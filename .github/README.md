# AI-Powered GitHub Issue Duplicates & Relations Detector

This GitHub Action automatically detects duplicate and related issues in your repository using embeddings and OpenAI's API. It helps maintain a clean issue tracker by identifying potential duplicates when new issues are created and cross-referencing related issues.

## 📋 Features

- **Automatic Duplicate Detection**: Identifies potential duplicate issues based on semantic similarity
- **Related Issues Cross-Reference**: Identifies issues that are related but not duplicates and adds cross-reference comments
- **Customizable Thresholds**: Configure similarity thresholds for duplicate and related issue detection
- **Automated Issue Management**: Adds "duplicate" labels to identified duplicate issues, closes them, and marks them as "not planned"
- **Sub-Issue Migration**: Automatically moves sub-issues from closed duplicate issues to the kept issue
- **Issue Type Prioritization**: Intelligently decides which issue to keep open based on issue type priority
- **Comprehensive Logging**: Detailed logging for troubleshooting and monitoring
- **Open Issues Only**: Only processes open issues to avoid duplicating closed issues

## 🚀 Setup

### Prerequisites

- GitHub repository with issues enabled
- OpenAI API key
- Issue types in your repository (epic, task, sub-task) for prioritization when handling duplicates

### Installation

1. Copy the `.github` directory to your repository
2. Configure the required secrets and variables (see below)
3. The action will automatically run when new issues are created, reopened or edited

## ⚙️ Configuration

### Required Secrets

Add these secrets to your repository:

| Secret Name | Description |
|-------------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `GITHUB_TOKEN` | GitHub token with issues:write and pull-requests:write permissions (automatically provided) |

### Optional Variables

**Note: The variables below are hardcoded in the codebase and not configurable through repository variables.** To customize these settings, you'll need to modify the code files directly:

| Variable Name | Description | Default | File Location | Code Variable |
|---------------|-------------|---------|--------------|--------------|
| `DUPLICATE_THRESHOLD` | Similarity threshold for duplicate detection (0-1) | 0.85 | `.github/scripts/duplicate_detector.py` | `embedding_low_threshold` in DuplicateDetector class |
| `RELATED_ISSUE_THRESHOLD` | Similarity threshold for related issues (0-1) | 0.82 | `.github/scripts/duplicate_detector.py` | `RELATED_ISSUE_THRESHOLD` constant |
| `EMBEDDING_MODEL` | OpenAI embedding model to use | text-embedding-3-large | `.github/scripts/issue_embedder.py` | `self.model` in IssueEmbedder class |
| `MAX_ISSUES_TO_PROCESS` | Maximum number of issues to process in a single run | Variable | `.github/scripts/detect_duplicates.py` | Handled internally |

To modify these settings:

1. **Update DUPLICATE_THRESHOLD**: Edit the `embedding_low_threshold = 0.85` value in the `__init__` method of the DuplicateDetector class in `.github/scripts/duplicate_detector.py` (around line 84)

2. **Update RELATED_ISSUE_THRESHOLD**: Edit the `RELATED_ISSUE_THRESHOLD = 0.82` constant near the top of `.github/scripts/duplicate_detector.py` (around line 49)

3. **Update EMBEDDING_MODEL**: Edit the `self.model = "text-embedding-3-large"` value in the `__init__` method of the IssueEmbedder class in `.github/scripts/issue_embedder.py` (around line 55)

Higher threshold values will lead to fewer false positives but might miss some genuine duplicates. Lower thresholds will catch more potential duplicates but may include more false positives.

## 📂 Directory Structure

```
.github/
├── scripts/
│   ├── duplicate_detector.py    # Core logic for duplicate detection
│   ├── update_embeddings.py     # Maintains issue embeddings database
│   ├── fetch_bulk_issues.py     # Fetches issues from GitHub API
│   └── ... (other utility scripts)
├── workflows/
│   └── ai-duplicate-detector.yaml  # Workflow definition for issue duplicate detection
└── README.md                    # This file
```

## 🔄 Workflows

### ai-duplicate-detector.yaml

This workflow runs when:
- A new issue is created (opened)
- An issue is reopened
- An issue is edited
- On manual trigger (workflow_dispatch)

The workflow includes a built-in 3.5-minute delay to allow for additional edits before processing:

```yaml
# From ai-duplicate-detector.yaml
steps:
  - name: Wait for potential edits
    if: github.event.issue
    run: |
      echo "Waiting 3.5 minutes for potential additional edits..."
      sleep 210
```

The workflow also has concurrency control to prevent multiple runs for the same issue:

```yaml
# From ai-duplicate-detector.yaml
concurrency:
  # Use issue number if available, otherwise use a unique identifier
  group: ${{ github.event.issue.number || github.run_id }}
  # Cancel in-progress runs
  cancel-in-progress: true
```

## 💾 Database

The action maintains an SQLite database (`embeddings.db`) that stores:
- Issue IDs, titles and content
- Computed embeddings for each issue
- Processing history

The database is automatically updated when the action runs.

## 🧠 How It Works

1. When a new issue is created, reopened, or edited:
   - The action waits 3.5 minutes to allow for additional edits by the author
   - The action fetches the issue content (only processes open issues)
   - Generates an embedding using OpenAI's API
   - Compares the embedding to existing open issues
   - Identifies potential duplicates and related issues based on similarity scores

2. For duplicate issues (similarity > DUPLICATE_THRESHOLD):
   - Determines which issue to keep open based on issue type priority
   - Adds a comment linking to the chosen "main" issue
   - Applies the "duplicate" label to the other issue

3. For related issues (similarity between RELATED_ISSUE_THRESHOLD and DUPLICATE_THRESHOLD):
   - Adds cross-reference comments to both issues

### Issue Type Prioritization

When duplicate issues are detected, the system intelligently decides which issue to keep open based on the issue type. The priority is hardcoded in the `get_issue_priority` function in `detect_duplicates.py`:

1. Epic (highest priority - value 3)
2. Task (medium priority - value 2)
3. Sub-task (lowest priority - value 1)

The function first checks the issue type field, and if not available, falls back to checking for keywords in the title.

For example:
- If an "epic" issue and a "task" issue are duplicates, the "epic" issue will be kept open
- If two "task" issues are duplicates, the older one will be kept open

This ensures that more significant issues (like epics) take precedence over less significant ones when duplicates are found.

## 🔧 Advanced Configuration

### Customizing Similarity Thresholds

Adjust the thresholds in the repository variables to fine-tune detection sensitivity:
- Higher threshold = fewer false positives but might miss some duplicates
- Lower threshold = catches more potential duplicates but more false positives

### Processing Delay

The processing delay is hardcoded to 3.5 minutes (210 seconds) in the workflow file. To change this delay, you'll need to modify the `sleep 210` value in the `ai-duplicate-detector.yaml` file:

```yaml
- name: Wait for potential edits
  if: github.event.issue
  run: |
    echo "Waiting 3.5 minutes for potential additional edits..."
    sleep 210  # Change this value to adjust the delay (in seconds)
```

## 🔍 Troubleshooting

### Common Issues

1. **Action not running**: Ensure the workflow file is properly configured and GitHub Actions is enabled for your repository.

2. **No duplicates detected**: Check the threshold values - they might be set too high.

3. **Too many false positives**: Increase the threshold values.

4. **API Rate Limiting**: If you hit GitHub API rate limits, consider increasing the schedule interval.

5. **Issue prioritization not working**: Verify your repository has issue types properly configured (epic, task, sub-task).

### Logs

Check the GitHub Actions logs for detailed information about each run.

## 📊 Example Output

When a duplicate is detected:
```
Potential duplicate of #42 found: Similarity score 0.89
Issue #42 type: epic, Issue #123 type: task
Keeping issue #42 open (higher priority type)
Adding comment to issue #123 referencing the original issue #42
Adding "duplicate" label to issue #123
```

When related issues are found:
```
Related issue #56 found: Similarity score 0.83
Adding cross-reference comments to issues #123 and #56
```

## 📝 License

MIT

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 