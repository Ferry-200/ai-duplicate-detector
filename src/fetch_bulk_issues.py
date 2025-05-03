"""Fetch and cache GitHub issues.

This script fetches all open issues from a GitHub repository and caches them locally for later use.
It handles pagination, rate limiting, retries, and provides comprehensive error handling and logging.

Key Features:
- Fetches all open issues using GitHub's REST API
- Handles API pagination automatically
- Implements exponential backoff retry logic
- Respects GitHub API rate limits
- Filters out pull requests from results
- Atomic cache file updates
- Comprehensive logging of operations and errors

Environment Variables Required:
    GITHUB_TOKEN: GitHub API authentication token
    GITHUB_REPOSITORY: Repository in format "owner/repo"

Dependencies:
    - requests: For GitHub API calls
    - pathlib: For path handling
    - logging: For operation logging
    - json: For cache file operations

The script follows this process:
1. Validates environment variables
2. Sets up logging to both console and file
3. Fetches issues page by page with retry logic
4. Filters out pull requests from results
5. Saves results to cache with atomic write operations

Cache Structure:
    The cache file contains:
    - timestamp: When the cache was created
    - repository: The repository the issues are from
    - issues: Array of issue objects from GitHub API
"""

import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import requests
from requests.exceptions import RequestException
from detect_duplicates import get_github_api_headers, get_repo_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(".github/cache/fetch_issues.log")
    ]
)

def fetch_issues(owner: str, repo: str, headers: dict, max_retries: int = 3) -> List[Dict]:
    """Fetch all open issues from the repository using pagination with retry logic.
    
    Args:
        owner (str): Repository owner/organization name
        repo (str): Repository name
        headers (dict): GitHub API headers including authentication
        max_retries (int, optional): Maximum number of retry attempts per request. Defaults to 3.
    
    Returns:
        List[Dict]: List of issue objects from GitHub API, excluding pull requests
        
    Raises:
        RequestException: If API requests fail after all retries
        
    The function implements:
    - Pagination handling (100 issues per page)
    - Exponential backoff for retries
    - Rate limit detection and handling
    - Automatic filtering of pull requests
    - Comprehensive logging of progress and rate limits
    
    Rate Limit Handling:
    - Checks for rate limit headers on every response
    - Respects Retry-After headers when rate limited
    - Implements exponential backoff with jitter
    - Logs detailed rate limit information
    """
    issues = []
    page = 1
    per_page = 100
    
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            'state': 'open',
            'per_page': per_page,
            'page': page,
            'sort': 'created',
            'direction': 'desc'
        }
        
        for attempt in range(max_retries):
            try:
                logging.info(f"Fetching page {page} of issues (attempt {attempt+1}/{max_retries})")
                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                
                # Log rate limit headers immediately after successful request
                try:
                    reset_timestamp = int(response.headers.get('x-ratelimit-reset', 0))
                    reset_time_str = datetime.utcfromtimestamp(reset_timestamp).isoformat() + 'Z' if reset_timestamp else 'N/A'
                    headers_to_log = {
                        'limit': response.headers.get('x-ratelimit-limit'),
                        'remaining': response.headers.get('x-ratelimit-remaining'),
                        'used': response.headers.get('x-ratelimit-used'),
                        'reset': reset_time_str,
                        'resource': response.headers.get('x-ratelimit-resource')
                    }
                    logging.info(f"GitHub API Rate Limit Status: {headers_to_log}")
                except Exception as log_e:
                    logging.warning(f"Could not log rate limit status: {log_e}")

                batch = response.json()
                if not batch:
                    logging.info("No more issues found")
                    return issues
                    
                # Filter out pull requests
                filtered_batch = [issue for issue in batch if 'pull_request' not in issue]
                issues.extend(filtered_batch)
                
                logging.info(f"Fetched page {page}: {len(filtered_batch)} issues (filtered from {len(batch)} items)")
                
                if len(batch) < per_page:
                    logging.info("Reached last page of results")
                    return issues
                    
                page += 1
                # Small delay to avoid hitting rate limits
                time.sleep(0.5)
                break
                
            except RequestException as e:
                # Log rate limit headers even on failure if available
                if hasattr(e, 'response') and e.response is not None:
                     try:
                         reset_timestamp = int(e.response.headers.get('x-ratelimit-reset', 0))
                         reset_time_str = datetime.utcfromtimestamp(reset_timestamp).isoformat() + 'Z' if reset_timestamp else 'N/A'
                         headers_to_log = {
                             'status': e.response.status_code,
                             'limit': e.response.headers.get('x-ratelimit-limit'),
                             'remaining': e.response.headers.get('x-ratelimit-remaining'),
                             'used': e.response.headers.get('x-ratelimit-used'),
                             'reset': reset_time_str,
                             'resource': e.response.headers.get('x-ratelimit-resource')
                         }
                         logging.warning(f"GitHub API Rate Limit Status on FAILED request: {headers_to_log}")
                     except Exception as log_e:
                         logging.warning(f"Could not log rate limit status on failed request: {log_e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff

                    # Check for Retry-After header specifically for 403/429
                    retry_after = None
                    if hasattr(e, 'response') and e.response is not None and e.response.status_code in [403, 429]:
                        retry_after_header = e.response.headers.get('Retry-After')
                        if retry_after_header:
                            try:
                                retry_after = int(retry_after_header)
                                logging.warning(f"Rate limit likely hit. Respecting Retry-After header: waiting {retry_after} seconds.")
                                wait_time = retry_after + 1 # Add buffer
                            except ValueError:
                                logging.warning(f"Could not parse Retry-After header: {retry_after_header}")

                    logging.warning(f"Request failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to fetch issues after {max_retries} attempts: {str(e)}")
                    if hasattr(e, 'response') and e.response:
                        logging.error(f"Response code: {e.response.status_code}")
                        logging.error(f"Response text: {e.response.text}")
                    raise
    
    return issues

def validate_environment() -> tuple:
    """Validate required environment variables and return their values.
    
    Returns:
        tuple: (github_token, repository_name)
        
    Raises:
        EnvironmentError: If any required variables are missing
        
    Required Environment Variables:
        GITHUB_TOKEN: Personal access token for GitHub API authentication
        GITHUB_REPOSITORY: Repository name in format "owner/repo"
    """
    token = os.environ.get('GITHUB_TOKEN')
    repository = os.environ.get('GITHUB_REPOSITORY')
    
    missing = []
    if not token:
        missing.append('GITHUB_TOKEN')
    if not repository:
        missing.append('GITHUB_REPOSITORY')
    
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    
    return token, repository

def save_to_cache(issues: List[Dict], cache_dir: Path, repository: str) -> Path:
    """Save fetched issues to a cache file with atomic write operations.
    
    Args:
        issues (List[Dict]): List of issue objects to cache
        cache_dir (Path): Directory to store cache files
        repository (str): Repository name for cache metadata
        
    Returns:
        Path: Path to the created cache file
        
    Raises:
        IOError: If file operations fail
        OSError: If directory operations fail
        
    The cache file is written atomically by:
    1. Writing to a temporary file
    2. Renaming the temporary file to the target filename
    
    Cache Format:
        {
            "timestamp": "YYYYMMDD_HHMMSS",
            "repository": "owner/repo",
            "issues": [...]
        }
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_data = {
        'timestamp': timestamp,
        'repository': repository,
        'issues': issues
    }
    
    cache_file = cache_dir / "open_issues_latest.json"
    temp_file = cache_dir / "open_issues_latest.json.tmp"
    
    try:
        # Write to temporary file first
        with open(temp_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        # Rename to target file (atomic operation)
        temp_file.rename(cache_file)
        logging.info(f"Successfully saved {len(issues)} issues to {cache_file}")
        
        return cache_file
    except (IOError, OSError) as e:
        logging.error(f"Failed to save cache file: {str(e)}")
        if temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass
        raise

def main():
    """Main execution function with comprehensive error handling.
    
    This function orchestrates the entire process:
    1. Validates environment variables
    2. Sets up GitHub API access
    3. Creates cache directory
    4. Fetches all open issues
    5. Saves issues to cache
    
    The function implements comprehensive error handling and logging,
    with different exit codes for different types of failures:
    - Environment errors: Exit code 1
    - GitHub API errors: Exit code 1
    - Unexpected errors: Exit code 1
    """
    start_time = time.time()
    
    try:
        # Validate environment
        logging.info("Validating environment variables")
        token, repository = validate_environment()
        
        # Get repository info
        owner, repo = get_repo_info(repository)
        headers = get_github_api_headers(token)
        
        # Create cache directory
        cache_dir = Path(".github/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch all open issues
        logging.info(f"Fetching open issues for {owner}/{repo}")
        issues = fetch_issues(owner, repo, headers)
        logging.info(f"Found {len(issues)} open issues")
        
        if not issues:
            logging.warning("No issues found. This could be normal for new repositories.")
        
        # Save to cache
        cache_file = save_to_cache(issues, cache_dir, repository)
        
        elapsed = time.time() - start_time
        logging.info(f"Process completed successfully in {elapsed:.2f} seconds")
        
    except EnvironmentError as e:
        logging.critical(f"Environment error: {str(e)}")
        sys.exit(1)
    except RequestException as e:
        logging.critical(f"GitHub API error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 