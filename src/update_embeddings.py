#!/usr/bin/env python3
"""Update embeddings for all issues before duplicate detection.

This script manages the process of updating and maintaining embeddings for GitHub issues,
implementing efficient multi-threaded processing, comprehensive error handling, and
performance tracking.

Key Features:
- Multi-threaded embedding generation for improved performance
- Comprehensive error handling and logging
- Performance tracking and metrics collection
- Automatic cleaning of closed issues
- OpenAI API usage monitoring
- Persistent storage of performance metrics

The script follows this process:
1. Loads cached issues from local storage
2. Validates and processes issue data
3. Cleans embeddings for closed issues
4. Updates embeddings for all active issues using multiple threads
5. Tracks and saves performance metrics

Dependencies:
    - openai: For API access and embedding generation
    - concurrent.futures: For multi-threaded processing
    - logging: For comprehensive logging
    - json: For cache and metrics file operations
    - pathlib: For file path handling

Environment Variables Required:
    OPENAI_API_KEY: OpenAI API authentication token

Cache Structure:
    The script expects issues cached in either:
    - List format: Direct array of issue objects
    - Dict format: {'timestamp': str, 'issues': List[Dict]}

Performance Metrics:
    Tracks and saves:
    - Total issues processed
    - Successful/failed updates
    - Processing rate
    - OpenAI API usage statistics
    - Execution time
"""

import json
import logging
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from duplicate_detector import DuplicateDetector

# Configure logging with file handler for persistent logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(".github/cache/update_embeddings.log")
    ]
)

class PerformanceTracker:
    """Tracks and reports performance metrics during embedding updates.
    
    This class provides real-time tracking and periodic reporting of performance
    metrics during the embedding update process, including processing rates,
    success/failure counts, and elapsed time.
    
    Attributes:
        start_time (float): Timestamp when tracking started
        issue_count (int): Total number of issues processed
        failed_issues (int): Count of failed embedding updates
        last_log_time (float): Timestamp of last progress log
        log_interval (int): Seconds between progress log updates
    """
    
    def __init__(self):
        """Initialize the performance tracker with default values."""
        self.start_time = time.time()
        self.issue_count = 0
        self.failed_issues = 0
        self.last_log_time = self.start_time
        self.log_interval = 5  # seconds
    
    def update(self, success: bool = True):
        """Update metrics with a processed issue and log progress periodically.
        
        Args:
            success (bool): Whether the issue was processed successfully
        """
        self.issue_count += 1
        if not success:
            self.failed_issues += 1
        
        # Log progress periodically
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            elapsed = current_time - self.start_time
            rate = self.issue_count / elapsed if elapsed > 0 else 0
            logging.info(f"Processed {self.issue_count} issues ({rate:.2f} issues/sec), {self.failed_issues} failures")
            self.last_log_time = current_time
    
    def get_summary(self) -> Dict:
        """Get summary of performance metrics.
        
        Returns:
            Dict: Summary metrics including total issues, failures, elapsed time,
                 and processing rate
        """
        elapsed = time.time() - self.start_time
        return {
            "total_issues": self.issue_count,
            "failed_issues": self.failed_issues,
            "elapsed_seconds": elapsed,
            "issues_per_second": self.issue_count / elapsed if elapsed > 0 else 0
        }

def load_cached_issues() -> List[Dict]:
    """Load and validate cached issues from storage.
    
    Returns:
        List[Dict]: List of issue objects from cache
        
    Raises:
        FileNotFoundError: If cache file doesn't exist
        ValueError: If cache format is invalid
        json.JSONDecodeError: If cache file is corrupted
        
    The function handles both list and dict cache formats, provides detailed
    error logging, and validates cache file integrity.
    """
    cache_file = Path(".github/cache/open_issues_latest.json")
    
    if not cache_file.exists():
        raise FileNotFoundError(f"No cached issues found at {cache_file}. Run fetch_bulk_issues.py first.")
    
    try:
        with open(cache_file) as f:
            cache_data = json.load(f)
            
        # Handle both possible formats
        if isinstance(cache_data, list):
            logging.info(f"Loaded {len(cache_data)} issues from cache (list format)")
            return cache_data
        elif isinstance(cache_data, dict) and 'issues' in cache_data:
            issue_count = len(cache_data['issues'])
            timestamp = cache_data.get('timestamp', 'unknown')
            logging.info(f"Loaded {issue_count} issues from cache (timestamp: {timestamp})")
            return cache_data['issues']
        else:
            raise ValueError(f"Invalid cache file format. Expected list or dict with 'issues' key, got {type(cache_data)}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse cache file: {str(e)}")
        # Check if file is empty or corrupted
        file_size = cache_file.stat().st_size
        if file_size == 0:
            logging.error("Cache file is empty")
        elif file_size < 1000:
            with open(cache_file) as f:
                content = f.read()
            logging.error(f"Cache file content (first 200 chars): {content[:200]}")
        raise

def process_issues(issues: List[Dict]) -> List[Dict]:
    """Process and validate issue data with detailed error tracking.
    
    Args:
        issues (List[Dict]): Raw issue data from cache
        
    Returns:
        List[Dict]: Processed and validated issue objects
        
    The function performs comprehensive validation of issue data:
    - Ensures required fields exist and have correct types
    - Tracks different types of validation errors
    - Provides detailed logging of validation failures
    - Handles both direct API response and cached formats
    """
    start_time = time.time()
    processed = []
    error_counts = {'number': 0, 'title': 0, 'body': 0, 'updated_at': 0, 'type': 0}
    
    for idx, issue in enumerate(issues):
        try:
            # Handle both direct API response and cached format
            if isinstance(issue, dict):
                # Ensure required fields exist and are of correct type
                processed_issue = {}
                
                # Process and validate number
                try:
                    number = issue.get("number")
                    if number is None:
                        error_counts['number'] += 1
                        logging.warning(f"Issue at index {idx} has no number field")
                        continue
                    processed_issue["number"] = int(number)
                except (ValueError, TypeError):
                    error_counts['number'] += 1
                    logging.warning(f"Invalid issue number: {issue.get('number')} (type: {type(issue.get('number'))})")
                    continue
                
                # Process and validate title
                try:
                    title = issue.get("title")
                    if not title:
                        error_counts['title'] += 1
                        logging.warning(f"Issue #{processed_issue['number']} has no title")
                        continue
                    processed_issue["title"] = str(title)
                except Exception:
                    error_counts['title'] += 1
                    logging.warning(f"Invalid issue title for #{processed_issue['number']}")
                    continue
                
                # Process body and updated_at (non-critical fields)
                try:
                    processed_issue["body"] = str(issue.get("body", ""))
                except Exception:
                    error_counts['body'] += 1
                    processed_issue["body"] = ""
                    logging.warning(f"Invalid issue body for #{processed_issue['number']}, using empty string")
                
                try:
                    processed_issue["updated_at"] = str(issue.get("updated_at", ""))
                except Exception:
                    error_counts['updated_at'] += 1
                    processed_issue["updated_at"] = ""
                    logging.warning(f"Invalid updated_at for #{processed_issue['number']}, using empty string")
                
                # Add to processed list
                processed.append(processed_issue)
            else:
                error_counts['type'] += 1
                logging.warning(f"Skipping non-dict issue data at index {idx}: {type(issue)}")
                
        except Exception as e:
            logging.warning(f"Unexpected error processing issue at index {idx}: {str(e)}")
            if isinstance(issue, dict):
                logging.warning(f"Problem with issue: {issue.get('number', 'unknown')}")
            continue
    
    # Log summary of processed issues
    elapsed = time.time() - start_time
    if not processed:
        logging.error("No valid issues found after processing!")
    else:
        logging.info(f"Successfully processed {len(processed)} issues in {elapsed:.2f} seconds")
        
    # Log error summary if any errors occurred
    if sum(error_counts.values()) > 0:
        logging.warning(f"Processing errors: {error_counts}")
    
    return processed

def update_issue_embeddings(detector: DuplicateDetector, issues: List[Dict], max_workers: int = 4) -> Tuple[int, int]:
    """Update embeddings for issues using multi-threaded processing.
    
    Args:
        detector (DuplicateDetector): Instance of duplicate detector
        issues (List[Dict]): List of processed issue objects
        max_workers (int): Maximum number of concurrent threads
        
    Returns:
        Tuple[int, int]: Count of successful and failed updates
        
    The function implements:
    - Multi-threaded processing for improved performance
    - Real-time progress tracking
    - Comprehensive error handling
    - Detailed logging of successes and failures
    """
    tracker = PerformanceTracker()
    success_count = 0
    failure_count = 0
    
    # Optimize by using multiple threads for API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_issue = {
            executor.submit(update_single_issue_embedding, detector, issue): issue 
            for issue in issues
        }
        
        # Process as they complete
        for future in as_completed(future_to_issue):
            issue = future_to_issue[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                tracker.update(success)
            except Exception as e:
                logging.error(f"Error updating embedding for issue #{issue.get('number')}: {str(e)}")
                failure_count += 1
                tracker.update(False)
    
    # Log final summary
    summary = tracker.get_summary()
    logging.info(f"Embedding update completed: {success_count} successes, {failure_count} failures in {summary['elapsed_seconds']:.2f}s")
    return success_count, failure_count

def update_single_issue_embedding(detector: DuplicateDetector, issue: Dict) -> bool:
    """Update embedding for a single issue with error handling.
    
    Args:
        detector (DuplicateDetector): Instance of duplicate detector
        issue (Dict): Issue object to update
        
    Returns:
        bool: True if update successful, False otherwise
        
    The function:
    - Validates required issue data
    - Checks if update is needed based on timestamp
    - Generates and stores new embedding if needed
    - Provides detailed error logging
    """
    try:
        issue_number = issue.get('number')
        title = issue.get('title', '')
        body = issue.get('body', '')
        updated_at = issue.get('updated_at', '')
        
        if not issue_number or not title:
            logging.error(f"Missing required data for issue: {issue}")
            return False
        
        needs_update = detector.store.needs_update(issue_number, updated_at)
        if needs_update:
            embedding = detector.embedder.get_embedding(title, body)
            detector.store.store_embedding(issue_number, title, body, embedding, updated_at)
            logging.debug(f"Updated embedding for issue #{issue_number}")
            return True
        else:
            logging.debug(f"No update needed for issue #{issue_number}")
            return True
    except Exception as e:
        logging.error(f"Error updating embedding for issue {issue.get('number', 'unknown')}: {str(e)}")
        return False

def clean_closed_issues(detector: DuplicateDetector, open_issues: List[Dict]) -> int:
    """Remove embeddings for closed issues from the database.
    
    Args:
        detector (DuplicateDetector): Instance of duplicate detector
        open_issues (List[Dict]): List of currently open issues
        
    Returns:
        int: Number of closed issues removed from database
        
    The function:
    - Identifies issues in database that are no longer open
    - Removes embeddings for closed issues
    - Provides detailed logging of cleaning process
    """
    logging.info("Cleaning closed issues from embeddings database...")
    
    # Get all issue numbers from the embeddings database
    stored_issues = detector.store.get_all_issue_metadata()
    
    # Create a set of open issue numbers for fast lookup
    open_issue_numbers = {issue['number'] for issue in open_issues}
    
    # Find issues in the database that are not in the open issues list
    closed_issues = [issue_num for issue_num in stored_issues.keys() 
                    if issue_num not in open_issue_numbers]
    
    if not closed_issues:
        logging.info("No closed issues found in embeddings database")
        return 0
    
    # Remove closed issues from the database
    for issue_num in closed_issues:
        logging.info(f"Removing closed issue #{issue_num} from embeddings database")
        detector.store.remove_embedding(issue_num)
    
    logging.info(f"Removed {len(closed_issues)} closed issues from embeddings database")
    return len(closed_issues)

def save_performance_metrics(metrics: Dict):
    """Save performance metrics to a persistent file.
    
    Args:
        metrics (Dict): Performance metrics to save
        
    The function:
    - Maintains a history of recent runs (last 10)
    - Adds timestamps to metrics
    - Handles file I/O with error handling
    - Provides detailed logging of save operations
    """
    try:
        metrics_file = Path(".github/cache/embedding_update_metrics.json")
        
        # Load existing metrics if available
        history = []
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = []
            except:
                history = []
        
        # Add timestamp to metrics
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history and save
        history.append(metrics)
        # Keep only last 10 runs
        if len(history) > 10:
            history = history[-10:]
            
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save performance metrics: {str(e)}")

def main():
    """Main function coordinating the embedding update process.
    
    The function orchestrates the entire embedding update process:
    1. Validates environment variables
    2. Loads and processes cached issues
    3. Initializes duplicate detector
    4. Cleans closed issues
    5. Updates embeddings with multi-threading
    6. Collects and saves performance metrics
    
    Provides comprehensive error handling and detailed logging
    throughout the process.
    """
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logging.critical("OPENAI_API_KEY environment variable is not set.")
            sys.exit(1)
        
        # Load cached issues
        logging.info("Loading cached issues...")
        cached_issues = load_cached_issues()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logging.info(f"Loaded {len(cached_issues)} issues from cache (timestamp: {timestamp})")
        
        # Process and validate issues
        logging.info("Processing issue data...")
        issues = process_issues(cached_issues)
        
        # Initialize duplicate detector (which also initializes embedding store)
        logging.info("Initializing duplicate detector")
        detector = DuplicateDetector(api_key=openai_api_key)
        
        # Clean closed issues from the embeddings database
        cleaned_count = clean_closed_issues(detector, issues)
        
        # Determine optimal number of workers based on issue count
        max_workers = min(10, max(2, len(issues) // 10))
        logging.info(f"Using {max_workers} worker threads for embedding updates")
        
        # Update embeddings for all issues
        logging.info(f"Updating embeddings for {len(issues)} issues...")
        success_count, failure_count = update_issue_embeddings(detector, issues, max_workers)
        
        # Log success/failure counts
        logging.info(f"Embedding update completed: {success_count} successes, {failure_count} failures, {cleaned_count} closed issues cleaned")
        
        # Get OpenAI usage stats if available, with multiple fallback methods
        openai_usage = {}
        try:
            if hasattr(detector, "rate_limiter"):
                rate_limiter = detector.rate_limiter
                # Try different approaches to get stats
                if hasattr(rate_limiter, "get_usage_stats") and callable(rate_limiter.get_usage_stats):
                    openai_usage = rate_limiter.get_usage_stats()
                # Fallback: Try to collect stats directly from attributes if they exist
                elif hasattr(rate_limiter, "embedding_api_calls"):
                    openai_usage = {
                        "embedding_api_calls": getattr(rate_limiter, "embedding_api_calls", 0),
                        "embedding_tokens": getattr(rate_limiter, "embedding_tokens", 0),
                        "gpt_api_calls": getattr(rate_limiter, "gpt_api_calls", 0),
                        "gpt_tokens": getattr(rate_limiter, "gpt_tokens", 0),
                        "rate_limit_retries": getattr(rate_limiter, "rate_limit_retries", 0)
                    }
                logging.info(f"OpenAI usage stats: {openai_usage}")
        except Exception as e:
            logging.warning(f"Could not collect OpenAI usage stats: {str(e)}")
            openai_usage = {"error": "Failed to collect stats"}
        
        # Save metrics
        metrics = {
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "issues_processed": len(issues),
            "successful_updates": success_count,
            "failed_updates": failure_count,
            "closed_issues_cleaned": cleaned_count,
            "openai_usage": openai_usage
        }
        save_performance_metrics(metrics)
        
    except Exception as e:
        logging.critical(f"Error in embedding update process: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        sys.exit(1) 