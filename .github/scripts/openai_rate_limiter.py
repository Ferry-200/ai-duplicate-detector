"""Rate limiter for OpenAI API calls.

This module provides rate limiting functionality for OpenAI API calls, handling both
embedding and GPT model requests. It implements sophisticated tracking and throttling
mechanisms to stay within OpenAI's rate limits.

Key Features:
- Separate tracking for embedding and GPT model usage
- Per-minute and daily request limits
- Token consumption monitoring
- Automatic waiting and throttling
- Exponential backoff for rate limit errors

The rate limiter tracks:
- Requests per minute (RPM)
- Tokens per minute (TPM)
- Tokens per day (TPD)
- Requests per day (RPD) for GPT models

Rate Limits Enforced:
Embeddings (text-embedding-3-large):
    - 3,000 requests per minute
    - 1M tokens per minute
    - 3M tokens per day

GPT (gpt-3.5-turbo-1106):
    - 500 requests per minute
    - 200K tokens per minute
    - 2M tokens per day
    - 10K requests per day
"""

import time
import re
from typing import List, Tuple

class OpenAIRateLimiter:
    """Rate limiter for OpenAI API calls with separate tracking for embeddings and GPT.
    
    This class provides comprehensive rate limiting for OpenAI API calls, ensuring
    compliance with both short-term (per-minute) and long-term (daily) limits.
    It maintains separate tracking for embedding and GPT model usage, implementing
    automatic waiting and throttling when approaching limits.
    
    Attributes:
        embedding_rpm (int): Maximum embedding requests per minute
        embedding_tpm (int): Maximum embedding tokens per minute
        embedding_tpd (int): Maximum embedding tokens per day
        gpt_rpm (int): Maximum GPT requests per minute
        gpt_tpm (int): Maximum GPT tokens per minute
        gpt_tpd (int): Maximum GPT tokens per day
        gpt_rpd (int): Maximum GPT requests per day
        embedding_requests (List[Tuple[float, int]]): Recent embedding requests with timestamps and tokens
        embedding_daily_tokens (List[Tuple[float, int]]): Daily embedding token usage
        gpt_requests (List[Tuple[float, int]]): Recent GPT requests with timestamps and tokens
        gpt_daily_tokens (List[Tuple[float, int]]): Daily GPT token usage
        gpt_daily_requests (List[float]): Timestamps of daily GPT requests
        last_retry_wait (float): Last retry wait time for exponential backoff
    """
    
    def __init__(self):
        """Initialize rate limiters for both models."""
        # text-embedding-3-large limits
        self.embedding_rpm = 3000  # requests per minute
        self.embedding_tpm = 1_000_000  # tokens per minute
        self.embedding_tpd = 3_000_000  # tokens per day
        
        # gpt-3.5-turbo-1106 limits
        self.gpt_rpm = 500  # requests per minute
        self.gpt_tpm = 200_000  # tokens per minute
        self.gpt_tpd = 2_000_000  # tokens per day
        self.gpt_rpd = 10_000  # requests per day
        
        # Tracking for embeddings
        self.embedding_requests = []  # [(timestamp, tokens)]
        self.embedding_daily_tokens = []  # [(timestamp, tokens)]
        
        # Tracking for GPT
        self.gpt_requests = []  # [(timestamp, tokens)]
        self.gpt_daily_tokens = []  # [(timestamp, tokens)]
        self.gpt_daily_requests = []  # [timestamp]
        
        self.last_retry_wait = 1  # Initial retry wait in seconds
    
    def _clean_old_records(self, records: List[Tuple[float, int]], window_seconds: int) -> List[Tuple[float, int]]:
        """Remove records older than the specified time window.
        
        Args:
            records (List[Tuple[float, int]]): List of (timestamp, tokens) records
            window_seconds (int): Time window in seconds
            
        Returns:
            List[Tuple[float, int]]: Filtered list with only recent records
        """
        now = time.time()
        return [(t, tokens) for t, tokens in records if now - t < window_seconds]
    
    def _clean_old_timestamps(self, timestamps: List[float], window_seconds: int) -> List[float]:
        """Remove timestamps older than the specified time window.
        
        Args:
            timestamps (List[float]): List of timestamps
            window_seconds (int): Time window in seconds
            
        Returns:
            List[float]: Filtered list with only recent timestamps
        """
        now = time.time()
        return [t for t in timestamps if now - t < window_seconds]
    
    def _sum_tokens(self, records: List[Tuple[float, int]]) -> int:
        """Calculate total tokens from a list of records.
        
        Args:
            records (List[Tuple[float, int]]): List of (timestamp, tokens) records
            
        Returns:
            int: Total token count
        """
        return sum(tokens for _, tokens in records)
    
    def wait_if_needed_embedding(self, tokens: int):
        """Wait if approaching embedding rate limits.
        
        Implements waiting logic for embedding API calls, checking against
        per-minute and daily limits for both requests and tokens.
        
        Args:
            tokens (int): Number of tokens in the pending request
        """
        now = time.time()
        
        # Clean old records
        self.embedding_requests = self._clean_old_records(self.embedding_requests, 60)  # 1 minute window
        self.embedding_daily_tokens = self._clean_old_records(self.embedding_daily_tokens, 86400)  # 24 hour window
        
        # Check request rate
        if len(self.embedding_requests) >= self.embedding_rpm:
            oldest = self.embedding_requests[0][0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                print(f"\nApproaching embedding RPM limit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.embedding_requests = []
        
        # Check token rate (per minute)
        minute_tokens = self._sum_tokens(self.embedding_requests)
        if minute_tokens + tokens > self.embedding_tpm:
            print(f"\nApproaching embedding TPM limit. Waiting for next minute window...")
            time.sleep(60)
            self.embedding_requests = []
        
        # Check daily token rate
        daily_tokens = self._sum_tokens(self.embedding_daily_tokens)
        if daily_tokens + tokens > self.embedding_tpd:
            print(f"\nDaily embedding token limit reached. Waiting for reset...")
            time.sleep(3600)  # Wait an hour and try again
            self.embedding_daily_tokens = []
    
    def wait_if_needed_gpt(self, tokens: int):
        """Wait if approaching GPT rate limits.
        
        Implements waiting logic for GPT API calls, checking against
        per-minute and daily limits for requests and tokens.
        
        Args:
            tokens (int): Number of tokens in the pending request
        """
        now = time.time()
        
        # Clean old records
        self.gpt_requests = self._clean_old_records(self.gpt_requests, 60)  # 1 minute window
        self.gpt_daily_tokens = self._clean_old_records(self.gpt_daily_tokens, 86400)  # 24 hour window
        self.gpt_daily_requests = self._clean_old_timestamps(self.gpt_daily_requests, 86400)  # 24 hour window
        
        # Check request rate (per minute)
        if len(self.gpt_requests) >= self.gpt_rpm:
            oldest = self.gpt_requests[0][0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                print(f"\nApproaching GPT RPM limit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.gpt_requests = []
        
        # Check token rate (per minute)
        minute_tokens = self._sum_tokens(self.gpt_requests)
        if minute_tokens + tokens > self.gpt_tpm:
            print(f"\nApproaching GPT TPM limit. Waiting for next minute window...")
            time.sleep(60)
            self.gpt_requests = []
        
        # Check daily request rate
        if len(self.gpt_daily_requests) >= self.gpt_rpd:
            oldest = self.gpt_daily_requests[0]
            wait_time = 86400 - (now - oldest)
            print(f"\nDaily GPT request limit reached. Waiting {wait_time/3600:.1f} hours...")
            time.sleep(wait_time)
            self.gpt_daily_requests = []
        
        # Check daily token rate
        daily_tokens = self._sum_tokens(self.gpt_daily_tokens)
        if daily_tokens + tokens > self.gpt_tpd:
            print(f"\nDaily GPT token limit reached. Waiting for reset...")
            time.sleep(3600)  # Wait an hour and try again
            self.gpt_daily_tokens = []
    
    def record_embedding_usage(self, tokens: int):
        """Record embedding API usage.
        
        Args:
            tokens (int): Number of tokens used in the request
        """
        now = time.time()
        self.embedding_requests.append((now, tokens))
        self.embedding_daily_tokens.append((now, tokens))
    
    def record_gpt_usage(self, tokens: int):
        """Record GPT API usage.
        
        Args:
            tokens (int): Number of tokens used in the request
        """
        now = time.time()
        self.gpt_requests.append((now, tokens))
        self.gpt_daily_tokens.append((now, tokens))
        self.gpt_daily_requests.append(now)
    
    def handle_rate_limit_error(self, error_message: str) -> float:
        """Handle rate limit error with exponential backoff.
        
        Implements exponential backoff strategy when rate limits are hit,
        extracting wait time from error messages when available.
        
        Args:
            error_message (str): Error message from the API
            
        Returns:
            float: The actual wait time used
            
        Note:
            The wait time doubles with each retry but is capped at 5 minutes.
            A 2-second buffer is added to the suggested wait time.
        """
        wait_match = re.search(r"wait (\d+) seconds", error_message)
        wait_time = int(wait_match.group(1)) if wait_match else self.last_retry_wait
        
        wait_time += 2  # Add buffer
        print(f"\nRate limit exceeded. Waiting {wait_time} seconds before retrying...")
        time.sleep(wait_time)
        
        self.last_retry_wait = min(wait_time * 2, 300)  # Cap at 5 minutes
        return wait_time 