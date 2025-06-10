from collections import deque
from threading import Lock
import time
import logging

logger = logging.getLogger(__name__)

class TokenRateLimiter:
    """Tracks token usage over a sliding 60-second window and enforces rate limits."""
    
    def __init__(self, max_tokens_per_minute: int = 35000, window_seconds: int = 60):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.window_seconds = window_seconds
        self.token_history = deque()  # (timestamp, token_count) tuples
        self.lock = Lock()
        
    def _cleanup_expired_entries(self, current_time: float):
        """Remove entries older than the window."""
        cutoff_time = current_time - self.window_seconds
        while self.token_history and self.token_history[0][0] < cutoff_time:
            self.token_history.popleft()
    
    def _get_current_usage(self, current_time: float) -> int:
        """Get total tokens used in the current window."""
        self._cleanup_expired_entries(current_time)
        return sum(tokens for _, tokens in self.token_history)
    
    def track(self, token_count: int):
        """Add token usage to the sliding window."""
        with self.lock:
            current_time = time.time()
            self.token_history.append((current_time, token_count))
            self._cleanup_expired_entries(current_time)
            
            current_usage = self._get_current_usage(current_time)
            logger.debug(f"Added {token_count} tokens. Current window usage: {current_usage}/{self.max_tokens_per_minute}")
    
    def wait_if_needed(self, estimated_tokens: int) -> float:
        """
        Check if adding estimated_tokens would exceed the limit.
        If so, calculate and return wait time. Otherwise return 0.
        """
        with self.lock:
            current_time = time.time()
            current_usage = self._get_current_usage(current_time)
            
            if current_usage + estimated_tokens <= self.max_tokens_per_minute:
                return 0  # No wait needed
            
            # Find when enough tokens will expire to make room
            needed_space = (current_usage + estimated_tokens) - self.max_tokens_per_minute
            
            # Look through history to find when enough tokens will expire
            cumulative_expired = 0
            wait_time = 0
            
            for timestamp, tokens in self.token_history:
                cumulative_expired += tokens
                if cumulative_expired >= needed_space:
                    # Wait until this entry expires
                    wait_time = max(0, (timestamp + self.window_seconds) - current_time)
                    break
            
            if wait_time > 0:
                logger.info(f"Rate limit check: {current_usage + estimated_tokens}/{self.max_tokens_per_minute} tokens would exceed limit. Waiting {wait_time:.1f}s")
            
            return wait_time
    
    def estimate_tokens(self, payload: dict) -> int:
        """Rough estimation of input tokens for a payload."""
        total_tokens = 0
        
        # Tools schema tokens (from your docs: ~346 tokens base + tool definitions)
        if payload.get("tools"):
            total_tokens += 346  # Base tool system prompt
            for tool in payload["tools"]:
                # Rough estimate: tool name + description + schema
                tool_str = str(tool)
                total_tokens += len(tool_str.split()) * 1.3
        
        # System prompt tokens
        if payload.get("system"):
            for block in payload["system"]:
                if block.get("type") == "text":
                    total_tokens += len(block.get("text", "").split()) * 1.3
        
        # Message tokens
        for message in payload.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, str):
                total_tokens += len(content.split()) * 1.3
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        total_tokens += len(item.get("text", "").split()) * 1.3
                    elif item.get("type") == "image":
                        # Images are roughly 85 tokens per 512x512 tile
                        # Conservative estimate: ~1000 tokens per image
                        total_tokens += 1000
                    elif item.get("type") == "tool_result":
                        total_tokens += len(str(item.get("content", "")).split()) * 1.3
        
        # Thinking mode tokens (if enabled)
        if payload.get("thinking"):
            total_tokens += 100  # Small overhead for thinking mode
        
        return int(total_tokens)

    def apply(self, payload: dict):
        """Apply rate limiting before making a request."""
        estimated_tokens = self.estimate_tokens(payload)
        
        wait_time = self.wait_if_needed(estimated_tokens)
        if wait_time > 0:
            logger.warning(f"Rate limiting: sleeping {wait_time:.1f}s to avoid exceeding 35k tokens/min")
            time.sleep(wait_time)