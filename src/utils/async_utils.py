"""
Async utility functions and helpers.

This module provides utilities for async operations including concurrency control,
async context managers, and other async-related helpers used throughout the application.
"""

import asyncio
import time
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


async def run_async(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a synchronous function in a thread pool to avoid blocking the event loop.
    
    Args:
        func: Synchronous function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function execution
    
    Examples:
        >>> import time
        >>> def blocking_operation(duration: int) -> str:
        ...     time.sleep(duration)
        ...     return f"Completed after {duration}s"
        >>> result = await run_async(blocking_operation, 2)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


async def gather_with_concurrency(
    coroutines: List[Awaitable[T]], 
    max_concurrency: int = 10,
    return_exceptions: bool = False
) -> List[Union[T, Exception]]:
    """
    Execute coroutines with controlled concurrency using semaphore.
    
    Args:
        coroutines: List of coroutines to execute
        max_concurrency: Maximum number of concurrent executions
        return_exceptions: Whether to return exceptions or raise them
    
    Returns:
        List of results from coroutine execution
    
    Examples:
        >>> async def fetch_data(url: str) -> dict:
        ...     # Simulate API call
        ...     await asyncio.sleep(1)
        ...     return {"url": url, "data": "response"}
        >>> 
        >>> urls = ["url1", "url2", "url3"]
        >>> coroutines = [fetch_data(url) for url in urls]
        >>> results = await gather_with_concurrency(coroutines, max_concurrency=2)
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def controlled_coroutine(coro: Awaitable[T]) -> Union[T, Exception]:
        async with semaphore:
            try:
                return await coro
            except Exception as e:
                if return_exceptions:
                    return e
                raise
    
    controlled_coroutines = [controlled_coroutine(coro) for coro in coroutines]
    return await asyncio.gather(*controlled_coroutines, return_exceptions=return_exceptions)


class AsyncBatch:
    """
    Utility for processing items in batches asynchronously.
    
    Processes a large number of items by breaking them into smaller batches
    and processing each batch with controlled concurrency.
    
    Examples:
        >>> async def process_item(item: int) -> int:
        ...     await asyncio.sleep(0.1)  # Simulate processing
        ...     return item * 2
        >>> 
        >>> items = list(range(100))
        >>> async_batch = AsyncBatch(process_item, batch_size=10, max_concurrency=3)
        >>> results = await async_batch.process(items)
    """
    
    def __init__(self, 
                 processor: Callable[[T], Awaitable[Any]],
                 batch_size: int = 50,
                 max_concurrency: int = 5):
        self.processor = processor
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
    
    async def process(self, items: List[T]) -> List[Any]:
        """Process items in batches with concurrency control."""
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        logger.info(
            "Processing items in batches",
            total_items=len(items),
            num_batches=len(batches),
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency
        )
        
        batch_coroutines = [self._process_batch(batch) for batch in batches]
        batch_results = await gather_with_concurrency(
            batch_coroutines, 
            max_concurrency=self.max_concurrency
        )
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results
    
    async def _process_batch(self, batch: List[T]) -> List[Any]:
        """Process a single batch of items."""
        start_time = time.time()
        
        try:
            coroutines = [self.processor(item) for item in batch]
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            processing_time = time.time() - start_time
            
            logger.debug(
                "Batch processed",
                batch_size=len(batch),
                processing_time=processing_time,
                success_count=sum(1 for r in results if not isinstance(r, Exception)),
                error_count=sum(1 for r in results if isinstance(r, Exception))
            )
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Batch processing failed",
                batch_size=len(batch),
                processing_time=processing_time,
                error=str(e)
            )
            raise


@asynccontextmanager
async def async_timer(operation_name: str = "Operation"):
    """
    Async context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
    
    Examples:
        >>> async with async_timer("Data processing"):
        ...     await process_data()
        ...     # Will log the execution time
    """
    start_time = time.time()
    logger.info(f"{operation_name} started")
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger.info(
            f"{operation_name} completed",
            execution_time=execution_time
        )


class AsyncRetry:
    """
    Async retry mechanism with exponential backoff.
    
    Provides configurable retry logic for async operations with
    exponential backoff and maximum retry attempts.
    
    Examples:
        >>> retry = AsyncRetry(max_attempts=3, base_delay=1.0)
        >>> 
        >>> @retry
        ... async def unreliable_operation():
        ...     # This might fail sometimes
        ...     if random.random() < 0.7:
        ...         raise Exception("Random failure")
        ...     return "Success"
        >>> 
        >>> result = await unreliable_operation()
    """
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_factor: float = 2.0,
                 exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_factor = exponential_factor
        self.exceptions = exceptions
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator for adding retry logic to async functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    
                    if attempt == self.max_attempts:
                        logger.error(
                            "Function failed after all retry attempts",
                            function=func.__name__,
                            attempts=attempt,
                            error=str(e)
                        )
                        raise
                    
                    delay = min(
                        self.base_delay * (self.exponential_factor ** (attempt - 1)),
                        self.max_delay
                    )
                    
                    logger.warning(
                        "Function attempt failed, retrying",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                        delay=delay,
                        error=str(e)
                    )
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper


class AsyncRateLimiter:
    """
    Async rate limiter for controlling request frequency.
    
    Limits the rate of async operations using a token bucket algorithm
    to prevent overwhelming external APIs or services.
    
    Examples:
        >>> rate_limiter = AsyncRateLimiter(rate=10, per=60)  # 10 requests per minute
        >>> 
        >>> async def make_api_call():
        ...     async with rate_limiter:
        ...         # Make actual API call
        ...         return await call_external_api()
    """
    
    def __init__(self, rate: int, per: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of operations allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
    
    async def acquire(self):
        """Acquire a token from the rate limiter."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on time passed
            self.tokens = min(
                self.rate,
                self.tokens + time_passed * (self.rate / self.per)
            )
            
            if self.tokens < 1:
                # Need to wait for next token
                wait_time = (1 - self.tokens) * (self.per / self.rate)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


async def async_timeout(coro: Awaitable[T], timeout: float) -> T:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
    
    Returns:
        Result of coroutine execution
    
    Raises:
        asyncio.TimeoutError: If coroutine doesn't complete within timeout
    
    Examples:
        >>> async def slow_operation():
        ...     await asyncio.sleep(10)
        ...     return "Done"
        >>> 
        >>> try:
        ...     result = await async_timeout(slow_operation(), timeout=5.0)
        ... except asyncio.TimeoutError:
        ...     print("Operation timed out")
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout} seconds")
        raise


def async_lru_cache(maxsize: int = 128, ttl: Optional[float] = None):
    """
    LRU cache decorator for async functions with optional TTL.
    
    Args:
        maxsize: Maximum cache size
        ttl: Time to live in seconds (optional)
    
    Examples:
        >>> @async_lru_cache(maxsize=100, ttl=300)  # Cache for 5 minutes
        ... async def expensive_operation(param: str) -> dict:
        ...     await asyncio.sleep(2)  # Simulate expensive operation
        ...     return {"param": param, "result": "computed"}
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))
            current_time = time.time()
            
            # Check if cached result exists and is still valid
            if key in cache:
                if ttl is None or (current_time - cache_times[key]) < ttl:
                    return cache[key]
                else:
                    # TTL expired, remove from cache
                    del cache[key]
                    del cache_times[key]
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = min(cache_times.keys(), key=cache_times.get)
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[key] = result
            cache_times[key] = current_time
            
            return result
        
        return wrapper
    return decorator 