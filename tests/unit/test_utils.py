"""
Unit tests for utility functions.

Tests the text processing utilities, async helpers, and logging functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.utils.text_utils import (
    clean_text, extract_code_blocks, extract_python_imports,
    extract_function_signatures, calculate_similarity,
    generate_text_hash, truncate_text, extract_urls,
    remove_html_tags, split_into_sentences, TextSummarizer
)
from src.utils.async_utils import (
    run_async, gather_with_concurrency, AsyncBatch,
    AsyncRetry, AsyncRateLimiter
)
from src.utils.logger import get_logger


class TestTextUtils:
    """Test cases for text utility functions."""

    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        text = "  This is   some text with\\nextra   spaces  "
        result = clean_text(text, preserve_code=False)
        assert result == "This is some text with extra spaces"

    def test_extract_code_blocks_fenced(self):
        """Test extraction of fenced code blocks."""
        text = '''
Here's some Python code:
```python
def hello():
    print("Hello, world!")
```
And that's it.
'''
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].language == 'python'
        assert 'def hello():' in blocks[0].content
        assert not blocks[0].is_inline

    def test_extract_code_blocks_indented(self):
        """Test extraction of indented code blocks."""
        text = '''
Here's some indented code:

    def hello():
        print("Hello, world!")
    
    x = 42

Regular text continues here.
'''
        blocks = extract_code_blocks(text)
        # Should find the indented block
        indented_blocks = [b for b in blocks if not b.is_inline and 'def hello():' in b.content]
        assert len(indented_blocks) >= 1

    def test_extract_python_imports(self):
        """Test Python import extraction."""
        code = '''
import os
from typing import List, Dict
import numpy as np
from datetime import datetime, timedelta

def some_function():
    pass
'''
        imports = extract_python_imports(code)
        assert 'import os' in imports
        assert 'import numpy as np' in imports
        assert any('from typing import' in imp for imp in imports)

    def test_extract_function_signatures(self):
        """Test Python function signature extraction."""
        code = '''
def hello_world():
    print("Hello!")

async def fetch_data(url: str) -> dict:
    return {}

class MyClass:
    def method(self, param: int) -> str:
        return str(param)
'''
        signatures = extract_function_signatures(code)
        assert len(signatures) >= 3
        assert any('def hello_world()' in sig for sig in signatures)
        assert any('async def fetch_data' in sig for sig in signatures)
        assert any('def method' in sig for sig in signatures)

    @pytest.mark.parametrize("text1,text2,method,expected_range", [
        ("hello world", "hello python", "sequence", (0.3, 0.8)),
        ("python programming", "python coding", "jaccard", (0.1, 0.8)),
        ("machine learning", "deep learning", "cosine", (0.3, 0.8)),
    ])
    def test_calculate_similarity(self, text1, text2, method, expected_range):
        """Test similarity calculation methods."""
        similarity = calculate_similarity(text1, text2, method)
        assert expected_range[0] <= similarity <= expected_range[1]

    def test_generate_text_hash(self):
        """Test text hashing functionality."""
        text = "Hello, world!"
        
        # Test MD5
        md5_hash = generate_text_hash(text, 'md5')
        assert len(md5_hash) == 32
        assert md5_hash == generate_text_hash(text, 'md5')  # Consistency
        
        # Test SHA256
        sha256_hash = generate_text_hash(text, 'sha256')
        assert len(sha256_hash) == 64
        
        # Different algorithms should produce different hashes
        assert md5_hash != sha256_hash

    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a long text that needs to be truncated"
        truncated = truncate_text(text, 20)
        assert len(truncated) == 20
        assert truncated.endswith("...")
        
        # Short text should not be truncated
        short_text = "Short"
        assert truncate_text(short_text, 20) == short_text

    def test_extract_urls(self):
        """Test URL extraction from text."""
        text = "Visit https://example.com or http://test.org for more info."
        urls = extract_urls(text)
        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "http://test.org" in urls

    def test_remove_html_tags(self):
        """Test HTML tag removal."""
        html_text = "<p>Hello <b>world</b>!</p>"
        clean = remove_html_tags(html_text)
        assert clean == "Hello world!"
        
        # Test with HTML entities
        html_with_entities = "&lt;p&gt;Hello &amp; goodbye&lt;/p&gt;"
        clean_entities = remove_html_tags(html_with_entities)
        assert "<p>Hello & goodbye</p>" == clean_entities

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        text = "Hello world. How are you? Fine, thanks!"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert "Hello world" in sentences[0]
        assert "How are you" in sentences[1]
        assert "Fine, thanks!" in sentences[2]

    def test_text_summarizer(self):
        """Test text summarization."""
        long_text = """
        Machine learning is a subset of artificial intelligence. 
        It focuses on algorithms that can learn from data. 
        Deep learning is a subset of machine learning. 
        Neural networks are the foundation of deep learning. 
        Python is a popular programming language for machine learning.
        """
        
        summarizer = TextSummarizer(max_sentences=2)
        summary = summarizer.summarize(long_text.strip())
        
        # Should be shorter than original
        original_sentences = split_into_sentences(long_text.strip())
        summary_sentences = split_into_sentences(summary)
        assert len(summary_sentences) <= min(2, len(original_sentences))


class TestAsyncUtils:
    """Test cases for async utility functions."""

    @pytest.mark.asyncio
    async def test_run_async(self):
        """Test running sync function in async context."""
        def blocking_operation(x: int) -> int:
            return x * 2
        
        result = await run_async(blocking_operation, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self):
        """Test gathering coroutines with concurrency control."""
        async def mock_coroutine(value: int) -> int:
            await asyncio.sleep(0.01)  # Simulate async work
            return value * 2
        
        coroutines = [mock_coroutine(i) for i in range(5)]
        results = await gather_with_concurrency(coroutines, max_concurrency=2)
        
        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test async batch processing."""
        async def process_item(item: int) -> int:
            await asyncio.sleep(0.001)  # Simulate processing
            return item * 2
        
        items = list(range(10))
        batch_processor = AsyncBatch(process_item, batch_size=3, max_concurrency=2)
        results = await batch_processor.process(items)
        
        assert len(results) == 10
        # Results should be doubled values (but may be in different order due to async)
        expected_results = [i * 2 for i in items]
        assert sorted(results) == sorted(expected_results)

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async retry mechanism with successful operation."""
        call_count = 0
        
        @AsyncRetry(max_attempts=3, base_delay=0.01)
        async def sometimes_failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Simulated failure")
            return "Success"
        
        result = await sometimes_failing_operation()
        assert result == "Success"
        assert call_count == 2  # Failed once, succeeded on second attempt

    @pytest.mark.asyncio
    async def test_async_retry_max_attempts(self):
        """Test async retry mechanism reaching max attempts."""
        @AsyncRetry(max_attempts=2, base_delay=0.01)
        async def always_failing_operation():
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            await always_failing_operation()

    @pytest.mark.asyncio
    async def test_async_rate_limiter(self):
        """Test async rate limiting."""
        rate_limiter = AsyncRateLimiter(rate=2, per=0.1)  # 2 operations per 0.1 seconds
        
        start_time = asyncio.get_event_loop().time()
        
        # First two operations should be fast
        async with rate_limiter:
            pass
        async with rate_limiter:
            pass
        
        # Third operation should be delayed
        async with rate_limiter:
            pass
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        # Should take at least some time due to rate limiting
        assert elapsed_time >= 0.05  # Some delay expected


class TestLogger:
    """Test cases for logging functionality."""

    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger(__name__)
        assert logger is not None
        
        # Test with context
        logger_with_context = get_logger(__name__, service="test_service")
        assert logger_with_context is not None

    def test_logger_context(self):
        """Test logger context binding."""
        logger = get_logger(__name__, test_context="test_value")
        # This is mainly testing that it doesn't crash
        # Full functionality testing would require more complex setup
        assert logger is not None 