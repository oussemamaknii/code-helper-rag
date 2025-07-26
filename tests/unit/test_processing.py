"""
Unit tests for data processing components.

Tests the base processor, Python analyzer, chunking strategies,
specialized processors, and processing pipeline functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.processing.base_processor import (
    BaseProcessor, ProcessedChunk, ProcessingStatus, ProcessingMetrics,
    DefaultChunkValidator
)
from src.processing.python_analyzer import (
    PythonASTAnalyzer, CodeElement, CodeElementType, CodeAnalysis
)
from src.processing.chunking_strategies import (
    ChunkingStrategy, ChunkingConfig, FixedSizeChunkingStrategy,
    FunctionLevelChunkingStrategy, QAPairChunkingStrategy,
    ChunkingStrategyFactory
)
from src.processing.code_processor import CodeProcessor
from src.processing.qa_processor import QAProcessor
from src.processing.pipeline import DataProcessingPipeline, ProcessingPipelineConfig
from src.ingestion.base_collector import CollectedItem


class TestProcessedChunk:
    """Test cases for ProcessedChunk."""
    
    def test_processed_chunk_creation(self):
        """Test creating a processed chunk."""
        chunk = ProcessedChunk(
            id="test_chunk_id",
            content="test chunk content",
            chunk_type="function",
            metadata={"key": "value"},
            source_item_id="source_123",
            source_type="github_code"
        )
        
        assert chunk.id == "test_chunk_id"
        assert chunk.content == "test chunk content"
        assert chunk.chunk_type == "function"
        assert chunk.metadata["key"] == "value"
        assert chunk.source_item_id == "source_123"
        assert chunk.source_type == "github_code"
        assert isinstance(chunk.processed_at, datetime)
    
    def test_processed_chunk_to_dict(self):
        """Test converting processed chunk to dictionary."""
        chunk = ProcessedChunk(
            id="test_chunk_id",
            content="test content",
            chunk_type="class",
            metadata={"complexity": 5},
            source_item_id="source_123",
            source_type="github_code"
        )
        
        chunk_dict = chunk.to_dict()
        
        assert chunk_dict["id"] == "test_chunk_id"
        assert chunk_dict["content"] == "test content"
        assert chunk_dict["chunk_type"] == "class"
        assert chunk_dict["metadata"]["complexity"] == 5
        assert chunk_dict["source_item_id"] == "source_123"
        assert chunk_dict["source_type"] == "github_code"
        assert "processed_at" in chunk_dict
    
    def test_processed_chunk_length(self):
        """Test chunk length calculation."""
        chunk = ProcessedChunk(
            id="test",
            content="Hello World",
            chunk_type="test",
            metadata={},
            source_item_id="source",
            source_type="test"
        )
        
        assert len(chunk) == 11


class TestProcessingMetrics:
    """Test cases for ProcessingMetrics."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ProcessingMetrics(
            processed_items=100,
            successful_items=85,
            failed_items=15,
            chunks_created=200
        )
        
        assert metrics.success_rate == 85.0
        assert abs(metrics.chunks_per_item - 2.35) < 0.01  # 200/85 with tolerance
    
    def test_processing_time_calculation(self):
        """Test processing time calculation."""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()
        
        metrics = ProcessingMetrics(
            processed_items=50,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics.processing_time is not None
        assert metrics.processing_time >= 0
        # items_per_second might be None if processing_time is 0
        if metrics.processing_time > 0:
            assert metrics.items_per_second is not None


class TestDefaultChunkValidator:
    """Test cases for DefaultChunkValidator."""
    
    def test_valid_chunk(self):
        """Test validation of valid chunk."""
        validator = DefaultChunkValidator(
            min_content_length=10,
            max_content_length=1000
        )
        
        chunk = ProcessedChunk(
            id="test_id",
            content="This is a valid chunk with sufficient content",
            chunk_type="function",
            metadata={},
            source_item_id="source",
            source_type="test"
        )
        
        assert validator.validate(chunk) is True
    
    def test_content_too_short(self):
        """Test validation of chunk with content too short."""
        validator = DefaultChunkValidator(min_content_length=20)
        
        chunk = ProcessedChunk(
            id="test_id",
            content="short",
            chunk_type="function",
            metadata={},
            source_item_id="source",
            source_type="test"
        )
        
        assert validator.validate(chunk) is False
    
    def test_required_metadata_missing(self):
        """Test validation with missing required metadata."""
        validator = DefaultChunkValidator(
            required_metadata_keys=["complexity_score"]
        )
        
        chunk = ProcessedChunk(
            id="test_id",
            content="Valid content here",
            chunk_type="function",
            metadata={},
            source_item_id="source",
            source_type="test"
        )
        
        assert validator.validate(chunk) is False


class MockProcessor(BaseProcessor):
    """Mock processor for testing BaseProcessor functionality."""
    
    def __init__(self, chunks_to_return=None, **kwargs):
        super().__init__("MockProcessor", **kwargs)
        self.chunks_to_return = chunks_to_return or []
    
    async def get_supported_types(self):
        return ["mock_type"]
    
    async def process_item(self, item):
        for i, chunk_content in enumerate(self.chunks_to_return):
            yield ProcessedChunk(
                id=f"mock_chunk_{i}",
                content=chunk_content,
                chunk_type="mock",
                metadata={"index": i},
                source_item_id=item.id,
                source_type=item.source_type
            )


class TestBaseProcessor:
    """Test cases for BaseProcessor."""
    
    @pytest.mark.asyncio
    async def test_process_items(self):
        """Test processing multiple items."""
        mock_chunks = ["chunk1", "chunk2", "chunk3"]
        processor = MockProcessor(chunks_to_return=mock_chunks)
        
        items = [
            CollectedItem(
                id="item1",
                content="test content",
                metadata={},
                source_type="mock_type"
            )
        ]
        
        chunks = []
        async for chunk in processor.process_items(items):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0].content == "chunk1"
        assert processor.status == ProcessingStatus.COMPLETED
        assert processor.metrics.successful_items == 1
        assert processor.metrics.chunks_created == 3
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        processor = MockProcessor()
        
        health_status = await processor.health_check()
        
        assert health_status["processor"] == "MockProcessor"
        assert health_status["status"] == "healthy"
        assert "supported_types" in health_status
    
    def test_get_metrics(self):
        """Test getting processor metrics."""
        processor = MockProcessor()
        processor.metrics.processed_items = 10
        processor.metrics.successful_items = 8
        processor.metrics.chunks_created = 20
        
        metrics = processor.get_metrics()
        
        assert metrics["processor_name"] == "MockProcessor"
        assert metrics["processed_items"] == 10
        assert metrics["successful_items"] == 8
        assert metrics["chunks_created"] == 20


class TestPythonASTAnalyzer:
    """Test cases for PythonASTAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = PythonASTAnalyzer()
        assert analyzer is not None
    
    def test_valid_python_check(self):
        """Test valid Python code detection."""
        analyzer = PythonASTAnalyzer()
        
        valid_code = "def hello():\n    print('Hello, World!')"
        invalid_code = "def hello(\n    print('Invalid syntax'"
        
        assert analyzer.is_valid_python(valid_code) is True
        assert analyzer.is_valid_python(invalid_code) is False
    
    def test_analyze_simple_function(self):
        """Test analyzing a simple function."""
        analyzer = PythonASTAnalyzer()
        
        code = """
def add_numbers(a, b):
    \"\"\"Add two numbers together.\"\"\"
    return a + b
"""
        
        analysis = analyzer.analyze_code(code)
        
        assert len(analysis.elements) == 1
        assert analysis.elements[0].element_type == CodeElementType.FUNCTION
        assert analysis.elements[0].name == "add_numbers"
        assert analysis.elements[0].docstring == "Add two numbers together."
        assert "a" in analysis.elements[0].metadata["args"]
        assert "b" in analysis.elements[0].metadata["args"]
    
    def test_analyze_class_with_methods(self):
        """Test analyzing a class with methods."""
        analyzer = PythonASTAnalyzer()
        
        code = """
class Calculator:
    \"\"\"A simple calculator class.\"\"\"
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
"""
        
        analysis = analyzer.analyze_code(code)
        
        # Should find 1 class and 2 methods
        classes = analysis.get_elements_by_type(CodeElementType.CLASS)
        methods = analysis.get_elements_by_type(CodeElementType.METHOD)
        
        assert len(classes) == 1
        assert len(methods) == 2
        
        assert classes[0].name == "Calculator"
        assert classes[0].docstring == "A simple calculator class."
        
        method_names = [m.name for m in methods]
        assert "add" in method_names
        assert "multiply" in method_names
    
    def test_extract_functions(self):
        """Test extracting function information."""
        analyzer = PythonASTAnalyzer()
        
        code = """
def simple_func():
    pass

async def async_func(param1, param2="default"):
    \"\"\"An async function.\"\"\"
    await some_operation()
"""
        
        functions = analyzer.extract_functions(code)
        
        assert len(functions) == 2
        
        simple_func = next(f for f in functions if f["name"] == "simple_func")
        async_func = next(f for f in functions if f["name"] == "async_func")
        
        assert simple_func["type"] == "function"
        assert simple_func["is_async"] is False
        
        assert async_func["type"] == "function"
        assert async_func["is_async"] is True
        assert async_func["docstring"] == "An async function."
        assert "param1" in async_func["args"]
    
    def test_extract_imports(self):
        """Test extracting import statements."""
        analyzer = PythonASTAnalyzer()
        
        code = """
import os
import sys
from typing import List, Dict
from collections import defaultdict
"""
        
        imports, from_imports = analyzer.extract_imports(code)
        
        assert "os" in imports
        assert "sys" in imports
        assert "typing" in from_imports
        assert "List" in from_imports["typing"]
        assert "Dict" in from_imports["typing"]
        assert "collections" in from_imports
        assert "defaultdict" in from_imports["collections"]
    
    def test_code_quality_score(self):
        """Test code quality scoring."""
        analyzer = PythonASTAnalyzer()
        
        good_code = '''
def well_documented_function(param1: int, param2: str) -> bool:
    """
    This function is well documented and has type hints.
    
    Args:
        param1: An integer parameter
        param2: A string parameter
        
    Returns:
        A boolean result
    """
    # This is a helpful comment
    if param1 > 0:
        return param2.isalpha()
    return False
'''
        
        poor_code = '''
def f(x,y):
    return x+y if x>0 else y*2
'''
        
        good_score = analyzer.get_code_quality_score(good_code)
        poor_score = analyzer.get_code_quality_score(poor_code)
        
        assert good_score > poor_score
        assert good_score > 50  # Should be relatively high
        assert poor_score <= 50  # Should be relatively low


class TestChunkingStrategies:
    """Test cases for chunking strategies."""
    
    def test_chunking_config(self):
        """Test chunking configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FUNCTION_LEVEL,
            max_chunk_size=1000,
            chunk_overlap=100,
            min_chunk_size=50
        )
        
        assert config.strategy == ChunkingStrategy.FUNCTION_LEVEL
        assert config.max_chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50
    
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=20
        )
        
        strategy = FixedSizeChunkingStrategy(config)
        
        content = "This is a long piece of text that needs to be split into multiple chunks for processing."
        metadata = {"source_item_id": "test", "source_type": "test"}
        
        chunks = strategy.chunk_content(content, metadata)
        
        assert len(chunks) > 1  # Should create multiple chunks
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chunk_size + 10  # Allow for boundary flexibility
            assert chunk.chunk_type == "fixed_size"
    
    def test_function_level_chunking(self):
        """Test function-level chunking strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FUNCTION_LEVEL,
            max_chunk_size=1000,
            min_chunk_size=10  # Lower minimum for testing
        )
        
        strategy = FunctionLevelChunkingStrategy(config)
        
        python_code = '''import os

def function_one():
    """First function."""
    return "one"

def function_two(param):
    """Second function."""
    return param * 2

class MyClass:
    """A test class."""
    
    def method_one(self):
        return "method"
'''
        
        metadata = {"source_item_id": "test", "source_type": "github_code"}
        
        chunks = strategy.chunk_content(python_code, metadata)
        
        # Should create chunks for functions, methods, and class
        # Note: module chunk might not be created if module-level content is too short
        chunk_types = [chunk.chunk_type for chunk in chunks]
        
        assert "function" in chunk_types
        assert "class" in chunk_types
        # Module chunk is optional - only created if module-level code is substantial enough
        
        # Check function chunk metadata
        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) >= 2  # Should have at least 2 functions
        
        func_names = [c.metadata.get("function_name") for c in function_chunks]
        assert "function_one" in func_names
        assert "function_two" in func_names

    def test_qa_pair_chunking(self):
        """Test Q&A pair chunking strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.QA_PAIR,
            max_chunk_size=1000
        )
        
        strategy = QAPairChunkingStrategy(config)
        
        qa_content = '''Title: How to use Python lists?

Question:
I want to create and manipulate lists in Python. How do I do this?

Answer:
You can create lists using square brackets:

```python
my_list = [1, 2, 3, 4, 5]
```

Then you can manipulate them with various methods.'''
        
        metadata = {
            "source_item_id": "test",
            "source_type": "stackoverflow_qa",
            "question_id": 123,
            "answer_id": 456
        }
        
        chunks = strategy.chunk_content(qa_content, metadata)
        
        chunk_types = [chunk.chunk_type for chunk in chunks]
        
        assert "question" in chunk_types
        assert "answer" in chunk_types
        
        # Check question chunk
        question_chunks = [c for c in chunks if c.chunk_type == "question"]
        assert len(question_chunks) == 1
        assert "Title:" in question_chunks[0].content
        
        # Check answer chunk
        answer_chunks = [c for c in chunks if c.chunk_type == "answer"]
        assert len(answer_chunks) >= 1
        assert "```python" in answer_chunks[0].content

    def test_chunking_strategy_factory(self):
        """Test chunking strategy factory."""
        # Test fixed size strategy creation
        config1 = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE)
        strategy1 = ChunkingStrategyFactory.create_strategy(config1)
        assert isinstance(strategy1, FixedSizeChunkingStrategy)
        
        # Test function level strategy creation
        config2 = ChunkingConfig(strategy=ChunkingStrategy.FUNCTION_LEVEL)
        strategy2 = ChunkingStrategyFactory.create_strategy(config2)
        assert isinstance(strategy2, FunctionLevelChunkingStrategy)
        
        # Test Q&A strategy creation
        config3 = ChunkingConfig(strategy=ChunkingStrategy.QA_PAIR)
        strategy3 = ChunkingStrategyFactory.create_strategy(config3)
        assert isinstance(strategy3, QAPairChunkingStrategy)
    
    def test_optimal_strategy_selection(self):
        """Test optimal strategy selection based on content type."""
        # GitHub code should use function level
        github_strategy = ChunkingStrategyFactory.get_optimal_strategy(
            "github_code", "def hello(): pass"
        )
        assert github_strategy == ChunkingStrategy.FUNCTION_LEVEL
        
        # Invalid Python should use fixed size
        invalid_python_strategy = ChunkingStrategyFactory.get_optimal_strategy(
            "github_code", "invalid python syntax ("
        )
        assert invalid_python_strategy == ChunkingStrategy.FIXED_SIZE
        
        # Stack Overflow should use Q&A pair
        so_strategy = ChunkingStrategyFactory.get_optimal_strategy(
            "stackoverflow_qa", "Question: How to...?"
        )
        assert so_strategy == ChunkingStrategy.QA_PAIR


class TestCodeProcessor:
    """Test cases for CodeProcessor."""
    
    def test_code_processor_initialization(self):
        """Test code processor initialization."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            processor = CodeProcessor()
            assert processor.name == "CodeProcessor"
    
    @pytest.mark.asyncio
    async def test_get_supported_types(self):
        """Test getting supported types."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            processor = CodeProcessor()
            supported_types = await processor.get_supported_types()
            assert "github_code" in supported_types
    
    @pytest.mark.asyncio
    async def test_process_python_code_item(self):
        """Test processing a Python code item."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            processor = CodeProcessor()
            
            item = CollectedItem(
                id="test_code_item",
                content='''def hello_world():
    """Print hello world."""
    print("Hello, World!")
    
class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
''',
                metadata={
                    "file_path": "hello.py",
                    "repository_name": "test/repo"
                },
                source_type="github_code"
            )
            
            chunks = []
            async for chunk in processor.process_item(item):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            
            # Should have function, class, and possibly module chunks
            chunk_types = [chunk.chunk_type for chunk in chunks]
            assert "function" in chunk_types or "class" in chunk_types


class TestQAProcessor:
    """Test cases for QAProcessor."""
    
    def test_qa_processor_initialization(self):
        """Test Q&A processor initialization."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            processor = QAProcessor()
            assert processor.name == "QAProcessor"
    
    @pytest.mark.asyncio
    async def test_get_supported_types(self):
        """Test getting supported types."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            processor = QAProcessor()
            supported_types = await processor.get_supported_types()
            assert "stackoverflow_qa" in supported_types
    
    @pytest.mark.asyncio
    async def test_process_qa_item(self):
        """Test processing a Q&A item."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            processor = QAProcessor()
            
            item = CollectedItem(
                id="test_qa_item",
                content='''Title: How to create a list in Python?

Question:
I'm new to Python and want to create a list. How do I do this?

Answer:
You can create a list using square brackets:

```python
my_list = [1, 2, 3, 4, 5]
empty_list = []
```

Lists are mutable and can hold different data types.''',
                metadata={
                    "question_id": 12345,
                    "answer_id": 67890,
                    "tags": ["python", "list"],
                    "question_score": 10,
                    "answer_score": 15,
                    "is_accepted": True,
                    "has_code": True
                },
                source_type="stackoverflow_qa"
            )
            
            chunks = []
            async for chunk in processor.process_item(item):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            
            # Should have question and answer chunks
            chunk_types = [chunk.chunk_type for chunk in chunks]
            assert "question" in chunk_types
            assert "answer" in chunk_types


class TestDataProcessingPipeline:
    """Test cases for DataProcessingPipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            config = ProcessingPipelineConfig(
                enable_code_processor=True,
                enable_qa_processor=True
            )
            
            with patch('src.processing.pipeline.CodeProcessor') as mock_code:
                with patch('src.processing.pipeline.QAProcessor') as mock_qa:
                    pipeline = DataProcessingPipeline(config)
                    
                    assert pipeline.config == config
                    assert pipeline.status.value == "idle"
    
    def test_pipeline_config(self):
        """Test pipeline configuration."""
        config = ProcessingPipelineConfig(
            enable_code_processor=False,
            enable_qa_processor=True,
            batch_size=20,
            max_concurrent_processors=5
        )
        
        assert config.enable_code_processor is False
        assert config.enable_qa_processor is True
        assert config.batch_size == 20
        assert config.max_concurrent_processors == 5
    
    def test_get_pipeline_status(self):
        """Test getting pipeline status."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            config = ProcessingPipelineConfig(
                enable_code_processor=False,
                enable_qa_processor=False
            )
            
            pipeline = DataProcessingPipeline(config)
            status = pipeline.get_pipeline_status()
            
            assert "status" in status
            assert "metrics" in status
            assert "processor_metrics" in status
            assert "config" in status
            assert status["config"]["code_processor_enabled"] is False
            assert status["config"]["qa_processor_enabled"] is False 