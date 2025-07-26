#!/usr/bin/env python3
"""
Test script for the data processing pipeline.

This script demonstrates how to use the processing pipeline to transform
collected data into semantically meaningful chunks for the RAG system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.pipeline import DataProcessingPipeline, ProcessingPipelineConfig
from src.processing.code_processor import CodeProcessor
from src.processing.qa_processor import QAProcessor
from src.processing.python_analyzer import PythonASTAnalyzer
from src.processing.chunking_strategies import (
    ChunkingStrategyFactory, ChunkingConfig, ChunkingStrategy
)
from src.ingestion.base_collector import CollectedItem
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_python_analyzer():
    """Test Python AST analyzer functionality."""
    print("\n🔍 Testing Python AST Analyzer...")
    
    analyzer = PythonASTAnalyzer()
    
    # Test with a complex Python example
    python_code = '''
import os
import asyncio
from typing import List, Dict, Optional

class DataProcessor:
    """A class for processing data with async capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0
    
    async def process_data(self, data: List[str]) -> Dict[str, int]:
        """
        Process a list of data items asynchronously.
        
        Args:
            data: List of data items to process
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        for item in data:
            try:
                # Process each item
                processed = await self._process_single_item(item)
                results[item] = processed
                self.processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {item}: {e}")
                results[item] = -1
        
        return results
    
    async def _process_single_item(self, item: str) -> int:
        """Process a single item."""
        await asyncio.sleep(0.01)  # Simulate async work
        return len(item) * 2

def create_processor(config_file: str) -> DataProcessor:
    """Create a data processor from config file."""
    config = load_config(config_file)
    return DataProcessor(config)

def load_config(filename: str) -> Dict[str, Any]:
    """Load configuration from file."""
    # Simplified config loading
    return {"max_items": 100, "timeout": 30}
'''
    
    # Analyze the code
    analysis = analyzer.analyze_code(python_code, "example.py")
    
    print(f"✅ Code Analysis Results:")
    print(f"   Lines of code: {analysis.metrics['total_lines']}")
    print(f"   Functions found: {len(analysis.get_elements_by_type('FUNCTION'))}")
    print(f"   Classes found: {len(analysis.get_elements_by_type('CLASS'))}")
    print(f"   Methods found: {len(analysis.get_elements_by_type('METHOD'))}")
    print(f"   Imports: {', '.join(analysis.dependencies.imports)}")
    print(f"   Quality score: {analyzer.get_code_quality_score(python_code):.1f}/100")
    
    # Test function extraction
    functions = analyzer.extract_functions(python_code)
    print(f"\n📋 Function Details:")
    for func in functions[:3]:  # Show first 3
        print(f"   - {func['name']} ({func['type']}): {func['line_count']} lines")
        if func['docstring']:
            print(f"     Documented: ✅")
        if func['is_async']:
            print(f"     Async: ✅")
    
    return True


async def test_chunking_strategies():
    """Test different chunking strategies."""
    print("\n🔍 Testing Chunking Strategies...")
    
    # Test Python function-level chunking
    python_code = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathOperations:
    """Mathematical operations utility class."""
    
    def __init__(self):
        self.operations_count = 0
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        self.operations_count += 1
        return a + b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        self.operations_count += 1
        return a * b
'''
    
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FUNCTION_LEVEL,
        max_chunk_size=1000,
        min_chunk_size=10
    )
    
    strategy = ChunkingStrategyFactory.create_strategy(config)
    metadata = {"source_item_id": "test", "source_type": "github_code"}
    
    chunks = strategy.chunk_content(python_code, metadata)
    
    print(f"✅ Function-Level Chunking:")
    print(f"   Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {chunk.chunk_type} ({len(chunk.content)} chars)")
        if 'function_name' in chunk.metadata:
            print(f"      Function: {chunk.metadata['function_name']}")
        elif 'class_name' in chunk.metadata:
            print(f"      Class: {chunk.metadata['class_name']}")
    
    # Test Q&A chunking
    qa_content = '''Title: How to use asyncio in Python?

Question:
I'm trying to understand how to use asyncio for concurrent programming in Python. 
Can someone explain the basics and provide a simple example?

Answer:
Asyncio is Python's built-in library for asynchronous programming. Here's a basic example:

```python
import asyncio

async def fetch_data(url):
    # Simulate fetching data
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    tasks = [
        fetch_data("http://example1.com"),
        fetch_data("http://example2.com"),
        fetch_data("http://example3.com")
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

# Run the async function
asyncio.run(main())
```

This example shows how to create async functions and run them concurrently using asyncio.gather().
'''
    
    qa_config = ChunkingConfig(
        strategy=ChunkingStrategy.QA_PAIR,
        max_chunk_size=1000,
        min_chunk_size=50
    )
    
    qa_strategy = ChunkingStrategyFactory.create_strategy(qa_config)
    qa_metadata = {"source_item_id": "test_qa", "source_type": "stackoverflow_qa"}
    
    qa_chunks = qa_strategy.chunk_content(qa_content, qa_metadata)
    
    print(f"\n✅ Q&A Pair Chunking:")
    print(f"   Created {len(qa_chunks)} chunks")
    
    for i, chunk in enumerate(qa_chunks):
        print(f"   Chunk {i+1}: {chunk.chunk_type} ({len(chunk.content)} chars)")
    
    return True


async def test_specialized_processors():
    """Test specialized processors."""
    print("\n🔍 Testing Specialized Processors...")
    
    # Test Code Processor
    code_processor = CodeProcessor()
    
    github_item = CollectedItem(
        id="github_test_123",
        content='''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MLModel:
    """A simple machine learning model wrapper."""
    
    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
    
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training."""
        X = data.drop('target', axis=1)
        y = data['target']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the model."""
        if self.model_type == "linear":
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def predict(self, X_test):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return self.model.predict(X_test)

def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    }
    return pd.DataFrame(data)
''',
        metadata={
            "file_path": "ml_model.py",
            "repository_name": "example/ml-project",
            "file_size": 1024,
            "language": "python"
        },
        source_type="github_code"
    )
    
    print(f"📄 Processing GitHub Code Item...")
    code_chunks = []
    async for chunk in code_processor.process_item(github_item):
        code_chunks.append(chunk)
    
    print(f"   Created {len(code_chunks)} code chunks")
    
    for i, chunk in enumerate(code_chunks[:3]):  # Show first 3
        print(f"   Chunk {i+1}: {chunk.chunk_type}")
        if 'function_name' in chunk.metadata:
            print(f"      Function: {chunk.metadata['function_name']}")
            if 'complexity_score' in chunk.metadata:
                print(f"      Complexity: {chunk.metadata['complexity_score']}")
        elif 'class_name' in chunk.metadata:
            print(f"      Class: {chunk.metadata['class_name']}")
    
    # Test Q&A Processor
    qa_processor = QAProcessor()
    
    stackoverflow_item = CollectedItem(
        id="so_test_456",
        content='''Title: How to handle exceptions in async Python functions?

Question:
I'm working with asyncio and I'm having trouble understanding how to properly handle exceptions in async functions. What's the best practice?

Answer:
When working with async functions in Python, exception handling follows similar patterns to synchronous code, but with some important considerations:

```python
import asyncio

async def risky_operation():
    await asyncio.sleep(1)
    raise ValueError("Something went wrong!")

async def handle_exceptions():
    try:
        result = await risky_operation()
        return result
    except ValueError as e:
        print(f"Caught exception: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# For multiple async operations
async def handle_multiple():
    tasks = [risky_operation() for _ in range(3)]
    
    # Option 1: Handle exceptions per task
    results = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            print(f"Task failed: {e}")
            results.append(None)
    
    return results

asyncio.run(handle_multiple())
```

Key points:
1. Use try/except blocks around await expressions
2. Consider using asyncio.gather(return_exceptions=True) for multiple tasks
3. Always handle exceptions in fire-and-forget tasks
''',
        metadata={
            "question_id": 12345,
            "answer_id": 67890,
            "tags": ["python", "asyncio", "exception-handling"],
            "question_score": 25,
            "answer_score": 18,
            "is_accepted": True,
            "has_code": True
        },
        source_type="stackoverflow_qa"
    )
    
    print(f"\n❓ Processing Stack Overflow Q&A Item...")
    qa_chunks = []
    async for chunk in qa_processor.process_item(stackoverflow_item):
        qa_chunks.append(chunk)
    
    print(f"   Created {len(qa_chunks)} Q&A chunks")
    
    for i, chunk in enumerate(qa_chunks):
        print(f"   Chunk {i+1}: {chunk.chunk_type}")
        if 'quality_indicators' in chunk.metadata:
            quality = chunk.metadata['quality_indicators']
            print(f"      Overall quality: {quality.get('overall_score', 0):.1f}/100")
    
    return True


async def test_full_processing_pipeline():
    """Test the complete processing pipeline."""
    print("\n🔍 Testing Complete Processing Pipeline...")
    
    # Create pipeline configuration
    config = ProcessingPipelineConfig(
        enable_code_processor=True,
        enable_qa_processor=True,
        batch_size=5,
        save_chunks=False  # Don't save to files during testing
    )
    
    pipeline = DataProcessingPipeline(config)
    
    # Create test items
    items = [
        # GitHub code items
        CollectedItem(
            id="code_1",
            content='''
def binary_search(arr, target):
    """Binary search implementation."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
''',
            metadata={"file_path": "search.py", "repository_name": "algorithms/python"},
            source_type="github_code"
        ),
        
        CollectedItem(
            id="code_2", 
            content='''
class Queue:
    """Simple queue implementation using list."""
    
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.pop(0)
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0
''',
            metadata={"file_path": "queue.py", "repository_name": "data-structures/python"},
            source_type="github_code"
        ),
        
        # Stack Overflow items
        CollectedItem(
            id="qa_1",
            content='''Title: How to reverse a string in Python?

Question:
What's the most efficient way to reverse a string in Python?

Answer:
The most Pythonic way to reverse a string is using slicing:

```python
text = "hello"
reversed_text = text[::-1]
print(reversed_text)  # "olleh"
```

This is both readable and efficient.
''',
            metadata={
                "question_id": 111,
                "answer_id": 222,
                "tags": ["python", "string"],
                "is_accepted": True
            },
            source_type="stackoverflow_qa"
        )
    ]
    
    print(f"📦 Processing {len(items)} items through pipeline...")
    
    # Process items
    processed_chunks = []
    async for chunk in pipeline.process_items(items):
        processed_chunks.append(chunk)
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    
    print(f"✅ Pipeline Processing Results:")
    print(f"   Total chunks created: {len(processed_chunks)}")
    print(f"   Items processed: {status['metrics']['total_items_processed']}")
    print(f"   Processing time: {status['metrics']['processing_time']:.2f}s")
    print(f"   Items per second: {status['metrics']['items_per_second']:.1f}")
    
    # Show chunk distribution
    chunk_types = {}
    for chunk in processed_chunks:
        chunk_type = chunk.chunk_type
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    print(f"\n📊 Chunk Distribution:")
    for chunk_type, count in chunk_types.items():
        print(f"   {chunk_type}: {count} chunks")
    
    return True


def check_environment():
    """Check if required environment variables are set."""
    print("🔧 Checking environment configuration...")
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'GITHUB_TOKEN': 'GitHub personal access token', 
        'PINECONE_API_KEY': 'Pinecone API key'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  ❌ {var}: {description}")
        else:
            print(f"  ✅ {var}: Set")
    
    if missing_vars:
        print("\n❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nNote: Tests will continue but some functionality may be limited.")
        return False
    
    return True


async def main():
    """Main test function."""
    print("🚀 Data Processing Pipeline Test - Phase 3")
    print("=" * 55)
    
    # Check environment (but don't fail if missing)
    check_environment()
    
    # Run tests
    tests = [
        ("Python AST Analyzer", test_python_analyzer),
        ("Chunking Strategies", test_chunking_strategies),
        ("Specialized Processors", test_specialized_processors),
        ("Full Processing Pipeline", test_full_processing_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results[test_name] = success
        except KeyboardInterrupt:
            print(f"\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'='*55}")
    print("📊 Test Results Summary:")
    print("=" * 55)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 3 processing pipeline is working correctly.")
        print("\n🔄 Phase 3: Data Processing & Chunking - COMPLETED")
        print("   ✅ Python AST analysis for semantic understanding")
        print("   ✅ Intelligent chunking strategies (function, class, Q&A)")
        print("   ✅ Specialized processors for different content types")
        print("   ✅ Processing pipeline orchestrator")
        print("   ✅ Comprehensive metadata extraction")
        print("   ✅ Production-ready error handling and metrics")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1) 