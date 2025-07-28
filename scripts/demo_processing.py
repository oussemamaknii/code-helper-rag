#!/usr/bin/env python3
"""
Standalone Processing Demo - Python Code Helper RAG System

This demonstrates the data processing capabilities without requiring
API keys or external dependencies. Shows real Python code analysis.
"""

import ast
import re
from typing import List, Dict, Any
from dataclasses import dataclass

print("🐍 Python Code Helper - Processing Demo")
print("=" * 50)
print("Demonstrating intelligent code processing capabilities")
print("Using REAL Python code analysis (no mocking)")
print("=" * 50)

@dataclass
class CodeChunk:
    """Represents a processed code chunk."""
    content: str
    chunk_type: str
    metadata: Dict[str, Any]
    size: int

class SimplePythonAnalyzer:
    """Simplified Python AST analyzer."""
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code using AST."""
        try:
            tree = ast.parse(code)
            
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity_score': 0,
                'docstrings': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'has_docstring': ast.get_docstring(node) is not None
                    }
                    analysis['functions'].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'has_docstring': ast.get_docstring(node) is not None
                    }
                    analysis['classes'].append(class_info)
                    
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append(alias.name)
                    else:
                        module = node.module or ''
                        for alias in node.names:
                            analysis['imports'].append(f"{module}.{alias.name}")
            
            # Simple complexity calculation
            analysis['complexity_score'] = len(analysis['functions']) * 2 + len(analysis['classes']) * 3
            
            return analysis
            
        except SyntaxError as e:
            return {'error': f"Syntax error: {e}"}

class SimpleChunker:
    """Simplified intelligent chunking."""
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        self.analyzer = SimplePythonAnalyzer()
    
    def chunk_code(self, code: str) -> List[CodeChunk]:
        """Chunk code intelligently based on structure."""
        chunks = []
        
        # Analyze the code structure
        analysis = self.analyzer.analyze_code(code)
        
        if 'error' in analysis:
            # Fallback to simple text chunking
            return self._simple_text_chunk(code)
        
        # Try to chunk by functions and classes
        lines = code.split('\n')
        current_chunk = []
        current_size = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this line starts a function or class
            if line.strip().startswith(('def ', 'class ')):
                # If we have content, save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk), 
                        'code_block',
                        current_size
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Extract the function/class
                func_lines = [line]
                func_size = line_size
                i += 1
                
                # Get the full function/class body
                indent_level = len(line) - len(line.lstrip())
                while i < len(lines):
                    next_line = lines[i]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    if (next_line.strip() and 
                        next_indent <= indent_level and 
                        not next_line.strip().startswith(('#', '"""', "'''"))):
                        break
                    
                    func_lines.append(next_line)
                    func_size += len(next_line) + 1
                    i += 1
                
                # Create function/class chunk
                chunk_type = 'function' if line.strip().startswith('def ') else 'class'
                chunks.append(self._create_chunk(
                    '\n'.join(func_lines), 
                    chunk_type,
                    func_size
                ))
                
            else:
                current_chunk.append(line)
                current_size += line_size
                i += 1
                
                # If chunk is getting too large, split it
                if current_size > self.max_chunk_size:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk), 
                        'code_block',
                        current_size
                    ))
                    current_chunk = []
                    current_size = 0
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                '\n'.join(current_chunk), 
                'code_block',
                current_size
            ))
        
        return chunks
    
    def _simple_text_chunk(self, text: str) -> List[CodeChunk]:
        """Fallback simple text chunking."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1
            if current_size + word_size > self.max_chunk_size and current_chunk:
                chunks.append(self._create_chunk(
                    ' '.join(current_chunk),
                    'text_block',
                    current_size
                ))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(self._create_chunk(
                ' '.join(current_chunk),
                'text_block',
                current_size
            ))
        
        return chunks
    
    def _create_chunk(self, content: str, chunk_type: str, size: int) -> CodeChunk:
        """Create a code chunk with metadata."""
        metadata = {}
        
        if chunk_type in ['function', 'class']:
            # Extract name
            first_line = content.split('\n')[0].strip()
            if chunk_type == 'function':
                match = re.search(r'def\s+(\w+)', first_line)
                if match:
                    metadata['name'] = match.group(1)
            elif chunk_type == 'class':
                match = re.search(r'class\s+(\w+)', first_line)
                if match:
                    metadata['name'] = match.group(1)
        
        # Count lines
        metadata['lines'] = len(content.split('\n'))
        
        # Check for docstrings
        metadata['has_docstring'] = '"""' in content or "'''" in content
        
        return CodeChunk(
            content=content,
            chunk_type=chunk_type,
            metadata=metadata,
            size=size
        )

def demo_code_analysis():
    """Demonstrate code analysis capabilities."""
    print("\n🔍 1. Python Code Analysis Demo")
    print("-" * 40)
    
    # Sample Python code for analysis
    sample_code = '''
import os
import sys
from typing import List, Dict, Optional

class DataProcessor:
    """A class for processing data efficiently."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration."""
        self.config = config
        self.processed_count = 0
    
    def process_item(self, item: str) -> Optional[str]:
        """Process a single item and return result."""
        if not item or not item.strip():
            return None
        
        # Clean the item
        cleaned = item.strip().lower()
        
        # Apply transformations
        result = self._apply_transformations(cleaned)
        
        self.processed_count += 1
        return result
    
    def _apply_transformations(self, text: str) -> str:
        """Apply various text transformations."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            'processed_count': self.processed_count,
            'config_items': len(self.config)
        }

def main():
    """Main processing function."""
    processor = DataProcessor({'mode': 'strict', 'debug': True})
    
    items = ['Hello World!', '  Python  ', '', 'Data@Science#']
    results = []
    
    for item in items:
        result = processor.process_item(item)
        if result:
            results.append(result)
    
    print(f"Processed {len(results)} items")
    print(f"Stats: {processor.get_stats()}")
    
    return results

if __name__ == "__main__":
    main()
'''
    
    print("📄 Analyzing sample Python code...")
    print(f"Code length: {len(sample_code)} characters")
    
    analyzer = SimplePythonAnalyzer()
    analysis = analyzer.analyze_code(sample_code)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Functions found: {len(analysis['functions'])}")
    for func in analysis['functions']:
        print(f"     • {func['name']}() - line {func['line']} - {len(func['args'])} args")
    
    print(f"   Classes found: {len(analysis['classes'])}")
    for cls in analysis['classes']:
        print(f"     • {cls['name']} - line {cls['line']} - {len(cls['methods'])} methods")
    
    print(f"   Imports found: {len(analysis['imports'])}")
    for imp in analysis['imports'][:5]:  # Show first 5
        print(f"     • {imp}")
    
    print(f"   Complexity score: {analysis['complexity_score']}")

def demo_intelligent_chunking():
    """Demonstrate intelligent chunking."""
    print("\n🧩 2. Intelligent Chunking Demo")
    print("-" * 40)
    
    # Sample code with different structures
    sample_code = '''
# Configuration and imports
import json
import logging
from pathlib import Path

# Global configuration
CONFIG = {
    'debug': True,
    'max_size': 1000
}

class QuickSort:
    """Implementation of QuickSort algorithm."""
    
    def sort(self, arr: List[int]) -> List[int]:
        """Sort array using quicksort."""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return self.sort(left) + middle + self.sort(right)

def binary_search(arr: List[int], target: int) -> int:
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

# Utility functions
def setup_logging():
    logging.basicConfig(level=logging.INFO)

def main():
    print("Algorithm demo")
'''
    
    print("📄 Chunking sample algorithm code...")
    print(f"Original code length: {len(sample_code)} characters")
    
    chunker = SimpleChunker(max_chunk_size=300)
    chunks = chunker.chunk_code(sample_code)
    
    print(f"\n🧩 Chunking Results:")
    print(f"   Total chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n   Chunk {i}: {chunk.chunk_type}")
        print(f"     Size: {chunk.size} characters ({chunk.metadata.get('lines', 0)} lines)")
        if 'name' in chunk.metadata:
            print(f"     Name: {chunk.metadata['name']}")
        print(f"     Has docstring: {chunk.metadata.get('has_docstring', False)}")
        
        # Show first few lines
        lines = chunk.content.split('\n')[:3]
        for line in lines:
            if line.strip():
                print(f"     Preview: {line.strip()[:50]}...")
                break

def demo_processing_pipeline():
    """Demonstrate the processing pipeline."""
    print("\n⚙️ 3. Processing Pipeline Demo")
    print("-" * 40)
    
    # Simulate processing different types of content
    contents = [
        ("GitHub Python file", '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class MathUtils:
    @staticmethod
    def factorial(n):
        return 1 if n <= 1 else n * MathUtils.factorial(n-1)
'''),
        ("Stack Overflow Q&A", '''
Question: How to reverse a string in Python?

Answer: There are several ways to reverse a string in Python:

1. Using slicing:
```python
text = "hello"
reversed_text = text[::-1]
print(reversed_text)  # "olleh"
```

2. Using reversed() function:
```python
text = "hello"
reversed_text = ''.join(reversed(text))
```

3. Using a loop:
```python
def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result
```
'''),
    ]
    
    chunker = SimpleChunker(max_chunk_size=200)
    
    total_chunks = 0
    for content_type, content in contents:
        print(f"\n📝 Processing: {content_type}")
        print(f"   Content length: {len(content)} characters")
        
        chunks = chunker.chunk_code(content)
        total_chunks += len(chunks)
        
        print(f"   Chunks created: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"     {i}. {chunk.chunk_type} ({chunk.size} chars)")
    
    print(f"\n📊 Pipeline Summary:")
    print(f"   Total content types processed: {len(contents)}")
    print(f"   Total chunks created: {total_chunks}")
    print(f"   Average chunks per content: {total_chunks/len(contents):.1f}")

async def main():
    """Main demonstration function."""
    try:
        demo_code_analysis()
        demo_intelligent_chunking()
        demo_processing_pipeline()
        
        print(f"\n🎉 Processing Demo Complete!")
        print(f"\n📊 Summary:")
        print(f"   ✅ Python AST analysis working correctly")
        print(f"   ✅ Intelligent code chunking by structure")
        print(f"   ✅ Metadata extraction and classification")
        print(f"   ✅ Multi-content type processing")
        
        print(f"\n🌟 Real Processing Features Demonstrated:")
        print(f"   • Abstract Syntax Tree (AST) analysis")
        print(f"   • Function and class extraction")
        print(f"   • Import dependency analysis")
        print(f"   • Intelligent code chunking")
        print(f"   • Metadata enrichment")
        print(f"   • Multi-format content handling")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 