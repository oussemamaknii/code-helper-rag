"""
Code processor for GitHub repository content.

This module implements a specialized processor for Python code from GitHub
repositories, using AST analysis and intelligent chunking strategies.
"""

from typing import AsyncGenerator, List, Dict, Any
import hashlib

from src.processing.base_processor import BaseProcessor, ProcessedChunk
from src.processing.python_analyzer import PythonASTAnalyzer
from src.processing.chunking_strategies import (
    ChunkingStrategyFactory, ChunkingConfig, ChunkingStrategy
)
from src.ingestion.base_collector import CollectedItem
from src.utils.logger import get_logger
from src.utils.text_utils import extract_code_blocks, clean_text
from src.config.settings import settings

logger = get_logger(__name__)


class CodeProcessor(BaseProcessor):
    """
    Specialized processor for GitHub code content.
    
    Features:
    - Python AST analysis for semantic understanding
    - Function-level and class-level chunking
    - Code quality assessment
    - Import and dependency extraction
    - Intelligent fallback for non-Python files
    """
    
    def __init__(self, **kwargs):
        """Initialize code processor."""
        super().__init__(name="CodeProcessor", **kwargs)
        
        self.analyzer = PythonASTAnalyzer()
        
        # Configure chunking strategies
        self.python_config = ChunkingConfig(
            strategy=ChunkingStrategy.FUNCTION_LEVEL,
            max_chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=10,  # Lower minimum for better chunking
            preserve_structure=True,
            include_context=True
        )
        
        self.fallback_config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=100
        )
    
    async def get_supported_types(self) -> List[str]:
        """Get supported source types."""
        return ["github_code"]
    
    async def process_item(self, item: CollectedItem) -> AsyncGenerator[ProcessedChunk, None]:
        """
        Process a GitHub code item into semantic chunks.
        
        Args:
            item: Collected GitHub code item
            
        Yields:
            ProcessedChunk: Semantic code chunks
        """
        self.logger.debug(
            "Processing GitHub code item",
            item_id=item.id,
            file_path=item.metadata.get('file_path'),
            file_size=len(item.content)
        )
        
        # Determine if this is Python code
        file_path = item.metadata.get('file_path', '')
        is_python = file_path.endswith('.py')
        
        if is_python and self.analyzer.is_valid_python(item.content):
            # Use Python-specific processing
            async for chunk in self._process_python_code(item):
                yield chunk
        else:
            # Use generic text processing
            async for chunk in self._process_generic_code(item):
                yield chunk
    
    async def _process_python_code(self, item: CollectedItem) -> AsyncGenerator[ProcessedChunk, None]:
        """Process Python code with AST analysis."""
        
        # Perform code analysis
        analysis = self.analyzer.analyze_code(item.content, 
                                            filename=item.metadata.get('file_path', 'unknown'))
        
        # Enhanced metadata with analysis results
        enhanced_metadata = {
            **item.metadata,
            'source_item_id': item.id,
            'analysis': {
                'metrics': analysis.metrics,
                'imports': analysis.dependencies.imports,
                'from_imports': analysis.dependencies.from_imports,
                'function_calls': analysis.dependencies.function_calls[:20],  # Limit size
                'syntax_errors': analysis.syntax_errors,
                'quality_score': self.analyzer.get_code_quality_score(item.content)
            },
            'functions': [
                {
                    'name': func['name'],
                    'type': func['type'],
                    'line_count': func['line_count'],
                    'has_docstring': func['docstring'] is not None,
                    'complexity_score': func['complexity_score']
                }
                for func in self.analyzer.extract_functions(item.content)
            ],
            'classes': [
                {
                    'name': cls['name'],
                    'method_count': cls['method_count'],
                    'line_count': cls['line_count'],
                    'has_docstring': cls['docstring'] is not None
                }
                for cls in self.analyzer.extract_classes(item.content)
            ]
        }
        
        # Create chunking strategy
        strategy = ChunkingStrategyFactory.create_strategy(self.python_config)
        
        # Generate chunks
        chunks = strategy.chunk_content(item.content, enhanced_metadata)
        
        for chunk in chunks:
            # Additional processing for code chunks
            await self._enrich_code_chunk(chunk, analysis)
            yield chunk
    
    async def _process_generic_code(self, item: CollectedItem) -> AsyncGenerator[ProcessedChunk, None]:
        """Process non-Python code files."""
        
        # Try to extract code blocks if it's markdown or similar
        code_blocks = extract_code_blocks(item.content)
        
        enhanced_metadata = {
            **item.metadata,
            'source_item_id': item.id,
            'code_blocks_found': len(code_blocks),
            'estimated_language': self._estimate_language(item.content, item.metadata.get('file_path', ''))
        }
        
        # Use fixed-size chunking strategy
        strategy = ChunkingStrategyFactory.create_strategy(self.fallback_config)
        chunks = strategy.chunk_content(item.content, enhanced_metadata)
        
        for chunk in chunks:
            yield chunk
    
    async def _enrich_code_chunk(self, chunk: ProcessedChunk, analysis) -> None:
        """Enrich code chunk with additional analysis."""
        
        # Add chunk-specific analysis
        if chunk.chunk_type == "function":
            # Find the function element for this chunk
            function_name = chunk.metadata.get('function_name')
            if function_name:
                for element in analysis.elements:
                    if element.name == function_name:
                        chunk.metadata.update({
                            'docstring': element.docstring,
                            'full_signature': element.content.split('\n')[0],
                            'dependencies': self._extract_chunk_dependencies(element.content)
                        })
                        break
        
        elif chunk.chunk_type == "class":
            # Add class-specific metadata
            class_name = chunk.metadata.get('class_name')
            if class_name:
                for element in analysis.elements:
                    if element.name == class_name:
                        chunk.metadata.update({
                            'docstring': element.docstring,
                            'inheritance': self._extract_inheritance(element.content)
                        })
                        break
        
        # Add code patterns
        chunk.metadata['patterns'] = self._extract_code_patterns(chunk.content)
    
    def _extract_chunk_dependencies(self, content: str) -> Dict[str, List[str]]:
        """Extract dependencies from chunk content."""
        dependencies = {
            'imports': [],
            'function_calls': [],
            'variables': []
        }
        
        try:
            imports, from_imports = self.analyzer.extract_imports(content)
            dependencies['imports'] = imports
            dependencies['from_imports'] = from_imports
            
            # Extract function calls (simple pattern matching)
            import re
            function_calls = re.findall(r'(\w+)\s*\(', content)
            dependencies['function_calls'] = list(set(function_calls))
            
        except Exception as e:
            self.logger.debug(f"Error extracting dependencies: {e}")
        
        return dependencies
    
    def _extract_inheritance(self, content: str) -> List[str]:
        """Extract class inheritance information."""
        import re
        
        # Simple regex to find class inheritance
        match = re.search(r'class\s+\w+\s*\(([^)]+)\)', content)
        if match:
            parents = [p.strip() for p in match.group(1).split(',')]
            return parents
        return []
    
    def _extract_code_patterns(self, content: str) -> Dict[str, bool]:
        """Extract common code patterns."""
        patterns = {
            'has_async': 'async def' in content or 'await ' in content,
            'has_decorators': '@' in content,
            'has_type_hints': '->' in content or ': ' in content,
            'has_docstring': '"""' in content or "'''" in content,
            'has_comments': '#' in content,
            'has_exceptions': 'try:' in content or 'except' in content,
            'has_loops': 'for ' in content or 'while ' in content,
            'has_conditionals': 'if ' in content or 'elif ' in content,
            'has_classes': 'class ' in content,
            'has_functions': 'def ' in content,
            'has_imports': 'import ' in content or 'from ' in content
        }
        
        return patterns
    
    def _estimate_language(self, content: str, file_path: str) -> str:
        """Estimate programming language from content and file path."""
        
        # Check file extension first
        if file_path:
            ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
            
            language_map = {
                'py': 'python',
                'js': 'javascript', 
                'ts': 'typescript',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
                'go': 'go',
                'rs': 'rust',
                'rb': 'ruby',
                'php': 'php',
                'swift': 'swift',
                'kt': 'kotlin',
                'scala': 'scala',
                'sh': 'shell',
                'sql': 'sql',
                'html': 'html',
                'css': 'css',
                'json': 'json',
                'xml': 'xml',
                'yaml': 'yaml',
                'yml': 'yaml',
                'md': 'markdown',
                'txt': 'text'
            }
            
            if ext in language_map:
                return language_map[ext]
        
        # Try to detect from content patterns
        content_lower = content.lower()
        
        if 'def ' in content and 'import ' in content:
            return 'python'
        elif 'function ' in content and ('var ' in content or 'let ' in content):
            return 'javascript'
        elif 'public class' in content and 'import java' in content:
            return 'java'
        elif '#include' in content and 'int main' in content:
            return 'c'
        elif 'func ' in content and 'package main' in content:
            return 'go'
        
        return 'unknown'
    
    async def _perform_health_check(self) -> None:
        """Perform health check for code processor."""
        
        # Test Python analyzer
        test_code = "def hello():\n    print('Hello, World!')\n"
        
        try:
            analysis = self.analyzer.analyze_code(test_code)
            if not analysis.elements:
                raise Exception("AST analyzer not working properly")
            
            # Test chunking strategy
            strategy = ChunkingStrategyFactory.create_strategy(self.python_config)
            chunks = strategy.chunk_content(test_code, {'source_item_id': 'test', 'source_type': 'test'})
            
            if not chunks:
                raise Exception("Chunking strategy not working properly")
            
            self.logger.debug("Code processor health check passed")
            
        except Exception as e:
            self.logger.error(f"Code processor health check failed: {e}")
            raise 