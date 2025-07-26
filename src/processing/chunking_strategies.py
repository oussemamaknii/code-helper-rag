"""
Chunking strategies for different types of content.

This module provides various intelligent chunking strategies to split content
into semantically meaningful chunks optimized for vector search and retrieval.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.processing.base_processor import ProcessedChunk
from src.processing.python_analyzer import PythonASTAnalyzer, CodeElement, CodeElementType
from src.utils.logger import get_logger
from src.utils.text_utils import clean_text, count_tokens_approximate
from src.config.settings import settings

logger = get_logger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    FUNCTION_LEVEL = "function_level"
    CLASS_LEVEL = "class_level"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    QA_PAIR = "qa_pair"


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    strategy: ChunkingStrategy
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    preserve_structure: bool = True
    include_context: bool = True
    max_chunks_per_item: int = 50


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize chunking strategy."""
        self.config = config
        self.logger = get_logger(__name__, strategy=config.strategy.value)
    
    @abstractmethod
    def chunk_content(self, content: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """
        Split content into chunks.
        
        Args:
            content: Content to chunk
            metadata: Metadata from source item
            
        Returns:
            List of processed chunks
        """
        pass
    
    def _create_chunk(self, 
                     content: str, 
                     chunk_type: str,
                     metadata: Dict[str, Any],
                     source_item_id: str,
                     source_type: str,
                     chunk_index: int = 0) -> ProcessedChunk:
        """Create a processed chunk with proper ID and metadata."""
        
        # Generate unique chunk ID
        chunk_id = hashlib.md5(
            f"{source_item_id}:{chunk_type}:{chunk_index}:{content[:100]}".encode()
        ).hexdigest()
        
        # Enhance metadata
        enhanced_metadata = {
            **metadata,
            'chunk_index': chunk_index,
            'chunk_size': len(content),
            'token_count': count_tokens_approximate(content),
            'strategy': self.config.strategy.value
        }
        
        return ProcessedChunk(
            id=chunk_id,
            content=content.strip(),
            chunk_type=chunk_type,
            metadata=enhanced_metadata,
            source_item_id=source_item_id,
            source_type=source_type
        )
    
    def _should_create_chunk(self, content: str) -> bool:
        """Check if content meets minimum requirements for chunking."""
        clean_content = content.strip()
        return (
            len(clean_content) >= self.config.min_chunk_size and
            bool(clean_content)
        )


class FixedSizeChunkingStrategy(BaseChunkingStrategy):
    """Fixed-size chunking with overlap."""
    
    def chunk_content(self, content: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """Split content into fixed-size chunks with overlap."""
        chunks = []
        source_item_id = metadata.get('source_item_id', 'unknown')
        source_type = metadata.get('source_type', 'unknown')
        
        # Clean content
        clean_content = clean_text(content)
        
        if not self._should_create_chunk(clean_content):
            return chunks
        
        # Split into chunks
        start = 0
        chunk_index = 0
        
        while start < len(clean_content):
            end = start + self.config.max_chunk_size
            
            # Try to find a good break point (sentence, paragraph, line)
            if end < len(clean_content):
                # Look for sentence boundaries
                sentence_end = clean_content.rfind('.', start, end)
                if sentence_end > start + self.config.min_chunk_size:
                    end = sentence_end + 1
                else:
                    # Look for line boundaries
                    line_end = clean_content.rfind('\n', start, end)
                    if line_end > start + self.config.min_chunk_size:
                        end = line_end + 1
                    else:
                        # Look for word boundaries
                        word_end = clean_content.rfind(' ', start, end)
                        if word_end > start + self.config.min_chunk_size:
                            end = word_end + 1
            
            chunk_content = clean_content[start:end].strip()
            
            if self._should_create_chunk(chunk_content):
                chunk = self._create_chunk(
                    content=chunk_content,
                    chunk_type="fixed_size",
                    metadata=metadata,
                    source_item_id=source_item_id,
                    source_type=source_type,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(end - self.config.chunk_overlap, start + 1)
            
            # Safety check to prevent infinite loop
            if len(chunks) >= self.config.max_chunks_per_item:
                break
        
        return chunks


class FunctionLevelChunkingStrategy(BaseChunkingStrategy):
    """Chunk Python code at function and method level."""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.analyzer = PythonASTAnalyzer()
    
    def chunk_content(self, content: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """Split Python code into function-level chunks."""
        chunks = []
        source_item_id = metadata.get('source_item_id', 'unknown')
        source_type = metadata.get('source_type', 'unknown')
        
        # Analyze code structure
        analysis = self.analyzer.analyze_code(content)
        
        if analysis.syntax_errors:
            # Fall back to fixed-size chunking for invalid Python
            self.logger.debug(
                "Syntax errors found, falling back to fixed-size chunking",
                errors=analysis.syntax_errors
            )
            fallback_strategy = FixedSizeChunkingStrategy(self.config)
            return fallback_strategy.chunk_content(content, metadata)
        
        code_lines = content.splitlines()
        chunk_index = 0
        
        # Create chunks for functions and methods
        for element in analysis.elements:
            if element.element_type in [CodeElementType.FUNCTION, CodeElementType.METHOD]:
                chunk_content = self._extract_function_chunk(element, code_lines, analysis)
                
                if self._should_create_chunk(chunk_content):
                    # Enhanced metadata for function chunks
                    func_metadata = {
                        **metadata,
                        'function_name': element.name,
                        'function_type': element.element_type.value,
                        'start_line': element.start_line,
                        'end_line': element.end_line,
                        'line_count': element.line_count,
                        'has_docstring': bool(element.docstring),
                        'complexity_score': element.complexity_score,
                        'parent_class': element.parent,
                        'args': element.metadata.get('args', []),
                        'decorators': element.metadata.get('decorators', []),
                        'is_async': element.metadata.get('is_async', False)
                    }
                    
                    chunk = self._create_chunk(
                        content=chunk_content,
                        chunk_type="function",
                        metadata=func_metadata,
                        source_item_id=source_item_id,
                        source_type=source_type,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        # Create chunks for classes (without their methods)
        for element in analysis.elements:
            if element.element_type == CodeElementType.CLASS:
                chunk_content = self._extract_class_chunk(element, code_lines, analysis)
                
                if self._should_create_chunk(chunk_content):
                    # Get methods in this class
                    class_methods = [
                        e.name for e in analysis.elements
                        if e.element_type == CodeElementType.METHOD and e.parent == element.name
                    ]
                    
                    class_metadata = {
                        **metadata,
                        'class_name': element.name,
                        'start_line': element.start_line,
                        'end_line': element.end_line,
                        'line_count': element.line_count,
                        'has_docstring': bool(element.docstring),
                        'complexity_score': element.complexity_score,
                        'method_count': len(class_methods),
                        'methods': class_methods,
                        'decorators': element.metadata.get('decorators', [])
                    }
                    
                    chunk = self._create_chunk(
                        content=chunk_content,
                        chunk_type="class",
                        metadata=class_metadata,
                        source_item_id=source_item_id,
                        source_type=source_type,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        # Create chunk for module-level code (imports, constants, etc.)
        module_content = self._extract_module_level_code(code_lines, analysis.elements)
        if self._should_create_chunk(module_content):
            module_metadata = {
                **metadata,
                'imports': analysis.dependencies.imports,
                'from_imports': analysis.dependencies.from_imports,
                'metrics': analysis.metrics
            }
            
            chunk = self._create_chunk(
                content=module_content,
                chunk_type="module",
                metadata=module_metadata,
                source_item_id=source_item_id,
                source_type=source_type,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_function_chunk(self, element: CodeElement, code_lines: List[str], analysis) -> str:
        """Extract function chunk with optional context."""
        lines = []
        
        # Add docstring context if needed
        if self.config.include_context and element.parent:
            # For methods, include class context
            class_element = next(
                (e for e in analysis.elements 
                 if e.element_type == CodeElementType.CLASS and e.name == element.parent),
                None
            )
            if class_element and class_element.docstring:
                lines.append(f"# Class: {element.parent}")
                lines.append(f'"""{class_element.docstring}"""')
                lines.append("")
        
        # Add the function content
        function_lines = code_lines[element.start_line-1:element.end_line]
        lines.extend(function_lines)
        
        return '\n'.join(lines)
    
    def _extract_class_chunk(self, element: CodeElement, code_lines: List[str], analysis) -> str:
        """Extract class definition without method implementations."""
        lines = []
        
        # Get class definition lines
        class_lines = code_lines[element.start_line-1:element.end_line]
        
        # Extract only class definition and docstring, skip method implementations
        in_method = False
        method_indent = 0
        
        for line in class_lines:
            # Check if we're entering a method
            if re.match(r'\s+def\s+', line) or re.match(r'\s+async\s+def\s+', line):
                in_method = True
                method_indent = len(line) - len(line.lstrip())
                # Add method signature only
                lines.append(line.rstrip() + "  # Implementation omitted")
                continue
            
            # Check if we're exiting a method
            if in_method:
                current_indent = len(line) - len(line.lstrip()) if line.strip() else method_indent + 1
                if current_indent <= method_indent and line.strip():
                    in_method = False
                else:
                    continue  # Skip method implementation
            
            # Add non-method lines
            if not in_method:
                lines.append(line.rstrip())
        
        return '\n'.join(lines)
    
    def _extract_module_level_code(self, code_lines: List[str], elements: List[CodeElement]) -> str:
        """Extract module-level code (imports, constants, etc.)."""
        lines = []
        covered_lines = set()
        
        # Mark lines covered by functions and classes
        for element in elements:
            for line_num in range(element.start_line, element.end_line + 1):
                covered_lines.add(line_num)
        
        # Extract uncovered lines (module-level code)
        for i, line in enumerate(code_lines, 1):
            if i not in covered_lines and line.strip():
                lines.append(line.rstrip())
        
        return '\n'.join(lines)


class QAPairChunkingStrategy(BaseChunkingStrategy):
    """Chunk Stack Overflow Q&A pairs."""
    
    def chunk_content(self, content: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """Split Q&A content into question and answer chunks."""
        chunks = []
        source_item_id = metadata.get('source_item_id', 'unknown')
        source_type = metadata.get('source_type', 'unknown')
        
        # Parse Q&A structure
        lines = content.splitlines()
        question_lines = []
        answer_lines = []
        current_section = None
        
        for line in lines:
            if line.startswith("Title:"):
                current_section = "title"
                question_lines.append(line)
            elif line.startswith("Question:"):
                current_section = "question"
                question_lines.append(line)
            elif line.startswith("Answer:"):
                current_section = "answer"
                answer_lines.append(line)
            elif current_section == "question":
                question_lines.append(line)
            elif current_section == "answer":
                answer_lines.append(line)
            else:
                question_lines.append(line)  # Default to question
        
        chunk_index = 0
        
        # Create question chunk
        question_content = '\n'.join(question_lines).strip()
        if self._should_create_chunk(question_content):
            question_metadata = {
                **metadata,
                'qa_part': 'question',
                'question_id': metadata.get('question_id'),
                'question_title': metadata.get('question_title'),
                'question_score': metadata.get('question_score', 0),
                'tags': metadata.get('tags', []),
                'view_count': metadata.get('view_count', 0)
            }
            
            chunk = self._create_chunk(
                content=question_content,
                chunk_type="question",
                metadata=question_metadata,
                source_item_id=source_item_id,
                source_type=source_type,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
            chunk_index += 1
        
        # Create answer chunk
        answer_content = '\n'.join(answer_lines).strip()
        if self._should_create_chunk(answer_content):
            answer_metadata = {
                **metadata,
                'qa_part': 'answer',
                'answer_id': metadata.get('answer_id'),
                'answer_score': metadata.get('answer_score', 0),
                'is_accepted': metadata.get('is_accepted', False),
                'has_code': metadata.get('has_code', False)
            }
            
            chunk = self._create_chunk(
                content=answer_content,
                chunk_type="answer",
                metadata=answer_metadata,
                source_item_id=source_item_id,
                source_type=source_type,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
            chunk_index += 1
        
        # If answer is very long, split it further
        if len(answer_content) > self.config.max_chunk_size * 2:
            fixed_strategy = FixedSizeChunkingStrategy(self.config)
            answer_chunks = fixed_strategy.chunk_content(answer_content, answer_metadata)
            
            for i, chunk in enumerate(answer_chunks):
                chunk.chunk_type = "answer_part"
                chunk.metadata['answer_part_index'] = i
                chunk.metadata['total_answer_parts'] = len(answer_chunks)
                chunks.append(chunk)
        
        return chunks


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""
    
    @staticmethod
    def create_strategy(config: ChunkingConfig) -> BaseChunkingStrategy:
        """Create a chunking strategy based on configuration."""
        
        if config.strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunkingStrategy(config)
        elif config.strategy == ChunkingStrategy.FUNCTION_LEVEL:
            return FunctionLevelChunkingStrategy(config)
        elif config.strategy == ChunkingStrategy.QA_PAIR:
            return QAPairChunkingStrategy(config)
        else:
            # Default to fixed size
            logger.warning(f"Unknown strategy {config.strategy}, using fixed_size")
            return FixedSizeChunkingStrategy(config)
    
    @staticmethod
    def get_optimal_strategy(source_type: str, content: str) -> ChunkingStrategy:
        """Get optimal chunking strategy based on content type."""
        
        if source_type == "github_code":
            # Check if it's valid Python code
            analyzer = PythonASTAnalyzer()
            if analyzer.is_valid_python(content):
                return ChunkingStrategy.FUNCTION_LEVEL
            else:
                return ChunkingStrategy.FIXED_SIZE
        
        elif source_type == "stackoverflow_qa":
            return ChunkingStrategy.QA_PAIR
        
        else:
            return ChunkingStrategy.FIXED_SIZE 