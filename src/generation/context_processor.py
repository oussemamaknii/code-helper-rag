"""
Context processing for LLM input optimization.

This module handles context compression, relevance filtering, and context window
management to optimize retrieved content for LLM generation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.vector.similarity_search import RetrievalResult
from src.utils.logger import get_logger
from src.utils.text_utils import clean_text, count_tokens_approximate

logger = get_logger(__name__)


class ContextCompressionStrategy(Enum):
    """Strategies for compressing context information."""
    NONE = "none"
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    EXTRACT_KEY_POINTS = "extract_key_points"
    RELEVANCE_FILTER = "relevance_filter"
    HYBRID = "hybrid"


@dataclass
class RelevanceFilter:
    """Filter for determining relevance of context chunks."""
    min_score: float = 0.7
    max_chunks: int = 10
    prefer_code: bool = True
    prefer_recent: bool = True
    diversity_threshold: float = 0.8
    
    def should_include(self, chunk: RetrievalResult, query: str) -> bool:
        """Determine if chunk should be included based on relevance."""
        # Score threshold
        if chunk.final_score < self.min_score:
            return False
        
        # Content quality check
        if len(chunk.content.strip()) < 50:
            return False
        
        # Avoid duplicates (simple check)
        if chunk.content.count('\n') < 2 and len(chunk.content) < 100:
            return False
        
        return True


@dataclass 
class ContextWindow:
    """Represents a context window with token limits."""
    max_tokens: int
    reserved_tokens: int = 500  # Reserved for system prompt and response
    current_tokens: int = 0
    chunks: List[RetrievalResult] = field(default_factory=list)
    
    @property
    def available_tokens(self) -> int:
        """Get available tokens for context."""
        return max(0, self.max_tokens - self.reserved_tokens - self.current_tokens)
    
    @property
    def is_full(self) -> bool:
        """Check if context window is full."""
        return self.available_tokens <= 0
    
    def can_add_chunk(self, chunk: RetrievalResult) -> bool:
        """Check if chunk can be added without exceeding limits."""
        chunk_tokens = count_tokens_approximate(chunk.content)
        return chunk_tokens <= self.available_tokens
    
    def add_chunk(self, chunk: RetrievalResult) -> bool:
        """Add chunk to context window if it fits."""
        chunk_tokens = count_tokens_approximate(chunk.content)
        
        if chunk_tokens <= self.available_tokens:
            self.chunks.append(chunk)
            self.current_tokens += chunk_tokens
            return True
        
        return False


class ContextProcessor:
    """Processes and optimizes context for LLM consumption."""
    
    def __init__(self,
                 max_context_tokens: int = 8000,
                 compression_strategy: ContextCompressionStrategy = ContextCompressionStrategy.HYBRID,
                 relevance_filter: Optional[RelevanceFilter] = None):
        """
        Initialize context processor.
        
        Args:
            max_context_tokens: Maximum tokens for context
            compression_strategy: Strategy for context compression
            relevance_filter: Filter for relevance-based selection
        """
        self.max_context_tokens = max_context_tokens
        self.compression_strategy = compression_strategy
        self.relevance_filter = relevance_filter or RelevanceFilter()
        
        self.logger = get_logger(__name__, component="context_processor")
    
    def process_context(self,
                       chunks: List[RetrievalResult],
                       query: str,
                       **kwargs) -> Dict[str, Any]:
        """
        Process retrieved chunks into optimized context.
        
        Args:
            chunks: Retrieved chunks to process
            query: Original user query
            **kwargs: Additional processing parameters
            
        Returns:
            Dict containing processed context information
        """
        self.logger.debug(
            "Processing context",
            chunk_count=len(chunks),
            strategy=self.compression_strategy.value,
            max_tokens=self.max_context_tokens
        )
        
        # Step 1: Filter for relevance
        relevant_chunks = self._filter_relevant_chunks(chunks, query)
        
        # Step 2: Ensure diversity
        diverse_chunks = self._ensure_diversity(relevant_chunks, query)
        
        # Step 3: Apply compression strategy
        processed_chunks = self._apply_compression(diverse_chunks, query)
        
        # Step 4: Fit into context window
        final_chunks = self._fit_context_window(processed_chunks)
        
        # Step 5: Organize by type and priority
        organized_context = self._organize_context(final_chunks, query)
        
        self.logger.info(
            "Context processing completed",
            original_chunks=len(chunks),
            relevant_chunks=len(relevant_chunks),
            diverse_chunks=len(diverse_chunks),
            final_chunks=len(final_chunks),
            total_tokens=sum(count_tokens_approximate(chunk.content) for chunk in final_chunks)
        )
        
        return organized_context
    
    def _filter_relevant_chunks(self,
                               chunks: List[RetrievalResult],
                               query: str) -> List[RetrievalResult]:
        """Filter chunks based on relevance criteria."""
        relevant_chunks = []
        
        for chunk in chunks:
            if self.relevance_filter.should_include(chunk, query):
                relevant_chunks.append(chunk)
                
                # Apply max chunks limit
                if len(relevant_chunks) >= self.relevance_filter.max_chunks:
                    break
        
        # Sort by final score (highest first)
        relevant_chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        return relevant_chunks
    
    def _ensure_diversity(self,
                         chunks: List[RetrievalResult],
                         query: str) -> List[RetrievalResult]:
        """Ensure diversity in selected chunks to avoid redundancy."""
        if len(chunks) <= 3:
            return chunks
        
        diverse_chunks = [chunks[0]]  # Always include the top result
        
        for chunk in chunks[1:]:
            # Check similarity with already selected chunks
            is_diverse = True
            
            for selected_chunk in diverse_chunks:
                # Simple similarity check based on content overlap
                chunk_words = set(chunk.content.lower().split())
                selected_words = set(selected_chunk.content.lower().split())
                
                if chunk_words and selected_words:
                    overlap = len(chunk_words.intersection(selected_words))
                    similarity = overlap / len(chunk_words.union(selected_words))
                    
                    if similarity > self.relevance_filter.diversity_threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_chunks.append(chunk)
                
                # Limit to reasonable number for diversity
                if len(diverse_chunks) >= 8:
                    break
        
        return diverse_chunks
    
    def _apply_compression(self,
                          chunks: List[RetrievalResult],
                          query: str) -> List[RetrievalResult]:
        """Apply compression strategy to chunks."""
        if self.compression_strategy == ContextCompressionStrategy.NONE:
            return chunks
        
        compressed_chunks = []
        
        for chunk in chunks:
            if self.compression_strategy == ContextCompressionStrategy.TRUNCATE:
                compressed_chunk = self._truncate_chunk(chunk)
            elif self.compression_strategy == ContextCompressionStrategy.EXTRACT_KEY_POINTS:
                compressed_chunk = self._extract_key_points(chunk, query)
            elif self.compression_strategy == ContextCompressionStrategy.HYBRID:
                compressed_chunk = self._hybrid_compression(chunk, query)
            else:
                compressed_chunk = chunk
            
            compressed_chunks.append(compressed_chunk)
        
        return compressed_chunks
    
    def _truncate_chunk(self, chunk: RetrievalResult) -> RetrievalResult:
        """Truncate chunk content to fit better in context."""
        max_chars = 1500  # Approximately 300-400 tokens
        
        if len(chunk.content) <= max_chars:
            return chunk
        
        # Try to truncate at natural boundaries
        truncated_content = chunk.content[:max_chars]
        
        # Try to end at a complete line
        last_newline = truncated_content.rfind('\n')
        if last_newline > max_chars * 0.8:  # If we can keep 80% of content
            truncated_content = truncated_content[:last_newline]
        
        # Add truncation indicator
        truncated_content += "\n... [truncated]"
        
        # Create new result with truncated content
        truncated_result = RetrievalResult(
            id=chunk.id,
            content=truncated_content,
            score=chunk.score,
            chunk_type=chunk.chunk_type,
            source_type=chunk.source_type,
            metadata=chunk.metadata,
            rerank_score=chunk.rerank_score,
            explanation=f"{chunk.explanation} [truncated]" if chunk.explanation else "[truncated]"
        )
        
        return truncated_result
    
    def _extract_key_points(self, chunk: RetrievalResult, query: str) -> RetrievalResult:
        """Extract key points from chunk content."""
        lines = chunk.content.split('\n')
        key_lines = []
        
        query_words = set(query.lower().split())
        
        # Score each line by relevance to query
        scored_lines = []
        for line in lines:
            line_words = set(line.lower().split())
            relevance_score = len(query_words.intersection(line_words))
            
            # Boost lines with code patterns
            if any(pattern in line for pattern in ['def ', 'class ', 'import ', '=', '(', ')']):
                relevance_score += 2
            
            # Boost lines with documentation
            if any(pattern in line for pattern in ['"""', "'''", '#']):
                relevance_score += 1
            
            if line.strip():  # Non-empty lines
                scored_lines.append((relevance_score, line))
        
        # Sort by relevance and take top lines
        scored_lines.sort(key=lambda x: x[0], reverse=True)
        key_lines = [line for _, line in scored_lines[:20]]  # Top 20 lines
        
        # Reconstruct content
        key_content = '\n'.join(key_lines)
        if len(key_content) < len(chunk.content) * 0.5:
            key_content += f"\n... [extracted {len(key_lines)} key lines from {len(lines)} total]"
        
        # Create new result
        key_result = RetrievalResult(
            id=chunk.id,
            content=key_content,
            score=chunk.score,
            chunk_type=chunk.chunk_type,
            source_type=chunk.source_type,
            metadata=chunk.metadata,
            rerank_score=chunk.rerank_score,
            explanation=f"{chunk.explanation} [key points extracted]" if chunk.explanation else "[key points]"
        )
        
        return key_result
    
    def _hybrid_compression(self, chunk: RetrievalResult, query: str) -> RetrievalResult:
        """Apply hybrid compression strategy."""
        # For code chunks, prefer key point extraction
        if chunk.chunk_type in ['function', 'class', 'module']:
            return self._extract_key_points(chunk, query)
        
        # For text chunks, prefer truncation
        else:
            return self._truncate_chunk(chunk)
    
    def _fit_context_window(self, chunks: List[RetrievalResult]) -> List[RetrievalResult]:
        """Fit chunks into available context window."""
        context_window = ContextWindow(max_tokens=self.max_context_tokens)
        fitted_chunks = []
        
        # Sort chunks by priority (score and type)
        prioritized_chunks = sorted(
            chunks,
            key=lambda x: (
                x.final_score,
                1 if x.chunk_type in ['function', 'class'] else 0,  # Prefer code
                -len(x.content)  # Prefer shorter content when scores are equal
            ),
            reverse=True
        )
        
        for chunk in prioritized_chunks:
            if context_window.can_add_chunk(chunk):
                context_window.add_chunk(chunk)
                fitted_chunks.append(chunk)
            else:
                # Try to fit a truncated version
                truncated_chunk = self._truncate_chunk(chunk)
                if context_window.can_add_chunk(truncated_chunk):
                    context_window.add_chunk(truncated_chunk)
                    fitted_chunks.append(truncated_chunk)
        
        self.logger.debug(
            "Context window fitting completed",
            fitted_chunks=len(fitted_chunks),
            total_tokens=context_window.current_tokens,
            available_tokens=context_window.available_tokens
        )
        
        return fitted_chunks
    
    def _organize_context(self,
                         chunks: List[RetrievalResult],
                         query: str) -> Dict[str, Any]:
        """Organize chunks by type and create structured context."""
        
        # Categorize chunks
        code_chunks = []
        qa_chunks = []
        doc_chunks = []
        
        for chunk in chunks:
            if chunk.chunk_type in ['function', 'class', 'module']:
                code_chunks.append(chunk)
            elif chunk.chunk_type in ['qa_pair', 'question', 'answer']:
                qa_chunks.append(chunk)
            else:
                doc_chunks.append(chunk)
        
        # Create organized context
        organized_context = {
            'code_examples': [
                {
                    'content': chunk.content,
                    'type': chunk.chunk_type,
                    'score': chunk.final_score,
                    'source': chunk.source_type,
                    'explanation': chunk.explanation
                }
                for chunk in code_chunks
            ],
            'qa_examples': [
                {
                    'content': chunk.content,
                    'score': chunk.final_score,
                    'source': chunk.source_type,
                    'explanation': chunk.explanation
                }
                for chunk in qa_chunks
            ],
            'documentation': [
                {
                    'content': chunk.content,
                    'score': chunk.final_score,
                    'source': chunk.source_type,
                    'explanation': chunk.explanation
                }
                for chunk in doc_chunks
            ],
            'summary': {
                'total_chunks': len(chunks),
                'code_chunks': len(code_chunks),
                'qa_chunks': len(qa_chunks),
                'doc_chunks': len(doc_chunks),
                'avg_relevance_score': sum(chunk.final_score for chunk in chunks) / len(chunks) if chunks else 0,
                'total_tokens': sum(count_tokens_approximate(chunk.content) for chunk in chunks),
                'processing_strategy': self.compression_strategy.value
            }
        }
        
        return organized_context
    
    def compress_for_followup(self,
                             previous_context: Dict[str, Any],
                             new_query: str) -> Dict[str, Any]:
        """Compress previous context for follow-up questions."""
        # Keep only the most relevant parts of previous context
        compressed_context = {
            'code_examples': previous_context.get('code_examples', [])[:3],  # Top 3
            'qa_examples': previous_context.get('qa_examples', [])[:2],      # Top 2
            'documentation': previous_context.get('documentation', [])[:1],   # Top 1
            'summary': {
                **previous_context.get('summary', {}),
                'compressed_for_followup': True,
                'followup_query': new_query
            }
        }
        
        return compressed_context 