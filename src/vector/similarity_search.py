"""
Similarity search and retrieval engine.

This module provides advanced similarity search capabilities including
query processing, embedding generation, vector search, and result reranking.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from src.vector.embeddings import BaseEmbeddingService, EmbeddingServiceFactory, EmbeddingRequest
from src.vector.pinecone_store import PineconeVectorStore, SearchQuery, SearchResult
from src.utils.logger import get_logger
from src.utils.async_utils import async_timer
from src.utils.text_utils import clean_text, extract_code_blocks
from src.config.settings import settings

logger = get_logger(__name__)


class SearchType(Enum):
    """Types of search queries."""
    SEMANTIC = "semantic"
    CODE = "code"
    QA = "qa"
    HYBRID = "hybrid"


class RerankingStrategy(Enum):
    """Reranking strategies for search results."""
    NONE = "none"
    SCORE_FUSION = "score_fusion"
    SEMANTIC_RERANK = "semantic_rerank"
    QUALITY_BOOST = "quality_boost"


@dataclass
class QueryContext:
    """Context information for search queries."""
    user_intent: Optional[str] = None
    programming_language: Optional[str] = None
    difficulty_level: Optional[str] = None
    domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SearchRequest:
    """Request for similarity search."""
    query: str
    top_k: int = 10
    search_type: SearchType = SearchType.SEMANTIC
    context: Optional[QueryContext] = None
    filters: Optional[Dict[str, Any]] = None
    reranking: RerankingStrategy = RerankingStrategy.SCORE_FUSION
    namespace: Optional[str] = None
    include_metadata: bool = True


@dataclass
class RetrievalResult:
    """Enhanced search result with additional information."""
    id: str
    content: str
    score: float
    chunk_type: str
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rerank_score: Optional[float] = None
    explanation: Optional[str] = None
    
    @property
    def final_score(self) -> float:
        """Get final score (reranked if available)."""
        return self.rerank_score if self.rerank_score is not None else self.score
    
    @property
    def function_name(self) -> Optional[str]:
        """Get function name if this is a function chunk."""
        return self.metadata.get('function_name')
    
    @property
    def class_name(self) -> Optional[str]:
        """Get class name if this is a class chunk."""
        return self.metadata.get('class_name')
    
    @property
    def repository_name(self) -> Optional[str]:
        """Get repository name for GitHub code."""
        return self.metadata.get('repository_name')
    
    @property
    def quality_score(self) -> Optional[float]:
        """Get quality score if available."""
        return self.metadata.get('quality_score')


class QueryProcessor:
    """Processes and enhances search queries."""
    
    def __init__(self):
        """Initialize query processor."""
        self.logger = get_logger(__name__, component="query_processor")
    
    async def process_query(self, request: SearchRequest) -> Tuple[str, Dict[str, Any]]:
        """
        Process and enhance search query.
        
        Args:
            request: Search request
            
        Returns:
            Tuple of (processed_query, enhanced_filters)
        """
        original_query = request.query
        processed_query = original_query
        enhanced_filters = request.filters.copy() if request.filters else {}
        
        # Detect search type if not specified
        if request.search_type == SearchType.SEMANTIC:
            detected_type = self._detect_search_type(original_query)
            if detected_type != SearchType.SEMANTIC:
                request.search_type = detected_type
        
        # Clean and normalize query
        processed_query = clean_text(processed_query)
        
        # Extract context from query
        context_info = self._extract_context_from_query(processed_query)
        if context_info:
            if not request.context:
                request.context = QueryContext()
            
            # Update context with extracted information
            for key, value in context_info.items():
                if value and not getattr(request.context, key, None):
                    setattr(request.context, key, value)
        
        # Apply query expansion
        processed_query = await self._expand_query(processed_query, request.search_type)
        
        # Build search filters based on context
        search_filters = self._build_search_filters(request.context, enhanced_filters)
        
        self.logger.debug(
            "Query processed",
            original_query=original_query,
            processed_query=processed_query,
            search_type=request.search_type.value,
            filters=search_filters
        )
        
        return processed_query, search_filters
    
    def _detect_search_type(self, query: str) -> SearchType:
        """Detect the type of search based on query content."""
        query_lower = query.lower()
        
        # Code-related indicators
        code_indicators = [
            'function', 'class', 'method', 'variable', 'import', 'def ', 'return',
            'if ', 'for ', 'while ', 'try:', 'except:', 'with ', 'lambda',
            '()', '[]', '{}', '->', '=>', '&&', '||'
        ]
        
        # Q&A indicators
        qa_indicators = [
            'how to', 'how do', 'how can', 'what is', 'what does', 'why',
            'when', 'where', 'which', 'who', '?'
        ]
        
        code_score = sum(1 for indicator in code_indicators if indicator in query_lower)
        qa_score = sum(1 for indicator in qa_indicators if indicator in query_lower)
        
        if code_score > qa_score and code_score > 1:
            return SearchType.CODE
        elif qa_score > 0:
            return SearchType.QA
        else:
            return SearchType.SEMANTIC
    
    def _extract_context_from_query(self, query: str) -> Dict[str, Any]:
        """Extract context information from query."""
        context = {}
        query_lower = query.lower()
        
        # Programming language detection
        languages = ['python', 'javascript', 'java', 'c++', 'go', 'rust', 'ruby', 'php']
        for lang in languages:
            if lang in query_lower:
                context['programming_language'] = lang
                break
        
        # Difficulty level detection
        if any(word in query_lower for word in ['beginner', 'basic', 'simple', 'intro']):
            context['difficulty_level'] = 'beginner'
        elif any(word in query_lower for word in ['advanced', 'complex', 'expert']):
            context['difficulty_level'] = 'advanced'
        elif any(word in query_lower for word in ['intermediate', 'medium']):
            context['difficulty_level'] = 'intermediate'
        
        # Domain detection
        domains = {
            'web': ['web', 'http', 'api', 'rest', 'frontend', 'backend'],
            'data_science': ['data', 'machine learning', 'ml', 'ai', 'pandas', 'numpy'],
            'database': ['database', 'sql', 'query', 'table', 'orm'],
            'async': ['async', 'await', 'concurrency', 'parallel', 'threading']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                context['domain'] = domain
                break
        
        return context
    
    async def _expand_query(self, query: str, search_type: SearchType) -> str:
        """Expand query with synonyms and related terms."""
        # Simple query expansion - can be enhanced with more sophisticated methods
        expansions = {
            SearchType.CODE: {
                'function': 'function method def',
                'class': 'class object type',
                'variable': 'variable var let const',
                'list': 'list array sequence',
                'dict': 'dict dictionary map object',
                'loop': 'loop for while iterate',
                'error': 'error exception exception handling'
            },
            SearchType.QA: {
                'how to': 'how to how do how can tutorial guide',
                'what is': 'what is what does definition explain',
                'why': 'why reason explanation cause',
                'best practice': 'best practice recommended approach pattern'
            }
        }
        
        if search_type in expansions:
            query_words = query.lower().split()
            expanded_terms = []
            
            for term, expansion in expansions[search_type].items():
                if term in query.lower():
                    expanded_terms.extend(expansion.split())
            
            if expanded_terms:
                # Add relevant terms to query
                unique_expanded = list(set(expanded_terms) - set(query_words))
                if unique_expanded:
                    query += " " + " ".join(unique_expanded[:3])  # Add top 3 expansions
        
        return query
    
    def _build_search_filters(self, 
                            context: Optional[QueryContext], 
                            base_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters based on context."""
        filters = base_filters.copy()
        
        if not context:
            return filters
        
        # Add programming language filter
        if context.programming_language:
            if context.programming_language == 'python':
                # Most of our data is Python, so this is always applicable
                pass
        
        # Add domain-specific filters
        if context.domain:
            if context.domain == 'web':
                # Look for web-related content
                filters.setdefault('$or', []).append({
                    'tags_list': {'$in': ['web', 'api', 'http', 'rest', 'flask', 'django']}
                })
            elif context.domain == 'data_science':
                filters.setdefault('$or', []).append({
                    'tags_list': {'$in': ['pandas', 'numpy', 'data', 'ml', 'machine-learning']}
                })
        
        # Add quality filters based on difficulty level
        if context.difficulty_level:
            if context.difficulty_level == 'beginner':
                # Prefer simpler content
                filters.setdefault('complexity_score', {})['$lt'] = 10
            elif context.difficulty_level == 'advanced':
                # Prefer more complex content
                filters.setdefault('complexity_score', {})['$gt'] = 5
        
        return filters


class ResultReranker:
    """Reranks search results using various strategies."""
    
    def __init__(self, embedding_service: Optional[BaseEmbeddingService] = None):
        """
        Initialize result reranker.
        
        Args:
            embedding_service: Embedding service for semantic reranking
        """
        self.embedding_service = embedding_service
        self.logger = get_logger(__name__, component="result_reranker")
    
    async def rerank_results(self,
                           results: List[SearchResult],
                           query: str,
                           strategy: RerankingStrategy,
                           context: Optional[QueryContext] = None) -> List[RetrievalResult]:
        """
        Rerank search results using the specified strategy.
        
        Args:
            results: Original search results
            query: Original search query
            strategy: Reranking strategy to use
            context: Query context for enhanced ranking
            
        Returns:
            List of reranked results
        """
        if not results:
            return []
        
        # Convert to RetrievalResult objects
        retrieval_results = []
        for result in results:
            retrieval_result = RetrievalResult(
                id=result.id,
                content=result.content or "",
                score=result.score,
                chunk_type=result.chunk_type or "unknown",
                source_type=result.source_type or "unknown",
                metadata=result.metadata
            )
            retrieval_results.append(retrieval_result)
        
        # Apply reranking strategy
        if strategy == RerankingStrategy.NONE:
            return retrieval_results
        
        elif strategy == RerankingStrategy.SCORE_FUSION:
            return await self._score_fusion_rerank(retrieval_results, query, context)
        
        elif strategy == RerankingStrategy.SEMANTIC_RERANK:
            return await self._semantic_rerank(retrieval_results, query)
        
        elif strategy == RerankingStrategy.QUALITY_BOOST:
            return await self._quality_boost_rerank(retrieval_results, context)
        
        else:
            self.logger.warning(f"Unknown reranking strategy: {strategy}")
            return retrieval_results
    
    async def _score_fusion_rerank(self,
                                 results: List[RetrievalResult],
                                 query: str,
                                 context: Optional[QueryContext]) -> List[RetrievalResult]:
        """Rerank using score fusion with multiple signals."""
        
        for result in results:
            # Start with original vector similarity score
            fusion_score = result.score
            
            # Quality boost
            quality_score = result.metadata.get('quality_score', 50.0) / 100.0
            fusion_score += quality_score * 0.1
            
            # Content length normalization (prefer substantial content)
            content_length = len(result.content)
            if 50 <= content_length <= 2000:
                fusion_score += 0.05
            elif content_length > 2000:
                fusion_score += 0.02
            
            # Source type preferences
            source_boosts = {
                'github_code': 0.1,
                'stackoverflow_qa': 0.05
            }
            if result.source_type in source_boosts:
                fusion_score += source_boosts[result.source_type]
            
            # Chunk type preferences for different contexts
            if context:
                if context.domain == 'web' and 'web' in result.content.lower():
                    fusion_score += 0.05
                elif context.domain == 'data_science' and any(
                    term in result.content.lower() 
                    for term in ['pandas', 'numpy', 'data', 'ml']
                ):
                    fusion_score += 0.05
            
            # Function/class name matching
            if result.chunk_type == 'function' and result.function_name:
                query_words = query.lower().split()
                if any(word in result.function_name.lower() for word in query_words):
                    fusion_score += 0.1
            
            result.rerank_score = fusion_score
            result.explanation = "Score fusion with quality and context signals"
        
        # Sort by fusion score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    async def _semantic_rerank(self,
                             results: List[RetrievalResult],
                             query: str) -> List[RetrievalResult]:
        """Rerank using semantic similarity with query."""
        
        if not self.embedding_service:
            self.logger.warning("No embedding service available for semantic reranking")
            return results
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Calculate similarity with each result
            for result in results:
                # This would require storing original embeddings or regenerating them
                # For now, we'll use a simpler approach based on text similarity
                
                # Simple keyword matching boost
                query_words = set(query.lower().split())
                content_words = set(result.content.lower().split())
                overlap = len(query_words.intersection(content_words))
                
                semantic_boost = min(overlap * 0.02, 0.2)  # Cap at 0.2
                result.rerank_score = result.score + semantic_boost
                result.explanation = f"Semantic rerank with {overlap} keyword matches"
            
        except Exception as e:
            self.logger.error(f"Semantic reranking failed: {e}")
            return results
        
        # Sort by semantic score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    async def _quality_boost_rerank(self,
                                  results: List[RetrievalResult],
                                  context: Optional[QueryContext]) -> List[RetrievalResult]:
        """Rerank with quality-based boosting."""
        
        for result in results:
            quality_boost = 0.0
            
            # Code quality boost
            if result.chunk_type in ['function', 'class']:
                # Boost documented code
                if result.metadata.get('has_docstring'):
                    quality_boost += 0.1
                
                # Boost lower complexity for beginners
                complexity = result.metadata.get('complexity_score', 5)
                if context and context.difficulty_level == 'beginner' and complexity < 5:
                    quality_boost += 0.05
                elif context and context.difficulty_level == 'advanced' and complexity > 10:
                    quality_boost += 0.05
            
            # Q&A quality boost
            elif result.chunk_type in ['question', 'answer']:
                # Boost accepted answers
                if result.metadata.get('is_accepted'):
                    quality_boost += 0.15
                
                # Boost high-scored content
                score = result.metadata.get('answer_score', 0)
                if score > 10:
                    quality_boost += 0.1
                elif score > 5:
                    quality_boost += 0.05
            
            result.rerank_score = result.score + quality_boost
            result.explanation = f"Quality boost: +{quality_boost:.3f}"
        
        # Sort by quality-boosted score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results


class SimilaritySearchEngine:
    """
    Advanced similarity search engine.
    
    Features:
    - Multi-modal search (semantic, code, Q&A)
    - Query processing and expansion
    - Vector similarity search
    - Advanced result reranking
    - Context-aware filtering
    """
    
    def __init__(self,
                 vector_store: PineconeVectorStore,
                 embedding_service: Optional[BaseEmbeddingService] = None):
        """
        Initialize similarity search engine.
        
        Args:
            vector_store: Vector database for similarity search
            embedding_service: Service for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service or EmbeddingServiceFactory.get_default_service()
        
        self.query_processor = QueryProcessor()
        self.result_reranker = ResultReranker(self.embedding_service)
        
        self.logger = get_logger(__name__, component="similarity_search")
    
    async def search(self, request: SearchRequest) -> List[RetrievalResult]:
        """
        Perform similarity search.
        
        Args:
            request: Search request with query and parameters
            
        Returns:
            List of ranked search results
        """
        self.logger.info(
            "Performing similarity search",
            query=request.query[:100],
            top_k=request.top_k,
            search_type=request.search_type.value
        )
        
        try:
            async with async_timer("Complete similarity search"):
                
                # Process query
                processed_query, filters = await self.query_processor.process_query(request)
                
                # Generate query embedding
                async with async_timer("Query embedding generation"):
                    query_embedding = await self.embedding_service.generate_embedding(processed_query)
                
                # Create search query
                search_query = SearchQuery(
                    vector=query_embedding.embedding,
                    top_k=min(request.top_k * 2, 50),  # Get more results for reranking
                    filter=filters if filters else None,
                    namespace=request.namespace,
                    include_metadata=request.include_metadata
                )
                
                # Perform vector search
                async with async_timer("Vector database search"):
                    search_results = await self.vector_store.search(search_query, request.namespace)
                
                # Rerank results
                async with async_timer("Result reranking"):
                    final_results = await self.result_reranker.rerank_results(
                        search_results,
                        request.query,
                        request.reranking,
                        request.context
                    )
                
                # Trim to requested top_k
                final_results = final_results[:request.top_k]
                
                self.logger.info(
                    "Search completed",
                    results_found=len(final_results),
                    top_score=final_results[0].final_score if final_results else None,
                    reranking_applied=request.reranking.value
                )
                
                return final_results
                
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def search_code(self,
                         query: str,
                         top_k: int = 10,
                         programming_language: str = "python",
                         **kwargs) -> List[RetrievalResult]:
        """
        Search for code examples.
        
        Args:
            query: Code search query
            top_k: Number of results to return
            programming_language: Programming language filter
            **kwargs: Additional search parameters
            
        Returns:
            List of code search results
        """
        context = QueryContext(
            programming_language=programming_language,
            **kwargs
        )
        
        request = SearchRequest(
            query=query,
            top_k=top_k,
            search_type=SearchType.CODE,
            context=context,
            reranking=RerankingStrategy.QUALITY_BOOST
        )
        
        return await self.search(request)
    
    async def search_qa(self,
                       question: str,
                       top_k: int = 10,
                       difficulty_level: Optional[str] = None,
                       **kwargs) -> List[RetrievalResult]:
        """
        Search for Q&A content.
        
        Args:
            question: Question to search for
            top_k: Number of results to return
            difficulty_level: Difficulty level preference
            **kwargs: Additional search parameters
            
        Returns:
            List of Q&A search results
        """
        context = QueryContext(
            difficulty_level=difficulty_level,
            **kwargs
        )
        
        request = SearchRequest(
            query=question,
            top_k=top_k,
            search_type=SearchType.QA,
            context=context,
            reranking=RerankingStrategy.SCORE_FUSION
        )
        
        return await self.search(request)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for search engine."""
        try:
            # Test embedding service
            embedding_health = await self.embedding_service.health_check()
            
            # Test vector store
            vector_health = await self.vector_store.health_check()
            
            # Test basic search
            test_request = SearchRequest(
                query="test search query",
                top_k=1
            )
            test_results = await self.search(test_request)
            
            return {
                'service': 'similarity_search_engine',
                'status': 'healthy',
                'embedding_service': embedding_health,
                'vector_store': vector_health,
                'test_search_results': len(test_results),
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'service': 'similarity_search_engine',
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            } 