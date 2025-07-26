"""
Generation pipeline orchestrator.

This module coordinates the entire LLM generation workflow including
retrieval, context processing, prompt building, and response generation.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.generation.llm_providers import BaseLLMProvider, LLMProviderFactory
from src.generation.prompt_engineering import (
    PromptTemplateLibrary, PromptBuilder, PromptType, PromptContext
)
from src.generation.context_processor import ContextProcessor, ContextCompressionStrategy
from src.generation.response_generator import (
    ContextualResponseGenerator, ChainOfThoughtGenerator, GeneratedResponse
)
from src.vector.similarity_search import SimilaritySearchEngine, SearchRequest, SearchType
from src.utils.logger import get_logger
from src.utils.async_utils import async_timer
from src.config.settings import settings

logger = get_logger(__name__)


class GenerationPipelineStatus(Enum):
    """Generation pipeline status."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class GenerationPipelineConfig:
    """Configuration for generation pipeline."""
    # LLM settings
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    llm_rate_limit: Optional[int] = 60
    
    # Context processing
    max_context_tokens: int = 8000
    context_compression: ContextCompressionStrategy = ContextCompressionStrategy.HYBRID
    
    # Retrieval settings
    retrieval_top_k: int = 10
    min_relevance_score: float = 0.7
    
    # Response settings
    enable_chain_of_thought: bool = False
    enable_source_attribution: bool = True
    response_temperature: float = 0.7
    max_response_tokens: int = 2048
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_concurrent_generations: int = 5
    
    # Health check settings
    health_check_interval: int = 300  # 5 minutes
    
    @classmethod
    def from_settings(cls) -> "GenerationPipelineConfig":
        """Create configuration from global settings."""
        return cls(
            llm_provider=getattr(settings, 'llm_provider', 'openai'),
            llm_model=getattr(settings, 'llm_model', None),
            llm_rate_limit=getattr(settings, 'llm_rate_limit', 60),
            
            max_context_tokens=getattr(settings, 'max_context_tokens', 8000),
            context_compression=ContextCompressionStrategy(
                getattr(settings, 'context_compression', 'hybrid')
            ),
            
            retrieval_top_k=getattr(settings, 'retrieval_top_k', 10),
            min_relevance_score=getattr(settings, 'min_relevance_score', 0.7),
            
            enable_chain_of_thought=getattr(settings, 'enable_chain_of_thought', False),
            response_temperature=getattr(settings, 'response_temperature', 0.7),
            max_response_tokens=getattr(settings, 'max_response_tokens', 2048),
            
            enable_caching=getattr(settings, 'enable_caching', True),
            max_concurrent_generations=getattr(settings, 'max_concurrent_generations', 5)
        )


@dataclass
class GenerationMetrics:
    """Metrics for generation pipeline operations."""
    # Request metrics
    total_requests: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    
    # Performance metrics  
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0
    
    # Quality metrics
    avg_confidence_score: float = 0.0
    avg_sources_per_response: float = 0.0
    
    # Resource metrics
    total_tokens_used: int = 0
    avg_tokens_per_request: float = 0.0
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Timestamps
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_request_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_generations / self.total_requests) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return (self.cache_hits / total_cache_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_requests': self.total_requests,
            'successful_generations': self.successful_generations,
            'failed_generations': self.failed_generations,
            'success_rate': self.success_rate,
            'avg_retrieval_time': self.avg_retrieval_time,
            'avg_generation_time': self.avg_generation_time,
            'avg_total_time': self.avg_total_time,
            'avg_confidence_score': self.avg_confidence_score,
            'avg_sources_per_response': self.avg_sources_per_response,
            'total_tokens_used': self.total_tokens_used,
            'avg_tokens_per_request': self.avg_tokens_per_request,
            'cache_hit_rate': self.cache_hit_rate,
            'start_time': self.start_time.isoformat(),
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None
        }


class GenerationPipeline:
    """
    Generation pipeline orchestrator.
    
    Features:
    - Coordinates retrieval, context processing, and generation
    - Manages LLM providers and prompt templates
    - Provides caching and performance optimization
    - Monitors health and performance metrics
    - Supports both simple and chain-of-thought generation
    """
    
    def __init__(self, 
                 search_engine: SimilaritySearchEngine,
                 config: Optional[GenerationPipelineConfig] = None):
        """
        Initialize generation pipeline.
        
        Args:
            search_engine: Similarity search engine for retrieval
            config: Pipeline configuration (uses default from settings if not provided)
        """
        self.search_engine = search_engine
        self.config = config or GenerationPipelineConfig.from_settings()
        self.status = GenerationPipelineStatus.INITIALIZING
        self.metrics = GenerationMetrics()
        
        # Initialize components
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.template_library: Optional[PromptTemplateLibrary] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.context_processor: Optional[ContextProcessor] = None
        self.response_generator: Optional[ContextualResponseGenerator] = None
        self.cot_generator: Optional[ChainOfThoughtGenerator] = None
        
        # Caching
        self._response_cache: Dict[str, GeneratedResponse] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None
        
        # Concurrency control
        self._generation_semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)
        
        self.logger = get_logger(__name__, component="generation_pipeline")
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            self.logger.info("Initializing generation pipeline")
            
            # Initialize LLM provider
            self.llm_provider = LLMProviderFactory.create_provider(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                rate_limit=self.config.llm_rate_limit
            )
            
            # Initialize prompt components
            self.template_library = PromptTemplateLibrary()
            self.prompt_builder = PromptBuilder()
            
            # Initialize context processor
            self.context_processor = ContextProcessor(
                max_context_tokens=self.config.max_context_tokens,
                compression_strategy=self.config.context_compression
            )
            
            # Initialize response generators
            self.response_generator = ContextualResponseGenerator(
                llm_provider=self.llm_provider,
                prompt_builder=self.prompt_builder,
                context_processor=self.context_processor
            )
            
            if self.config.enable_chain_of_thought:
                self.cot_generator = ChainOfThoughtGenerator(self.response_generator)
            
            # Run health checks
            await self._run_health_checks()
            
            # Start background health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            self.status = GenerationPipelineStatus.READY
            self.logger.info("Generation pipeline initialized successfully")
            
        except Exception as e:
            self.status = GenerationPipelineStatus.ERROR
            self.logger.error(f"Failed to initialize generation pipeline: {e}")
            raise
    
    async def generate(self,
                      query: str,
                      context_metadata: Optional[Dict[str, Any]] = None,
                      use_chain_of_thought: Optional[bool] = None) -> GeneratedResponse:
        """
        Generate response for a query using the complete RAG pipeline.
        
        Args:
            query: User query
            context_metadata: Additional context metadata
            use_chain_of_thought: Override default CoT setting
            
        Returns:
            GeneratedResponse: Complete generated response
        """
        if self.status != GenerationPipelineStatus.READY:
            raise RuntimeError(f"Pipeline not ready. Status: {self.status}")
        
        async with self._generation_semaphore:
            return await self._generate_internal(query, context_metadata, use_chain_of_thought)
    
    async def _generate_internal(self,
                               query: str,
                               context_metadata: Optional[Dict[str, Any]] = None,
                               use_chain_of_thought: Optional[bool] = None) -> GeneratedResponse:
        """Internal generation method with metrics tracking."""
        
        self.status = GenerationPipelineStatus.PROCESSING
        self.metrics.total_requests += 1
        self.metrics.last_request_time = datetime.utcnow()
        request_start = datetime.utcnow()
        
        self.logger.info(
            "Starting generation request",
            query=query[:100],
            has_context=bool(context_metadata)
        )
        
        try:
            async with async_timer("Complete generation pipeline"):
                
                # Step 1: Check cache
                cache_key = self._generate_cache_key(query, context_metadata)
                if self.config.enable_caching and cache_key in self._response_cache:
                    self.metrics.cache_hits += 1
                    cached_response = self._response_cache[cache_key]
                    self.logger.info("Returning cached response")
                    return cached_response
                
                self.metrics.cache_misses += 1
                
                # Step 2: Retrieve relevant content
                async with async_timer("Retrieval") as retrieval_timer:
                    retrieved_chunks = await self._retrieve_content(query, context_metadata)
                
                # Step 3: Detect query intent and select template
                prompt_type = self.template_library.detect_prompt_type(query)
                template = self.template_library.get_template(prompt_type)
                
                if not template:
                    raise ValueError(f"No template found for prompt type: {prompt_type}")
                
                # Step 4: Generate response
                async with async_timer("Generation") as generation_timer:
                    if use_chain_of_thought or (use_chain_of_thought is None and self.config.enable_chain_of_thought):
                        response = await self.cot_generator.generate_with_reasoning(
                            query, retrieved_chunks, template, context_metadata
                        )
                    else:
                        response = await self.response_generator.generate_response(
                            query, retrieved_chunks, template, context_metadata
                        )
                
                # Step 5: Update metrics
                total_time = (datetime.utcnow() - request_start).total_seconds()
                self._update_metrics(response, retrieval_timer.elapsed, generation_timer.elapsed, total_time)
                
                # Step 6: Cache response
                if self.config.enable_caching:
                    self._response_cache[cache_key] = response
                    
                    # Clean old cache entries
                    if len(self._response_cache) > 1000:  # Limit cache size
                        oldest_key = min(self._response_cache.keys())
                        del self._response_cache[oldest_key]
                
                self.metrics.successful_generations += 1
                
                self.logger.info(
                    "Generation completed successfully",
                    response_length=len(response.content),
                    sources_count=len(response.sources),
                    confidence=response.confidence_score,
                    total_time=total_time
                )
                
                return response
                
        except Exception as e:
            self.metrics.failed_generations += 1
            self.logger.error(f"Generation failed: {e}")
            
            # Return error response
            error_response = GeneratedResponse(
                content=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                response_type=prompt_type if 'prompt_type' in locals() else None,
                confidence_score=0.0,
                metadata={'error': str(e), 'query': query}
            )
            
            return error_response
            
        finally:
            self.status = GenerationPipelineStatus.READY
    
    async def _retrieve_content(self,
                              query: str,
                              context_metadata: Optional[Dict[str, Any]] = None) -> List:
        """Retrieve relevant content using the search engine."""
        
        # Build search request
        search_request = SearchRequest(
            query=query,
            top_k=self.config.retrieval_top_k,
            search_type=SearchType.SEMANTIC  # Default, could be enhanced with intent detection
        )
        
        # Add context-based filters if available
        if context_metadata:
            search_request.context = context_metadata
        
        # Perform search
        results = await self.search_engine.search(search_request)
        
        # Filter by minimum relevance score
        filtered_results = [
            result for result in results 
            if result.final_score >= self.config.min_relevance_score
        ]
        
        self.logger.debug(
            "Content retrieval completed",
            total_results=len(results),
            filtered_results=len(filtered_results),
            min_score=self.config.min_relevance_score
        )
        
        return filtered_results
    
    def _generate_cache_key(self, 
                           query: str, 
                           context_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for request."""
        import hashlib
        
        key_components = [query]
        if context_metadata:
            # Sort keys for consistent hashing
            sorted_metadata = sorted(context_metadata.items())
            key_components.append(str(sorted_metadata))
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_metrics(self, 
                       response: GeneratedResponse,
                       retrieval_time: float,
                       generation_time: float,
                       total_time: float) -> None:
        """Update pipeline metrics."""
        
        # Update averages
        n = self.metrics.total_requests
        
        self.metrics.avg_retrieval_time = (
            (self.metrics.avg_retrieval_time * (n - 1)) + retrieval_time
        ) / n
        
        self.metrics.avg_generation_time = (
            (self.metrics.avg_generation_time * (n - 1)) + generation_time
        ) / n
        
        self.metrics.avg_total_time = (
            (self.metrics.avg_total_time * (n - 1)) + total_time
        ) / n
        
        self.metrics.avg_confidence_score = (
            (self.metrics.avg_confidence_score * (n - 1)) + response.confidence_score
        ) / n
        
        self.metrics.avg_sources_per_response = (
            (self.metrics.avg_sources_per_response * (n - 1)) + len(response.sources)
        ) / n
        
        # Update token usage
        tokens_used = response.generation_metrics.get('total_tokens', 0)
        self.metrics.total_tokens_used += tokens_used
        self.metrics.avg_tokens_per_request = self.metrics.total_tokens_used / n
    
    async def _run_health_checks(self) -> Dict[str, Any]:
        """Run health checks on all components."""
        health_status = {}
        
        # Check LLM provider
        if self.llm_provider:
            health_status['llm_provider'] = await self.llm_provider.health_check()
        
        # Check search engine
        if self.search_engine:
            health_status['search_engine'] = await self.search_engine.health_check()
        
        # Overall health
        all_healthy = all(
            status.get('status') == 'healthy'
            for status in health_status.values()
        )
        
        health_status['pipeline'] = {
            'status': 'healthy' if all_healthy else 'degraded',
            'pipeline_status': self.status.value,
            'last_check': datetime.utcnow().isoformat()
        }
        
        self._last_health_check = datetime.utcnow()
        return health_status
    
    async def _health_monitor(self) -> None:
        """Background task for health monitoring."""
        while self.status != GenerationPipelineStatus.SHUTDOWN:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self.status != GenerationPipelineStatus.SHUTDOWN:
                    health_status = await self._run_health_checks()
                    
                    # Log any unhealthy services
                    for service, status in health_status.items():
                        if status.get('status') != 'healthy':
                            self.logger.warning(
                                f"Health check failed for {service}",
                                status=status.get('status'),
                                error=status.get('error')
                            )
                    
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return await self._run_health_checks()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        return {
            'pipeline_status': self.status.value,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'metrics': self.metrics.to_dict(),
            'config': {
                'llm_provider': self.config.llm_provider,
                'llm_model': self.config.llm_model,
                'max_context_tokens': self.config.max_context_tokens,
                'enable_chain_of_thought': self.config.enable_chain_of_thought,
                'enable_caching': self.config.enable_caching
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the pipeline and cleanup resources."""
        self.logger.info("Shutting down generation pipeline")
        
        self.status = GenerationPipelineStatus.SHUTDOWN
        
        # Cancel health monitoring task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Clear cache
        self._response_cache.clear()
        
        self.logger.info("Generation pipeline shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown() 