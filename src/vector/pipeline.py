"""
Vector storage pipeline orchestrator.

This module coordinates the entire vector storage workflow including
embedding generation, vector database operations, and search functionality.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.processing.base_processor import ProcessedChunk
from src.vector.embeddings import (
    BaseEmbeddingService, 
    EmbeddingServiceFactory, 
    EmbeddingRequest,
    EmbeddingResult
)
from src.vector.pinecone_store import PineconeVectorStore
from src.vector.similarity_search import SimilaritySearchEngine, SearchRequest, RetrievalResult
from src.utils.logger import get_logger
from src.utils.async_utils import gather_with_concurrency, async_timer
from src.config.settings import settings

logger = get_logger(__name__)


class VectorPipelineStatus(Enum):
    """Vector pipeline status."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class VectorPipelineConfig:
    """Configuration for vector storage pipeline."""
    # Embedding settings
    embedding_provider: str = "openai"
    embedding_model: Optional[str] = None
    embedding_batch_size: int = 100
    embedding_rate_limit: Optional[int] = 500
    
    # Vector store settings
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    vector_dimension: int = 1536
    vector_metric: str = "cosine"
    default_namespace: Optional[str] = None
    
    # Processing settings
    max_concurrent_embeddings: int = 10
    max_concurrent_upserts: int = 5
    upsert_batch_size: int = 100
    
    # Search settings
    default_top_k: int = 10
    max_search_results: int = 50
    
    # Health check settings
    health_check_interval: int = 300  # 5 minutes
    
    @classmethod
    def from_settings(cls) -> "VectorPipelineConfig":
        """Create configuration from global settings."""
        return cls(
            embedding_provider=getattr(settings, 'embedding_provider', 'openai'),
            embedding_model=getattr(settings, 'embedding_model', None),
            embedding_batch_size=getattr(settings, 'embedding_batch_size', 100),
            embedding_rate_limit=getattr(settings, 'embedding_rate_limit', 500),
            
            pinecone_api_key=getattr(settings, 'pinecone_api_key', None),
            pinecone_environment=getattr(settings, 'pinecone_environment', None),
            pinecone_index_name=getattr(settings, 'pinecone_index_name', None),
            vector_dimension=getattr(settings, 'vector_dimension', 1536),
            vector_metric=getattr(settings, 'vector_metric', 'cosine'),
            default_namespace=getattr(settings, 'default_namespace', None),
            
            max_concurrent_embeddings=getattr(settings, 'max_concurrent_embeddings', 10),
            max_concurrent_upserts=getattr(settings, 'max_concurrent_upserts', 5),
            upsert_batch_size=getattr(settings, 'upsert_batch_size', 100),
            
            default_top_k=getattr(settings, 'default_top_k', 10),
            max_search_results=getattr(settings, 'max_search_results', 50),
            
            health_check_interval=getattr(settings, 'health_check_interval', 300)
        )


@dataclass
class VectorPipelineMetrics:
    """Metrics for vector pipeline operations."""
    # Processing metrics
    total_chunks_processed: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    successful_upserts: int = 0
    failed_upserts: int = 0
    
    # Performance metrics
    embedding_time: float = 0.0
    upsert_time: float = 0.0
    total_processing_time: float = 0.0
    
    # Search metrics
    total_searches: int = 0
    average_search_time: float = 0.0
    
    # Timestamps
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    @property
    def embedding_success_rate(self) -> float:
        """Calculate embedding success rate."""
        total = self.successful_embeddings + self.failed_embeddings
        return (self.successful_embeddings / total * 100) if total > 0 else 0.0
    
    @property
    def upsert_success_rate(self) -> float:
        """Calculate upsert success rate."""
        total = self.successful_upserts + self.failed_upserts
        return (self.successful_upserts / total * 100) if total > 0 else 0.0
    
    @property
    def chunks_per_second(self) -> Optional[float]:
        """Calculate processing rate."""
        if self.total_processing_time > 0:
            return self.total_chunks_processed / self.total_processing_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_chunks_processed': self.total_chunks_processed,
            'successful_embeddings': self.successful_embeddings,
            'failed_embeddings': self.failed_embeddings,
            'successful_upserts': self.successful_upserts,
            'failed_upserts': self.failed_upserts,
            'embedding_success_rate': self.embedding_success_rate,
            'upsert_success_rate': self.upsert_success_rate,
            'embedding_time': self.embedding_time,
            'upsert_time': self.upsert_time,
            'total_processing_time': self.total_processing_time,
            'total_searches': self.total_searches,
            'average_search_time': self.average_search_time,
            'chunks_per_second': self.chunks_per_second,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class VectorStoragePipeline:
    """
    Vector storage pipeline orchestrator.
    
    Features:
    - Coordinates embedding generation and vector storage
    - Manages concurrent processing with rate limiting
    - Provides unified search interface
    - Monitors health and performance metrics
    - Handles error recovery and retry logic
    """
    
    def __init__(self, config: Optional[VectorPipelineConfig] = None):
        """
        Initialize vector storage pipeline.
        
        Args:
            config: Pipeline configuration (uses default from settings if not provided)
        """
        self.config = config or VectorPipelineConfig.from_settings()
        self.status = VectorPipelineStatus.INITIALIZING
        self.metrics = VectorPipelineMetrics()
        
        # Initialize components
        self.embedding_service: Optional[BaseEmbeddingService] = None
        self.vector_store: Optional[PineconeVectorStore] = None
        self.search_engine: Optional[SimilaritySearchEngine] = None
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None
        
        self.logger = get_logger(__name__, component="vector_pipeline")
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            self.logger.info("Initializing vector storage pipeline")
            
            # Initialize embedding service
            self.embedding_service = EmbeddingServiceFactory.create_service(
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
                rate_limit=self.config.embedding_rate_limit
            )
            
            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                api_key=self.config.pinecone_api_key,
                environment=self.config.pinecone_environment,
                index_name=self.config.pinecone_index_name,
                dimension=self.config.vector_dimension,
                metric=self.config.vector_metric,
                namespace=self.config.default_namespace
            )
            
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Initialize search engine
            self.search_engine = SimilaritySearchEngine(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service
            )
            
            # Run health checks
            await self._run_health_checks()
            
            # Start background health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            self.status = VectorPipelineStatus.READY
            self.logger.info("Vector storage pipeline initialized successfully")
            
        except Exception as e:
            self.status = VectorPipelineStatus.ERROR
            self.logger.error(f"Failed to initialize vector pipeline: {e}")
            raise
    
    async def process_chunks(self,
                           chunks: List[ProcessedChunk],
                           namespace: Optional[str] = None,
                           overwrite_existing: bool = False) -> Dict[str, Any]:
        """
        Process chunks through the vector storage pipeline.
        
        Args:
            chunks: List of processed chunks to store
            namespace: Namespace for vector storage
            overwrite_existing: Whether to overwrite existing vectors
            
        Returns:
            Processing results and metrics
        """
        if not chunks:
            return {'message': 'No chunks to process'}
        
        if self.status != VectorPipelineStatus.READY:
            raise RuntimeError(f"Pipeline not ready. Status: {self.status}")
        
        self.status = VectorPipelineStatus.PROCESSING
        self.metrics.start_time = datetime.utcnow()
        self.metrics.total_chunks_processed = len(chunks)
        
        self.logger.info(
            "Starting vector processing pipeline",
            chunk_count=len(chunks),
            namespace=namespace
        )
        
        try:
            async with async_timer("Complete vector processing pipeline"):
                
                # Step 1: Generate embeddings
                async with async_timer("Embedding generation"):
                    embeddings = await self._generate_embeddings_concurrent(chunks)
                    self.metrics.successful_embeddings = len([e for e in embeddings if 'error' not in e.metadata])
                    self.metrics.failed_embeddings = len([e for e in embeddings if 'error' in e.metadata])
                
                # Step 2: Store vectors
                async with async_timer("Vector storage"):
                    # Filter out failed embeddings
                    successful_pairs = []
                    for chunk, embedding in zip(chunks, embeddings):
                        if 'error' not in embedding.metadata:
                            successful_pairs.append((chunk, embedding))
                    
                    if successful_pairs:
                        successful_chunks, successful_embeddings = zip(*successful_pairs)
                        upsert_result = await self.vector_store.upsert_chunks(
                            list(successful_chunks),
                            list(successful_embeddings),
                            namespace
                        )
                        self.metrics.successful_upserts = upsert_result.get('upserted_count', 0)
                        self.metrics.failed_upserts = upsert_result.get('failed_count', 0)
                    else:
                        self.logger.warning("No successful embeddings to store")
                
                self.metrics.end_time = datetime.utcnow()
                self.metrics.total_processing_time = (
                    self.metrics.end_time - self.metrics.start_time
                ).total_seconds()
                
                result = {
                    'processed_chunks': len(chunks),
                    'successful_embeddings': self.metrics.successful_embeddings,
                    'failed_embeddings': self.metrics.failed_embeddings,
                    'successful_upserts': self.metrics.successful_upserts,
                    'failed_upserts': self.metrics.failed_upserts,
                    'embedding_success_rate': self.metrics.embedding_success_rate,
                    'upsert_success_rate': self.metrics.upsert_success_rate,
                    'processing_time': self.metrics.total_processing_time,
                    'chunks_per_second': self.metrics.chunks_per_second,
                    'namespace': namespace
                }
                
                self.logger.info(
                    "Vector processing pipeline completed",
                    **result
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Vector processing pipeline failed: {e}")
            raise
        
        finally:
            self.status = VectorPipelineStatus.READY
    
    async def _generate_embeddings_concurrent(self, 
                                            chunks: List[ProcessedChunk]) -> List[EmbeddingResult]:
        """Generate embeddings with controlled concurrency."""
        
        # Create embedding requests
        requests = []
        for chunk in chunks:
            request = EmbeddingRequest(
                text=chunk.content,
                id=chunk.id,
                metadata={
                    'chunk_type': chunk.chunk_type,
                    'source_type': chunk.source_type,
                    'source_item_id': chunk.source_item_id
                }
            )
            requests.append(request)
        
        # Generate embeddings with concurrency control
        async def generate_batch(batch_requests: List[EmbeddingRequest]) -> List[EmbeddingResult]:
            return await self.embedding_service.generate_embeddings(batch_requests)
        
        # Split into batches for concurrent processing
        batch_size = self.config.embedding_batch_size
        batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
        
        # Process batches concurrently
        tasks = [generate_batch(batch) for batch in batches]
        batch_results = await gather_with_concurrency(
            self.config.max_concurrent_embeddings,
            *tasks
        )
        
        # Flatten results
        embeddings = []
        for batch_result in batch_results:
            embeddings.extend(batch_result)
        
        return embeddings
    
    async def search(self, request: SearchRequest) -> List[RetrievalResult]:
        """
        Perform similarity search.
        
        Args:
            request: Search request
            
        Returns:
            List of search results
        """
        if self.status not in [VectorPipelineStatus.READY, VectorPipelineStatus.PROCESSING]:
            raise RuntimeError(f"Pipeline not ready for search. Status: {self.status}")
        
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized")
        
        search_start = datetime.utcnow()
        
        try:
            results = await self.search_engine.search(request)
            
            # Update search metrics
            search_time = (datetime.utcnow() - search_start).total_seconds()
            self.metrics.total_searches += 1
            
            # Update average search time
            total_time = self.metrics.average_search_time * (self.metrics.total_searches - 1) + search_time
            self.metrics.average_search_time = total_time / self.metrics.total_searches
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def search_code(self, 
                         query: str, 
                         top_k: int = 10, 
                         **kwargs) -> List[RetrievalResult]:
        """Convenience method for code search."""
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized")
        
        return await self.search_engine.search_code(query, top_k, **kwargs)
    
    async def search_qa(self, 
                       question: str, 
                       top_k: int = 10, 
                       **kwargs) -> List[RetrievalResult]:
        """Convenience method for Q&A search."""
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized")
        
        return await self.search_engine.search_qa(question, top_k, **kwargs)
    
    async def _run_health_checks(self) -> Dict[str, Any]:
        """Run health checks on all components."""
        health_status = {}
        
        # Check embedding service
        if self.embedding_service:
            health_status['embedding_service'] = await self.embedding_service.health_check()
        
        # Check vector store
        if self.vector_store:
            health_status['vector_store'] = await self.vector_store.health_check()
        
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
        while self.status != VectorPipelineStatus.SHUTDOWN:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self.status != VectorPipelineStatus.SHUTDOWN:
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
        metrics_dict = self.metrics.to_dict()
        
        # Add additional runtime metrics
        if self.vector_store:
            try:
                vector_stats = await self.vector_store._get_index_stats()
                metrics_dict['vector_store_stats'] = vector_stats
            except Exception as e:
                self.logger.warning(f"Could not get vector store stats: {e}")
        
        return {
            'pipeline_status': self.status.value,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'metrics': metrics_dict
        }
    
    async def shutdown(self) -> None:
        """Shutdown the pipeline and cleanup resources."""
        self.logger.info("Shutting down vector storage pipeline")
        
        self.status = VectorPipelineStatus.SHUTDOWN
        
        # Cancel health monitoring task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Vector storage pipeline shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown() 