"""
Pinecone vector database integration.

This module provides integration with Pinecone for storing, indexing,
and retrieving vector embeddings with metadata.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from src.vector.embeddings import EmbeddingResult
from src.processing.base_processor import ProcessedChunk
from src.utils.logger import get_logger
from src.utils.async_utils import AsyncRetry, async_timer
from src.config.settings import settings

logger = get_logger(__name__)


class IndexType(Enum):
    """Pinecone index types."""
    SERVERLESS = "serverless"
    POD = "pod"


@dataclass
class VectorRecord:
    """Vector record for storage in Pinecone."""
    id: str
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_pinecone_format(self) -> Dict[str, Any]:
        """Convert to Pinecone API format."""
        return {
            'id': self.id,
            'values': self.values,
            'metadata': self.metadata
        }


@dataclass
class SearchResult:
    """Search result from vector database."""
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def content(self) -> Optional[str]:
        """Get content from metadata."""
        return self.metadata.get('content')
    
    @property
    def source_type(self) -> Optional[str]:
        """Get source type from metadata."""
        return self.metadata.get('source_type')
    
    @property
    def chunk_type(self) -> Optional[str]:
        """Get chunk type from metadata."""
        return self.metadata.get('chunk_type')


@dataclass
class SearchQuery:
    """Search query for vector database."""
    vector: List[float]
    top_k: int = 10
    namespace: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_values: bool = False
    
    def to_pinecone_format(self) -> Dict[str, Any]:
        """Convert to Pinecone API format."""
        query = {
            'vector': self.vector,
            'top_k': self.top_k,
            'include_metadata': self.include_metadata,
            'include_values': self.include_values
        }
        
        if self.namespace:
            query['namespace'] = self.namespace
        
        if self.filter:
            query['filter'] = self.filter
        
        return query


class PineconeVectorStore:
    """
    Pinecone vector database integration.
    
    Features:
    - Async operations for all database interactions
    - Batch upsert for efficient data ingestion
    - Advanced filtering and metadata search
    - Namespace management for data organization
    - Health monitoring and error handling
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 environment: Optional[str] = None,
                 index_name: Optional[str] = None,
                 dimension: int = 1536,
                 metric: str = "cosine",
                 namespace: Optional[str] = None):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key (uses settings if not provided)
            environment: Pinecone environment (uses settings if not provided)
            index_name: Name of the Pinecone index (uses settings if not provided)
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            namespace: Default namespace for operations
        """
        self.api_key = api_key or settings.pinecone_api_key
        self.environment = environment or getattr(settings, 'pinecone_environment', 'us-east1-gcp')
        self.index_name = index_name or getattr(settings, 'pinecone_index_name', 'python-code-helper')
        self.dimension = dimension
        self.metric = metric
        self.namespace = namespace
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize Pinecone client
        try:
            import pinecone
            self.pinecone = pinecone
            
            # Initialize Pinecone
            self.pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Get or create index
            self.index = None
            
        except ImportError:
            raise ImportError("pinecone-client package is required for Pinecone integration")
        
        # Set up retry mechanism
        self.retry_decorator = AsyncRetry(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0
        )
        
        self.logger = get_logger(__name__, index=self.index_name, namespace=self.namespace)
    
    async def initialize(self) -> None:
        """Initialize the Pinecone index."""
        try:
            # Check if index exists
            if self.index_name not in self.pinecone.list_indexes():
                self.logger.info(
                    "Creating new Pinecone index",
                    index_name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric
                )
                
                # Create index
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=self.pinecone.ServerlessSpec(
                        cloud='aws',
                        region=self.environment.split('-')[0] if '-' in self.environment else 'us-east-1'
                    )
                )
                
                # Wait for index to be ready
                await self._wait_for_index_ready()
            
            # Connect to index
            self.index = self.pinecone.Index(self.index_name)
            
            self.logger.info(
                "Pinecone index initialized successfully",
                index_name=self.index_name,
                stats=await self._get_index_stats()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    
    async def _wait_for_index_ready(self, timeout: int = 300) -> None:
        """Wait for index to be ready."""
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                index_description = self.pinecone.describe_index(self.index_name)
                if index_description.status['ready']:
                    return
                
                self.logger.debug("Waiting for index to be ready...")
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.warning(f"Error checking index status: {e}")
                await asyncio.sleep(5)
        
        raise TimeoutError(f"Index {self.index_name} not ready after {timeout} seconds")
    
    async def upsert_vectors(self,
                           vectors: List[VectorRecord],
                           namespace: Optional[str] = None,
                           batch_size: int = 100) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone index.
        
        Args:
            vectors: List of vector records to upsert
            namespace: Namespace for the vectors (uses default if not provided)
            batch_size: Batch size for upsert operations
            
        Returns:
            Upsert operation results
        """
        if not self.index:
            await self.initialize()
        
        namespace = namespace or self.namespace
        
        self.logger.info(
            "Upserting vectors to Pinecone",
            count=len(vectors),
            namespace=namespace,
            batch_size=batch_size
        )
        
        upserted_count = 0
        failed_count = 0
        
        # Process in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                async with async_timer(f"Pinecone upsert batch {i//batch_size + 1}"):
                    # Convert to Pinecone format
                    pinecone_vectors = [vec.to_pinecone_format() for vec in batch]
                    
                    # Upsert with retry
                    @self.retry_decorator
                    async def upsert_batch():
                        # Run in thread pool since Pinecone client is not async
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            None,
                            lambda: self.index.upsert(
                                vectors=pinecone_vectors,
                                namespace=namespace
                            )
                        )
                    
                    result = await upsert_batch()
                    upserted_count += result.upserted_count
                    
                    self.logger.debug(
                        "Batch upserted successfully",
                        batch_number=i//batch_size + 1,
                        batch_size=len(batch),
                        upserted=result.upserted_count
                    )
                    
            except Exception as e:
                failed_count += len(batch)
                self.logger.error(
                    "Failed to upsert batch",
                    batch_number=i//batch_size + 1,
                    error=str(e)
                )
        
        result = {
            'total_vectors': len(vectors),
            'upserted_count': upserted_count,
            'failed_count': failed_count,
            'success_rate': upserted_count / len(vectors) * 100 if vectors else 0
        }
        
        self.logger.info(
            "Vector upsert completed",
            **result
        )
        
        return result
    
    async def search(self,
                    query: SearchQuery,
                    namespace: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query: Search query
            namespace: Namespace to search in (uses default if not provided)
            
        Returns:
            List of search results
        """
        if not self.index:
            await self.initialize()
        
        namespace = namespace or self.namespace
        
        self.logger.debug(
            "Searching vectors in Pinecone",
            top_k=query.top_k,
            namespace=namespace,
            has_filter=bool(query.filter)
        )
        
        try:
            # Search with retry
            @self.retry_decorator
            async def search_vectors():
                # Run in thread pool since Pinecone client is not async
                loop = asyncio.get_event_loop()
                
                query_dict = query.to_pinecone_format()
                if namespace:
                    query_dict['namespace'] = namespace
                
                return await loop.run_in_executor(
                    None,
                    lambda: self.index.query(**query_dict)
                )
            
            response = await search_vectors()
            
            # Convert results
            results = []
            for match in response.matches:
                result = SearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata or {}
                )
                results.append(result)
            
            self.logger.debug(
                "Search completed",
                results_found=len(results),
                top_score=results[0].score if results else None
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def delete_vectors(self,
                           ids: List[str],
                           namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Namespace (uses default if not provided)
            
        Returns:
            Delete operation results
        """
        if not self.index:
            await self.initialize()
        
        namespace = namespace or self.namespace
        
        self.logger.info(
            "Deleting vectors from Pinecone",
            count=len(ids),
            namespace=namespace
        )
        
        try:
            # Delete with retry
            @self.retry_decorator
            async def delete_vectors():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.index.delete(
                        ids=ids,
                        namespace=namespace
                    )
                )
            
            await delete_vectors()
            
            result = {
                'deleted_count': len(ids),
                'success': True
            }
            
            self.logger.info("Vectors deleted successfully", **result)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {e}")
            return {
                'deleted_count': 0,
                'success': False,
                'error': str(e)
            }
    
    async def upsert_chunks(self,
                          chunks: List[ProcessedChunk],
                          embeddings: List[EmbeddingResult],
                          namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Upsert processed chunks with their embeddings.
        
        Args:
            chunks: List of processed chunks
            embeddings: List of corresponding embeddings
            namespace: Namespace for storage
            
        Returns:
            Upsert operation results
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings lists must have the same length")
        
        # Create vector records
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            # Prepare metadata
            metadata = {
                'content': chunk.content,
                'chunk_type': chunk.chunk_type,
                'source_type': chunk.source_type,
                'source_item_id': chunk.source_item_id,
                'processed_at': chunk.processed_at.isoformat(),
                **chunk.metadata
            }
            
            # Clean metadata for Pinecone (remove nested objects and large values)
            clean_metadata = self._clean_metadata(metadata)
            
            vector = VectorRecord(
                id=chunk.id,
                values=embedding.embedding,
                metadata=clean_metadata
            )
            vectors.append(vector)
        
        return await self.upsert_vectors(vectors, namespace)
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for Pinecone storage."""
        clean = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
            
            # Handle different types
            if isinstance(value, (str, int, float, bool)):
                # Truncate long strings
                if isinstance(value, str) and len(value) > 1000:
                    clean[key] = value[:1000] + "..."
                else:
                    clean[key] = value
            elif isinstance(value, list):
                # Convert list to string if it's small enough
                if len(str(value)) <= 200:
                    clean[f"{key}_list"] = str(value)
            elif isinstance(value, dict):
                # Flatten simple dictionaries
                if len(str(value)) <= 200:
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (str, int, float, bool)):
                            clean[f"{key}_{sub_key}"] = sub_value
        
        return clean
    
    async def _get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            if not self.index:
                return {}
            
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: self.index.describe_index_stats()
            )
            
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get index stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Pinecone connection."""
        try:
            if not self.index:
                await self.initialize()
            
            # Get index stats as health check
            stats = await self._get_index_stats()
            
            return {
                'service': 'pinecone_vector_store',
                'index_name': self.index_name,
                'status': 'healthy',
                'stats': stats,
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'service': 'pinecone_vector_store',
                'index_name': self.index_name,
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def create_namespace_if_needed(self, namespace: str) -> None:
        """Create namespace if it doesn't exist (Pinecone creates automatically)."""
        # Pinecone creates namespaces automatically when data is inserted
        # This method is here for consistency with other vector databases
        pass
    
    async def list_namespaces(self) -> List[str]:
        """List all namespaces in the index."""
        try:
            stats = await self._get_index_stats()
            return list(stats.get('namespaces', {}).keys())
        except Exception as e:
            self.logger.warning(f"Could not list namespaces: {e}")
            return [] 