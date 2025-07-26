"""
Embedding service for generating vector representations.

This module provides embedding generation using various providers like OpenAI,
with support for different embedding models and batch processing.
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from src.utils.logger import get_logger
from src.utils.async_utils import AsyncRetry, AsyncRateLimiter, async_timer
from src.config.settings import settings

logger = get_logger(__name__)


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    HUGGING_FACE = "hugging_face"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class EmbeddingModel(Enum):
    """Supported embedding models."""
    # OpenAI models
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    
    # Sentence Transformers models
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"


@dataclass
class EmbeddingRequest:
    """Request for generating embeddings."""
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    id: str
    text: str
    embedding: List[float]
    model: str
    provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'text': self.text,
            'embedding': self.embedding,
            'model': self.model,
            'provider': self.provider,
            'metadata': self.metadata,
            'dimension': self.dimension,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class EmbeddingBatch:
    """Batch of embedding requests."""
    requests: List[EmbeddingRequest]
    batch_id: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.requests)
    
    @property
    def total_tokens(self) -> int:
        """Estimate total tokens in batch."""
        return sum(len(req.text.split()) for req in self.requests)


class BaseEmbeddingService:
    """Base class for embedding services."""
    
    def __init__(self, 
                 model: Union[str, EmbeddingModel],
                 provider: Union[str, EmbeddingProvider],
                 batch_size: int = 100,
                 max_retries: int = 3,
                 rate_limit: Optional[int] = None):
        """
        Initialize embedding service.
        
        Args:
            model: Embedding model to use
            provider: Embedding provider
            batch_size: Maximum batch size for processing
            max_retries: Maximum retry attempts
            rate_limit: Rate limit (requests per minute)
        """
        self.model = model.value if isinstance(model, EmbeddingModel) else model
        self.provider = provider.value if isinstance(provider, EmbeddingProvider) else provider
        self.batch_size = batch_size
        
        # Set up retry mechanism
        self.retry_decorator = AsyncRetry(
            max_attempts=max_retries,
            base_delay=1.0,
            max_delay=60.0
        )
        
        # Set up rate limiter if specified
        self.rate_limiter = (
            AsyncRateLimiter(rate=rate_limit, per=60.0)
            if rate_limit else None
        )
        
        self.logger = get_logger(__name__, provider=self.provider, model=self.model)
    
    async def generate_embedding(self, text: str, **kwargs) -> EmbeddingResult:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResult: Generated embedding
        """
        request = EmbeddingRequest(text=text, metadata=kwargs)
        results = await self.generate_embeddings([request])
        return results[0]
    
    async def generate_embeddings(self, 
                                requests: List[EmbeddingRequest]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            requests: List of embedding requests
            
        Returns:
            List of embedding results
        """
        if not requests:
            return []
        
        self.logger.info(
            "Generating embeddings",
            count=len(requests),
            model=self.model,
            provider=self.provider
        )
        
        results = []
        
        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i:i + self.batch_size]
            batch = EmbeddingBatch(batch_requests)
            
            try:
                async with async_timer(f"Embedding batch {batch.batch_id}"):
                    batch_results = await self._generate_batch_embeddings(batch)
                    results.extend(batch_results)
                    
                    self.logger.debug(
                        "Batch processed successfully",
                        batch_id=batch.batch_id,
                        batch_size=batch.size,
                        results=len(batch_results)
                    )
                    
            except Exception as e:
                self.logger.error(
                    "Failed to process batch",
                    batch_id=batch.batch_id,
                    error=str(e)
                )
                # Create error results for failed batch
                for req in batch_requests:
                    error_result = EmbeddingResult(
                        id=req.id,
                        text=req.text,
                        embedding=[0.0] * self._get_embedding_dimension(),
                        model=self.model,
                        provider=self.provider,
                        metadata={**req.metadata, 'error': str(e)}
                    )
                    results.append(error_result)
        
        self.logger.info(
            "Embedding generation completed",
            total_requests=len(requests),
            successful_results=sum(1 for r in results if 'error' not in r.metadata),
            failed_results=sum(1 for r in results if 'error' in r.metadata)
        )
        
        return results
    
    async def _generate_batch_embeddings(self, 
                                       batch: EmbeddingBatch) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch (to be implemented by subclasses).
        
        Args:
            batch: Batch of embedding requests
            
        Returns:
            List of embedding results
        """
        raise NotImplementedError("Subclasses must implement _generate_batch_embeddings")
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model."""
        # Default dimensions for common models
        dimension_map = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "BAAI/bge-large-en-v1.5": 1024
        }
        return dimension_map.get(self.model, 1536)  # Default to 1536
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for embedding service."""
        try:
            # Test with a simple text
            test_text = "This is a test sentence for health check."
            result = await self.generate_embedding(test_text)
            
            return {
                'service': f"{self.provider}_embeddings",
                'model': self.model,
                'status': 'healthy',
                'test_dimension': result.dimension,
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'service': f"{self.provider}_embeddings",
                'model': self.model,
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }


class OpenAIEmbeddingService(BaseEmbeddingService):
    """OpenAI embedding service implementation."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: EmbeddingModel = EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
                 **kwargs):
        """
        Initialize OpenAI embedding service.
        
        Args:
            api_key: OpenAI API key (uses settings if not provided)
            model: OpenAI embedding model to use
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            model=model,
            provider=EmbeddingProvider.OPENAI,
            **kwargs
        )
        
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Import OpenAI client
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embedding service")
    
    async def _generate_batch_embeddings(self, 
                                       batch: EmbeddingBatch) -> List[EmbeddingResult]:
        """Generate embeddings using OpenAI API."""
        
        # Apply rate limiting
        if self.rate_limiter:
            async with self.rate_limiter:
                pass
        
        # Extract texts from requests
        texts = [req.text for req in batch.requests]
        
        # Call OpenAI API with retry
        @self.retry_decorator
        async def call_openai_api():
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            return response
        
        response = await call_openai_api()
        
        # Process results
        results = []
        for i, (req, embedding_data) in enumerate(zip(batch.requests, response.data)):
            result = EmbeddingResult(
                id=req.id,
                text=req.text,
                embedding=embedding_data.embedding,
                model=self.model,
                provider=self.provider,
                metadata=req.metadata
            )
            results.append(result)
        
        return results


class SentenceTransformersEmbeddingService(BaseEmbeddingService):
    """Sentence Transformers embedding service implementation."""
    
    def __init__(self, 
                 model: EmbeddingModel = EmbeddingModel.ALL_MINILM_L6_V2,
                 device: str = "cpu",
                 **kwargs):
        """
        Initialize Sentence Transformers embedding service.
        
        Args:
            model: Sentence Transformers model to use
            device: Device to run model on (cpu/cuda)
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            model=model,
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            **kwargs
        )
        
        self.device = device
        
        # Import and load model
        try:
            from sentence_transformers import SentenceTransformer
            self.model_instance = SentenceTransformer(self.model, device=device)
        except ImportError:
            raise ImportError("sentence-transformers package is required")
    
    async def _generate_batch_embeddings(self, 
                                       batch: EmbeddingBatch) -> List[EmbeddingResult]:
        """Generate embeddings using Sentence Transformers."""
        
        # Extract texts from requests
        texts = [req.text for req in batch.requests]
        
        # Generate embeddings (run in thread pool since it's CPU-bound)
        import asyncio
        loop = asyncio.get_event_loop()
        
        embeddings = await loop.run_in_executor(
            None, 
            self.model_instance.encode, 
            texts
        )
        
        # Convert to list format
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        # Process results
        results = []
        for req, embedding in zip(batch.requests, embeddings):
            result = EmbeddingResult(
                id=req.id,
                text=req.text,
                embedding=embedding,
                model=self.model,
                provider=self.provider,
                metadata=req.metadata
            )
            results.append(result)
        
        return results


class EmbeddingServiceFactory:
    """Factory for creating embedding services."""
    
    @staticmethod
    def create_service(provider: Union[str, EmbeddingProvider],
                      model: Optional[Union[str, EmbeddingModel]] = None,
                      **kwargs) -> BaseEmbeddingService:
        """
        Create an embedding service.
        
        Args:
            provider: Embedding provider
            model: Embedding model (uses default if not specified)
            **kwargs: Additional arguments for the service
            
        Returns:
            BaseEmbeddingService: Configured embedding service
        """
        if isinstance(provider, str):
            provider = EmbeddingProvider(provider)
        
        if provider == EmbeddingProvider.OPENAI:
            default_model = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
            if model:
                if isinstance(model, str):
                    model = EmbeddingModel(model)
            else:
                model = default_model
            
            return OpenAIEmbeddingService(model=model, **kwargs)
        
        elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            default_model = EmbeddingModel.ALL_MINILM_L6_V2
            if model:
                if isinstance(model, str):
                    model = EmbeddingModel(model)
            else:
                model = default_model
            
            return SentenceTransformersEmbeddingService(model=model, **kwargs)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def get_default_service() -> BaseEmbeddingService:
        """Get default embedding service based on configuration."""
        provider = getattr(settings, 'embedding_provider', 'openai')
        model = getattr(settings, 'embedding_model', None)
        
        return EmbeddingServiceFactory.create_service(
            provider=provider,
            model=model,
            batch_size=getattr(settings, 'embedding_batch_size', 100),
            rate_limit=getattr(settings, 'embedding_rate_limit', 500)
        ) 