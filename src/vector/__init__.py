"""
Vector storage and retrieval module.

This module handles vector database operations, embedding generation,
similarity search, and retrieval functionality for the RAG system.
"""

from .embeddings import (
    EmbeddingProvider,
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingBatch,
    BaseEmbeddingService,
    OpenAIEmbeddingService,
    SentenceTransformersEmbeddingService,
    EmbeddingServiceFactory
)

from .pinecone_store import (
    IndexType,
    VectorRecord,
    SearchResult,
    SearchQuery,
    PineconeVectorStore
)

from .similarity_search import (
    SearchType,
    RerankingStrategy,
    QueryContext,
    SearchRequest,
    RetrievalResult,
    QueryProcessor,
    ResultReranker,
    SimilaritySearchEngine
)

__all__ = [
    # Embeddings
    "EmbeddingProvider",
    "EmbeddingModel", 
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingBatch",
    "BaseEmbeddingService",
    "OpenAIEmbeddingService",
    "SentenceTransformersEmbeddingService",
    "EmbeddingServiceFactory",
    
    # Vector Store
    "IndexType",
    "VectorRecord",
    "SearchResult", 
    "SearchQuery",
    "PineconeVectorStore",
    
    # Similarity Search
    "SearchType",
    "RerankingStrategy",
    "QueryContext",
    "SearchRequest",
    "RetrievalResult",
    "QueryProcessor",
    "ResultReranker",
    "SimilaritySearchEngine",
    
    # Pipeline
    "VectorStoragePipeline",
    "VectorPipelineConfig",
    "VectorPipelineStatus"
]

# Import pipeline after other components to avoid circular imports
from .pipeline import (
    VectorStoragePipeline,
    VectorPipelineConfig,
    VectorPipelineStatus
) 