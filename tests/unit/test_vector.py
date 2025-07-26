"""
Unit tests for vector storage and retrieval components.

Tests cover embedding services, vector database operations,
similarity search, and pipeline orchestration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from src.vector.embeddings import (
    EmbeddingProvider,
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingBatch,
    BaseEmbeddingService,
    EmbeddingServiceFactory
)

from src.vector.pinecone_store import (
    VectorRecord,
    SearchResult,
    SearchQuery
)

from src.vector.similarity_search import (
    SearchType,
    RerankingStrategy,
    QueryContext,
    SearchRequest,
    RetrievalResult,
    QueryProcessor,
    ResultReranker,
    SimilaritySearchEngine
)

from src.vector.pipeline import (
    VectorPipelineConfig,
    VectorPipelineMetrics,
    VectorPipelineStatus
)

from src.processing.base_processor import ProcessedChunk


class TestEmbeddingComponents:
    """Test embedding-related components."""
    
    def test_embedding_request_creation(self):
        """Test embedding request creation and ID generation."""
        # Test with explicit ID
        request = EmbeddingRequest(
            text="test text",
            id="test-id",
            metadata={'key': 'value'}
        )
        assert request.id == "test-id"
        assert request.text == "test text"
        assert request.metadata == {'key': 'value'}
        
        # Test with auto-generated ID
        request_auto = EmbeddingRequest(text="test text")
        assert request_auto.id is not None
        assert len(request_auto.id) == 32  # MD5 hash length
    
    def test_embedding_result_properties(self):
        """Test embedding result properties and methods."""
        embedding = [0.1, 0.2, 0.3]
        result = EmbeddingResult(
            id="test-id",
            text="test text",
            embedding=embedding,
            model="test-model",
            provider="test-provider",
            metadata={'key': 'value'}
        )
        
        assert result.dimension == 3
        assert result.id == "test-id"
        
        result_dict = result.to_dict()
        assert result_dict['id'] == "test-id"
        assert result_dict['embedding'] == embedding
        assert result_dict['dimension'] == 3
        assert 'created_at' in result_dict
    
    def test_embedding_batch_properties(self):
        """Test embedding batch properties."""
        requests = [
            EmbeddingRequest(text="text 1"),
            EmbeddingRequest(text="text 2 with more words"),
            EmbeddingRequest(text="text 3")
        ]
        
        batch = EmbeddingBatch(requests=requests)
        assert batch.size == 3
        assert batch.total_tokens == 9  # Corrected word count: 2 + 5 + 2 = 9
        assert batch.batch_id is not None
    
    @pytest.mark.asyncio
    async def test_base_embedding_service(self):
        """Test base embedding service abstract functionality."""
        
        class MockEmbeddingService(BaseEmbeddingService):
            async def _generate_batch_embeddings(self, batch):
                results = []
                for req in batch.requests:
                    result = EmbeddingResult(
                        id=req.id,
                        text=req.text,
                        embedding=[0.1, 0.2, 0.3],
                        model=self.model,
                        provider=self.provider,
                        metadata=req.metadata
                    )
                    results.append(result)
                return results
        
        service = MockEmbeddingService(
            model="test-model",
            provider="test-provider",
            batch_size=2
        )
        
        # Test single embedding
        result = await service.generate_embedding("test text")
        assert result.text == "test text"
        assert result.embedding == [0.1, 0.2, 0.3]
        
        # Test batch embeddings
        requests = [
            EmbeddingRequest(text="text 1"),
            EmbeddingRequest(text="text 2"),
            EmbeddingRequest(text="text 3")
        ]
        
        results = await service.generate_embeddings(requests)
        assert len(results) == 3
        assert all(r.embedding == [0.1, 0.2, 0.3] for r in results)
    
    @pytest.mark.asyncio
    async def test_openai_embedding_service(self):
        """Test OpenAI embedding service."""
        
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        # Mock the openai module and settings
        with patch('builtins.__import__') as mock_import:
            mock_openai = Mock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'openai':
                    return mock_openai
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            with patch('src.config.settings') as mock_settings:
                mock_settings.openai_api_key = "test-key"
                
                from src.vector.embeddings import OpenAIEmbeddingService
                
                service = OpenAIEmbeddingService(
                    api_key="test-key",
                    model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL
                )
                
                requests = [
                    EmbeddingRequest(text="text 1"),
                    EmbeddingRequest(text="text 2")
                ]
                
                results = await service.generate_embeddings(requests)
                
                assert len(results) == 2
                assert results[0].embedding == [0.1, 0.2, 0.3]
                assert results[1].embedding == [0.4, 0.5, 0.6]
                
                # Verify API call
                mock_client.embeddings.create.assert_called_once()
    
    def test_embedding_service_factory(self):
        """Test embedding service factory."""
        
        # Test with mock settings
        with patch('src.config.settings') as mock_settings:
            mock_settings.openai_api_key = 'test-key'
            
            # Mock the import for OpenAI
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        mock_openai = Mock()
                        mock_openai.AsyncOpenAI.return_value = Mock()
                        return mock_openai
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = import_side_effect
                
                service = EmbeddingServiceFactory.create_service(
                    provider=EmbeddingProvider.OPENAI,
                    model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL
                )
                assert service.model == "text-embedding-3-small"


class TestVectorStore:
    """Test vector store components."""
    
    def test_vector_record_creation(self):
        """Test vector record creation and conversion."""
        metadata = {'key': 'value', 'type': 'test'}
        record = VectorRecord(
            id="test-id",
            values=[0.1, 0.2, 0.3],
            metadata=metadata
        )
        
        pinecone_format = record.to_pinecone_format()
        assert pinecone_format['id'] == "test-id"
        assert pinecone_format['values'] == [0.1, 0.2, 0.3]
        assert pinecone_format['metadata'] == metadata
    
    def test_search_result_properties(self):
        """Test search result properties."""
        metadata = {
            'content': 'test content',
            'source_type': 'github_code',
            'chunk_type': 'function'
        }
        
        result = SearchResult(
            id="test-id",
            score=0.95,
            metadata=metadata
        )
        
        assert result.content == 'test content'
        assert result.source_type == 'github_code'
        assert result.chunk_type == 'function'
    
    def test_search_query_creation(self):
        """Test search query creation and conversion."""
        query = SearchQuery(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            namespace="test-namespace",
            filter={'type': 'function'},
            include_metadata=True
        )
        
        pinecone_format = query.to_pinecone_format()
        assert pinecone_format['vector'] == [0.1, 0.2, 0.3]
        assert pinecone_format['top_k'] == 5
        assert pinecone_format['namespace'] == "test-namespace"
        assert pinecone_format['filter'] == {'type': 'function'}
        assert pinecone_format['include_metadata'] is True
    
    @pytest.mark.asyncio
    async def test_pinecone_vector_store_initialization(self):
        """Test Pinecone vector store initialization."""
        
        # Mock Pinecone
        mock_pinecone = Mock()
        mock_pinecone.list_indexes.return_value = ['existing-index']
        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index
        
        # Mock the import
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'pinecone':
                    return mock_pinecone
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            from src.vector.pinecone_store import PineconeVectorStore
            
            store = PineconeVectorStore(
                api_key="test-key",
                environment="test-env",
                index_name="existing-index"
            )
            
            await store.initialize()
            
            assert store.index == mock_index
            mock_pinecone.init.assert_called_once_with(
                api_key="test-key",
                environment="test-env"
            )
    
    @pytest.mark.asyncio
    async def test_pinecone_vector_store_upsert(self):
        """Test vector upsert operation."""
        
        # Mock Pinecone components
        mock_pinecone = Mock()
        mock_pinecone.list_indexes.return_value = ['test-index']
        mock_index = Mock()
        mock_upsert_result = Mock()
        mock_upsert_result.upserted_count = 2
        mock_index.upsert.return_value = mock_upsert_result
        mock_pinecone.Index.return_value = mock_index
        
        # Mock the import
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'pinecone':
                    return mock_pinecone
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            from src.vector.pinecone_store import PineconeVectorStore
            
            store = PineconeVectorStore(
                api_key="test-key",
                environment="test-env",
                index_name="test-index"
            )
            
            await store.initialize()
            
            # Test upsert
            vectors = [
                VectorRecord(id="id1", values=[0.1, 0.2]),
                VectorRecord(id="id2", values=[0.3, 0.4])
            ]
            
            result = await store.upsert_vectors(vectors)
            
            assert result['total_vectors'] == 2
            assert result['upserted_count'] == 2
            assert result['success_rate'] == 100.0
    
    def test_metadata_cleaning(self):
        """Test metadata cleaning for Pinecone storage."""
        
        # Mock the import to avoid requiring pinecone-client
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'pinecone':
                    return Mock()
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            from src.vector.pinecone_store import PineconeVectorStore
            
            store = PineconeVectorStore(api_key="test-key")
            
            metadata = {
                'content': 'test content',
                'long_string': 'x' * 1500,  # Too long
                'nested_dict': {'key': 'value'},
                'nested_list': [1, 2, 3],
                'none_value': None,
                'valid_int': 42,
                'valid_bool': True
            }
            
            cleaned = store._clean_metadata(metadata)
            
            assert cleaned['content'] == 'test content'
            assert len(cleaned['long_string']) <= 1003  # Truncated + "..."
            assert cleaned['nested_dict_key'] == 'value'
            assert cleaned['nested_list_list'] == '[1, 2, 3]'
            assert 'none_value' not in cleaned
            assert cleaned['valid_int'] == 42
            assert cleaned['valid_bool'] is True


class TestSimilaritySearch:
    """Test similarity search components."""
    
    def test_query_context_creation(self):
        """Test query context creation."""
        context = QueryContext(
            programming_language="python",
            difficulty_level="intermediate",
            domain="data_science",
            tags=["pandas", "numpy"]
        )
        
        assert context.programming_language == "python"
        assert context.difficulty_level == "intermediate"
        assert context.domain == "data_science"
        assert "pandas" in context.tags
    
    def test_search_request_creation(self):
        """Test search request creation."""
        context = QueryContext(programming_language="python")
        
        request = SearchRequest(
            query="how to sort a list",
            top_k=5,
            search_type=SearchType.CODE,
            context=context,
            reranking=RerankingStrategy.QUALITY_BOOST
        )
        
        assert request.query == "how to sort a list"
        assert request.top_k == 5
        assert request.search_type == SearchType.CODE
        assert request.context.programming_language == "python"
        assert request.reranking == RerankingStrategy.QUALITY_BOOST
    
    def test_retrieval_result_properties(self):
        """Test retrieval result properties."""
        metadata = {
            'function_name': 'sort_list',
            'class_name': 'ListUtils',
            'repository_name': 'python-utils',
            'quality_score': 85.0
        }
        
        result = RetrievalResult(
            id="test-id",
            content="def sort_list(items): return sorted(items)",
            score=0.92,
            chunk_type="function",
            source_type="github_code",
            metadata=metadata,
            rerank_score=0.95
        )
        
        assert result.final_score == 0.95  # Uses rerank_score
        assert result.function_name == 'sort_list'
        assert result.class_name == 'ListUtils'
        assert result.repository_name == 'python-utils'
        assert result.quality_score == 85.0
    
    @pytest.mark.asyncio
    async def test_query_processor(self):
        """Test query processing and enhancement."""
        processor = QueryProcessor()
        
        # Test query processing
        request = SearchRequest(
            query="how to sort a python list",
            search_type=SearchType.SEMANTIC
        )
        
        processed_query, filters = await processor.process_query(request)
        
        assert "how to sort a python list" in processed_query
        assert isinstance(filters, dict)
    
    def test_query_type_detection(self):
        """Test automatic query type detection."""
        processor = QueryProcessor()
        
        # Code query
        code_type = processor._detect_search_type("def function() return list")
        assert code_type == SearchType.CODE
        
        # Q&A query
        qa_type = processor._detect_search_type("how to sort a list in Python?")
        assert qa_type == SearchType.QA
        
        # Semantic query
        semantic_type = processor._detect_search_type("machine learning algorithms")
        assert semantic_type == SearchType.SEMANTIC
    
    def test_context_extraction(self):
        """Test context extraction from queries."""
        processor = QueryProcessor()
        
        # Test language detection
        context = processor._extract_context_from_query("python list sorting")
        assert context.get('programming_language') == 'python'
        
        # Test difficulty detection
        context = processor._extract_context_from_query("advanced machine learning")
        assert context.get('difficulty_level') == 'advanced'
        
        # Test domain detection
        context = processor._extract_context_from_query("web api development")
        assert context.get('domain') == 'web'
    
    @pytest.mark.asyncio
    async def test_result_reranker_score_fusion(self):
        """Test score fusion reranking strategy."""
        
        # Mock search results
        search_results = [
            SearchResult(
                id="result1",
                score=0.8,
                metadata={
                    'content': 'def sort_list(items): return sorted(items)',
                    'quality_score': 90.0,
                    'source_type': 'github_code',
                    'chunk_type': 'function',
                    'function_name': 'sort_list'
                }
            ),
            SearchResult(
                id="result2", 
                score=0.85,
                metadata={
                    'content': 'items.sort()',
                    'quality_score': 60.0,
                    'source_type': 'stackoverflow_qa',
                    'chunk_type': 'answer'
                }
            )
        ]
        
        reranker = ResultReranker()
        results = await reranker.rerank_results(
            search_results,
            "sort list",
            RerankingStrategy.SCORE_FUSION
        )
        
        assert len(results) == 2
        assert all(r.rerank_score is not None for r in results)
        # First result should have higher final score due to quality boost
        assert results[0].final_score > results[1].final_score
    
    @pytest.mark.asyncio
    async def test_similarity_search_engine(self):
        """Test similarity search engine integration."""
        
        # Mock components
        mock_embedding_service = AsyncMock()
        mock_embedding_result = Mock()
        mock_embedding_result.embedding = [0.1, 0.2, 0.3]
        mock_embedding_service.generate_embedding.return_value = mock_embedding_result
        
        mock_vector_store = AsyncMock()
        mock_search_results = [
            SearchResult(
                id="result1",
                score=0.9,
                metadata={
                    'content': 'test content',
                    'chunk_type': 'function',
                    'source_type': 'github_code'
                }
            )
        ]
        mock_vector_store.search.return_value = mock_search_results
        
        # Create search engine
        engine = SimilaritySearchEngine(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        # Test search
        request = SearchRequest(
            query="test query",
            top_k=5
        )
        
        results = await engine.search(request)
        
        assert len(results) == 1
        assert results[0].content == 'test content'
        
        # Verify component calls
        mock_embedding_service.generate_embedding.assert_called_once()
        mock_vector_store.search.assert_called_once()


class TestVectorPipeline:
    """Test vector pipeline orchestration."""
    
    def test_pipeline_config_creation(self):
        """Test pipeline configuration."""
        config = VectorPipelineConfig(
            embedding_provider="openai",
            embedding_batch_size=50,
            max_concurrent_embeddings=5
        )
        
        assert config.embedding_provider == "openai"
        assert config.embedding_batch_size == 50
        assert config.max_concurrent_embeddings == 5
    
    def test_pipeline_metrics(self):
        """Test pipeline metrics calculation."""
        metrics = VectorPipelineMetrics(
            total_chunks_processed=100,
            successful_embeddings=95,
            failed_embeddings=5,
            successful_upserts=90,
            failed_upserts=10,
            total_processing_time=60.0
        )
        
        assert metrics.embedding_success_rate == 95.0
        assert metrics.upsert_success_rate == 90.0
        assert metrics.chunks_per_second == pytest.approx(100/60, rel=1e-2)
        
        metrics_dict = metrics.to_dict()
        assert 'total_chunks_processed' in metrics_dict
        assert 'embedding_success_rate' in metrics_dict
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        
        config = VectorPipelineConfig(
            pinecone_api_key="test-key",
            embedding_provider="openai"
        )
        
        # Mock components
        with patch('src.vector.pipeline.EmbeddingServiceFactory') as mock_factory, \
             patch('src.vector.pipeline.PineconeVectorStore') as mock_store_class, \
             patch('src.vector.pipeline.SimilaritySearchEngine') as mock_engine_class:
            
            mock_embedding_service = AsyncMock()
            mock_factory.create_service.return_value = mock_embedding_service
            
            mock_vector_store = AsyncMock()
            mock_store_class.return_value = mock_vector_store
            
            mock_search_engine = AsyncMock()
            mock_engine_class.return_value = mock_search_engine
            
            # Mock health checks
            mock_embedding_service.health_check.return_value = {'status': 'healthy'}
            mock_vector_store.health_check.return_value = {'status': 'healthy'}
            mock_search_engine.health_check.return_value = {'status': 'healthy'}
            
            from src.vector.pipeline import VectorStoragePipeline
            
            pipeline = VectorStoragePipeline(config)
            await pipeline.initialize()
            
            assert pipeline.status == VectorPipelineStatus.READY
            assert pipeline.embedding_service == mock_embedding_service
            assert pipeline.vector_store == mock_vector_store
            assert pipeline.search_engine == mock_search_engine
    
    @pytest.mark.asyncio
    async def test_pipeline_process_chunks(self):
        """Test chunk processing through pipeline."""
        
        # Create test chunks
        chunks = [
            ProcessedChunk(
                id="chunk1",
                content="test content 1",
                chunk_type="function",
                source_type="github_code",
                source_item_id="item1",
                metadata={}
            ),
            ProcessedChunk(
                id="chunk2",
                content="test content 2", 
                chunk_type="class",
                source_type="github_code",
                source_item_id="item2",
                metadata={}
            )
        ]
        
        # Mock services
        mock_embedding_service = AsyncMock()
        mock_embeddings = [
            EmbeddingResult(
                id="chunk1",
                text="test content 1",
                embedding=[0.1, 0.2, 0.3],
                model="test-model",
                provider="test-provider"
            ),
            EmbeddingResult(
                id="chunk2",
                text="test content 2",
                embedding=[0.4, 0.5, 0.6],
                model="test-model",
                provider="test-provider"
            )
        ]
        mock_embedding_service.generate_embeddings.return_value = mock_embeddings
        
        mock_vector_store = AsyncMock()
        mock_vector_store.upsert_chunks.return_value = {
            'upserted_count': 2,
            'failed_count': 0
        }
        
        # Create pipeline
        from src.vector.pipeline import VectorStoragePipeline
        
        pipeline = VectorStoragePipeline()
        pipeline.status = VectorPipelineStatus.READY
        pipeline.embedding_service = mock_embedding_service
        pipeline.vector_store = mock_vector_store
        
        # Process chunks
        result = await pipeline.process_chunks(chunks)
        
        assert result['processed_chunks'] == 2
        assert result['successful_embeddings'] == 2
        assert result['successful_upserts'] == 2
        assert result['embedding_success_rate'] == 100.0
        
        # Verify service calls
        mock_embedding_service.generate_embeddings.assert_called_once()
        mock_vector_store.upsert_chunks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_search_integration(self):
        """Test search functionality integration."""
        
        mock_search_engine = AsyncMock()
        mock_results = [
            RetrievalResult(
                id="result1",
                content="test result",
                score=0.9,
                chunk_type="function",
                source_type="github_code"
            )
        ]
        mock_search_engine.search.return_value = mock_results
        
        from src.vector.pipeline import VectorStoragePipeline
        
        pipeline = VectorStoragePipeline()
        pipeline.status = VectorPipelineStatus.READY
        pipeline.search_engine = mock_search_engine
        
        request = SearchRequest(query="test query")
        results = await pipeline.search(request)
        
        assert len(results) == 1
        assert results[0].content == "test result"
        assert pipeline.metrics.total_searches == 1
        
        mock_search_engine.search.assert_called_once_with(request)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 