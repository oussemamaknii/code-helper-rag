"""
Unit tests for LLM integration and response generation components.

Tests cover LLM providers, prompt engineering, context processing,
response generation, and pipeline orchestration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from src.generation.llm_providers import (
    LLMProvider,
    LLMModel,
    GenerationRequest,
    GenerationResponse,
    GenerationConfig,
    BaseLLMProvider,
    LLMProviderFactory
)

from src.generation.prompt_engineering import (
    PromptType,
    FewShotExample,
    PromptContext,
    PromptTemplate,
    PromptBuilder,
    PromptTemplateLibrary
)

from src.generation.context_processor import (
    ContextCompressionStrategy,
    RelevanceFilter,
    ContextWindow,
    ContextProcessor
)

from src.generation.response_generator import (
    ResponseType,
    SourceAttribution,
    GeneratedResponse,
    ContextualResponseGenerator,
    ChainOfThoughtGenerator
)

from src.generation.pipeline import (
    GenerationPipelineConfig,
    GenerationMetrics,
    GenerationPipeline,
    GenerationPipelineStatus
)

from src.vector.similarity_search import RetrievalResult


class TestLLMProviders:
    """Test LLM provider components."""
    
    def test_generation_config_creation(self):
        """Test generation config creation and conversion."""
        config = GenerationConfig(
            temperature=0.8,
            max_tokens=1000,
            top_p=0.9,
            stop_sequences=["END", "STOP"]
        )
        
        assert config.temperature == 0.8
        assert config.max_tokens == 1000
        assert config.stop_sequences == ["END", "STOP"]
        
        config_dict = config.to_dict()
        assert config_dict['temperature'] == 0.8
        assert config_dict['stop'] == ["END", "STOP"]
    
    def test_generation_request_creation(self):
        """Test generation request creation with ID generation."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Test with explicit ID
        request = GenerationRequest(
            messages=messages,
            request_id="test-id"
        )
        assert request.request_id == "test-id"
        assert request.messages == messages
        
        # Test with auto-generated ID
        request_auto = GenerationRequest(messages=messages)
        assert request_auto.request_id is not None
        assert request_auto.request_id.startswith("gen_")
    
    def test_generation_response_properties(self):
        """Test generation response properties and methods."""
        response = GenerationResponse(
            request_id="test-id",
            text="Generated response",
            model="gpt-4",
            provider="openai",
            usage={
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            }
        )
        
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 50
        assert response.total_tokens == 150
        
        response_dict = response.to_dict()
        assert response_dict['text'] == "Generated response"
        assert response_dict['usage']['total_tokens'] == 150
    
    @pytest.mark.asyncio
    async def test_base_llm_provider(self):
        """Test base LLM provider abstract functionality."""
        
        class MockLLMProvider(BaseLLMProvider):
            async def _generate_response(self, request):
                return GenerationResponse(
                    request_id=request.request_id,
                    text="Mock response",
                    model=self.model,
                    provider=self.provider
                )
            
            async def _generate_stream_response(self, request):
                yield "Mock"
                yield " stream"
                yield " response"
        
        provider = MockLLMProvider(
            model="test-model",
            provider="test-provider"
        )
        
        # Test single generation
        request = GenerationRequest(
            messages=[{"role": "user", "content": "test"}]
        )
        response = await provider.generate(request)
        
        assert response.text == "Mock response"
        assert response.model == "test-model"
        assert response.provider == "test-provider"
        
        # Test streaming generation
        stream_chunks = []
        async for chunk in provider.generate_stream(request):
            stream_chunks.append(chunk)
        
        assert "".join(stream_chunks) == "Mock stream response"
        
        # Test health check
        health = await provider.health_check()
        assert health['service'] == "test-provider_llm"
        assert health['model'] == "test-model"
    
    @pytest.mark.asyncio
    async def test_openai_llm_provider_mock(self):
        """Test OpenAI LLM provider with mocked client."""
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "OpenAI response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "resp-123"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'openai':
                    mock_openai = Mock()
                    mock_openai.AsyncOpenAI.return_value = mock_client
                    return mock_openai
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            from src.generation.llm_providers import OpenAILLMProvider
            
            provider = OpenAILLMProvider(
                model=LLMModel.GPT_4_TURBO,
                api_key="test-key"
            )
            
            request = GenerationRequest(
                messages=[{"role": "user", "content": "test"}]
            )
            
            response = await provider.generate(request)
            
            assert response.text == "OpenAI response"
            assert response.total_tokens == 15
            assert response.metadata['finish_reason'] == "stop"
    
    def test_llm_provider_factory(self):
        """Test LLM provider factory."""
        
        # Mock the imports to avoid requiring actual packages
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'openai':
                    mock_openai = Mock()
                    mock_openai.AsyncOpenAI.return_value = Mock()
                    return mock_openai
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            with patch('src.config.settings') as mock_settings:
                mock_settings.openai_api_key = 'test-key'
                
                provider = LLMProviderFactory.create_provider(
                    provider=LLMProvider.OPENAI,
                    model=LLMModel.GPT_4_TURBO
                )
                
                assert provider.model == "gpt-4-turbo-preview"
                assert provider.provider == "openai"


class TestPromptEngineering:
    """Test prompt engineering components."""
    
    def test_few_shot_example_creation(self):
        """Test few-shot example creation and conversion."""
        example = FewShotExample(
            input="What is Python?",
            output="Python is a programming language",
            context="Programming basics",
            metadata={'difficulty': 'beginner'}
        )
        
        assert example.input == "What is Python?"
        assert example.output == "Python is a programming language"
        assert example.context == "Programming basics"
        
        example_dict = example.to_dict()
        assert example_dict['metadata']['difficulty'] == 'beginner'
    
    def test_prompt_context_creation(self):
        """Test prompt context creation."""
        retrieval_results = [
            RetrievalResult(
                id="result1",
                content="def hello(): print('Hello')",
                score=0.9,
                chunk_type="function",
                source_type="github_code"
            )
        ]
        
        context = PromptContext(
            query="How to create a function?",
            retrieved_chunks=retrieval_results,
            programming_language="python",
            difficulty_level="beginner"
        )
        
        assert context.query == "How to create a function?"
        assert len(context.retrieved_chunks) == 1
        assert context.programming_language == "python"
    
    def test_prompt_template_rendering(self):
        """Test prompt template rendering with Jinja2."""
        template = PromptTemplate(
            name="Test Template",
            prompt_type=PromptType.CODE_EXPLANATION,
            template="You are helping with {{ language }}. Query: {{ query }}",
            description="Test template"
        )
        
        rendered = template.render({
            'language': 'Python',
            'query': 'How to use loops?'
        })
        
        assert "Python" in rendered
        assert "How to use loops?" in rendered
    
    def test_prompt_builder(self):
        """Test prompt builder functionality."""
        builder = PromptBuilder()
        
        template = PromptTemplate(
            name="Test Template",
            prompt_type=PromptType.CODE_EXPLANATION,
            template="You are a {{ programming_language }} expert. Help with: {{ query }}",
            description="Test template",
            few_shot_examples=[
                FewShotExample(
                    input="What is a variable?",
                    output="A variable stores data"
                )
            ]
        )
        
        retrieval_results = [
            RetrievalResult(
                id="result1",
                content="x = 5",
                score=0.9,
                chunk_type="function",
                source_type="github_code"
            )
        ]
        
        context = PromptContext(
            query="How to create variables?",
            retrieved_chunks=retrieval_results,
            programming_language="Python"
        )
        
        messages = builder.build_prompt(template, context)
        
        # Should have system message, example exchange, and user query
        assert len(messages) >= 3
        assert messages[0]['role'] == 'system'
        assert messages[-1]['role'] == 'user'
        assert messages[-1]['content'] == "How to create variables?"
    
    def test_prompt_template_library(self):
        """Test prompt template library."""
        library = PromptTemplateLibrary()
        
        # Test getting templates
        code_template = library.get_template(PromptType.CODE_EXPLANATION)
        assert code_template is not None
        assert code_template.prompt_type == PromptType.CODE_EXPLANATION
        
        qa_template = library.get_template(PromptType.QA_RESPONSE)
        assert qa_template is not None
        
        # Test query type detection
        assert library.detect_prompt_type("write a function") == PromptType.CODE_GENERATION
        assert library.detect_prompt_type("explain this code") == PromptType.CODE_EXPLANATION
        assert library.detect_prompt_type("fix this error") == PromptType.DEBUGGING_HELP
        assert library.detect_prompt_type("what is python") == PromptType.QA_RESPONSE


class TestContextProcessor:
    """Test context processing components."""
    
    def test_relevance_filter(self):
        """Test relevance filter functionality."""
        filter = RelevanceFilter(
            min_score=0.7,
            max_chunks=5
        )
        
        # High score chunk should be included
        high_score_chunk = RetrievalResult(
            id="high",
            content="This is good content with enough text to pass quality checks",
            score=0.8,
            chunk_type="function",
            source_type="github_code"
        )
        assert filter.should_include(high_score_chunk, "test query")
        
        # Low score chunk should be excluded
        low_score_chunk = RetrievalResult(
            id="low",
            content="Good content",
            score=0.5,
            chunk_type="function",
            source_type="github_code"
        )
        assert not filter.should_include(low_score_chunk, "test query")
        
        # Short content should be excluded
        short_chunk = RetrievalResult(
            id="short",
            content="Short",
            score=0.8,
            chunk_type="function",
            source_type="github_code"
        )
        assert not filter.should_include(short_chunk, "test query")
    
    def test_context_window(self):
        """Test context window management."""
        window = ContextWindow(max_tokens=1000, reserved_tokens=200)
        
        assert window.available_tokens == 800
        assert not window.is_full
        
        # Add a chunk
        chunk = RetrievalResult(
            id="test",
            content="This is test content for the chunk",
            score=0.9,
            chunk_type="function",
            source_type="github_code"
        )
        
        # Should be able to add the chunk
        assert window.can_add_chunk(chunk)
        added = window.add_chunk(chunk)
        assert added
        assert len(window.chunks) == 1
        assert window.current_tokens > 0
    
    def test_context_processor(self):
        """Test context processor functionality."""
        processor = ContextProcessor(
            max_context_tokens=1000,
            compression_strategy=ContextCompressionStrategy.HYBRID
        )
        
        chunks = [
            RetrievalResult(
                id="chunk1",
                content="def function1(): pass\n" * 20,  # Longer content
                score=0.9,
                chunk_type="function",
                source_type="github_code"
            ),
            RetrievalResult(
                id="chunk2",
                content="Q: How to use loops?\nA: Use for loops for iteration",
                score=0.8,
                chunk_type="qa_pair",
                source_type="stackoverflow_qa"
            )
        ]
        
        processed_context = processor.process_context(chunks, "How to write functions?")
        
        assert 'code_examples' in processed_context
        assert 'qa_examples' in processed_context
        assert 'summary' in processed_context
        
        summary = processed_context['summary']
        assert summary['total_chunks'] <= len(chunks)
        assert summary['code_chunks'] >= 0
        assert summary['qa_chunks'] >= 0


class TestResponseGeneration:
    """Test response generation components."""
    
    def test_source_attribution(self):
        """Test source attribution creation."""
        attribution = SourceAttribution(
            chunk_id="chunk1",
            content_snippet="def hello(): print('Hello')",
            source_type="github_code",
            relevance_score=0.9,
            confidence=0.8,
            usage_type="example"
        )
        
        assert attribution.chunk_id == "chunk1"
        assert attribution.relevance_score == 0.9
        assert attribution.usage_type == "example"
        
        attribution_dict = attribution.to_dict()
        assert attribution_dict['confidence'] == 0.8
    
    def test_generated_response(self):
        """Test generated response creation and properties."""
        sources = [
            SourceAttribution(
                chunk_id="chunk1",
                content_snippet="code snippet",
                source_type="github_code",
                relevance_score=0.9
            ),
            SourceAttribution(
                chunk_id="chunk2", 
                content_snippet="qa snippet",
                source_type="stackoverflow_qa",
                relevance_score=0.7
            )
        ]
        
        response = GeneratedResponse(
            content="This is a generated response",
            response_type=ResponseType.CODE_EXPLANATION,
            sources=sources,
            confidence_score=0.85,
            reasoning_steps=["Step 1", "Step 2"]
        )
        
        assert response.has_sources
        assert len(response.primary_sources) == 2
        assert response.primary_sources[0].relevance_score == 0.9  # Should be sorted
        
        response_dict = response.to_dict()
        assert response_dict['response_type'] == 'code_explanation'
        assert len(response_dict['sources']) == 2
    
    @pytest.mark.asyncio
    async def test_contextual_response_generator(self):
        """Test contextual response generator."""
        
        # Mock LLM provider
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = GenerationResponse(
            request_id="test",
            text="This is a helpful explanation using the provided code examples.",
            model="test-model",
            provider="test-provider",
            usage={'prompt_tokens': 50, 'completion_tokens': 25, 'total_tokens': 75}
        )
        
        # Mock prompt builder
        mock_prompt_builder = Mock()
        mock_prompt_builder.build_prompt.return_value = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Explain this code"}
        ]
        
        # Mock context processor
        mock_context_processor = Mock()
        mock_context_processor.process_context.return_value = {
            'code_examples': [],
            'qa_examples': [],
            'summary': {'total_chunks': 1}
        }
        
        generator = ContextualResponseGenerator(
            llm_provider=mock_llm,
            prompt_builder=mock_prompt_builder,
            context_processor=mock_context_processor
        )
        
        template = PromptTemplate(
            name="Test Template",
            prompt_type=PromptType.CODE_EXPLANATION,
            template="Test template",
            description="Test"
        )
        
        chunks = [
            RetrievalResult(
                id="chunk1",
                content="def hello(): print('Hello')",
                score=0.9,
                chunk_type="function",
                source_type="github_code"
            )
        ]
        
        response = await generator.generate_response(
            query="Explain this function",
            retrieved_chunks=chunks,
            template=template
        )
        
        assert response.content == "This is a helpful explanation using the provided code examples."
        assert response.response_type in [ResponseType.CODE_EXPLANATION, ResponseType.DIRECT_ANSWER]
        assert response.confidence_score > 0
        assert response.generation_metrics['total_tokens'] == 75
    
    @pytest.mark.asyncio
    async def test_chain_of_thought_generator(self):
        """Test chain-of-thought generator."""
        
        # Mock base generator
        base_generator = AsyncMock()
        base_generator.generate_response.return_value = GeneratedResponse(
            content="REASONING:\n1. First I analyze the query\n2. Then I find relevant info\n\nANSWER:\nThis is the final answer",
            response_type=ResponseType.CODE_EXPLANATION,
            confidence_score=0.8
        )
        
        cot_generator = ChainOfThoughtGenerator(base_generator)
        
        template = PromptTemplate(
            name="Test Template",
            prompt_type=PromptType.CODE_EXPLANATION,
            template="Test template",
            description="Test"
        )
        
        response = await cot_generator.generate_with_reasoning(
            query="Test query",
            retrieved_chunks=[],
            template=template
        )
        
        assert response.content == "This is the final answer"
        assert len(response.reasoning_steps) > 0
        assert "analyze the query" in response.reasoning_steps[0].lower()


class TestGenerationPipeline:
    """Test generation pipeline orchestration."""
    
    def test_pipeline_config_creation(self):
        """Test pipeline configuration."""
        config = GenerationPipelineConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            max_context_tokens=4000,
            enable_chain_of_thought=True
        )
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.max_context_tokens == 4000
        assert config.enable_chain_of_thought is True
    
    def test_generation_metrics(self):
        """Test generation metrics calculation."""
        metrics = GenerationMetrics(
            total_requests=10,
            successful_generations=8,
            failed_generations=2,
            cache_hits=3,
            cache_misses=7
        )
        
        assert metrics.success_rate == 80.0
        assert metrics.cache_hit_rate == 30.0
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict['success_rate'] == 80.0
        assert 'total_requests' in metrics_dict
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        
        # Mock search engine
        mock_search_engine = AsyncMock()
        mock_search_engine.health_check.return_value = {'status': 'healthy'}
        
        config = GenerationPipelineConfig(
            llm_provider="openai",
            llm_model="gpt-4"
        )
        
        # Mock all the components
        with patch('src.generation.pipeline.LLMProviderFactory') as mock_factory, \
             patch('src.generation.pipeline.PromptTemplateLibrary') as mock_template_lib, \
             patch('src.generation.pipeline.PromptBuilder') as mock_prompt_builder, \
             patch('src.generation.pipeline.ContextProcessor') as mock_context_proc, \
             patch('src.generation.pipeline.ContextualResponseGenerator') as mock_resp_gen:
            
            mock_llm = AsyncMock()
            mock_llm.health_check.return_value = {'status': 'healthy'}
            mock_factory.create_provider.return_value = mock_llm
            
            mock_template_lib.return_value = Mock()
            mock_prompt_builder.return_value = Mock()
            mock_context_proc.return_value = Mock()
            mock_resp_gen.return_value = Mock()
            
            pipeline = GenerationPipeline(
                search_engine=mock_search_engine,
                config=config
            )
            
            await pipeline.initialize()
            
            assert pipeline.status == GenerationPipelineStatus.READY
            assert pipeline.llm_provider == mock_llm
            assert pipeline.template_library is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_generation_flow(self):
        """Test complete generation flow."""
        
        # Mock search engine
        mock_search_engine = AsyncMock()
        mock_search_results = [
            RetrievalResult(
                id="result1",
                content="def hello(): print('Hello')",
                score=0.9,
                chunk_type="function",
                source_type="github_code"
            )
        ]
        mock_search_engine.search.return_value = mock_search_results
        
        # Mock all components
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = GenerationResponse(
            request_id="test",
            text="This function prints Hello",
            model="gpt-4",
            provider="openai",
            usage={'total_tokens': 50}
        )
        
        mock_template_lib = Mock()
        mock_template = Mock()
        mock_template.name = "Test Template"
        mock_template.prompt_type = PromptType.CODE_EXPLANATION
        mock_template_lib.detect_prompt_type.return_value = PromptType.CODE_EXPLANATION
        mock_template_lib.get_template.return_value = mock_template
        
        mock_prompt_builder = Mock()
        mock_prompt_builder.build_prompt.return_value = [
            {"role": "user", "content": "test"}
        ]
        
        mock_context_processor = Mock()
        mock_context_processor.process_context.return_value = {
            'summary': {'total_chunks': 1}
        }
        
        mock_resp_generator = AsyncMock()
        mock_resp_generator.generate_response.return_value = GeneratedResponse(
            content="This function prints Hello",
            response_type=ResponseType.CODE_EXPLANATION,
            confidence_score=0.8,
            generation_metrics={'total_tokens': 50}
        )
        
        # Create pipeline
        pipeline = GenerationPipeline(search_engine=mock_search_engine)
        pipeline.status = GenerationPipelineStatus.READY
        pipeline.llm_provider = mock_llm
        pipeline.template_library = mock_template_lib
        pipeline.prompt_builder = mock_prompt_builder
        pipeline.context_processor = mock_context_processor
        pipeline.response_generator = mock_resp_generator
        
        # Generate response
        response = await pipeline.generate("Explain this function")
        
        assert response.content == "This function prints Hello"
        assert response.response_type == ResponseType.CODE_EXPLANATION
        assert pipeline.metrics.total_requests == 1
        assert pipeline.metrics.successful_generations == 1
        
        # Verify component calls
        mock_search_engine.search.assert_called_once()
        mock_resp_generator.generate_response.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 