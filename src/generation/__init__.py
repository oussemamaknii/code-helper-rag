"""
LLM integration and response generation module.

This module handles LLM integration, prompt engineering, response generation,
and context grounding for the RAG system.
"""

from .llm_providers import (
    LLMProvider,
    LLMModel,
    GenerationRequest,
    GenerationResponse,
    GenerationConfig,
    BaseLLMProvider,
    OpenAILLMProvider,
    AnthropicLLMProvider,
    LLMProviderFactory
)

from .prompt_engineering import (
    PromptTemplate,
    PromptType,
    PromptContext,
    FewShotExample,
    PromptBuilder,
    PromptTemplateLibrary
)

from .response_generator import (
    ResponseType,
    SourceAttribution,
    GeneratedResponse,
    ContextualResponseGenerator,
    ChainOfThoughtGenerator
)

from .context_processor import (
    ContextCompressionStrategy,
    RelevanceFilter,
    ContextProcessor,
    ContextWindow
)

from .pipeline import (
    GenerationPipeline,
    GenerationPipelineConfig,
    GenerationPipelineStatus,
    GenerationMetrics
)

__all__ = [
    # LLM Providers
    "LLMProvider",
    "LLMModel",
    "GenerationRequest",
    "GenerationResponse", 
    "GenerationConfig",
    "BaseLLMProvider",
    "OpenAILLMProvider",
    "AnthropicLLMProvider",
    "LLMProviderFactory",
    
    # Prompt Engineering
    "PromptTemplate",
    "PromptType",
    "PromptContext",
    "FewShotExample",
    "PromptBuilder",
    "PromptTemplateLibrary",
    
    # Response Generation
    "ResponseType",
    "SourceAttribution",
    "GeneratedResponse",
    "ContextualResponseGenerator",
    "ChainOfThoughtGenerator",
    
    # Context Processing
    "ContextCompressionStrategy",
    "RelevanceFilter",
    "ContextProcessor",
    "ContextWindow",
    
    # Pipeline
    "GenerationPipeline",
    "GenerationPipelineConfig",
    "GenerationPipelineStatus",
    "GenerationMetrics"
] 