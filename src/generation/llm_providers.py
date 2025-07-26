"""
Multi-provider LLM integration for response generation.

This module provides a unified interface for different LLM providers
with support for OpenAI, Anthropic, and extensible architecture for others.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from src.utils.logger import get_logger
from src.utils.async_utils import AsyncRetry, AsyncRateLimiter, async_timer
from src.config.settings import settings

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGING_FACE = "hugging_face"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


class LLMModel(Enum):
    """Supported LLM models."""
    # OpenAI models
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    
    # Anthropic models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Open source models (via Ollama or Hugging Face)
    LLAMA_2_70B = "llama2:70b"
    LLAMA_2_13B = "llama2:13b"
    CODELLAMA_34B = "codellama:34b"
    MISTRAL_7B = "mistral:7b"


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stop': self.stop_sequences if self.stop_sequences else None,
            'stream': self.stream
        }


@dataclass
class GenerationRequest:
    """Request for LLM generation."""
    messages: List[Dict[str, str]]
    config: GenerationConfig = field(default_factory=GenerationConfig)
    context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate request ID if not provided."""
        if not self.request_id:
            self.request_id = f"gen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"


@dataclass
class GenerationResponse:
    """Response from LLM generation."""
    request_id: str
    text: str
    model: str
    provider: str
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens used."""
        return self.usage.get('prompt_tokens', 0)
    
    @property
    def completion_tokens(self) -> int:
        """Get completion tokens generated."""
        return self.usage.get('completion_tokens', 0)
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get('total_tokens', self.prompt_tokens + self.completion_tokens)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'request_id': self.request_id,
            'text': self.text,
            'model': self.model,
            'provider': self.provider,
            'usage': self.usage,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class BaseLLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self,
                 model: Union[str, LLMModel],
                 provider: Union[str, LLMProvider],
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_retries: int = 3,
                 rate_limit: Optional[int] = None):
        """
        Initialize LLM provider.
        
        Args:
            model: LLM model to use
            provider: LLM provider
            api_key: API key for the provider
            base_url: Base URL for API calls
            max_retries: Maximum retry attempts
            rate_limit: Rate limit (requests per minute)
        """
        self.model = model.value if isinstance(model, LLMModel) else model
        self.provider = provider.value if isinstance(provider, LLMProvider) else provider
        self.api_key = api_key
        self.base_url = base_url
        
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
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate response for a single request.
        
        Args:
            request: Generation request
            
        Returns:
            GenerationResponse: Generated response
        """
        self.logger.debug(
            "Generating response",
            request_id=request.request_id,
            messages_count=len(request.messages),
            max_tokens=request.config.max_tokens
        )
        
        try:
            async with async_timer(f"LLM generation {request.request_id}"):
                response = await self._generate_response(request)
                
                self.logger.info(
                    "Response generated successfully",
                    request_id=request.request_id,
                    response_length=len(response.text),
                    total_tokens=response.total_tokens
                )
                
                return response
                
        except Exception as e:
            self.logger.error(
                "Response generation failed",
                request_id=request.request_id,
                error=str(e)
            )
            # Return error response
            return GenerationResponse(
                request_id=request.request_id,
                text=f"Error generating response: {str(e)}",
                model=self.model,
                provider=self.provider,
                metadata={'error': str(e)}
            )
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """
        Generate streaming response.
        
        Args:
            request: Generation request with stream=True
            
        Yields:
            str: Partial response chunks
        """
        # Set streaming in config
        request.config.stream = True
        
        self.logger.debug(
            "Starting streaming generation",
            request_id=request.request_id
        )
        
        try:
            async for chunk in self._generate_stream_response(request):
                yield chunk
                
        except Exception as e:
            self.logger.error(
                "Streaming generation failed",
                request_id=request.request_id,
                error=str(e)
            )
            yield f"Error: {str(e)}"
    
    async def _generate_response(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _generate_response")
    
    async def _generate_stream_response(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _generate_stream_response")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for LLM provider."""
        try:
            # Test with a simple request
            test_request = GenerationRequest(
                messages=[{"role": "user", "content": "Hello"}],
                config=GenerationConfig(max_tokens=10, temperature=0.0)
            )
            
            response = await self.generate(test_request)
            
            return {
                'service': f"{self.provider}_llm",
                'model': self.model,
                'status': 'healthy' if not response.metadata.get('error') else 'unhealthy',
                'test_tokens': response.total_tokens,
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'service': f"{self.provider}_llm",
                'model': self.model,
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self,
                 model: LLMModel = LLMModel.GPT_4_TURBO,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize OpenAI LLM provider.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (uses settings if not provided)
            base_url: Base URL for API calls
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            model=model,
            provider=LLMProvider.OPENAI,
            api_key=api_key or settings.openai_api_key,
            base_url=base_url,
            **kwargs
        )
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Import OpenAI client
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("openai package is required for OpenAI LLM provider")
    
    async def _generate_response(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response using OpenAI API."""
        
        # Apply rate limiting
        if self.rate_limiter:
            async with self.rate_limiter:
                pass
        
        # Call OpenAI API with retry
        @self.retry_decorator
        async def call_openai_api():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=request.messages,
                **request.config.to_dict()
            )
            return response
        
        response = await call_openai_api()
        
        # Extract response text
        text = response.choices[0].message.content or ""
        
        # Extract usage information
        usage = {}
        if response.usage:
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        return GenerationResponse(
            request_id=request.request_id,
            text=text,
            model=self.model,
            provider=self.provider,
            usage=usage,
            metadata={
                'finish_reason': response.choices[0].finish_reason,
                'response_id': response.id
            }
        )
    
    async def _generate_stream_response(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI API."""
        
        # Apply rate limiting
        if self.rate_limiter:
            async with self.rate_limiter:
                pass
        
        # Call OpenAI streaming API with retry
        @self.retry_decorator
        async def call_openai_stream():
            return await self.client.chat.completions.create(
                model=self.model,
                messages=request.messages,
                **request.config.to_dict()
            )
        
        stream = await call_openai_stream()
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""
    
    def __init__(self,
                 model: LLMModel = LLMModel.CLAUDE_3_SONNET,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize Anthropic LLM provider.
        
        Args:
            model: Anthropic model to use
            api_key: Anthropic API key (uses settings if not provided)
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            model=model,
            provider=LLMProvider.ANTHROPIC,
            api_key=api_key or getattr(settings, 'anthropic_api_key', None),
            **kwargs
        )
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Import Anthropic client
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic LLM provider")
    
    async def _generate_response(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response using Anthropic API."""
        
        # Apply rate limiting
        if self.rate_limiter:
            async with self.rate_limiter:
                pass
        
        # Convert OpenAI format messages to Anthropic format
        system_message = ""
        messages = []
        
        for msg in request.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Call Anthropic API with retry
        @self.retry_decorator
        async def call_anthropic_api():
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=request.config.max_tokens,
                temperature=request.config.temperature,
                system=system_message if system_message else None,
                messages=messages
            )
            return response
        
        response = await call_anthropic_api()
        
        # Extract response text
        text = ""
        if response.content:
            text = "".join([block.text for block in response.content if hasattr(block, 'text')])
        
        # Extract usage information
        usage = {}
        if response.usage:
            usage = {
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
        
        return GenerationResponse(
            request_id=request.request_id,
            text=text,
            model=self.model,
            provider=self.provider,
            usage=usage,
            metadata={
                'stop_reason': response.stop_reason,
                'response_id': response.id
            }
        )
    
    async def _generate_stream_response(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response using Anthropic API."""
        
        # Apply rate limiting
        if self.rate_limiter:
            async with self.rate_limiter:
                pass
        
        # Convert messages format
        system_message = ""
        messages = []
        
        for msg in request.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Call Anthropic streaming API with retry
        @self.retry_decorator
        async def call_anthropic_stream():
            return await self.client.messages.create(
                model=self.model,
                max_tokens=request.config.max_tokens,
                temperature=request.config.temperature,
                system=system_message if system_message else None,
                messages=messages,
                stream=True
            )
        
        stream = await call_anthropic_stream()
        
        async for chunk in stream:
            if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                yield chunk.delta.text


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider: Union[str, LLMProvider],
                       model: Optional[Union[str, LLMModel]] = None,
                       **kwargs) -> BaseLLMProvider:
        """
        Create an LLM provider.
        
        Args:
            provider: LLM provider
            model: LLM model (uses default if not specified)
            **kwargs: Additional arguments for the provider
            
        Returns:
            BaseLLMProvider: Configured LLM provider
        """
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        
        if provider == LLMProvider.OPENAI:
            default_model = LLMModel.GPT_4_TURBO
            if model:
                if isinstance(model, str):
                    # Try to find matching enum value
                    for llm_model in LLMModel:
                        if llm_model.value == model:
                            model = llm_model
                            break
                    else:
                        # Use string directly if no enum match
                        pass
            else:
                model = default_model
            
            return OpenAILLMProvider(model=model, **kwargs)
        
        elif provider == LLMProvider.ANTHROPIC:
            default_model = LLMModel.CLAUDE_3_SONNET
            if model:
                if isinstance(model, str):
                    # Try to find matching enum value
                    for llm_model in LLMModel:
                        if llm_model.value == model:
                            model = llm_model
                            break
                    else:
                        # Use string directly if no enum match
                        pass
            else:
                model = default_model
            
            return AnthropicLLMProvider(model=model, **kwargs)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_default_provider() -> BaseLLMProvider:
        """Get default LLM provider based on configuration."""
        provider = getattr(settings, 'llm_provider', 'openai')
        model = getattr(settings, 'llm_model', None)
        
        return LLMProviderFactory.create_provider(
            provider=provider,
            model=model,
            rate_limit=getattr(settings, 'llm_rate_limit', 60)
        ) 