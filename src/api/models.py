"""
Pydantic models for API requests and responses.

This module defines the data models used for API communication,
including request/response models, validation, and serialization.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, EmailStr
from pydantic.types import UUID4


class QueryType(str, Enum):
    """Types of queries supported by the system."""
    CODE_EXPLANATION = "code_explanation"
    CODE_GENERATION = "code_generation"
    DEBUGGING_HELP = "debugging_help"
    QA_RESPONSE = "qa_response"
    CONCEPT_EXPLANATION = "concept_explanation"
    BEST_PRACTICES = "best_practices"
    CODE_REVIEW = "code_review"
    TUTORIAL = "tutorial"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    PHP = "php"
    RUBY = "ruby"
    CSHARP = "csharp"


class DifficultyLevel(str, Enum):
    """Difficulty levels for responses."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SourceType(str, Enum):
    """Types of sources in the knowledge base."""
    GITHUB_CODE = "github_code"
    STACKOVERFLOW_QA = "stackoverflow_qa"
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"
    BLOG_POST = "blog_post"


# Request Models

class ContextMetadata(BaseModel):
    """Context metadata for requests."""
    programming_language: Optional[ProgrammingLanguage] = ProgrammingLanguage.PYTHON
    difficulty_level: Optional[DifficultyLevel] = DifficultyLevel.INTERMEDIATE
    domain: Optional[str] = Field(None, description="Specific domain or topic")
    include_examples: bool = Field(True, description="Include code examples in response")
    include_tests: bool = Field(False, description="Include test cases when generating code")
    style_guide: Optional[str] = Field(None, description="Code style guide to follow")
    max_response_length: Optional[int] = Field(None, ge=100, le=10000, description="Maximum response length")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message or question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context continuity")
    context: Optional[ContextMetadata] = Field(default_factory=ContextMetadata, description="Additional context metadata")
    use_chain_of_thought: Optional[bool] = Field(None, description="Enable chain-of-thought reasoning")
    stream: bool = Field(False, description="Enable streaming response")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "How does the quicksort algorithm work in Python?",
                "context": {
                    "programming_language": "python",
                    "difficulty_level": "intermediate",
                    "include_examples": True
                },
                "use_chain_of_thought": False,
                "stream": False
            }
        }


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    search_type: str = Field("hybrid", description="Type of search: semantic, keyword, hybrid, or code")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for search")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "binary search algorithm implementation",
                "top_k": 5,
                "search_type": "hybrid",
                "filters": {
                    "programming_language": "python",
                    "source_type": "github_code"
                }
            }
        }


# Response Models

class SourceAttribution(BaseModel):
    """Source attribution information."""
    chunk_id: str = Field(..., description="Unique identifier of the source chunk")
    content_snippet: str = Field(..., description="Relevant snippet from the source")
    source_type: SourceType = Field(..., description="Type of source")
    url: Optional[str] = Field(None, description="URL to the original source")
    title: Optional[str] = Field(None, description="Title of the source")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in attribution")
    usage_type: str = Field(..., description="How the source was used (example, reference, inspiration)")


class ResponseMetrics(BaseModel):
    """Metrics for the generated response."""
    generation_time: float = Field(..., description="Time taken to generate response (seconds)")
    tokens_used: int = Field(..., description="Total tokens consumed")
    prompt_tokens: int = Field(..., description="Tokens used for prompt")
    completion_tokens: int = Field(..., description="Tokens used for completion")
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Generated response text")
    conversation_id: str = Field(..., description="Conversation ID")
    query_type: QueryType = Field(..., description="Detected query type")
    sources: List[SourceAttribution] = Field(default_factory=list, description="Source attributions")
    reasoning_steps: Optional[List[str]] = Field(None, description="Chain-of-thought reasoning steps")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up question suggestions")
    metrics: ResponseMetrics = Field(..., description="Response generation metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Quicksort is a divide-and-conquer algorithm that works by selecting a 'pivot' element...",
                "conversation_id": "conv_123456",
                "query_type": "code_explanation",
                "sources": [
                    {
                        "chunk_id": "github_repo_123_chunk_456",
                        "content_snippet": "def quicksort(arr): ...",
                        "source_type": "github_code",
                        "url": "https://github.com/example/algorithms",
                        "relevance_score": 0.95,
                        "confidence": 0.88,
                        "usage_type": "example"
                    }
                ],
                "suggestions": [
                    "What's the time complexity of quicksort?",
                    "Show me a Python implementation",
                    "Compare quicksort with mergesort"
                ],
                "metrics": {
                    "generation_time": 1.23,
                    "tokens_used": 156,
                    "confidence_score": 0.92
                }
            }
        }


class StreamingChatResponse(BaseModel):
    """Streaming response model for chat endpoint."""
    content: str = Field(..., description="Partial response content")
    conversation_id: str = Field(..., description="Conversation ID")
    is_complete: bool = Field(False, description="Whether this is the final chunk")
    chunk_index: int = Field(..., description="Index of this chunk")
    
    # Only included in the final chunk
    sources: Optional[List[SourceAttribution]] = Field(None, description="Source attributions (final chunk only)")
    query_type: Optional[QueryType] = Field(None, description="Detected query type (final chunk only)")
    metrics: Optional[ResponseMetrics] = Field(None, description="Response metrics (final chunk only)")


class SearchResult(BaseModel):
    """Individual search result."""
    chunk_id: str = Field(..., description="Unique identifier of the chunk")
    content: str = Field(..., description="Content of the chunk")
    title: Optional[str] = Field(None, description="Title or summary")
    source_type: SourceType = Field(..., description="Type of source")
    url: Optional[str] = Field(None, description="URL to original source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    highlights: List[str] = Field(default_factory=list, description="Highlighted matching text")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Time taken for search (seconds)")
    query_interpretation: Optional[str] = Field(None, description="How the query was interpreted")
    suggestions: List[str] = Field(default_factory=list, description="Query suggestions")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "chunk_id": "stackoverflow_123456",
                        "content": "Binary search is an efficient algorithm for finding an item...",
                        "source_type": "stackoverflow_qa",
                        "relevance_score": 0.94,
                        "highlights": ["binary search", "algorithm", "efficient"]
                    }
                ],
                "total_results": 15,
                "search_time": 0.045,
                "suggestions": ["binary search implementation", "binary search complexity"]
            }
        }


# System Models

class SystemHealthResponse(BaseModel):
    """System health check response."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    uptime: float = Field(..., description="System uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Component health status")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-26T12:00:00Z",
                "uptime": 3600.0,
                "components": {
                    "generation_pipeline": {"status": "healthy", "latency": 0.123},
                    "vector_search": {"status": "healthy", "latency": 0.045},
                    "llm_provider": {"status": "healthy", "model": "gpt-4-turbo"}
                },
                "metrics": {
                    "total_requests": 1500,
                    "avg_response_time": 1.2,
                    "success_rate": 0.998
                }
            }
        }


# User and Authentication Models

class UserProfile(BaseModel):
    """User profile information."""
    user_id: str = Field(..., description="Unique user identifier")
    email: Optional[EmailStr] = Field(None, description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")
    usage_stats: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")


class APIKeyInfo(BaseModel):
    """API key information."""
    key_id: str = Field(..., description="API key identifier")
    name: Optional[str] = Field(None, description="Human-readable key name")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Key creation timestamp")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    usage_count: int = Field(0, description="Total usage count")
    rate_limit: Optional[int] = Field(None, description="Rate limit (requests per minute)")
    is_active: bool = Field(True, description="Whether the key is active")


class APIKeyCreateRequest(BaseModel):
    """Request to create a new API key."""
    name: Optional[str] = Field(None, description="Human-readable name for the key")
    rate_limit: Optional[int] = Field(None, ge=1, le=1000, description="Rate limit (requests per minute)")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


# Error Models

class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail = Field(..., description="Error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded. Please try again later.",
                    "details": {
                        "current_rate": 65,
                        "limit": 60,
                        "reset_at": "2025-01-26T12:01:00Z"
                    }
                },
                "timestamp": "2025-01-26T12:00:00Z",
                "request_id": "req_123456789"
            }
        } 