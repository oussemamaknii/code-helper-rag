"""
FastAPI application and API endpoints.

This module provides REST and streaming APIs for the Python Code Helper RAG system,
including chat endpoints, search functionality, and system management.
"""

from .app import create_app, app
from .endpoints import router as api_router
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest, 
    SearchResponse,
    StreamingChatResponse,
    SystemHealthResponse,
    UserProfile,
    APIKeyInfo
)
from .auth import (
    get_current_user,
    create_api_key,
    validate_api_key,
    APIKeyAuth,
    UserAuth
)
from .middleware import (
    RateLimitingMiddleware,
    CORSMiddleware,
    LoggingMiddleware
)
from .dependencies import (
    get_generation_pipeline,
    get_search_engine,
    get_rate_limiter,
    get_current_user_profile
)

__all__ = [
    # App
    "create_app",
    "app",
    "api_router",
    
    # Models
    "ChatRequest", 
    "ChatResponse",
    "SearchRequest",
    "SearchResponse", 
    "StreamingChatResponse",
    "SystemHealthResponse",
    "UserProfile",
    "APIKeyInfo",
    
    # Auth
    "get_current_user",
    "create_api_key", 
    "validate_api_key",
    "APIKeyAuth",
    "UserAuth",
    
    # Middleware
    "RateLimitingMiddleware",
    "CORSMiddleware", 
    "LoggingMiddleware",
    
    # Dependencies
    "get_generation_pipeline",
    "get_search_engine",
    "get_rate_limiter",
    "get_current_user_profile"
] 