"""
FastAPI application factory and configuration.

This module creates and configures the main FastAPI application with middleware,
routers, exception handlers, and startup/shutdown events.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.config.settings import settings
from src.utils.logger import get_logger
from src.generation.pipeline import GenerationPipeline, GenerationPipelineConfig
from src.vector.similarity_search import SimilaritySearchEngine
from src.api.endpoints import router as api_router
from src.api.middleware import RateLimitingMiddleware, LoggingMiddleware
from src.api.exceptions import setup_exception_handlers

logger = get_logger(__name__)

# Global application state
app_state = {
    "generation_pipeline": None,
    "search_engine": None,
    "startup_time": None,
    "health_status": "starting"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Python Code Helper API")
    app_state["startup_time"] = datetime.utcnow()
    
    try:
        # Initialize search engine (mock for now)
        logger.info("Initializing search engine...")
        # app_state["search_engine"] = await initialize_search_engine()
        
        # Initialize generation pipeline
        logger.info("Initializing generation pipeline...")
        config = GenerationPipelineConfig.from_settings()
        # pipeline = GenerationPipeline(app_state["search_engine"], config)
        # await pipeline.initialize()
        # app_state["generation_pipeline"] = pipeline
        
        app_state["health_status"] = "healthy"
        logger.info("API startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        app_state["health_status"] = "unhealthy"
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down Python Code Helper API")
        
        if app_state["generation_pipeline"]:
            await app_state["generation_pipeline"].shutdown()
            
        logger.info("API shutdown completed")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    
    # Create FastAPI app with custom OpenAPI configuration
    app = FastAPI(
        title="Python Code Helper API",
        description="""
        🐍 **Python Code Helper RAG System API**
        
        An intelligent programming assistant powered by Retrieval-Augmented Generation (RAG).
        Get help with Python programming through contextual, source-attributed responses.
        
        ## Features
        
        * 🤖 **Multi-LLM Support**: OpenAI GPT, Anthropic Claude, and more
        * 🔍 **Hybrid Search**: Semantic + keyword search across GitHub repos and Stack Overflow
        * 📝 **Context-Aware Responses**: Intelligent responses with source attribution
        * 🧠 **Chain-of-Thought**: Explicit reasoning for complex queries
        * ⚡ **Real-time Streaming**: Get responses as they're generated
        * 🔐 **Secure**: API key authentication and rate limiting
        
        ## Data Sources
        
        * **GitHub Repositories**: Popular Python projects and code examples
        * **Stack Overflow**: Community Q&A with validated answers
        * **Documentation**: Official Python and library documentation
        
        ## Usage Examples
        
        ```bash
        # Simple chat request
        curl -X POST "/api/v1/chat" \\
             -H "Authorization: Bearer YOUR_API_KEY" \\
             -H "Content-Type: application/json" \\
             -d '{"message": "How does quicksort work?"}'
        
        # Streaming chat
        curl -X POST "/api/v1/chat/stream" \\
             -H "Authorization: Bearer YOUR_API_KEY" \\
             -H "Content-Type: application/json" \\
             -d '{"message": "Explain Python decorators"}' \\
             --no-buffer
        ```
        """,
        version="1.0.0",
        terms_of_service="https://github.com/your-org/python-code-helper/blob/main/TERMS.md",
        contact={
            "name": "Python Code Helper Team",
            "url": "https://github.com/your-org/python-code-helper",
            "email": "support@pythoncodehelper.dev"
        },
        license_info={
            "name": "MIT License",
            "url": "https://github.com/your-org/python-code-helper/blob/main/LICENSE"
        },
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add routers
    app.include_router(api_router, prefix="/api/v1")
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Add custom routes
    setup_custom_routes(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Set up middleware for the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(settings, 'cors_origins', ['*']),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom rate limiting middleware
    app.add_middleware(RateLimitingMiddleware)
    
    # Custom logging middleware
    app.add_middleware(LoggingMiddleware)


def setup_custom_routes(app: FastAPI) -> None:
    """Set up custom routes for the application."""
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Python Code Helper API",
            "version": "1.0.0",
            "description": "Intelligent programming assistant powered by RAG",
            "status": app_state["health_status"],
            "uptime": (
                datetime.utcnow() - app_state["startup_time"]
            ).total_seconds() if app_state["startup_time"] else 0,
            "docs_url": "/docs",
            "endpoints": {
                "chat": "/api/v1/chat",
                "stream": "/api/v1/chat/stream", 
                "search": "/api/v1/search",
                "health": "/api/v1/health"
            }
        }
    
    @app.get("/health", include_in_schema=False)
    async def health_check():
        """Simple health check endpoint."""
        return {
            "status": app_state["health_status"],
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": (
                datetime.utcnow() - app_state["startup_time"]
            ).total_seconds() if app_state["startup_time"] else 0
        }


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Custom OpenAPI schema generator.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Dict: Custom OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": "API Key authentication. Use format: 'Bearer YOUR_API_KEY'"
        }
    }
    
    # Add security to all endpoints by default
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method != "options":
                openapi_schema["paths"][path][method]["security"] = [
                    {"ApiKeyAuth": []}
                ]
    
    # Add custom examples
    openapi_schema["components"]["examples"] = {
        "ChatRequest": {
            "summary": "Simple question",
            "value": {
                "message": "How does quicksort algorithm work?",
                "context": {
                    "programming_language": "python",
                    "difficulty_level": "intermediate"
                }
            }
        },
        "CodeGenerationRequest": {
            "summary": "Code generation request",
            "value": {
                "message": "Write a binary search function in Python",
                "context": {
                    "programming_language": "python",
                    "include_tests": True,
                    "style": "pep8"
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create the main application instance
app = create_app()

# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)


def get_app_state() -> Dict[str, Any]:
    """Get current application state."""
    return app_state.copy()


def get_generation_pipeline() -> Optional[GenerationPipeline]:
    """Get the generation pipeline instance."""
    return app_state.get("generation_pipeline")


def get_search_engine() -> Optional[SimilaritySearchEngine]:
    """Get the search engine instance."""
    return app_state.get("search_engine") 