"""
Application settings and configuration management.

This module defines the application settings using Pydantic Settings for type-safe
configuration management with environment variable support.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ============================================================================
    # API KEYS
    # ============================================================================
    
    openai_api_key: str = Field(
        ..., 
        description="OpenAI API key for LLM operations"
    )
    
    github_token: str = Field(
        ..., 
        description="GitHub personal access token for repository access"
    )
    
    pinecone_api_key: str = Field(
        ..., 
        description="Pinecone API key for vector database operations"
    )
    
    stackoverflow_api_key: Optional[str] = Field(
        None, 
        description="Stack Overflow API key (optional, for higher rate limits)"
    )
    
    # ============================================================================
    # VECTOR DATABASE CONFIGURATION
    # ============================================================================
    
    pinecone_environment: str = Field(
        default="us-east-1-aws",
        description="Pinecone environment region"
    )
    
    pinecone_index_name: str = Field(
        default="python-code-helper",
        description="Name of the Pinecone index"
    )
    
    vector_dimensions: int = Field(
        default=384,
        description="Vector dimensions (must match embedding model)"
    )
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformers embedding model name"
    )
    
    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model for text generation"
    )
    
    # ============================================================================
    # PROCESSING CONFIGURATION
    # ============================================================================
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum chunk size in tokens"
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks in tokens"
    )
    
    max_tokens: int = Field(
        default=4000,
        ge=500,
        le=32000,
        description="Maximum tokens for LLM generation"
    )
    
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for processing operations"
    )
    
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent API requests"
    )
    
    # ============================================================================
    # SERVER CONFIGURATION
    # ============================================================================
    
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    
    port: int = Field(
        default=8000,
        ge=1000,
        le=65535,
        description="Server port number"
    )
    
    workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes"
    )
    
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
        description="CORS allowed origins"
    )
    
    api_prefix: str = Field(
        default="/api/v1",
        description="API route prefix"
    )
    
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum request size in bytes"
    )
    
    request_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Request timeout in seconds"
    )
    
    # ============================================================================
    # REDIS CONFIGURATION
    # ============================================================================
    
    redis_host: str = Field(
        default="localhost",
        description="Redis host address"
    )
    
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port number"
    )
    
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password (optional)"
    )
    
    redis_url: Optional[str] = Field(
        default=None,
        description="Complete Redis URL (overrides individual settings)"
    )
    
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default cache TTL in seconds"
    )
    
    search_cache_ttl: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Search results cache TTL in seconds"
    )
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (optional)"
    )
    
    enable_structured_logging: bool = Field(
        default=True,
        description="Enable structured logging with metadata"
    )
    
    # ============================================================================
    # MONITORING & METRICS
    # ============================================================================
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics collection"
    )
    
    metrics_port: int = Field(
        default=8001,
        ge=1000,
        le=65535,
        description="Prometheus metrics server port"
    )
    
    # ============================================================================
    # SEARCH CONFIGURATION
    # ============================================================================
    
    semantic_search_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid search"
    )
    
    keyword_search_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid search"
    )
    
    max_search_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of search results"
    )
    
    default_search_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default number of search results"
    )
    
    # ============================================================================
    # DATA SOURCE CONFIGURATION
    # ============================================================================
    
    github_min_stars: int = Field(
        default=100,
        ge=0,
        description="Minimum stars for GitHub repositories"
    )
    
    github_max_file_size: int = Field(
        default=100000,
        ge=1000,
        description="Maximum file size to process (bytes)"
    )
    
    github_languages: List[str] = Field(
        default=["python"],
        description="Programming languages to crawl"
    )
    
    github_exclude_forks: bool = Field(
        default=True,
        description="Exclude forked repositories"
    )
    
    so_min_score: int = Field(
        default=5,
        ge=0,
        description="Minimum score for Stack Overflow questions"
    )
    
    so_max_questions: int = Field(
        default=10000,
        ge=100,
        description="Maximum Stack Overflow questions to process"
    )
    
    so_tags: List[str] = Field(
        default=["python", "django", "flask", "pandas", "numpy"],
        description="Stack Overflow tags to search"
    )
    
    # ============================================================================
    # DEVELOPMENT SETTINGS
    # ============================================================================
    
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    auto_reload: bool = Field(
        default=True,
        description="Enable auto-reload in development"
    )
    
    # ============================================================================
    # COMPUTED PROPERTIES
    # ============================================================================
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def redis_connection_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_url:
            return self.redis_url
        
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # ============================================================================
    # VALIDATORS
    # ============================================================================
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v.lower()
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, v):
        """Parse comma-separated origins string into list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @field_validator("github_languages", "so_tags")
    @classmethod
    def validate_lists(cls, v):
        """Parse comma-separated strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance for dependency injection."""
    return settings 