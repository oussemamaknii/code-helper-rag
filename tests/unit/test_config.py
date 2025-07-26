"""
Unit tests for configuration management.

Tests the Pydantic settings implementation and environment variable handling.
"""

import pytest
import os
from unittest.mock import patch

from src.config.settings import Settings, get_settings


class TestSettings:
    """Test cases for Settings class."""

    def test_settings_initialization(self):
        """Test that settings can be initialized with default values."""
        # Mock required environment variables
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_openai_key',
            'GITHUB_TOKEN': 'test_github_token',
            'PINECONE_API_KEY': 'test_pinecone_key'
        }):
            settings = Settings()
            
            assert settings.openai_api_key == 'test_openai_key'
            assert settings.github_token == 'test_github_token'
            assert settings.pinecone_api_key == 'test_pinecone_key'
            
            # Test defaults
            assert settings.pinecone_environment == 'us-east-1-aws'
            assert settings.embedding_model == 'all-MiniLM-L6-v2'
            assert settings.chunk_size == 1000
            assert settings.chunk_overlap == 200

    def test_environment_properties(self):
        """Test environment-related properties."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'ENVIRONMENT': 'development'
        }):
            settings = Settings()
            
            assert settings.is_development is True
            assert settings.is_production is False

    def test_redis_connection_url(self):
        """Test Redis connection URL generation."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'REDIS_DB': '0'
        }):
            settings = Settings()
            
            expected_url = 'redis://localhost:6379/0'
            assert settings.redis_connection_url == expected_url

    def test_validator_log_level(self):
        """Test log level validation."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'LOG_LEVEL': 'debug'
        }):
            settings = Settings()
            assert settings.log_level == 'DEBUG'  # Should be uppercase

    def test_validator_search_weights(self):
        """Test search weight validation."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'SEMANTIC_SEARCH_WEIGHT': '0.6',
            'KEYWORD_SEARCH_WEIGHT': '0.4'
        }):
            settings = Settings()
            assert abs(settings.semantic_search_weight + settings.keyword_search_weight - 1.0) < 0.01

    def test_validator_chunk_overlap(self):
        """Test chunk overlap validation."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'CHUNK_SIZE': '1000',
            'CHUNK_OVERLAP': '200'
        }):
            settings = Settings()
            assert settings.chunk_overlap < settings.chunk_size

    def test_get_settings_function(self):
        """Test the get_settings dependency injection function."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert settings.openai_api_key == 'test_key'

    def test_project_root_property(self):
        """Test project root directory detection."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            settings = Settings()
            project_root = settings.project_root
            
            # Should be a Path object
            assert hasattr(project_root, 'exists')
            # Should end with our project directory
            assert 'code helper rag' in str(project_root) or 'python-code-helper' in str(project_root)

    @pytest.mark.parametrize("log_format,expected", [
        ("json", "json"),
        ("JSON", "json"),
        ("text", "text"),
        ("TEXT", "text")
    ])
    def test_log_format_validation(self, log_format, expected):
        """Test log format validation and normalization."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'LOG_FORMAT': log_format
        }):
            settings = Settings()
            assert settings.log_format == expected

    def test_list_field_parsing(self):
        """Test parsing of comma-separated list fields."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'ALLOWED_ORIGINS': 'http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000',
            'SO_TAGS': 'python,django,flask,fastapi'
        }):
            settings = Settings()
            
            expected_origins = ['http://localhost:3000', 'http://localhost:8080', 'http://127.0.0.1:3000']
            expected_tags = ['python', 'django', 'flask', 'fastapi']
            
            assert settings.allowed_origins == expected_origins
            assert settings.so_tags == expected_tags 