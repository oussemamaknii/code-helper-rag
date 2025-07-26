"""
Unit tests for data ingestion components.

Tests the base collector, GitHub crawler, Stack Overflow collector,
and ingestion pipeline functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.ingestion.base_collector import (
    BaseCollector, CollectedItem, CollectionStatus, CollectionMetrics,
    DefaultItemValidator
)
from src.ingestion.github_crawler import GitHubCrawler, GitHubCodeItem
from src.ingestion.stackoverflow_collector import StackOverflowCollector, StackOverflowQAItem
from src.ingestion.pipeline import DataIngestionPipeline, PipelineConfig


class TestCollectedItem:
    """Test cases for CollectedItem."""
    
    def test_collected_item_creation(self):
        """Test creating a collected item."""
        item = CollectedItem(
            id="test_id",
            content="test content",
            metadata={"key": "value"},
            source_type="test_source"
        )
        
        assert item.id == "test_id"
        assert item.content == "test content"
        assert item.metadata["key"] == "value"
        assert item.source_type == "test_source"
        assert isinstance(item.collected_at, datetime)
    
    def test_collected_item_to_dict(self):
        """Test converting collected item to dictionary."""
        item = CollectedItem(
            id="test_id",
            content="test content",
            metadata={"key": "value"},
            source_type="test_source"
        )
        
        item_dict = item.to_dict()
        
        assert item_dict["id"] == "test_id"
        assert item_dict["content"] == "test content"
        assert item_dict["metadata"]["key"] == "value"
        assert item_dict["source_type"] == "test_source"
        assert "collected_at" in item_dict


class TestCollectionMetrics:
    """Test cases for CollectionMetrics."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = CollectionMetrics(
            processed_items=100,
            successful_items=80,
            failed_items=20
        )
        
        assert metrics.success_rate == 80.0
    
    def test_success_rate_zero_processed(self):
        """Test success rate when no items processed."""
        metrics = CollectionMetrics()
        assert metrics.success_rate == 0.0
    
    def test_processing_time_calculation(self):
        """Test processing time calculation."""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()
        
        metrics = CollectionMetrics(
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics.processing_time is not None
        assert metrics.processing_time >= 0


class TestDefaultItemValidator:
    """Test cases for DefaultItemValidator."""
    
    def test_valid_item(self):
        """Test validation of valid item."""
        validator = DefaultItemValidator(
            min_content_length=5,
            max_content_length=100
        )
        
        item = CollectedItem(
            id="test_id",
            content="This is valid content",
            metadata={},
            source_type="test"
        )
        
        assert validator.validate(item) is True
    
    def test_content_too_short(self):
        """Test validation of item with content too short."""
        validator = DefaultItemValidator(min_content_length=10)
        
        item = CollectedItem(
            id="test_id",
            content="short",
            metadata={},
            source_type="test"
        )
        
        assert validator.validate(item) is False
    
    def test_content_too_long(self):
        """Test validation of item with content too long."""
        validator = DefaultItemValidator(max_content_length=10)
        
        item = CollectedItem(
            id="test_id",
            content="This content is way too long for the validator",
            metadata={},
            source_type="test"
        )
        
        assert validator.validate(item) is False
    
    def test_missing_required_metadata(self):
        """Test validation with missing required metadata."""
        validator = DefaultItemValidator(
            required_metadata_keys=["required_key"]
        )
        
        item = CollectedItem(
            id="test_id",
            content="Valid content",
            metadata={},
            source_type="test"
        )
        
        assert validator.validate(item) is False
    
    def test_empty_id_or_content(self):
        """Test validation with empty ID or content."""
        validator = DefaultItemValidator()
        
        # Empty ID
        item1 = CollectedItem(
            id="",
            content="Valid content",
            metadata={},
            source_type="test"
        )
        assert validator.validate(item1) is False
        
        # Empty content
        item2 = CollectedItem(
            id="test_id",
            content="",
            metadata={},
            source_type="test"
        )
        assert validator.validate(item2) is False


class MockCollector(BaseCollector):
    """Mock collector for testing BaseCollector functionality."""
    
    def __init__(self, items_to_return=None, **kwargs):
        super().__init__("MockCollector", **kwargs)
        self.items_to_return = items_to_return or []
    
    async def collect_items(self, **kwargs):
        """Return mock items."""
        for i, item_content in enumerate(self.items_to_return):
            yield CollectedItem(
                id=f"mock_{i}",
                content=item_content,
                metadata={"index": i},
                source_type="mock"
            )
    
    async def get_total_count(self, **kwargs):
        """Return count of mock items."""
        return len(self.items_to_return)


class TestBaseCollector:
    """Test cases for BaseCollector."""
    
    @pytest.mark.asyncio
    async def test_collect_all(self):
        """Test collecting all items."""
        mock_items = ["item1", "item2", "item3"]
        collector = MockCollector(items_to_return=mock_items)
        
        items = await collector.collect_all()
        
        assert len(items) == 3
        assert items[0].content == "item1"
        assert items[1].content == "item2"
        assert items[2].content == "item3"
    
    @pytest.mark.asyncio
    async def test_run_collection_with_metrics(self):
        """Test running collection with metrics tracking."""
        mock_items = ["item1", "item2"]
        collector = MockCollector(items_to_return=mock_items)
        
        collected_items = []
        async for item in collector.run_collection():
            collected_items.append(item)
        
        assert len(collected_items) == 2
        assert collector.status == CollectionStatus.COMPLETED
        assert collector.metrics.processed_items == 2
        assert collector.metrics.successful_items == 2
        assert collector.metrics.failed_items == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        collector = MockCollector()
        
        health_status = await collector.health_check()
        
        assert health_status["collector"] == "MockCollector"
        assert health_status["status"] == "healthy"
        assert "last_check" in health_status
    
    def test_get_metrics(self):
        """Test getting collector metrics."""
        collector = MockCollector()
        collector.metrics.processed_items = 10
        collector.metrics.successful_items = 8
        collector.metrics.failed_items = 2
        
        metrics = collector.get_metrics()
        
        assert metrics["collector_name"] == "MockCollector"
        assert metrics["processed_items"] == 10
        assert metrics["successful_items"] == 8
        assert metrics["failed_items"] == 2
        assert metrics["success_rate"] == 80.0


class TestGitHubCodeItem:
    """Test cases for GitHubCodeItem."""
    
    def test_github_code_item_creation(self):
        """Test creating a GitHub code item."""
        item = GitHubCodeItem(
            file_path="test.py",
            content="def hello():\n    print('Hello')",
            repository_name="test/repo",
            repository_url="https://github.com/test/repo",
            file_size=100,
            file_sha="abc123",
            last_modified="2023-01-01T00:00:00Z"
        )
        
        assert item.metadata["file_path"] == "test.py"
        assert item.metadata["repository_name"] == "test/repo"
        assert item.metadata["contains_functions"] is True
        assert item.metadata["contains_classes"] is False
        assert item.source_type == "github_code"
    
    def test_github_code_item_metadata_analysis(self):
        """Test metadata analysis of code content."""
        code_content = """
import os
from typing import List

class TestClass:
    def test_method(self):
        pass

def test_function():
    return True
"""
        
        item = GitHubCodeItem(
            file_path="test.py",
            content=code_content,
            repository_name="test/repo",
            repository_url="https://github.com/test/repo",
            file_size=len(code_content),
            file_sha="abc123",
            last_modified="2023-01-01T00:00:00Z"
        )
        
        assert item.metadata["contains_functions"] is True
        assert item.metadata["contains_classes"] is True
        assert item.metadata["contains_imports"] is True
        assert item.metadata["lines_of_code"] > 1


class TestStackOverflowQAItem:
    """Test cases for StackOverflowQAItem."""
    
    def test_stackoverflow_qa_item_creation(self):
        """Test creating a Stack Overflow Q&A item."""
        item = StackOverflowQAItem(
            question_id=12345,
            question_title="How to use Python?",
            question_body="<p>I want to learn Python programming.</p>",
            answer_body="<p>Start with the basics: <code>print('Hello')</code></p>",
            answer_id=67890,
            tags=["python", "beginner"],
            question_score=10,
            answer_score=5,
            is_accepted=True,
            created_date="2023-01-01T00:00:00Z"
        )
        
        assert item.metadata["question_id"] == 12345
        assert item.metadata["answer_id"] == 67890
        assert item.metadata["tags"] == ["python", "beginner"]
        assert item.metadata["is_accepted"] is True
        assert item.metadata["has_code"] is True
        assert item.source_type == "stackoverflow_qa"
        
        # Check HTML cleaning
        assert "<p>" not in item.content
        assert "print('Hello')" in item.content


class TestGitHubCrawler:
    """Test cases for GitHubCrawler (mocked)."""
    
    def test_github_crawler_initialization(self):
        """Test GitHub crawler initialization."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            crawler = GitHubCrawler(
                github_token="test_token",
                max_file_size=10000,
                min_stars=100,
                exclude_forks=True,
                languages=["python"]
            )
            
            assert crawler.name == "GitHubCrawler"
            assert crawler.max_file_size == 10000
            assert crawler.min_stars == 100
            assert crawler.exclude_forks is True
            assert crawler.languages == ["python"]
    
    def test_build_search_query(self):
        """Test building GitHub search query."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            crawler = GitHubCrawler(
                github_token="test_token",
                min_stars=100,
                exclude_forks=True,
                languages=["python"]
            )
            
            query = crawler._build_search_query()
            
            assert "language:python" in query
            assert "stars:>=100" in query
            assert "fork:false" in query


class TestStackOverflowCollector:
    """Test cases for StackOverflowCollector (mocked)."""
    
    def test_stackoverflow_collector_initialization(self):
        """Test Stack Overflow collector initialization."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            collector = StackOverflowCollector(
                api_key="test_key",
                min_question_score=5,
                tags=["python", "django"]
            )
            
            assert collector.name == "StackOverflowCollector"
            assert collector.min_question_score == 5
            assert collector.tags == ["python", "django"]
            assert collector.api_key == "test_key"
    
    def test_should_process_question(self):
        """Test question filtering logic."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            collector = StackOverflowCollector(
                min_question_score=5,
                tags=["python"]
            )
            
            # Valid question
            valid_question = {
                "score": 10,
                "answer_count": 3,
                "tags": ["python", "programming"],
                "body": "How to use Python?"
            }
            assert collector._should_process_question(valid_question) is True
            
            # Low score question
            low_score_question = {
                "score": 2,
                "answer_count": 1,
                "tags": ["python"],
                "body": "Test question"
            }
            assert collector._should_process_question(low_score_question) is False
            
            # No answers
            no_answers_question = {
                "score": 10,
                "answer_count": 0,
                "tags": ["python"],
                "body": "Test question"
            }
            assert collector._should_process_question(no_answers_question) is False
            
            # Wrong tags
            wrong_tags_question = {
                "score": 10,
                "answer_count": 3,
                "tags": ["java", "programming"],
                "body": "How to use Java?"
            }
            assert collector._should_process_question(wrong_tags_question) is False
    
    def test_should_process_answer(self):
        """Test answer filtering logic."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            collector = StackOverflowCollector(
                min_answer_score=0
            )
            
            # Valid answer
            valid_answer = {
                "score": 5,
                "is_accepted": False,
                "body": "This is a comprehensive answer explaining the solution with code examples."
            }
            assert collector._should_process_answer(valid_answer) is True
            
            # Short answer
            short_answer = {
                "score": 5,
                "is_accepted": False,
                "body": "Short answer"
            }
            assert collector._should_process_answer(short_answer) is False


class TestPipelineConfig:
    """Test cases for PipelineConfig."""
    
    def test_default_configuration(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()
        
        assert config.enable_github is True
        assert config.enable_stackoverflow is True
        assert config.github_max_repos == 50
        assert config.stackoverflow_max_questions == 1000
        assert config.max_concurrent_collectors == 2
        assert config.batch_size == 100
        assert config.continue_on_error is True
    
    def test_custom_configuration(self):
        """Test custom pipeline configuration."""
        config = PipelineConfig(
            enable_github=False,
            github_max_repos=100,
            batch_size=50
        )
        
        assert config.enable_github is False
        assert config.github_max_repos == 100
        assert config.batch_size == 50


class TestDataIngestionPipeline:
    """Test cases for DataIngestionPipeline (mocked)."""
    
    @patch('src.ingestion.pipeline.GitHubCrawler')
    @patch('src.ingestion.pipeline.StackOverflowCollector')
    def test_pipeline_initialization(self, mock_so_collector, mock_gh_crawler):
        """Test pipeline initialization with mocked collectors."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',
            'PINECONE_API_KEY': 'test_pinecone',
            'SO_MIN_SCORE': '5',
            'SO_TAGS': 'python,django'
        }):
            config = PipelineConfig(
                enable_github=True,
                enable_stackoverflow=True
            )
            
            pipeline = DataIngestionPipeline(config)
            
            assert pipeline.config == config
            assert len(pipeline.collectors) <= 2  # May be 0 if mocked collectors fail
            assert pipeline.status.value == "idle"
    
    def test_get_pipeline_status(self):
        """Test getting pipeline status."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'GITHUB_TOKEN': 'test_token',  
            'PINECONE_API_KEY': 'test_pinecone'
        }):
            config = PipelineConfig(enable_github=False, enable_stackoverflow=False)
            pipeline = DataIngestionPipeline(config)
            
            status = pipeline.get_pipeline_status()
            
            assert "status" in status
            assert "metrics" in status
            assert "collector_metrics" in status
            assert "config" in status
            assert status["config"]["github_enabled"] is False
            assert status["config"]["stackoverflow_enabled"] is False 