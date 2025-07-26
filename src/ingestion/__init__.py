"""
Data ingestion module.

This module handles data collection from various sources including
GitHub repositories and Stack Overflow Q&As with async processing
and error handling.
"""

from .base_collector import (
    BaseCollector, 
    CollectedItem, 
    CollectionStatus, 
    CollectionMetrics,
    DefaultItemValidator
)
from .github_crawler import GitHubCrawler, GitHubCodeItem
from .stackoverflow_collector import StackOverflowCollector, StackOverflowQAItem
from .pipeline import DataIngestionPipeline, PipelineConfig, PipelineStatus

__all__ = [
    "BaseCollector",
    "CollectedItem", 
    "CollectionStatus",
    "CollectionMetrics", 
    "DefaultItemValidator",
    "GitHubCrawler",
    "GitHubCodeItem",
    "StackOverflowCollector", 
    "StackOverflowQAItem",
    "DataIngestionPipeline",
    "PipelineConfig",
    "PipelineStatus"
] 