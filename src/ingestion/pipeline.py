"""
Data ingestion pipeline for orchestrating multiple collectors.

This module provides a pipeline that can coordinate multiple data collectors,
handle errors, track progress, and manage the overall ingestion process.
"""

import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from src.ingestion.base_collector import BaseCollector, CollectedItem, CollectionStatus
from src.ingestion.github_crawler import GitHubCrawler
from src.ingestion.stackoverflow_collector import StackOverflowCollector
from src.utils.logger import get_logger
from src.utils.async_utils import gather_with_concurrency, async_timer
from src.config.settings import settings

logger = get_logger(__name__)


class PipelineStatus(Enum):
    """Status of the ingestion pipeline."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline."""
    
    # GitHub configuration
    enable_github: bool = True
    github_max_repos: int = 50
    github_max_files_per_repo: int = 50
    
    # Stack Overflow configuration  
    enable_stackoverflow: bool = True
    stackoverflow_max_questions: int = 1000
    stackoverflow_max_answers_per_question: int = 5
    
    # Processing configuration
    max_concurrent_collectors: int = 2
    batch_size: int = 100
    
    # Output configuration
    save_to_file: bool = False
    output_directory: str = "./data/ingested"
    
    # Error handling
    continue_on_error: bool = True
    max_errors_per_collector: int = 100


@dataclass
class PipelineMetrics:
    """Metrics for the entire pipeline."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_items_collected: int = 0
    items_by_source: Dict[str, int] = field(default_factory=dict)
    collector_metrics: Dict[str, Dict] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate total processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def items_per_second(self) -> Optional[float]:
        """Calculate processing rate."""
        processing_time = self.processing_time
        if processing_time and processing_time > 0:
            return self.total_items_collected / processing_time
        return None


class DataIngestionPipeline:
    """
    Orchestrates multiple data collectors for comprehensive ingestion.
    
    Features:
    - Concurrent execution of multiple collectors
    - Progress tracking and metrics
    - Error handling and recovery
    - Configurable output options
    - Health monitoring
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.status = PipelineStatus.IDLE
        self.metrics = PipelineMetrics()
        
        # Initialize collectors
        self.collectors: Dict[str, BaseCollector] = {}
        
        self.logger = get_logger(__name__, pipeline="data_ingestion")
        
        self._initialize_collectors()
        
        self.logger.info(
            "Pipeline initialized",
            github_enabled=self.config.enable_github,
            stackoverflow_enabled=self.config.enable_stackoverflow,
            max_concurrent=self.config.max_concurrent_collectors
        )
    
    def _initialize_collectors(self) -> None:
        """Initialize all configured collectors."""
        if self.config.enable_github:
            try:
                github_collector = GitHubCrawler(
                    github_token=settings.github_token,
                    max_file_size=settings.github_max_file_size,
                    min_stars=settings.github_min_stars,
                    exclude_forks=settings.github_exclude_forks,
                    languages=settings.github_languages,
                    batch_size=self.config.batch_size
                )
                self.collectors["github"] = github_collector
                self.logger.info("GitHub collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize GitHub collector: {e}")
                if not self.config.continue_on_error:
                    raise
        
        if self.config.enable_stackoverflow:
            try:
                so_collector = StackOverflowCollector(
                    api_key=settings.stackoverflow_api_key,
                    min_question_score=settings.so_min_score,
                    tags=settings.so_tags,
                    batch_size=self.config.batch_size
                )
                self.collectors["stackoverflow"] = so_collector
                self.logger.info("Stack Overflow collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Stack Overflow collector: {e}")
                if not self.config.continue_on_error:
                    raise
    
    async def run_pipeline(self) -> AsyncGenerator[CollectedItem, None]:
        """
        Run the complete ingestion pipeline.
        
        Yields:
            CollectedItem: Items collected from all sources
        """
        self.logger.info("Starting data ingestion pipeline")
        
        self.status = PipelineStatus.RUNNING
        self.metrics.start_time = datetime.utcnow()
        
        try:
            async with async_timer("Complete pipeline execution"):
                
                # Run health checks first
                await self._run_health_checks()
                
                # Create collector tasks
                collector_tasks = []
                
                if "github" in self.collectors:
                    github_task = self._run_github_collection()
                    collector_tasks.append(("github", github_task))
                
                if "stackoverflow" in self.collectors:
                    so_task = self._run_stackoverflow_collection()
                    collector_tasks.append(("stackoverflow", so_task))
                
                if not collector_tasks:
                    self.logger.warning("No collectors available to run")
                    return
                
                # Run collectors with controlled concurrency
                async for item in self._run_collectors_concurrently(collector_tasks):
                    yield item
            
            self.status = PipelineStatus.COMPLETED
            self.logger.info(
                "Pipeline completed successfully",
                total_items=self.metrics.total_items_collected,
                processing_time=self.metrics.processing_time,
                items_per_second=self.metrics.items_per_second
            )
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.metrics.errors.append(f"Pipeline error: {str(e)}")
            self.logger.error(f"Pipeline failed: {e}")
            
            if not self.config.continue_on_error:
                raise
        
        finally:
            self.metrics.end_time = datetime.utcnow()
    
    async def _run_collectors_concurrently(self, 
                                          collector_tasks: List[tuple]) -> AsyncGenerator[CollectedItem, None]:
        """Run collectors concurrently and yield items."""
        
        # Convert async generators to async iterators
        async_iterators = []
        for name, task in collector_tasks:
            async_iterators.append(self._collect_with_metrics(name, task))
        
        # Use a simple approach to interleave results
        active_iterators = list(async_iterators)
        
        while active_iterators:
            # Create tasks for getting next item from each iterator
            next_tasks = []
            for i, iterator in enumerate(active_iterators):
                task = asyncio.create_task(self._get_next_item(iterator))
                next_tasks.append((i, task))
            
            if not next_tasks:
                break
            
            # Wait for first completed task
            done, pending = await asyncio.wait(
                [task for _, task in next_tasks],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Process completed tasks
            completed_indices = []
            for i, task in next_tasks:
                if task in done:
                    try:
                        item = await task
                        if item is not None:
                            yield item
                        else:
                            # Iterator exhausted
                            completed_indices.append(i)
                    except Exception as e:
                        self.logger.error(f"Error in collector: {e}")
                        completed_indices.append(i)
            
            # Remove exhausted iterators
            for i in sorted(completed_indices, reverse=True):
                active_iterators.pop(i)
    
    async def _get_next_item(self, async_iterator) -> Optional[CollectedItem]:
        """Get next item from async iterator."""
        try:
            return await async_iterator.__anext__()
        except StopAsyncIteration:
            return None
    
    async def _collect_with_metrics(self, 
                                   collector_name: str, 
                                   collection_task) -> AsyncGenerator[CollectedItem, None]:
        """Collect items and track metrics."""
        try:
            async for item in collection_task:
                # Update metrics
                self.metrics.total_items_collected += 1
                source_type = item.source_type
                self.metrics.items_by_source[source_type] = (
                    self.metrics.items_by_source.get(source_type, 0) + 1
                )
                
                # Save to file if configured
                if self.config.save_to_file:
                    await self._save_item_to_file(item)
                
                yield item
                
        except Exception as e:
            error_msg = f"{collector_name} collector error: {str(e)}"
            self.metrics.errors.append(error_msg)
            self.logger.error(error_msg)
            
            if not self.config.continue_on_error:
                raise
        
        finally:
            # Store collector metrics
            if collector_name in self.collectors:
                collector = self.collectors[collector_name]
                self.metrics.collector_metrics[collector_name] = collector.get_metrics()
    
    async def _run_github_collection(self) -> AsyncGenerator[CollectedItem, None]:
        """Run GitHub collection."""
        collector = self.collectors["github"]
        
        async for item in collector.run_collection(
            max_repos=self.config.github_max_repos,
            max_files_per_repo=self.config.github_max_files_per_repo
        ):
            yield item
    
    async def _run_stackoverflow_collection(self) -> AsyncGenerator[CollectedItem, None]:
        """Run Stack Overflow collection."""
        collector = self.collectors["stackoverflow"]
        
        # Use context manager for session management
        async with collector:
            async for item in collector.run_collection(
                max_questions=self.config.stackoverflow_max_questions,
                max_answers_per_question=self.config.stackoverflow_max_answers_per_question
            ):
                yield item
    
    async def _run_health_checks(self) -> None:
        """Run health checks on all collectors."""
        self.logger.info("Running collector health checks")
        
        health_tasks = []
        for name, collector in self.collectors.items():
            health_tasks.append(collector.health_check())
        
        try:
            health_results = await gather_with_concurrency(
                health_tasks, 
                max_concurrency=len(self.collectors)
            )
            
            for i, (name, result) in enumerate(zip(self.collectors.keys(), health_results)):
                if isinstance(result, Exception):
                    self.logger.error(f"Health check failed for {name}: {result}")
                    if not self.config.continue_on_error:
                        raise result
                else:
                    self.logger.info(f"Health check passed for {name}")
                    
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            if not self.config.continue_on_error:
                raise
    
    async def _save_item_to_file(self, item: CollectedItem) -> None:
        """Save collected item to file."""
        try:
            import os
            import aiofiles
            
            # Create output directory if it doesn't exist
            os.makedirs(self.config.output_directory, exist_ok=True)
            
            # Create filename based on source type and date
            date_str = datetime.utcnow().strftime("%Y%m%d")
            filename = f"{item.source_type}_{date_str}.jsonl"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # Append item to file
            async with aiofiles.open(filepath, mode='a', encoding='utf-8') as f:
                json_line = json.dumps(item.to_dict(), ensure_ascii=False)
                await f.write(json_line + '\n')
                
        except Exception as e:
            self.logger.warning(f"Failed to save item to file: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            "status": self.status.value,
            "metrics": {
                "total_items_collected": self.metrics.total_items_collected,
                "items_by_source": self.metrics.items_by_source,
                "processing_time": self.metrics.processing_time,
                "items_per_second": self.metrics.items_per_second,
                "error_count": len(self.metrics.errors),
                "start_time": self.metrics.start_time.isoformat() if self.metrics.start_time else None,
                "end_time": self.metrics.end_time.isoformat() if self.metrics.end_time else None
            },
            "collector_metrics": self.metrics.collector_metrics,
            "config": {
                "github_enabled": self.config.enable_github,
                "stackoverflow_enabled": self.config.enable_stackoverflow,
                "max_concurrent_collectors": self.config.max_concurrent_collectors,
                "batch_size": self.config.batch_size
            }
        }
    
    async def stop_pipeline(self) -> None:
        """Stop the pipeline gracefully."""
        self.logger.info("Stopping pipeline")
        self.status = PipelineStatus.CANCELLED
        
        # TODO: Implement graceful shutdown logic
        # This would involve cancelling running tasks and cleaning up resources 