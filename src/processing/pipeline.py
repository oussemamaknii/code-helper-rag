"""
Data processing pipeline orchestrator.

This module provides a pipeline that coordinates multiple data processors,
handles routing, tracks progress, and manages the overall processing workflow.
"""

import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.processing.base_processor import BaseProcessor, ProcessedChunk, ProcessingStatus
from src.processing.code_processor import CodeProcessor
from src.processing.qa_processor import QAProcessor
from src.ingestion.base_collector import CollectedItem
from src.utils.logger import get_logger
from src.utils.async_utils import gather_with_concurrency, async_timer
from src.config.settings import settings

logger = get_logger(__name__)


class ProcessingPipelineStatus(Enum):
    """Status of the processing pipeline."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingPipelineConfig:
    """Configuration for the processing pipeline."""
    
    # Processor configuration
    enable_code_processor: bool = True
    enable_qa_processor: bool = True
    max_concurrent_processors: int = 3
    
    # Processing parameters
    batch_size: int = 10
    max_items_per_batch: int = 100
    
    # Error handling
    continue_on_error: bool = True
    max_errors_per_processor: int = 50
    
    # Output configuration
    save_chunks: bool = False
    output_directory: str = "./data/processed"


@dataclass
class PipelineMetrics:
    """Metrics for the entire processing pipeline."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_items_processed: int = 0
    total_chunks_created: int = 0
    items_by_source: Dict[str, int] = field(default_factory=dict)
    chunks_by_type: Dict[str, int] = field(default_factory=dict)
    processor_metrics: Dict[str, Dict] = field(default_factory=dict)
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
            return self.total_items_processed / processing_time
        return None
    
    @property
    def chunks_per_item(self) -> float:
        """Calculate average chunks created per item."""
        if self.total_items_processed == 0:
            return 0.0
        return self.total_chunks_created / self.total_items_processed


class DataProcessingPipeline:
    """
    Orchestrates multiple data processors for comprehensive content processing.
    
    Features:
    - Automatic processor routing based on content type
    - Concurrent processing with controlled resource usage
    - Progress tracking and comprehensive metrics
    - Error handling and recovery
    - Health monitoring for all processors
    """
    
    def __init__(self, config: Optional[ProcessingPipelineConfig] = None):
        """
        Initialize the processing pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or ProcessingPipelineConfig()
        self.status = ProcessingPipelineStatus.IDLE
        self.metrics = PipelineMetrics()
        
        # Initialize processors
        self.processors: Dict[str, BaseProcessor] = {}
        self.logger = get_logger(__name__, pipeline="data_processing")
        
        self._initialize_processors()
        
        self.logger.info(
            "Processing pipeline initialized",
            code_processor_enabled=self.config.enable_code_processor,
            qa_processor_enabled=self.config.enable_qa_processor,
            max_concurrent=self.config.max_concurrent_processors
        )
    
    def _initialize_processors(self) -> None:
        """Initialize all configured processors."""
        
        if self.config.enable_code_processor:
            try:
                code_processor = CodeProcessor(
                    batch_size=self.config.batch_size,
                    max_retries=3
                )
                self.processors["code"] = code_processor
                self.logger.info("Code processor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize code processor: {e}")
                if not self.config.continue_on_error:
                    raise
        
        if self.config.enable_qa_processor:
            try:
                qa_processor = QAProcessor(
                    batch_size=self.config.batch_size,
                    max_retries=3
                )
                self.processors["qa"] = qa_processor
                self.logger.info("Q&A processor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Q&A processor: {e}")
                if not self.config.continue_on_error:
                    raise
    
    async def process_items(self, items: List[CollectedItem]) -> AsyncGenerator[ProcessedChunk, None]:
        """
        Process collected items through appropriate processors.
        
        Args:
            items: List of collected items to process
            
        Yields:
            ProcessedChunk: Processed chunks from all items
        """
        self.logger.info("Starting data processing pipeline", item_count=len(items))
        
        self.status = ProcessingPipelineStatus.RUNNING
        self.metrics.start_time = datetime.utcnow()
        
        try:
            async with async_timer("Complete processing pipeline"):
                
                # Run health checks first
                await self._run_health_checks()
                
                # Route items to appropriate processors
                processor_batches = self._route_items_to_processors(items)
                
                if not processor_batches:
                    self.logger.warning("No items could be routed to processors")
                    return
                
                # Process items with controlled concurrency
                async for chunk in self._process_batches_concurrently(processor_batches):
                    yield chunk
            
            self.status = ProcessingPipelineStatus.COMPLETED
            self.logger.info(
                "Processing pipeline completed successfully",
                total_items=self.metrics.total_items_processed,
                total_chunks=self.metrics.total_chunks_created,
                processing_time=self.metrics.processing_time,
                items_per_second=self.metrics.items_per_second,
                chunks_per_item=self.metrics.chunks_per_item
            )
            
        except Exception as e:
            self.status = ProcessingPipelineStatus.FAILED
            self.metrics.errors.append(f"Pipeline error: {str(e)}")
            self.logger.error(f"Processing pipeline failed: {e}")
            
            if not self.config.continue_on_error:
                raise
        
        finally:
            self.metrics.end_time = datetime.utcnow()
    
    def _route_items_to_processors(self, items: List[CollectedItem]) -> Dict[str, List[CollectedItem]]:
        """Route items to appropriate processors based on source type."""
        
        processor_batches = {}
        
        for item in items:
            source_type = item.source_type
            
            # Route based on source type
            if source_type == "github_code" and "code" in self.processors:
                if "code" not in processor_batches:
                    processor_batches["code"] = []
                processor_batches["code"].append(item)
                
            elif source_type == "stackoverflow_qa" and "qa" in self.processors:
                if "qa" not in processor_batches:
                    processor_batches["qa"] = []
                processor_batches["qa"].append(item)
                
            else:
                self.logger.warning(
                    "No processor available for item",
                    item_id=item.id,
                    source_type=source_type
                )
        
        # Log routing results
        for processor_name, batch_items in processor_batches.items():
            self.logger.info(
                "Items routed to processor",
                processor=processor_name,
                item_count=len(batch_items)
            )
        
        return processor_batches
    
    async def _process_batches_concurrently(self, 
                                          processor_batches: Dict[str, List[CollectedItem]]) -> AsyncGenerator[ProcessedChunk, None]:
        """Process batches concurrently across processors."""
        
        # Create processing tasks
        processing_tasks = []
        
        for processor_name, items in processor_batches.items():
            if processor_name in self.processors:
                processor = self.processors[processor_name]
                task = self._process_with_metrics(processor_name, processor, items)
                processing_tasks.append(task)
        
        if not processing_tasks:
            return
        
        # Use async generators concurrently
        active_iterators = []
        for task in processing_tasks:
            active_iterators.append(task)
        
        # Process results as they become available
        while active_iterators:
            # Create tasks for getting next chunk from each iterator
            next_tasks = []
            for i, iterator in enumerate(active_iterators):
                task = asyncio.create_task(self._get_next_chunk(iterator))
                next_tasks.append((i, task))
            
            if not next_tasks:
                break
            
            # Wait for first completed task
            done, pending = await asyncio.wait(
                [task for _, task in next_tasks],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks (they'll be recreated in next iteration)
            for task in pending:
                task.cancel()
            
            # Process completed tasks
            completed_indices = []
            for i, task in next_tasks:
                if task in done:
                    try:
                        chunk = await task
                        if chunk is not None:
                            yield chunk
                        else:
                            # Iterator exhausted
                            completed_indices.append(i)
                    except Exception as e:
                        self.logger.error(f"Error in processor: {e}")
                        completed_indices.append(i)
            
            # Remove exhausted iterators
            for i in sorted(completed_indices, reverse=True):
                active_iterators.pop(i)
    
    async def _get_next_chunk(self, async_iterator) -> Optional[ProcessedChunk]:
        """Get next chunk from async iterator."""
        try:
            return await async_iterator.__anext__()
        except StopAsyncIteration:
            return None
    
    async def _process_with_metrics(self, 
                                   processor_name: str, 
                                   processor: BaseProcessor,
                                   items: List[CollectedItem]) -> AsyncGenerator[ProcessedChunk, None]:
        """Process items and track metrics."""
        
        try:
            async for chunk in processor.process_items(items):
                # Update pipeline metrics
                self.metrics.total_chunks_created += 1
                
                # Track chunk types
                chunk_type = chunk.chunk_type
                self.metrics.chunks_by_type[chunk_type] = (
                    self.metrics.chunks_by_type.get(chunk_type, 0) + 1
                )
                
                # Track source types
                source_type = chunk.source_type
                self.metrics.items_by_source[source_type] = (
                    self.metrics.items_by_source.get(source_type, 0) + 1
                )
                
                # Save chunk if configured
                if self.config.save_chunks:
                    await self._save_chunk(chunk)
                
                yield chunk
                
        except Exception as e:
            error_msg = f"{processor_name} processor error: {str(e)}"
            self.metrics.errors.append(error_msg)
            self.logger.error(error_msg)
            
            if not self.config.continue_on_error:
                raise
        
        finally:
            # Store processor metrics
            self.metrics.processor_metrics[processor_name] = processor.get_metrics()
            
            # Update items processed count
            processor_metrics = processor.get_metrics()
            self.metrics.total_items_processed += processor_metrics.get('processed_items', 0)
    
    async def _save_chunk(self, chunk: ProcessedChunk) -> None:
        """Save processed chunk to file."""
        try:
            import os
            import json
            import aiofiles
            
            # Create output directory if it doesn't exist
            os.makedirs(self.config.output_directory, exist_ok=True)
            
            # Create filename based on chunk type and date
            date_str = datetime.utcnow().strftime("%Y%m%d")
            filename = f"{chunk.chunk_type}_{date_str}.jsonl"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # Append chunk to file
            async with aiofiles.open(filepath, mode='a', encoding='utf-8') as f:
                json_line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                await f.write(json_line + '\n')
                
        except Exception as e:
            self.logger.warning(f"Failed to save chunk to file: {e}")
    
    async def _run_health_checks(self) -> None:
        """Run health checks on all processors."""
        self.logger.info("Running processor health checks")
        
        health_tasks = []
        for name, processor in self.processors.items():
            health_tasks.append(processor.health_check())
        
        try:
            health_results = await gather_with_concurrency(
                health_tasks, 
                max_concurrency=len(self.processors)
            )
            
            for i, (name, result) in enumerate(zip(self.processors.keys(), health_results)):
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
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current processing pipeline status and metrics."""
        return {
            "status": self.status.value,
            "metrics": {
                "total_items_processed": self.metrics.total_items_processed,
                "total_chunks_created": self.metrics.total_chunks_created,
                "items_by_source": self.metrics.items_by_source,
                "chunks_by_type": self.metrics.chunks_by_type,
                "processing_time": self.metrics.processing_time,
                "items_per_second": self.metrics.items_per_second,
                "chunks_per_item": self.metrics.chunks_per_item,
                "error_count": len(self.metrics.errors),
                "start_time": self.metrics.start_time.isoformat() if self.metrics.start_time else None,
                "end_time": self.metrics.end_time.isoformat() if self.metrics.end_time else None
            },
            "processor_metrics": self.metrics.processor_metrics,
            "config": {
                "code_processor_enabled": self.config.enable_code_processor,
                "qa_processor_enabled": self.config.enable_qa_processor,
                "max_concurrent_processors": self.config.max_concurrent_processors,
                "batch_size": self.config.batch_size
            }
        }
    
    async def stop_pipeline(self) -> None:
        """Stop the processing pipeline gracefully."""
        self.logger.info("Stopping processing pipeline")
        self.status = ProcessingPipelineStatus.CANCELLED
        
        # TODO: Implement graceful shutdown logic
        # This would involve cancelling running tasks and cleaning up resources
    
    def add_processor(self, name: str, processor: BaseProcessor) -> None:
        """Add a custom processor to the pipeline."""
        self.processors[name] = processor
        self.logger.info(f"Added custom processor: {name}")
    
    def remove_processor(self, name: str) -> bool:
        """Remove a processor from the pipeline."""
        if name in self.processors:
            del self.processors[name]
            self.logger.info(f"Removed processor: {name}")
            return True
        return False
    
    def get_processor(self, name: str) -> Optional[BaseProcessor]:
        """Get a processor by name."""
        return self.processors.get(name) 