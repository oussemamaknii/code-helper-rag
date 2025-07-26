"""
Base processor abstract class for data transformation.

This module provides the abstract base class and common functionality
for all data processors in the processing pipeline.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.ingestion.base_collector import CollectedItem
from src.utils.logger import get_logger
from src.utils.async_utils import AsyncRetry, async_timer
from src.config.settings import settings

logger = get_logger(__name__)

T = TypeVar('T')


class ProcessingStatus(Enum):
    """Status of data processing operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingMetrics:
    """Metrics for data processing operations."""
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    chunks_created: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100
    
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
            return self.processed_items / processing_time
        return None
    
    @property
    def chunks_per_item(self) -> float:
        """Calculate average chunks created per item."""
        if self.successful_items == 0:
            return 0.0
        return self.chunks_created / self.successful_items


@dataclass
class ProcessedChunk:
    """Base class for processed data chunks."""
    id: str
    content: str
    chunk_type: str
    metadata: Dict[str, Any]
    source_item_id: str
    source_type: str
    processed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'content': self.content,
            'chunk_type': self.chunk_type,
            'metadata': self.metadata,
            'source_item_id': self.source_item_id,
            'source_type': self.source_type,
            'processed_at': self.processed_at.isoformat()
        }
    
    def __len__(self) -> int:
        """Return content length."""
        return len(self.content)


class ChunkValidator(Protocol):
    """Protocol for validating processed chunks."""
    
    def validate(self, chunk: ProcessedChunk) -> bool:
        """Validate a processed chunk."""
        ...


class BaseProcessor(ABC):
    """
    Abstract base class for all data processors.
    
    Provides common functionality for data processing including:
    - Error handling and retries
    - Progress tracking and metrics
    - Chunk validation
    - Async processing
    """
    
    def __init__(self, 
                 name: str,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 batch_size: int = 10,
                 validator: Optional[ChunkValidator] = None):
        """
        Initialize base processor.
        
        Args:
            name: Processor name for logging
            max_retries: Maximum retry attempts for failed processing
            retry_delay: Base delay between retries in seconds
            batch_size: Size of processing batches
            validator: Optional chunk validator
        """
        self.name = name
        self.batch_size = batch_size
        self.validator = validator
        
        # Set up retry mechanism
        self.retry_decorator = AsyncRetry(
            max_attempts=max_retries,
            base_delay=retry_delay,
            max_delay=60.0
        )
        
        # Initialize metrics
        self.metrics = ProcessingMetrics()
        self.status = ProcessingStatus.PENDING
        
        # Logger with processor context
        self.logger = get_logger(__name__, processor=self.name)
    
    @abstractmethod
    async def process_item(self, item: CollectedItem) -> AsyncGenerator[ProcessedChunk, None]:
        """
        Abstract method to process a collected item into chunks.
        
        Args:
            item: Collected item to process
            
        Yields:
            ProcessedChunk: Processed chunks from the item
        """
        pass
    
    @abstractmethod
    async def get_supported_types(self) -> List[str]:
        """
        Get list of source types this processor can handle.
        
        Returns:
            List of supported source types
        """
        pass
    
    async def can_process(self, item: CollectedItem) -> bool:
        """
        Check if this processor can handle the given item.
        
        Args:
            item: Item to check
            
        Returns:
            True if processor can handle this item
        """
        supported_types = await self.get_supported_types()
        return item.source_type in supported_types
    
    async def process_items(self, items: List[CollectedItem]) -> AsyncGenerator[ProcessedChunk, None]:
        """
        Process multiple items and yield chunks.
        
        Args:
            items: List of collected items to process
            
        Yields:
            ProcessedChunk: Processed chunks from all items
        """
        self.logger.info("Starting batch processing", item_count=len(items))
        
        # Initialize metrics
        self.status = ProcessingStatus.RUNNING
        self.metrics.start_time = datetime.utcnow()
        self.metrics.total_items = len(items)
        
        try:
            async with async_timer(f"{self.name} batch processing"):
                async for chunk in self._process_items_with_handling(items):
                    yield chunk
                    
            self.status = ProcessingStatus.COMPLETED
            self.logger.info(
                "Batch processing completed successfully",
                processed_items=self.metrics.processed_items,
                chunks_created=self.metrics.chunks_created,
                success_rate=self.metrics.success_rate,
                processing_time=self.metrics.processing_time
            )
            
        except Exception as e:
            self.status = ProcessingStatus.FAILED
            self.metrics.errors.append(str(e))
            self.logger.error(
                "Batch processing failed",
                error=str(e),
                processed_items=self.metrics.processed_items
            )
            raise
        finally:
            self.metrics.end_time = datetime.utcnow()
    
    async def _process_items_with_handling(self, items: List[CollectedItem]) -> AsyncGenerator[ProcessedChunk, None]:
        """Internal method that handles processing with error recovery."""
        for item in items:
            try:
                # Check if we can process this item
                if not await self.can_process(item):
                    self.logger.debug(
                        "Skipping unsupported item",
                        item_id=item.id,
                        source_type=item.source_type
                    )
                    continue
                
                # Process item and yield chunks
                item_chunk_count = 0
                async for chunk in self._process_single_item(item):
                    if chunk:
                        yield chunk
                        item_chunk_count += 1
                        self.metrics.chunks_created += 1
                
                # Update metrics
                self.metrics.processed_items += 1
                self.metrics.successful_items += 1
                
                self.logger.debug(
                    "Item processed successfully",
                    item_id=item.id,
                    chunks_created=item_chunk_count,
                    content_length=len(item.content)
                )
                
            except Exception as e:
                self.metrics.processed_items += 1
                self.metrics.failed_items += 1
                self.metrics.errors.append(f"Item {item.id}: {str(e)}")
                self.logger.error(
                    "Failed to process item",
                    item_id=item.id,
                    error=str(e)
                )
                # Continue with next item instead of failing entire batch
                continue
    
    async def _process_single_item(self, item: CollectedItem) -> AsyncGenerator[ProcessedChunk, None]:
        """Process a single item with validation and error handling."""
        try:
            # Process item with retry logic
            @self.retry_decorator
            async def process_with_retry():
                chunks = []
                async for chunk in self.process_item(item):
                    chunks.append(chunk)
                return chunks
            
            chunks = await process_with_retry()
            
            for chunk in chunks:
                # Validate chunk if validator is provided
                if self.validator and not self.validator.validate(chunk):
                    self.logger.warning(
                        "Chunk validation failed",
                        chunk_id=chunk.id,
                        chunk_type=chunk.chunk_type
                    )
                    continue
                
                # Apply any chunk transformations
                processed_chunk = await self._transform_chunk(chunk)
                
                yield processed_chunk
                
        except Exception as e:
            self.logger.error(
                "Error processing single item",
                item_id=item.id,
                error=str(e)
            )
            raise
    
    async def _transform_chunk(self, chunk: ProcessedChunk) -> ProcessedChunk:
        """
        Transform chunk before yielding (can be overridden by subclasses).
        
        Args:
            chunk: Chunk to transform
            
        Returns:
            Transformed chunk
        """
        return chunk
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics."""
        return {
            'processor_name': self.name,
            'status': self.status.value,
            'total_items': self.metrics.total_items,
            'processed_items': self.metrics.processed_items,
            'successful_items': self.metrics.successful_items,
            'failed_items': self.metrics.failed_items,
            'chunks_created': self.metrics.chunks_created,
            'success_rate': self.metrics.success_rate,
            'processing_time': self.metrics.processing_time,
            'items_per_second': self.metrics.items_per_second,
            'chunks_per_item': self.metrics.chunks_per_item,
            'error_count': len(self.metrics.errors),
            'last_errors': self.metrics.errors[-5:] if self.metrics.errors else []
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the processor.
        
        Returns:
            Health status dictionary
        """
        try:
            # Override in subclasses for specific health checks
            await self._perform_health_check()
            
            supported_types = await self.get_supported_types()
            
            return {
                'processor': self.name,
                'status': 'healthy',
                'last_check': datetime.utcnow().isoformat(),
                'supported_types': supported_types,
                'batch_size': self.batch_size
            }
            
        except Exception as e:
            return {
                'processor': self.name,
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def _perform_health_check(self) -> None:
        """Override in subclasses for specific health checks."""
        pass


class DefaultChunkValidator:
    """Default implementation of chunk validator."""
    
    def __init__(self, 
                 min_content_length: int = 10,
                 max_content_length: int = 8000,
                 required_metadata_keys: Optional[List[str]] = None):
        """
        Initialize validator.
        
        Args:
            min_content_length: Minimum chunk content length
            max_content_length: Maximum chunk content length  
            required_metadata_keys: Required metadata keys
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.required_metadata_keys = required_metadata_keys or []
    
    def validate(self, chunk: ProcessedChunk) -> bool:
        """Validate processed chunk."""
        # Check content length
        if not (self.min_content_length <= len(chunk.content) <= self.max_content_length):
            return False
        
        # Check required metadata keys
        for key in self.required_metadata_keys:
            if key not in chunk.metadata:
                return False
        
        # Check for empty or None values
        if not chunk.id or not chunk.content.strip():
            return False
        
        return True 