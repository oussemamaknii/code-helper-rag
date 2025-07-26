"""
Base collector abstract class for data ingestion.

This module provides the abstract base class and common functionality
for all data collectors in the ingestion pipeline.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.utils.logger import get_logger
from src.utils.async_utils import AsyncRetry, AsyncRateLimiter, async_timer
from src.config.settings import settings

logger = get_logger(__name__)

T = TypeVar('T')


class CollectionStatus(Enum):
    """Status of data collection process."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CollectionMetrics:
    """Metrics for data collection process."""
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
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


@dataclass
class CollectedItem:
    """Base class for collected data items."""
    id: str
    content: str
    metadata: Dict[str, Any]
    source_type: str
    collected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'source_type': self.source_type,
            'collected_at': self.collected_at.isoformat()
        }


class ItemValidator(Protocol):
    """Protocol for validating collected items."""
    
    def validate(self, item: CollectedItem) -> bool:
        """Validate a collected item."""
        ...


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    Provides common functionality for data collection including:
    - Rate limiting
    - Error handling and retries
    - Progress tracking
    - Async processing
    """
    
    def __init__(self, 
                 name: str,
                 rate_limit: Optional[int] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 batch_size: int = 100,
                 validator: Optional[ItemValidator] = None):
        """
        Initialize base collector.
        
        Args:
            name: Collector name for logging
            rate_limit: Requests per minute (None for no limit)
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries in seconds
            batch_size: Size of processing batches
            validator: Optional item validator
        """
        self.name = name
        self.batch_size = batch_size
        self.validator = validator
        
        # Set up rate limiter if specified
        self.rate_limiter = (
            AsyncRateLimiter(rate=rate_limit, per=60.0) 
            if rate_limit else None
        )
        
        # Set up retry mechanism
        self.retry_decorator = AsyncRetry(
            max_attempts=max_retries,
            base_delay=retry_delay,
            max_delay=60.0
        )
        
        # Initialize metrics
        self.metrics = CollectionMetrics()
        self.status = CollectionStatus.PENDING
        
        # Logger with collector context
        self.logger = get_logger(__name__, collector=self.name)
    
    @abstractmethod
    async def collect_items(self, **kwargs) -> AsyncGenerator[CollectedItem, None]:
        """
        Abstract method to collect items from data source.
        
        Yields:
            CollectedItem: Individual collected items
        """
        pass
    
    @abstractmethod
    async def get_total_count(self, **kwargs) -> Optional[int]:
        """
        Get total number of items available for collection.
        
        Returns:
            Optional count of total items (None if unknown)
        """
        pass
    
    async def collect_all(self, **kwargs) -> List[CollectedItem]:
        """
        Collect all items and return as list.
        
        Args:
            **kwargs: Arguments passed to collect_items
            
        Returns:
            List of collected items
        """
        items = []
        async for item in self.run_collection(**kwargs):
            items.append(item)
        return items
    
    async def run_collection(self, **kwargs) -> AsyncGenerator[CollectedItem, None]:
        """
        Run the collection process with metrics tracking.
        
        Args:
            **kwargs: Arguments passed to collect_items
            
        Yields:
            CollectedItem: Validated and processed items
        """
        self.logger.info("Starting data collection", **kwargs)
        
        # Initialize metrics
        self.status = CollectionStatus.RUNNING
        self.metrics.start_time = datetime.utcnow()
        self.metrics.total_items = await self.get_total_count(**kwargs) or 0
        
        try:
            async with async_timer(f"{self.name} collection"):
                async for item in self._collect_with_processing(**kwargs):
                    yield item
                    
            self.status = CollectionStatus.COMPLETED
            self.logger.info(
                "Collection completed successfully",
                processed_items=self.metrics.processed_items,
                success_rate=self.metrics.success_rate,
                processing_time=self.metrics.processing_time
            )
            
        except Exception as e:
            self.status = CollectionStatus.FAILED
            self.metrics.errors.append(str(e))
            self.logger.error(
                "Collection failed",
                error=str(e),
                processed_items=self.metrics.processed_items
            )
            raise
        finally:
            self.metrics.end_time = datetime.utcnow()
    
    async def _collect_with_processing(self, **kwargs) -> AsyncGenerator[CollectedItem, None]:
        """Internal method that handles rate limiting and validation."""
        batch = []
        
        async for raw_item in self.collect_items(**kwargs):
            # Apply rate limiting
            if self.rate_limiter:
                async with self.rate_limiter:
                    pass
            
            # Process item
            processed_item = await self._process_item(raw_item)
            if processed_item:
                batch.append(processed_item)
                
                # Yield batch when full
                if len(batch) >= self.batch_size:
                    for item in batch:
                        yield item
                    batch = []
        
        # Yield remaining items
        for item in batch:
            yield item
    
    async def _process_item(self, item: CollectedItem) -> Optional[CollectedItem]:
        """
        Process and validate individual item.
        
        Args:
            item: Raw collected item
            
        Returns:
            Processed item or None if validation fails
        """
        self.metrics.processed_items += 1
        
        try:
            # Validate item if validator is provided
            if self.validator and not self.validator.validate(item):
                self.logger.warning(
                    "Item validation failed",
                    item_id=item.id,
                    source_type=item.source_type
                )
                self.metrics.failed_items += 1
                return None
            
            # Apply any item transformations
            processed_item = await self._transform_item(item)
            
            self.metrics.successful_items += 1
            self.logger.debug(
                "Item processed successfully",
                item_id=item.id,
                content_length=len(item.content)
            )
            
            return processed_item
            
        except Exception as e:
            self.metrics.failed_items += 1
            self.metrics.errors.append(f"Item {item.id}: {str(e)}")
            self.logger.error(
                "Failed to process item",
                item_id=item.id,
                error=str(e)
            )
            return None
    
    async def _transform_item(self, item: CollectedItem) -> CollectedItem:
        """
        Transform item before yielding (can be overridden by subclasses).
        
        Args:
            item: Item to transform
            
        Returns:
            Transformed item
        """
        return item
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current collection metrics."""
        return {
            'collector_name': self.name,
            'status': self.status.value,
            'total_items': self.metrics.total_items,
            'processed_items': self.metrics.processed_items,
            'successful_items': self.metrics.successful_items,
            'failed_items': self.metrics.failed_items,
            'success_rate': self.metrics.success_rate,
            'processing_time': self.metrics.processing_time,
            'items_per_second': self.metrics.items_per_second,
            'error_count': len(self.metrics.errors),
            'last_errors': self.metrics.errors[-5:] if self.metrics.errors else []
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the collector.
        
        Returns:
            Health status dictionary
        """
        try:
            # Override in subclasses for specific health checks
            await self._perform_health_check()
            
            return {
                'collector': self.name,
                'status': 'healthy',
                'last_check': datetime.utcnow().isoformat(),
                'rate_limit_configured': self.rate_limiter is not None,
                'batch_size': self.batch_size
            }
            
        except Exception as e:
            return {
                'collector': self.name,
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def _perform_health_check(self) -> None:
        """Override in subclasses for specific health checks."""
        pass


class DefaultItemValidator:
    """Default implementation of item validator."""
    
    def __init__(self, 
                 min_content_length: int = 10,
                 max_content_length: int = 1000000,
                 required_metadata_keys: Optional[List[str]] = None):
        """
        Initialize validator.
        
        Args:
            min_content_length: Minimum content length
            max_content_length: Maximum content length  
            required_metadata_keys: Required metadata keys
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.required_metadata_keys = required_metadata_keys or []
    
    def validate(self, item: CollectedItem) -> bool:
        """Validate collected item."""
        # Check content length
        if not (self.min_content_length <= len(item.content) <= self.max_content_length):
            return False
        
        # Check required metadata keys
        for key in self.required_metadata_keys:
            if key not in item.metadata:
                return False
        
        # Check for empty or None values
        if not item.id or not item.content.strip():
            return False
        
        return True 