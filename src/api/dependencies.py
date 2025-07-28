"""
Dependency injection for FastAPI application.

This module provides dependency functions for injecting components
like generation pipeline, search engine, and other services.
"""

from typing import Optional, Any
from fastapi import Depends, HTTPException, status

from src.api.app import get_generation_pipeline as _get_generation_pipeline
from src.api.app import get_search_engine as _get_search_engine
from src.api.auth import get_current_user_optional, api_key_manager, user_manager
from src.api.models import UserProfile
from src.generation.pipeline import GenerationPipeline
from src.vector.similarity_search import SimilaritySearchEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Component Dependencies

async def get_generation_pipeline() -> Optional[GenerationPipeline]:
    """
    Get the generation pipeline instance.
    
    Returns:
        GenerationPipeline: Pipeline instance if available, None otherwise
    """
    pipeline = _get_generation_pipeline()
    if pipeline is None:
        logger.warning("Generation pipeline not available")
    return pipeline


async def get_search_engine() -> Optional[SimilaritySearchEngine]:
    """
    Get the search engine instance.
    
    Returns:
        SimilaritySearchEngine: Search engine instance if available, None otherwise
    """
    search_engine = _get_search_engine()
    if search_engine is None:
        logger.warning("Search engine not available")
    return search_engine


async def get_generation_pipeline_required() -> GenerationPipeline:
    """
    Get the generation pipeline instance (required).
    
    Returns:
        GenerationPipeline: Pipeline instance
        
    Raises:
        HTTPException: If pipeline is not available
    """
    pipeline = await get_generation_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation pipeline is not available. Please try again later."
        )
    return pipeline


async def get_search_engine_required() -> SimilaritySearchEngine:
    """
    Get the search engine instance (required).
    
    Returns:
        SimilaritySearchEngine: Search engine instance
        
    Raises:
        HTTPException: If search engine is not available
    """
    search_engine = await get_search_engine()
    if search_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search engine is not available. Please try again later."
        )
    return search_engine


# Authentication Dependencies

async def get_current_user(user: Optional[UserProfile] = Depends(get_current_user_optional)) -> Optional[UserProfile]:
    """
    Get current authenticated user.
    
    Returns:
        UserProfile: Current user if authenticated, None otherwise
    """
    return user


async def get_current_user_profile(user: Optional[UserProfile] = Depends(get_current_user)) -> Optional[UserProfile]:
    """
    Get current user profile with usage tracking.
    
    Returns:
        UserProfile: Current user profile
    """
    if user:
        # Update last activity
        user_manager.update_user_activity(user.user_id)
    
    return user


# Rate Limiting Dependencies

class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Number of requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.logger = get_logger(__name__, component="rate_limiter")
    
    async def check_rate_limit(self, user_id: str) -> None:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Rate limiting is handled by middleware
        # This is a placeholder for endpoint-specific rate limiting
        pass


async def get_rate_limiter() -> RateLimiter:
    """
    Get rate limiter instance.
    
    Returns:
        RateLimiter: Rate limiter instance
    """
    return RateLimiter()


# Request Context Dependencies

class RequestContext:
    """Request context information."""
    
    def __init__(self, 
                 user: Optional[UserProfile] = None,
                 request_id: Optional[str] = None,
                 client_ip: Optional[str] = None):
        """
        Initialize request context.
        
        Args:
            user: Current user
            request_id: Request identifier
            client_ip: Client IP address
        """
        self.user = user
        self.request_id = request_id
        self.client_ip = client_ip
        self.start_time = None
        self.metadata = {}


async def get_request_context(
    user: Optional[UserProfile] = Depends(get_current_user)
) -> RequestContext:
    """
    Get request context with user information.
    
    Args:
        user: Current user from dependency injection
        
    Returns:
        RequestContext: Request context object
    """
    return RequestContext(user=user)


# Validation Dependencies

class QueryValidator:
    """Validator for query parameters and request data."""
    
    def __init__(self, max_query_length: int = 10000):
        """
        Initialize query validator.
        
        Args:
            max_query_length: Maximum allowed query length
        """
        self.max_query_length = max_query_length
        self.logger = get_logger(__name__, component="validator")
    
    def validate_query(self, query: str) -> str:
        """
        Validate and sanitize query string.
        
        Args:
            query: Query string to validate
            
        Returns:
            str: Validated query string
            
        Raises:
            HTTPException: If query is invalid
        """
        if not query or not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        query = query.strip()
        
        if len(query) > self.max_query_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query too long. Maximum length is {self.max_query_length} characters."
            )
        
        # Basic sanitization
        suspicious_patterns = ['<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
        query_lower = query.lower()
        
        for pattern in suspicious_patterns:
            if pattern in query_lower:
                self.logger.warning(
                    "Suspicious query detected",
                    query=query[:100],
                    pattern=pattern
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid query content"
                )
        
        return query


async def get_query_validator() -> QueryValidator:
    """
    Get query validator instance.
    
    Returns:
        QueryValidator: Validator instance
    """
    return QueryValidator()


# Service Health Dependencies

class ServiceHealthChecker:
    """Service health checker for dependencies."""
    
    def __init__(self):
        """Initialize service health checker."""
        self.logger = get_logger(__name__, component="health_checker")
    
    async def check_generation_pipeline(self, 
                                      pipeline: Optional[GenerationPipeline] = Depends(get_generation_pipeline)
                                      ) -> bool:
        """
        Check if generation pipeline is healthy.
        
        Args:
            pipeline: Generation pipeline instance
            
        Returns:
            bool: True if healthy, False otherwise
        """
        if pipeline is None:
            return False
        
        try:
            health_status = await pipeline.get_health_status()
            return health_status.get('pipeline', {}).get('status') == 'healthy'
        except Exception as e:
            self.logger.error(f"Pipeline health check failed: {e}")
            return False
    
    async def check_search_engine(self, 
                                search_engine: Optional[SimilaritySearchEngine] = Depends(get_search_engine)
                                ) -> bool:
        """
        Check if search engine is healthy.
        
        Args:
            search_engine: Search engine instance
            
        Returns:
            bool: True if healthy, False otherwise
        """
        if search_engine is None:
            return False
        
        try:
            health_status = await search_engine.health_check()
            return health_status.get('status') == 'healthy'
        except Exception as e:
            self.logger.error(f"Search engine health check failed: {e}")
            return False


async def get_service_health_checker() -> ServiceHealthChecker:
    """
    Get service health checker instance.
    
    Returns:
        ServiceHealthChecker: Health checker instance
    """
    return ServiceHealthChecker()


# Caching Dependencies

class CacheManager:
    """Cache manager for request caching."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000
        self.logger = get_logger(__name__, component="cache_manager")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value if found and not expired, None otherwise
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            # Check if expired
            import time
            if time.time() - timestamp < self.cache_ttl:
                return value
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        import time
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            # Remove 10% of oldest entries
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])[:int(self.max_cache_size * 0.1)]
            for old_key in oldest_keys:
                del self.cache[old_key]
        
        self.cache[key] = (value, time.time())
    
    def generate_key(self, prefix: str, **kwargs) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            prefix: Key prefix
            **kwargs: Key parameters
            
        Returns:
            str: Generated cache key
        """
        import hashlib
        import json
        
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(kwargs, sort_keys=True, default=str)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:8]
        
        return f"{prefix}:{param_hash}"


async def get_cache_manager() -> CacheManager:
    """
    Get cache manager instance.
    
    Returns:
        CacheManager: Cache manager instance
    """
    return CacheManager()


# Analytics Dependencies

class AnalyticsTracker:
    """Analytics tracker for request metrics."""
    
    def __init__(self):
        """Initialize analytics tracker."""
        self.events = []
        self.max_events = 10000
        self.logger = get_logger(__name__, component="analytics")
    
    def track_event(self, event_type: str, user_id: Optional[str] = None, 
                   metadata: Optional[dict] = None) -> None:
        """
        Track an analytics event.
        
        Args:
            event_type: Type of event
            user_id: User identifier
            metadata: Additional event metadata
        """
        import time
        
        event = {
            'type': event_type,
            'user_id': user_id,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        self.logger.debug(
            "Event tracked",
            event_type=event_type,
            user_id=user_id
        )
    
    def get_stats(self) -> dict:
        """
        Get analytics statistics.
        
        Returns:
            dict: Analytics statistics
        """
        if not self.events:
            return {"total_events": 0}
        
        # Basic statistics
        total_events = len(self.events)
        event_types = {}
        unique_users = set()
        
        for event in self.events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if event['user_id']:
                unique_users.add(event['user_id'])
        
        return {
            'total_events': total_events,
            'event_types': event_types,
            'unique_users': len(unique_users),
            'latest_event': self.events[-1]['timestamp'] if self.events else None
        }


async def get_analytics_tracker() -> AnalyticsTracker:
    """
    Get analytics tracker instance.
    
    Returns:
        AnalyticsTracker: Analytics tracker instance
    """
    return AnalyticsTracker() 