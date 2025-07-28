"""
Custom middleware for FastAPI application.

This module provides middleware for rate limiting, logging, request tracking,
and other cross-cutting concerns.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with sliding window algorithm.
    
    Features:
    - Per-user rate limiting based on API key or IP
    - Configurable rate limits per endpoint
    - Sliding window implementation
    - Automatic cleanup of old entries
    """
    
    def __init__(self, app, default_rate_limit: int = 60):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            default_rate_limit: Default requests per minute
        """
        super().__init__(app)
        self.default_rate_limit = default_rate_limit
        
        # Rate limit storage: {user_id: deque of timestamps}
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Endpoint-specific rate limits
        self.endpoint_limits = {
            "/api/v1/chat": 30,           # 30 requests per minute for chat
            "/api/v1/chat/stream": 10,    # 10 requests per minute for streaming
            "/api/v1/search": 100,        # 100 requests per minute for search
            "/api/v1/health": 300         # 300 requests per minute for health checks
        }
        
        # Last cleanup time
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
        self.logger = get_logger(__name__, component="rate_limiter")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with rate limiting."""
        
        # Extract user identifier
        user_id = self._get_user_identifier(request)
        
        # Get rate limit for this endpoint
        endpoint = request.url.path
        rate_limit = self.endpoint_limits.get(endpoint, self.default_rate_limit)
        
        # Check rate limit
        current_time = time.time()
        
        try:
            await self._check_rate_limit(user_id, rate_limit, current_time)
        except HTTPException as e:
            # Log rate limit violation
            self.logger.warning(
                "Rate limit exceeded",
                user_id=user_id,
                endpoint=endpoint,
                rate_limit=rate_limit
            )
            
            # Return rate limit error response
            return Response(
                content=f'{{"error": "{e.detail}", "retry_after": 60}}',
                status_code=e.status_code,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 60)),
                    "Retry-After": "60"
                }
            )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # Add rate limit headers to response
        remaining = await self._get_remaining_requests(user_id, rate_limit, current_time)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        response.headers["X-Response-Time"] = f"{processing_time:.3f}s"
        
        # Periodic cleanup
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        return response
    
    def _get_user_identifier(self, request: Request) -> str:
        """Get user identifier from request."""
        # Try to get user ID from authentication
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return user_id
        
        # Try to get API key
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:]  # Remove 'Bearer ' prefix
            return f"api_key_{api_key[:8]}"  # Use first 8 chars for identification
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        
        return f"ip_{client_ip}"
    
    async def _check_rate_limit(self, user_id: str, rate_limit: int, current_time: float) -> None:
        """Check if request exceeds rate limit."""
        user_requests = self.request_counts[user_id]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        while user_requests and user_requests[0] < cutoff_time:
            user_requests.popleft()
        
        # Check if rate limit exceeded
        if len(user_requests) >= rate_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        user_requests.append(current_time)
    
    async def _get_remaining_requests(self, user_id: str, rate_limit: int, current_time: float) -> int:
        """Get remaining requests for user."""
        user_requests = self.request_counts[user_id]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        while user_requests and user_requests[0] < cutoff_time:
            user_requests.popleft()
        
        return max(0, rate_limit - len(user_requests))
    
    async def _cleanup_old_entries(self, current_time: float) -> None:
        """Clean up old rate limit entries."""
        cutoff_time = current_time - 120  # Keep last 2 minutes
        
        users_to_remove = []
        for user_id, requests in self.request_counts.items():
            # Remove old requests
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Mark empty queues for removal
            if not requests:
                users_to_remove.append(user_id)
        
        # Remove empty entries
        for user_id in users_to_remove:
            del self.request_counts[user_id]
        
        self.logger.debug(
            "Rate limit cleanup completed",
            active_users=len(self.request_counts),
            removed_users=len(users_to_remove)
        )


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for request/response tracking.
    
    Features:
    - Request/response logging with timing
    - Error logging with context
    - User activity tracking
    - Performance monitoring
    """
    
    def __init__(self, app):
        """Initialize logging middleware."""
        super().__init__(app)
        self.logger = get_logger(__name__, component="request_logger")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with logging."""
        
        # Generate request ID
        request_id = f"req_{int(time.time() * 1000000)}"
        request.state.request_id = request_id
        
        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get('User-Agent', 'unknown')
        
        # Get user identifier
        user_id = getattr(request.state, 'user_id', 'anonymous')
        
        start_time = time.time()
        
        # Log request
        self.logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=client_ip,
            user_agent=user_agent[:100],  # Truncate long user agents
            user_id=user_id
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            processing_time = time.time() - start_time
            
            # Log successful response
            self.logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                processing_time=processing_time,
                response_size=response.headers.get('content-length', 'unknown')
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log error
            self.logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                processing_time=processing_time,
                error_type=type(e).__name__
            )
            
            # Re-raise the exception
            raise


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for headers and basic security measures.
    
    Features:
    - Security headers injection
    - Request validation
    - Basic attack prevention
    """
    
    def __init__(self, app):
        """Initialize security middleware."""
        super().__init__(app)
        self.logger = get_logger(__name__, component="security")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with security measures."""
        
        # Basic request validation
        await self._validate_request(request)
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    async def _validate_request(self, request: Request) -> None:
        """Validate incoming request for basic security."""
        
        # Check request method
        if request.method not in ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD']:
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail="Method not allowed"
            )
        
        # Check content length (prevent large payloads)
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                length = int(content_length)
                if length > 10 * 1024 * 1024:  # 10MB limit
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Request entity too large"
                    )
            except ValueError:
                pass  # Invalid content-length header, let FastAPI handle it
        
        # Basic path validation
        path = request.url.path
        suspicious_patterns = ['..', '<script', 'javascript:', 'vbscript:']
        if any(pattern in path.lower() for pattern in suspicious_patterns):
            self.logger.warning(
                "Suspicious request detected",
                path=path,
                client_ip=request.client.host if request.client else "unknown"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request"
            )
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'"
        )
        
        # HSTS header for HTTPS
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Performance monitoring middleware.
    
    Features:
    - Response time tracking
    - Performance metrics collection
    - Slow request detection
    """
    
    def __init__(self, app, slow_request_threshold: float = 5.0):
        """
        Initialize performance middleware.
        
        Args:
            app: FastAPI application
            slow_request_threshold: Threshold for slow request warning (seconds)
        """
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.logger = get_logger(__name__, component="performance")
        
        # Performance metrics
        self.request_times: deque = deque(maxlen=1000)  # Keep last 1000 requests
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0
        })
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with performance monitoring."""
        
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        # Process request
        response = await call_next(request)
        
        processing_time = time.time() - start_time
        
        # Update metrics
        self.request_times.append(processing_time)
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['total_time'] += processing_time
        stats['min_time'] = min(stats['min_time'], processing_time)
        stats['max_time'] = max(stats['max_time'], processing_time)
        
        # Log slow requests
        if processing_time > self.slow_request_threshold:
            self.logger.warning(
                "Slow request detected",
                endpoint=endpoint,
                processing_time=processing_time,
                threshold=self.slow_request_threshold
            )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{processing_time:.3f}s"
        
        return response
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.request_times:
            return {"message": "No requests processed yet"}
        
        # Calculate overall stats
        times = list(self.request_times)
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate percentiles
        sorted_times = sorted(times)
        n = len(sorted_times)
        p50 = sorted_times[int(n * 0.5)]
        p95 = sorted_times[int(n * 0.95)]
        p99 = sorted_times[int(n * 0.99)]
        
        # Endpoint stats
        endpoint_summary = {}
        for endpoint, stats in self.endpoint_stats.items():
            if stats['count'] > 0:
                endpoint_summary[endpoint] = {
                    'count': stats['count'],
                    'avg_time': stats['total_time'] / stats['count'],
                    'min_time': stats['min_time'],
                    'max_time': stats['max_time']
                }
        
        return {
            'overall': {
                'total_requests': len(times),
                'avg_response_time': avg_time,
                'min_response_time': min_time,
                'max_response_time': max_time,
                'p50_response_time': p50,
                'p95_response_time': p95,
                'p99_response_time': p99
            },
            'endpoints': endpoint_summary,
            'slow_requests': sum(1 for t in times if t > self.slow_request_threshold)
        } 