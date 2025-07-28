"""
Exception handlers for FastAPI application.

This module provides custom exception handlers for consistent error responses
and proper HTTP status codes.
"""

import traceback
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from pydantic import ValidationError

from src.api.models import ErrorResponse, ErrorDetail
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Set up exception handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions with consistent error format."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        # Log the exception
        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=str(exc.detail),
            path=request.url.path,
            method=request.method,
            request_id=request_id
        )
        
        # Map HTTP status codes to error codes
        error_code_mapping = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED", 
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            408: "REQUEST_TIMEOUT",
            409: "CONFLICT",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "UNPROCESSABLE_ENTITY",
            429: "RATE_LIMIT_EXCEEDED",
            500: "INTERNAL_SERVER_ERROR",
            502: "BAD_GATEWAY",
            503: "SERVICE_UNAVAILABLE",
            504: "GATEWAY_TIMEOUT"
        }
        
        error_code = error_code_mapping.get(exc.status_code, "HTTP_ERROR")
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code=error_code,
                message=str(exc.detail),
                details={"status_code": exc.status_code}
            ),
            request_id=request_id
        )
        
        # Add rate limit headers if applicable
        headers = {}
        if exc.status_code == 429 and hasattr(exc, 'headers') and exc.headers:
            headers.update(exc.headers)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict(),
            headers=headers
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request validation errors."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        # Extract validation error details
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        logger.warning(
            "Request validation error",
            path=request.url.path,
            method=request.method,
            errors=error_details,
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                details={
                    "errors": error_details,
                    "error_count": len(error_details)
                }
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.dict()
        )
    
    @app.exception_handler(ResponseValidationError)
    async def response_validation_exception_handler(request: Request, exc: ResponseValidationError) -> JSONResponse:
        """Handle response validation errors (internal server errors)."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        logger.error(
            "Response validation error",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="Internal server error occurred",
                details={"type": "response_validation_error"}
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict()
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle Pydantic validation errors."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        # Extract validation error details
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        logger.warning(
            "Pydantic validation error",
            path=request.url.path,
            method=request.method,
            errors=error_details,
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Data validation failed",
                details={"errors": error_details}
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.dict()
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle value errors."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        logger.warning(
            "Value error",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INVALID_VALUE",
                message="Invalid value provided",
                details={"original_error": str(exc)}
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.dict()
        )
    
    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        """Handle key errors."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        logger.warning(
            "Key error",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="MISSING_FIELD",
                message=f"Required field missing: {str(exc)}",
                details={"missing_key": str(exc)}
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.dict()
        )
    
    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError) -> JSONResponse:
        """Handle timeout errors."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        logger.error(
            "Timeout error",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="REQUEST_TIMEOUT",
                message="Request timed out",
                details={"timeout_error": str(exc)}
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            content=error_response.dict()
        )
    
    @app.exception_handler(ConnectionError)
    async def connection_error_handler(request: Request, exc: ConnectionError) -> JSONResponse:
        """Handle connection errors."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        logger.error(
            "Connection error",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="SERVICE_UNAVAILABLE",
                message="External service unavailable",
                details={"connection_error": str(exc)}
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other exceptions."""
        
        request_id = getattr(request.state, 'request_id', None)
        
        # Log the full traceback for debugging
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred",
                details={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                }
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict()
        )


# Custom Exception Classes

class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception."""
    
    def __init__(self, message: str = "Rate limit exceeded", 
                 retry_after: int = 60,
                 current_rate: int = 0,
                 limit: int = 60):
        """
        Initialize rate limit exception.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            current_rate: Current request rate
            limit: Rate limit
        """
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after)
            }
        )
        
        self.retry_after = retry_after
        self.current_rate = current_rate
        self.limit = limit


class ServiceUnavailable(HTTPException):
    """Service unavailable exception."""
    
    def __init__(self, service_name: str, message: str = None):
        """
        Initialize service unavailable exception.
        
        Args:
            service_name: Name of the unavailable service
            message: Custom error message
        """
        if message is None:
            message = f"{service_name} is currently unavailable"
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=message
        )
        
        self.service_name = service_name


class InvalidAPIKey(HTTPException):
    """Invalid API key exception."""
    
    def __init__(self, message: str = "Invalid API key"):
        """
        Initialize invalid API key exception.
        
        Args:
            message: Error message
        """
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
            headers={"WWW-Authenticate": "Bearer"}
        )


class QuotaExceeded(HTTPException):
    """Quota exceeded exception."""
    
    def __init__(self, quota_type: str, current: int, limit: int):
        """
        Initialize quota exceeded exception.
        
        Args:
            quota_type: Type of quota (tokens, requests, etc.)
            current: Current usage
            limit: Usage limit
        """
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"{quota_type.title()} quota exceeded. Current: {current}, Limit: {limit}"
        )
        
        self.quota_type = quota_type
        self.current = current
        self.limit = limit


class ValidationException(HTTPException):
    """Custom validation exception."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        """
        Initialize validation exception.
        
        Args:
            field: Field that failed validation
            message: Validation error message
            value: Invalid value
        """
        detail = f"Validation error for field '{field}': {message}"
        if value is not None:
            detail += f" (value: {value})"
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )
        
        self.field = field
        self.message = message
        self.value = value 