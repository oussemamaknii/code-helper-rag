"""
Authentication and authorization system.

This module provides API key-based authentication, user management,
and authorization for the FastAPI application.
"""

import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.api.models import UserProfile, APIKeyInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Security scheme for API key authentication
security = HTTPBearer(auto_error=False)


class APIKeyManager:
    """
    API key management system.
    
    Features:
    - API key generation and validation
    - Rate limiting per key
    - Usage tracking
    - Key expiration
    """
    
    def __init__(self):
        """Initialize API key manager."""
        # In-memory storage for demo (use database in production)
        self.api_keys: Dict[str, APIKeyInfo] = {}
        self.key_to_user: Dict[str, str] = {}
        
        # Create a default API key for testing
        self._create_default_key()
        
        self.logger = get_logger(__name__, component="api_key_manager")
    
    def _create_default_key(self) -> None:
        """Create a default API key for testing."""
        default_key = "test_key_12345"
        default_user_id = "test_user"
        
        key_info = APIKeyInfo(
            key_id="default_key",
            name="Default Test Key",
            created_at=datetime.utcnow(),
            rate_limit=60,
            is_active=True
        )
        
        self.api_keys[default_key] = key_info
        self.key_to_user[default_key] = default_user_id
        
        self.logger.info("Created default API key for testing", key_id="default_key")
    
    def generate_api_key(self, user_id: str, name: Optional[str] = None, 
                        rate_limit: Optional[int] = None,
                        expires_at: Optional[datetime] = None) -> str:
        """
        Generate a new API key.
        
        Args:
            user_id: User identifier
            name: Human-readable name for the key
            rate_limit: Rate limit for this key (requests per minute)
            expires_at: Expiration timestamp
            
        Returns:
            str: Generated API key
        """
        # Generate secure random key
        api_key = f"pch_{secrets.token_urlsafe(32)}"
        key_id = f"key_{secrets.token_hex(8)}"
        
        # Create key info
        key_info = APIKeyInfo(
            key_id=key_id,
            name=name or f"API Key {datetime.utcnow().strftime('%Y-%m-%d')}",
            created_at=datetime.utcnow(),
            rate_limit=rate_limit,
            is_active=True
        )
        
        # Store key
        self.api_keys[api_key] = key_info
        self.key_to_user[api_key] = user_id
        
        self.logger.info(
            "API key generated",
            key_id=key_id,
            user_id=user_id,
            rate_limit=rate_limit
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[tuple[APIKeyInfo, str]]:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            tuple: (APIKeyInfo, user_id) if valid, None otherwise
        """
        if not api_key or api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        user_id = self.key_to_user[api_key]
        
        # Check if key is active
        if not key_info.is_active:
            self.logger.warning("Inactive API key used", key_id=key_info.key_id)
            return None
        
        # Check expiration (if set)
        if hasattr(key_info, 'expires_at') and key_info.expires_at:
            if datetime.utcnow() > key_info.expires_at:
                self.logger.warning("Expired API key used", key_id=key_info.key_id)
                return None
        
        # Update usage
        key_info.last_used = datetime.utcnow()
        key_info.usage_count += 1
        
        return key_info, user_id
    
    def deactivate_api_key(self, api_key: str) -> bool:
        """
        Deactivate an API key.
        
        Args:
            api_key: API key to deactivate
            
        Returns:
            bool: True if successful, False otherwise
        """
        if api_key in self.api_keys:
            self.api_keys[api_key].is_active = False
            self.logger.info(
                "API key deactivated",
                key_id=self.api_keys[api_key].key_id
            )
            return True
        return False
    
    def get_user_keys(self, user_id: str) -> list[tuple[str, APIKeyInfo]]:
        """
        Get all API keys for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            list: List of (api_key, APIKeyInfo) tuples
        """
        user_keys = []
        for api_key, stored_user_id in self.key_to_user.items():
            if stored_user_id == user_id:
                user_keys.append((api_key, self.api_keys[api_key]))
        
        return user_keys


class UserManager:
    """
    User management system.
    
    Features:
    - User profile management
    - Usage tracking
    - Preferences storage
    """
    
    def __init__(self):
        """Initialize user manager."""
        # In-memory storage for demo (use database in production)
        self.users: Dict[str, UserProfile] = {}
        
        # Create a default test user
        self._create_default_user()
        
        self.logger = get_logger(__name__, component="user_manager")
    
    def _create_default_user(self) -> None:
        """Create a default test user."""
        default_user = UserProfile(
            user_id="test_user",
            email="test@example.com",
            name="Test User",
            created_at=datetime.utcnow(),
            usage_stats={
                "total_requests": 0,
                "total_tokens": 0,
                "avg_response_time": 0.0
            },
            preferences={
                "default_language": "python",
                "difficulty_level": "intermediate",
                "include_examples": True
            }
        )
        
        self.users["test_user"] = default_user
        self.logger.info("Created default test user", user_id="test_user")
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserProfile: User profile if found, None otherwise
        """
        return self.users.get(user_id)
    
    def create_user(self, user_id: str, email: Optional[str] = None, 
                   name: Optional[str] = None) -> UserProfile:
        """
        Create a new user.
        
        Args:
            user_id: Unique user identifier
            email: User email address
            name: User display name
            
        Returns:
            UserProfile: Created user profile
        """
        user = UserProfile(
            user_id=user_id,
            email=email,
            name=name or user_id,
            created_at=datetime.utcnow(),
            usage_stats={},
            preferences={}
        )
        
        self.users[user_id] = user
        
        self.logger.info(
            "User created",
            user_id=user_id,
            email=email
        )
        
        return user
    
    def update_user_activity(self, user_id: str, 
                           tokens_used: int = 0,
                           response_time: float = 0.0) -> None:
        """
        Update user activity statistics.
        
        Args:
            user_id: User identifier
            tokens_used: Number of tokens used in request
            response_time: Response time for request
        """
        if user_id in self.users:
            user = self.users[user_id]
            user.last_active = datetime.utcnow()
            
            # Update usage stats
            stats = user.usage_stats
            stats["total_requests"] = stats.get("total_requests", 0) + 1
            stats["total_tokens"] = stats.get("total_tokens", 0) + tokens_used
            
            # Update average response time
            total_requests = stats["total_requests"]
            current_avg = stats.get("avg_response_time", 0.0)
            stats["avg_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )


# Global instances
api_key_manager = APIKeyManager()
user_manager = UserManager()


# Authentication dependencies

async def get_api_key_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """
    Extract API key from Authorization header (optional).
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        str: API key if present, None otherwise
    """
    if credentials is None:
        return None
    
    if credentials.scheme.lower() != "bearer":
        return None
    
    return credentials.credentials


async def get_api_key_required(api_key: Optional[str] = Depends(get_api_key_optional)) -> str:
    """
    Extract API key from Authorization header (required).
    
    Args:
        api_key: API key from optional dependency
        
    Returns:
        str: Valid API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return api_key


async def validate_api_key(api_key: str = Depends(get_api_key_required)) -> tuple[APIKeyInfo, str]:
    """
    Validate API key and return key info and user ID.
    
    Args:
        api_key: API key to validate
        
    Returns:
        tuple: (APIKeyInfo, user_id)
        
    Raises:
        HTTPException: If API key is invalid
    """
    result = api_key_manager.validate_api_key(api_key)
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return result


async def get_current_user(validation_result: tuple[APIKeyInfo, str] = Depends(validate_api_key)) -> UserProfile:
    """
    Get current user from validated API key.
    
    Args:
        validation_result: Result from API key validation
        
    Returns:
        UserProfile: Current user profile
        
    Raises:
        HTTPException: If user not found
    """
    key_info, user_id = validation_result
    
    user = user_manager.get_user(user_id)
    if user is None:
        # Create user if not exists (for development)
        user = user_manager.create_user(user_id)
    
    return user


async def get_current_user_optional(api_key: Optional[str] = Depends(get_api_key_optional)) -> Optional[UserProfile]:
    """
    Get current user from API key (optional).
    
    Args:
        api_key: API key (optional)
        
    Returns:
        UserProfile: Current user profile if authenticated, None otherwise
    """
    if not api_key:
        return None
    
    result = api_key_manager.validate_api_key(api_key)
    if result is None:
        return None
    
    key_info, user_id = result
    return user_manager.get_user(user_id)


# Convenience functions

def create_api_key(user_id: str, name: Optional[str] = None, 
                  rate_limit: Optional[int] = None) -> str:
    """
    Create a new API key for a user.
    
    Args:
        user_id: User identifier
        name: Human-readable key name
        rate_limit: Rate limit for this key
        
    Returns:
        str: Generated API key
    """
    return api_key_manager.generate_api_key(user_id, name, rate_limit)


def require_api_key(api_key: str = Depends(get_api_key_required)) -> str:
    """
    Dependency that requires a valid API key.
    
    Args:
        api_key: API key from request
        
    Returns:
        str: Validated API key
    """
    return api_key


class APIKeyAuth:
    """API key authentication class for use with FastAPI dependencies."""
    
    def __init__(self, required: bool = True):
        """
        Initialize API key authentication.
        
        Args:
            required: Whether API key is required
        """
        self.required = required
    
    async def __call__(self, api_key: Optional[str] = Depends(get_api_key_optional)) -> Optional[str]:
        """
        Authenticate request with API key.
        
        Args:
            api_key: API key from request
            
        Returns:
            str: Validated API key
            
        Raises:
            HTTPException: If API key is required but missing/invalid
        """
        if self.required and not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if api_key:
            result = api_key_manager.validate_api_key(api_key)
            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        return api_key


class UserAuth:
    """User authentication class for use with FastAPI dependencies."""
    
    def __init__(self, required: bool = True):
        """
        Initialize user authentication.
        
        Args:
            required: Whether authentication is required
        """
        self.required = required
    
    async def __call__(self, 
                      api_key: Optional[str] = Depends(get_api_key_optional)) -> Optional[UserProfile]:
        """
        Authenticate user from API key.
        
        Args:
            api_key: API key from request
            
        Returns:
            UserProfile: Authenticated user profile
            
        Raises:
            HTTPException: If authentication is required but fails
        """
        if not api_key:
            if self.required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            return None
        
        result = api_key_manager.validate_api_key(api_key)
        if result is None:
            if self.required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            return None
        
        key_info, user_id = result
        user = user_manager.get_user(user_id)
        
        if user is None and self.required:
            # Create user if not exists
            user = user_manager.create_user(user_id)
        
        return user 