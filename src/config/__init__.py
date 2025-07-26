"""
Configuration management module.

This module provides centralized configuration management using Pydantic Settings
with support for environment variables, type validation, and development/production
environment handling.
"""

from .settings import Settings, settings

__all__ = ["Settings", "settings"] 