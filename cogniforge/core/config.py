"""
Configuration management for CogniForge.

This module handles all application configuration including
environment variables and API keys.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any
from functools import lru_cache
import warnings

from dotenv import load_dotenv


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        """Initialize settings and load environment variables."""
        # Find the project root directory (where .env file should be)
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        
        # Load .env file if it exists
        env_file = self.base_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✓ Loaded environment from: {env_file}")
        else:
            # Try to load .env.example as fallback (for development)
            env_example = self.base_dir / ".env.example"
            if env_example.exists():
                load_dotenv(env_example)
                warnings.warn(
                    f"Using .env.example file. Please create a .env file for production use.",
                    UserWarning
                )
        
        # API Configuration
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.api_reload: bool = os.getenv("API_RELOAD", "true").lower() == "true"
        
        # Application Settings
        self.app_name: str = os.getenv("APP_NAME", "CogniForge")
        self.app_version: str = os.getenv("APP_VERSION", "0.1.0")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        
        # OpenAI Configuration
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key and self.openai_api_key.startswith("sk-proj-XXX"):
            warnings.warn(
                "OpenAI API key appears to be a placeholder. Please set a valid API key in .env file.",
                UserWarning
            )
            self.openai_api_key = None
        
        # PyBullet Configuration
        self.pybullet_gui: bool = os.getenv("PYBULLET_GUI", "false").lower() == "true"
        self.pybullet_options: str = os.getenv("PYBULLET_OPTIONS", "")
        
        # ML/AI Settings
        self.torch_device: str = os.getenv("TORCH_DEVICE", "cpu")
        self.model_path: Path = Path(os.getenv("MODEL_PATH", "./models"))
        self.data_path: Path = Path(os.getenv("DATA_PATH", "./data"))
        
        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_file: Optional[str] = os.getenv("LOG_FILE")
        
        # Database (optional)
        self.database_url: Optional[str] = os.getenv("DATABASE_URL")
        
        # External APIs
        self.external_api_url: Optional[str] = os.getenv("EXTERNAL_API_URL")
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [self.model_path, self.data_path]
        
        if self.log_file:
            log_dir = Path(self.log_file).parent
            directories.append(log_dir)
        
        for directory in directories:
            directory = Path(directory)
            if not directory.is_absolute():
                directory = self.base_dir / directory
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to settings."""
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        """Check if a setting exists."""
        return hasattr(self, key)
    
    def validate_openai_config(self) -> bool:
        """
        Validate OpenAI configuration.
        
        Returns:
            bool: True if OpenAI is properly configured, False otherwise.
        """
        if not self.openai_api_key:
            return False
        
        # Check if it's a valid format (basic check)
        if not self.openai_api_key.startswith("sk-"):
            warnings.warn("OpenAI API key format appears invalid.", UserWarning)
            return False
        
        return True
    
    def get_openai_client(self):
        """
        Get an initialized OpenAI client.
        
        Returns:
            OpenAI client instance or None if not configured.
        """
        if not self.validate_openai_config():
            raise ValueError(
                "OpenAI API key is not configured. "
                "Please set OPENAI_API_KEY in your .env file."
            )
        
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.openai_api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package is not installed. "
                "Install it with: pip install openai"
            )
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """
        Convert settings to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive data like API keys.
        
        Returns:
            Dictionary of settings.
        """
        result = {}
        
        for key in dir(self):
            if key.startswith("_"):
                continue
            
            value = getattr(self, key)
            
            # Skip methods
            if callable(value):
                continue
            
            # Mask sensitive data unless explicitly requested
            if not include_sensitive and "api_key" in key.lower() and value:
                value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            
            result[key] = value
        
        return result
    
    def __repr__(self) -> str:
        """String representation of settings."""
        return f"<Settings: {self.app_name} v{self.app_version}>"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        items = []
        for key, value in self.to_dict().items():
            items.append(f"  {key}: {value}")
        
        return f"Settings:\n" + "\n".join(items)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings instance (cached).
    """
    return Settings()


# Create a global settings instance
settings = get_settings()


# Utility functions for common operations
def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from settings.
    
    Returns:
        OpenAI API key or None if not configured.
    """
    return settings.openai_api_key if settings.validate_openai_config() else None


def is_debug_mode() -> bool:
    """Check if application is in debug mode."""
    return settings.debug


def get_app_info() -> dict:
    """Get application information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
    }


# Example usage and validation on import
if __name__ == "__main__":
    # Print current settings
    print(settings)
    
    # Validate OpenAI configuration
    if settings.validate_openai_config():
        print("\n✓ OpenAI API key is configured")
    else:
        print("\n⚠ OpenAI API key is not configured or invalid")
    
    # Show app info
    print(f"\nApp Info: {get_app_info()}")