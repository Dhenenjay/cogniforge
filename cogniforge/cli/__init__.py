"""
CogniForge Command-Line Interface Module

This module provides CLI tools with support for deterministic mode.
"""

from .train_cli import create_parser, setup_reproducibility

__all__ = [
    "create_parser",
    "setup_reproducibility"
]

__version__ = "1.0.0"