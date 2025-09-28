"""
Deterministic Mode Configuration for PyTorch

This module provides utilities for enabling full deterministic behavior in PyTorch,
including CPU operations, with support for command-line --deterministic flag.
"""

import os
import warnings
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import torch
import numpy as np
import random

logger = logging.getLogger(__name__)


@dataclass
class DeterministicConfig:
    """Configuration for PyTorch deterministic mode."""
    
    # Core deterministic settings
    enable_deterministic: bool = False
    
    # PyTorch CPU deterministic settings
    torch_deterministic_algorithms: bool = True
    torch_warn_only: bool = False  # If True, warn instead of error on non-deterministic ops
    
    # CUDA deterministic settings (if GPU available)
    cuda_deterministic: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False  # Must be False for determinism
    
    # Additional deterministic settings
    cublas_workspace_config: bool = True  # Set CUBLAS workspace for determinism
    
    # Environment variables
    set_python_hash_seed: bool = True
    python_hash_seed: int = 0
    
    # Warnings
    show_warnings: bool = True
    strict_mode: bool = False  # If True, raise errors on non-deterministic operations


class DeterministicMode:
    """
    Manager for PyTorch deterministic mode.
    
    This class handles all aspects of enabling deterministic behavior in PyTorch,
    including CPU and GPU operations.
    """
    
    def __init__(self, config: Optional[DeterministicConfig] = None):
        """
        Initialize deterministic mode manager.
        
        Args:
            config: Deterministic configuration settings
        """
        self.config = config or DeterministicConfig()
        self._original_state = {}
        self._is_enabled = False
    
    def enable(self, warn_only: bool = False) -> Dict[str, Any]:
        """
        Enable full deterministic mode for PyTorch.
        
        Args:
            warn_only: If True, warn on non-deterministic operations instead of erroring
            
        Returns:
            Dictionary containing all deterministic settings applied
        """
        if self._is_enabled:
            logger.warning("Deterministic mode already enabled")
            return self.get_current_state()
        
        settings_applied = {}
        
        # Store original state
        self._store_original_state()
        
        # Set PyTorch deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                mode = 'warn' if (warn_only or self.config.torch_warn_only) else True
                torch.use_deterministic_algorithms(mode)
                settings_applied['torch_deterministic_algorithms'] = mode
                logger.info(f"PyTorch deterministic algorithms: {mode}")
            except RuntimeError as e:
                if self.config.strict_mode:
                    raise
                logger.warning(f"Could not set deterministic algorithms: {e}")
                settings_applied['torch_deterministic_algorithms'] = 'failed'
        
        # Set CUDA deterministic settings if available
        if torch.cuda.is_available():
            # cuDNN deterministic
            if self.config.cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                settings_applied['cudnn_deterministic'] = True
                logger.info("cuDNN deterministic: enabled")
            
            # Disable cuDNN benchmark for determinism
            if not self.config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = False
                settings_applied['cudnn_benchmark'] = False
                logger.info("cuDNN benchmark: disabled (for determinism)")
            
            # Set CUBLAS workspace config for determinism
            if self.config.cublas_workspace_config:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                settings_applied['cublas_workspace'] = ':4096:8'
                logger.info("CUBLAS workspace configured for determinism")
        else:
            logger.info("CUDA not available - CPU deterministic mode only")
        
        # Set environment variables
        if self.config.set_python_hash_seed:
            os.environ['PYTHONHASHSEED'] = str(self.config.python_hash_seed)
            settings_applied['python_hash_seed'] = self.config.python_hash_seed
            logger.info(f"PYTHONHASHSEED set to: {self.config.python_hash_seed}")
        
        # Additional CPU-specific deterministic settings
        if not torch.cuda.is_available():
            # CPU-specific optimizations that might affect determinism
            torch.set_num_threads(1)  # Single thread for full determinism
            settings_applied['num_threads'] = 1
            logger.info("CPU threads set to 1 for determinism")
        
        self._is_enabled = True
        
        # Log summary
        self._log_deterministic_summary(settings_applied)
        
        # Show warnings if enabled
        if self.config.show_warnings:
            self._show_deterministic_warnings()
        
        return settings_applied
    
    def disable(self) -> bool:
        """
        Disable deterministic mode and restore original settings.
        
        Returns:
            True if successfully disabled
        """
        if not self._is_enabled:
            logger.warning("Deterministic mode not currently enabled")
            return False
        
        # Restore original state
        self._restore_original_state()
        
        self._is_enabled = False
        logger.info("Deterministic mode disabled")
        
        return True
    
    def _store_original_state(self):
        """Store the original state before enabling deterministic mode."""
        self._original_state = {}
        
        # Store PyTorch settings
        if torch.cuda.is_available():
            self._original_state['cudnn_deterministic'] = torch.backends.cudnn.deterministic
            self._original_state['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        
        # Store environment variables
        self._original_state['pythonhashseed'] = os.environ.get('PYTHONHASHSEED')
        self._original_state['cublas_workspace'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        
        # Store thread count
        self._original_state['num_threads'] = torch.get_num_threads()
    
    def _restore_original_state(self):
        """Restore the original state."""
        if not self._original_state:
            return
        
        # Restore PyTorch settings
        if torch.cuda.is_available():
            if 'cudnn_deterministic' in self._original_state:
                torch.backends.cudnn.deterministic = self._original_state['cudnn_deterministic']
            if 'cudnn_benchmark' in self._original_state:
                torch.backends.cudnn.benchmark = self._original_state['cudnn_benchmark']
        
        # Restore environment variables
        if self._original_state.get('pythonhashseed'):
            os.environ['PYTHONHASHSEED'] = self._original_state['pythonhashseed']
        elif 'PYTHONHASHSEED' in os.environ:
            del os.environ['PYTHONHASHSEED']
        
        if self._original_state.get('cublas_workspace'):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = self._original_state['cublas_workspace']
        elif 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
            del os.environ['CUBLAS_WORKSPACE_CONFIG']
        
        # Restore thread count
        if 'num_threads' in self._original_state:
            torch.set_num_threads(self._original_state['num_threads'])
        
        # Disable deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(False)
            except RuntimeError:
                pass
    
    def _log_deterministic_summary(self, settings: Dict[str, Any]):
        """Log a summary of deterministic settings."""
        logger.info("="*60)
        logger.info("DETERMINISTIC MODE ENABLED")
        logger.info("="*60)
        
        for key, value in settings.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("="*60)
    
    def _show_deterministic_warnings(self):
        """Show warnings about deterministic mode limitations."""
        warnings_list = []
        
        if not torch.cuda.is_available():
            warnings_list.append(
                "CPU-only mode: Operations will be deterministic but may be slower"
            )
        
        warnings_list.extend([
            "Some operations may still be non-deterministic:",
            "  - Certain sparse operations",
            "  - Some backward passes for specific layers",
            "  - Operations involving atomicAdd on older GPUs"
        ])
        
        if warnings_list:
            logger.warning("\nDeterministic Mode Warnings:")
            for warning in warnings_list:
                logger.warning(f"  {warning}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of deterministic settings.
        
        Returns:
            Dictionary with current settings
        """
        state = {
            "enabled": self._is_enabled,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            state.update({
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            })
        
        state.update({
            "python_hash_seed": os.environ.get('PYTHONHASHSEED'),
            "cublas_workspace": os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            "num_threads": torch.get_num_threads(),
        })
        
        return state
    
    def check_operation_determinism(self, operation_name: str) -> bool:
        """
        Check if a specific operation is deterministic.
        
        Args:
            operation_name: Name of the operation to check
            
        Returns:
            True if operation is deterministic
        """
        # List of known non-deterministic operations
        non_deterministic_ops = {
            'scatter_add_': 'Use scatter_add with deterministic=True',
            'index_add_': 'May be non-deterministic on CUDA',
            'bmm': 'Use with deterministic algorithms enabled',
            'bincount': 'Non-deterministic on CUDA',
            'ctc_loss': 'Non-deterministic backward on CUDA',
            'grid_sample': 'Non-deterministic backward on CUDA',
        }
        
        if operation_name in non_deterministic_ops:
            logger.warning(
                f"Operation '{operation_name}' may be non-deterministic: "
                f"{non_deterministic_ops[operation_name]}"
            )
            return False
        
        return True


# Global instance
_deterministic_mode = None


def enable_deterministic_mode(
    warn_only: bool = False,
    cpu_only: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Enable PyTorch deterministic mode globally.
    
    Args:
        warn_only: If True, warn on non-deterministic ops instead of error
        cpu_only: If True, only set CPU deterministic settings
        verbose: If True, print detailed information
        
    Returns:
        Dictionary of applied settings
        
    Example:
        >>> from cogniforge.core.deterministic_mode import enable_deterministic_mode
        >>> settings = enable_deterministic_mode(warn_only=True)
        >>> print(f"Deterministic mode enabled: {settings}")
    """
    global _deterministic_mode
    
    config = DeterministicConfig(
        enable_deterministic=True,
        torch_warn_only=warn_only,
        show_warnings=verbose
    )
    
    if cpu_only:
        config.cuda_deterministic = False
        config.cudnn_deterministic = False
    
    _deterministic_mode = DeterministicMode(config)
    return _deterministic_mode.enable(warn_only=warn_only)


def disable_deterministic_mode() -> bool:
    """
    Disable PyTorch deterministic mode globally.
    
    Returns:
        True if successfully disabled
    """
    global _deterministic_mode
    
    if _deterministic_mode is None:
        logger.warning("Deterministic mode was not enabled")
        return False
    
    result = _deterministic_mode.disable()
    _deterministic_mode = None
    return result


def get_deterministic_state() -> Dict[str, Any]:
    """
    Get current state of deterministic settings.
    
    Returns:
        Dictionary with current deterministic settings
    """
    if _deterministic_mode is None:
        return {"enabled": False, "message": "Deterministic mode not initialized"}
    
    return _deterministic_mode.get_current_state()


def set_deterministic_debug_mode(enabled: bool = True):
    """
    Enable debug mode for deterministic operations.
    
    This will cause PyTorch to error when a non-deterministic operation is called.
    
    Args:
        enabled: If True, enable debug mode
    """
    if hasattr(torch, 'set_deterministic_debug_mode'):
        torch.set_deterministic_debug_mode(enabled)
        logger.info(f"Deterministic debug mode: {'enabled' if enabled else 'disabled'}")
    else:
        logger.warning("Deterministic debug mode not available in this PyTorch version")


if __name__ == "__main__":
    """Demonstration of deterministic mode."""
    
    print("\n" + "#"*60)
    print("# DETERMINISTIC MODE DEMONSTRATION")
    print("#"*60)
    
    # Test 1: Enable deterministic mode
    print("\n1. Enabling Deterministic Mode:")
    print("-"*40)
    settings = enable_deterministic_mode(warn_only=True, verbose=True)
    print(f"Settings applied: {settings}")
    
    # Test 2: Check reproducibility
    print("\n2. Testing Reproducibility:")
    print("-"*40)
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Generate random tensors
    tensor1 = torch.randn(10, 10)
    
    # Reset seed and generate again
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    tensor2 = torch.randn(10, 10)
    
    # Check if identical
    identical = torch.allclose(tensor1, tensor2)
    print(f"Tensors identical: {identical}")
    
    # Test 3: Get current state
    print("\n3. Current Deterministic State:")
    print("-"*40)
    state = get_deterministic_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Test 4: Disable deterministic mode
    print("\n4. Disabling Deterministic Mode:")
    print("-"*40)
    success = disable_deterministic_mode()
    print(f"Disabled successfully: {success}")
    
    print("\n✅ Deterministic mode demonstration complete!")