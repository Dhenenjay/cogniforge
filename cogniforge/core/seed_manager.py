"""
Seed Management Module for Reproducible Experiments

This module provides comprehensive seed management for NumPy, PyTorch, and Python's
random module, ensuring reproducibility and proper logging of all seeds in run summaries.
"""

import os
import random
import logging
import json
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import hashlib

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SeedConfig:
    """Configuration for seed management."""
    
    # Main seed (if None, will be auto-generated)
    master_seed: Optional[int] = None
    
    # Individual seeds (derived from master if None)
    numpy_seed: Optional[int] = None
    torch_seed: Optional[int] = None
    python_seed: Optional[int] = None
    
    # PyTorch specific
    cuda_deterministic: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False  # Disable for determinism
    
    # Environment seeds
    env_seed: Optional[int] = None
    worker_seed: Optional[int] = None
    
    # Auto-generation
    auto_generate: bool = True  # Auto-generate if master_seed is None
    use_time_based: bool = False  # Use time for auto-generation
    
    # Logging
    log_seeds: bool = True
    save_to_file: bool = True
    log_file_path: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize seeds after dataclass creation."""
        if self.master_seed is None and self.auto_generate:
            if self.use_time_based:
                # Time-based seed for uniqueness
                self.master_seed = int(time.time() * 1000) % (2**32 - 1)
            else:
                # Random seed
                self.master_seed = random.randint(0, 2**32 - 1)
        
        # Derive individual seeds from master seed if not provided
        if self.master_seed is not None:
            if self.numpy_seed is None:
                self.numpy_seed = self._derive_seed(self.master_seed, "numpy")
            if self.torch_seed is None:
                self.torch_seed = self._derive_seed(self.master_seed, "torch")
            if self.python_seed is None:
                self.python_seed = self._derive_seed(self.master_seed, "python")
            if self.env_seed is None:
                self.env_seed = self._derive_seed(self.master_seed, "env")
            if self.worker_seed is None:
                self.worker_seed = self._derive_seed(self.master_seed, "worker")
    
    def _derive_seed(self, master: int, component: str) -> int:
        """Derive a component seed from master seed."""
        # Use hash to create different but deterministic seeds
        combined = f"{master}_{component}"
        hash_val = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        return hash_val % (2**32 - 1)


@dataclass
class SeedState:
    """Current state of all random seeds."""
    
    master_seed: int
    numpy_seed: int
    torch_seed: int
    python_seed: int
    cuda_seed: Optional[int] = None
    env_seed: Optional[int] = None
    worker_seed: Optional[int] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    torch_version: str = field(default_factory=lambda: torch.__version__)
    numpy_version: str = field(default_factory=lambda: np.__version__)
    cuda_available: bool = field(default_factory=torch.cuda.is_available)
    cuda_version: Optional[str] = field(default_factory=lambda: torch.version.cuda if torch.cuda.is_available() else None)
    
    # PyTorch settings
    cuda_deterministic: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class SeedManager:
    """
    Centralized seed manager for all random number generators.
    
    This manager ensures reproducibility by setting seeds for:
    - NumPy
    - PyTorch (CPU and CUDA)
    - Python's random module
    - Environment variables
    
    It also logs all seeds for experiment tracking.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure single manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the seed manager."""
        if not self._initialized:
            self.config: Optional[SeedConfig] = None
            self.current_state: Optional[SeedState] = None
            self.history: List[SeedState] = []
            self._original_state: Optional[Dict[str, Any]] = None
            SeedManager._initialized = True
    
    def set_seeds(
        self,
        seed: Optional[int] = None,
        config: Optional[SeedConfig] = None,
        **kwargs
    ) -> SeedState:
        """
        Set all random seeds for reproducibility.
        
        Args:
            seed: Master seed value (overrides config)
            config: SeedConfig object with detailed settings
            **kwargs: Additional keyword arguments for SeedConfig
            
        Returns:
            SeedState object containing all set seeds
            
        Example:
            >>> manager = SeedManager()
            >>> state = manager.set_seeds(42)
            >>> print(f"Seeds set: {state.to_dict()}")
            
            >>> # Or with config
            >>> config = SeedConfig(master_seed=42, cuda_deterministic=True)
            >>> state = manager.set_seeds(config=config)
        """
        # Create or update config
        if config is None:
            config = SeedConfig(master_seed=seed, **kwargs)
        elif seed is not None:
            config.master_seed = seed
        
        self.config = config
        
        # Store original state if first time
        if self._original_state is None:
            self._store_original_state()
        
        # Set NumPy seed
        if config.numpy_seed is not None:
            np.random.seed(config.numpy_seed)
            if config.log_seeds:
                logger.info(f"NumPy seed set to: {config.numpy_seed}")
        
        # Set Python random seed
        if config.python_seed is not None:
            random.seed(config.python_seed)
            if config.log_seeds:
                logger.info(f"Python random seed set to: {config.python_seed}")
        
        # Set PyTorch seeds
        if config.torch_seed is not None:
            torch.manual_seed(config.torch_seed)
            if config.log_seeds:
                logger.info(f"PyTorch seed set to: {config.torch_seed}")
            
            # CUDA seeds
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.torch_seed)
                torch.cuda.manual_seed_all(config.torch_seed)
                if config.log_seeds:
                    logger.info(f"CUDA seed set to: {config.torch_seed}")
        
        # Set PyTorch deterministic behavior
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = config.cudnn_deterministic
            torch.backends.cudnn.benchmark = config.cudnn_benchmark
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(config.cuda_deterministic)
                except RuntimeError as e:
                    logger.warning(f"Could not set deterministic algorithms: {e}")
        
        # Set environment variables for additional reproducibility
        if config.env_seed is not None:
            os.environ['PYTHONHASHSEED'] = str(config.env_seed)
            if config.log_seeds:
                logger.info(f"PYTHONHASHSEED set to: {config.env_seed}")
        
        # Create state object
        self.current_state = SeedState(
            master_seed=config.master_seed or 0,
            numpy_seed=config.numpy_seed or 0,
            torch_seed=config.torch_seed or 0,
            python_seed=config.python_seed or 0,
            cuda_seed=config.torch_seed if torch.cuda.is_available() else None,
            env_seed=config.env_seed,
            worker_seed=config.worker_seed,
            cuda_deterministic=config.cuda_deterministic,
            cudnn_deterministic=config.cudnn_deterministic,
            cudnn_benchmark=config.cudnn_benchmark
        )
        
        # Add to history
        self.history.append(self.current_state)
        
        # Log summary
        if config.log_seeds:
            self._log_seed_summary()
        
        # Save to file if requested
        if config.save_to_file:
            self.save_seeds_to_file(config.log_file_path)
        
        return self.current_state
    
    def _store_original_state(self):
        """Store the original state of random generators."""
        self._original_state = {
            'numpy_state': np.random.get_state(),
            'python_state': random.getstate(),
            'torch_state': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            self._original_state['cuda_state'] = torch.cuda.get_rng_state()
            self._original_state['cudnn_deterministic'] = torch.backends.cudnn.deterministic
            self._original_state['cudnn_benchmark'] = torch.backends.cudnn.benchmark
    
    def restore_original_state(self):
        """Restore the original state of random generators."""
        if self._original_state is None:
            logger.warning("No original state to restore")
            return
        
        np.random.set_state(self._original_state['numpy_state'])
        random.setstate(self._original_state['python_state'])
        torch.set_rng_state(self._original_state['torch_state'])
        
        if torch.cuda.is_available() and 'cuda_state' in self._original_state:
            torch.cuda.set_rng_state(self._original_state['cuda_state'])
            torch.backends.cudnn.deterministic = self._original_state['cudnn_deterministic']
            torch.backends.cudnn.benchmark = self._original_state['cudnn_benchmark']
        
        logger.info("Restored original random state")
    
    def _log_seed_summary(self):
        """Log a summary of all set seeds."""
        if self.current_state is None:
            return
        
        logger.info("="*60)
        logger.info("SEED CONFIGURATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Master Seed: {self.current_state.master_seed}")
        logger.info(f"NumPy Seed: {self.current_state.numpy_seed}")
        logger.info(f"PyTorch Seed: {self.current_state.torch_seed}")
        logger.info(f"Python Random Seed: {self.current_state.python_seed}")
        
        if self.current_state.cuda_available:
            logger.info(f"CUDA Seed: {self.current_state.cuda_seed}")
            logger.info(f"CUDA Deterministic: {self.current_state.cuda_deterministic}")
            logger.info(f"cuDNN Deterministic: {self.current_state.cudnn_deterministic}")
            logger.info(f"cuDNN Benchmark: {self.current_state.cudnn_benchmark}")
        
        if self.current_state.env_seed is not None:
            logger.info(f"Environment Seed: {self.current_state.env_seed}")
        
        logger.info(f"Timestamp: {self.current_state.timestamp}")
        logger.info("="*60)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current seed configuration.
        
        Returns:
            Dictionary with seed information for logging
        """
        if self.current_state is None:
            return {"error": "No seeds have been set"}
        
        summary = {
            "seeds": {
                "master": self.current_state.master_seed,
                "numpy": self.current_state.numpy_seed,
                "torch": self.current_state.torch_seed,
                "python": self.current_state.python_seed,
            },
            "environment": {
                "torch_version": self.current_state.torch_version,
                "numpy_version": self.current_state.numpy_version,
                "cuda_available": self.current_state.cuda_available,
                "cuda_version": self.current_state.cuda_version,
            },
            "deterministic_settings": {
                "cuda_deterministic": self.current_state.cuda_deterministic,
                "cudnn_deterministic": self.current_state.cudnn_deterministic,
                "cudnn_benchmark": self.current_state.cudnn_benchmark,
            },
            "timestamp": self.current_state.timestamp,
            "history_length": len(self.history)
        }
        
        return summary
    
    def save_seeds_to_file(self, filepath: Optional[Path] = None):
        """
        Save current seed configuration to a file.
        
        Args:
            filepath: Path to save the seed configuration
        """
        if self.current_state is None:
            logger.warning("No seeds to save")
            return
        
        if filepath is None:
            # Default path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"seeds_{timestamp}.json")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(self.current_state.to_json())
        
        logger.info(f"Seed configuration saved to: {filepath}")
    
    def load_seeds_from_file(self, filepath: Path) -> SeedState:
        """
        Load and apply seed configuration from a file.
        
        Args:
            filepath: Path to the seed configuration file
            
        Returns:
            SeedState loaded from file
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create config from loaded data
        config = SeedConfig(
            master_seed=data['master_seed'],
            numpy_seed=data['numpy_seed'],
            torch_seed=data['torch_seed'],
            python_seed=data['python_seed'],
            env_seed=data.get('env_seed'),
            worker_seed=data.get('worker_seed'),
            cuda_deterministic=data.get('cuda_deterministic', True),
            cudnn_deterministic=data.get('cudnn_deterministic', True),
            cudnn_benchmark=data.get('cudnn_benchmark', False)
        )
        
        # Apply seeds
        state = self.set_seeds(config=config)
        
        logger.info(f"Seed configuration loaded from: {filepath}")
        
        return state
    
    def get_worker_seed(self, worker_id: int) -> int:
        """
        Get a unique seed for a worker process.
        
        Args:
            worker_id: ID of the worker process
            
        Returns:
            Unique seed for the worker
        """
        if self.current_state is None:
            base_seed = int(time.time()) % (2**32 - 1)
        else:
            base_seed = self.current_state.worker_seed or self.current_state.master_seed
        
        # Create unique seed for worker
        worker_seed = (base_seed + worker_id * 100) % (2**32 - 1)
        
        return worker_seed
    
    def log_to_tensorboard(self, writer, step: int = 0):
        """
        Log seed information to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter
            step: Global step for logging
        """
        if self.current_state is None:
            return
        
        writer.add_text('seeds/master', str(self.current_state.master_seed), step)
        writer.add_text('seeds/numpy', str(self.current_state.numpy_seed), step)
        writer.add_text('seeds/torch', str(self.current_state.torch_seed), step)
        writer.add_text('seeds/python', str(self.current_state.python_seed), step)
        
        # Log full configuration as JSON
        writer.add_text('seeds/full_config', self.current_state.to_json(), step)
    
    def log_to_wandb(self, run=None):
        """
        Log seed information to Weights & Biases.
        
        Args:
            run: W&B run object (uses wandb.run if None)
        """
        try:
            import wandb
            
            if run is None:
                run = wandb.run
            
            if run is None or self.current_state is None:
                return
            
            # Log as config
            run.config.update({
                "seed_master": self.current_state.master_seed,
                "seed_numpy": self.current_state.numpy_seed,
                "seed_torch": self.current_state.torch_seed,
                "seed_python": self.current_state.python_seed,
                "cuda_deterministic": self.current_state.cuda_deterministic,
            })
            
            # Log as summary
            run.summary["seeds"] = self.current_state.to_dict()
            
            logger.info("Seeds logged to W&B")
            
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")


# Global singleton instance
_seed_manager = SeedManager()


def set_global_seeds(
    seed: Optional[int] = None,
    numpy: bool = True,
    torch: bool = True,
    python: bool = True,
    cuda_deterministic: bool = True,
    log: bool = True,
    save: bool = False
) -> SeedState:
    """
    Convenience function to set all seeds globally.
    
    Args:
        seed: Master seed value (auto-generated if None)
        numpy: Set NumPy seed
        torch: Set PyTorch seed
        python: Set Python random seed
        cuda_deterministic: Enable CUDA deterministic mode
        log: Log seed configuration
        save: Save seeds to file
        
    Returns:
        SeedState with all set seeds
        
    Example:
        >>> from cogniforge.core.seed_manager import set_global_seeds
        >>> state = set_global_seeds(42)
        >>> print(f"Master seed: {state.master_seed}")
    """
    config = SeedConfig(
        master_seed=seed,
        cuda_deterministic=cuda_deterministic,
        log_seeds=log,
        save_to_file=save
    )
    
    # Disable specific seeds if requested
    if not numpy:
        config.numpy_seed = None
    if not torch:
        config.torch_seed = None
    if not python:
        config.python_seed = None
    
    return _seed_manager.set_seeds(config=config)


def get_seed_manager() -> SeedManager:
    """Get the global seed manager instance."""
    return _seed_manager


def log_seeds_to_run_summary(
    run_name: str = "experiment",
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a run summary with seed information.
    
    Args:
        run_name: Name of the experimental run
        additional_info: Additional information to include
        
    Returns:
        Complete run summary dictionary
        
    Example:
        >>> summary = log_seeds_to_run_summary("my_experiment")
        >>> print(json.dumps(summary, indent=2))
    """
    manager = get_seed_manager()
    
    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "seeds": manager.get_summary() if manager.current_state else None,
    }
    
    if additional_info:
        summary.update(additional_info)
    
    # Log the summary
    logger.info("\n" + "="*60)
    logger.info("RUN SUMMARY")
    logger.info("="*60)
    logger.info(json.dumps(summary, indent=2, default=str))
    logger.info("="*60 + "\n")
    
    return summary


if __name__ == "__main__":
    """Demonstration of seed management functionality."""
    
    print("\n" + "#"*60)
    print("# SEED MANAGER DEMONSTRATION")
    print("#"*60)
    
    # Test 1: Basic seed setting
    print("\n1. Basic Seed Setting:")
    print("-"*40)
    state1 = set_global_seeds(42)
    print(f"Master seed: {state1.master_seed}")
    print(f"NumPy seed: {state1.numpy_seed}")
    print(f"PyTorch seed: {state1.torch_seed}")
    
    # Test 2: Auto-generated seeds
    print("\n2. Auto-Generated Seeds:")
    print("-"*40)
    state2 = set_global_seeds()  # No seed provided
    print(f"Auto-generated master seed: {state2.master_seed}")
    
    # Test 3: Custom configuration
    print("\n3. Custom Configuration:")
    print("-"*40)
    config = SeedConfig(
        master_seed=12345,
        cuda_deterministic=True,
        cudnn_benchmark=False
    )
    manager = get_seed_manager()
    state3 = manager.set_seeds(config=config)
    
    # Test 4: Save and load seeds
    print("\n4. Save and Load Seeds:")
    print("-"*40)
    manager.save_seeds_to_file("test_seeds.json")
    print("Seeds saved to test_seeds.json")
    
    # Reset and load
    set_global_seeds(99999)
    print(f"Changed to seed: {manager.current_state.master_seed}")
    
    loaded_state = manager.load_seeds_from_file("test_seeds.json")
    print(f"Loaded seed: {loaded_state.master_seed}")
    
    # Test 5: Run summary
    print("\n5. Run Summary with Seeds:")
    print("-"*40)
    summary = log_seeds_to_run_summary(
        run_name="test_experiment",
        additional_info={
            "model": "test_model",
            "dataset": "test_data",
            "epochs": 100
        }
    )
    
    # Clean up
    Path("test_seeds.json").unlink(missing_ok=True)
    
    print("\n✅ Seed manager demonstration complete!")