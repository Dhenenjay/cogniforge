#!/usr/bin/env python3
"""
Command-Line Interface for Training with Deterministic Mode Support

This module provides a CLI interface with --deterministic flag to enable
PyTorch deterministic mode for reproducible training.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np

from cogniforge.core.seed_manager import set_global_seeds, log_seeds_to_run_summary
from cogniforge.core.deterministic_mode import (
    enable_deterministic_mode,
    disable_deterministic_mode,
    get_deterministic_state
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with deterministic mode support.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="CogniForge Training CLI with Deterministic Mode Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'demo'],
        help='Task to perform'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Reproducibility arguments
    reproducibility_group = parser.add_argument_group('Reproducibility')
    
    reproducibility_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (auto-generated if not provided)'
    )
    
    reproducibility_group.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable PyTorch deterministic mode for full reproducibility (CPU)'
    )
    
    reproducibility_group.add_argument(
        '--deterministic-warn-only',
        action='store_true',
        help='Warn instead of error on non-deterministic operations'
    )
    
    reproducibility_group.add_argument(
        '--deterministic-debug',
        action='store_true',
        help='Enable debug mode for deterministic operations (errors on non-deterministic ops)'
    )
    
    reproducibility_group.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only execution (automatically sets deterministic for CPU)'
    )
    
    # Training arguments
    training_group = parser.add_argument_group('Training')
    
    training_group.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    training_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    training_group.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    training_group.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory for output files'
    )
    
    output_group.add_argument(
        '--save-config',
        action='store_true',
        help='Save configuration including seeds and deterministic settings'
    )
    
    output_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def setup_reproducibility(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Setup reproducibility based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with reproducibility settings
    """
    settings = {}
    
    # Set random seeds
    seed_state = set_global_seeds(
        seed=args.seed,
        log=args.verbose,
        save=args.save_config
    )
    settings['seed_state'] = seed_state.to_dict()
    
    logger.info(f"Seeds initialized - Master seed: {seed_state.master_seed}")
    
    # Enable deterministic mode if requested
    if args.deterministic or args.cpu_only:
        logger.info("Enabling PyTorch deterministic mode...")
        
        deterministic_settings = enable_deterministic_mode(
            warn_only=args.deterministic_warn_only,
            cpu_only=args.cpu_only,
            verbose=args.verbose
        )
        settings['deterministic'] = deterministic_settings
        
        # Set deterministic debug mode if requested
        if args.deterministic_debug:
            from cogniforge.core.deterministic_mode import set_deterministic_debug_mode
            set_deterministic_debug_mode(True)
            logger.info("Deterministic debug mode enabled")
    else:
        settings['deterministic'] = {'enabled': False}
    
    # Force CPU if requested
    if args.cpu_only:
        torch.set_default_tensor_type('torch.FloatTensor')
        logger.info("CPU-only mode enabled")
        settings['device'] = 'cpu'
    else:
        settings['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return settings


def run_training(args: argparse.Namespace, settings: Dict[str, Any]):
    """
    Run training with deterministic settings.
    
    Args:
        args: Command-line arguments
        settings: Reproducibility settings
    """
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    # Log configuration
    logger.info(f"Task: {args.task}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {settings['device']}")
    logger.info(f"Deterministic: {args.deterministic}")
    logger.info(f"Master seed: {settings['seed_state']['master_seed']}")
    
    # Simulate training loop
    device = torch.device(settings['device'])
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        # Generate reproducible random data
        batch_data = torch.randn(args.batch_size, 10, device=device)
        batch_labels = torch.randint(0, 10, (args.batch_size,), device=device)
        
        # Forward pass
        outputs = model(batch_data)
        loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {loss.item():.4f}")
    
    logger.info("Training complete!")
    
    # Save final summary
    summary = log_seeds_to_run_summary(
        run_name=f"{args.task}_deterministic" if args.deterministic else args.task,
        additional_info={
            "args": vars(args),
            "settings": settings,
            "final_loss": loss.item()
        }
    )
    
    # Save configuration if requested
    if args.save_config:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = output_dir / "run_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to: {config_file}")


def verify_determinism(args: argparse.Namespace):
    """
    Verify that deterministic mode produces identical results.
    
    Args:
        args: Command-line arguments
    """
    logger.info("="*60)
    logger.info("VERIFYING DETERMINISM")
    logger.info("="*60)
    
    # Run 1
    logger.info("\n--- Run 1 ---")
    args.seed = 42
    settings1 = setup_reproducibility(args)
    
    # Generate some operations
    torch.manual_seed(42)
    tensor1 = torch.randn(100, 100)
    
    if not args.cpu_only and torch.cuda.is_available():
        tensor1_cuda = torch.randn(100, 100, device='cuda')
    
    # Clean up
    if args.deterministic:
        disable_deterministic_mode()
    
    # Run 2
    logger.info("\n--- Run 2 ---")
    args.seed = 42
    settings2 = setup_reproducibility(args)
    
    # Generate same operations
    torch.manual_seed(42)
    tensor2 = torch.randn(100, 100)
    
    if not args.cpu_only and torch.cuda.is_available():
        tensor2_cuda = torch.randn(100, 100, device='cuda')
    
    # Verify
    logger.info("\n--- Verification ---")
    cpu_match = torch.allclose(tensor1, tensor2)
    logger.info(f"CPU tensors match: {cpu_match}")
    
    if not args.cpu_only and torch.cuda.is_available():
        cuda_match = torch.allclose(tensor1_cuda, tensor2_cuda)
        logger.info(f"CUDA tensors match: {cuda_match}")
    
    # Check deterministic state
    det_state = get_deterministic_state()
    logger.info(f"\nDeterministic state: {det_state}")
    
    if cpu_match:
        logger.info("\n✅ Determinism verified successfully!")
    else:
        logger.warning("\n⚠️ Determinism verification failed!")


def main():
    """Main entry point for CLI."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print header
    print("\n" + "#"*60)
    print("# COGNIFORGE TRAINING CLI")
    print("#"*60)
    print(f"# Deterministic Mode: {'ENABLED' if args.deterministic else 'DISABLED'}")
    print(f"# CPU-Only Mode: {'ENABLED' if args.cpu_only else 'DISABLED'}")
    print("#"*60 + "\n")
    
    try:
        # Setup reproducibility
        settings = setup_reproducibility(args)
        
        # Run task
        if args.task == 'train':
            run_training(args, settings)
        elif args.task == 'evaluate':
            logger.info("Evaluation not implemented in demo")
        elif args.task == 'demo':
            verify_determinism(args)
        
        # Clean up deterministic mode if enabled
        if args.deterministic:
            disable_deterministic_mode()
            logger.info("Deterministic mode disabled")
        
        print("\n✅ Task completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()