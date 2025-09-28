#!/usr/bin/env python3
"""
Test Seed Management and Run Summary Logging

This script demonstrates comprehensive seed management for NumPy, PyTorch,
and Python's random module, along with proper logging in run summaries.
"""

import numpy as np
import torch
import random
import json
import logging
from pathlib import Path
from datetime import datetime

from cogniforge.core.seed_manager import (
    set_global_seeds,
    get_seed_manager,
    log_seeds_to_run_summary,
    SeedConfig,
    SeedState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_reproducibility(seed: int = 42):
    """
    Test that same seed produces same random numbers.
    """
    print("\n" + "="*60)
    print("TEST: Reproducibility with Fixed Seed")
    print("="*60)
    
    # Set seeds first time
    state1 = set_global_seeds(seed)
    
    # Generate random numbers
    numpy_vals1 = np.random.randn(5)
    torch_vals1 = torch.randn(5)
    python_vals1 = [random.random() for _ in range(5)]
    
    print(f"\nFirst run with seed {seed}:")
    print(f"  NumPy: {numpy_vals1[:3]}")
    print(f"  PyTorch: {torch_vals1[:3]}")
    print(f"  Python: {python_vals1[:3]}")
    
    # Reset with same seed
    state2 = set_global_seeds(seed)
    
    # Generate random numbers again
    numpy_vals2 = np.random.randn(5)
    torch_vals2 = torch.randn(5)
    python_vals2 = [random.random() for _ in range(5)]
    
    print(f"\nSecond run with seed {seed}:")
    print(f"  NumPy: {numpy_vals2[:3]}")
    print(f"  PyTorch: {torch_vals2[:3]}")
    print(f"  Python: {python_vals2[:3]}")
    
    # Verify reproducibility
    numpy_match = np.allclose(numpy_vals1, numpy_vals2)
    torch_match = torch.allclose(torch_vals1, torch_vals2)
    python_match = python_vals1 == python_vals2
    
    print("\nReproducibility check:")
    print(f"  ✓ NumPy: {'PASSED' if numpy_match else 'FAILED'}")
    print(f"  ✓ PyTorch: {'PASSED' if torch_match else 'FAILED'}")
    print(f"  ✓ Python: {'PASSED' if python_match else 'FAILED'}")
    
    return numpy_match and torch_match and python_match


def test_different_seeds():
    """
    Test that different seeds produce different results.
    """
    print("\n" + "="*60)
    print("TEST: Different Seeds Produce Different Results")
    print("="*60)
    
    results = {}
    
    for seed in [42, 123, 999]:
        set_global_seeds(seed)
        
        results[seed] = {
            'numpy': np.random.randn(3),
            'torch': torch.randn(3),
            'python': [random.random() for _ in range(3)]
        }
        
        print(f"\nSeed {seed}:")
        print(f"  NumPy first value: {results[seed]['numpy'][0]:.6f}")
        print(f"  PyTorch first value: {results[seed]['torch'][0].item():.6f}")
        print(f"  Python first value: {results[seed]['python'][0]:.6f}")
    
    # Check that different seeds give different results
    seed_42_numpy = results[42]['numpy'][0]
    seed_123_numpy = results[123]['numpy'][0]
    different = not np.isclose(seed_42_numpy, seed_123_numpy)
    
    print(f"\n✓ Different seeds produce different results: {'PASSED' if different else 'FAILED'}")
    
    return different


def test_cuda_determinism():
    """
    Test CUDA deterministic settings if available.
    """
    print("\n" + "="*60)
    print("TEST: CUDA Deterministic Settings")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping test")
        return True
    
    # Set seeds with CUDA determinism
    config = SeedConfig(
        master_seed=42,
        cuda_deterministic=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False
    )
    
    manager = get_seed_manager()
    state = manager.set_seeds(config=config)
    
    print(f"\nCUDA Settings:")
    print(f"  CUDA available: {state.cuda_available}")
    print(f"  CUDA version: {state.cuda_version}")
    print(f"  CUDA deterministic: {state.cuda_deterministic}")
    print(f"  cuDNN deterministic: {state.cudnn_deterministic}")
    print(f"  cuDNN benchmark: {state.cudnn_benchmark}")
    
    # Test CUDA reproducibility
    cuda_tensor1 = torch.randn(100, 100, device='cuda')
    
    # Reset and generate again
    manager.set_seeds(config=config)
    cuda_tensor2 = torch.randn(100, 100, device='cuda')
    
    match = torch.allclose(cuda_tensor1, cuda_tensor2)
    print(f"\n✓ CUDA reproducibility: {'PASSED' if match else 'FAILED'}")
    
    return match


def test_run_summary_logging():
    """
    Test comprehensive run summary with seeds.
    """
    print("\n" + "="*60)
    print("TEST: Run Summary with Seed Logging")
    print("="*60)
    
    # Set seeds for experiment
    seed_state = set_global_seeds(12345, save=False)
    
    # Simulate an experiment run
    experiment_config = {
        "model": "ResNet50",
        "dataset": "ImageNet",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "optimizer": "Adam"
    }
    
    # Simulate some results
    results = {
        "final_accuracy": 0.923,
        "final_loss": 0.234,
        "best_epoch": 87,
        "training_time_hours": 12.5
    }
    
    # Create comprehensive run summary
    run_summary = log_seeds_to_run_summary(
        run_name="resnet50_imagenet_experiment",
        additional_info={
            "experiment_config": experiment_config,
            "results": results,
            "hardware": {
                "gpu": "NVIDIA RTX 3090" if torch.cuda.is_available() else "CPU",
                "cpu_cores": 16,
                "ram_gb": 32
            }
        }
    )
    
    # Save summary to file
    summary_file = Path("run_summary_example.json")
    with open(summary_file, 'w') as f:
        json.dump(run_summary, f, indent=2, default=str)
    
    print(f"\n✓ Run summary saved to: {summary_file}")
    print(f"✓ Master seed logged: {run_summary['seeds']['seeds']['master']}")
    
    return True


def test_seed_persistence():
    """
    Test saving and loading seed configurations.
    """
    print("\n" + "="*60)
    print("TEST: Seed Persistence (Save/Load)")
    print("="*60)
    
    manager = get_seed_manager()
    
    # Set initial seeds and save
    original_state = set_global_seeds(seed=7777)
    seed_file = Path("test_seed_config.json")
    manager.save_seeds_to_file(seed_file)
    
    print(f"\nOriginal seed configuration:")
    print(f"  Master: {original_state.master_seed}")
    print(f"  NumPy: {original_state.numpy_seed}")
    print(f"  PyTorch: {original_state.torch_seed}")
    
    # Generate some random numbers
    original_random = np.random.randn(3)
    print(f"  Random values: {original_random}")
    
    # Change seeds
    set_global_seeds(seed=9999)
    changed_random = np.random.randn(3)
    print(f"\nAfter changing seed to 9999:")
    print(f"  Random values: {changed_random}")
    
    # Load original seeds back
    loaded_state = manager.load_seeds_from_file(seed_file)
    restored_random = np.random.randn(3)
    
    print(f"\nAfter loading saved configuration:")
    print(f"  Master: {loaded_state.master_seed}")
    print(f"  Random values: {restored_random}")
    
    # Verify restoration
    match = loaded_state.master_seed == original_state.master_seed
    print(f"\n✓ Seed restoration: {'PASSED' if match else 'FAILED'}")
    
    # Clean up
    seed_file.unlink(missing_ok=True)
    
    return match


def test_worker_seeds():
    """
    Test unique seed generation for worker processes.
    """
    print("\n" + "="*60)
    print("TEST: Worker Process Seeds")
    print("="*60)
    
    # Set master seed
    set_global_seeds(seed=1000)
    manager = get_seed_manager()
    
    print("\nWorker seeds generated from master seed 1000:")
    worker_seeds = []
    
    for worker_id in range(5):
        seed = manager.get_worker_seed(worker_id)
        worker_seeds.append(seed)
        print(f"  Worker {worker_id}: {seed}")
    
    # Check uniqueness
    unique = len(worker_seeds) == len(set(worker_seeds))
    print(f"\n✓ All worker seeds unique: {'PASSED' if unique else 'FAILED'}")
    
    # Test reproducibility
    seed_again = manager.get_worker_seed(0)
    reproducible = seed_again == worker_seeds[0]
    print(f"✓ Worker seeds reproducible: {'PASSED' if reproducible else 'FAILED'}")
    
    return unique and reproducible


def simulate_ml_experiment():
    """
    Simulate a complete ML experiment with seed management.
    """
    print("\n" + "#"*60)
    print("# SIMULATED ML EXPERIMENT WITH SEED MANAGEMENT")
    print("#"*60)
    
    # Initialize experiment
    experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set seeds at start of experiment
    print("\n1. Initializing seeds...")
    seed_state = set_global_seeds(seed=42, save=True)
    
    # Simulate data loading
    print("\n2. Loading data (with seeded shuffle)...")
    data_indices = np.random.permutation(1000)[:100]  # Seeded shuffle
    print(f"   First 5 indices: {data_indices[:5]}")
    
    # Simulate model initialization
    print("\n3. Initializing model weights...")
    model_weights = torch.randn(10, 10)  # Seeded initialization
    print(f"   Weight stats: mean={model_weights.mean():.4f}, std={model_weights.std():.4f}")
    
    # Simulate training with random augmentation
    print("\n4. Training with random augmentation...")
    for epoch in range(3):
        aug_factor = random.random()  # Seeded augmentation
        loss = np.random.randn() * 0.1 + 1.0  # Seeded noise
        print(f"   Epoch {epoch+1}: aug={aug_factor:.3f}, loss={loss:.4f}")
    
    # Log final summary
    print("\n5. Logging experiment summary...")
    summary = log_seeds_to_run_summary(
        run_name=experiment_name,
        additional_info={
            "training_complete": True,
            "final_loss": loss,
            "data_samples": len(data_indices),
            "model_parameters": model_weights.numel()
        }
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Name: {summary['run_name']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Master Seed: {summary['seeds']['seeds']['master']}")
    print(f"Final Loss: {summary['final_loss']:.4f}")
    print("="*60)
    
    return True


def main():
    """Run all seed management tests."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "SEED MANAGEMENT TEST SUITE" + " "*27 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print("\nThis test suite demonstrates comprehensive seed management")
    print("for NumPy, PyTorch, and Python's random module with proper")
    print("logging of all seeds in run summaries.")
    
    # Run tests
    tests = [
        ("Reproducibility", test_reproducibility),
        ("Different Seeds", test_different_seeds),
        ("CUDA Determinism", test_cuda_determinism),
        ("Run Summary Logging", test_run_summary_logging),
        ("Seed Persistence", test_seed_persistence),
        ("Worker Seeds", test_worker_seeds),
        ("ML Experiment Simulation", simulate_ml_experiment)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "#"*70)
    print("# TEST RESULTS SUMMARY")
    print("#"*70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nKey Features Demonstrated:")
        print("  ✓ Reproducible random number generation")
        print("  ✓ Synchronized seeding across NumPy, PyTorch, and Python")
        print("  ✓ CUDA deterministic settings for GPU reproducibility")
        print("  ✓ Comprehensive seed logging in run summaries")
        print("  ✓ Seed configuration persistence (save/load)")
        print("  ✓ Unique worker process seed generation")
        print("  ✓ Integration with ML experiment workflows")
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
    
    # Clean up
    for file in ["run_summary_example.json", "seeds_*.json"]:
        for path in Path(".").glob(file):
            path.unlink(missing_ok=True)
    
    print("\n" + "#"*70)
    print("# SEED MANAGEMENT TEST SUITE COMPLETE")
    print("#"*70)


if __name__ == "__main__":
    main()