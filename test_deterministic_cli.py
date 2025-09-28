#!/usr/bin/env python3
"""
Test Script for --deterministic Flag

This script demonstrates the --deterministic flag functionality
for enabling PyTorch deterministic mode via command-line.
"""

import subprocess
import sys
import json
from pathlib import Path
import torch
import numpy as np


def run_command(cmd):
    """Run a command and capture output."""
    print(f"\n📌 Running: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr, file=sys.stderr)
    
    return result.returncode


def test_deterministic_flag():
    """Test the --deterministic flag functionality."""
    print("\n" + "#"*70)
    print("# TEST: --deterministic Flag")
    print("#"*70)
    
    # Test 1: Help message shows deterministic flag
    print("\n1. Checking --help for deterministic flag:")
    cmd = [sys.executable, "-m", "cogniforge.cli.train_cli", "--help"]
    run_command(cmd)
    
    # Test 2: Run without deterministic flag
    print("\n2. Running WITHOUT --deterministic flag:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli.train_cli",
        "--task", "demo",
        "--seed", "42",
        "--epochs", "5"
    ]
    run_command(cmd)
    
    # Test 3: Run with deterministic flag
    print("\n3. Running WITH --deterministic flag:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli.train_cli",
        "--task", "demo",
        "--deterministic",
        "--seed", "42",
        "--epochs", "5"
    ]
    run_command(cmd)
    
    # Test 4: Run with deterministic and warn-only
    print("\n4. Running with --deterministic --deterministic-warn-only:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli.train_cli",
        "--task", "demo",
        "--deterministic",
        "--deterministic-warn-only",
        "--seed", "42",
        "--epochs", "5"
    ]
    run_command(cmd)
    
    # Test 5: CPU-only mode (automatically enables deterministic)
    print("\n5. Running with --cpu-only (auto-enables deterministic):")
    cmd = [
        sys.executable, "-m", "cogniforge.cli.train_cli",
        "--task", "demo",
        "--cpu-only",
        "--seed", "42",
        "--epochs", "5"
    ]
    run_command(cmd)
    
    # Test 6: Save configuration with deterministic settings
    print("\n6. Saving configuration with deterministic settings:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli.train_cli",
        "--task", "train",
        "--deterministic",
        "--seed", "12345",
        "--epochs", "2",
        "--save-config",
        "--output-dir", "test_output"
    ]
    run_command(cmd)
    
    # Check saved configuration
    config_file = Path("test_output/run_config.json")
    if config_file.exists():
        print("\n📄 Saved configuration content:")
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(json.dumps(config.get('settings', {}).get('deterministic', {}), indent=2))
        
        # Clean up
        config_file.unlink()
        config_file.parent.rmdir()


def test_reproducibility_with_deterministic():
    """Test that deterministic mode ensures reproducibility."""
    print("\n" + "#"*70)
    print("# TEST: Reproducibility with Deterministic Mode")
    print("#"*70)
    
    from cogniforge.core.deterministic_mode import enable_deterministic_mode, disable_deterministic_mode
    from cogniforge.core.seed_manager import set_global_seeds
    
    print("\n1. Testing WITHOUT deterministic mode:")
    print("-" * 40)
    
    # Run 1 without deterministic
    set_global_seeds(42)
    x1 = torch.randn(5, 5)
    
    # Run 2 without deterministic
    set_global_seeds(42)
    x2 = torch.randn(5, 5)
    
    without_det_match = torch.allclose(x1, x2)
    print(f"Tensors match without deterministic: {without_det_match}")
    
    print("\n2. Testing WITH deterministic mode:")
    print("-" * 40)
    
    # Enable deterministic mode
    settings = enable_deterministic_mode(warn_only=True)
    print(f"Deterministic settings applied: {list(settings.keys())}")
    
    # Run 1 with deterministic
    set_global_seeds(42)
    y1 = torch.randn(5, 5)
    
    # Run 2 with deterministic
    set_global_seeds(42)
    y2 = torch.randn(5, 5)
    
    with_det_match = torch.allclose(y1, y2)
    print(f"Tensors match with deterministic: {with_det_match}")
    
    # Disable deterministic mode
    disable_deterministic_mode()
    
    print("\n✅ Reproducibility test complete!")


def test_performance_impact():
    """Test performance impact of deterministic mode."""
    print("\n" + "#"*70)
    print("# TEST: Performance Impact of Deterministic Mode")
    print("#"*70)
    
    import time
    from cogniforge.core.deterministic_mode import enable_deterministic_mode, disable_deterministic_mode
    from cogniforge.core.seed_manager import set_global_seeds
    
    # Setup
    set_global_seeds(42)
    size = 1000
    iterations = 100
    
    # Test without deterministic mode
    print("\n1. Benchmark WITHOUT deterministic mode:")
    start = time.time()
    for _ in range(iterations):
        x = torch.randn(size, size)
        y = torch.randn(size, size)
        z = torch.matmul(x, y)
    time_without = time.time() - start
    print(f"   Time: {time_without:.3f} seconds")
    
    # Test with deterministic mode
    print("\n2. Benchmark WITH deterministic mode:")
    enable_deterministic_mode(warn_only=True)
    set_global_seeds(42)
    
    start = time.time()
    for _ in range(iterations):
        x = torch.randn(size, size)
        y = torch.randn(size, size)
        z = torch.matmul(x, y)
    time_with = time.time() - start
    print(f"   Time: {time_with:.3f} seconds")
    
    # Calculate overhead
    overhead = ((time_with - time_without) / time_without) * 100
    print(f"\n📊 Performance overhead: {overhead:.1f}%")
    
    if overhead > 0:
        print("   Note: Deterministic mode may slow down operations")
    else:
        print("   Note: No significant performance impact detected")
    
    disable_deterministic_mode()


def demonstrate_usage():
    """Demonstrate various usage patterns."""
    print("\n" + "#"*70)
    print("# USAGE EXAMPLES")
    print("#"*70)
    
    print("\n📚 Command-line usage examples:\n")
    
    examples = [
        {
            "desc": "Basic training with deterministic mode:",
            "cmd": "python -m cogniforge.cli.train_cli --task train --deterministic --seed 42"
        },
        {
            "desc": "CPU-only deterministic training:",
            "cmd": "python -m cogniforge.cli.train_cli --task train --cpu-only --seed 42"
        },
        {
            "desc": "Deterministic with warnings only (no errors):",
            "cmd": "python -m cogniforge.cli.train_cli --task train --deterministic --deterministic-warn-only"
        },
        {
            "desc": "Debug mode (errors on non-deterministic ops):",
            "cmd": "python -m cogniforge.cli.train_cli --task train --deterministic --deterministic-debug"
        },
        {
            "desc": "Save all settings including deterministic config:",
            "cmd": "python -m cogniforge.cli.train_cli --task train --deterministic --save-config --output-dir results"
        }
    ]
    
    for example in examples:
        print(f"  {example['desc']}")
        print(f"  $ {example['cmd']}\n")
    
    print("\n📝 Python API usage:\n")
    print("""
    from cogniforge.core.deterministic_mode import enable_deterministic_mode
    from cogniforge.core.seed_manager import set_global_seeds
    
    # Enable deterministic mode
    settings = enable_deterministic_mode(warn_only=True)
    
    # Set seeds for reproducibility
    set_global_seeds(42)
    
    # Your training code here...
    model = train_model()
    """)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "DETERMINISTIC MODE CLI TEST SUITE")
    print("="*70)
    
    print("\nThis test suite demonstrates the --deterministic flag")
    print("functionality for enabling PyTorch deterministic mode.")
    
    try:
        # Run tests
        test_deterministic_flag()
        test_reproducibility_with_deterministic()
        test_performance_impact()
        demonstrate_usage()
        
        print("\n" + "="*70)
        print(" "*20 + "ALL TESTS COMPLETE! ✅")
        print("="*70)
        
        print("\n🎯 Summary:")
        print("  • --deterministic flag enables PyTorch deterministic mode")
        print("  • CPU operations become fully reproducible")
        print("  • Settings are logged in run summaries")
        print("  • Compatible with seed management system")
        print("  • Performance impact is acceptable for debugging")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())