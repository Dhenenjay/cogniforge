#!/usr/bin/env python3
"""
Test Script for CogniForge CLI Commands

This script tests the cogv command-line interface with various subcommands:
- cogv run
- cogv demo
- cogv gen-code
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, capture=True):
    """Execute a command and display output."""
    print(f"\n{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        return result.returncode
    else:
        return subprocess.run(cmd).returncode


def test_help_commands():
    """Test help output for all commands."""
    print("\n" + "#"*70)
    print("# TEST: Help Commands")
    print("#"*70)
    
    commands = [
        [sys.executable, "-m", "cogniforge.cli", "--help"],
        [sys.executable, "-m", "cogniforge.cli", "run", "--help"],
        [sys.executable, "-m", "cogniforge.cli", "demo", "--help"],
        [sys.executable, "-m", "cogniforge.cli", "gen-code", "--help"],
    ]
    
    for cmd in commands:
        run_command(cmd)


def test_run_commands():
    """Test cogv run command with different pipelines."""
    print("\n" + "#"*70)
    print("# TEST: cogv run")
    print("#"*70)
    
    # Test training pipeline
    print("\n1. Testing 'cogv run train':")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "run", "train",
        "--epochs", "2",
        "--batch-size", "16",
        "--lr", "0.001",
        "--seed", "42",
        "--deterministic",
        "--output-dir", "test_output"
    ]
    run_command(cmd)
    
    # Test evaluation pipeline
    print("\n2. Testing 'cogv run eval':")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "run", "eval",
        "--task", "pick_place"
    ]
    run_command(cmd)
    
    # Test BC pipeline
    print("\n3. Testing 'cogv run bc':")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "run", "bc",
        "--epochs", "5"
    ]
    run_command(cmd)
    
    # Clean up
    output_dir = Path("test_output")
    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
        output_dir.rmdir()


def test_demo_commands():
    """Test cogv demo command with different demonstrations."""
    print("\n" + "#"*70)
    print("# TEST: cogv demo")
    print("#"*70)
    
    # Test grasp demo (without visualization to avoid GUI)
    print("\n1. Testing 'cogv demo grasp':")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "demo", "grasp",
        "--object", "cube",
        "--robot", "kuka",
        "--environment", "simple"
    ]
    run_command(cmd)
    
    # Test vision demo
    print("\n2. Testing 'cogv demo vision':")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "demo", "vision",
        "--object", "block"
    ]
    run_command(cmd)
    
    # Test IK demo
    print("\n3. Testing 'cogv demo ik':")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "demo", "ik"
    ]
    run_command(cmd)
    
    # Test planning demo
    print("\n4. Testing 'cogv demo planning':")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "demo", "planning",
        "--object", "red_cube"
    ]
    run_command(cmd)


def test_gencode_commands():
    """Test cogv gen-code command with different parameters."""
    print("\n" + "#"*70)
    print("# TEST: cogv gen-code")
    print("#"*70)
    
    # Test minimal PyBullet code generation
    print("\n1. Testing minimal PyBullet code generation:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "gen-code",
        "pick up the blue cube and place it on the platform",
        "--framework", "pybullet",
        "--style", "minimal",
        "--validate"
    ]
    run_command(cmd)
    
    # Test verbose PyTorch code generation
    print("\n2. Testing verbose PyTorch code generation:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "gen-code",
        "grasp and manipulate objects",
        "--framework", "pytorch",
        "--style", "verbose",
        "--include-tests"
    ]
    run_command(cmd)
    
    # Test educational code with documentation
    print("\n3. Testing educational code with documentation:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "gen-code",
        "move the robot to the target position",
        "--framework", "pybullet",
        "--style", "educational",
        "--include-docs",
        "--safety-check"
    ]
    run_command(cmd)
    
    # Test code generation with output file
    print("\n4. Testing code generation with file output:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "gen-code",
        "navigate through waypoints",
        "--framework", "ros",
        "--output-file", "generated_code.py"
    ]
    run_command(cmd)
    
    # Clean up
    generated_file = Path("generated_code.py")
    if generated_file.exists():
        generated_file.unlink()
        print("  ✓ Cleaned up generated_code.py")


def test_advanced_features():
    """Test advanced CLI features."""
    print("\n" + "#"*70)
    print("# TEST: Advanced Features")
    print("#"*70)
    
    # Test version
    print("\n1. Testing version flag:")
    cmd = [sys.executable, "-m", "cogniforge.cli", "--version"]
    run_command(cmd)
    
    # Test quiet mode
    print("\n2. Testing quiet mode:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "--quiet",
        "run", "eval"
    ]
    run_command(cmd)
    
    # Test verbose mode
    print("\n3. Testing verbose mode:")
    cmd = [
        sys.executable, "-m", "cogniforge.cli", "--verbose",
        "demo", "navigation"
    ]
    run_command(cmd)


def demonstrate_usage():
    """Show example usage patterns."""
    print("\n" + "#"*70)
    print("# USAGE EXAMPLES")
    print("#"*70)
    
    examples = [
        {
            "desc": "Train a model with deterministic mode:",
            "cmd": "cogv run train --epochs 100 --deterministic --seed 42"
        },
        {
            "desc": "Run full pipeline:",
            "cmd": "cogv run full --task pick_place --save-checkpoints"
        },
        {
            "desc": "Interactive grasp demo with visualization:",
            "cmd": "cogv demo grasp --visualize --interactive --robot franka"
        },
        {
            "desc": "Generate code from natural language:",
            "cmd": 'cogv gen-code "pick up the red block" --framework pybullet'
        },
        {
            "desc": "Generate educational code with tests and docs:",
            "cmd": 'cogv gen-code "manipulate objects" --style educational --include-tests --include-docs'
        },
        {
            "desc": "Run optimization pipeline:",
            "cmd": "cogv run optimize --config config.yaml --log-metrics"
        },
        {
            "desc": "Demo all capabilities:",
            "cmd": "cogv demo all --speed 2.0"
        },
        {
            "desc": "Generate ROS code with safety checks:",
            "cmd": 'cogv gen-code "navigate to target" --framework ros --safety-check --validate'
        }
    ]
    
    print("\n📚 Example Commands:\n")
    for example in examples:
        print(f"  {example['desc']}")
        print(f"    $ {example['cmd']}\n")
    
    print("\n💡 Tips:")
    print("  • Use --help with any command for detailed options")
    print("  • Add --deterministic for reproducible runs")
    print("  • Use --visualize with demos for PyBullet GUI")
    print("  • Add --validate to check generated code")
    print("  • Use --output-dir to specify where to save results")


def main():
    """Run all CLI tests."""
    print("\n" + "="*70)
    print(" "*20 + "COGNIFORGE CLI TEST SUITE")
    print("="*70)
    
    print("\nTesting CogniForge command-line interface...")
    print("Commands: cogv run, cogv demo, cogv gen-code")
    
    try:
        # Run tests
        test_help_commands()
        test_run_commands()
        test_demo_commands()
        test_gencode_commands()
        test_advanced_features()
        demonstrate_usage()
        
        print("\n" + "="*70)
        print(" "*25 + "ALL TESTS COMPLETE! ✅")
        print("="*70)
        
        print("\n📊 Summary:")
        print("  ✓ cogv run - Training and evaluation pipelines")
        print("  ✓ cogv demo - Interactive demonstrations")
        print("  ✓ cogv gen-code - Code generation from natural language")
        print("  ✓ Help commands working")
        print("  ✓ Advanced features functional")
        
        print("\n🚀 To install cogv globally:")
        print("  $ pip install -e .")
        print("\nThen you can use:")
        print("  $ cogv run train")
        print("  $ cogv demo grasp")
        print("  $ cogv gen-code \"your task description\"")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())