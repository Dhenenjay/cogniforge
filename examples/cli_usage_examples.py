#!/usr/bin/env python3
"""
CogniForge CLI Usage Examples

This script demonstrates how to use the cogv command-line interface
to interact with CogniForge programmatically or from the command line.
"""

import subprocess
import sys
import json
from pathlib import Path


def run_training_with_deterministic():
    """Example: Train a model with deterministic mode enabled."""
    print("\n" + "="*60)
    print("EXAMPLE: Training with Deterministic Mode")
    print("="*60)
    
    # Command to run training with deterministic mode
    cmd = [
        "cogv", "run", "train",
        "--epochs", "10",
        "--batch-size", "32",
        "--lr", "0.001",
        "--seed", "42",
        "--deterministic",  # Enable deterministic mode
        "--output-dir", "outputs/deterministic_run",
        "--log-metrics"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis command will:")
    print("  • Set a fixed seed (42) for reproducibility")
    print("  • Enable PyTorch deterministic mode")
    print("  • Log all metrics and configurations")
    print("  • Save outputs to 'outputs/deterministic_run'")
    
    # Uncomment to actually run:
    # subprocess.run(cmd)


def run_demo_with_visualization():
    """Example: Run an interactive grasp demo with visualization."""
    print("\n" + "="*60)
    print("EXAMPLE: Interactive Grasp Demo")
    print("="*60)
    
    cmd = [
        "cogv", "demo", "grasp",
        "--object", "red_cube",
        "--robot", "franka",
        "--environment", "tabletop",
        "--visualize",      # Enable PyBullet GUI
        "--interactive",    # Allow user interaction
        "--speed", "0.5"    # Slow down for visibility
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis command will:")
    print("  • Launch PyBullet GUI with a Franka robot")
    print("  • Create a tabletop environment with a red cube")
    print("  • Enable interactive control")
    print("  • Run at half speed for better visualization")
    
    # Uncomment to run:
    # subprocess.run(cmd)


def generate_code_with_validation():
    """Example: Generate code from natural language with validation."""
    print("\n" + "="*60)
    print("EXAMPLE: Code Generation with Validation")
    print("="*60)
    
    task_description = "pick up the blue cube and stack it on the red cube"
    
    cmd = [
        "cogv", "gen-code",
        task_description,
        "--framework", "pybullet",
        "--model", "gpt-4",
        "--style", "educational",
        "--include-tests",
        "--include-docs",
        "--validate",       # Validate generated code
        "--safety-check",   # Perform safety checks
        "--output-file", "generated_stacking.py"
    ]
    
    print(f"Command: cogv gen-code \"{task_description}\" ...")
    print("\nThis command will:")
    print("  • Use GPT-4 to generate PyBullet code")
    print("  • Include educational comments")
    print("  • Generate unit tests")
    print("  • Add documentation")
    print("  • Validate the code syntax")
    print("  • Perform safety checks")
    print("  • Save to 'generated_stacking.py'")
    
    # Uncomment to run:
    # subprocess.run(cmd)


def batch_experiments():
    """Example: Run multiple experiments with different seeds."""
    print("\n" + "="*60)
    print("EXAMPLE: Batch Experiments")
    print("="*60)
    
    seeds = [42, 123, 456, 789, 1000]
    
    for seed in seeds:
        cmd = [
            "cogv", "run", "train",
            "--epochs", "50",
            "--seed", str(seed),
            "--deterministic",
            "--output-dir", f"experiments/seed_{seed}",
            "--quiet"  # Suppress output for batch runs
        ]
        
        print(f"Running experiment with seed {seed}...")
        # Uncomment to run:
        # subprocess.run(cmd)
    
    print("\nBatch experiments setup:")
    print(f"  • Running {len(seeds)} experiments")
    print("  • Each with a different seed for statistical analysis")
    print("  • Results saved in separate directories")


def pipeline_with_config():
    """Example: Run a full pipeline with configuration file."""
    print("\n" + "="*60)
    print("EXAMPLE: Pipeline with Configuration File")
    print("="*60)
    
    # Create a sample config file
    config = {
        "model": {
            "type": "transformer",
            "hidden_dim": 256,
            "num_layers": 6
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": 100
        },
        "data": {
            "dataset": "pick_place",
            "augmentation": True
        }
    }
    
    config_path = Path("config_example.yaml")
    
    print("Sample configuration:")
    print(json.dumps(config, indent=2))
    
    cmd = [
        "cogv", "run", "full",
        "--config", str(config_path),
        "--task", "pick_place",
        "--deterministic",
        "--save-checkpoints",
        "--log-metrics"
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("\nThis command will:")
    print("  • Load configuration from YAML file")
    print("  • Run the full training pipeline")
    print("  • Save checkpoints during training")
    print("  • Log all metrics")


def advanced_code_generation():
    """Example: Advanced code generation with multiple constraints."""
    print("\n" + "="*60)
    print("EXAMPLE: Advanced Code Generation")
    print("="*60)
    
    task = "navigate through a maze avoiding obstacles and reach the goal"
    
    cmd = [
        "cogv", "gen-code",
        task,
        "--framework", "ros",
        "--language", "python",
        "--style", "modular",
        "--model", "gpt-4",
        "--include-tests",
        "--include-docs",
        "--validate",
        "--safety-check",
        "--max-length", "500",
        "--output-file", "navigation_controller.py"
    ]
    
    print(f"Task: \"{task}\"")
    print("\nGeneration parameters:")
    print("  • Framework: ROS (Robot Operating System)")
    print("  • Style: Modular (separated into functions/classes)")
    print("  • Includes: Tests, documentation, validation, safety")
    print("  • Max length: 500 lines")


def combining_commands():
    """Example: Combining multiple commands in a workflow."""
    print("\n" + "="*60)
    print("EXAMPLE: Complete Workflow")
    print("="*60)
    
    workflow = [
        {
            "desc": "Step 1: Generate code for the task",
            "cmd": ["cogv", "gen-code", "grasp red cube", "--framework", "pybullet", 
                   "--output-file", "grasp_task.py"]
        },
        {
            "desc": "Step 2: Test the generated code",
            "cmd": ["cogv", "demo", "grasp", "--script", "grasp_task.py", 
                   "--visualize"]
        },
        {
            "desc": "Step 3: Train a policy using BC",
            "cmd": ["cogv", "run", "bc", "--demos", "grasp_demos.pkl", 
                   "--epochs", "100", "--deterministic"]
        },
        {
            "desc": "Step 4: Evaluate the trained policy",
            "cmd": ["cogv", "run", "eval", "--checkpoint", "bc_model.pth", 
                   "--task", "grasp"]
        }
    ]
    
    print("Complete workflow:")
    for i, step in enumerate(workflow, 1):
        print(f"\n{step['desc']}")
        print(f"  $ {' '.join(step['cmd'])}")


def parallel_execution():
    """Example: Running commands in parallel."""
    print("\n" + "="*60)
    print("EXAMPLE: Parallel Execution")
    print("="*60)
    
    print("Running multiple demos in parallel:\n")
    
    demos = [
        ["cogv", "demo", "grasp", "--object", "cube"],
        ["cogv", "demo", "navigation", "--map", "maze"],
        ["cogv", "demo", "manipulation", "--task", "stack"],
        ["cogv", "demo", "planning", "--goal", "sort_objects"]
    ]
    
    print("Commands to run in parallel:")
    for cmd in demos:
        print(f"  $ {' '.join(cmd)} &")
    
    print("\nNote: Add '&' at the end to run in background (Unix/Linux)")
    print("      Or use 'Start-Process' in PowerShell for parallel execution")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "COGNIFORGE CLI USAGE EXAMPLES")
    print("="*70)
    
    print("\nThese examples demonstrate various ways to use the cogv CLI.")
    print("Uncomment the subprocess.run() lines to actually execute commands.\n")
    
    # Run all example functions
    run_training_with_deterministic()
    run_demo_with_visualization()
    generate_code_with_validation()
    batch_experiments()
    pipeline_with_config()
    advanced_code_generation()
    combining_commands()
    parallel_execution()
    
    print("\n" + "="*70)
    print("KEY FEATURES DEMONSTRATED:")
    print("="*70)
    print("""
✅ Deterministic Training
   - Fixed seeds for reproducibility
   - PyTorch deterministic mode
   
✅ Interactive Demos
   - Visualization with PyBullet
   - Interactive control
   
✅ Code Generation
   - Natural language to code
   - Multiple frameworks supported
   - Validation and safety checks
   
✅ Batch Processing
   - Multiple experiments
   - Different configurations
   
✅ Complex Workflows
   - Chaining commands
   - Complete pipelines
   
✅ Parallel Execution
   - Running multiple tasks
   - Efficient resource usage
""")
    
    print("\n🚀 Quick Start:")
    print("  1. Install: pip install -e .")
    print("  2. Test: cogv --help")
    print("  3. Run: cogv run train --deterministic")
    
    print("\n📚 Documentation:")
    print("  • Use --help with any command")
    print("  • Check examples/ directory for more scripts")
    print("  • Read docs/ for detailed guides")


if __name__ == "__main__":
    main()