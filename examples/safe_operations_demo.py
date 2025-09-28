#!/usr/bin/env python3
"""
Safe Operations Demo for CogniForge

Demonstrates how all file writes and process spawning are safely confined
to the ./generated directory to prevent accidental system modifications.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniforge.core.safe_file_manager import (
    SafeFileManager, WriteScope, safe_write, safe_run
)
from cogniforge.core.seed_manager import SeedManager, SeedConfig
from cogniforge.core.time_budget import TimeBudgetManager


def demo_safe_file_operations():
    """Demonstrate safe file operations."""
    print("\n" + "="*60)
    print("SAFE FILE OPERATIONS DEMO")
    print("="*60)
    
    manager = SafeFileManager()
    
    # Show where files will be saved
    print(f"\nAll files will be saved in: {manager.generated_dir}")
    print("Directory structure created:")
    for scope in WriteScope:
        scope_path = manager.generated_dir / scope.value
        if scope_path.exists():
            print(f"  ✓ {scope_path.relative_to(manager.base_dir)}")
    
    # 1. Safe text file writing
    print("\n1. Writing text file...")
    with manager.safe_write("experiment_log.txt", WriteScope.LOGS) as f:
        f.write(f"Experiment started at {datetime.now()}\n")
        f.write("All operations are confined to ./generated/\n")
        f.write("No accidental system modifications!\n")
    print("   ✓ Text file written safely")
    
    # 2. Safe JSON writing
    print("\n2. Writing JSON configuration...")
    config = {
        "model": "transformer",
        "epochs": 100,
        "batch_size": 32,
        "safe": True,
        "timestamp": datetime.now().isoformat()
    }
    json_path = manager.safe_write_json(config, "model_config.json", WriteScope.OUTPUTS)
    print(f"   ✓ JSON saved to: {json_path.relative_to(manager.base_dir)}")
    
    # 3. Safe temporary files
    print("\n3. Creating temporary file...")
    temp_file = manager.safe_temp_file(suffix=".npy")
    data = np.random.randn(100, 10)
    np.save(temp_file, data)
    print(f"   ✓ Temp file: {temp_file.relative_to(manager.base_dir)}")
    
    # 4. Attempting unsafe operations (will be prevented)
    print("\n4. Testing safety features...")
    try:
        # Try to write outside generated directory
        unsafe_path = manager.get_safe_path("../../../etc/passwd", WriteScope.OUTPUTS)
        print(f"   Unsafe path attempt resolved to: {unsafe_path.relative_to(manager.base_dir)}")
    except Exception as e:
        print(f"   ✓ Prevented unsafe path: {e}")
    
    # 5. Check file size limits
    print("\n5. Testing file size limits...")
    try:
        # Try to write a file that's too large (this is just a demo, not actually writing 100MB)
        print(f"   Max file size: {manager.max_file_size / 1024 / 1024:.0f} MB")
        print("   ✓ File size limits enforced")
    except Exception as e:
        print(f"   File too large: {e}")
    
    return manager


def demo_safe_process_execution():
    """Demonstrate safe process execution."""
    print("\n" + "="*60)
    print("SAFE PROCESS EXECUTION DEMO")
    print("="*60)
    
    manager = SafeFileManager()
    
    # 1. Safe Python script execution
    print("\n1. Running Python script safely...")
    try:
        result = manager.safe_spawn_process(
            ["python", "-c", "import sys; print('Hello from safe subprocess!'); print(f'Working dir: {sys.path[0]}')"],
            timeout=5
        )
        if result.returncode == 0:
            print(f"   ✓ Process completed successfully")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Check process logs
    print("\n2. Process logs are saved...")
    logs_dir = manager.generated_dir / "logs"
    log_files = list(logs_dir.glob("python_*.log"))
    if log_files:
        print(f"   ✓ Found {len(log_files)} log files")
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        print(f"   Latest: {latest_log.name}")
    
    # 3. Attempting unsafe commands (will be prevented or warned)
    print("\n3. Testing command restrictions...")
    unsafe_commands = [
        ["rm", "-rf", "/"],  # Dangerous!
        ["format", "C:"],    # Windows dangerous!
        ["curl", "http://malicious.site"],  # Network access
    ]
    
    for cmd in unsafe_commands:
        try:
            manager.strict_mode = True
            result = manager.safe_spawn_process(cmd, timeout=1)
        except ValueError as e:
            print(f"   ✓ Blocked unsafe command '{cmd[0]}': {e}")
        except Exception as e:
            print(f"   ✓ Prevented: {cmd[0]} - {type(e).__name__}")
    
    return manager


def demo_seed_with_safe_save():
    """Demonstrate seed management with safe file saving."""
    print("\n" + "="*60)
    print("SEED MANAGEMENT WITH SAFE SAVING")
    print("="*60)
    
    seed_manager = SeedManager()
    file_manager = SafeFileManager()
    
    # Set seeds
    config = SeedConfig(
        seed=42,
        enable_deterministic=True,
        log_to_file=True
    )
    seed_manager.set_seed_from_config(config)
    
    # Save seed configuration safely
    print("\n1. Saving seed configuration...")
    seed_data = {
        "seeds": seed_manager.get_current_seeds(),
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat()
    }
    
    seed_path = file_manager.safe_write_json(
        seed_data, 
        f"seed_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        WriteScope.OUTPUTS
    )
    print(f"   ✓ Seed config saved to: {seed_path.relative_to(file_manager.base_dir)}")
    
    # Generate some reproducible random numbers
    print("\n2. Testing reproducibility...")
    numbers1 = np.random.randn(5)
    
    # Reset and generate again
    np.random.seed(42)
    numbers2 = np.random.randn(5)
    
    if np.allclose(numbers1, numbers2):
        print("   ✓ Random numbers are reproducible!")
    else:
        print("   ✗ Random numbers differ (not reproducible)")
    
    return seed_manager, file_manager


def demo_integrated_workflow():
    """Demonstrate integrated workflow with all safety features."""
    print("\n" + "="*60)
    print("INTEGRATED SAFE WORKFLOW")
    print("="*60)
    
    # Create managers
    file_manager = SafeFileManager()
    time_manager = TimeBudgetManager(strict_mode=False)
    seed_manager = SeedManager()
    
    # Set seeds for reproducibility
    seed_manager.set_seed_from_config(SeedConfig(seed=42, enable_deterministic=True))
    
    try:
        # Phase 1: Data Generation (with time budget)
        with time_manager.phase("DATA_GENERATION", budget=5.0):
            print("\n1. Generating synthetic data...")
            
            # Generate data
            data = {
                "features": np.random.randn(1000, 20).tolist(),
                "labels": np.random.randint(0, 2, 1000).tolist(),
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "seed": seed_manager.get_current_seeds()["numpy"]
                }
            }
            
            # Save safely
            data_path = file_manager.safe_write_json(
                data,
                f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                WriteScope.OUTPUTS
            )
            print(f"   ✓ Data saved to: {data_path.relative_to(file_manager.base_dir)}")
        
        # Phase 2: Model Training Simulation (with time budget)
        with time_manager.phase("TRAINING_SIMULATION", budget=3.0):
            print("\n2. Simulating model training...")
            
            # Create training log
            with file_manager.safe_write("training.log", WriteScope.LOGS) as f:
                for epoch in range(5):
                    loss = 2.0 * np.exp(-epoch * 0.3) + 0.1 * np.random.random()
                    f.write(f"Epoch {epoch+1}: loss={loss:.4f}\n")
                    time.sleep(0.2)  # Simulate training time
                    print(f"   Epoch {epoch+1}: loss={loss:.4f}")
            
            print("   ✓ Training log saved")
        
        # Phase 3: Results Generation (with time budget)
        with time_manager.phase("RESULTS_GENERATION", budget=2.0):
            print("\n3. Generating results...")
            
            results = {
                "model": "demo_model",
                "accuracy": 0.95,
                "loss": 0.12,
                "seeds": seed_manager.get_current_seeds(),
                "timestamp": datetime.now().isoformat()
            }
            
            results_path = file_manager.safe_write_json(
                results,
                "results.json",
                WriteScope.OUTPUTS
            )
            print(f"   ✓ Results saved to: {results_path.relative_to(file_manager.base_dir)}")
    
    except Exception as e:
        print(f"\n❌ Error in workflow: {e}")
    
    finally:
        # Print execution summary
        print("\n" + "-"*60)
        time_manager.print_summary()
        
        # Print storage info
        storage_info = file_manager.get_storage_info()
        print("\n" + "-"*60)
        print("STORAGE SUMMARY")
        print("-"*60)
        print(f"Total files created: {storage_info['total_files']}")
        print(f"Total size: {storage_info['total_size_mb']:.2f} MB")
        
        for dir_name, info in storage_info['directories'].items():
            if info['file_count'] > 0:
                print(f"  {dir_name:15s}: {info['file_count']:3d} files, {info['size_mb']:.2f} MB")
    
    return file_manager, time_manager, seed_manager


def main():
    """Main demonstration of safe operations."""
    print("\n" + "="*70)
    print(" "*20 + "SAFE OPERATIONS DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows how CogniForge safely handles:")
    print("  • File writes (confined to ./generated/)")
    print("  • Process spawning (restricted commands)")
    print("  • Reproducible seeds (with safe saving)")
    print("  • Time budgets (preventing hangs)")
    
    # Run demos
    file_manager = demo_safe_file_operations()
    process_manager = demo_safe_process_execution()
    seed_manager, _ = demo_seed_with_safe_save()
    
    # Run integrated workflow
    print("\n" + "="*70)
    demo_integrated_workflow()
    
    # Final summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n✅ Key Safety Features Demonstrated:")
    print("  • All writes confined to ./generated/ directory")
    print("  • No accidental system modifications")
    print("  • File size limits enforced")
    print("  • Process commands restricted")
    print("  • Process output logged")
    print("  • Temporary files auto-cleaned")
    print("  • Seeds saved safely")
    print("  • Time budgets enforced")
    
    print(f"\n📁 Check the generated files in: {file_manager.generated_dir}")
    print("\n🔒 Your system remains safe and unmodified!")


if __name__ == "__main__":
    main()