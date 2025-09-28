#!/usr/bin/env python3
"""
Example of using Time Budget Manager in CogniForge

Demonstrates how to enforce time limits for different phases with graceful
abort and helpful error messages.
"""

import sys
import time
import random
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniforge.core.time_budget import (
    TimeBudgetManager, TimeoutException, Phase
)
from cogniforge.core.seed_manager import SeedManager, SeedConfig


def cleanup_training():
    """Cleanup handler for training phase."""
    print("🧹 Cleaning up training resources...")
    print("   - Saving checkpoint...")
    print("   - Releasing GPU memory...")
    print("   - Closing data loaders...")
    print("✅ Cleanup complete!")


def simulate_data_loading(manager: TimeBudgetManager):
    """Simulate data loading with time budget."""
    print("\n" + "="*60)
    print("📊 DATA LOADING PHASE")
    print("="*60)
    
    try:
        with manager.phase("DATA_LOADING", budget=5.0) as phase_stats:
            print("Loading dataset...")
            
            # Simulate loading batches
            num_batches = 10
            for i in range(num_batches):
                # Check remaining time
                remaining = phase_stats.budget - phase_stats.elapsed
                if remaining < 1.0:
                    print(f"⚠️  Low on time! Stopping at batch {i}/{num_batches}")
                    break
                
                print(f"  Loading batch {i+1}/{num_batches}...")
                time.sleep(0.4)  # Simulate I/O
            
            print("✅ Data loading complete!")
            
    except TimeoutException as e:
        print(e)
        return False
    
    return True


def simulate_training(manager: TimeBudgetManager, epochs: int = 5):
    """Simulate model training with per-epoch time budgets."""
    print("\n" + "="*60)
    print("🤖 TRAINING PHASE")
    print("="*60)
    
    # Register cleanup handler
    manager.register_abort_handler("TRAINING", cleanup_training)
    
    losses = []
    
    for epoch in range(epochs):
        try:
            # Each epoch has its own budget
            with manager.phase(f"EPOCH_{epoch+1}", budget=3.0) as phase_stats:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print("-" * 30)
                
                # Training steps
                steps = 20
                for step in range(steps):
                    # Check if we're running low on time
                    if phase_stats.elapsed > phase_stats.budget * 0.8:
                        print(f"⚠️  Running low on time, completing epoch early")
                        break
                    
                    # Simulate training step
                    loss = 2.0 * np.exp(-(epoch * steps + step) * 0.01) + 0.1 * random.random()
                    
                    if step % 5 == 0:
                        print(f"  Step {step:3d}: loss = {loss:.4f}")
                    
                    time.sleep(0.1)  # Simulate computation
                
                losses.append(loss)
                print(f"✅ Epoch {epoch+1} complete: final loss = {loss:.4f}")
                
        except TimeoutException as e:
            print(e)
            print(f"❌ Training aborted at epoch {epoch+1}")
            break
    
    return losses


def simulate_code_generation(manager: TimeBudgetManager, task: str):
    """Simulate code generation with time budget."""
    print("\n" + "="*60)
    print("💻 CODE GENERATION PHASE")
    print("="*60)
    print(f"Task: {task}")
    
    try:
        with manager.phase("CODE_GENERATION", budget=10.0) as phase_stats:
            # Step 1: Parse task
            print("  1. Parsing task description...")
            time.sleep(1.0)
            
            # Step 2: Generate code
            print("  2. Generating code...")
            time.sleep(2.0)
            
            # Step 3: Validate
            print("  3. Validating generated code...")
            time.sleep(1.0)
            
            # Step 4: Optimize
            if phase_stats.elapsed < phase_stats.budget * 0.7:
                print("  4. Optimizing code...")
                time.sleep(1.5)
            else:
                print("  4. Skipping optimization (low on time)")
            
            code = f"""
def {task.replace(' ', '_')}():
    # Generated code for: {task}
    print("Executing task...")
    return True
"""
            print("✅ Code generation complete!")
            return code
            
    except TimeoutException as e:
        print(e)
        return None


def simulate_simulation(manager: TimeBudgetManager):
    """Simulate PyBullet simulation with per-step budgets."""
    print("\n" + "="*60)
    print("🎮 SIMULATION PHASE")
    print("="*60)
    
    try:
        # Initialization phase
        with manager.phase("SIMULATION_INIT", budget=5.0):
            print("Initializing simulation...")
            print("  - Loading physics engine...")
            time.sleep(1.0)
            print("  - Loading robot model...")
            time.sleep(1.0)
            print("  - Setting up environment...")
            time.sleep(0.5)
            print("✅ Simulation initialized!")
        
        # Simulation loop with strict per-step budget
        num_steps = 100
        print(f"\nRunning {num_steps} simulation steps...")
        
        completed_steps = 0
        for step in range(num_steps):
            try:
                # Very tight budget per step (50ms)
                with manager.phase(f"SIM_STEP_{step}", budget=0.05):
                    # Simulate physics computation
                    time.sleep(0.03 + random.random() * 0.03)
                    completed_steps += 1
                    
                    if step % 20 == 0:
                        print(f"  Step {step:3d}/100")
                        
            except TimeoutException:
                print(f"⚠️  Step {step} exceeded budget, continuing...")
                continue
        
        print(f"✅ Simulation complete: {completed_steps}/{num_steps} steps")
        
    except TimeoutException as e:
        print(e)
        return False
    
    return True


@TimeBudgetManager().timeout(5.0)
def slow_api_call(endpoint: str):
    """Simulate an API call with automatic timeout."""
    print(f"Calling API: {endpoint}")
    time.sleep(2.0)  # Simulate network delay
    return {"status": "success", "data": "response"}


def main():
    """Main example demonstrating time budget management."""
    print("\n" + "="*70)
    print(" "*20 + "TIME BUDGET MANAGER EXAMPLE")
    print("="*70)
    
    # Create manager with strict mode (will abort on timeout)
    manager = TimeBudgetManager(strict_mode=True, global_timeout=60.0)
    
    # Start global timer
    manager.start_global_timer()
    
    # Configure custom budgets
    manager.set_phase_budget("DATA_LOADING", 5.0)
    manager.set_phase_budget("CODE_GENERATION", 10.0)
    
    # Initialize seed for reproducibility
    seed_mgr = SeedManager()
    seed_mgr.set_seed_from_config(SeedConfig(seed=42, enable_deterministic=True))
    
    try:
        # 1. Data Loading Phase
        success = simulate_data_loading(manager)
        if not success:
            print("⚠️  Data loading failed, but continuing...")
        
        # 2. Training Phase
        losses = simulate_training(manager, epochs=3)
        if losses:
            print(f"\n📈 Training results: Final loss = {losses[-1]:.4f}")
        
        # 3. Code Generation Phase
        code = simulate_code_generation(manager, "pick up red cube")
        if code:
            print(f"\n📝 Generated code preview:")
            print(code[:200] + "...")
        
        # 4. Simulation Phase
        # Switch to non-strict mode for simulation
        manager.strict_mode = False
        simulate_simulation(manager)
        
        # 5. API Call with decorator
        print("\n" + "="*60)
        print("🌐 API CALL PHASE")
        print("="*60)
        try:
            result = slow_api_call("https://api.example.com/data")
            print(f"✅ API Response: {result}")
        except TimeoutException as e:
            print(f"❌ API call timed out")
        
    except TimeoutException as e:
        print(f"\n❌ FATAL: {e}")
    
    finally:
        # Print execution summary
        print("\n" + "="*70)
        manager.print_summary()
        
        # Get detailed statistics
        summary = manager.get_summary()
        
        print("\n📊 EFFICIENCY ANALYSIS")
        print("="*70)
        
        # Find most and least efficient phases
        if summary['phases']:
            efficiencies = [
                (p['phase'], p['budget']/p['elapsed']*100 if p['elapsed'] > 0 else 0)
                for p in summary['phases']
                if p['completed']
            ]
            
            if efficiencies:
                efficiencies.sort(key=lambda x: x[1], reverse=True)
                
                print("\n🏆 Most Efficient Phases:")
                for phase, eff in efficiencies[:3]:
                    print(f"  {phase:30s} {eff:5.1f}% efficiency")
                
                if len(efficiencies) > 3:
                    print("\n⚠️  Least Efficient Phases:")
                    for phase, eff in efficiencies[-3:]:
                        print(f"  {phase:30s} {eff:5.1f}% efficiency")
        
        print("\n💡 RECOMMENDATIONS")
        print("="*70)
        
        # Provide recommendations based on results
        aborted = summary['aborted_phases']
        if aborted > 0:
            print(f"⚠️  {aborted} phases were aborted due to timeout")
            print("   Consider:")
            print("   - Increasing time budgets for critical phases")
            print("   - Optimizing slow operations")
            print("   - Using async/parallel processing")
        else:
            print("✅ All phases completed within budget!")
        
        total_time = summary['total_elapsed']
        if total_time > 30:
            print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
            print("   Consider caching results for faster subsequent runs")


if __name__ == "__main__":
    main()