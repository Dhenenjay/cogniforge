#!/usr/bin/env python3
"""
CMA-ES Sparkline Integration Example

This example demonstrates how to integrate the sparkline visualization
with your CMA-ES optimization loop for real-time progress monitoring.
"""

import numpy as np
import json
from pathlib import Path
from cogniforge.ui.cmaes_visualizer import CMAESVisualizer, CMAESMetrics

def objective_function(x):
    """Example objective function (Rosenbrock function)."""
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def simple_cmaes_with_visualization():
    """Simple CMA-ES optimization with sparkline visualization."""
    
    print("=" * 70)
    print("CMA-ES OPTIMIZATION WITH REAL-TIME SPARKLINE VISUALIZATION")
    print("=" * 70)
    
    # Initialize visualizer
    visualizer = CMAESVisualizer(history_size=100, update_interval=0.5)
    
    # CMA-ES parameters
    dim = 5  # Problem dimension
    population_size = 20
    mean = np.random.randn(dim)  # Initial mean
    sigma = 1.0  # Initial step size
    
    # Simple CMA-ES loop (simplified for demonstration)
    best_ever = float('inf')
    
    for iteration in range(1, 51):
        # Generate population
        population = np.random.randn(population_size, dim)
        scaled_pop = mean + sigma * population
        
        # Evaluate
        costs = [objective_function(x) for x in scaled_pop]
        
        # Get statistics
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        # Update best ever
        improvement = 0
        if best_cost < best_ever:
            improvement = best_ever - best_cost
            best_ever = best_cost
        
        # Update mean (simplified - just move toward best)
        mean = 0.9 * mean + 0.1 * scaled_pop[best_idx]
        
        # Decay sigma
        sigma *= 0.98
        
        # Check convergence
        converged = sigma < 1e-6 or best_cost < 1e-6
        
        # Create metrics
        metrics = CMAESMetrics(
            iteration=iteration,
            best_cost=best_cost,
            mean_cost=mean_cost,
            std_cost=std_cost,
            population_size=population_size,
            sigma=sigma,
            converged=converged,
            improvement=improvement
        )
        
        # Update visualization (only prints if update interval has passed)
        ui_update = visualizer.update(metrics)
        
        # Print detailed update every 10 iterations
        if iteration % 10 == 0:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}")
            print(f"{'='*70}")
            visualizer.print_update(metrics)
        
        # Early stopping
        if converged:
            print(f"\n🎯 CONVERGED at iteration {iteration}!")
            print(f"   Final cost: {best_cost:.6f}")
            print(f"   Final sigma: {sigma:.3e}")
            break
    
    # Save history
    visualizer.save_history("cmaes_optimization_history.json")
    print("\n✅ Optimization history saved!")
    
    # Show final summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"Total iterations: {iteration}")
    print(f"Best cost found: {best_ever:.6f}")
    print(f"Total improvements: {visualizer.total_improvements}")
    
    # Print final sparkline
    print("\nFinal Cost Trajectory:")
    final_sparkline = visualizer.sparkline.generate_sparkline(
        list(visualizer.best_cost_history),
        width=60,
        style="blocks",
        show_values=True
    )
    print(f"  {final_sparkline}")

def load_and_display_history():
    """Load and display saved optimization history."""
    
    print("\n" + "="*70)
    print("LOADING SAVED OPTIMIZATION HISTORY")
    print("="*70)
    
    # Check for existing checkpoint files
    checkpoint_dir = Path("metrics/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*_cmaes_best.json"))
        
        if checkpoints:
            print(f"\nFound {len(checkpoints)} checkpoint files:")
            for i, checkpoint in enumerate(checkpoints, 1):
                print(f"  {i}. {checkpoint.name}")
                
                # Load and display
                with open(checkpoint, 'r') as f:
                    data = json.load(f)
                
                print(f"     Iteration: {data.get('iteration', 'N/A')}")
                print(f"     Best Cost: {data.get('best_cost', 'N/A'):.4f}")
                print(f"     Converged: {data.get('converged', False)}")
                
                # Create mini sparkline from single point
                if 'best_cost' in data:
                    from cogniforge.ui.cmaes_visualizer import SparklineGenerator
                    generator = SparklineGenerator()
                    costs = [data['best_cost']] * 10  # Simulate history
                    sparkline = generator.generate_sparkline(
                        costs, width=20, style="blocks", show_values=False
                    )
                    print(f"     Progress: {sparkline}")
                print()

def demonstrate_sparkline_styles():
    """Demonstrate different sparkline visualization styles."""
    
    print("\n" + "="*70)
    print("SPARKLINE VISUALIZATION STYLES")
    print("="*70)
    
    # Generate sample cost data
    iterations = np.arange(0, 50)
    costs = list(100 * np.exp(-iterations/10) + 10 + np.random.normal(0, 2, 50))
    
    from cogniforge.ui.cmaes_visualizer import SparklineGenerator
    generator = SparklineGenerator()
    
    print("\n1. Unicode Blocks (default):")
    sparkline = generator.generate_sparkline(costs, width=50, style="blocks")
    print(f"   {sparkline}")
    
    print("\n2. ASCII (compatible with all terminals):")
    sparkline = generator.generate_sparkline(costs, width=50, style="ascii")
    print(f"   {sparkline}")
    
    print("\n3. Dots (compact):")
    sparkline = generator.generate_sparkline(costs, width=50, style="dots")
    print(f"   {sparkline}")
    
    print("\n4. Braille (ultra-smooth):")
    sparkline = generator.generate_sparkline(costs, width=50, style="braille")
    print(f"   {sparkline}")
    
    print("\n5. Compact view (for UI status bar):")
    compact = generator.generate_sparkline(costs[-20:], width=15, style="blocks", show_values=False)
    print(f"   Status: [Iter 50] {compact} Cost: {costs[-1]:.2f}")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║         📊 CMA-ES SPARKLINE VISUALIZATION INTEGRATION 📊           ║
║                                                                    ║
║  This demo shows how to integrate real-time sparkline progress    ║
║  visualization into your CMA-ES optimization workflow.            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Menu
    print("\nSelect demo:")
    print("1. Run CMA-ES optimization with live sparklines")
    print("2. Load and display saved optimization history")
    print("3. Demonstrate different sparkline styles")
    print("4. Run all demos")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        simple_cmaes_with_visualization()
    elif choice == "2":
        load_and_display_history()
    elif choice == "3":
        demonstrate_sparkline_styles()
    elif choice == "4":
        simple_cmaes_with_visualization()
        load_and_display_history()
        demonstrate_sparkline_styles()
    else:
        print("Invalid choice. Running default demo...")
        simple_cmaes_with_visualization()
    
    print("\n✨ Demo complete!")