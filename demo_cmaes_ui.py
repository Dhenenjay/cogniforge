#!/usr/bin/env python3
"""
CMA-ES UI Integration Demo

Complete demonstration of CMA-ES optimization with real-time sparkline
visualization integrated into the UI components.
"""

import numpy as np
import time
from cogniforge.ui.cmaes_visualizer import CMAESVisualizer, CMAESMetrics, SparklineGenerator

def print_ui_dashboard(visualizer, metrics, width=80):
    """Print a complete UI dashboard with sparklines."""
    
    # Get the latest sparklines
    cost_sparkline = visualizer.sparkline.generate_sparkline(
        list(visualizer.best_cost_history),
        width=30,
        style="blocks",
        show_values=False
    )
    
    sigma_sparkline = visualizer.sparkline.generate_sparkline(
        list(visualizer.sigma_history),
        width=20,
        style="dots",
        show_values=False
    )
    
    # Calculate progress
    progress = min(1.0, (50 - metrics.iteration) / 50) if metrics.iteration <= 50 else 1.0
    progress_bar = visualizer._create_progress_bar(progress, width=25)
    
    # Print dashboard
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print(f"║{'CMA-ES OPTIMIZATION DASHBOARD':^{width-2}}║")
    print("╠" + "═" * (width - 2) + "╣")
    
    # Status line
    status = "CONVERGED ✓" if metrics.converged else "OPTIMIZING ⟳"
    print(f"║ Status: {status:<20} Iteration: {metrics.iteration:>4} │ Population: {metrics.population_size:>3}   ║")
    
    # Cost display with sparkline
    print("╠" + "─" * (width - 2) + "╣")
    print(f"║ {'COST TRAJECTORY':^{width-4}} ║")
    print(f"║ Best:  {metrics.best_cost:>10.4f}  │ {cost_sparkline:<30} │ Trend: {'↓' if len(visualizer.best_cost_history) > 1 and visualizer.best_cost_history[-1] < visualizer.best_cost_history[-2] else '→':>2}  ║")
    print(f"║ Mean:  {metrics.mean_cost:>10.4f}  │ Std: {metrics.std_cost:>8.4f} │ Improvements: {visualizer.total_improvements:>3}     ║")
    
    # Sigma display with sparkline
    print("╠" + "─" * (width - 2) + "╣")
    print(f"║ {'STEP SIZE (σ)':^{width-4}} ║")
    print(f"║ Current: {metrics.sigma:.3e}  │ {sigma_sparkline:<20} │ Min: {min(visualizer.sigma_history) if visualizer.sigma_history else 0:.3e}  ║")
    
    # Progress bar
    print("╠" + "─" * (width - 2) + "╣")
    print(f"║ Progress: {progress_bar} │ {int(progress*100):>3}% │ Stagnation: {metrics.iteration - visualizer.last_improvement_iter:>3} ║")
    
    print("╚" + "═" * (width - 2) + "╝")

def run_demo():
    """Run the complete CMA-ES UI demo."""
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║                    📊 CMA-ES SPARKLINE UI INTEGRATION 📊                   ║
    ║                                                                            ║
    ║     Real-time optimization progress with sparkline visualizations         ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize visualizer
    visualizer = CMAESVisualizer(history_size=100)
    
    # Simulate CMA-ES optimization
    print("\n🚀 Starting optimization...")
    time.sleep(1)
    
    # Initial parameters
    best_cost = 1000.0
    sigma = 1.0
    np.random.seed(42)
    
    for iteration in range(1, 31):
        # Simulate cost improvement
        improvement_rate = np.exp(-iteration / 20)
        noise = np.random.uniform(-5, 5)
        best_cost = max(0.1, best_cost * (0.95 + 0.05 * improvement_rate) + noise)
        
        # Update sigma
        sigma *= 0.95
        
        # Create metrics
        metrics = CMAESMetrics(
            iteration=iteration,
            best_cost=best_cost,
            mean_cost=best_cost + abs(np.random.normal(10, 5)),
            std_cost=abs(np.random.normal(5, 2)),
            population_size=20,
            sigma=sigma,
            converged=(sigma < 1e-6 or best_cost < 1.0)
        )
        
        # Update visualizer
        visualizer.update(metrics)
        
        # Clear screen (optional - comment out for full history)
        # print("\033[2J\033[H")  # ANSI escape codes to clear screen
        
        # Print dashboard
        print_ui_dashboard(visualizer, metrics)
        
        # Show compact status line (for status bar)
        compact = visualizer._generate_compact_view(metrics)
        print(f"\nStatus Bar: {compact}")
        
        # Delay for animation effect
        time.sleep(0.2)
        
        # Stop if converged
        if metrics.converged:
            print("\n" + "="*80)
            print("🎯 OPTIMIZATION CONVERGED!")
            print("="*80)
            break
    
    # Final summary
    print("\n" + "╔" + "═" * 78 + "╗")
    print(f"║{'FINAL OPTIMIZATION SUMMARY':^78}║")
    print("╠" + "═" * 78 + "╣")
    
    # Generate final sparkline
    final_sparkline = visualizer.sparkline.generate_sparkline(
        list(visualizer.best_cost_history),
        width=60,
        style="blocks",
        show_values=True
    )
    
    print(f"║ Final Best Cost: {min(visualizer.best_cost_history):>10.4f}{' '*47}║")
    print(f"║ Total Iterations: {iteration:>9}{' '*48}║")
    print(f"║ Total Improvements: {visualizer.total_improvements:>7}{' '*48}║")
    print("╠" + "─" * 78 + "╣")
    print(f"║ Cost Trajectory:{' '*60}║")
    print(f"║  {final_sparkline:<75} ║")
    print("╚" + "═" * 78 + "╝")
    
    # Save history
    visualizer.save_history("ui_demo_history.json")
    print("\n✅ Optimization history saved to 'ui_demo_history.json'")

def show_live_comparison():
    """Show comparison of different sparkline styles in real-time."""
    
    print("\n" + "="*80)
    print("LIVE SPARKLINE COMPARISON")
    print("="*80)
    
    generator = SparklineGenerator()
    costs = []
    
    print("\nGenerating optimization data...")
    print("(Watch the sparklines grow in real-time)\n")
    
    for i in range(30):
        # Generate cost
        cost = 100 * np.exp(-i/10) + 10 + np.random.normal(0, 2)
        costs.append(cost)
        
        # Clear previous lines (ANSI escape codes)
        if i > 0:
            print("\033[4A", end="")  # Move cursor up 4 lines
        
        # Generate sparklines in different styles
        blocks = generator.generate_sparkline(costs, width=40, style="blocks", show_values=False)
        ascii_line = generator.generate_sparkline(costs, width=40, style="ascii", show_values=False)
        dots = generator.generate_sparkline(costs, width=40, style="dots", show_values=False)
        
        print(f"Blocks: {blocks}")
        print(f"ASCII:  {ascii_line}")
        print(f"Dots:   {dots}")
        print(f"Iter: {i+1:3} │ Cost: {cost:8.2f}")
        
        time.sleep(0.1)
    
    print("\n✨ Comparison complete!")

if __name__ == "__main__":
    print("""
    ┌────────────────────────────────────────────────────────────────────────┐
    │                     CMA-ES UI SPARKLINE INTEGRATION                    │
    ├────────────────────────────────────────────────────────────────────────┤
    │  This demo showcases real-time sparkline visualizations for CMA-ES     │
    │  optimization integrated into UI dashboards and status displays.       │
    └────────────────────────────────────────────────────────────────────────┘
    """)
    
    print("\nSelect demo mode:")
    print("1. Full Dashboard Demo (animated)")
    print("2. Live Sparkline Comparison")
    print("3. Run Both Demos")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        run_demo()
    elif choice == "2":
        show_live_comparison()
    elif choice == "3":
        run_demo()
        input("\nPress Enter to continue to sparkline comparison...")
        show_live_comparison()
    else:
        print("Running default demo...")
        run_demo()
    
    print("\n" + "="*80)
    print("✨ CMA-ES UI SPARKLINE DEMO COMPLETE!")
    print("="*80)