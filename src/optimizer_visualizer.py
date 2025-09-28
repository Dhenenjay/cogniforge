"""
Optimization Visualizer with Best Cost Tracking and ETA

This module provides real-time visualization of optimization progress,
showing best cost improvements at each iteration with an ETA progress bar.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from collections import deque
import threading
from colorama import init, Fore, Style, Back
import warnings
import sys
from abc import ABC, abstractmethod

# Initialize colorama
init(autoreset=True)

# Suppress matplotlib warnings
warnings.filterwarnings("ignore")

# Set matplotlib backend
plt.switch_backend('TkAgg')


@dataclass
class OptimizationConfig:
    """Configuration for optimization with visualization"""
    max_iterations: int = 100
    tolerance: float = 1e-6
    population_size: int = 50
    display_interval: int = 1
    eta_window: int = 10  # Window for ETA calculation
    verbose: bool = True
    plot_update_interval: float = 0.1  # Update plot every 100ms
    show_particles: bool = True  # Show particle positions (if applicable)
    convergence_patience: int = 20  # Stop if no improvement for N iterations


class OptimizationVisualizer:
    """Real-time optimization progress visualizer with ETA"""
    
    def __init__(self, max_iterations: int = 100, 
                 problem_name: str = "Optimization",
                 target_cost: Optional[float] = None):
        """
        Initialize optimizer visualizer
        
        Args:
            max_iterations: Maximum iterations
            problem_name: Name of optimization problem
            target_cost: Target cost value (if known)
        """
        self.max_iterations = max_iterations
        self.problem_name = problem_name
        self.target_cost = target_cost
        
        # Data storage
        self.iterations = []
        self.best_costs = []
        self.current_costs = []
        self.improvements = []
        self.times = []
        self.start_time = None
        self.eta_history = deque(maxlen=10)
        
        # Best cost tracking
        self.best_cost = float('inf')
        self.best_iteration = 0
        self.best_solution = None
        self.improvement_count = 0
        
        # Setup visualization
        self._setup_plot()
        
        # Threading
        self.lock = threading.Lock()
        self.update_thread = None
        self.stop_event = threading.Event()
        
    def _setup_plot(self):
        """Setup the visualization plot"""
        self.fig = plt.figure(figsize=(15, 8), facecolor='#f5f5f5')
        self.fig.suptitle(f'🎯 {self.problem_name} Optimization Progress', 
                          fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main cost plot (2x2 grid)
        self.ax_cost = self.fig.add_subplot(gs[:2, :2])
        self._setup_cost_plot()
        
        # Improvement rate plot
        self.ax_improvement = self.fig.add_subplot(gs[2, :2])
        self._setup_improvement_plot()
        
        # ETA and progress
        self.ax_progress = self.fig.add_subplot(gs[0, 2])
        self._setup_progress_plot()
        
        # Statistics
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self._setup_stats_plot()
        
        # Best solution info
        self.ax_best = self.fig.add_subplot(gs[2, 2])
        self._setup_best_plot()
        
        # Initialize plot elements
        self.best_line, = self.ax_cost.plot([], [], 'g-', linewidth=2.5, 
                                            label='Best Cost', marker='o', markersize=4)
        self.current_line, = self.ax_cost.plot([], [], 'b--', linewidth=1.5, 
                                               alpha=0.6, label='Current Cost')
        
        if self.target_cost is not None:
            self.ax_cost.axhline(y=self.target_cost, color='r', linestyle='--', 
                                alpha=0.5, label=f'Target: {self.target_cost:.4f}')
        
        # Improvement bars
        self.improvement_bars = None
        
        # Progress bar
        self.progress_bar = mpatches.Rectangle((0, 0.4), 0, 0.2, 
                                              facecolor='#4CAF50', alpha=0.8)
        self.ax_progress.add_patch(self.progress_bar)
        
        # ETA bar
        self.eta_bar = mpatches.Rectangle((0, 0.1), 1, 0.2, 
                                         facecolor='#2196F3', alpha=0.3)
        self.ax_progress.add_patch(self.eta_bar)
        
        # Text elements
        self.progress_text = self.ax_progress.text(0.5, 0.7, '', ha='center', 
                                                   fontsize=10, fontweight='bold')
        self.eta_text = self.ax_progress.text(0.5, 0.2, '', ha='center', 
                                             fontsize=10, color='blue')
        
        # Stats text
        self.stats_texts = {}
        stats_labels = ['Current Iter:', 'Best Cost:', 'Best Iter:', 
                       'Improvements:', 'Convergence:', 'Time Elapsed:']
        for i, label in enumerate(stats_labels):
            y_pos = 0.9 - i * 0.15
            self.ax_stats.text(0.05, y_pos, label, fontsize=9, fontweight='bold')
            text = self.ax_stats.text(0.55, y_pos, '--', fontsize=9)
            self.stats_texts[label] = text
        
        # Best solution text
        self.best_solution_text = self.ax_best.text(0.5, 0.5, 
                                                    'Waiting for first iteration...', 
                                                    ha='center', va='center', 
                                                    fontsize=10, wrap=True)
        
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)
        
    def _setup_cost_plot(self):
        """Setup cost evolution plot"""
        self.ax_cost.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        self.ax_cost.set_ylabel('Cost', fontsize=11, fontweight='bold')
        self.ax_cost.set_title('📉 Cost Evolution', fontsize=12, fontweight='bold')
        self.ax_cost.grid(True, alpha=0.3, linestyle='--')
        self.ax_cost.set_facecolor('white')
        self.ax_cost.legend(loc='upper right', framealpha=0.9)
        
    def _setup_improvement_plot(self):
        """Setup improvement rate plot"""
        self.ax_improvement.set_xlabel('Iteration', fontsize=10)
        self.ax_improvement.set_ylabel('Improvement', fontsize=10)
        self.ax_improvement.set_title('📊 Cost Improvements per Iteration', 
                                     fontsize=11, fontweight='bold')
        self.ax_improvement.grid(True, alpha=0.3)
        self.ax_improvement.set_facecolor('white')
        
    def _setup_progress_plot(self):
        """Setup progress and ETA plot"""
        self.ax_progress.set_xlim(0, 1)
        self.ax_progress.set_ylim(0, 1)
        self.ax_progress.set_title('⏱️ Progress & ETA', fontsize=11, fontweight='bold')
        self.ax_progress.set_xticks([])
        self.ax_progress.set_yticks([])
        self.ax_progress.set_facecolor('#f9f9f9')
        
    def _setup_stats_plot(self):
        """Setup statistics display"""
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.set_title('📈 Optimization Statistics', fontsize=11, fontweight='bold')
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor('#f9f9f9')
        
    def _setup_best_plot(self):
        """Setup best solution display"""
        self.ax_best.set_xlim(0, 1)
        self.ax_best.set_ylim(0, 1)
        self.ax_best.set_title('🏆 Best Solution', fontsize=11, fontweight='bold')
        self.ax_best.axis('off')
        self.ax_best.set_facecolor('#fffef0')
        
    def start(self):
        """Start the visualizer"""
        self.start_time = time.time()
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop(self):
        """Stop the visualizer"""
        self.stop_event.set()
        if self.update_thread:
            self.update_thread.join(timeout=1)
            
    def update(self, iteration: int, current_cost: float, 
              best_cost: float, best_solution: Any = None):
        """
        Update optimization progress
        
        Args:
            iteration: Current iteration number
            current_cost: Cost at current iteration
            best_cost: Best cost found so far
            best_solution: Best solution found (optional)
        """
        with self.lock:
            self.iterations.append(iteration)
            self.current_costs.append(current_cost)
            self.best_costs.append(best_cost)
            
            # Track improvements
            if best_cost < self.best_cost:
                improvement = self.best_cost - best_cost if self.best_cost != float('inf') else 0
                self.improvements.append(improvement)
                self.best_cost = best_cost
                self.best_iteration = iteration
                self.best_solution = best_solution
                self.improvement_count += 1
                
                # Print improvement message
                if improvement > 0:
                    print(f"{Fore.GREEN}✨ Iteration {iteration}: "
                          f"Best cost improved to {best_cost:.6f} "
                          f"(improvement: {improvement:.6f}){Style.RESET_ALL}")
            else:
                self.improvements.append(0)
                
            # Update times
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)
            
            # Calculate ETA
            if len(self.iterations) > 1:
                recent_time = self.times[-1] - (self.times[-2] if len(self.times) > 1 else 0)
                self.eta_history.append(recent_time)
                avg_time_per_iter = np.mean(self.eta_history)
                remaining_iters = self.max_iterations - iteration
                eta_seconds = avg_time_per_iter * remaining_iters
            else:
                eta_seconds = 0
                
    def _update_loop(self):
        """Background update loop for visualization"""
        while not self.stop_event.is_set():
            try:
                self._update_plot()
                time.sleep(0.1)
            except Exception as e:
                print(f"Visualization error: {e}")
                break
                
    def _update_plot(self):
        """Update all plot elements"""
        with self.lock:
            if not self.iterations:
                return
                
            # Update cost lines
            self.best_line.set_data(self.iterations, self.best_costs)
            self.current_line.set_data(self.iterations, self.current_costs)
            
            # Auto-scale cost plot
            self.ax_cost.set_xlim(0, max(self.max_iterations, self.iterations[-1] + 1))
            all_costs = self.best_costs + self.current_costs
            if all_costs:
                cost_range = max(all_costs) - min(all_costs)
                margin = 0.1 * cost_range if cost_range > 0 else 0.1
                self.ax_cost.set_ylim(min(all_costs) - margin, max(all_costs) + margin)
                
            # Update improvement bars
            if self.improvements:
                self.ax_improvement.clear()
                self.ax_improvement.set_title('📊 Cost Improvements per Iteration', 
                                            fontsize=11, fontweight='bold')
                self.ax_improvement.set_xlabel('Iteration', fontsize=10)
                self.ax_improvement.set_ylabel('Improvement', fontsize=10)
                self.ax_improvement.grid(True, alpha=0.3)
                
                # Color bars based on improvement magnitude
                colors = ['green' if imp > 0 else 'lightgray' for imp in self.improvements]
                self.ax_improvement.bar(self.iterations, self.improvements, 
                                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                self.ax_improvement.set_xlim(0, max(self.max_iterations, self.iterations[-1] + 1))
                
            # Update progress bar
            progress = self.iterations[-1] / self.max_iterations
            self.progress_bar.set_width(progress)
            
            # Color based on progress
            if progress < 0.33:
                self.progress_bar.set_facecolor('#4CAF50')  # Green
            elif progress < 0.66:
                self.progress_bar.set_facecolor('#FFC107')  # Yellow
            else:
                self.progress_bar.set_facecolor('#FF5722')  # Orange
                
            # Update progress text
            self.progress_text.set_text(f'{self.iterations[-1]}/{self.max_iterations} '
                                       f'({progress*100:.1f}%)')
            
            # Calculate and update ETA
            if len(self.eta_history) > 0:
                avg_time = np.mean(self.eta_history)
                remaining = self.max_iterations - self.iterations[-1]
                eta_seconds = avg_time * remaining
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                self.eta_text.set_text(f'ETA: {eta_str}')
                
                # Update ETA bar position
                eta_progress = 1 - (remaining / self.max_iterations)
                self.eta_bar.set_x(eta_progress)
                self.eta_bar.set_width(1 - eta_progress)
            else:
                self.eta_text.set_text('ETA: Calculating...')
                
            # Update statistics
            elapsed = self.times[-1] if self.times else 0
            convergence_rate = (self.best_costs[0] - self.best_costs[-1]) / self.best_costs[0] * 100 \
                              if self.best_costs[0] != 0 else 0
                              
            self.stats_texts['Current Iter:'].set_text(f'{self.iterations[-1]}')
            self.stats_texts['Best Cost:'].set_text(f'{self.best_cost:.6f}')
            self.stats_texts['Best Iter:'].set_text(f'{self.best_iteration}')
            self.stats_texts['Improvements:'].set_text(f'{self.improvement_count}')
            self.stats_texts['Convergence:'].set_text(f'{convergence_rate:.1f}%')
            self.stats_texts['Time Elapsed:'].set_text(f'{elapsed:.1f}s')
            
            # Color code best cost based on improvement
            if self.improvement_count > 0:
                self.stats_texts['Best Cost:'].set_color('green')
            else:
                self.stats_texts['Best Cost:'].set_color('black')
                
            # Update best solution display
            if self.best_solution is not None:
                if isinstance(self.best_solution, (list, np.ndarray)):
                    if len(self.best_solution) <= 5:
                        solution_str = f"Solution:\n{np.array(self.best_solution).round(4)}"
                    else:
                        solution_str = f"Solution (first 5):\n{np.array(self.best_solution[:5]).round(4)}\n..."
                else:
                    solution_str = f"Best found at iter {self.best_iteration}"
                    
                self.best_solution_text.set_text(solution_str)
                
        # Refresh display
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except:
            pass


class OptimizationTracker:
    """Tracks and displays optimization progress with automatic visualization"""
    
    def __init__(self, optimizer_name: str = "Optimizer",
                 max_iterations: int = 100,
                 target_cost: Optional[float] = None,
                 show_live: bool = True):
        """
        Initialize optimization tracker
        
        Args:
            optimizer_name: Name of the optimizer
            max_iterations: Maximum iterations
            target_cost: Target cost if known
            show_live: Show live visualization
        """
        self.optimizer_name = optimizer_name
        self.max_iterations = max_iterations
        self.target_cost = target_cost
        self.show_live = show_live
        
        self.visualizer = None
        if show_live:
            self.visualizer = OptimizationVisualizer(
                max_iterations, 
                optimizer_name,
                target_cost
            )
            
        self.history = {
            'iterations': [],
            'best_costs': [],
            'current_costs': [],
            'improvements': [],
            'solutions': []
        }
        
    def start(self):
        """Start tracking"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 Starting {self.optimizer_name} Optimization{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        print(f"📊 Max Iterations: {self.max_iterations}")
        if self.target_cost:
            print(f"🎯 Target Cost: {self.target_cost}")
        print(f"📈 Live Visualization: {'Enabled' if self.show_live else 'Disabled'}\n")
        
        if self.visualizer:
            self.visualizer.start()
            
    def update(self, iteration: int, current_cost: float, 
              best_cost: float, best_solution: Any = None):
        """Update tracking with new iteration data"""
        # Store in history
        self.history['iterations'].append(iteration)
        self.history['current_costs'].append(current_cost)
        self.history['best_costs'].append(best_cost)
        self.history['solutions'].append(best_solution)
        
        # Calculate improvement
        if len(self.history['best_costs']) > 1:
            prev_best = self.history['best_costs'][-2]
            improvement = prev_best - best_cost if best_cost < prev_best else 0
        else:
            improvement = 0
            
        self.history['improvements'].append(improvement)
        
        # Update visualizer
        if self.visualizer:
            self.visualizer.update(iteration, current_cost, best_cost, best_solution)
            
        # Check if target reached
        if self.target_cost and best_cost <= self.target_cost:
            print(f"\n{Fore.GREEN}🎉 Target cost reached! "
                  f"Best: {best_cost:.6f} ≤ Target: {self.target_cost:.6f}{Style.RESET_ALL}")
            return True
            
        return False
        
    def finish(self):
        """Finish tracking and show summary"""
        if self.visualizer:
            self.visualizer.stop()
            
        # Print summary
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Optimization Summary{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        if self.history['best_costs']:
            final_best = min(self.history['best_costs'])
            best_iter = self.history['best_costs'].index(final_best)
            total_improvements = sum(1 for imp in self.history['improvements'] if imp > 0)
            
            print(f"✅ Best Cost: {final_best:.6f} (at iteration {best_iter})")
            print(f"📈 Total Improvements: {total_improvements}")
            print(f"🔄 Total Iterations: {len(self.history['iterations'])}")
            
            if len(self.history['best_costs']) > 1:
                initial_cost = self.history['best_costs'][0]
                reduction = (initial_cost - final_best) / initial_cost * 100
                print(f"📉 Cost Reduction: {reduction:.1f}%")


# Example optimizers with visualization

class SimpleOptimizer:
    """Example optimizer with visualization"""
    
    def __init__(self, func: Callable, dim: int = 2, 
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer
        
        Args:
            func: Objective function to minimize
            dim: Problem dimension
            config: Optimization configuration
        """
        self.func = func
        self.dim = dim
        self.config = config or OptimizationConfig()
        
    def optimize(self, show_progress: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run optimization with visualization
        
        Args:
            show_progress: Show progress visualization
            
        Returns:
            Best solution and cost
        """
        # Initialize tracker
        tracker = OptimizationTracker(
            "Simple Optimizer",
            self.config.max_iterations,
            show_live=show_progress
        )
        tracker.start()
        
        # Initialize solution
        best_solution = np.random.randn(self.dim)
        best_cost = self.func(best_solution)
        
        # Optimization loop
        for iteration in range(self.config.max_iterations):
            # Generate new candidate
            candidate = best_solution + np.random.randn(self.dim) * 0.1
            candidate_cost = self.func(candidate)
            
            # Update if better
            if candidate_cost < best_cost:
                best_solution = candidate
                best_cost = candidate_cost
                
            # Update tracker
            target_reached = tracker.update(
                iteration, 
                candidate_cost, 
                best_cost,
                best_solution
            )
            
            if target_reached:
                break
                
            # Small delay for visualization
            time.sleep(0.01)
            
        tracker.finish()
        return best_solution, best_cost


def demo_optimization():
    """Demonstrate optimization with visualization"""
    
    print(f"{Fore.CYAN}{'='*70}")
    print(f" OPTIMIZATION VISUALIZATION DEMO")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # Define test function (Rosenbrock)
    def rosenbrock(x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                  for i in range(len(x) - 1))
    
    # Create optimizer
    config = OptimizationConfig(
        max_iterations=200,
        population_size=30,
        tolerance=1e-6
    )
    
    optimizer = SimpleOptimizer(rosenbrock, dim=5, config=config)
    
    # Run optimization with visualization
    best_solution, best_cost = optimizer.optimize(show_progress=True)
    
    print(f"\n{Fore.GREEN}✅ Optimization complete!{Style.RESET_ALL}")
    print(f"Best solution: {best_solution.round(4)}")
    print(f"Best cost: {best_cost:.6f}")
    
    print("\nPress Enter to close visualization and exit...")
    input()
    plt.close('all')


if __name__ == "__main__":
    demo_optimization()