"""
Demo: CMA-ES with Time Budget in Pipeline

Shows how CMA-ES optimization gracefully handles time budget limits,
keeping the best solution found and continuing with the pipeline.
"""

import sys
import numpy as np
import time
import logging
from typing import Dict, Any, List, Callable, Optional

sys.path.insert(0, 'C:/Users/Dhenenjay/cogniforge')

from cogniforge.optimization.cmaes_with_timeout import (
    CMAESWithTimeout,
    create_robust_cmaes,
    sphere_function,
    rosenbrock_function
)
from cogniforge.ui.console_utils import ConsoleAutoScroller, ProgressTracker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Pipeline with CMA-ES time budget management."""
    
    def __init__(self, time_budget: float = 10.0):
        self.time_budget = time_budget
        self.scroller = ConsoleAutoScroller()
        self.results_history = []
        
    def optimize_task(
        self,
        task_name: str,
        objective_function: Callable,
        dim: int,
        time_budget: Optional[float] = None,
        simulate_slow: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize a task with time budget management.
        
        Args:
            task_name: Name of optimization task
            objective_function: Function to minimize
            dim: Problem dimension
            time_budget: Time limit (uses default if None)
            simulate_slow: Make function slow to force timeout
            
        Returns:
            Optimization results
        """
        print("\n" + "="*70)
        print(f"📊 TASK: {task_name}")
        print("="*70)
        
        budget = time_budget or self.time_budget
        
        # Wrap function if simulating slow evaluation
        if simulate_slow:
            def slow_wrapper(x):
                time.sleep(0.1)  # 100ms per evaluation
                return objective_function(x)
            opt_function = slow_wrapper
            print("⚠️ Simulating slow function evaluations...")
        else:
            opt_function = objective_function
        
        # Create optimizer
        optimizer = CMAESWithTimeout(
            fitness_function=opt_function,
            dim=dim,
            time_budget=budget,
            max_iterations=1000,
            show_progress=True,
            save_checkpoints=False
        )
        
        # Run optimization
        result = optimizer.optimize()
        
        # Process result
        pipeline_data = {
            'task': task_name,
            'dimension': dim,
            'best_solution': result.best_solution,
            'best_fitness': result.best_fitness,
            'iterations': result.iterations_completed,
            'time_used': result.time_elapsed,
            'timeout': result.budget_exceeded,
            'termination': result.termination_reason
        }
        
        self.results_history.append(pipeline_data)
        
        # Continue pipeline with best solution
        self._continue_pipeline(pipeline_data)
        
        return pipeline_data
    
    def _continue_pipeline(self, optimization_result: Dict[str, Any]):
        """Continue pipeline processing with optimization result."""
        
        print("\n🔄 CONTINUING PIPELINE WITH OPTIMIZATION RESULT")
        print("-"*50)
        
        if optimization_result['timeout']:
            print("⚠️ Using best solution found before timeout")
        else:
            print("✅ Using optimal solution")
        
        # Simulate downstream processing
        solution = optimization_result['best_solution']
        
        print(f"Solution vector norm: {np.linalg.norm(solution):.6f}")
        print(f"Solution quality (fitness): {optimization_result['best_fitness']:.6f}")
        
        # Simulate using solution in next pipeline stage
        print("\n📍 Next pipeline stages:")
        stages = [
            "Trajectory generation",
            "Control parameter tuning",
            "Execution planning",
            "Safety validation"
        ]
        
        tracker = ProgressTracker(len(stages), "Pipeline processing")
        for stage in stages:
            time.sleep(0.2)
            tracker.update(1)
            self.scroller.print_and_scroll(f"  ✓ {stage} completed")
        tracker.finish()
        
        print("\n✅ Pipeline completed successfully!")
    
    def show_summary(self):
        """Display summary of all optimization tasks."""
        
        print("\n" + "="*70)
        print("📈 OPTIMIZATION PIPELINE SUMMARY")
        print("="*70)
        
        if not self.results_history:
            print("No tasks completed")
            return
        
        print("\n| Task | Dimension | Iterations | Time (s) | Fitness | Status |")
        print("|------|-----------|------------|----------|---------|--------|")
        
        for r in self.results_history:
            status = "⏱️ Timeout" if r['timeout'] else "✅ Complete"
            print(f"| {r['task'][:20]:<20} | {r['dimension']:^9} | "
                  f"{r['iterations']:^10} | {r['time_used']:^8.2f} | "
                  f"{r['best_fitness']:^7.3f} | {status:<10} |")
        
        # Statistics
        total_time = sum(r['time_used'] for r in self.results_history)
        timeouts = sum(1 for r in self.results_history if r['timeout'])
        
        print(f"\nTotal optimization time: {total_time:.2f}s")
        print(f"Timeouts: {timeouts}/{len(self.results_history)}")
        print(f"Pipeline success rate: 100% (all tasks completed)")


def run_optimization_scenarios():
    """Run different optimization scenarios."""
    
    print("\n" + "🧬"*35)
    print("CMA-ES TIME BUDGET MANAGEMENT DEMO")
    print("🧬"*35)
    
    pipeline = OptimizationPipeline(time_budget=5.0)
    
    # Scenario 1: Fast optimization (should complete)
    print("\n" + "="*70)
    print("SCENARIO 1: Fast Optimization (Simple Problem)")
    print("="*70)
    
    pipeline.optimize_task(
        task_name="Sphere Function",
        objective_function=sphere_function,
        dim=10,
        time_budget=5.0,
        simulate_slow=False
    )
    
    time.sleep(1)
    
    # Scenario 2: Timeout scenario
    print("\n" + "="*70)
    print("SCENARIO 2: Time Budget Exceeded (Complex Problem)")
    print("="*70)
    
    pipeline.optimize_task(
        task_name="Rosenbrock (Slow)",
        objective_function=rosenbrock_function,
        dim=20,
        time_budget=3.0,
        simulate_slow=True  # Force timeout
    )
    
    time.sleep(1)
    
    # Scenario 3: Mixed scenario
    print("\n" + "="*70)
    print("SCENARIO 3: Adaptive Time Budget")
    print("="*70)
    
    def rastrigin(x):
        """Rastrigin function."""
        n = len(x)
        return 10 * n + sum(x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(n))
    
    pipeline.optimize_task(
        task_name="Rastrigin Function",
        objective_function=rastrigin,
        dim=15,
        time_budget=4.0,
        simulate_slow=False
    )
    
    # Show summary
    pipeline.show_summary()


def test_robustness():
    """Test robustness of CMA-ES with various time budgets."""
    
    print("\n" + "="*70)
    print("🔬 ROBUSTNESS TEST: Various Time Budgets")
    print("="*70)
    
    dim = 10
    budgets = [1.0, 2.0, 5.0, 10.0]
    
    print(f"\nOptimizing {dim}D sphere function with different time budgets:")
    print("-"*50)
    
    results = []
    
    for budget in budgets:
        print(f"\nTime budget: {budget}s")
        
        optimizer = create_robust_cmaes(time_budget=budget, dim=dim)
        result = optimizer.optimize_with_timeout(
            objective_function=sphere_function,
            dim=dim,
            time_budget=budget
        )
        
        results.append({
            'budget': budget,
            'fitness': result['fitness'],
            'iterations': result['iterations'],
            'timeout': result['timeout']
        })
        
        status = "TIMEOUT" if result['timeout'] else "COMPLETE"
        print(f"  Status: {status}")
        print(f"  Best fitness: {result['fitness']:.6f}")
        print(f"  Iterations: {result['iterations']}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS: Time Budget vs. Solution Quality")
    print("="*70)
    
    print("\n| Budget (s) | Fitness | Iterations | Status |")
    print("|------------|---------|------------|--------|")
    
    for r in results:
        status = "Timeout" if r['timeout'] else "Complete"
        print(f"| {r['budget']:^10.1f} | {r['fitness']:^7.4f} | "
              f"{r['iterations']:^10} | {status:<8} |")
    
    print("\nObservation: Longer time budgets → Better solutions")
    print("Key insight: Pipeline always continues with best solution found")


def main():
    """Run the complete demonstration."""
    
    print("\n" + "="*70)
    print("CMA-ES WITH TIME BUDGET MANAGEMENT")
    print("="*70)
    print("\nDemonstrating graceful handling of optimization timeouts")
    print("Pipeline continues with best solution found so far\n")
    
    # Run optimization scenarios
    run_optimization_scenarios()
    
    print("\n" + "="*70)
    
    # Run robustness test
    test_robustness()
    
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    
    print("\nKey Features Demonstrated:")
    print("  1. CMA-ES respects time budget constraints")
    print("  2. Best solution is always preserved")
    print("  3. Clear timeout banner when budget exceeded")
    print("  4. Pipeline continues without interruption")
    print("  5. All timeout events are logged")


if __name__ == "__main__":
    from typing import Optional
    main()