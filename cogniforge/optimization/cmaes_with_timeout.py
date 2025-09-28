"""
CMA-ES Optimizer with Time Budget Management

Implements CMA-ES optimization that gracefully handles time limits by
keeping the best solution found so far and proceeding with the pipeline.
"""

import numpy as np
import time
import logging
from typing import Callable, Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import signal
import threading
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CMAESResult:
    """Result from CMA-ES optimization."""
    best_solution: np.ndarray
    best_fitness: float
    iterations_completed: int
    time_elapsed: float
    budget_exceeded: bool
    convergence_reached: bool
    population_stats: Dict[str, Any]
    termination_reason: str


class TimeoutException(Exception):
    """Raised when time budget is exceeded."""
    pass


class CMAESWithTimeout:
    """CMA-ES optimizer with time budget management."""
    
    def __init__(
        self,
        fitness_function: Callable,
        dim: int,
        time_budget: float = 60.0,
        max_iterations: int = 1000,
        population_size: Optional[int] = None,
        sigma_init: float = 0.3,
        show_progress: bool = True,
        save_checkpoints: bool = True
    ):
        """
        Initialize CMA-ES optimizer with timeout.
        
        Args:
            fitness_function: Function to minimize
            dim: Problem dimension
            time_budget: Maximum time in seconds
            max_iterations: Maximum iterations
            population_size: Population size (auto if None)
            sigma_init: Initial step size
            show_progress: Display progress
            save_checkpoints: Save best solution periodically
        """
        self.fitness_function = fitness_function
        self.dim = dim
        self.time_budget = time_budget
        self.max_iterations = max_iterations
        self.sigma_init = sigma_init
        self.show_progress = show_progress
        self.save_checkpoints = save_checkpoints
        
        # Population size
        self.popsize = population_size or (4 + int(3 * np.log(dim)))
        
        # Initialize CMA-ES parameters
        self._initialize_parameters()
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_iteration = 0
        
        # Time tracking
        self.start_time = None
        self.time_elapsed = 0
        
        # Statistics
        self.history = {
            'fitness': [],
            'mean': [],
            'sigma': [],
            'time': []
        }
        
    def _initialize_parameters(self):
        """Initialize CMA-ES parameters."""
        # Strategy parameters
        self.mu = self.popsize // 2  # Parent number
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / 
                       ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Dynamic parameters
        self.mean = np.random.randn(self.dim)  # Initial mean
        self.sigma = self.sigma_init  # Step size
        self.pc = np.zeros(self.dim)  # Evolution path for C
        self.ps = np.zeros(self.dim)  # Evolution path for sigma
        self.C = np.eye(self.dim)  # Covariance matrix
        self.invsqrtC = np.eye(self.dim)  # C^-1/2
        self.eigeneval = 0  # Track when to update eigendecomposition
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        self.iteration_count = 0  # Initialize iteration counter
        
    def optimize(
        self,
        initial_solution: Optional[np.ndarray] = None
    ) -> CMAESResult:
        """
        Run CMA-ES optimization with time budget management.
        
        Args:
            initial_solution: Optional starting point
            
        Returns:
            CMAESResult with best solution found
        """
        self.start_time = time.time()
        
        # Set initial solution
        if initial_solution is not None:
            self.mean = initial_solution.copy()
        
        # Display header
        if self.show_progress:
            self._display_header()
        
        # Main optimization loop
        iteration = 0
        budget_exceeded = False
        convergence_reached = False
        termination_reason = "max_iterations"
        
        try:
            while iteration < self.max_iterations:
                # Check time budget
                self.time_elapsed = time.time() - self.start_time
                if self.time_elapsed > self.time_budget:
                    budget_exceeded = True
                    termination_reason = "time_budget_exceeded"
                    logger.warning(f"⏱️ Time budget ({self.time_budget}s) exceeded at iteration {iteration}")
                    if self.show_progress:
                        self._display_timeout_banner()
                    break
                
                # Generate and evaluate population
                fitness_values = self._iteration_step()
                
                # Track best solution
                min_idx = np.argmin(fitness_values)
                if fitness_values[min_idx] < self.best_fitness:
                    self.best_fitness = fitness_values[min_idx]
                    self.best_solution = self.population[min_idx].copy()
                    self.best_iteration = iteration
                    
                    # Save checkpoint
                    if self.save_checkpoints:
                        self._save_checkpoint()
                
                # Update statistics
                self.history['fitness'].append(self.best_fitness)
                self.history['mean'].append(self.mean.copy())
                self.history['sigma'].append(self.sigma)
                self.history['time'].append(self.time_elapsed)
                
                # Display progress
                if self.show_progress and iteration % 10 == 0:
                    self._display_progress(iteration, fitness_values)
                
                # Check convergence
                if self._check_convergence():
                    convergence_reached = True
                    termination_reason = "convergence"
                    logger.info(f"✅ Converged at iteration {iteration}")
                    break
                
                iteration += 1
                
        except KeyboardInterrupt:
            termination_reason = "user_interrupt"
            logger.warning("⚠️ Optimization interrupted by user")
            
        except Exception as e:
            termination_reason = f"error: {str(e)}"
            logger.error(f"Error during optimization: {e}")
        
        # Ensure we have a valid solution
        if self.best_solution is None:
            logger.warning("No valid solution found, using current mean")
            self.best_solution = self.mean.copy()
            self.best_fitness = self.fitness_function(self.best_solution)
        
        # Final display
        if self.show_progress:
            self._display_final_result(iteration, budget_exceeded)
        
        # Return result
        return CMAESResult(
            best_solution=self.best_solution,
            best_fitness=self.best_fitness,
            iterations_completed=iteration,
            time_elapsed=self.time_elapsed,
            budget_exceeded=budget_exceeded,
            convergence_reached=convergence_reached,
            population_stats={
                'final_mean': self.mean,
                'final_sigma': self.sigma,
                'best_iteration': self.best_iteration,
                'population_size': self.popsize
            },
            termination_reason=termination_reason
        )
    
    def _iteration_step(self) -> np.ndarray:
        """Perform one CMA-ES iteration."""
        # Generate population
        self.population = np.zeros((self.popsize, self.dim))
        for i in range(self.popsize):
            self.population[i] = self.mean + self.sigma * np.dot(self.invsqrtC, np.random.randn(self.dim))
        
        # Evaluate fitness
        fitness_values = np.array([self.fitness_function(x) for x in self.population])
        
        # Sort by fitness
        sorted_idx = np.argsort(fitness_values)
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.dot(self.weights, self.population[sorted_idx[:self.mu]])
        
        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  np.dot(self.invsqrtC, self.mean - old_mean) / self.sigma
        
        hsig = (np.linalg.norm(self.ps) / 
                np.sqrt(1 - (1 - self.cs) ** (2 * (self.iteration_count + 1))) / 
                self.chiN < 1.4 + 2 / (self.dim + 1))
        
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  (self.mean - old_mean) / self.sigma
        
        # Update covariance matrix
        artmp = (self.population[sorted_idx[:self.mu]] - old_mean) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) + 
                            (1 - hsig) * self.cc * (2 - self.cc) * self.C) +
                  self.cmu * np.dot(artmp.T, np.dot(np.diag(self.weights), artmp)))
        
        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * 
                            (np.linalg.norm(self.ps) / self.chiN - 1))
        
        # Update eigendecomposition periodically
        if self.eigeneval == 0 or self.iteration_count % (10 * self.dim) == 0:
            self._update_eigendecomposition()
            self.eigeneval = self.iteration_count
        
        self.iteration_count = getattr(self, 'iteration_count', 0) + 1
        
        return fitness_values
    
    def _update_eigendecomposition(self):
        """Update eigendecomposition of covariance matrix."""
        try:
            self.C = (self.C + self.C.T) / 2  # Enforce symmetry
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid negative eigenvalues
            self.invsqrtC = np.dot(eigenvectors, 
                                   np.dot(np.diag(1.0 / np.sqrt(eigenvalues)), 
                                         eigenvectors.T))
        except np.linalg.LinAlgError:
            logger.warning("Eigendecomposition failed, resetting covariance")
            self.C = np.eye(self.dim)
            self.invsqrtC = np.eye(self.dim)
    
    def _check_convergence(self) -> bool:
        """Check convergence criteria."""
        # Simple convergence based on sigma and fitness improvement
        if self.sigma < 1e-10:
            return True
        
        if len(self.history['fitness']) > 20:
            recent_fitness = self.history['fitness'][-20:]
            if np.std(recent_fitness) < 1e-10:
                return True
        
        return False
    
    def _save_checkpoint(self):
        """Save current best solution to file."""
        checkpoint = {
            'best_solution': self.best_solution.tolist(),
            'best_fitness': float(self.best_fitness),
            'iteration': self.best_iteration,
            'time_elapsed': self.time_elapsed,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"cmaes_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _display_header(self):
        """Display optimization header."""
        print("\n" + "="*70)
        print("🔬 CMA-ES OPTIMIZATION WITH TIME BUDGET")
        print("="*70)
        print(f"Dimension: {self.dim}")
        print(f"Population size: {self.popsize}")
        print(f"Time budget: {self.time_budget}s")
        print(f"Max iterations: {self.max_iterations}")
        print("-"*70)
        print("Iter | Best Fitness | Sigma | Time (s) | Progress")
        print("-"*70)
    
    def _display_progress(self, iteration: int, fitness_values: np.ndarray):
        """Display optimization progress."""
        progress = min(self.time_elapsed / self.time_budget * 100, 100)
        avg_fitness = np.mean(fitness_values)
        print(f"{iteration:4d} | {self.best_fitness:11.6f} | {self.sigma:.3e} | "
              f"{self.time_elapsed:7.2f} | {progress:3.0f}%")
    
    def _display_timeout_banner(self):
        """Display timeout warning banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║              ⏱️  CMA-ES TIME BUDGET EXCEEDED                ║
║                                                              ║
║         Keeping best solution found so far and proceeding   ║
║         Optimization quality may be suboptimal              ║
╚══════════════════════════════════════════════════════════════╝"""
        print(banner)
    
    def _display_final_result(self, iterations: int, budget_exceeded: bool):
        """Display final optimization result."""
        print("-"*70)
        if budget_exceeded:
            print("⏱️ TIME BUDGET EXCEEDED - Using best solution found")
        else:
            print("✅ OPTIMIZATION COMPLETED")
        print(f"Best fitness: {self.best_fitness:.6f}")
        print(f"Found at iteration: {self.best_iteration}")
        print(f"Total iterations: {iterations}")
        print(f"Time elapsed: {self.time_elapsed:.2f}s")
        print("="*70)


class RobustCMAESOptimizer:
    """Wrapper for robust CMA-ES optimization in pipelines."""
    
    def __init__(
        self,
        default_time_budget: float = 60.0,
        fallback_solution: Optional[np.ndarray] = None,
        log_timeouts: bool = True
    ):
        """
        Initialize robust CMA-ES optimizer.
        
        Args:
            default_time_budget: Default time limit in seconds
            fallback_solution: Solution to use if optimization fails
            log_timeouts: Whether to log timeout events
        """
        self.default_time_budget = default_time_budget
        self.fallback_solution = fallback_solution
        self.log_timeouts = log_timeouts
        self.timeout_log = []
        
    def optimize_with_timeout(
        self,
        objective_function: Callable,
        dim: int,
        time_budget: Optional[float] = None,
        initial_guess: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run CMA-ES with timeout handling.
        
        Args:
            objective_function: Function to minimize
            dim: Problem dimension
            time_budget: Time limit (uses default if None)
            initial_guess: Starting point
            **kwargs: Additional CMA-ES parameters
            
        Returns:
            Dictionary with solution and metadata
        """
        time_budget = time_budget or self.default_time_budget
        
        # Create optimizer
        optimizer = CMAESWithTimeout(
            fitness_function=objective_function,
            dim=dim,
            time_budget=time_budget,
            **kwargs
        )
        
        # Run optimization
        result = optimizer.optimize(initial_solution=initial_guess)
        
        # Handle timeout
        if result.budget_exceeded:
            self._handle_timeout(result)
        
        # Package result for pipeline
        return {
            'solution': result.best_solution,
            'fitness': result.best_fitness,
            'success': not result.budget_exceeded,
            'timeout': result.budget_exceeded,
            'iterations': result.iterations_completed,
            'time_used': result.time_elapsed,
            'termination': result.termination_reason,
            'stats': result.population_stats
        }
    
    def _handle_timeout(self, result: CMAESResult):
        """Handle timeout event."""
        timeout_entry = {
            'timestamp': datetime.now().isoformat(),
            'iterations_completed': result.iterations_completed,
            'time_elapsed': result.time_elapsed,
            'best_fitness': result.best_fitness,
            'termination': result.termination_reason
        }
        
        self.timeout_log.append(timeout_entry)
        
        if self.log_timeouts:
            logger.warning(
                f"CMA-ES timeout: {result.iterations_completed} iterations "
                f"in {result.time_elapsed:.2f}s, best fitness: {result.best_fitness:.6f}"
            )
            
            # Save to file
            log_file = f"cmaes_timeouts_{datetime.now().strftime('%Y%m%d')}.json"
            try:
                with open(log_file, 'a') as f:
                    json.dump(timeout_entry, f)
                    f.write('\n')
            except:
                pass
    
    def get_timeout_statistics(self) -> Dict[str, Any]:
        """Get statistics about timeouts."""
        if not self.timeout_log:
            return {'total_timeouts': 0}
        
        return {
            'total_timeouts': len(self.timeout_log),
            'avg_iterations_at_timeout': np.mean([t['iterations_completed'] for t in self.timeout_log]),
            'avg_fitness_at_timeout': np.mean([t['best_fitness'] for t in self.timeout_log]),
            'recent_timeouts': self.timeout_log[-5:]
        }


def create_robust_cmaes(
    time_budget: float = 60.0,
    dim: int = 10,
    show_progress: bool = True
) -> RobustCMAESOptimizer:
    """
    Create a robust CMA-ES optimizer for pipelines.
    
    Args:
        time_budget: Maximum optimization time
        dim: Problem dimension
        show_progress: Display progress
        
    Returns:
        Configured optimizer
    """
    # Create default fallback solution (center of search space)
    fallback = np.zeros(dim)
    
    return RobustCMAESOptimizer(
        default_time_budget=time_budget,
        fallback_solution=fallback,
        log_timeouts=True
    )


# Test functions
def sphere_function(x):
    """Simple sphere function for testing."""
    return np.sum(x**2)


def rosenbrock_function(x):
    """Rosenbrock function for testing."""
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x) - 1))


def test_timeout_handling():
    """Test CMA-ES with timeout handling."""
    
    print("="*70)
    print("TESTING CMA-ES TIMEOUT HANDLING")
    print("="*70)
    
    # Test 1: Quick optimization (should complete)
    print("\n1. Fast optimization (5s budget for simple problem):")
    print("-"*40)
    
    optimizer1 = CMAESWithTimeout(
        fitness_function=sphere_function,
        dim=5,
        time_budget=5.0,
        max_iterations=100,
        show_progress=True
    )
    
    result1 = optimizer1.optimize()
    print(f"\nResult: {'TIMEOUT' if result1.budget_exceeded else 'COMPLETED'}")
    print(f"Best fitness: {result1.best_fitness:.6f}")
    print(f"Solution norm: {np.linalg.norm(result1.best_solution):.6f}")
    
    # Test 2: Timeout scenario (complex problem, short budget)
    print("\n2. Timeout scenario (2s budget for hard problem):")
    print("-"*40)
    
    def slow_function(x):
        """Slow function to force timeout."""
        time.sleep(0.05)  # Simulate expensive evaluation
        return rosenbrock_function(x)
    
    optimizer2 = CMAESWithTimeout(
        fitness_function=slow_function,
        dim=10,
        time_budget=2.0,
        max_iterations=1000,
        show_progress=True
    )
    
    result2 = optimizer2.optimize()
    print(f"\nResult: {'TIMEOUT' if result2.budget_exceeded else 'COMPLETED'}")
    print(f"Best fitness: {result2.best_fitness:.6f}")
    print(f"Iterations completed: {result2.iterations_completed}")
    
    # Test 3: Pipeline integration
    print("\n3. Pipeline integration test:")
    print("-"*40)
    
    robust_optimizer = create_robust_cmaes(time_budget=3.0, dim=8)
    
    pipeline_result = robust_optimizer.optimize_with_timeout(
        objective_function=sphere_function,
        dim=8,
        time_budget=3.0
    )
    
    print(f"Pipeline continues: {'YES' if pipeline_result['solution'] is not None else 'NO'}")
    print(f"Solution available: {pipeline_result['solution'] is not None}")
    print(f"Timeout occurred: {pipeline_result['timeout']}")
    print(f"Best fitness: {pipeline_result['fitness']:.6f}")
    
    # Show statistics
    stats = robust_optimizer.get_timeout_statistics()
    print(f"\nTimeout statistics:")
    print(f"  Total timeouts: {stats['total_timeouts']}")
    
    print("\n" + "="*70)
    print("✅ CMA-ES timeout handling test complete!")
    print("="*70)


if __name__ == "__main__":
    test_timeout_handling()