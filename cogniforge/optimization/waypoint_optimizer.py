"""
Waypoint Optimizer using CMA-ES

Optimizes robot waypoints using Covariance Matrix Adaptation Evolution Strategy
for smooth, collision-free trajectories.
"""

import numpy as np
from typing import Callable, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for waypoint optimization."""
    n_waypoints: int = 5
    waypoint_dim: int = 3  # 3D waypoints by default
    population_size: Optional[int] = None  # 4 + floor(3*ln(n)) if None
    max_iterations: int = 1000
    target_fitness: float = 1e-10
    sigma_init: float = 0.3  # Initial step size
    seed: Optional[int] = None
    verbose: bool = False
    
    # CMA-ES specific parameters
    c_sigma: Optional[float] = None  # Path length control
    d_sigma: Optional[float] = None  # Damping for step-size
    c_c: Optional[float] = None  # Covariance matrix learning rate
    c_1: Optional[float] = None  # Rank-one update learning rate
    c_mu: Optional[float] = None  # Rank-mu update learning rate
    
    # Constraints
    bounds_min: Optional[np.ndarray] = None
    bounds_max: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Compute default values after initialization."""
        n = self.n_waypoints * self.waypoint_dim
        
        if self.population_size is None:
            # Avoid divide by zero for n=0 or negative
            if n <= 0:
                self.population_size = 10  # Default minimum
            else:
                self.population_size = 4 + int(3 * np.log(max(n, 1)))
        
        # Compute CMA-ES parameters if not provided
        mu = self.population_size // 2
        
        if self.c_sigma is None:
            self.c_sigma = (mu + 2) / (n + mu + 5)
        
        if self.d_sigma is None:
            self.d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + self.c_sigma
        
        if self.c_c is None:
            self.c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        
        if self.c_1 is None:
            self.c_1 = 2 / ((n + 1.3) ** 2 + mu)
        
        if self.c_mu is None:
            self.c_mu = min(1 - self.c_1, 2 * (mu - 2 + 1/mu) / ((n + 2) ** 2 + mu))


@dataclass
class OptimizationResult:
    """Result of waypoint optimization."""
    best_waypoints: np.ndarray
    best_cost: float
    n_iterations: int
    n_evaluations: int
    converged: bool
    history: Dict[str, List[float]] = field(default_factory=dict)
    message: str = ""
    
    def reshape_waypoints(self, n_waypoints: int, waypoint_dim: int) -> np.ndarray:
        """Reshape flat waypoint array to (n_waypoints, waypoint_dim)."""
        return self.best_waypoints.reshape(n_waypoints, waypoint_dim)


class CostFunction(ABC):
    """Abstract base class for cost functions."""
    
    @abstractmethod
    def __call__(self, waypoints: np.ndarray) -> float:
        """Evaluate cost for given waypoints."""
        pass
    
    def gradient(self, waypoints: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Compute numerical gradient (for validation)."""
        grad = np.zeros_like(waypoints)
        for i in range(len(waypoints)):
            waypoints_plus = waypoints.copy()
            waypoints_minus = waypoints.copy()
            waypoints_plus[i] += epsilon
            waypoints_minus[i] -= epsilon
            grad[i] = (self(waypoints_plus) - self(waypoints_minus)) / (2 * epsilon)
        return grad


class ParabolaCost(CostFunction):
    """Simple parabolic cost function for testing."""
    
    def __init__(self, center: Optional[np.ndarray] = None, scale: float = 1.0):
        """
        Initialize parabolic cost function.
        
        Args:
            center: Global minimum location
            scale: Scaling factor for the parabola
        """
        self.center = center
        self.scale = scale
    
    def __call__(self, waypoints: np.ndarray) -> float:
        """Compute sum of squared distances from center."""
        if self.center is None:
            self.center = np.zeros_like(waypoints)
        
        diff = waypoints - self.center
        return self.scale * np.sum(diff ** 2)


class RosenbockCost(CostFunction):
    """Rosenbrock function for more challenging optimization."""
    
    def __init__(self, a: float = 1.0, b: float = 100.0):
        """
        Initialize Rosenbrock cost function.
        
        Args:
            a: First parameter (usually 1.0)
            b: Second parameter (usually 100.0)
        """
        self.a = a
        self.b = b
    
    def __call__(self, waypoints: np.ndarray) -> float:
        """Compute Rosenbrock function value."""
        x = waypoints[:-1]
        y = waypoints[1:]
        return np.sum(self.b * (y - x**2)**2 + (self.a - x)**2)


class TrajectorySmoothnesssCost(CostFunction):
    """Cost function for smooth trajectories."""
    
    def __init__(self, n_waypoints: int, waypoint_dim: int,
                 smoothness_weight: float = 1.0,
                 collision_weight: float = 10.0,
                 obstacles: Optional[List[Tuple[np.ndarray, float]]] = None):
        """
        Initialize trajectory smoothness cost.
        
        Args:
            n_waypoints: Number of waypoints
            waypoint_dim: Dimension of each waypoint
            smoothness_weight: Weight for smoothness term
            collision_weight: Weight for collision avoidance
            obstacles: List of (center, radius) tuples for spherical obstacles
        """
        self.n_waypoints = n_waypoints
        self.waypoint_dim = waypoint_dim
        self.smoothness_weight = smoothness_weight
        self.collision_weight = collision_weight
        self.obstacles = obstacles or []
    
    def __call__(self, waypoints_flat: np.ndarray) -> float:
        """Compute trajectory cost."""
        # Reshape waypoints
        waypoints = waypoints_flat.reshape(self.n_waypoints, self.waypoint_dim)
        
        cost = 0.0
        
        # Smoothness cost (minimize acceleration)
        if self.n_waypoints >= 3:
            velocities = np.diff(waypoints, axis=0)
            accelerations = np.diff(velocities, axis=0)
            smoothness_cost = np.sum(accelerations ** 2)
            cost += self.smoothness_weight * smoothness_cost
        
        # Collision cost
        for waypoint in waypoints:
            for obstacle_center, obstacle_radius in self.obstacles:
                distance = np.linalg.norm(waypoint - obstacle_center)
                if distance < obstacle_radius:
                    penetration = obstacle_radius - distance
                    cost += self.collision_weight * penetration ** 2
        
        return cost


class CMAESOptimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy optimizer.
    
    Based on Hansen, N. (2006). The CMA Evolution Strategy: A Tutorial.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.n = config.n_waypoints * config.waypoint_dim
        
        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Strategy parameters
        self.lambda_ = config.population_size
        self.mu = self.lambda_ // 2
        
        # Selection weights
        weights_prime = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_prime / np.sum(weights_prime)
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        
        # Adaptation parameters
        self.c_sigma = config.c_sigma
        self.d_sigma = config.d_sigma
        self.c_c = config.c_c
        self.c_1 = config.c_1
        self.c_mu = config.c_mu
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset optimizer state."""
        # Mean of the distribution
        self.mean = np.random.randn(self.n) * 0.1
        
        # Step size
        self.sigma = self.config.sigma_init
        
        # Covariance matrix C = B * D^2 * B^T
        self.C = np.eye(self.n)
        self.B = np.eye(self.n)  # Eigenvectors
        self.D = np.ones(self.n)  # Square root of eigenvalues
        
        # Evolution paths
        self.p_sigma = np.zeros(self.n)  # For step size control
        self.p_c = np.zeros(self.n)  # For covariance matrix adaptation
        
        # Expectation of ||N(0,I)||
        self.chi_n = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))
        
        # History
        self.history = {
            'best_cost': [],
            'mean_cost': [],
            'sigma': [],
            'condition_number': []
        }
    
    def optimize(self, cost_function: Callable[[np.ndarray], float],
                initial_guess: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Optimize waypoints using CMA-ES.
        
        Args:
            cost_function: Function to minimize
            initial_guess: Initial waypoint values
            
        Returns:
            Optimization result
        """
        if initial_guess is not None:
            self.mean = initial_guess.flatten()
        
        best_waypoints = self.mean.copy()
        best_cost = float('inf')
        n_evaluations = 0
        converged = False
        
        for iteration in range(self.config.max_iterations):
            # Generate population
            population = self._sample_population()
            
            # Evaluate fitness
            costs = np.array([cost_function(ind) for ind in population])
            n_evaluations += self.lambda_
            
            # Sort by fitness
            sorted_indices = np.argsort(costs)
            costs_sorted = costs[sorted_indices]
            population_sorted = population[sorted_indices]
            
            # Update best solution
            if costs_sorted[0] < best_cost:
                best_cost = costs_sorted[0]
                best_waypoints = population_sorted[0].copy()
            
            # Record history
            self.history['best_cost'].append(best_cost)
            self.history['mean_cost'].append(np.mean(costs))
            self.history['sigma'].append(self.sigma)
            self.history['condition_number'].append(np.max(self.D) / np.min(self.D))
            
            # Check convergence
            if best_cost < self.config.target_fitness:
                converged = True
                message = f"Converged: fitness {best_cost:.2e} < target {self.config.target_fitness:.2e}"
                break
            
            if self.sigma < 1e-10:
                message = f"Converged: step size {self.sigma:.2e} too small"
                break
            
            # Update distribution
            self._update_distribution(population_sorted[:self.mu])
            
            # Verbose output
            if self.config.verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: best_cost={best_cost:.6e}, "
                          f"sigma={self.sigma:.6e}, cond={self.history['condition_number'][-1]:.2f}")
        
        else:
            message = f"Maximum iterations ({self.config.max_iterations}) reached"
        
        return OptimizationResult(
            best_waypoints=best_waypoints,
            best_cost=best_cost,
            n_iterations=iteration + 1,
            n_evaluations=n_evaluations,
            converged=converged,
            history=self.history,
            message=message
        )
    
    def _sample_population(self) -> np.ndarray:
        """Sample new population from the distribution."""
        population = np.zeros((self.lambda_, self.n))
        
        for i in range(self.lambda_):
            z = np.random.randn(self.n)
            y = self.B @ (self.D * z)
            x = self.mean + self.sigma * y
            
            # Apply bounds if specified
            if self.config.bounds_min is not None:
                x = np.maximum(x, self.config.bounds_min.flatten())
            if self.config.bounds_max is not None:
                x = np.minimum(x, self.config.bounds_max.flatten())
            
            population[i] = x
        
        return population
    
    def _update_distribution(self, selected_population: np.ndarray):
        """Update the distribution parameters based on selected individuals."""
        # Compute new mean
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * selected_population, axis=0)
        
        # Update evolution paths
        y_mean = (self.mean - old_mean) / self.sigma
        
        # Compute C^(-1/2) * y_mean
        C_inv_half = self.B @ np.diag(1 / self.D) @ self.B.T
        z_mean = C_inv_half @ y_mean
        
        # Update sigma evolution path
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                      np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z_mean
        
        # Update sigma
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * 
                            (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        
        # Update covariance evolution path
        # Note: simplified h_sigma calculation (iteration not available in this scope)
        h_sigma = 1 if np.linalg.norm(self.p_sigma) < (1.4 + 2/(self.n + 1)) * self.chi_n else 0
        
        self.p_c = (1 - self.c_c) * self.p_c + \
                  h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_mean
        
        # Update covariance matrix
        # Rank-one update
        self.C = (1 - self.c_1 - self.c_mu) * self.C + \
                self.c_1 * np.outer(self.p_c, self.p_c)
        
        # Rank-mu update
        for i in range(self.mu):
            y_i = (selected_population[i] - old_mean) / self.sigma
            self.C += self.c_mu * self.weights[i] * np.outer(y_i, y_i)
        
        # Eigendecomposition for sampling
        eigenvalues, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(eigenvalues, 1e-10))  # Ensure positive


def optimize_waypoints_cma(
    cost_function: Callable[[np.ndarray], float],
    config: OptimizationConfig,
    initial_guess: Optional[np.ndarray] = None
) -> OptimizationResult:
    """
    Optimize waypoints using CMA-ES.
    
    Args:
        cost_function: Function to minimize
        config: Optimization configuration
        initial_guess: Initial waypoint values
        
    Returns:
        Optimization result
    """
    optimizer = CMAESOptimizer(config)
    return optimizer.optimize(cost_function, initial_guess)


# Example usage
if __name__ == "__main__":
    # Test on simple parabola
    print("Testing CMA-ES on parabolic cost function")
    print("=" * 60)
    
    # Create configuration
    config = OptimizationConfig(
        n_waypoints=3,
        waypoint_dim=2,
        max_iterations=100,
        verbose=True,
        seed=42
    )
    
    # Create cost function (parabola centered at origin)
    cost_fn = ParabolaCost(center=np.zeros(6), scale=1.0)
    
    # Initial guess (away from optimum)
    initial_guess = np.ones(6) * 2.0
    
    # Optimize
    result = optimize_waypoints_cma(cost_fn, config, initial_guess)
    
    # Print results
    print(f"\nOptimization Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Evaluations: {result.n_evaluations}")
    print(f"  Best cost: {result.best_cost:.6e}")
    print(f"  Message: {result.message}")
    
    # Reshape and print waypoints
    waypoints = result.reshape_waypoints(3, 2)
    print(f"\nOptimized waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i}: [{wp[0]:.6f}, {wp[1]:.6f}]")
    
    print(f"\nDistance from optimum: {np.linalg.norm(result.best_waypoints):.6e}")