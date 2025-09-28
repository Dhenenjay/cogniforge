#!/usr/bin/env python3
"""
Unit Tests for CMA-ES Waypoint Optimizer

Tests optimization convergence on toy cost functions including parabolas,
Rosenbrock function, and trajectory smoothness costs.
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniforge.optimization.waypoint_optimizer import (
    OptimizationConfig, OptimizationResult, CMAESOptimizer,
    ParabolaCost, RosenbockCost, TrajectorySmoothnesssCost,
    optimize_waypoints_cma, CostFunction
)


class TestParabolaConvergence(unittest.TestCase):
    """Test CMA-ES convergence on parabolic cost functions."""
    
    def test_simple_parabola_convergence(self):
        """Test convergence on simple parabola centered at origin."""
        # Configuration
        config = OptimizationConfig(
            n_waypoints=3,
            waypoint_dim=2,
            max_iterations=200,
            target_fitness=1e-8,
            sigma_init=1.0,
            seed=42,
            verbose=False
        )
        
        # Cost function: f(x) = ||x||^2
        cost_fn = ParabolaCost(center=np.zeros(6), scale=1.0)
        
        # Initial guess (far from optimum)
        initial_guess = np.ones(6) * 5.0
        initial_cost = cost_fn(initial_guess)
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config, initial_guess)
        
        # Assertions
        self.assertTrue(result.converged, 
                       f"Failed to converge: {result.message}")
        
        self.assertLess(result.best_cost, 1e-6,
                       f"Final cost {result.best_cost:.2e} not small enough")
        
        self.assertLess(np.linalg.norm(result.best_waypoints), 1e-3,
                       f"Solution too far from origin: {np.linalg.norm(result.best_waypoints):.2e}")
        
        # Check improvement
        improvement = initial_cost / (result.best_cost + 1e-10)
        self.assertGreater(improvement, 1e6,
                          f"Insufficient improvement: {improvement:.2e}x")
        
        print(f"\nParabola test: converged in {result.n_iterations} iterations")
        print(f"  Initial cost: {initial_cost:.2e}")
        print(f"  Final cost: {result.best_cost:.2e}")
        print(f"  Distance from optimum: {np.linalg.norm(result.best_waypoints):.2e}")
    
    def test_shifted_parabola(self):
        """Test convergence on parabola with shifted center."""
        # Configuration
        config = OptimizationConfig(
            n_waypoints=2,
            waypoint_dim=3,
            max_iterations=300,
            target_fitness=1e-8,
            seed=123
        )
        
        # Cost function centered at (1, 2, 3, -1, -2, -3)
        center = np.array([1, 2, 3, -1, -2, -3])
        cost_fn = ParabolaCost(center=center, scale=2.0)
        
        # Initial guess
        initial_guess = np.zeros(6)
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config, initial_guess)
        
        # Check convergence to shifted center
        distance_from_center = np.linalg.norm(result.best_waypoints - center)
        
        self.assertLess(distance_from_center, 1e-3,
                       f"Solution {distance_from_center:.2e} away from center")
        
        self.assertLess(result.best_cost, 1e-6,
                       f"Final cost {result.best_cost:.2e} too high")
    
    def test_high_dimensional_parabola(self):
        """Test convergence on high-dimensional parabola."""
        # Configuration for 10 waypoints in 3D
        config = OptimizationConfig(
            n_waypoints=10,
            waypoint_dim=3,
            max_iterations=500,
            target_fitness=1e-6,
            seed=456
        )
        
        # 30-dimensional parabola
        cost_fn = ParabolaCost(center=np.zeros(30), scale=1.0)
        
        # Random initial guess
        np.random.seed(456)
        initial_guess = np.random.randn(30) * 3.0
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config, initial_guess)
        
        # Check convergence
        self.assertLess(result.best_cost, 1e-4,
                       f"High-dim parabola cost {result.best_cost:.2e} too high")
        
        print(f"\nHigh-dimensional test: {result.n_iterations} iterations, "
              f"final cost: {result.best_cost:.2e}")


class TestRosenbrockConvergence(unittest.TestCase):
    """Test CMA-ES on the challenging Rosenbrock function."""
    
    def test_rosenbrock_2d(self):
        """Test convergence on 2D Rosenbrock function."""
        # Configuration
        config = OptimizationConfig(
            n_waypoints=1,
            waypoint_dim=2,
            max_iterations=1000,
            target_fitness=1e-6,
            sigma_init=0.5,
            seed=789
        )
        
        # Rosenbrock function
        cost_fn = RosenbockCost(a=1.0, b=100.0)
        
        # Initial guess
        initial_guess = np.array([-1.0, 1.0])
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config, initial_guess)
        
        # Rosenbrock minimum is at (1, 1) with value 0
        expected = np.array([1.0, 1.0])
        distance = np.linalg.norm(result.best_waypoints - expected)
        
        self.assertLess(distance, 0.1,
                       f"Rosenbrock solution {distance:.2e} away from (1,1)")
        
        self.assertLess(result.best_cost, 0.01,
                       f"Rosenbrock cost {result.best_cost:.2e} too high")
        
        print(f"\nRosenbrock test: {result.n_iterations} iterations")
        print(f"  Solution: {result.best_waypoints}")
        print(f"  Expected: {expected}")
        print(f"  Distance: {distance:.3f}")
    
    def test_rosenbrock_higher_dim(self):
        """Test on higher-dimensional Rosenbrock."""
        # Configuration for 4D Rosenbrock
        config = OptimizationConfig(
            n_waypoints=2,
            waypoint_dim=2,
            max_iterations=2000,
            target_fitness=1e-4,
            sigma_init=0.3,
            seed=111
        )
        
        cost_fn = RosenbockCost()
        initial_guess = np.zeros(4)
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config, initial_guess)
        
        # Check if close to all ones (global optimum)
        expected = np.ones(4)
        distance = np.linalg.norm(result.best_waypoints - expected)
        
        self.assertLess(distance, 0.5,
                       f"4D Rosenbrock distance {distance:.2e} from optimum")


class TestTrajectoryOptimization(unittest.TestCase):
    """Test trajectory optimization with smoothness and obstacles."""
    
    def test_straight_line_trajectory(self):
        """Test that straight line is optimal without obstacles."""
        # Configuration
        config = OptimizationConfig(
            n_waypoints=5,
            waypoint_dim=2,
            max_iterations=300,
            target_fitness=1e-4,
            seed=222
        )
        
        # Cost function with only smoothness
        cost_fn = TrajectorySmoothnesssCost(
            n_waypoints=5,
            waypoint_dim=2,
            smoothness_weight=1.0,
            collision_weight=0.0,
            obstacles=[]
        )
        
        # Initial waypoints: zigzag pattern
        initial = np.array([
            [0, 0],
            [1, 1],
            [2, -1],
            [3, 1],
            [4, 0]
        ]).flatten()
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config, initial)
        
        # Check if trajectory is approximately straight
        waypoints = result.reshape_waypoints(5, 2)
        
        # Check smoothness (small accelerations)
        velocities = np.diff(waypoints, axis=0)
        accelerations = np.diff(velocities, axis=0)
        max_acceleration = np.max(np.abs(accelerations))
        
        self.assertLess(max_acceleration, 0.5,
                       f"Trajectory not smooth enough: max accel = {max_acceleration:.3f}")
        
        print(f"\nStraight line test: {result.n_iterations} iterations")
        print(f"  Max acceleration: {max_acceleration:.3f}")
    
    def test_obstacle_avoidance(self):
        """Test trajectory optimization with obstacle avoidance."""
        # Configuration
        config = OptimizationConfig(
            n_waypoints=5,
            waypoint_dim=2,
            max_iterations=500,
            target_fitness=1e-2,
            seed=333
        )
        
        # Add obstacle in the middle
        obstacles = [
            (np.array([2.0, 0.0]), 0.8)  # Obstacle at (2, 0) with radius 0.8
        ]
        
        cost_fn = TrajectorySmoothnesssCost(
            n_waypoints=5,
            waypoint_dim=2,
            smoothness_weight=1.0,
            collision_weight=100.0,
            obstacles=obstacles
        )
        
        # Initial straight line through obstacle
        initial = np.array([
            [0, 0],
            [1, 0],
            [2, 0],  # Goes through obstacle!
            [3, 0],
            [4, 0]
        ]).flatten()
        
        initial_cost = cost_fn(initial)
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config, initial)
        
        # Check that trajectory avoids obstacle
        waypoints = result.reshape_waypoints(5, 2)
        
        for i, wp in enumerate(waypoints):
            distance_to_obstacle = np.linalg.norm(wp - obstacles[0][0])
            self.assertGreater(distance_to_obstacle, obstacles[0][1] - 0.1,
                             f"Waypoint {i} too close to obstacle: {distance_to_obstacle:.3f}")
        
        # Check improvement
        self.assertLess(result.best_cost, initial_cost * 0.1,
                       f"Insufficient cost reduction: {result.best_cost:.2e} vs {initial_cost:.2e}")
        
        print(f"\nObstacle avoidance test: {result.n_iterations} iterations")
        print(f"  Initial cost: {initial_cost:.2e}")
        print(f"  Final cost: {result.best_cost:.2e}")
        print("  Waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"    {i}: [{wp[0]:.2f}, {wp[1]:.2f}]")


class TestCMAESProperties(unittest.TestCase):
    """Test CMA-ES algorithm properties and behavior."""
    
    def test_population_size_scaling(self):
        """Test that population size scales with dimensionality."""
        dimensions = [2, 6, 12, 24, 48]
        
        for dim in dimensions:
            config = OptimizationConfig(
                n_waypoints=dim // 3,
                waypoint_dim=3
            )
            
            expected_min = 4 + int(3 * np.log(dim))
            self.assertGreaterEqual(config.population_size, expected_min,
                                   f"Population size too small for dim={dim}")
            
            print(f"Dim {dim}: population size = {config.population_size}")
    
    def test_convergence_history(self):
        """Test that optimization history is properly recorded."""
        config = OptimizationConfig(
            n_waypoints=2,
            waypoint_dim=2,
            max_iterations=50,
            seed=444
        )
        
        cost_fn = ParabolaCost(center=np.zeros(4), scale=1.0)
        initial = np.ones(4) * 2.0
        
        result = optimize_waypoints_cma(cost_fn, config, initial)
        
        # Check history
        self.assertIn('best_cost', result.history)
        self.assertIn('mean_cost', result.history)
        self.assertIn('sigma', result.history)
        self.assertIn('condition_number', result.history)
        
        # Check monotonic improvement in best cost
        best_costs = result.history['best_cost']
        for i in range(1, len(best_costs)):
            self.assertLessEqual(best_costs[i], best_costs[i-1] * 1.001,
                               f"Best cost increased at iteration {i}")
        
        # Check sigma adaptation
        sigmas = result.history['sigma']
        self.assertLess(sigmas[-1], sigmas[0],
                       "Sigma did not decrease during optimization")
        
        print(f"\nHistory test: {len(best_costs)} iterations recorded")
        print(f"  Initial sigma: {sigmas[0]:.3f}")
        print(f"  Final sigma: {sigmas[-1]:.3e}")
        print(f"  Cost reduction: {best_costs[0]:.2e} -> {best_costs[-1]:.2e}")
    
    def test_bounds_enforcement(self):
        """Test that bounds constraints are enforced."""
        config = OptimizationConfig(
            n_waypoints=3,
            waypoint_dim=2,
            max_iterations=100,
            bounds_min=np.array([-1, -1] * 3),
            bounds_max=np.array([1, 1] * 3),
            seed=555
        )
        
        # Cost function with optimum outside bounds
        center = np.array([2, 2, 2, 2, 2, 2])  # Outside [-1, 1] bounds
        cost_fn = ParabolaCost(center=center, scale=1.0)
        
        # Optimize
        result = optimize_waypoints_cma(cost_fn, config)
        
        # Check that solution respects bounds
        self.assertTrue(np.all(result.best_waypoints >= -1.001),
                       f"Solution violates lower bounds: {np.min(result.best_waypoints)}")
        
        self.assertTrue(np.all(result.best_waypoints <= 1.001),
                       f"Solution violates upper bounds: {np.max(result.best_waypoints)}")
        
        # Solution should be at boundary
        expected_bounded = np.ones(6)  # Closest point to (2,2,2,2,2,2) within bounds
        distance = np.linalg.norm(result.best_waypoints - expected_bounded)
        
        self.assertLess(distance, 0.1,
                       f"Bounded solution {distance:.2e} from expected")
        
        print(f"\nBounds test: solution at boundary as expected")
        print(f"  Min value: {np.min(result.best_waypoints):.3f}")
        print(f"  Max value: {np.max(result.best_waypoints):.3f}")
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        config1 = OptimizationConfig(
            n_waypoints=2,
            waypoint_dim=2,
            max_iterations=50,
            seed=12345
        )
        
        config2 = OptimizationConfig(
            n_waypoints=2,
            waypoint_dim=2,
            max_iterations=50,
            seed=12345
        )
        
        cost_fn = ParabolaCost(center=np.ones(4), scale=1.0)
        
        # Run twice with same seed
        result1 = optimize_waypoints_cma(cost_fn, config1)
        result2 = optimize_waypoints_cma(cost_fn, config2)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1.best_waypoints,
            result2.best_waypoints,
            decimal=10,
            err_msg="Results not reproducible with same seed"
        )
        
        self.assertEqual(result1.n_iterations, result2.n_iterations,
                        "Iteration counts differ with same seed")
        
        print("\nReproducibility test: Results match with same seed ✓")


class TestConvergenceRates(unittest.TestCase):
    """Test convergence rates on different problem types."""
    
    def test_convergence_rate_comparison(self):
        """Compare convergence rates on different cost functions."""
        config = OptimizationConfig(
            n_waypoints=2,
            waypoint_dim=2,
            max_iterations=200,
            target_fitness=1e-6,
            seed=777
        )
        
        # Test functions
        test_cases = [
            ("Parabola", ParabolaCost(center=np.zeros(4), scale=1.0), np.ones(4)*2),
            ("Shifted Parabola", ParabolaCost(center=np.ones(4)*0.5, scale=2.0), np.zeros(4)),
            ("Rosenbrock", RosenbockCost(a=1.0, b=10.0), np.zeros(4))
        ]
        
        print("\nConvergence Rate Comparison:")
        print("-" * 50)
        
        for name, cost_fn, initial in test_cases:
            result = optimize_waypoints_cma(cost_fn, config, initial)
            
            # Calculate convergence rate
            history = result.history['best_cost']
            if len(history) > 10:
                # Log-linear fit for exponential convergence
                early = np.mean(history[:10])
                late = np.mean(history[-10:])
                if early > 0 and late > 0:
                    convergence_rate = np.log(late/early) / len(history)
                else:
                    convergence_rate = 0
            else:
                convergence_rate = 0
            
            print(f"{name:20s}: {result.n_iterations:3d} iters, "
                  f"cost={result.best_cost:.2e}, "
                  f"rate={convergence_rate:.3f}")
    
    def test_dimension_scaling(self):
        """Test how performance scales with problem dimension."""
        dimensions = [2, 4, 8, 16, 32]
        
        print("\nDimension Scaling Test:")
        print("-" * 50)
        
        for dim in dimensions:
            n_waypoints = dim // 2
            waypoint_dim = 2
            
            config = OptimizationConfig(
                n_waypoints=n_waypoints,
                waypoint_dim=waypoint_dim,
                max_iterations=500,
                target_fitness=1e-4,
                seed=888
            )
            
            cost_fn = ParabolaCost(center=np.zeros(dim), scale=1.0)
            initial = np.ones(dim) * 2.0
            
            result = optimize_waypoints_cma(cost_fn, config, initial)
            
            print(f"Dim {dim:3d}: {result.n_iterations:4d} iters, "
                  f"{result.n_evaluations:5d} evals, "
                  f"cost={result.best_cost:.2e}")


def visualize_convergence(result: OptimizationResult, title: str = "Convergence"):
    """Visualize convergence history (optional, for debugging)."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Best cost
        axes[0, 0].semilogy(result.history['best_cost'])
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Best Cost')
        axes[0, 0].set_title('Best Cost Evolution')
        axes[0, 0].grid(True)
        
        # Mean cost
        axes[0, 1].semilogy(result.history['mean_cost'])
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Mean Cost')
        axes[0, 1].set_title('Population Mean Cost')
        axes[0, 1].grid(True)
        
        # Sigma
        axes[1, 0].semilogy(result.history['sigma'])
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Sigma')
        axes[1, 0].set_title('Step Size Evolution')
        axes[1, 0].grid(True)
        
        # Condition number
        axes[1, 1].plot(result.history['condition_number'])
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Condition Number')
        axes[1, 1].set_title('Covariance Matrix Condition')
        axes[1, 1].grid(True)
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        pass  # Matplotlib not available


def run_tests():
    """Run all waypoint optimizer tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestParabolaConvergence))
    suite.addTests(loader.loadTestsFromTestCase(TestRosenbrockConvergence))
    suite.addTests(loader.loadTestsFromTestCase(TestTrajectoryOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestCMAESProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestConvergenceRates))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("CMA-ES WAYPOINT OPTIMIZER TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! CMA-ES converges correctly on toy problems.")
    else:
        print("\n❌ Some tests failed. Review errors above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)