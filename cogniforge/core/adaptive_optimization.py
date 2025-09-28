"""
Adaptive Optimization Module with BC Fallback

This module implements an adaptive optimization strategy that monitors improvement
and falls back to the BC policy if optimization fails to improve after N iterations,
then proceeds directly to vision refinement and code generation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import copy
from pathlib import Path
import json

from cogniforge.core.optimization import optimize_waypoints_cma
from cogniforge.core.refinement import refine_with_optimal_trajectory
from cogniforge.core.metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class OptimizationStatus(Enum):
    """Status of the optimization process."""
    IMPROVING = "improving"
    PLATEAU = "plateau"
    DEGRADING = "degrading"
    CONVERGED = "converged"
    FAILED = "failed"
    BC_FALLBACK = "bc_fallback"


@dataclass
class AdaptiveOptimizationConfig:
    """Configuration for adaptive optimization with BC fallback."""
    
    # Optimization parameters
    max_iterations: int = 100
    population_size: int = 16
    sigma0: float = 0.1
    
    # Improvement monitoring
    improvement_threshold: float = 0.01  # Minimum relative improvement per check
    patience: int = 10  # Iterations without improvement before fallback
    check_interval: int = 5  # Check improvement every N iterations
    
    # BC fallback parameters
    bc_fallback_enabled: bool = True
    min_iterations_before_fallback: int = 20  # Don't fallback too early
    keep_bc_policy: bool = True  # Keep BC policy when falling back
    
    # Early stopping
    early_stopping_enabled: bool = True
    convergence_threshold: float = 1e-4  # Absolute improvement for convergence
    min_reward_threshold: float = -float('inf')  # Minimum acceptable reward
    
    # Vision and codegen
    proceed_to_vision_on_fallback: bool = True
    proceed_to_codegen_on_fallback: bool = True
    
    # Logging and monitoring
    verbose: bool = True
    log_interval: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: Optional[Path] = None
    
    # Performance tracking
    track_metrics: bool = True
    metrics_window: int = 10  # Window for moving averages


@dataclass
class OptimizationState:
    """Current state of the optimization process."""
    
    iteration: int = 0
    best_reward: float = -float('inf')
    current_reward: float = -float('inf')
    rewards_history: List[float] = field(default_factory=list)
    
    # Improvement tracking
    iterations_without_improvement: int = 0
    last_improvement_iteration: int = 0
    improvement_rate: float = 0.0
    
    # Status
    status: OptimizationStatus = OptimizationStatus.IMPROVING
    converged: bool = False
    fallback_triggered: bool = False
    
    # Policies
    current_policy: Optional[nn.Module] = None
    best_policy: Optional[nn.Module] = None
    bc_policy: Optional[nn.Module] = None
    
    # Metrics
    total_time: float = 0.0
    evaluation_count: int = 0
    
    def update(self, reward: float, iteration: int):
        """Update state with new reward."""
        self.iteration = iteration
        self.current_reward = reward
        self.rewards_history.append(reward)
        self.evaluation_count += 1
        
        # Check for improvement
        if reward > self.best_reward:
            improvement = reward - self.best_reward
            self.improvement_rate = improvement / (abs(self.best_reward) + 1e-8)
            self.best_reward = reward
            self.last_improvement_iteration = iteration
            self.iterations_without_improvement = 0
            return True
        else:
            self.iterations_without_improvement += 1
            return False


class AdaptiveOptimizer:
    """
    Adaptive optimizer with BC fallback mechanism.
    
    This optimizer monitors improvement during optimization and falls back
    to the BC policy if no improvement is detected after N iterations,
    then proceeds to vision refinement and code generation.
    """
    
    def __init__(
        self,
        config: Optional[AdaptiveOptimizationConfig] = None,
        bc_policy: Optional[nn.Module] = None,
        cost_function: Optional[Callable] = None,
        env: Optional[Any] = None
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            config: Configuration for adaptive optimization
            bc_policy: Pre-trained BC policy to use as fallback
            cost_function: Cost/reward function for optimization
            env: Environment for evaluation
        """
        self.config = config or AdaptiveOptimizationConfig()
        self.bc_policy = bc_policy
        self.cost_function = cost_function
        self.env = env
        
        # Initialize state
        self.state = OptimizationState()
        if bc_policy:
            self.state.bc_policy = copy.deepcopy(bc_policy)
        
        # Metrics tracking
        self.metrics = MetricsTracker() if self.config.track_metrics else None
        
        # Checkpoint management
        if self.config.save_checkpoints and not self.config.checkpoint_dir:
            self.config.checkpoint_dir = Path("checkpoints/adaptive_opt")
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AdaptiveOptimizer initialized with patience={config.patience}")
    
    def check_improvement(self) -> Tuple[bool, OptimizationStatus]:
        """
        Check if optimization is improving.
        
        Returns:
            Tuple of (should_continue, status)
        """
        if self.state.iteration < self.config.min_iterations_before_fallback:
            # Too early to make decisions
            return True, OptimizationStatus.IMPROVING
        
        # Check for convergence
        if len(self.state.rewards_history) >= 2:
            recent_improvement = abs(
                self.state.rewards_history[-1] - self.state.rewards_history[-2]
            )
            if recent_improvement < self.config.convergence_threshold:
                logger.info("Optimization converged")
                return False, OptimizationStatus.CONVERGED
        
        # Check for plateau/degradation
        if self.state.iterations_without_improvement >= self.config.patience:
            if self.config.bc_fallback_enabled:
                logger.warning(
                    f"No improvement for {self.state.iterations_without_improvement} iterations. "
                    f"Triggering BC fallback."
                )
                return False, OptimizationStatus.BC_FALLBACK
            else:
                logger.warning("Optimization plateau detected but BC fallback disabled")
                return False, OptimizationStatus.PLATEAU
        
        # Check if we're still improving
        if self.state.iterations_without_improvement > self.config.patience // 2:
            return True, OptimizationStatus.PLATEAU
        
        return True, OptimizationStatus.IMPROVING
    
    def optimize(
        self,
        initial_waypoints: Optional[np.ndarray] = None,
        budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run adaptive optimization with BC fallback.
        
        Args:
            initial_waypoints: Initial waypoints for trajectory optimization
            budget: Maximum iterations (overrides config)
            
        Returns:
            Optimization results dictionary
        """
        start_time = time.time()
        budget = budget or self.config.max_iterations
        
        logger.info(
            f"Starting adaptive optimization (budget={budget}, "
            f"patience={self.config.patience}, BC fallback={'enabled' if self.config.bc_fallback_enabled else 'disabled'})"
        )
        
        # Initialize results
        results = {
            "success": False,
            "final_policy": None,
            "final_reward": -float('inf'),
            "bc_fallback_used": False,
            "optimization_iterations": 0,
            "total_time": 0.0,
            "status": None,
            "proceed_to_vision": False,
            "proceed_to_codegen": False,
            "convergence_history": [],
            "improvement_history": []
        }
        
        # Store BC policy baseline if available
        bc_baseline_reward = None
        if self.bc_policy and self.env:
            bc_baseline_reward = self._evaluate_policy(self.bc_policy)
            self.state.best_reward = bc_baseline_reward
            logger.info(f"BC baseline reward: {bc_baseline_reward:.4f}")
        
        # Main optimization loop
        iteration = 0
        should_continue = True
        
        while iteration < budget and should_continue:
            iteration += 1
            
            # Run optimization step
            if self.cost_function:
                # Use provided cost function
                reward = -self.cost_function(initial_waypoints.flatten())
            else:
                # Simulate optimization step
                reward = self._simulate_optimization_step(iteration)
            
            # Update state
            improved = self.state.update(reward, iteration)
            
            # Track metrics
            if self.metrics:
                self.metrics.track_value("optimization_reward", reward)
                self.metrics.track_value("improvement_rate", self.state.improvement_rate)
            
            # Log progress
            if iteration % self.config.log_interval == 0 and self.config.verbose:
                self._log_progress(iteration, reward, improved)
            
            # Check for improvement every N iterations
            if iteration % self.config.check_interval == 0:
                should_continue, status = self.check_improvement()
                self.state.status = status
                
                if not should_continue:
                    logger.info(f"Stopping optimization: {status.value}")
                    break
            
            # Save checkpoint if needed
            if self.config.save_checkpoints and iteration % 10 == 0:
                self._save_checkpoint(iteration)
            
            results["convergence_history"].append(reward)
            results["improvement_history"].append(improved)
        
        # Handle optimization outcome
        results["optimization_iterations"] = iteration
        results["status"] = self.state.status.value
        
        if self.state.status == OptimizationStatus.BC_FALLBACK:
            # Fallback to BC policy
            results = self._handle_bc_fallback(results)
        elif self.state.status == OptimizationStatus.CONVERGED:
            # Use best optimized policy
            results["success"] = True
            results["final_policy"] = self.state.best_policy or self.state.current_policy
            results["final_reward"] = self.state.best_reward
        else:
            # Partial success or failure
            results["success"] = self.state.best_reward > (bc_baseline_reward or -float('inf'))
            results["final_policy"] = self.state.best_policy or self.state.bc_policy
            results["final_reward"] = self.state.best_reward
        
        # Finalize
        results["total_time"] = time.time() - start_time
        
        # Generate summary
        self._generate_summary(results)
        
        return results
    
    def _handle_bc_fallback(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle BC fallback scenario.
        
        Args:
            results: Current results dictionary
            
        Returns:
            Updated results dictionary
        """
        logger.info("=" * 60)
        logger.info("BC FALLBACK TRIGGERED")
        logger.info("=" * 60)
        
        if self.state.bc_policy and self.config.keep_bc_policy:
            # Keep BC policy as final policy
            results["final_policy"] = self.state.bc_policy
            results["bc_fallback_used"] = True
            results["success"] = True  # Consider BC policy as success
            
            # Evaluate BC policy if we have environment
            if self.env:
                bc_reward = self._evaluate_policy(self.state.bc_policy)
                results["final_reward"] = bc_reward
                logger.info(f"Using BC policy with reward: {bc_reward:.4f}")
            else:
                results["final_reward"] = self.state.best_reward
            
            # Set flags to proceed to next stages
            if self.config.proceed_to_vision_on_fallback:
                results["proceed_to_vision"] = True
                logger.info("✓ Proceeding to vision refinement")
            
            if self.config.proceed_to_codegen_on_fallback:
                results["proceed_to_codegen"] = True
                logger.info("✓ Proceeding to code generation")
            
            logger.info(
                f"Optimization stopped early at iteration {results['optimization_iterations']}. "
                f"Using BC policy and proceeding to next stages."
            )
        else:
            # No BC policy available or not keeping it
            results["success"] = False
            results["final_policy"] = self.state.best_policy or self.state.current_policy
            results["final_reward"] = self.state.best_reward
            logger.warning("BC fallback triggered but no BC policy available")
        
        logger.info("=" * 60)
        
        return results
    
    def _evaluate_policy(self, policy: nn.Module) -> float:
        """
        Evaluate a policy in the environment.
        
        Args:
            policy: Policy to evaluate
            
        Returns:
            Average reward
        """
        if not self.env:
            # Simulate evaluation
            return np.random.randn() * 0.1 + 0.5
        
        # Run actual evaluation
        rewards = []
        for _ in range(3):  # Multiple rollouts
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    action = policy(torch.FloatTensor(obs))
                    if isinstance(action, torch.Tensor):
                        action = action.numpy()
                
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)
    
    def _simulate_optimization_step(self, iteration: int) -> float:
        """
        Simulate an optimization step for testing.
        
        Args:
            iteration: Current iteration
            
        Returns:
            Simulated reward
        """
        # Simulate improving then plateauing
        if iteration < 20:
            # Initial improvement
            base = -1.0 + iteration * 0.05
            noise = np.random.randn() * 0.01
        elif iteration < 40:
            # Slower improvement
            base = 0.0 + (iteration - 20) * 0.01
            noise = np.random.randn() * 0.02
        else:
            # Plateau with noise
            base = 0.2
            noise = np.random.randn() * 0.03
        
        return base + noise
    
    def _log_progress(self, iteration: int, reward: float, improved: bool):
        """Log optimization progress."""
        improvement_str = "↑" if improved else "→"
        logger.info(
            f"Iter {iteration:4d} | Reward: {reward:8.4f} {improvement_str} | "
            f"Best: {self.state.best_reward:8.4f} | "
            f"No improv: {self.state.iterations_without_improvement:2d}/{self.config.patience} | "
            f"Status: {self.state.status.value}"
        )
    
    def _save_checkpoint(self, iteration: int):
        """Save optimization checkpoint."""
        if not self.config.checkpoint_dir:
            return
        
        checkpoint = {
            "iteration": iteration,
            "state": {
                "best_reward": self.state.best_reward,
                "current_reward": self.state.current_reward,
                "rewards_history": self.state.rewards_history,
                "iterations_without_improvement": self.state.iterations_without_improvement,
                "status": self.state.status.value
            },
            "config": {
                "patience": self.config.patience,
                "improvement_threshold": self.config.improvement_threshold
            }
        }
        
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_iter_{iteration}.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _generate_summary(self, results: Dict[str, Any]):
        """Generate and log optimization summary."""
        logger.info("\n" + "=" * 60)
        logger.info("ADAPTIVE OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Status: {results['status']}")
        logger.info(f"Success: {results['success']}")
        logger.info(f"Iterations: {results['optimization_iterations']}")
        logger.info(f"Final reward: {results['final_reward']:.4f}")
        logger.info(f"BC fallback used: {results['bc_fallback_used']}")
        logger.info(f"Total time: {results['total_time']:.2f}s")
        
        if results['bc_fallback_used']:
            logger.info("\nBC Fallback Actions:")
            logger.info(f"  - Policy: {'BC policy retained' if self.config.keep_bc_policy else 'Best optimized policy'}")
            logger.info(f"  - Vision: {'Proceeding' if results['proceed_to_vision'] else 'Skipped'}")
            logger.info(f"  - Codegen: {'Proceeding' if results['proceed_to_codegen'] else 'Skipped'}")
        
        if self.metrics:
            logger.info("\nMetrics Summary:")
            summary = self.metrics.get_summary()
            for key, stats in summary.items():
                if 'reward' in key:
                    logger.info(f"  {key}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
        
        logger.info("=" * 60 + "\n")


def create_adaptive_optimizer(
    bc_policy: Optional[nn.Module] = None,
    patience: int = 10,
    improvement_threshold: float = 0.01,
    proceed_to_vision: bool = True,
    proceed_to_codegen: bool = True,
    verbose: bool = True
) -> AdaptiveOptimizer:
    """
    Convenience function to create an adaptive optimizer.
    
    Args:
        bc_policy: Pre-trained BC policy for fallback
        patience: Iterations without improvement before fallback
        improvement_threshold: Minimum improvement to continue
        proceed_to_vision: Whether to proceed to vision on fallback
        proceed_to_codegen: Whether to proceed to codegen on fallback
        verbose: Enable verbose logging
        
    Returns:
        Configured AdaptiveOptimizer instance
    
    Example:
        >>> bc_policy = train_bc_policy(demonstrations)
        >>> optimizer = create_adaptive_optimizer(
        ...     bc_policy=bc_policy,
        ...     patience=10,
        ...     proceed_to_vision=True
        ... )
        >>> results = optimizer.optimize(budget=50)
        >>> 
        >>> if results['bc_fallback_used']:
        >>>     print("Optimization failed, using BC policy")
        >>>     if results['proceed_to_vision']:
        >>>         # Continue with vision refinement
        >>>         vision_result = refine_with_vision(results['final_policy'])
    """
    config = AdaptiveOptimizationConfig(
        patience=patience,
        improvement_threshold=improvement_threshold,
        bc_fallback_enabled=True,
        keep_bc_policy=True,
        proceed_to_vision_on_fallback=proceed_to_vision,
        proceed_to_codegen_on_fallback=proceed_to_codegen,
        verbose=verbose
    )
    
    return AdaptiveOptimizer(config=config, bc_policy=bc_policy)


if __name__ == "__main__":
    """Demonstration of adaptive optimization with BC fallback."""
    import matplotlib.pyplot as plt
    
    # Create a dummy BC policy
    class DummyPolicy(nn.Module):
        def __init__(self, input_dim=10, output_dim=3):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return torch.tanh(self.fc(x))
    
    bc_policy = DummyPolicy()
    
    # Test 1: Optimization that plateaus (triggers BC fallback)
    print("\n" + "#" * 60)
    print("# TEST 1: Optimization with Plateau (BC Fallback)")
    print("#" * 60)
    
    optimizer1 = create_adaptive_optimizer(
        bc_policy=bc_policy,
        patience=10,
        proceed_to_vision=True,
        proceed_to_codegen=True
    )
    
    results1 = optimizer1.optimize(budget=50)
    
    print(f"\nResults:")
    print(f"  - BC fallback triggered: {results1['bc_fallback_used']}")
    print(f"  - Proceed to vision: {results1['proceed_to_vision']}")
    print(f"  - Proceed to codegen: {results1['proceed_to_codegen']}")
    
    # Test 2: Successful optimization (no fallback needed)
    print("\n" + "#" * 60)
    print("# TEST 2: Successful Optimization (No Fallback)")
    print("#" * 60)
    
    # Create optimizer with different cost function that improves
    class ImprovingOptimizer(AdaptiveOptimizer):
        def _simulate_optimization_step(self, iteration):
            # Continuously improving function
            return -1.0 + iteration * 0.1 + np.random.randn() * 0.01
    
    optimizer2 = ImprovingOptimizer(
        config=AdaptiveOptimizationConfig(patience=10, verbose=True),
        bc_policy=bc_policy
    )
    
    results2 = optimizer2.optimize(budget=30)
    
    print(f"\nResults:")
    print(f"  - BC fallback triggered: {results2['bc_fallback_used']}")
    print(f"  - Final reward: {results2['final_reward']:.4f}")
    print(f"  - Status: {results2['status']}")
    
    # Plot convergence histories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Plateauing optimization
    ax1.plot(results1['convergence_history'], 'b-', alpha=0.7, label='Optimization')
    if results1['bc_fallback_used']:
        fallback_iter = results1['optimization_iterations']
        ax1.axvline(x=fallback_iter, color='r', linestyle='--', label=f'BC Fallback (iter {fallback_iter})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title('Optimization with Plateau (BC Fallback)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Successful optimization
    ax2.plot(results2['convergence_history'], 'g-', alpha=0.7, label='Optimization')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reward')
    ax2.set_title('Successful Optimization (No Fallback)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Adaptive Optimization with BC Fallback', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('adaptive_optimization.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved to 'adaptive_optimization.png'")
    
    print("\n✅ Adaptive optimization demonstration complete!")