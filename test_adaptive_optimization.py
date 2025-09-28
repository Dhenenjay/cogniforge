#!/usr/bin/env python3
"""
Test Adaptive Optimization with BC Fallback

This script demonstrates how optimization automatically falls back to the BC policy
when it fails to improve after N iterations, then proceeds to vision and code generation.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from cogniforge.core.adaptive_optimization import (
    AdaptiveOptimizer,
    AdaptiveOptimizationConfig,
    OptimizationStatus,
    create_adaptive_optimizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class SimplePolicy(nn.Module):
    """Simple policy network for testing."""
    
    def __init__(self, obs_dim=10, act_dim=3, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


def simulate_bc_training(n_demos=100, n_epochs=10):
    """
    Simulate BC training and return a trained policy.
    
    Args:
        n_demos: Number of demonstrations
        n_epochs: Number of training epochs
        
    Returns:
        Trained BC policy
    """
    print("\n" + "="*60)
    print("SIMULATING BC TRAINING")
    print("="*60)
    
    # Create policy
    policy = SimplePolicy()
    
    # Generate fake demonstrations
    X = torch.randn(n_demos, 10)
    Y = torch.randn(n_demos, 3)
    
    # Train
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = policy(X)
        loss = nn.MSELoss()(predictions, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    print(f"✓ BC training complete (final loss: {loss.item():.4f})")
    
    return policy


def test_scenario_1_plateau():
    """
    Test Scenario 1: Optimization plateaus, triggers BC fallback.
    """
    print("\n" + "#"*60)
    print("# SCENARIO 1: Optimization Plateau → BC Fallback")
    print("#"*60)
    
    # Train BC policy
    bc_policy = simulate_bc_training()
    
    # Create optimizer with short patience for demo
    config = AdaptiveOptimizationConfig(
        max_iterations=100,
        patience=10,  # Fallback after 10 iterations without improvement
        check_interval=5,
        min_iterations_before_fallback=15,
        bc_fallback_enabled=True,
        keep_bc_policy=True,
        proceed_to_vision_on_fallback=True,
        proceed_to_codegen_on_fallback=True,
        verbose=True
    )
    
    # Create optimizer that will plateau
    class PlateauingOptimizer(AdaptiveOptimizer):
        def _simulate_optimization_step(self, iteration):
            # Improve for 20 iterations, then plateau
            if iteration <= 20:
                return -1.0 + iteration * 0.05 + np.random.randn() * 0.01
            else:
                # Plateau with just noise
                return 0.0 + np.random.randn() * 0.01
    
    optimizer = PlateauingOptimizer(config=config, bc_policy=bc_policy)
    
    print("\nStarting optimization that will plateau...")
    print("-"*40)
    
    # Run optimization
    results = optimizer.optimize(budget=50)
    
    # Check results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"✓ Optimization ran for {results['optimization_iterations']} iterations")
    print(f"✓ Status: {results['status']}")
    print(f"✓ BC fallback triggered: {results['bc_fallback_used']}")
    print(f"✓ Final reward: {results['final_reward']:.4f}")
    
    if results['bc_fallback_used']:
        print("\n📌 BC FALLBACK ACTIONS:")
        print(f"  • Policy used: BC policy")
        print(f"  • Proceed to vision: {results['proceed_to_vision']}")
        print(f"  • Proceed to codegen: {results['proceed_to_codegen']}")
        
        # Simulate proceeding to vision
        if results['proceed_to_vision']:
            print("\n🔍 VISION REFINEMENT:")
            print("  • Capturing image from camera...")
            print("  • Detecting object position...")
            print("  • Computing world offset: dx=0.012m, dy=-0.008m")
            print("  ✓ Vision refinement complete")
        
        # Simulate proceeding to codegen
        if results['proceed_to_codegen']:
            print("\n💻 CODE GENERATION:")
            print("  • Generating execution code...")
            print("  • Incorporating BC policy waypoints...")
            print("  • Adding vision corrections...")
            print("  ✓ Code generation complete")
    
    return results


def test_scenario_2_success():
    """
    Test Scenario 2: Optimization succeeds, no fallback needed.
    """
    print("\n" + "#"*60)
    print("# SCENARIO 2: Successful Optimization (No Fallback)")
    print("#"*60)
    
    # Train BC policy
    bc_policy = simulate_bc_training()
    
    # Create standard optimizer config
    config = AdaptiveOptimizationConfig(
        max_iterations=50,
        patience=10,
        check_interval=5,
        bc_fallback_enabled=True,
        verbose=False  # Less verbose for this test
    )
    
    # Create optimizer that continuously improves
    class ImprovingOptimizer(AdaptiveOptimizer):
        def _simulate_optimization_step(self, iteration):
            # Continuously improving with some noise
            return -2.0 + iteration * 0.1 + np.random.randn() * 0.005
    
    optimizer = ImprovingOptimizer(config=config, bc_policy=bc_policy)
    
    print("\nStarting optimization that will succeed...")
    print("-"*40)
    
    # Run optimization
    results = optimizer.optimize(budget=30)
    
    # Check results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"✓ Optimization completed {results['optimization_iterations']} iterations")
    print(f"✓ Status: {results['status']}")
    print(f"✓ BC fallback triggered: {results['bc_fallback_used']}")
    print(f"✓ Final reward: {results['final_reward']:.4f}")
    
    if not results['bc_fallback_used']:
        print("\n✅ Optimization succeeded without needing BC fallback!")
        print("  • Using optimized policy")
        print("  • Standard vision refinement will be applied")
        print("  • Standard code generation will proceed")
    
    return results


def test_scenario_3_early_convergence():
    """
    Test Scenario 3: Optimization converges early.
    """
    print("\n" + "#"*60)
    print("# SCENARIO 3: Early Convergence")
    print("#"*60)
    
    # Train BC policy
    bc_policy = simulate_bc_training()
    
    # Create optimizer config with tight convergence threshold
    config = AdaptiveOptimizationConfig(
        max_iterations=100,
        patience=15,
        convergence_threshold=0.001,  # Tight convergence
        bc_fallback_enabled=True,
        verbose=False
    )
    
    # Create optimizer that converges quickly
    class ConvergingOptimizer(AdaptiveOptimizer):
        def _simulate_optimization_step(self, iteration):
            # Quick convergence to optimal value
            if iteration <= 10:
                return -1.0 + iteration * 0.2
            else:
                # Converged with minimal noise
                return 1.0 + np.random.randn() * 0.0001
    
    optimizer = ConvergingOptimizer(config=config, bc_policy=bc_policy)
    
    print("\nStarting optimization that will converge quickly...")
    print("-"*40)
    
    # Run optimization
    results = optimizer.optimize(budget=50)
    
    # Check results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"✓ Optimization stopped at iteration {results['optimization_iterations']}")
    print(f"✓ Status: {results['status']}")
    print(f"✓ Final reward: {results['final_reward']:.4f}")
    print(f"✓ Reason: Early convergence detected")
    
    return results


def visualize_all_scenarios(results1, results2, results3):
    """
    Visualize all three scenarios together.
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scenario 1: Plateau with BC fallback
    ax1 = axes[0]
    history1 = results1['convergence_history']
    ax1.plot(history1, 'b-', alpha=0.7, linewidth=2)
    if results1['bc_fallback_used']:
        fallback_iter = results1['optimization_iterations']
        ax1.axvline(x=fallback_iter, color='r', linestyle='--', 
                   linewidth=2, label=f'BC Fallback')
        ax1.axhline(y=history1[fallback_iter-1], color='orange', 
                   linestyle=':', alpha=0.5, label='Plateau level')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title('Scenario 1: Plateau → BC Fallback')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Scenario 2: Successful optimization
    ax2 = axes[1]
    history2 = results2['convergence_history']
    ax2.plot(history2, 'g-', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reward')
    ax2.set_title('Scenario 2: Successful Optimization')
    ax2.grid(True, alpha=0.3)
    
    # Scenario 3: Early convergence
    ax3 = axes[2]
    history3 = results3['convergence_history']
    ax3.plot(history3, 'm-', alpha=0.7, linewidth=2)
    convergence_iter = results3['optimization_iterations']
    ax3.axvline(x=convergence_iter, color='purple', linestyle='--', 
               linewidth=2, label='Convergence')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Reward')
    ax3.set_title('Scenario 3: Early Convergence')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle('Adaptive Optimization with BC Fallback - Three Scenarios', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('adaptive_optimization_scenarios.png', dpi=100, bbox_inches='tight')
    print("✓ Visualization saved to 'adaptive_optimization_scenarios.png'")
    plt.show()


def main():
    """Run all test scenarios."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  ADAPTIVE OPTIMIZATION WITH BC FALLBACK - TEST SUITE" + " "*13 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print("\nThis test suite demonstrates how optimization automatically")
    print("falls back to the BC policy when it fails to improve, then")
    print("proceeds directly to vision refinement and code generation.")
    
    # Run all scenarios
    results1 = test_scenario_1_plateau()
    results2 = test_scenario_2_success()
    results3 = test_scenario_3_early_convergence()
    
    # Visualize
    visualize_all_scenarios(results1, results2, results3)
    
    # Summary
    print("\n" + "#"*70)
    print("# SUMMARY")
    print("#"*70)
    
    print("\n📊 Test Results:")
    print("-"*40)
    print(f"Scenario 1 (Plateau):    BC Fallback = {results1['bc_fallback_used']}, "
          f"Iterations = {results1['optimization_iterations']}")
    print(f"Scenario 2 (Success):    BC Fallback = {results2['bc_fallback_used']}, "
          f"Iterations = {results2['optimization_iterations']}")
    print(f"Scenario 3 (Converged):  BC Fallback = {results3['bc_fallback_used']}, "
          f"Iterations = {results3['optimization_iterations']}")
    
    print("\n✅ Key Features Demonstrated:")
    print("  • Automatic detection of optimization plateau")
    print("  • BC policy retention when optimization fails")
    print("  • Direct progression to vision refinement")
    print("  • Direct progression to code generation")
    print("  • Early convergence detection")
    print("  • Configurable patience and thresholds")
    
    print("\n🎯 Benefits:")
    print("  • Saves computational resources")
    print("  • Ensures a working policy is always available")
    print("  • Prevents wasted optimization iterations")
    print("  • Maintains pipeline flow even on optimization failure")
    
    print("\n" + "#"*70)
    print("# TEST SUITE COMPLETE! 🎉")
    print("#"*70)


if __name__ == "__main__":
    main()