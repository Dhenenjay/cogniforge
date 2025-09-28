#!/usr/bin/env python3
"""
Demonstration of metrics tracking for CogniForge pipeline.
Shows BC loss, CMA-ES cost, PPO rewards, and vision offset tracking.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniforge.core.metrics_tracker import (
    MetricsTracker,
    track_bc_loss,
    track_cmaes_cost,
    track_ppo_reward,
    track_vision_offset,
    get_global_tracker
)
from cogniforge.core.logging_utils import set_request_context


def simulate_bc_training(num_epochs: int = 10):
    """Simulate Behavior Cloning training with loss tracking."""
    print("\n" + "="*60)
    print("BEHAVIOR CLONING TRAINING")
    print("="*60)
    
    initial_loss = 0.8
    for epoch in range(1, num_epochs + 1):
        # Simulate decreasing loss
        loss = initial_loss * np.exp(-0.2 * epoch) + np.random.normal(0, 0.01)
        accuracy = min(0.95, 0.5 + 0.05 * epoch + np.random.normal(0, 0.02))
        
        # Track the metrics
        metrics = track_bc_loss(
            epoch=epoch,
            loss=loss,
            learning_rate=0.001 * (0.9 ** (epoch // 3)),  # Decay every 3 epochs
            batch_size=32,
            accuracy=accuracy,
            validation_loss=loss * 1.1,
            gradient_norm=np.random.uniform(0.5, 2.0)
        )
        
        print(f"Epoch {epoch}: Loss={metrics.loss:.4f}, Accuracy={metrics.accuracy:.3f}")
        time.sleep(0.1)  # Simulate training time
    
    print(f"Training complete! Best loss: {get_global_tracker().best_bc_loss:.4f}")


def simulate_cmaes_optimization(num_iterations: int = 15):
    """Simulate CMA-ES optimization with cost tracking."""
    print("\n" + "="*60)
    print("CMA-ES OPTIMIZATION")
    print("="*60)
    
    initial_cost = 100.0
    sigma = 1.0
    population_size = 20
    
    for iteration in range(1, num_iterations + 1):
        # Simulate decreasing cost with some noise
        best_cost = initial_cost / (1 + 0.3 * iteration) + np.random.normal(0, 2)
        mean_cost = best_cost + np.random.uniform(5, 15)
        std_cost = np.random.uniform(3, 8)
        
        # Update sigma (step size)
        sigma *= 0.95
        
        # Check convergence
        converged = iteration >= num_iterations - 1 or best_cost < 10
        
        # Track the metrics
        metrics = track_cmaes_cost(
            iteration=iteration,
            best_cost=best_cost,
            mean_cost=mean_cost,
            std_cost=std_cost,
            population_size=population_size,
            sigma=sigma,
            converged=converged
        )
        
        print(f"Iteration {iteration}: Best Cost={metrics.best_cost:.2f}, "
              f"Sigma={metrics.sigma:.4f}, Converged={metrics.converged}")
        
        if converged:
            print("Optimization converged!")
            break
        
        time.sleep(0.1)
    
    print(f"Optimization complete! Best cost: {get_global_tracker().best_cmaes_cost:.2f}")


def simulate_ppo_training(num_episodes: int = 20):
    """Simulate PPO training with reward tracking."""
    print("\n" + "="*60)
    print("PPO TRAINING")
    print("="*60)
    
    for episode in range(1, num_episodes + 1):
        # Simulate improving rewards
        base_reward = -10 + 0.8 * episode
        avg_reward = base_reward + np.random.normal(0, 1)
        min_reward = avg_reward - np.random.uniform(2, 5)
        max_reward = avg_reward + np.random.uniform(2, 5)
        
        # Loss values
        value_loss = 0.1 / (1 + 0.05 * episode) + np.random.normal(0, 0.01)
        policy_loss = 0.05 / (1 + 0.03 * episode) + np.random.normal(0, 0.005)
        entropy = max(0.01, 0.5 - 0.02 * episode)
        
        # Track the metrics
        metrics = track_ppo_reward(
            episode=episode,
            average_reward=avg_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            value_loss=value_loss,
            policy_loss=policy_loss,
            entropy=entropy,
            explained_variance=np.random.uniform(0.8, 0.95),
            kl_divergence=np.random.uniform(0.001, 0.01)
        )
        
        print(f"Episode {episode}: Avg Reward={metrics.average_reward:.2f}, "
              f"Range=[{metrics.min_reward:.2f}, {metrics.max_reward:.2f}]")
        
        time.sleep(0.05)
    
    print(f"Training complete! Best reward: {get_global_tracker().best_ppo_reward:.2f}")


def simulate_vision_detection(num_detections: int = 10):
    """Simulate vision detections with offset tracking."""
    print("\n" + "="*60)
    print("VISION DETECTION OFFSETS")
    print("="*60)
    
    # Camera calibration: pixels to mm conversion (example: 0.5mm per pixel)
    pixel_to_mm = 0.5
    
    for i in range(1, num_detections + 1):
        # Simulate vision detections getting more accurate
        noise_level = 20 / (1 + 0.5 * i)
        
        # Pixel offsets (simulating object detection)
        dx_pixel = int(np.random.normal(0, noise_level))
        dy_pixel = int(np.random.normal(0, noise_level))
        
        # Convert to world coordinates (mm)
        dx_world = dx_pixel * pixel_to_mm
        dy_world = dy_pixel * pixel_to_mm
        
        # Confidence increases with practice
        confidence = min(0.98, 0.7 + 0.03 * i + np.random.uniform(-0.05, 0.05))
        
        # Processing time decreases with optimization
        processing_time = 50 - 2 * i + np.random.uniform(-5, 5)
        
        # Track the vision offsets
        offsets = track_vision_offset(
            dx_pixel=dx_pixel,
            dy_pixel=dy_pixel,
            dx_world=dx_world,
            dy_world=dy_world,
            confidence=confidence,
            detection_method="template_matching" if i % 2 else "deep_learning",
            depth_mm=250.0 + np.random.normal(0, 5),
            rotation_deg=np.random.uniform(-2, 2),
            scale_factor=1.0 + np.random.uniform(-0.05, 0.05),
            processing_time_ms=processing_time
        )
        
        print(f"Detection {i}: Pixel=({offsets.dx_pixel},{offsets.dy_pixel}), "
              f"World=({offsets.dx_world:.1f},{offsets.dy_world:.1f})mm, "
              f"Magnitude={offsets.world_magnitude:.1f}mm, "
              f"Aligned={offsets.is_aligned()}")
        
        time.sleep(0.1)
    
    print(f"Detection complete! Smallest offset: {get_global_tracker().smallest_vision_offset:.2f}mm")


def run_full_pipeline_simulation():
    """Run complete pipeline simulation with all metrics."""
    print("="*60)
    print("COGNIFORGE METRICS TRACKING DEMO")
    print("="*60)
    
    # Set up logging context
    request_id = f"demo-{int(time.time())}"
    set_request_context(request_id)
    
    # Initialize tracker
    tracker = get_global_tracker(request_id)
    
    # Run simulations
    simulate_bc_training(num_epochs=8)
    simulate_cmaes_optimization(num_iterations=10)
    simulate_ppo_training(num_episodes=15)
    simulate_vision_detection(num_detections=8)
    
    # Get and display summaries
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    
    # BC Summary
    bc_summary = tracker.get_bc_summary()
    if bc_summary:
        print("\nBehavior Cloning:")
        print(f"  - Total Epochs: {bc_summary['total_epochs']}")
        print(f"  - Best Loss: {bc_summary['best_loss']:.6f}")
        print(f"  - Final Loss: {bc_summary['final_loss']:.6f}")
        print(f"  - Loss Reduction: {bc_summary['loss_reduction']:.6f}")
        print(f"  - Convergence Rate: {bc_summary['convergence_rate']:.2%}")
    
    # CMA-ES Summary
    cmaes_summary = tracker.get_cmaes_summary()
    if cmaes_summary:
        print("\nCMA-ES Optimization:")
        print(f"  - Total Iterations: {cmaes_summary['total_iterations']}")
        print(f"  - Best Cost: {cmaes_summary['best_cost']:.2f}")
        print(f"  - Final Cost: {cmaes_summary['final_cost']:.2f}")
        print(f"  - Cost Reduction: {cmaes_summary['cost_reduction']:.2f}")
        print(f"  - Converged: {cmaes_summary['converged']}")
    
    # PPO Summary
    ppo_summary = tracker.get_ppo_summary()
    if ppo_summary:
        print("\nPPO Training:")
        print(f"  - Total Episodes: {ppo_summary['total_episodes']}")
        print(f"  - Best Reward: {ppo_summary['best_reward']:.2f}")
        print(f"  - Final Reward: {ppo_summary['final_reward']:.2f}")
        print(f"  - Reward Improvement: {ppo_summary['reward_improvement']:.2f}")
        print(f"  - Final Entropy: {ppo_summary['final_entropy']:.4f}")
    
    # Vision Summary
    vision_summary = tracker.get_vision_summary()
    if vision_summary:
        print("\nVision Detection:")
        print(f"  - Total Detections: {vision_summary['total_detections']}")
        print(f"  - Smallest Offset: {vision_summary['smallest_offset_mm']:.2f}mm")
        print(f"  - Average Offset: {vision_summary['average_offset_mm']:.2f}mm")
        print(f"  - Alignment Rate: {vision_summary['alignment_rate']:.2%}")
        print(f"  - Average Confidence: {vision_summary['average_confidence']:.3f}")
        print(f"  - Average Pixel Offset: ({vision_summary['average_dx_pixel']:.1f}, "
              f"{vision_summary['average_dy_pixel']:.1f})")
        print(f"  - Average World Offset: ({vision_summary['average_dx_world']:.1f}, "
              f"{vision_summary['average_dy_world']:.1f})mm")
    
    # Save metrics
    filepath = tracker.save_metrics()
    print(f"\n📊 Metrics saved to: {filepath}")
    
    # Generate plots if matplotlib is available
    plot_path = tracker.plot_metrics(save_plots=True)
    if plot_path:
        print(f"📈 Plots saved to: {plot_path}")
    
    # Get full summary
    full_summary = tracker.get_full_summary()
    print(f"\n⏱️ Total Duration: {full_summary['duration_seconds']:.2f} seconds")
    
    return tracker


if __name__ == "__main__":
    # Create metrics directory
    Path("metrics").mkdir(exist_ok=True)
    
    # Run the simulation
    tracker = run_full_pipeline_simulation()
    
    print("\n" + "="*60)
    print("✅ Metrics tracking demo complete!")
    print("Check the 'metrics' directory for saved data and plots.")
    print("="*60)