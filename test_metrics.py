#!/usr/bin/env python3
"""Simple test of metrics tracking."""

import sys
import time
import random
sys.path.insert(0, '.')

from cogniforge.core.metrics_tracker import MetricsTracker

# Create tracker
tracker = MetricsTracker('test-metrics-001')

print("Testing Metrics Tracking")
print("="*40)

# Track phase timings and success
phase_times = {}
phase_success = {}
final_distances = {}

# Track BC epochs
print("\n1. Tracking BC Loss per Epoch:")
start_time = time.time()
for epoch in range(1, 6):
    loss = 1.0 / epoch
    metrics = tracker.track_bc_epoch(
        epoch=epoch, 
        loss=loss, 
        learning_rate=0.001,
        accuracy=0.8 + 0.02 * epoch
    )
    print(f"   Epoch {epoch}: Loss={loss:.4f}")
    time.sleep(0.1)  # Simulate processing
phase_times['BC Training'] = time.time() - start_time
phase_success['BC Training'] = loss < 0.5  # Success if final loss < 0.5

# Track CMA-ES iterations  
print("\n2. Tracking CMA-ES Cost per Iteration:")
start_time = time.time()
for iter in range(1, 6):
    cost = 100.0 / iter
    metrics = tracker.track_cmaes_iteration(
        iteration=iter,
        best_cost=cost,
        mean_cost=cost * 1.2,
        std_cost=cost * 0.1,
        population_size=20,
        sigma=1.0 * (0.9 ** iter)
    )
    print(f"   Iteration {iter}: Cost={cost:.2f}")
    time.sleep(0.1)  # Simulate processing
phase_times['CMA-ES Optimization'] = time.time() - start_time
phase_success['CMA-ES Optimization'] = cost < 30  # Success if final cost < 30
final_distances['CMA-ES Cost'] = cost

# Track PPO episodes
print("\n3. Tracking PPO Average Reward:")
start_time = time.time()
for ep in range(1, 6):
    reward = -10.0 + ep * 2.0
    metrics = tracker.track_ppo_episode(
        episode=ep,
        average_reward=reward,
        min_reward=reward - 2,
        max_reward=reward + 2,
        value_loss=0.1 / ep,
        policy_loss=0.05 / ep,
        entropy=0.5 - 0.05 * ep
    )
    print(f"   Episode {ep}: Reward={reward:.2f}")
    time.sleep(0.1)  # Simulate processing
phase_times['PPO Training'] = time.time() - start_time
phase_success['PPO Training'] = reward > -2  # Success if final reward > -2
final_distances['PPO Reward Gap'] = abs(10.0 - reward)  # Gap from target reward of 10

# Track vision offsets
print("\n4. Tracking Vision Pixel/World Offsets:")
start_time = time.time()
for i in range(1, 6):
    dx_pixel = 10 - i * 2
    dy_pixel = 5 - i
    dx_world = dx_pixel * 0.5  # 0.5mm per pixel
    dy_world = dy_pixel * 0.5
    
    offsets = tracker.track_vision_offset(
        dx_pixel=dx_pixel,
        dy_pixel=dy_pixel,
        dx_world=dx_world,
        dy_world=dy_world,
        confidence=0.9 + i * 0.02
    )
    print(f"   Detection {i}: Pixel=({dx_pixel},{dy_pixel}), World=({dx_world:.1f},{dy_world:.1f})mm")
    time.sleep(0.05)  # Simulate processing
    
    # Store final offset distance
    if i == 5:
        final_pixel_distance = (dx_pixel**2 + dy_pixel**2)**0.5
        final_world_distance = (dx_world**2 + dy_world**2)**0.5
        
phase_times['Vision Detection'] = time.time() - start_time
phase_success['Vision Detection'] = final_world_distance < 2.0  # Success if final offset < 2mm
final_distances['Vision Pixel Offset'] = final_pixel_distance
final_distances['Vision World Offset (mm)'] = final_world_distance

# Get summaries
print("\n" + "="*40)
print("SUMMARIES")
print("="*40)

bc_summary = tracker.get_bc_summary()
print("\nBC Training:")
print(f"  Best Loss: {bc_summary['best_loss']:.4f}")
print(f"  Final Loss: {bc_summary['final_loss']:.4f}") 
print(f"  Total Epochs: {bc_summary['total_epochs']}")

cmaes_summary = tracker.get_cmaes_summary()
print("\nCMA-ES:")
print(f"  Best Cost: {cmaes_summary['best_cost']:.2f}")
print(f"  Final Cost: {cmaes_summary['final_cost']:.2f}")
print(f"  Total Iterations: {cmaes_summary['total_iterations']}")

ppo_summary = tracker.get_ppo_summary()
print("\nPPO:")
print(f"  Best Reward: {ppo_summary['best_reward']:.2f}")
print(f"  Final Reward: {ppo_summary['final_reward']:.2f}")
print(f"  Total Episodes: {ppo_summary['total_episodes']}")

vision_summary = tracker.get_vision_summary()
print("\nVision:")
print(f"  Smallest Offset: {vision_summary['smallest_offset_mm']:.2f}mm")
print(f"  Average Offset: {vision_summary['average_offset_mm']:.2f}mm")
print(f"  Alignment Rate: {vision_summary['alignment_rate']:.1%}")
print(f"  Total Detections: {vision_summary['total_detections']}")

# Save metrics
filepath = tracker.save_metrics()
print(f"\nMetrics saved to: {filepath}")

# Print comprehensive summary
print("\n" + "="*60)
print("COMPREHENSIVE PIPELINE SUMMARY")
print("="*60)

# Phase Timing Summary
print("\n1. TIME PER PHASE:")
print("-" * 40)
total_time = sum(phase_times.values())
for phase, duration in phase_times.items():
    percentage = (duration / total_time) * 100
    print(f"   {phase:<25} {duration:6.2f}s ({percentage:5.1f}%)")
print(f"   {'TOTAL':<25} {total_time:6.2f}s (100.0%)")

# Success Flags Summary
print("\n2. PHASE SUCCESS FLAGS:")
print("-" * 40)
for phase, success in phase_success.items():
    status = "[PASS]" if success else "[FAIL]"
    symbol = "✓" if success else "✗"
    print(f"   {phase:<25} {status:^8} {symbol}")

total_phases = len(phase_success)
successful_phases = sum(1 for s in phase_success.values() if s)
success_rate = (successful_phases / total_phases) * 100
print(f"\n   Overall Success Rate: {successful_phases}/{total_phases} ({success_rate:.0f}%)")

# Final Distances Summary
print("\n3. FINAL DISTANCES & METRICS:")
print("-" * 40)
for metric, value in final_distances.items():
    if "mm" in metric:
        print(f"   {metric:<30} {value:8.2f} mm")
    elif "Pixel" in metric:
        print(f"   {metric:<30} {value:8.2f} pixels")
    elif "Cost" in metric:
        print(f"   {metric:<30} {value:8.2f}")
    elif "Gap" in metric:
        print(f"   {metric:<30} {value:8.2f}")
    else:
        print(f"   {metric:<30} {value:8.2f}")

# Key Performance Indicators
print("\n4. KEY PERFORMANCE INDICATORS:")
print("-" * 40)

# Calculate aggregated metrics
avg_processing_time = total_time / 4  # 4 phases
bc_final = tracker.get_bc_summary()['final_loss']
cmaes_final = tracker.get_cmaes_summary()['final_cost']
ppo_final = tracker.get_ppo_summary()['final_reward']
vision_avg = tracker.get_vision_summary()['average_offset_mm']

print(f"   Average Phase Time:          {avg_processing_time:.2f}s")
print(f"   BC Final Loss:               {bc_final:.4f}")
print(f"   CMA-ES Final Cost:           {cmaes_final:.2f}")
print(f"   PPO Final Reward:            {ppo_final:.2f}")
print(f"   Vision Avg Offset:           {vision_avg:.2f}mm")
print(f"   Pipeline Success:            {'YES' if all(phase_success.values()) else 'NO'}")

# Final Status
print("\n" + "="*60)
if all(phase_success.values()):
    print("STATUS: PIPELINE COMPLETED SUCCESSFULLY")
else:
    failed_phases = [p for p, s in phase_success.items() if not s]
    print(f"STATUS: PIPELINE FAILED - Issues in: {', '.join(failed_phases)}")
print("="*60)
