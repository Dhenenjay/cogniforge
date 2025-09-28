#!/usr/bin/env python3
"""
Demonstration of integrated logging with console and SSE output.
Shows how to use the log_event helper throughout the pipeline.
"""

import asyncio
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from cogniforge.core.logging_utils import (
    log_event, 
    EventPhase, 
    LogLevel,
    set_request_context,
    get_execution_summary,
    log_info,
    log_warning,
    log_error,
    log_success,
    log_metric
)


def simulate_pipeline_execution():
    """Simulate a complete pipeline execution with logging."""
    
    # Set up logging context
    request_id = f"sim-{int(time.time())}"
    set_request_context(request_id, f"logs/{request_id}.json")
    
    # Phase 1: Planning
    log_event(EventPhase.PLANNING, "Initializing task planner")
    time.sleep(0.5)
    log_event(EventPhase.PLANNING, "Analyzing task requirements", 
              task_type="pick_and_place", complexity=0.7)
    time.sleep(0.5)
    log_event(EventPhase.PLANNING, "Task plan generated",
              num_steps=8, estimated_time=45)
    
    # Phase 2: Expert Demonstration
    log_event(EventPhase.EXPERT_DEMONSTRATION, "Starting expert demonstration")
    for step in range(1, 4):
        time.sleep(0.3)
        log_metric(EventPhase.EXPERT_DEMONSTRATION, f"Recording step {step}",
                   step=step, position_error=0.001 * step, gripper_state=step % 2)
    log_success(EventPhase.EXPERT_DEMONSTRATION, "Expert demonstration complete",
                num_trajectories=3)
    
    # Phase 3: Behavior Cloning
    log_event(EventPhase.BEHAVIOR_CLONING, "Initializing BC training")
    for epoch in range(1, 6):
        time.sleep(0.2)
        loss = 0.5 / epoch
        accuracy = 0.8 + (0.03 * epoch)
        
        log_metric(EventPhase.BEHAVIOR_CLONING, f"Epoch {epoch} complete",
                   epoch=epoch, loss=round(loss, 4), accuracy=round(accuracy, 3),
                   learning_rate=0.001)
        
        if loss > 0.2:
            log_warning(EventPhase.BEHAVIOR_CLONING, f"High loss detected in epoch {epoch}",
                        loss=round(loss, 4), threshold=0.2)
    
    log_success(EventPhase.BEHAVIOR_CLONING, "BC training converged",
                final_loss=0.1, final_accuracy=0.95)
    
    # Phase 4: Optimization
    log_event(EventPhase.OPTIMIZATION, "Starting policy optimization")
    for step in range(1, 11):
        time.sleep(0.1)
        reward = -0.5 + (0.1 * step)
        
        log_metric(EventPhase.OPTIMIZATION, f"Optimization step {step}",
                   step=step, reward=round(reward, 3), 
                   exploration_rate=round(0.9 - (0.08 * step), 2))
        
        if step % 3 == 0:
            log_info(EventPhase.OPTIMIZATION, f"Checkpoint saved at step {step}",
                     checkpoint_id=f"ckpt_{step}")
    
    log_success(EventPhase.OPTIMIZATION, "Policy optimization complete",
                best_reward=0.45, num_iterations=10)
    
    # Phase 5: Vision Refinement
    log_event(EventPhase.VISION_REFINEMENT, "Initializing vision system")
    
    # Simulate vision detections
    detections = [
        {"dx": 5, "dy": -3, "confidence": 0.92},
        {"dx": -2, "dy": 1, "confidence": 0.88},
        {"dx": 0, "dy": 0, "confidence": 0.95}
    ]
    
    for i, detection in enumerate(detections, 1):
        time.sleep(0.3)
        log_metric(EventPhase.VISION_REFINEMENT, f"Vision detection {i}",
                   dx=detection["dx"], dy=detection["dy"], 
                   confidence=detection["confidence"])
        
        if detection["confidence"] < 0.9:
            log_warning(EventPhase.VISION_REFINEMENT, "Low confidence detection",
                        detection_id=i, confidence=detection["confidence"])
    
    log_success(EventPhase.VISION_REFINEMENT, "Vision calibration complete",
                avg_confidence=0.92, total_corrections=3)
    
    # Phase 6: Code Generation
    log_event(EventPhase.CODE_GENERATION, "Generating executable code")
    time.sleep(0.5)
    log_event(EventPhase.CODE_GENERATION, "Translating waypoints to robot commands",
              num_waypoints=8, code_lines=301)
    time.sleep(0.3)
    log_success(EventPhase.CODE_GENERATION, "Code generation complete",
                output_file="generated/pick_place.py", size_kb=12.5)
    
    # Phase 7: Execution
    log_event(EventPhase.EXECUTION, "Starting robot execution")
    
    # Simulate waypoint execution
    waypoints = ["approach", "descend", "grasp", "lift", "transport", "place", "release", "retreat"]
    for i, wp in enumerate(waypoints, 1):
        time.sleep(0.2)
        success = i != 4  # Simulate one failure
        
        if success:
            log_metric(EventPhase.EXECUTION, f"Waypoint '{wp}' executed",
                       waypoint=i, action=wp, duration=0.2)
        else:
            log_error(EventPhase.EXECUTION, f"Failed to execute waypoint '{wp}'",
                      waypoint=i, action=wp, error="Motion planning failed")
            # Recovery
            log_info(EventPhase.EXECUTION, "Attempting recovery",
                     strategy="retry", waypoint=i)
            time.sleep(0.3)
            log_success(EventPhase.EXECUTION, f"Recovery successful for waypoint '{wp}'",
                        waypoint=i, action=wp)
    
    # Phase 8: Completion
    log_success(EventPhase.COMPLETED, "Pipeline execution completed successfully",
                total_time=time.time(), success_rate=1.0, 
                waypoints_executed=len(waypoints))
    
    # Get and display summary
    summary = get_execution_summary()
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    # Phase summary
    print("\nPhases Executed:")
    for phase, data in summary['phases'].items():
        print(f"  - {phase}: {data['message_count']} messages")
    
    # Metrics summary
    print("\nAggregated Metrics:")
    for metric, stats in summary['metrics'].items():
        if isinstance(stats, dict) and 'avg' in stats:
            print(f"  - {metric}: avg={stats['avg']:.3f}, "
                  f"min={stats['min']:.3f}, max={stats['max']:.3f}")
    
    print(f"\nTotal Execution Time: {summary['total_time']:.2f} seconds")
    print(f"Request ID: {summary['request_id']}")
    
    return summary


def simulate_sse_streaming():
    """Simulate SSE streaming with mock clients."""
    
    from cogniforge.core.logging_utils import add_sse_client, remove_sse_client
    
    class MockSSEClient:
        """Mock SSE client for demonstration."""
        def __init__(self, client_id):
            self.client_id = client_id
            self.messages = []
        
        def send(self, message):
            """Mock send method."""
            self.messages.append(message)
            print(f"[SSE Client {self.client_id}] Received: {len(self.messages)} messages")
    
    # Create mock clients
    client1 = MockSSEClient("client-1")
    client2 = MockSSEClient("client-2")
    
    # Add clients
    add_sse_client(client1)
    add_sse_client(client2)
    
    # Run mini pipeline with SSE streaming
    set_request_context("sse-demo-001")
    
    log_event(EventPhase.PLANNING, "SSE Demo: Planning phase", client_count=2)
    time.sleep(0.1)
    
    log_metric(EventPhase.BEHAVIOR_CLONING, "SSE Demo: Training",
               epoch=1, loss=0.123)
    time.sleep(0.1)
    
    # Remove one client
    remove_sse_client(client1)
    
    log_success(EventPhase.COMPLETED, "SSE Demo: Complete", 
                messages_sent=3)
    
    print(f"\nClient 1 received {len(client1.messages)} messages")
    print(f"Client 2 received {len(client2.messages)} messages")


if __name__ == "__main__":
    """Run the demonstration."""
    
    print("="*60)
    print("COGNIFORGE LOGGING INTEGRATION DEMO")
    print("="*60)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run main pipeline simulation
    print("\n1. Running pipeline simulation with logging...")
    summary = simulate_pipeline_execution()
    
    # Run SSE streaming simulation
    print("\n2. Testing SSE streaming...")
    simulate_sse_streaming()
    
    print("\n" + "="*60)
    print("Demo complete! Check the logs/ directory for JSON log files.")
    print("="*60)