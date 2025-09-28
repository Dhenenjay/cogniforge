"""
Demo: GPT Fallback Mechanism in Pipeline

Shows how the pipeline continues with fallback waypoints when GPT fails,
while still logging the error for debugging.
"""

import sys
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, 'C:/Users/Dhenenjay/cogniforge')

from cogniforge.core.expert_script_with_fallback import (
    ExpertScriptWithFallback, 
    create_robust_expert
)
from cogniforge.ui.console_utils import ConsoleAutoScroller, ProgressTracker
from cogniforge.ui.vision_display import VisionOffsetDisplay

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class RobustPipeline:
    """Pipeline with GPT fallback mechanism."""
    
    def __init__(self):
        self.scroller = ConsoleAutoScroller()
        self.vision_display = VisionOffsetDisplay()
        self.expert_generator = ExpertScriptWithFallback(
            use_gpt=True,
            log_failures=True
        )
        self.execution_history = []
        
    def execute_task(
        self,
        task_description: str,
        scene: Dict[str, Any],
        simulate_gpt_failure: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a task with automatic GPT fallback.
        
        Args:
            task_description: Natural language task
            scene: Scene configuration
            simulate_gpt_failure: Force GPT failure for testing
            
        Returns:
            Execution results
        """
        print("\n" + "="*70)
        print(f"🤖 EXECUTING TASK: {task_description}")
        print("="*70)
        
        # Temporarily force failure if requested
        original_use_gpt = self.expert_generator.use_gpt
        if simulate_gpt_failure:
            self.expert_generator.use_gpt = False
            print("⚠️ Simulating GPT failure for demonstration")
        
        # Step 1: Generate trajectory (with fallback)
        print("\n📝 Step 1: Generating expert trajectory...")
        print("-"*40)
        
        start_time = time.time()
        trajectory = self.expert_generator.generate_expert_trajectory(
            task_description,
            scene,
            max_retries=2
        )
        generation_time = time.time() - start_time
        
        # Display generation results
        method = trajectory.get('generation_method', 'unknown')
        if method == 'gpt':
            print(f"✅ GPT generated trajectory successfully")
            print(f"  Attempts: {trajectory.get('gpt_attempts', 1)}")
        elif method == 'fallback':
            print(f"🔄 Using FALLBACK waypoints (GPT unavailable)")
            print(f"  Pattern: {trajectory.get('fallback_pattern', 'default')}")
            if 'metadata' in trajectory and 'warning' in trajectory['metadata']:
                self.scroller.print_and_scroll(
                    f"  ⚠️ {trajectory['metadata']['warning']}"
                )
        
        print(f"  Generation time: {generation_time:.2f}s")
        print(f"  Waypoints generated: {len(trajectory['waypoints'])}")
        
        # Step 2: Validate trajectory
        print("\n🔍 Step 2: Validating trajectory...")
        print("-"*40)
        
        validation_result = self._validate_trajectory_safety(trajectory, scene)
        if validation_result['safe']:
            print("✅ Trajectory validated as SAFE")
        else:
            print(f"⚠️ Safety concerns: {validation_result['warnings']}")
        
        # Step 3: Execute trajectory (simulated)
        print("\n🚀 Step 3: Executing trajectory...")
        print("-"*40)
        
        execution_result = self._simulate_execution(trajectory, scene)
        
        # Step 4: Log results
        result = {
            'task': task_description,
            'generation_method': method,
            'generation_time': generation_time,
            'waypoint_count': len(trajectory['waypoints']),
            'execution': execution_result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.execution_history.append(result)
        
        # Restore original setting
        self.expert_generator.use_gpt = original_use_gpt
        
        return result
    
    def _validate_trajectory_safety(
        self,
        trajectory: Dict[str, Any],
        scene: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate trajectory for safety."""
        warnings = []
        
        waypoints = trajectory['waypoints']
        
        # Check workspace bounds
        if 'workspace' in scene:
            ws = scene['workspace']
            for i, (x, y, z) in enumerate(waypoints):
                if not (ws['x_min'] <= x <= ws['x_max']):
                    warnings.append(f"Waypoint {i} X out of bounds")
                if not (ws['y_min'] <= y <= ws['y_max']):
                    warnings.append(f"Waypoint {i} Y out of bounds")
                if not (ws['z_min'] <= z <= ws['z_max']):
                    warnings.append(f"Waypoint {i} Z out of bounds")
        
        # Check for large jumps
        for i in range(1, len(waypoints)):
            prev = np.array(waypoints[i-1])
            curr = np.array(waypoints[i])
            distance = np.linalg.norm(curr - prev)
            if distance > 0.5:  # 50cm jump
                warnings.append(f"Large jump ({distance:.2f}m) between waypoints {i-1} and {i}")
        
        return {
            'safe': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _simulate_execution(
        self,
        trajectory: Dict[str, Any],
        scene: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate trajectory execution with progress."""
        
        waypoints = trajectory['waypoints']
        gripper_actions = trajectory.get('gripper_actions', {})
        
        tracker = ProgressTracker(len(waypoints), "Executing trajectory")
        
        execution_log = []
        current_pos = [0, 0, 0.5]  # Starting position
        
        for i, waypoint in enumerate(waypoints):
            # Simulate movement
            time.sleep(0.2)  # Simulate execution time
            
            # Check for gripper action
            if i in gripper_actions:
                action = gripper_actions[i]
                self.scroller.print_and_scroll(f"  💡 Gripper: {action}")
                execution_log.append(f"Gripper {action} at waypoint {i}")
            
            # Simulate vision offset (for display demo)
            dx = np.random.randint(-10, 10)
            dy = np.random.randint(-10, 10)
            self.vision_display.print_compact_status(dx, dy)
            
            # Update position
            current_pos = list(waypoint)
            
            # Log annotation if available
            if 'annotations' in trajectory and i < len(trajectory['annotations']):
                annotation = trajectory['annotations'][i]
                execution_log.append(f"Waypoint {i}: {annotation}")
            
            tracker.update(1)
        
        tracker.finish()
        
        return {
            'success': True,
            'waypoints_executed': len(waypoints),
            'final_position': current_pos,
            'execution_log': execution_log
        }
    
    def show_statistics(self):
        """Display pipeline execution statistics."""
        print("\n" + "="*70)
        print("📊 PIPELINE STATISTICS")
        print("="*70)
        
        if not self.execution_history:
            print("No executions yet")
            return
        
        # Count generation methods
        gpt_count = sum(1 for e in self.execution_history if e['generation_method'] == 'gpt')
        fallback_count = sum(1 for e in self.execution_history if e['generation_method'] == 'fallback')
        
        print(f"\nTotal executions: {len(self.execution_history)}")
        print(f"  GPT successful: {gpt_count} ({gpt_count/len(self.execution_history)*100:.1f}%)")
        print(f"  Fallback used: {fallback_count} ({fallback_count/len(self.execution_history)*100:.1f}%)")
        
        # Average statistics
        avg_waypoints = np.mean([e['waypoint_count'] for e in self.execution_history])
        avg_time = np.mean([e['generation_time'] for e in self.execution_history])
        
        print(f"\nAverage waypoints: {avg_waypoints:.1f}")
        print(f"Average generation time: {avg_time:.2f}s")
        
        # GPT failure stats
        failure_stats = self.expert_generator.get_failure_stats()
        if failure_stats['total_failures'] > 0:
            print(f"\nGPT Failures logged: {failure_stats['total_failures']}")
            print(f"Error types: {failure_stats['error_types']}")


def main():
    """Run the fallback pipeline demonstration."""
    
    print("\n" + "="*70)
    print("🔄 GPT FALLBACK PIPELINE DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows how the pipeline continues with hardcoded")
    print("3-waypoint paths when GPT fails, while logging errors.\n")
    
    # Initialize pipeline
    pipeline = RobustPipeline()
    
    # Create test scene
    scene = {
        'objects': [
            {'name': 'red_cube', 'position': [0.4, 0.0, 0.1], 'graspable': True},
            {'name': 'blue_cube', 'position': [0.6, 0.2, 0.1], 'graspable': True},
            {'name': 'platform', 'position': [0.5, 0.1, 0.05], 'graspable': False}
        ],
        'robot_state': {'position': [0.0, 0.0, 0.5]},
        'gripper_state': 'open',
        'workspace': {
            'x_min': -0.8, 'x_max': 0.8,
            'y_min': -0.8, 'y_max': 0.8,
            'z_min': 0.0, 'z_max': 1.0
        }
    }
    
    # Test cases
    tasks = [
        ("Pick up the red cube and place it on the platform", False),
        ("Move the blue cube to a new location", True),  # Simulate failure
        ("Stack the red cube on the blue cube", False),
        ("Lift and rotate the object", True),  # Simulate failure
    ]
    
    for task, simulate_failure in tasks:
        result = pipeline.execute_task(task, scene, simulate_failure)
        
        print(f"\n✅ Task completed:")
        print(f"  Method used: {result['generation_method']}")
        print(f"  Success: {result['execution']['success']}")
        
        time.sleep(1)  # Brief pause between tasks
    
    # Show statistics
    pipeline.show_statistics()
    
    # Show GPT failure log summary
    print("\n" + "="*70)
    print("📋 GPT FAILURE LOG SUMMARY")
    print("="*70)
    
    failure_stats = pipeline.expert_generator.get_failure_stats()
    if failure_stats['total_failures'] > 0:
        print(f"\nTotal GPT failures logged: {failure_stats['total_failures']}")
        print("\nMost recent failures:")
        for i, failure in enumerate(failure_stats.get('recent_failures', [])[-3:], 1):
            print(f"\n  Failure {i}:")
            print(f"    Time: {failure['timestamp']}")
            print(f"    Error: {failure['error']}")
            print(f"    Task: {failure['prompt'][:50]}...")
    else:
        print("\nNo GPT failures logged (all attempts successful or fallback used)")
    
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. Pipeline continues even when GPT fails")
    print("  2. Fallback uses hardcoded 3-waypoint paths")
    print("  3. All GPT errors are logged for debugging")
    print("  4. System remains robust and operational")


if __name__ == "__main__":
    main()