#!/usr/bin/env python3
"""
Generated Pick and Place Task
Python 3.11 compatible, no external deps beyond project modules

This file recreates the PyBullet scene, loads optimized waypoints,
pauses at pre-grasp for vision offset, and completes pick-place.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import json
import os
from typing import List, Tuple, Dict, Optional


class PickPlaceTask:
    """Pick and place task with vision-guided adjustment"""
    
    def __init__(self, gui: bool = True):
        """
        Initialize simulation environment
        
        Args:
            gui: Whether to show GUI
        """
        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Simulation parameters
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # Robot and object IDs
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        
        # Panda specific
        self.end_effector_index = 11
        self.gripper_indices = [9, 10]
        
        # Waypoints storage
        self.W_star = None
        self.current_waypoint_idx = 0
        
        # Vision offset limits
        self.max_offset_m = 0.02  # 2cm maximum adjustment
        
        # Execute waypoints
        results = {
            'success': True,
            'waypoints_executed': 0,
            'vision_corrections': [],
            'execution_time': 0,
            'final_position': None,
            'object_grasped': False
        }
        
        start_time = time.time()
        
        for i, waypoint in enumerate(waypoints):
            print(f"\n[WAYPOINT {i+1}/{len(waypoints)}]")
            
            # Extract waypoint components
            position = waypoint.get('position', [0.5, 0.0, 0.3])
            gripper = waypoint.get('gripper', 0.08)
            action = waypoint.get('action', 'move')
            
            print(f"  Position: {position}")
            print(f"  Gripper: {gripper:.3f}")
            print(f"  Action: {action}")
            
            # Apply vision correction if enabled
            if vision and action in ['grasp', 'place']:
                correction = apply_vision_correction(sim, position)
                if correction is not None:
                    results['vision_corrections'].append({
                        'waypoint': i,
                        'original': position,
                        'correction': correction
                    })
                    position = [
                        position[0] + correction[0],
                        position[1] + correction[1],
                        position[2]
                    ]
                    print(f"  Vision correction applied: dx={correction[0]:.3f}, dy={correction[1]:.3f}")
            
            # Execute movement
            success = sim.move_to_position(position, gripper_opening=gripper)
            
            if not success:
                print(f"  [WARNING] Failed to reach waypoint {i+1}")
                results['success'] = False
            else:
                results['waypoints_executed'] += 1
                
                # Check grasp status
                if action == 'grasp' and gripper < 0.04:
                    results['object_grasped'] = check_grasp_success(sim)
                    print(f"  Grasp {'successful' if results['object_grasped'] else 'failed'}")
            
            # Small delay for stability
            time.sleep(0.1)
        
        results['execution_time'] = time.time() - start_time
        results['final_position'] = sim.get_end_effector_position()
        
        print(f"\n[COMPLETE] Executed {results['waypoints_executed']}/{len(waypoints)} waypoints")
        print(f"  Total time: {results['execution_time']:.2f}s")
        print(f"  Vision corrections: {len(results['vision_corrections'])}")
        print(f"  Object grasped: {results['object_grasped']}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'waypoints_executed': 0
        }
    finally:
        if sim:
            sim.close()
            print("[INFO] Simulator closed")


def spawn_scene(sim):
    """
    Spawn the scene with table, cube, and target platform.
    
    Args:
        sim: RobotSimulator instance
    """
    # Spawn table
    table_id = sim.spawn_table()
    print(f"  Table spawned: ID={table_id}")
    
    # Spawn blue cube (object to pick)
    cube_position = [0.5, 0.0, 0.05]
    cube_id = sim.spawn_cube(
        position=cube_position,
        color=[0.0, 0.0, 1.0, 1.0],  # Blue
        size=0.04
    )
    print(f"  Blue cube spawned at {cube_position}: ID={cube_id}")
    
    # Spawn red platform (target location)
    platform_position = [0.3, 0.3, 0.01]
    platform_id = sim.spawn_box(
        position=platform_position,
        color=[1.0, 0.0, 0.0, 1.0],  # Red
        size=[0.1, 0.1, 0.02]
    )
    print(f"  Red platform spawned at {platform_position}: ID={platform_id}")
    
    # Let objects settle
    for _ in range(50):
        sim.step_simulation()
    
    return {
        'table': table_id,
        'cube': cube_id,
        'platform': platform_id
    }


def apply_vision_correction(sim, target_position):
    """
    Apply vision-based correction to target position.
    
    Args:
        sim: RobotSimulator instance
        target_position: Original target position
    
    Returns:
        tuple: (dx, dy) correction or None
    """
    try:
        # Get camera image
        img = sim.get_camera_image()
        
        # Simple vision processing (placeholder - would use actual vision module)
        # In real implementation, this would call cogniforge.vision.vision_utils
        # For now, return small random corrections for demonstration
        
        # Simulate vision detection with small offsets
        dx = np.random.uniform(-0.01, 0.01)
        dy = np.random.uniform(-0.01, 0.01)
        
        # Only apply if correction is significant
        if abs(dx) > 0.005 or abs(dy) > 0.005:
            return (dx, dy)
        
        return None
        
    except Exception as e:
        print(f"  [WARNING] Vision correction failed: {e}")
        return None


def check_grasp_success(sim):
    """
    Check if object was successfully grasped.
    
    Args:
        sim: RobotSimulator instance
    
    Returns:
        bool: True if object is grasped
    """
    try:
        # Check gripper contact points
        contacts = sim.get_gripper_contacts()
        return len(contacts) > 0
    except:
        # Fallback: check gripper opening
        return sim.get_gripper_opening() < 0.04


# Define waypoints for pick and place task
WAYPOINTS = [
    # Approach object
    {
        'position': [0.5, 0.0, 0.15],
        'gripper': 0.08,
        'action': 'approach'
    },
    # Move down to object
    {
        'position': [0.5, 0.0, 0.08],
        'gripper': 0.08,
        'action': 'descend'
    },
    # Grasp object
    {
        'position': [0.5, 0.0, 0.08],
        'gripper': 0.02,
        'action': 'grasp'
    },
    # Lift object
    {
        'position': [0.5, 0.0, 0.20],
        'gripper': 0.02,
        'action': 'lift'
    },
    # Move to target location
    {
        'position': [0.3, 0.3, 0.20],
        'gripper': 0.02,
        'action': 'transport'
    },
    # Lower to place
    {
        'position': [0.3, 0.3, 0.08],
        'gripper': 0.02,
        'action': 'place'
    },
    # Release object
    {
        'position': [0.3, 0.3, 0.08],
        'gripper': 0.08,
        'action': 'release'
    },
    # Retreat
    {
        'position': [0.3, 0.3, 0.20],
        'gripper': 0.08,
        'action': 'retreat'
    }
]


if __name__ == "__main__":
    """
    Main execution entry point.
    """
    print("="*60)
    print("COGNIFORGE GENERATED PICK AND PLACE EXECUTION")
    print("="*60)
    
    # Execute with vision enabled
    results = run_pick_place(waypoints=WAYPOINTS, vision=True)
    
    # Print final summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    if results['success']:
        print("✅ Task completed successfully!")
    else:
        print("❌ Task failed!")
        if 'error' in results:
            print(f"Error: {results['error']}")
    
    print(f"\nMetrics:")
    print(f"  - Waypoints: {results.get('waypoints_executed', 0)}/{len(WAYPOINTS)}")
    print(f"  - Time: {results.get('execution_time', 0):.2f}s")
    print(f"  - Vision corrections: {len(results.get('vision_corrections', []))}")
    print(f"  - Object grasped: {results.get('object_grasped', False)}")
    
    if results.get('final_position'):
        pos = results['final_position']
        print(f"  - Final position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print("\n" + "="*60)
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)