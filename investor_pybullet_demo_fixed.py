#!/usr/bin/env python3
"""
INVESTOR DEMO PyBullet Simulation - Fixed Version
Proper cleanup on startup, no spurious executions
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import json
import logging
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class WaypointCommand:
    """Waypoint command structure"""
    request_id: str
    waypoints: List[Dict]
    execution_type: str
    timestamp: float

class InvestorPyBulletDemo:
    """Production-ready PyBullet demo for investor pitch"""
    
    def __init__(self):
        self.client = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.table_id = None
        
        # State tracking
        self.running = True
        self.executing = False
        self.current_phase = "waiting"
        self.execution_count = 0
        self.grasped_object = None
        
        # Command tracking
        self.waypoints_file = Path("shared_waypoints.json")
        self.last_request_id = None
        self.processed_requests = set()  # Track all processed requests
        
        # Smoothness settings per phase
        self.smoothness_settings = {
            "expert_demonstration": {
                "steps": 15,  # Fewer steps = jerkier
                "delay": 0.01,
                "interpolation": False  # No smoothing
            },
            "behavior_cloning": {
                "steps": 30,  # Medium smoothing
                "delay": 0.008,
                "interpolation": "partial"
            },
            "optimized": {
                "steps": 40,  # More steps = smoother
                "delay": 0.006,
                "interpolation": True
            },
            "vision_correction": {
                "steps": 35,
                "delay": 0.007,
                "interpolation": True
            }
        }
        
    def setup_simulation(self):
        """Initialize PyBullet with investor demo configuration"""
        
        # Clean up old state file on startup
        if self.waypoints_file.exists():
            try:
                self.waypoints_file.unlink()
                logger.info("Cleaned up old waypoints file on startup")
            except:
                pass
        
        # Connect to PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure simulation
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Camera for investor view
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3]
        )
        
        # Setup environment
        self.setup_environment()
        
        # Load and configure robot
        self.load_robot()
        
        # Set to home position
        self.set_robot_home()
        
        logger.info("PyBullet ready - waiting for execution command")
    
    def setup_environment(self):
        """Create investor demo environment"""
        
        # Load plane
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Create table
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.8, 0.8, 0.4]
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.8, 0.8, 0.4],
            rgbaColor=[0.6, 0.4, 0.2, 1]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0, 0, 0.4]
        )
        
        # Create blue cube - DELIBERATELY 2cm off for vision demo
        cube_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.03]
        )
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.03],
            rgbaColor=[0.2, 0.2, 0.8, 1]
        )
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.42, -0.01, 0.05]  # 2cm offset for vision correction demo
        )
        
        # Create green platform
        platform_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.02]
        )
        platform_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.02],
            rgbaColor=[0.2, 0.8, 0.2, 1]
        )
        self.platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0, 0.4, 0.02]
        )
        
        # Add visual markers
        p.addUserDebugText(
            "BLUE CUBE\n(2cm offset)",
            [0.42, -0.01, 0.15],
            textColorRGB=[0, 0, 1],
            textSize=0.8
        )
        
        p.addUserDebugText(
            "TARGET",
            [0, 0.4, 0.15],
            textColorRGB=[0, 1, 0],
            textSize=0.8
        )
    
    def load_robot(self):
        """Load UR5e robot"""
        try:
            # Try to load UR5e
            self.robot_id = p.loadURDF(
                "ur5e/ur5e.urdf",
                basePosition=[0, 0, 0.0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION
            )
            logger.info(f"Loaded UR5e with {p.getNumJoints(self.robot_id)} joints")
        except:
            # Fallback to Franka Panda
            self.robot_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=[0, 0, 0.0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            )
            logger.info(f"Loaded Franka Panda with {p.getNumJoints(self.robot_id)} joints")
    
    def set_robot_home(self):
        """Set robot to home position"""
        home_joints = [0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0, 0]
        for i in range(min(7, p.getNumJoints(self.robot_id))):
            p.resetJointState(self.robot_id, i, home_joints[i])
        self.open_gripper()
    
    def update_wrist_camera(self):
        """Update wrist-mounted camera view"""
        if self.robot_id is None or p.getNumJoints(self.robot_id) < 7:
            return
            
        ee_state = p.getLinkState(self.robot_id, 6)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        
        # Compute camera matrices
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=ee_pos,
            distance=0.3,
            yaw=0,
            pitch=-90,
            roll=0,
            upAxisIndex=2
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.33,
            nearVal=0.01,
            farVal=10.0
        )
        
        # Optional: capture image for vision system
        if self.current_phase == "vision_correction":
            p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )
    
    def execute_trajectory_with_quality(self, waypoints: List[Dict], execution_type: str):
        """Execute trajectory with different quality based on phase"""
        
        logger.info(f"Executing {execution_type} with {len(waypoints)} waypoints")
        self.executing = True
        self.current_phase = execution_type
        
        # Get smoothness settings for this execution type
        settings = self.smoothness_settings.get(
            execution_type,
            {"steps": 40, "delay": 0.008, "interpolation": True}
        )
        
        try:
            previous_joints = None
            
            for i, wp in enumerate(waypoints):
                if not self.running:
                    break
                
                target_pos = [wp.get('x', 0), wp.get('y', 0), wp.get('z', 0)]
                action = wp.get('action', 'move')
                
                # Calculate IK
                joint_poses = p.calculateInverseKinematics(
                    self.robot_id,
                    6,
                    target_pos,
                    maxNumIterations=100
                )
                
                # Apply interpolation based on execution type
                if settings["interpolation"] and previous_joints is not None:
                    # Smooth interpolation
                    for step in range(settings["steps"]):
                        alpha = (step + 1) / settings["steps"]
                        
                        if settings["interpolation"] == "partial":
                            # Partial smoothing for BC
                            alpha = alpha ** 0.7
                        elif settings["interpolation"] is True:
                            # Full smoothing for optimized
                            alpha = 0.5 * (1 - np.cos(np.pi * alpha))
                        
                        for j in range(min(7, p.getNumJoints(self.robot_id))):
                            if previous_joints:
                                interpolated = previous_joints[j] + alpha * (joint_poses[j] - previous_joints[j])
                            else:
                                interpolated = joint_poses[j]
                            
                            p.setJointMotorControl2(
                                self.robot_id,
                                j,
                                p.POSITION_CONTROL,
                                targetPosition=interpolated,
                                force=500,
                                maxVelocity=2.0 if execution_type == "expert_demonstration" else 1.0
                            )
                        
                        p.stepSimulation()
                        time.sleep(settings["delay"])
                        
                        # Update camera every few steps
                        if step % 5 == 0:
                            self.update_wrist_camera()
                else:
                    # Jerky movement for expert demonstration
                    for j in range(min(7, p.getNumJoints(self.robot_id))):
                        p.setJointMotorControl2(
                            self.robot_id,
                            j,
                            p.POSITION_CONTROL,
                            targetPosition=joint_poses[j],
                            force=500,
                            maxVelocity=3.0  # Faster for jerky look
                        )
                    
                    # Fewer steps for jerkier motion
                    for _ in range(settings["steps"]):
                        p.stepSimulation()
                        time.sleep(settings["delay"])
                    
                    self.update_wrist_camera()
                
                # Handle gripper actions
                if action == "grasp":
                    self.close_gripper()
                    if self.cube_id is not None:
                        self.grasped_object = self.cube_id
                elif action in ["release", "place"]:
                    self.open_gripper()
                    self.grasped_object = None
                
                # Move grasped object with gripper
                if self.grasped_object is not None:
                    ee_state = p.getLinkState(self.robot_id, 6)
                    ee_pos = ee_state[0]
                    p.resetBasePositionAndOrientation(
                        self.grasped_object,
                        [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05],
                        [0, 0, 0, 1]
                    )
                
                # Visual feedback
                quality_text = {
                    "expert_demonstration": "EXPERT (Jerky)",
                    "behavior_cloning": "BC (Smoother)",
                    "optimized": "OPTIMIZED (Smooth)",
                    "vision_correction": "VISION CORRECTION"
                }.get(execution_type, execution_type.upper())
                
                p.addUserDebugText(
                    f"{quality_text}: {action} ({i+1}/{len(waypoints)})",
                    target_pos,
                    textColorRGB=[0, 1, 0],
                    textSize=1.0,
                    lifeTime=2.0
                )
                
                previous_joints = joint_poses
                
                logger.info(f"  Waypoint {i+1}: {action} at {target_pos}")
            
            logger.info(f"{execution_type} completed")
            
            # Reset after each execution phase
            if execution_type != "vision_correction":
                time.sleep(1)
                self.reset_scene_for_next_phase()
            
        except Exception as e:
            logger.error(f"Error during {execution_type}: {e}")
        finally:
            self.executing = False
            self.current_phase = "ready"
    
    def close_gripper(self):
        """Close gripper"""
        if p.getNumJoints(self.robot_id) >= 9:
            p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=0.0, force=100)
            p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, targetPosition=0.0, force=100)
    
    def open_gripper(self):
        """Open gripper"""
        if p.getNumJoints(self.robot_id) >= 9:
            p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=0.05, force=100)
            p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, targetPosition=0.05, force=100)
    
    def reset_scene_for_next_phase(self):
        """Reset scene between execution phases"""
        logger.info("Resetting scene for next phase...")
        
        # Reset cube to slightly off position
        if self.cube_id is not None:
            p.resetBasePositionAndOrientation(
                self.cube_id,
                [0.42, -0.01, 0.05],  # Keep 2cm off
                [0, 0, 0, 1]
            )
        
        # Reset robot to home
        self.set_robot_home()
        self.grasped_object = None
        
        # Step simulation to settle
        for _ in range(50):
            p.stepSimulation()
            time.sleep(0.01)
    
    def check_for_commands(self):
        """Check for new commands from API - only process new unique requests"""
        
        if self.waypoints_file.exists():
            try:
                with open(self.waypoints_file, 'r') as f:
                    data = json.load(f)
                
                request_id = data.get('request_id', '')
                waypoints = data.get('waypoints', [])
                execution_type = data.get('execution_type', 'unknown')
                
                # Only execute if this is a NEW request we haven't processed
                if waypoints and not self.executing and request_id not in self.processed_requests:
                    logger.info(f"NEW COMMAND: {execution_type} for {request_id}")
                    self.processed_requests.add(request_id)
                    self.last_request_id = request_id
                    self.execution_count += 1
                    
                    # Execute in separate thread with appropriate quality
                    thread = threading.Thread(
                        target=self.execute_trajectory_with_quality,
                        args=(waypoints, execution_type)
                    )
                    thread.daemon = True
                    thread.start()
                    
            except Exception as e:
                pass  # Silent fail for file read
    
    def run(self):
        """Main loop - NO automatic movement"""
        
        # Setup
        self.setup_simulation()
        
        # Main loop
        frame = 0
        
        while self.running and p.isConnected():
            
            # Check for commands every 5 frames
            if frame % 5 == 0:
                self.check_for_commands()
            
            # Update camera periodically
            if frame % 30 == 0:
                self.update_wrist_camera()
            
            # Display status
            phase_text = {
                "waiting": "WAITING FOR COMMAND",
                "expert_demonstration": "EXPERT DEMO (Jerky)",
                "behavior_cloning": "BC EXECUTION (Better)",
                "optimized": "OPTIMIZED (Smooth)",
                "vision_correction": "VISION CORRECTION",
                "ready": "READY"
            }.get(self.current_phase, self.current_phase.upper())
            
            status_color = [0.5, 0.5, 0.5] if self.current_phase == "waiting" else [0, 1, 0]
            if self.executing:
                status_color = [1, 0.5, 0]
            
            p.addUserDebugText(
                f"CogniForge-V | {phase_text}",
                [-0.5, -0.5, 0.5],
                textColorRGB=status_color,
                textSize=1.2,
                lifeTime=0,
                replaceItemUniqueId=100
            )
            
            p.addUserDebugText(
                f"Executions: {self.execution_count}",
                [-0.5, -0.5, 0.6],
                textColorRGB=[0.8, 0.8, 0.8],
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=101
            )
            
            # Check for quit
            keys = p.getKeyboardEvents()
            if keys.get(ord('q')) or keys.get(ord('Q')):
                self.running = False
            
            frame += 1
            time.sleep(1/60)  # 60 FPS
        
        p.disconnect()
        logger.info("Investor demo ended")

def main():
    """Main execution with proper initialization"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Clean up any old shared files on startup
    shared_file = Path("shared_waypoints.json")
    if shared_file.exists():
        try:
            shared_file.unlink()
            logger.info("Cleaned up old shared file on startup")
        except:
            pass
    
    demo = InvestorPyBulletDemo()
    demo.run()

if __name__ == "__main__":
    main()