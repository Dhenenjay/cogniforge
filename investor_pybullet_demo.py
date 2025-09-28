#!/usr/bin/env python3
"""
CogniForge Investor Demo PyBullet
Perfect demonstration with different execution qualities
"""

import json
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

import pybullet as p
import pybullet_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvestorPyBulletDemo:
    """PyBullet demo for investor pitch - NO automatic movement"""
    
    def __init__(self):
        self.sim_id = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.running = True
        
        # Shared waypoints file
        self.waypoints_file = Path("shared_waypoints.json")
        self.last_check_time = 0
        self.last_request_id = None
        
        # Robot state
        self.current_phase = "waiting"
        self.executing = False
        self.grasped_object = None
        self.execution_count = 0
        
        # Smoothness settings for different phases
        self.smoothness_settings = {
            "expert_demonstration": {"steps": 30, "delay": 0.01, "interpolation": False},
            "behavior_cloning": {"steps": 40, "delay": 0.008, "interpolation": "partial"},
            "optimized": {"steps": 60, "delay": 0.005, "interpolation": True},
            "vision_correction": {"steps": 20, "delay": 0.005, "interpolation": True}
        }
        
    def setup_simulation(self):
        """Set up PyBullet simulation - NO automatic movement"""
        logger.info("Setting up PyBullet for investor demo...")
        
        # Connect with GUI
        self.sim_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure visualization
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # Physics
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # Manual stepping for precise control
        
        # Load environment
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Load Kuka robot
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            [0, 0, 0],
            useFixedBase=True
        )
        
        # Set robot to initial position
        self.set_robot_home()
        
        # Create scene
        self.create_investor_scene()
        
        # Set camera for best demo view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0.2, 0.2, 0.2]
        )
        
        logger.info("PyBullet ready - waiting for execution command")
        
    def create_investor_scene(self):
        """Create perfect scene for investor demo"""
        
        # Blue cube (deliberately spawn 2cm off for vision demo)
        cube_size = 0.05
        cube_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3,
            rgbaColor=[0.2, 0.3, 0.9, 1]  # Nice blue
        )
        cube_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3
        )
        
        # Spawn cube 2cm off from expected position
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.42, -0.01, 0.05]  # 2cm off intentionally
        )
        
        # Green platform
        platform_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.02],
            rgbaColor=[0.2, 0.8, 0.3, 1]  # Nice green
        )
        platform_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.02]
        )
        
        self.platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0, 0.4, 0.01]
        )
        
        # Add visual markers
        p.addUserDebugText(
            "Blue Cube",
            [0.42, -0.01, 0.15],
            textColorRGB=[0.2, 0.3, 0.9],
            textSize=1.0
        )
        
        p.addUserDebugText(
            "Target Platform",
            [0, 0.4, 0.15],
            textColorRGB=[0.2, 0.8, 0.3],
            textSize=1.0
        )
    
    def set_robot_home(self):
        """Set robot to home position - no movement"""
        home_joints = [0, -0.5, 0, -1.5, 0, 1.0, 0]
        for j in range(min(7, p.getNumJoints(self.robot_id))):
            p.resetJointState(self.robot_id, j, home_joints[j])
    
    def update_wrist_camera(self):
        """Update wrist camera view"""
        if self.robot_id is not None:
            # Get end-effector state
            link_state = p.getLinkState(self.robot_id, 6)
            end_effector_pos = link_state[0]
            
            # Wrist camera position
            camera_eye_pos = [
                end_effector_pos[0],
                end_effector_pos[1],
                end_effector_pos[2] - 0.1
            ]
            
            camera_target_pos = [
                end_effector_pos[0],
                end_effector_pos[1],
                end_effector_pos[2] - 0.3
            ]
            
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_eye_pos,
                cameraTargetPosition=camera_target_pos,
                cameraUpVector=[0, 0, 1]
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=320/240,
                nearVal=0.01,
                farVal=10
            )
            
            # Update camera preview panels
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
                
                # Handle gripper actions based on waypoint data
                gripper_value = wp.get('gripper', 0.05)
                if gripper_value < 0.01:  # Gripper should be closed
                    self.close_gripper()
                    if action == "grasp" and self.cube_id is not None:
                        self.grasped_object = self.cube_id
                elif action in ["release", "retreat"]:
                    self.open_gripper()
                    self.grasped_object = None
                # Keep gripper closed during lift, move, and place
                elif action in ["lift", "move", "place"]:
                    self.close_gripper()  # Ensure gripper stays closed
                
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
        """Check for new commands from API - NO automatic execution"""
        
        if self.waypoints_file.exists():
            try:
                mtime = self.waypoints_file.stat().st_mtime
                if mtime > self.last_check_time:
                    with open(self.waypoints_file, 'r') as f:
                        data = json.load(f)
                    
                    request_id = data.get('request_id', '')
                    waypoints = data.get('waypoints', [])
                    execution_type = data.get('execution_type', 'unknown')
                    
                    # Only execute NEW commands
                    if waypoints and not self.executing and request_id != self.last_request_id:
                        logger.info(f"NEW COMMAND: {execution_type} for {request_id}")
                        self.last_check_time = mtime
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
    demo = InvestorPyBulletDemo()
    demo.run()

if __name__ == "__main__":
    main()