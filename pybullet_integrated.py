#!/usr/bin/env python3
"""
CogniForge PyBullet Integrated Demo
Real execution with actual robot control
"""

import json
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
import requests

import numpy as np
import pybullet as p
import pybullet_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedPyBulletDemo:
    """PyBullet demo integrated with real CogniForge execution"""
    
    def __init__(self):
        self.api_url = "http://localhost:8001"  # Updated port
        self.sim_id = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.running = True
        
        # Shared waypoints file
        self.waypoints_file = Path("shared_waypoints.json")
        self.last_waypoint_time = 0
        self.last_request_id = None
        
        # Robot state
        self.current_phase = "idle"
        self.api_connected = False
        self.executing = False
        self.grasped_object = None
        self.execution_count = 0
        
    def setup_simulation(self):
        """Set up PyBullet simulation with real robot"""
        logger.info("Setting up PyBullet simulation...")
        
        # Connect with GUI
        self.sim_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Enable camera previews (wrist camera view)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # Physics
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        
        # Load environment
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Load Kuka robot (real robot model)
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            [0, 0, 0],
            useFixedBase=True
        )
        
        # Create real scene objects
        self.create_real_scene()
        
        # Set camera for better view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3]
        )
        
        logger.info("PyBullet simulation ready")
        
    def create_real_scene(self):
        """Create real scene objects matching CogniForge execution"""
        
        # Blue cube (object to pick)
        cube_size = 0.05
        cube_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3,
            rgbaColor=[0, 0, 1, 1]  # Blue
        )
        cube_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3
        )
        
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.4, 0, 0.05]  # Real position
        )
        
        # Green platform (target location)
        platform_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.02],
            rgbaColor=[0, 1, 0, 1]  # Green
        )
        platform_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.02]
        )
        
        self.platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0, 0.4, 0.01]  # Real position
        )
        
    def update_wrist_camera(self):
        """Update wrist camera preview panels"""
        
        if self.robot_id is not None:
            # Get end-effector state
            link_state = p.getLinkState(self.robot_id, 6)
            end_effector_pos = link_state[0]
            end_effector_orn = link_state[1]
            
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
            
            # Update camera images for preview panels
            p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )
    
    def execute_real_trajectory(self, waypoints: List[Dict[str, Any]]):
        """Execute real trajectory from CogniForge"""
        
        logger.info(f"Executing real trajectory with {len(waypoints)} waypoints")
        self.executing = True
        self.current_phase = "executing"
        
        try:
            for i, wp in enumerate(waypoints):
                if not self.running:
                    break
                    
                target_pos = [wp.get('x', 0), wp.get('y', 0), wp.get('z', 0)]
                action = wp.get('action', 'move')
                
                # Calculate inverse kinematics
                joint_poses = p.calculateInverseKinematics(
                    self.robot_id,
                    6,  # End effector link
                    target_pos,
                    maxNumIterations=100
                )
                
                # Move joints smoothly
                for j in range(min(7, p.getNumJoints(self.robot_id))):
                    p.setJointMotorControl2(
                        self.robot_id,
                        j,
                        p.POSITION_CONTROL,
                        targetPosition=joint_poses[j],
                        force=500,
                        maxVelocity=1.0
                    )
                
                # Gripper control based on action
                if action == "grasp":
                    self.close_gripper()
                    # Attach cube to gripper (simulate grasping)
                    if self.cube_id is not None:
                        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
                        if abs(cube_pos[0] - target_pos[0]) < 0.1 and abs(cube_pos[1] - target_pos[1]) < 0.1:
                            # Move cube with gripper
                            self.grasped_object = self.cube_id
                elif action == "release" or action == "place":
                    self.open_gripper()
                    self.grasped_object = None
                
                # Step simulation with more steps for smoother motion
                for step in range(60):  # Increased steps
                    p.stepSimulation()
                    time.sleep(1/240)
                    
                    # Move grasped object with gripper
                    if self.grasped_object is not None:
                        ee_state = p.getLinkState(self.robot_id, 6)
                        ee_pos = ee_state[0]
                        p.resetBasePositionAndOrientation(
                            self.grasped_object,
                            [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05],
                            [0, 0, 0, 1]
                        )
                    
                    # Update camera every 10 steps
                    if step % 10 == 0:
                        self.update_wrist_camera()
                
                # Visual feedback
                p.addUserDebugText(
                    f"Action: {action} ({i+1}/{len(waypoints)})",
                    target_pos,
                    textColorRGB=[0, 1, 0],
                    textSize=1.0,
                    lifeTime=1.0
                )
                
                logger.info(f"Waypoint {i+1}: {action} at {target_pos}")
            
            logger.info("Trajectory execution completed successfully")
            
            # Reset robot to home position after completion
            time.sleep(1)
            self.reset_to_home()
            
        except Exception as e:
            logger.error(f"Error during trajectory execution: {e}")
        finally:
            self.executing = False
            self.current_phase = "ready"
            self.grasped_object = None
    
    def close_gripper(self):
        """Close gripper fingers"""
        if p.getNumJoints(self.robot_id) >= 9:
            p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=0.0, force=100)
            p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, targetPosition=0.0, force=100)
    
    def open_gripper(self):
        """Open gripper fingers"""
        if p.getNumJoints(self.robot_id) >= 9:
            p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=0.05, force=100)
            p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, targetPosition=0.05, force=100)
    
    def reset_to_home(self):
        """Reset robot to home position"""
        logger.info("Resetting robot to home position")
        home_joints = [0, -0.5, 0, -1.5, 0, 1.0, 0]
        for j in range(min(7, p.getNumJoints(self.robot_id))):
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=home_joints[j],
                force=500
            )
        # Step simulation to move to home
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1/240)
    
    def reset_scene_objects(self):
        """Reset scene objects to initial positions"""
        logger.info("Resetting scene objects")
        if self.cube_id is not None:
            p.resetBasePositionAndOrientation(
                self.cube_id,
                [0.4, 0, 0.05],  # Original position
                [0, 0, 0, 1]
            )
        if self.platform_id is not None:
            p.resetBasePositionAndOrientation(
                self.platform_id,
                [0, 0.4, 0.01],  # Original position
                [0, 0, 0, 1]
            )
    
    def check_for_waypoints(self):
        """Check for new waypoints from API"""
        
        if self.waypoints_file.exists():
            try:
                mtime = self.waypoints_file.stat().st_mtime
                if mtime > self.last_waypoint_time:
                    with open(self.waypoints_file, 'r') as f:
                        data = json.load(f)
                    
                    request_id = data.get('request_id', '')
                    waypoints = data.get('waypoints', [])
                    
                    # Only execute if it's a new request and not currently executing
                    if waypoints and not self.executing and request_id != self.last_request_id:
                        logger.info(f"Received NEW task {request_id}: {len(waypoints)} waypoints")
                        self.last_waypoint_time = mtime
                        self.last_request_id = request_id
                        self.execution_count += 1
                        
                        # Reset scene objects to initial positions before new execution
                        self.reset_scene_objects()
                        
                        # Execute in separate thread
                        thread = threading.Thread(
                            target=self.execute_real_trajectory,
                            args=(waypoints,)
                        )
                        thread.daemon = True
                        thread.start()
                        
            except Exception as e:
                logger.error(f"Error reading waypoints: {e}")
    
    def check_api_connection(self):
        """Check connection to CogniForge API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=1)
            if response.status_code == 200:
                if not self.api_connected:
                    self.api_connected = True
                    logger.info("Connected to CogniForge API")
                return True
        except:
            if self.api_connected:
                self.api_connected = False
                logger.warning("Lost connection to CogniForge API")
        return False
    
    def run(self):
        """Main execution loop"""
        
        # Setup simulation
        self.setup_simulation()
        
        # Check API
        self.check_api_connection()
        
        # Main loop
        frame = 0
        last_api_check = 0
        
        while self.running and p.isConnected():
            current_time = time.time()
            
            # Update camera periodically
            if frame % 30 == 0:
                self.update_wrist_camera()
            
            # Check API connection every 5 seconds
            if current_time - last_api_check > 5:
                self.check_api_connection()
                last_api_check = current_time
            
            # Check for new waypoints
            if frame % 10 == 0:  # Check every 10 frames
                self.check_for_waypoints()
            
            # Display status
            status_text = f"Status: {'EXECUTING' if self.executing else self.current_phase.upper()}"
            api_text = f"API: {'CONNECTED' if self.api_connected else 'DISCONNECTED'}"
            exec_text = f"Tasks Executed: {self.execution_count}"
            
            # Status color based on state
            if self.executing:
                status_color = [1, 0.5, 0]  # Orange when executing
            elif self.current_phase == "ready":
                status_color = [0, 1, 0]  # Green when ready
            else:
                status_color = [0.5, 0.5, 0.5]  # Gray when idle
            
            p.addUserDebugText(
                status_text,
                [-0.5, -0.5, 0.5],
                textColorRGB=status_color,
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=100
            )
            
            p.addUserDebugText(
                api_text,
                [-0.5, -0.5, 0.6],
                textColorRGB=[0, 1, 0] if self.api_connected else [1, 0, 0],
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=101
            )
            
            p.addUserDebugText(
                exec_text,
                [-0.5, -0.5, 0.7],
                textColorRGB=[0.8, 0.8, 0.8],
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=102
            )
            
            p.addUserDebugText(
                "WRIST CAMERA (Real-time) | Press 'R' to reset | 'Q' to quit",
                [-0.5, -0.5, 0.8],
                textColorRGB=[0.5, 0.5, 1],
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=103
            )
            
            # Check for keyboard input
            keys = p.getKeyboardEvents()
            if keys.get(ord('q')) or keys.get(ord('Q')):
                self.running = False
                logger.info("Quit requested")
            elif keys.get(ord('r')) or keys.get(ord('R')):
                if not self.executing:
                    logger.info("Manual reset requested")
                    self.reset_scene_objects()
                    self.reset_to_home()
            
            frame += 1
            time.sleep(1/240)
        
        p.disconnect()
        logger.info("PyBullet demo ended")

def main():
    demo = IntegratedPyBulletDemo()
    demo.run()

if __name__ == "__main__":
    main()