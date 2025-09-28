#!/usr/bin/env python3
"""
CogniForge PyBullet Demo - Fixed Version
Single window with 3 camera preview panels as in original PyBullet
"""

import asyncio
import json
import logging
import sys
import time
import threading
from typing import Dict, Any, List
import requests

import numpy as np
import pybullet as p
import pybullet_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyBulletDemo:
    """PyBullet demo with 3 camera previews in same window"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.sim_id = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.running = True
        
        # Robot state
        self.current_phase = "idle"
        self.api_connected = False
        
    def setup_simulation(self):
        """Set up PyBullet simulation with 3 camera previews"""
        logger.info("Setting up PyBullet with camera previews...")
        
        # Connect with GUI
        self.sim_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # ENABLE the 3 camera preview panels (this is what was missing!)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # Update preview window titles (note: PyBullet doesn't support custom titles)
        # The previews will show "Wrist Camera RGB/Depth/Segmentation" when active
        
        # Physics
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        
        # Load environment
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Load Kuka robot
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            [0, 0, 0],
            useFixedBase=True
        )
        
        # Create scene objects
        self.create_scene_objects()
        
        # Set camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3]
        )
        
        logger.info("PyBullet ready with 3 camera previews")
        
    def create_scene_objects(self):
        """Create scene objects"""
        
        # Blue cube
        cube_size = 0.05
        cube_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3,
            rgbaColor=[0, 0, 1, 1]
        )
        cube_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3
        )
        
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.4, 0, 0.05]
        )
        
        # Green platform
        platform_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.02],
            rgbaColor=[0, 1, 0, 1]
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
        
    def update_camera_previews(self):
        """Update the 3 camera preview panels with wrist camera view"""
        
        # Get camera parameters
        width, height = 320, 240
        
        # Get wrist (end-effector) position and orientation
        if self.robot_id is not None:
            # Get end-effector link state (link 6 is typically the end-effector for Kuka)
            link_state = p.getLinkState(self.robot_id, 6)
            end_effector_pos = link_state[0]
            end_effector_orn = link_state[1]
            
            # Convert quaternion to rotation matrix for camera
            rot_matrix = p.getMatrixFromQuaternion(end_effector_orn)
            rot_matrix = np.array(rot_matrix).reshape(3, 3)
            
            # Camera offset from wrist (looking down from wrist)
            camera_offset = [0, 0, -0.1]  # 10cm below the wrist
            camera_eye_pos = [
                end_effector_pos[0] + camera_offset[0],
                end_effector_pos[1] + camera_offset[1],
                end_effector_pos[2] + camera_offset[2]
            ]
            
            # Look down from wrist position
            camera_target_pos = [
                end_effector_pos[0],
                end_effector_pos[1],
                end_effector_pos[2] - 0.2  # Look 20cm below wrist
            ]
            
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_eye_pos,
                cameraTargetPosition=camera_target_pos,
                cameraUpVector=[0, 0, 1]
            )
        else:
            # Fallback to static camera if robot not loaded
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1, 0, 0.5],
                cameraTargetPosition=[0, 0, 0.2],
                cameraUpVector=[0, 0, 1]
            )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.01,
            farVal=10
        )
        
        # Get camera image with all buffers
        img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        
        # The preview panels will automatically update
        
    def execute_trajectory(self, waypoints: List[Dict[str, float]]):
        """Execute trajectory"""
        
        for i, wp in enumerate(waypoints):
            target_pos = [wp['x'], wp['y'], wp['z']]
            
            # IK
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                6,
                target_pos,
                maxNumIterations=100
            )
            
            # Move joints
            for j in range(min(7, p.getNumJoints(self.robot_id))):
                p.setJointMotorControl2(
                    self.robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[j],
                    force=500
                )
            
            # Step
            for _ in range(20):
                p.stepSimulation()
                time.sleep(1/240)
                
            # Show progress
            p.addUserDebugText(
                f"Waypoint {i+1}/{len(waypoints)}",
                target_pos,
                textColorRGB=[0, 1, 0],
                textSize=1.0,
                lifeTime=0.5
            )
    
    def demo_pick_place(self):
        """Demo pick and place"""
        waypoints = [
            {"x": 0.4, "y": 0, "z": 0.3},
            {"x": 0.4, "y": 0, "z": 0.08},
            {"x": 0.4, "y": 0, "z": 0.3},
            {"x": 0, "y": 0.4, "z": 0.3},
            {"x": 0, "y": 0.4, "z": 0.08},
            {"x": 0, "y": 0.4, "z": 0.3},
        ]
        self.execute_trajectory(waypoints)
        
    async def monitor_api(self, request_id: str):
        """Monitor API events"""
        try:
            url = f"{self.api_url}/api/events/{request_id}"
            response = requests.get(url, stream=True, timeout=1)
            
            for line in response.iter_lines():
                if line and line.startswith(b'data: '):
                    try:
                        data = json.loads(line[6:])
                        
                        if 'phase' in data:
                            self.current_phase = data['phase']
                            logger.info(f"Phase: {data['phase']}")
                            
                        if data.get('type') == 'waypoints':
                            waypoints = data.get('waypoints', [])
                            if waypoints:
                                self.execute_trajectory(waypoints)
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"API monitoring failed: {e}")
            
    async def connect_api(self):
        """Connect to API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=1)
            if response.status_code == 200:
                self.api_connected = True
                logger.info("API connected")
                return True
        except:
            self.api_connected = False
            logger.info("Running standalone")
        return False
        
    def run(self):
        """Main loop"""
        
        # Setup
        self.setup_simulation()
        
        # Try API
        asyncio.run(self.connect_api())
        
        # Main loop
        frame = 0
        
        while self.running and p.isConnected():
            
            # Update camera previews periodically
            if frame % 30 == 0:
                self.update_camera_previews()
            
            # Show status
            p.addUserDebugText(
                f"Status: {self.current_phase} | API: {'Connected' if self.api_connected else 'Disconnected'}",
                [-0.5, -0.5, 0.5],
                textColorRGB=[0, 1, 0] if self.api_connected else [1, 1, 0],
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=100
            )
            
            # Show camera view type
            p.addUserDebugText(
                "Camera Previews: WRIST CAMERA VIEW",
                [-0.5, -0.5, 0.6],
                textColorRGB=[0.5, 0.5, 1],
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=101
            )
            
            # Auto demo
            if frame % 500 == 0 and frame > 0:
                if not self.api_connected:
                    threading.Thread(target=self.demo_pick_place).start()
            
            # Check quit
            keys = p.getKeyboardEvents()
            if keys.get(ord('q')):
                self.running = False
                
            frame += 1
            time.sleep(1/240)
            
        p.disconnect()
        logger.info("Demo ended")

def main():
    demo = PyBulletDemo()
    demo.run()

if __name__ == "__main__":
    main()