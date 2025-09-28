#!/usr/bin/env python3
"""
CogniForge PyBullet Demo with 3 Camera Views
Simple visualization with front, side, and top camera views
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

class PyBullet3ViewDemo:
    """PyBullet demo with 3 camera views"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.sim_id = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.running = True
        
        # Camera settings for 3 views
        self.camera_configs = {
            "front": {"distance": 1.5, "yaw": 0, "pitch": -20, "target": [0, 0, 0.3]},
            "side": {"distance": 1.5, "yaw": 90, "pitch": -20, "target": [0, 0, 0.3]},
            "top": {"distance": 2.0, "yaw": 45, "pitch": -89, "target": [0, 0, 0]}
        }
        
        self.current_view = "front"
        self.auto_rotate = False
        
        # Robot state for API-driven motion
        self.target_positions = []
        self.current_phase = "idle"
        self.api_connected = False
        self.current_request_id = None
        
    def setup_simulation(self):
        """Set up PyBullet simulation with minimal UI"""
        logger.info("Setting up PyBullet with 3 camera views...")
        
        # Connect with GUI
        self.sim_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure GUI (remove most panels)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Keep minimal GUI
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        # Physics
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
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
        
        # Add simple camera view buttons
        self.add_camera_controls()
        
        # Set initial camera view
        self.set_camera_view("front")
        
        logger.info("PyBullet setup complete with 3 views")
        
    def create_scene_objects(self):
        """Create simple scene objects"""
        
        # Blue cube (target object)
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
        
        # Green platform (target location)
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
        
    def add_camera_controls(self):
        """Add simple camera view controls"""
        
        # Camera view buttons
        self.view_front_btn = p.addUserDebugParameter("View: Front", 0, 1, 0)
        self.view_side_btn = p.addUserDebugParameter("View: Side", 0, 1, 0)
        self.view_top_btn = p.addUserDebugParameter("View: Top", 0, 1, 0)
        self.auto_rotate_btn = p.addUserDebugParameter("Auto Rotate", 0, 1, 0)
        
    def set_camera_view(self, view_name: str):
        """Set camera to specific view"""
        if view_name in self.camera_configs:
            config = self.camera_configs[view_name]
            p.resetDebugVisualizerCamera(
                cameraDistance=config["distance"],
                cameraYaw=config["yaw"],
                cameraPitch=config["pitch"],
                cameraTargetPosition=config["target"]
            )
            self.current_view = view_name
            
            # Update status text
            p.addUserDebugText(
                f"Camera View: {view_name.upper()}",
                [-0.5, -0.5, 0.6],
                textColorRGB=[1, 1, 1],
                textSize=1.5,
                lifeTime=2.0,
                replaceItemUniqueId=1
            )
    
    def update_camera_controls(self):
        """Check camera control buttons"""
        
        # Check view buttons
        if p.readUserDebugParameter(self.view_front_btn) > 0.5:
            self.set_camera_view("front")
            p.resetBaseDebugParam(self.view_front_btn, 0)
            
        if p.readUserDebugParameter(self.view_side_btn) > 0.5:
            self.set_camera_view("side")
            p.resetBaseDebugParam(self.view_side_btn, 0)
            
        if p.readUserDebugParameter(self.view_top_btn) > 0.5:
            self.set_camera_view("top")
            p.resetBaseDebugParam(self.view_top_btn, 0)
            
        # Auto rotate
        self.auto_rotate = p.readUserDebugParameter(self.auto_rotate_btn) > 0.5
        
        if self.auto_rotate:
            # Slowly rotate camera
            camera_info = p.getDebugVisualizerCamera()
            yaw = camera_info[8] + 0.5  # Slowly increase yaw
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_info[10],
                cameraYaw=yaw,
                cameraPitch=camera_info[9],
                cameraTargetPosition=camera_info[11]
            )
    
    def render_camera_views(self):
        """Render 3 camera views in small windows"""
        
        # Get window size
        width, height = 320, 240
        
        # Render front view
        front_config = self.camera_configs["front"]
        view_matrix_front = p.computeViewMatrixFromYawPitchRoll(
            front_config["target"],
            front_config["distance"],
            front_config["yaw"],
            front_config["pitch"],
            0,
            2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image (front view)
        img_front = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix_front,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Similarly for side and top views (if needed)
        # This would require additional rendering windows or texture displays
        
    def display_status(self):
        """Display current phase and status"""
        
        status_text = f"Phase: {self.current_phase}"
        api_status = "API: Connected" if self.api_connected else "API: Disconnected"
        
        p.addUserDebugText(
            status_text,
            [-0.5, -0.5, 0.5],
            textColorRGB=[0, 1, 0] if self.api_connected else [1, 0, 0],
            textSize=1.2,
            lifeTime=0,
            replaceItemUniqueId=10
        )
        
        p.addUserDebugText(
            api_status,
            [-0.5, -0.5, 0.45],
            textColorRGB=[0.7, 0.7, 0.7],
            textSize=1.0,
            lifeTime=0,
            replaceItemUniqueId=11
        )
    
    def execute_trajectory(self, waypoints: List[Dict[str, float]]):
        """Execute trajectory from API"""
        
        logger.info(f"Executing trajectory with {len(waypoints)} waypoints")
        
        for i, wp in enumerate(waypoints):
            target_pos = [wp['x'], wp['y'], wp['z']]
            
            # Calculate IK
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                6,  # End effector
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
            
            # Step simulation
            for _ in range(10):
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
    
    async def monitor_api_events(self, request_id: str):
        """Monitor API events and update visualization"""
        
        url = f"{self.api_url}/api/events/{request_id}"
        
        try:
            response = requests.get(url, stream=True, timeout=1)
            
            for line in response.iter_lines():
                if line and line.startswith(b'data: '):
                    try:
                        data = json.loads(line[6:])
                        
                        # Update phase
                        if 'phase' in data:
                            self.current_phase = data['phase']
                            logger.info(f"Phase: {data['phase']}")
                        
                        # Handle waypoints
                        if data.get('type') == 'waypoints':
                            waypoints = data.get('waypoints', [])
                            if waypoints:
                                threading.Thread(
                                    target=lambda: self.execute_trajectory(waypoints)
                                ).start()
                        
                        # Handle completion
                        if data.get('type') == 'complete':
                            logger.info("Execution completed!")
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to monitor API: {e}")
    
    async def connect_to_api(self):
        """Try to connect to API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=1)
            if response.status_code == 200:
                self.api_connected = True
                logger.info("Connected to API")
                return True
        except:
            self.api_connected = False
            logger.info("Running in standalone mode (API not available)")
        return False
    
    def simulate_pick_place(self):
        """Simple pick and place simulation"""
        
        # Pick and place waypoints
        waypoints = [
            {"x": 0.4, "y": 0, "z": 0.3},    # Above cube
            {"x": 0.4, "y": 0, "z": 0.08},   # Descend to cube
            {"x": 0.4, "y": 0, "z": 0.3},    # Lift
            {"x": 0, "y": 0.4, "z": 0.3},    # Move to platform
            {"x": 0, "y": 0.4, "z": 0.08},   # Descend
            {"x": 0, "y": 0.4, "z": 0.3},    # Lift away
        ]
        
        self.execute_trajectory(waypoints)
    
    def run(self):
        """Main run loop"""
        
        # Setup simulation
        self.setup_simulation()
        
        # Try to connect to API
        asyncio.run(self.connect_to_api())
        
        # If API connected, start monitoring in background
        if self.api_connected:
            # Get latest request ID from API if available
            pass
        
        # Main loop
        frame = 0
        last_sim_time = time.time()
        
        while self.running and p.isConnected():
            
            # Update camera controls
            self.update_camera_controls()
            
            # Display status
            self.display_status()
            
            # Auto demo every 300 frames
            if frame % 300 == 0 and frame > 0:
                if not self.api_connected:
                    # Run standalone demo
                    threading.Thread(target=self.simulate_pick_place).start()
            
            # Step simulation at 60Hz
            current_time = time.time()
            if current_time - last_sim_time > 1/60:
                p.stepSimulation()
                last_sim_time = current_time
            
            frame += 1
            
            # Check for quit
            keys = p.getKeyboardEvents()
            if keys.get(ord('q')):
                self.running = False
            
            time.sleep(1/240)
        
        # Cleanup
        p.disconnect()
        logger.info("Demo ended")

def main():
    """Main entry point"""
    demo = PyBullet3ViewDemo()
    demo.run()

if __name__ == "__main__":
    main()