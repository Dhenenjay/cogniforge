#!/usr/bin/env python3
"""
CogniForge Integrated Demo
Combines PyBullet visualization with API backend execution
"""

import asyncio
import json
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pybullet as p
import pybullet_data
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedDemo:
    """Integrated demo combining visualization and API"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.sim_id = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.running = False
        
        # Robot configuration
        self.robot_base_pos = [0, 0, 0]
        self.robot_base_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        # Task state
        self.current_phase = None
        self.current_progress = 0.0
        
    def setup_simulation(self):
        """Set up PyBullet simulation"""
        logger.info("Setting up PyBullet simulation...")
        
        # Connect to PyBullet
        self.sim_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set up environment
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Load plane
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Load robot (Kuka IIWA)
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            self.robot_base_pos,
            self.robot_base_orn,
            useFixedBase=True
        )
        
        # Create blue cube (target object)
        cube_size = 0.05
        cube_mass = 0.1
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
            baseMass=cube_mass,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.4, 0, 0.05]
        )
        
        # Create green platform (target location)
        platform_size = [0.15, 0.15, 0.02]
        platform_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=platform_size,
            rgbaColor=[0, 1, 0, 1]  # Green
        )
        platform_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=platform_size
        )
        
        self.platform_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0, 0.4, 0.01],
            useMaximalCoordinates=True
        )
        
        # Set camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3]
        )
        
        logger.info("Simulation setup complete")
    
    def update_visualization(self, phase: str, metrics: Dict[str, Any]):
        """Update PyBullet visualization based on phase and metrics"""
        
        if phase == "planning":
            # Show planning visualization
            p.addUserDebugText(
                "Phase: Planning Task",
                [0, 0, 0.8],
                textColorRGB=[1, 1, 0],
                textSize=1.5,
                lifeTime=2.0
            )
            
        elif phase == "expert_demonstration":
            # Animate expert demo
            self.animate_expert_demo()
            
        elif phase == "behavior_cloning":
            # Show BC training progress
            if "epoch_loss" in metrics:
                p.addUserDebugText(
                    f"BC Loss: {metrics['epoch_loss']:.4f}",
                    [0, 0, 0.7],
                    textColorRGB=[1, 0.5, 0],
                    textSize=1.5,
                    lifeTime=0.5
                )
            
        elif phase == "optimization":
            # Show optimization progress
            if "avg_reward" in metrics:
                p.addUserDebugText(
                    f"Reward: {metrics['avg_reward']:.3f}",
                    [0, 0, 0.6],
                    textColorRGB=[0, 1, 0.5],
                    textSize=1.5,
                    lifeTime=0.5
                )
                
        elif phase == "vision_refinement":
            # Show vision correction
            if "vision_offset" in metrics:
                offset_data = metrics["vision_offset"]
                if "world_offset" in offset_data:
                    dx = offset_data["world_offset"].get("dx", 0)
                    dy = offset_data["world_offset"].get("dy", 0)
                    
                    # Draw offset arrow
                    cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
                    corrected_pos = [
                        cube_pos[0] - dx/1000,  # Convert mm to m
                        cube_pos[1] - dy/1000,
                        cube_pos[2]
                    ]
                    
                    p.addUserDebugLine(
                        cube_pos,
                        corrected_pos,
                        lineColorRGB=[1, 0, 1],
                        lineWidth=3,
                        lifeTime=2.0
                    )
                    
                    p.addUserDebugText(
                        f"Vision Offset: {dx:.1f}mm, {dy:.1f}mm",
                        [0, 0, 0.5],
                        textColorRGB=[1, 0, 1],
                        textSize=1.5,
                        lifeTime=2.0
                    )
        
        elif phase == "execution":
            # Execute final trajectory
            self.execute_learned_trajectory()
    
    def animate_expert_demo(self):
        """Animate expert demonstration"""
        # Simple pick and place animation
        for t in range(50):
            # Move to cube
            if t < 15:
                progress = t / 15.0
                pos = [0.4 * progress, 0, 0.3]
            # Descend to cube
            elif t < 25:
                pos = [0.4, 0, 0.3 - 0.25 * ((t - 15) / 10.0)]
            # Lift cube
            elif t < 35:
                pos = [0.4, 0, 0.05 + 0.25 * ((t - 25) / 10.0)]
                # Move cube with gripper
                if t == 25:
                    logger.info("Grasping cube")
            # Move to platform
            elif t < 45:
                progress = (t - 35) / 10.0
                pos = [0.4 * (1 - progress), 0.4 * progress, 0.3]
                # Move cube along
                p.resetBasePositionAndOrientation(
                    self.cube_id,
                    pos,
                    self.robot_base_orn
                )
            # Place
            else:
                pos = [0, 0.4, 0.3 - 0.2 * ((t - 45) / 5.0)]
                p.resetBasePositionAndOrientation(
                    self.cube_id,
                    [0, 0.4, 0.05],
                    self.robot_base_orn
                )
            
            # Update robot visualization
            self.set_robot_ee_position(pos)
            p.stepSimulation()
            time.sleep(0.05)
    
    def execute_learned_trajectory(self):
        """Execute the learned trajectory"""
        logger.info("Executing learned trajectory...")
        # Similar to expert demo but with learned parameters
        self.animate_expert_demo()
    
    def set_robot_ee_position(self, position):
        """Set robot end-effector position using IK"""
        num_joints = p.getNumJoints(self.robot_id)
        
        # Get end-effector link
        ee_link = num_joints - 1
        
        # Calculate IK
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            ee_link,
            position,
            maxNumIterations=100
        )
        
        # Set joint positions
        for i in range(min(7, num_joints)):  # Kuka has 7 DOF
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500
            )
    
    async def monitor_api_events(self, request_id: str):
        """Monitor SSE events from API"""
        url = f"{self.api_url}/api/events/{request_id}"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    # Parse SSE data
                    if line.startswith(b'data: '):
                        data = json.loads(line[6:])
                        
                        # Update phase
                        if 'phase' in data:
                            self.current_phase = data['phase']
                            logger.info(f"Phase: {data['phase']} - {data.get('message', '')}")
                        
                        # Update progress
                        if 'progress' in data:
                            self.current_progress = data['progress']
                        
                        # Update visualization
                        if 'metrics' in data:
                            self.update_visualization(
                                data.get('phase', ''),
                                data['metrics']
                            )
                        
                        # Check completion
                        if data.get('type') == 'complete':
                            logger.info("Pipeline completed successfully!")
                            break
                        elif data.get('type') == 'error':
                            logger.error(f"Pipeline failed: {data.get('message')}")
                            break
                            
        except Exception as e:
            logger.error(f"Failed to monitor events: {e}")
    
    def run_demo(self, task_description: str):
        """Run the integrated demo"""
        
        logger.info(f"Starting integrated demo: {task_description}")
        
        # Setup simulation
        self.setup_simulation()
        
        # Send execution request to API
        logger.info("Sending execution request to API...")
        
        request_data = {
            "task_type": "pick_and_place",
            "task_description": task_description,
            "use_vision": True,
            "use_gpt_reward": False,
            "dry_run": False,
            "num_bc_epochs": 5,
            "num_optimization_steps": 20,
            "safety_checks": True
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/execute",
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            request_id = result['request_id']
            
            logger.info(f"Request ID: {request_id}")
            
            # Monitor events in separate thread
            event_thread = threading.Thread(
                target=lambda: asyncio.run(self.monitor_api_events(request_id))
            )
            event_thread.start()
            
            # Keep simulation running
            self.running = True
            while self.running:
                # Update GUI
                p.stepSimulation()
                
                # Add status text
                p.addUserDebugText(
                    f"Progress: {self.current_progress*100:.1f}%",
                    [-0.5, 0, 0.8],
                    textColorRGB=[1, 1, 1],
                    textSize=1.2,
                    lifeTime=0,
                    replaceItemUniqueId=1
                )
                
                if self.current_phase:
                    p.addUserDebugText(
                        f"Phase: {self.current_phase}",
                        [-0.5, 0, 0.7],
                        textColorRGB=[0.5, 0.5, 1],
                        textSize=1.2,
                        lifeTime=0,
                        replaceItemUniqueId=2
                    )
                
                time.sleep(0.01)
                
                # Check for completion
                if self.current_progress >= 1.0:
                    time.sleep(2)
                    self.running = False
            
            # Wait for event thread
            event_thread.join(timeout=5)
            
            logger.info("Demo completed!")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            logger.error("Make sure the API server is running: python api_server.py")
        
        except Exception as e:
            logger.error(f"Demo failed: {e}")
        
        finally:
            # Keep window open
            logger.info("Press any key in PyBullet window to exit...")
            while p.isConnected():
                p.stepSimulation()
                time.sleep(0.01)
                keys = p.getKeyboardEvents()
                if keys:
                    break
            
            p.disconnect()

def main():
    """Main entry point"""
    
    # Default task
    task = "Pick up the blue cube and place it on the green platform"
    
    # Check if custom task provided
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    
    # Create and run demo
    demo = IntegratedDemo()
    demo.run_demo(task)

if __name__ == "__main__":
    main()