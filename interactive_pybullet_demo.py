#!/usr/bin/env python3
"""
CogniForge Interactive PyBullet Demo
Full interactive control with all visualization features
"""

import asyncio
import json
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

import numpy as np
import pybullet as p
import pybullet_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractivePyBulletDemo:
    """Fully interactive PyBullet demo with all controls"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.sim_id = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.running = True
        
        # Robot state
        self.robot_joints = []
        self.ee_pos = [0.4, 0.0, 0.3]
        self.gripper_state = 0.0
        
        # Waypoints
        self.expert_waypoints = []
        self.optimized_waypoints = []
        self.current_waypoint_idx = 0
        self.show_waypoints = True
        self.show_diff = False
        
        # Control modes
        self.control_mode = "manual"  # manual, auto, playback
        self.undo_stack = []
        self.skills_library = {
            "push": self.skill_push,
            "slide": self.skill_slide,
            "stack": self.skill_stack,
            "grasp": self.skill_grasp,
            "place": self.skill_place
        }
        
        # Visualization
        self.waypoint_markers = []
        self.trajectory_lines = []
        self.text_ids = {}
        
        # API connection
        self.current_request_id = None
        self.api_connected = False
        
    def setup_simulation(self):
        """Set up interactive PyBullet simulation"""
        logger.info("Setting up interactive PyBullet simulation...")
        
        # Connect with GUI
        self.sim_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Physics settings
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
        
        # Get joint info
        self.robot_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:
                self.robot_joints.append(i)
        
        # Create objects
        self.create_scene_objects()
        
        # Set camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3]
        )
        
        # Add GUI controls
        self.setup_gui_controls()
        
        logger.info("Interactive simulation ready!")
    
    def create_scene_objects(self):
        """Create interactive scene objects"""
        
        # Blue cube
        cube_size = 0.05
        cube_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3,
            rgbaColor=[0, 0.3, 1, 1]
        )
        cube_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3
        )
        
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.4, 0.1, 0.05]
        )
        
        # Green platform
        platform_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.02],
            rgbaColor=[0, 1, 0.3, 1]
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
    
    def setup_gui_controls(self):
        """Setup interactive GUI parameters"""
        
        # Mode selector
        self.mode_param = p.addUserDebugParameter("Mode (0=Manual,1=Auto,2=Playback)", 0, 2, 0)
        
        # Robot controls
        self.x_param = p.addUserDebugParameter("X Position", -0.5, 0.5, 0.4)
        self.y_param = p.addUserDebugParameter("Y Position", -0.5, 0.5, 0.0)
        self.z_param = p.addUserDebugParameter("Z Position", 0.0, 0.6, 0.3)
        self.gripper_param = p.addUserDebugParameter("Gripper", 0, 1, 0)
        
        # Visualization controls
        self.show_waypoints_param = p.addUserDebugParameter("Show Waypoints", 0, 1, 1)
        self.show_diff_param = p.addUserDebugParameter("Show Diff", 0, 1, 0)
        self.waypoint_alpha_param = p.addUserDebugParameter("Waypoint Alpha", 0, 1, 0.5)
        
        # Skill execution
        self.skill_param = p.addUserDebugParameter("Skill (0-4)", 0, 4, 0)
        self.execute_skill_param = p.addUserDebugParameter("Execute Skill", 0, 1, 0)
        
        # Playback controls
        self.playback_speed_param = p.addUserDebugParameter("Playback Speed", 0.1, 2.0, 1.0)
        self.loop_playback_param = p.addUserDebugParameter("Loop Playback", 0, 1, 0)
    
    def process_keyboard_input(self):
        """Process keyboard input for interactive control"""
        keys = p.getKeyboardEvents()
        
        if keys.get(ord('q')):  # Quit
            self.running = False
            
        if keys.get(ord('h')):  # Help
            self.show_help()
            
        if keys.get(ord('r')):  # Reset
            self.reset_scene()
            
        if keys.get(ord('u')):  # Undo
            self.undo_last_action()
            
        if keys.get(ord('w')):  # Toggle waypoints
            self.show_waypoints = not self.show_waypoints
            
        if keys.get(ord('d')):  # Toggle diff view
            self.show_diff = not self.show_diff
            
        if keys.get(ord('s')):  # Save waypoint
            self.save_current_waypoint()
            
        if keys.get(ord('c')):  # Clear waypoints
            self.clear_waypoints()
            
        if keys.get(ord(' ')):  # Space - Execute
            self.execute_current_trajectory()
            
        # Arrow keys for fine control
        if keys.get(p.B3G_UP_ARROW):
            self.ee_pos[1] += 0.01
        if keys.get(p.B3G_DOWN_ARROW):
            self.ee_pos[1] -= 0.01
        if keys.get(p.B3G_LEFT_ARROW):
            self.ee_pos[0] -= 0.01
        if keys.get(p.B3G_RIGHT_ARROW):
            self.ee_pos[0] += 0.01
        if keys.get(p.B3G_PAGE_UP):
            self.ee_pos[2] += 0.01
        if keys.get(p.B3G_PAGE_DOWN):
            self.ee_pos[2] -= 0.01
    
    def update_robot_control(self):
        """Update robot based on current control mode"""
        
        mode = int(p.readUserDebugParameter(self.mode_param))
        
        if mode == 0:  # Manual control
            self.ee_pos[0] = p.readUserDebugParameter(self.x_param)
            self.ee_pos[1] = p.readUserDebugParameter(self.y_param)
            self.ee_pos[2] = p.readUserDebugParameter(self.z_param)
            self.gripper_state = p.readUserDebugParameter(self.gripper_param)
            
        elif mode == 1:  # Auto mode
            self.auto_execute_task()
            
        elif mode == 2:  # Playback mode
            self.playback_trajectory()
        
        # Apply IK
        self.apply_ik(self.ee_pos, self.gripper_state)
    
    def apply_ik(self, target_pos: List[float], gripper: float):
        """Apply inverse kinematics"""
        
        # Calculate IK for arm
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            6,  # End effector link
            target_pos,
            maxNumIterations=100
        )
        
        # Set joint positions
        for i, joint_id in enumerate(self.robot_joints[:7]):
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500
            )
        
        # Gripper control (if robot has gripper joints)
        # This is robot-specific
    
    def visualize_waypoints(self):
        """Visualize waypoints and trajectories"""
        
        # Clear old markers
        for marker in self.waypoint_markers:
            p.removeBody(marker)
        self.waypoint_markers = []
        
        for line in self.trajectory_lines:
            p.removeUserDebugItem(line)
        self.trajectory_lines = []
        
        if not self.show_waypoints:
            return
        
        alpha = p.readUserDebugParameter(self.waypoint_alpha_param)
        
        # Draw expert waypoints
        if self.expert_waypoints:
            prev_pos = None
            for i, wp in enumerate(self.expert_waypoints):
                pos = [wp['x'], wp['y'], wp['z']]
                
                # Create sphere marker
                visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.01,
                    rgbaColor=[1, 0.5, 0, alpha]
                )
                
                marker = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual,
                    basePosition=pos
                )
                self.waypoint_markers.append(marker)
                
                # Draw line
                if prev_pos:
                    line = p.addUserDebugLine(
                        prev_pos, pos,
                        lineColorRGB=[1, 0.5, 0],
                        lineWidth=2
                    )
                    self.trajectory_lines.append(line)
                prev_pos = pos
        
        # Draw optimized waypoints if showing diff
        if self.show_diff and self.optimized_waypoints:
            prev_pos = None
            for i, wp in enumerate(self.optimized_waypoints):
                pos = [wp['x'], wp['y'], wp['z']]
                
                # Create sphere marker
                visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.01,
                    rgbaColor=[0, 1, 0, alpha]
                )
                
                marker = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual,
                    basePosition=pos
                )
                self.waypoint_markers.append(marker)
                
                # Draw line
                if prev_pos:
                    line = p.addUserDebugLine(
                        prev_pos, pos,
                        lineColorRGB=[0, 1, 0],
                        lineWidth=2
                    )
                    self.trajectory_lines.append(line)
                prev_pos = pos
    
    def show_status_text(self):
        """Display status information"""
        
        # Clear old text
        for text_id in self.text_ids.values():
            p.removeUserDebugItem(text_id)
        self.text_ids = {}
        
        # Mode text
        mode = int(p.readUserDebugParameter(self.mode_param))
        mode_names = ["Manual", "Auto", "Playback"]
        self.text_ids['mode'] = p.addUserDebugText(
            f"Mode: {mode_names[mode]}",
            [-0.5, -0.5, 0.5],
            textColorRGB=[1, 1, 1],
            textSize=1.5
        )
        
        # Position text
        self.text_ids['pos'] = p.addUserDebugText(
            f"EE Pos: ({self.ee_pos[0]:.2f}, {self.ee_pos[1]:.2f}, {self.ee_pos[2]:.2f})",
            [-0.5, -0.5, 0.45],
            textColorRGB=[0.7, 0.7, 0.7],
            textSize=1.0
        )
        
        # Waypoint count
        self.text_ids['waypoints'] = p.addUserDebugText(
            f"Waypoints: Expert={len(self.expert_waypoints)}, Optimized={len(self.optimized_waypoints)}",
            [-0.5, -0.5, 0.4],
            textColorRGB=[0.7, 0.7, 0.7],
            textSize=1.0
        )
        
        # Help text
        self.text_ids['help'] = p.addUserDebugText(
            "Press 'H' for help | Arrow keys to move | Space to execute",
            [-0.5, -0.5, 0.35],
            textColorRGB=[0.5, 0.5, 0.5],
            textSize=0.8
        )
    
    def show_help(self):
        """Display help overlay"""
        help_text = """
=== INTERACTIVE CONTROLS ===
KEYBOARD:
  Arrow Keys: Move X/Y
  PageUp/Down: Move Z
  Space: Execute trajectory
  H: Show this help
  R: Reset scene
  U: Undo last action
  W: Toggle waypoints
  D: Toggle diff view
  S: Save current waypoint
  C: Clear waypoints
  Q: Quit

GUI SLIDERS:
  Mode: Manual/Auto/Playback
  Position: X/Y/Z control
  Gripper: Open/Close
  Skills: Execute predefined skills

MOUSE:
  Click+Drag: Rotate view
  Scroll: Zoom
  Shift+Click: Pan view
        """
        
        logger.info(help_text)
        
        # Also show in GUI
        p.addUserDebugText(
            help_text,
            [0, 0, 0.6],
            textColorRGB=[1, 1, 0],
            textSize=1.0,
            lifeTime=5.0
        )
    
    # Skill implementations
    def skill_push(self):
        """Execute push skill"""
        logger.info("Executing PUSH skill")
        # Move behind object
        obj_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
        push_start = [obj_pos[0] - 0.1, obj_pos[1], obj_pos[2] + 0.05]
        push_end = [obj_pos[0] + 0.1, obj_pos[1], obj_pos[2] + 0.05]
        
        self.ee_pos = push_start
        time.sleep(0.5)
        self.ee_pos = push_end
    
    def skill_slide(self):
        """Execute slide skill"""
        logger.info("Executing SLIDE skill")
        # Similar to push but lateral
        obj_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
        slide_start = [obj_pos[0], obj_pos[1] - 0.1, obj_pos[2] + 0.05]
        slide_end = [obj_pos[0], obj_pos[1] + 0.1, obj_pos[2] + 0.05]
        
        self.ee_pos = slide_start
        time.sleep(0.5)
        self.ee_pos = slide_end
    
    def skill_stack(self):
        """Execute stack skill"""
        logger.info("Executing STACK skill")
        # Pick and stack motion
        self.skill_grasp()
        time.sleep(0.5)
        self.ee_pos[2] += 0.1
        time.sleep(0.5)
        self.skill_place()
    
    def skill_grasp(self):
        """Execute grasp skill"""
        logger.info("Executing GRASP skill")
        obj_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
        
        # Approach from above
        self.ee_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
        time.sleep(0.5)
        self.ee_pos[2] = obj_pos[2] + 0.05
        time.sleep(0.3)
        self.gripper_state = 1.0  # Close
    
    def skill_place(self):
        """Execute place skill"""
        logger.info("Executing PLACE skill")
        platform_pos = p.getBasePositionAndOrientation(self.platform_id)[0]
        
        # Move to platform
        self.ee_pos = [platform_pos[0], platform_pos[1], platform_pos[2] + 0.15]
        time.sleep(0.5)
        self.ee_pos[2] = platform_pos[2] + 0.05
        time.sleep(0.3)
        self.gripper_state = 0.0  # Open
    
    def save_current_waypoint(self):
        """Save current position as waypoint"""
        waypoint = {
            'x': self.ee_pos[0],
            'y': self.ee_pos[1],
            'z': self.ee_pos[2],
            'gripper': self.gripper_state
        }
        self.expert_waypoints.append(waypoint)
        logger.info(f"Saved waypoint {len(self.expert_waypoints)}: {waypoint}")
    
    def clear_waypoints(self):
        """Clear all waypoints"""
        self.expert_waypoints = []
        self.optimized_waypoints = []
        logger.info("Cleared all waypoints")
    
    def undo_last_action(self):
        """Undo to previous state"""
        if self.undo_stack:
            state = self.undo_stack.pop()
            self.ee_pos = state['ee_pos']
            self.gripper_state = state['gripper']
            logger.info("Undone to previous state")
    
    def reset_scene(self):
        """Reset scene to initial state"""
        p.resetBasePositionAndOrientation(
            self.cube_id,
            [0.4, 0.1, 0.05],
            [0, 0, 0, 1]
        )
        self.ee_pos = [0.4, 0.0, 0.3]
        self.gripper_state = 0.0
        logger.info("Scene reset")
    
    def execute_current_trajectory(self):
        """Execute the current trajectory"""
        waypoints = self.optimized_waypoints if self.optimized_waypoints else self.expert_waypoints
        
        if not waypoints:
            logger.warning("No waypoints to execute")
            return
        
        logger.info(f"Executing trajectory with {len(waypoints)} waypoints")
        
        for i, wp in enumerate(waypoints):
            self.ee_pos = [wp['x'], wp['y'], wp['z']]
            self.gripper_state = wp.get('gripper', 0.0)
            self.apply_ik(self.ee_pos, self.gripper_state)
            time.sleep(0.1)
            
            # Update progress
            p.addUserDebugText(
                f"Waypoint {i+1}/{len(waypoints)}",
                self.ee_pos,
                textColorRGB=[0, 1, 0],
                textSize=1.0,
                lifeTime=0.5
            )
    
    def auto_execute_task(self):
        """Automatically execute pick and place task"""
        # Simple automated sequence
        if not hasattr(self, 'auto_step'):
            self.auto_step = 0
        
        steps = [
            ([0.4, 0.1, 0.3], 0.0),  # Above cube
            ([0.4, 0.1, 0.1], 0.0),  # Descend
            ([0.4, 0.1, 0.1], 1.0),  # Grasp
            ([0.4, 0.1, 0.3], 1.0),  # Lift
            ([0.0, 0.4, 0.3], 1.0),  # Move to platform
            ([0.0, 0.4, 0.1], 1.0),  # Descend
            ([0.0, 0.4, 0.1], 0.0),  # Release
            ([0.0, 0.4, 0.3], 0.0),  # Retreat
        ]
        
        if self.auto_step < len(steps):
            self.ee_pos, self.gripper_state = steps[self.auto_step]
            self.auto_step = (self.auto_step + 1) % len(steps)
    
    def playback_trajectory(self):
        """Playback recorded trajectory"""
        if not self.expert_waypoints:
            return
        
        if not hasattr(self, 'playback_idx'):
            self.playback_idx = 0
        
        speed = p.readUserDebugParameter(self.playback_speed_param)
        loop = p.readUserDebugParameter(self.loop_playback_param) > 0.5
        
        if self.playback_idx < len(self.expert_waypoints):
            wp = self.expert_waypoints[self.playback_idx]
            self.ee_pos = [wp['x'], wp['y'], wp['z']]
            self.gripper_state = wp.get('gripper', 0.0)
            self.playback_idx += int(speed)
        elif loop:
            self.playback_idx = 0
    
    async def connect_to_api(self):
        """Connect to API for real-time updates"""
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                self.api_connected = True
                logger.info("Connected to API")
                return True
        except:
            logger.warning("API not available, running in standalone mode")
            self.api_connected = False
        return False
    
    async def fetch_waypoints_from_api(self, request_id: str):
        """Fetch waypoints from API execution"""
        try:
            response = requests.get(f"{self.api_url}/api/task/{request_id}")
            if response.status_code == 200:
                data = response.json()
                waypoints = data.get('waypoints', {})
                self.expert_waypoints = waypoints.get('expert', [])
                self.optimized_waypoints = waypoints.get('optimized', [])
                logger.info(f"Fetched waypoints from API")
        except Exception as e:
            logger.error(f"Failed to fetch waypoints: {e}")
    
    def run(self):
        """Main run loop"""
        
        # Setup simulation
        self.setup_simulation()
        
        # Try to connect to API
        asyncio.run(self.connect_to_api())
        
        # Main loop
        last_update = time.time()
        
        while self.running and p.isConnected():
            # Process input
            self.process_keyboard_input()
            
            # Update robot
            self.update_robot_control()
            
            # Check skill execution
            if p.readUserDebugParameter(self.execute_skill_param) > 0.5:
                skill_idx = int(p.readUserDebugParameter(self.skill_param))
                skill_names = list(self.skills_library.keys())
                if skill_idx < len(skill_names):
                    skill_func = self.skills_library[skill_names[skill_idx]]
                    skill_func()
                
                # Reset button
                p.resetBaseDebugParam(self.execute_skill_param, 0)
            
            # Update visualizations (at lower frequency)
            if time.time() - last_update > 0.1:
                self.visualize_waypoints()
                self.show_status_text()
                last_update = time.time()
            
            # Step simulation
            p.stepSimulation()
            time.sleep(1/240)  # 240Hz
        
        # Cleanup
        p.disconnect()
        logger.info("Interactive demo ended")

def main():
    """Main entry point"""
    demo = InteractivePyBulletDemo()
    demo.run()

if __name__ == "__main__":
    main()