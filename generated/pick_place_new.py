#!/usr/bin/env python3
"""
Generated Pick and Place Task
Python 3.11 compatible, no external deps beyond project modules

This file recreates the PyBullet scene, loads optimized waypoints W_star,
pauses at pre-grasp to request vision offset via get_offset(),
applies clamped world nudge, and completes pick-place.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import json
import os
from typing import List, Tuple, Dict, Optional


class PickPlaceExecution:
    """Execute pick-place with optimized waypoints and vision feedback"""
    
    def __init__(self, headless: bool = False):
        """Initialize PyBullet environment"""
        self.client = p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Physics
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        p.setRealTimeSimulation(0)
        
        # IDs
        self.robot = None
        self.cube = None 
        self.platform = None
        
        # Panda constants
        self.ee_link = 11
        self.finger_joints = [9, 10]
        self.arm_joints = list(range(7))
        
        # Waypoints
        self.W_star = []
        
        # Vision nudge limit
        self.max_nudge_m = 0.015  # 1.5cm max
        
    def recreate_scene(self):
        """Recreate PyBullet scene with robot, cube, platform"""
        
        # Ground plane
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Panda robot at origin
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Set initial joint positions
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 0.785, 0.04, 0.04]
        for i in range(len(rest_poses)):
            p.resetJointState(self.robot, i, rest_poses[i])
        
        # Blue cube (4cm)
        cube_size = 0.04
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2]*3,
            rgbaColor=[0, 0.3, 1, 1]
        )
        cube_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2]*3
        )
        self.cube = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.5, 0, cube_size/2]
        )
        
        # Green platform (10x10x2cm)
        platform_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.01],
            rgbaColor=[0, 0.8, 0, 1]
        )
        platform_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.01]
        )
        self.platform = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0.3, 0.3, 0.01]
        )
        
        print("✓ Scene recreated")
        
    def load_waypoints(self, filepath: str = "W_star.json"):
        """Load optimized waypoints W_star from file"""
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.W_star = data.get('W_star', data.get('waypoints', []))
                print(f"✓ Loaded {len(self.W_star)} waypoints from {filepath}")
        else:
            # Generate default waypoints
            print("⚠ No W_star.json found, generating defaults")
            self.W_star = self._generate_default_waypoints()
            
    def _generate_default_waypoints(self) -> List[List[float]]:
        """Generate default waypoints if W_star not found"""
        
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        platform_pos, _ = p.getBasePositionAndOrientation(self.platform)
        
        waypoints = []
        
        # W1: Pre-grasp
        target1 = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
        joints1 = self._compute_ik(target1)
        waypoints.append(joints1)
        
        # W2: Grasp
        target2 = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.05]
        joints2 = self._compute_ik(target2)
        waypoints.append(joints2)
        
        # W3: Lift
        target3 = [cube_pos[0], cube_pos[1], 0.2]
        joints3 = self._compute_ik(target3)
        waypoints.append(joints3)
        
        # W4: Transport
        target4 = [platform_pos[0], platform_pos[1], 0.2]
        joints4 = self._compute_ik(target4)
        waypoints.append(joints4)
        
        # W5: Place
        target5 = [platform_pos[0], platform_pos[1], platform_pos[2] + 0.06]
        joints5 = self._compute_ik(target5)
        waypoints.append(joints5)
        
        return waypoints
        
    def _compute_ik(self, target_pos: List[float], 
                   target_orn: Optional[List[float]] = None) -> List[float]:
        """Compute IK for target pose"""
        
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        elif len(target_orn) == 3:
            target_orn = p.getQuaternionFromEuler(target_orn)
            
        ik_solution = p.calculateInverseKinematics(
            self.robot,
            self.ee_link,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        return list(ik_solution[:7])
        
    def get_offset(self) -> Dict[str, int]:
        """
        Request vision offset from wrist camera
        Returns pixel offset of blue cube from image center
        """
        
        # Get end-effector state
        ee_state = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        ee_pos = ee_state[4]
        ee_orn = ee_state[5]
        
        # Compute view matrix from EE pose
        rot_matrix = p.getMatrixFromQuaternion(ee_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # Camera positioned at EE looking down
        camera_eye_pos = ee_pos
        camera_target_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.5]
        camera_up_vector = [0, 1, 0]
        
        view_matrix = p.computeViewMatrix(
            camera_eye_pos,
            camera_target_pos,
            camera_up_vector
        )
        
        # Projection matrix (640x480 image)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=640/480,
            nearVal=0.01,
            farVal=1.0
        )
        
        # Capture image
        width, height = 640, 480
        img = p.getCameraImage(width, height, view_matrix, proj_matrix)
        
        # Get cube position in world
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        
        # Project cube to image (simplified)
        dx_world = cube_pos[0] - ee_pos[0]
        dy_world = cube_pos[1] - ee_pos[1]
        dz_world = ee_pos[2] - cube_pos[2]
        
        # Simplified projection to pixels
        focal_length_px = 600
        if dz_world > 0:
            dx_px = int(-focal_length_px * dx_world / dz_world)
            dy_px = int(-focal_length_px * dy_world / dz_world) 
        else:
            dx_px = dy_px = 0
            
        # Add small noise to simulate real vision
        dx_px += np.random.randint(-3, 4)
        dy_px += np.random.randint(-3, 4)
        
        print(f"  Vision offset: dx={dx_px}px, dy={dy_px}px")
        
        return {"dx_px": dx_px, "dy_px": dy_px}
        
    def apply_nudge(self, offset: Dict[str, int], 
                   current_joints: List[float]) -> List[float]:
        """
        Apply clamped world nudge based on vision offset
        Returns adjusted joint configuration
        """
        
        # Convert pixels to meters (rough calibration)
        pixels_per_meter = 5000  # At typical working distance
        dx_m = -offset["dx_px"] / pixels_per_meter
        dy_m = -offset["dy_px"] / pixels_per_meter
        
        # Clamp nudge
        dx_m = np.clip(dx_m, -self.max_nudge_m, self.max_nudge_m)
        dy_m = np.clip(dy_m, -self.max_nudge_m, self.max_nudge_m)
        
        print(f"  World nudge: dx={dx_m:.4f}m, dy={dy_m:.4f}m (clamped)")
        
        # Get current EE position
        ee_state = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        ee_pos = list(ee_state[4])
        
        # Apply nudge
        nudged_pos = [
            ee_pos[0] + dx_m,
            ee_pos[1] + dy_m,
            ee_pos[2]
        ]
        
        # Compute new IK
        nudged_joints = self._compute_ik(nudged_pos)
        
        return nudged_joints
        
    def move_to_joints(self, target_joints: List[float], duration: float = 1.5):
        """Execute smooth motion to target joint configuration"""
        
        # Get current joints
        current_joints = []
        for i in self.arm_joints:
            current_joints.append(p.getJointState(self.robot, i)[0])
            
        # Smooth interpolation
        steps = int(duration * 240)
        for step in range(steps):
            alpha = (step + 1) / steps
            # Smooth S-curve
            s = 6 * alpha**5 - 15 * alpha**4 + 10 * alpha**3
            
            for i, joint_id in enumerate(self.arm_joints):
                pos = current_joints[i] + s * (target_joints[i] - current_joints[i])
                p.setJointMotorControl2(
                    self.robot, joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=pos,
                    force=150
                )
                
            p.stepSimulation()
            time.sleep(1/240)
            
    def open_gripper(self):
        """Open gripper fingers"""
        for joint in self.finger_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=20
            )
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1/240)
            
    def close_gripper(self):
        """Close gripper fingers"""
        for joint in self.finger_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=40
            )
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1/240)
            
    def execute(self):
        """Main execution: load waypoints, apply vision, complete task"""
        
        print("\n" + "="*60)
        print(" PICK-PLACE EXECUTION WITH VISION")
        print("="*60)
        
        # Setup
        self.recreate_scene()
        self.load_waypoints()
        
        # Let scene settle
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1/480)
            
        # Execute waypoints
        print("\n[1] Moving to pre-grasp...")
        self.open_gripper()
        self.move_to_joints(self.W_star[0])
        
        print("\n[2] Pausing for vision offset...")
        time.sleep(0.5)
        offset = self.get_offset()
        
        print("\n[3] Applying vision nudge to grasp waypoint...")
        adjusted_grasp = self.apply_nudge(offset, self.W_star[1])
        
        print("\n[4] Moving to adjusted grasp position...")
        self.move_to_joints(adjusted_grasp)
        
        print("\n[5] Grasping...")
        self.close_gripper()
        time.sleep(0.5)
        
        print("\n[6] Lifting...")
        self.move_to_joints(self.W_star[2])
        
        print("\n[7] Transporting...")
        self.move_to_joints(self.W_star[3])
        
        print("\n[8] Placing...")
        self.move_to_joints(self.W_star[4])
        
        print("\n[9] Releasing...")
        self.open_gripper()
        time.sleep(0.5)
        
        print("\n[10] Retreating...")
        self.move_to_joints(self.W_star[2])
        
        print("\n" + "="*60)
        print(" ✓ PICK-PLACE COMPLETE")
        print("="*60)
        
        # Keep visualization
        if p.getConnectionInfo()['connectionMethod'] == p.GUI:
            print("\nPress Enter to exit...")
            input()
            
    def cleanup(self):
        """Disconnect from PyBullet"""
        p.disconnect()


def main():
    """Entry point for generated pick_place.py"""
    
    task = PickPlaceExecution(headless=False)
    
    try:
        task.execute()
    finally:
        task.cleanup()


if __name__ == "__main__":
    main()