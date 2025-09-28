#!/usr/bin/env python
"""
Cogniforge PyBullet Integration Demo
Demonstrates a simple robotic simulation using PyBullet
"""

import pybullet as p
import pybullet_data
import time
import math

class CogniforgeSimulation:
    """A simple physics simulation class for Cogniforge."""
    
    def __init__(self, gui=True):
        """Initialize the simulation environment."""
        # Connect to PyBullet
        if gui:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        # Set up simulation
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load environment
        self.planeId = p.loadURDF("plane.urdf")
        
        # Store robot IDs
        self.robots = []
        
    def add_robot(self, position=[0, 0, 1]):
        """Add a robot to the simulation."""
        # Load Kuka robot arm
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", position, useFixedBase=True)
        self.robots.append(robot_id)
        
        # Get joint info
        num_joints = p.getNumJoints(robot_id)
        print(f"Added robot with {num_joints} joints at position {position}")
        
        return robot_id
    
    def add_object(self, position=[2, 0, 0.5], color=[1, 0, 0, 1]):
        """Add a simple cube object to the simulation."""
        # Create a cube
        cube_size = 0.1
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size]*3)
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=color)
        
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=position
        )
        
        print(f"Added cube at position {position}")
        return cube_id
    
    def control_robot_joints(self, robot_id, joint_positions):
        """Control robot joint positions."""
        num_joints = p.getNumJoints(robot_id)
        
        # Set joint positions
        for joint_idx in range(min(len(joint_positions), num_joints)):
            p.setJointMotorControl2(
                robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[joint_idx]
            )
    
    def simulate_step(self):
        """Run one simulation step."""
        p.stepSimulation()
    
    def get_robot_state(self, robot_id):
        """Get current state of the robot."""
        num_joints = p.getNumJoints(robot_id)
        joint_states = []
        
        for i in range(num_joints):
            state = p.getJointState(robot_id, i)
            joint_states.append({
                'position': state[0],
                'velocity': state[1]
            })
        
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        
        return {
            'base_position': base_pos,
            'base_orientation': base_orn,
            'joint_states': joint_states
        }
    
    def cleanup(self):
        """Clean up the simulation."""
        p.disconnect()

def demo_simulation():
    """Run a demonstration simulation."""
    print("=" * 60)
    print("Cogniforge PyBullet Integration Demo")
    print("=" * 60)
    
    # Create simulation
    sim = CogniforgeSimulation(gui=False)  # Set to True for GUI
    
    # Add robots
    robot1 = sim.add_robot(position=[0, 0, 0])
    robot2 = sim.add_robot(position=[2, 0, 0])
    
    # Add objects
    for i in range(5):
        sim.add_object(
            position=[1, i*0.3 - 0.6, 0.5],
            color=[i/5, 0, 1-i/5, 1]
        )
    
    print("\nRunning simulation for 1000 steps...")
    
    # Simulate
    for step in range(1000):
        # Create sinusoidal joint motion
        time_val = step * 0.01
        joint_positions = [
            math.sin(time_val) * 0.5,
            math.cos(time_val) * 0.3,
            math.sin(time_val * 2) * 0.4,
            0, 0, 0, 0
        ]
        
        # Control robots
        sim.control_robot_joints(robot1, joint_positions)
        sim.control_robot_joints(robot2, [-jp for jp in joint_positions])  # Mirror motion
        
        # Step simulation
        sim.simulate_step()
        
        # Print state every 100 steps
        if step % 100 == 0:
            state = sim.get_robot_state(robot1)
            print(f"Step {step}: Robot at {[round(p, 3) for p in state['base_position']]}")
    
    print("\nSimulation complete!")
    
    # Clean up
    sim.cleanup()
    
    print("=" * 60)
    print("PyBullet integration successful!")
    print("=" * 60)

if __name__ == "__main__":
    demo_simulation()
    
    print("\nTo run with GUI visualization, edit the script and set gui=True")
    print("in the CogniforgeSimulation initialization.")