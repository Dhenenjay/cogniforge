"""
Example demonstrating waypoint-based motion control using move_through_waypoints().

This script shows how to:
1. Move through simple position waypoints
2. Include orientation constraints in waypoints  
3. Coordinate gripper actions with waypoint motion
4. Adjust motion speed and smoothness parameters
5. Record and visualize trajectories
6. Create complex pick-and-place sequences
"""

import time
import numpy as np
import logging
from cogniforge.core import RobotSimulator, RobotType, SimulationMode, SimulationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def demo_simple_waypoints(sim, robot_name):
    """Demonstrate simple position-only waypoint motion."""
    
    print(f"\n{'='*60}")
    print(f"SIMPLE WAYPOINT MOTION - {robot_name}")
    print('='*60)
    
    # Define waypoints as simple positions
    waypoints = [
        (0.5, 0.0, 0.4),   # Start position
        (0.5, 0.2, 0.4),   # Move right
        (0.3, 0.2, 0.5),   # Move up and forward
        (0.3, -0.2, 0.5),  # Move left
        (0.5, 0.0, 0.3),   # Return near start
    ]
    
    print(f"\n📍 Moving through {len(waypoints)} waypoints:")
    for i, wp in enumerate(waypoints, 1):
        print(f"   {i}. Position: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
    
    # Execute waypoint motion
    print("\n▶️  Executing motion...")
    sim.move_through_waypoints(
        robot_name,
        waypoints,
        steps_per_segment=60,  # 60 steps between waypoints
        action_repeat=4,        # Repeat each action 4 times for stability
    )
    
    print("✅ Motion complete!")


def demo_waypoints_with_orientation(sim, robot_name):
    """Demonstrate waypoints with position and orientation."""
    
    print(f"\n{'='*60}")
    print("WAYPOINTS WITH ORIENTATION")
    print('='*60)
    
    import pybullet as p
    
    # Define waypoints with orientation
    waypoints = [
        {
            'pos': (0.5, 0.0, 0.4),
            'orn': p.getQuaternionFromEuler([0, 0, 0])  # Straight
        },
        {
            'pos': (0.5, 0.2, 0.4),
            'orn': p.getQuaternionFromEuler([0, 0, np.pi/4])  # 45° yaw
        },
        {
            'pos': (0.4, 0.2, 0.5),
            'orn': p.getQuaternionFromEuler([0, np.pi/6, np.pi/4])  # Pitch + yaw
        },
        {
            'pos': (0.4, -0.1, 0.4),
            'orn': p.getQuaternionFromEuler([np.pi/8, 0, 0])  # Roll
        },
    ]
    
    print(f"\n📍 Moving through {len(waypoints)} oriented waypoints")
    
    # Execute with orientation control
    print("\n▶️  Executing oriented motion...")
    sim.move_through_waypoints(
        robot_name,
        waypoints,
        use_orientation=True,  # Enable orientation control
        steps_per_segment=80,   # Slower for smoother orientation changes
        action_repeat=4,
    )
    
    print("✅ Oriented motion complete!")


def demo_speed_variations(sim, robot_name):
    """Demonstrate different motion speeds."""
    
    print(f"\n{'='*60}")
    print("SPEED VARIATIONS")
    print('='*60)
    
    # Same waypoints, different speeds
    waypoints = [
        (0.5, 0.0, 0.4),
        (0.5, 0.2, 0.4),
        (0.5, 0.0, 0.4),
    ]
    
    speed_configs = [
        {"name": "Fast", "steps": 30, "repeat": 2},
        {"name": "Medium", "steps": 60, "repeat": 4},
        {"name": "Slow", "steps": 120, "repeat": 4},
    ]
    
    for config in speed_configs:
        print(f"\n⚡ {config['name']} motion:")
        print(f"   Steps per segment: {config['steps']}")
        print(f"   Action repeat: {config['repeat']}")
        
        start_time = time.time()
        
        sim.move_through_waypoints(
            robot_name,
            waypoints,
            steps_per_segment=config['steps'],
            action_repeat=config['repeat'],
            max_velocity=2.0 if config['name'] == 'Fast' else 1.0,
        )
        
        elapsed = time.time() - start_time
        print(f"   Time taken: {elapsed:.2f} seconds")


def demo_pick_and_place_sequence(sim, robot_name):
    """Demonstrate pick-and-place with waypoints and gripper control."""
    
    print(f"\n{'='*60}")
    print("PICK AND PLACE SEQUENCE")
    print('='*60)
    
    # Spawn objects to manipulate
    print("\n📦 Spawning objects...")
    
    # Spawn blocks at different positions
    block1 = sim.spawn_block(
        color_rgb=(1.0, 0.0, 0.0),
        size=0.04,
        position=(0.5, 0.15, 0.02),
        mass=0.05,
        block_name="red_block"
    )
    
    block2 = sim.spawn_block(
        color_rgb=(0.0, 1.0, 0.0),
        size=0.04,
        position=(0.5, -0.15, 0.02),
        mass=0.05,
        block_name="green_block"
    )
    
    # Let blocks settle
    for _ in range(50):
        sim.step()
    
    # Define pick-and-place waypoints
    waypoints = [
        (0.5, 0.0, 0.5),    # 0: Home position
        (0.5, 0.15, 0.3),   # 1: Above red block
        (0.5, 0.15, 0.05),  # 2: Grasp position for red block
        (0.5, 0.15, 0.3),   # 3: Lift red block
        (0.3, 0.0, 0.3),    # 4: Transport position
        (0.3, 0.0, 0.05),   # 5: Place position
        (0.3, 0.0, 0.3),    # 6: After placing
        (0.5, -0.15, 0.3),  # 7: Above green block
        (0.5, -0.15, 0.05), # 8: Grasp position for green block
        (0.5, -0.15, 0.3),  # 9: Lift green block
        (0.6, 0.0, 0.3),    # 10: Transport to new location
        (0.6, 0.0, 0.05),   # 11: Place position
        (0.6, 0.0, 0.3),    # 12: After placing
        (0.5, 0.0, 0.5),    # 13: Return home
    ]
    
    # Define gripper actions at specific waypoints
    gripper_actions = {
        0: 'open',    # Open at start
        2: 'close',   # Close to grasp red block
        5: 'open',    # Open to release red block
        8: 'close',   # Close to grasp green block
        11: 'open',   # Open to release green block
    }
    
    print(f"\n🦾 Executing pick-and-place sequence...")
    print(f"   {len(waypoints)} waypoints")
    print(f"   {len(gripper_actions)} gripper actions")
    
    # Execute the sequence
    sim.move_through_waypoints(
        robot_name,
        waypoints,
        steps_per_segment=60,
        action_repeat=4,
        gripper_actions=gripper_actions,
        force=80.0,
    )
    
    print("✅ Pick-and-place complete!")


def demo_trajectory_recording(sim, robot_name):
    """Demonstrate trajectory recording and analysis."""
    
    print(f"\n{'='*60}")
    print("TRAJECTORY RECORDING")
    print('='*60)
    
    # Define a figure-8 pattern
    waypoints = []
    n_points = 16
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        x = 0.5 + 0.1 * np.sin(t)
        y = 0.15 * np.sin(2 * t)
        z = 0.4 + 0.05 * np.cos(t)
        waypoints.append((x, y, z))
    
    print(f"\n📈 Recording figure-8 trajectory with {len(waypoints)} waypoints")
    
    # Execute with trajectory recording
    trajectory_data = sim.move_through_waypoints(
        robot_name,
        waypoints,
        steps_per_segment=40,
        action_repeat=2,
        return_trajectories=True,  # Enable recording
    )
    
    # Analyze recorded data
    print("\n📊 Trajectory Analysis:")
    print(f"   Joint trajectory points: {len(trajectory_data['joint_trajectory'])}")
    print(f"   EE positions recorded: {len(trajectory_data['ee_positions'])}")
    print(f"   Waypoint errors: {len(trajectory_data['ee_errors'])}")
    
    if trajectory_data['ee_errors']:
        avg_error = np.mean(trajectory_data['ee_errors'])
        max_error = np.max(trajectory_data['ee_errors'])
        print(f"   Average error: {avg_error:.4f} m")
        print(f"   Maximum error: {max_error:.4f} m")
    
    # Sample some trajectory points
    print("\n📍 Sample EE positions:")
    sample_indices = [0, len(trajectory_data['ee_positions'])//4, 
                     len(trajectory_data['ee_positions'])//2,
                     3*len(trajectory_data['ee_positions'])//4,
                     -1]
    
    for idx in sample_indices:
        if idx < len(trajectory_data['ee_positions']):
            pos = trajectory_data['ee_positions'][idx]
            print(f"   Point {idx}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")


def demo_complex_manipulation(sim, robot_name):
    """Demonstrate complex manipulation with multiple objects."""
    
    print(f"\n{'='*60}")
    print("COMPLEX MANIPULATION")
    print('='*60)
    
    # Create a more complex scene
    print("\n🏗️ Setting up complex scene...")
    
    # Spawn platform
    platform = sim.spawn_platform(
        color_rgb=(0.7, 0.7, 0.7),
        size=0.3,
        position=(0.5, 0, 0.01),
        height=0.01,
    )
    
    # Spawn multiple colored blocks in a grid
    blocks = []
    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0)]
    positions = [(0.4, 0.1), (0.4, -0.1), (0.6, 0.1), (0.6, -0.1)]
    
    for i, (color, (x, y)) in enumerate(zip(colors, positions)):
        block = sim.spawn_block(
            color_rgb=color,
            size=0.03,
            position=(x, y, 0.02),
            mass=0.03,
            block_name=f"block_{i}"
        )
        blocks.append(block)
    
    # Let scene settle
    for _ in range(100):
        sim.step()
    
    # Complex manipulation sequence: Stack blocks
    print("\n🏗️ Stacking blocks...")
    
    # Waypoints for stacking
    stack_position = (0.5, 0.0)
    stack_heights = [0.05, 0.08, 0.11, 0.14]  # Stacking heights
    
    all_waypoints = []
    all_gripper_actions = {}
    waypoint_idx = 0
    
    # Add initial position
    all_waypoints.append((0.5, 0.0, 0.4))
    all_gripper_actions[waypoint_idx] = 'open'
    waypoint_idx += 1
    
    for i, ((x, y), height) in enumerate(zip(positions, stack_heights)):
        # Move above block
        all_waypoints.append((x, y, 0.2))
        waypoint_idx += 1
        
        # Move down to grasp
        all_waypoints.append((x, y, 0.04))
        all_gripper_actions[waypoint_idx] = 'close'
        waypoint_idx += 1
        
        # Lift block
        all_waypoints.append((x, y, 0.2))
        waypoint_idx += 1
        
        # Move to stack position
        all_waypoints.append((stack_position[0], stack_position[1], 0.2))
        waypoint_idx += 1
        
        # Lower to stack
        all_waypoints.append((stack_position[0], stack_position[1], height))
        all_gripper_actions[waypoint_idx] = 'open'
        waypoint_idx += 1
        
        # Move up after placing
        all_waypoints.append((stack_position[0], stack_position[1], 0.2))
        waypoint_idx += 1
    
    # Return to home
    all_waypoints.append((0.5, 0.0, 0.4))
    
    print(f"   Total waypoints: {len(all_waypoints)}")
    print(f"   Gripper actions: {len(all_gripper_actions)}")
    
    # Execute stacking sequence
    sim.move_through_waypoints(
        robot_name,
        all_waypoints,
        steps_per_segment=50,
        action_repeat=3,
        gripper_actions=all_gripper_actions,
        force=60.0,
    )
    
    print("✅ Complex manipulation complete!")


def main():
    """Run the waypoint motion demonstration."""
    
    # Create simulator configuration
    config = SimulationConfig(
        gravity=(0.0, 0.0, -9.81),
        time_step=1.0 / 240.0,
        use_real_time=True,
        camera_distance=1.8,
        camera_yaw=45,
        camera_pitch=-30,
        camera_target=(0.5, 0, 0.2),
    )
    
    # Create simulator
    sim = RobotSimulator(config=config)
    
    try:
        print("🚀 Starting Waypoint Motion Demonstration...")
        sim.connect()
        
        # Load plane
        sim.load_plane()
        
        # Load robot
        print("\n🤖 Loading Franka Panda robot...")
        panda = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="panda",
        )
        
        # Run demonstrations
        
        # 1. Simple waypoints
        demo_simple_waypoints(sim, "panda")
        
        # 2. Waypoints with orientation
        demo_waypoints_with_orientation(sim, "panda")
        
        # 3. Speed variations
        demo_speed_variations(sim, "panda")
        
        # 4. Pick and place
        demo_pick_and_place_sequence(sim, "panda")
        
        # 5. Trajectory recording
        demo_trajectory_recording(sim, "panda")
        
        # 6. Complex manipulation
        demo_complex_manipulation(sim, "panda")
        
        # Reset to home
        print(f"\n{'='*60}")
        print("FINAL RESET")
        print('='*60)
        
        print("\n🏠 Resetting to home position...")
        sim.reset_robot("panda")
        sim.open_gripper("panda")
        
        # Keep simulation running if in GUI mode
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n▶️  Simulation running (press Ctrl+C to stop)...")
            print("   You can interact with the simulation:")
            print("   - Click and drag to rotate camera")
            print("   - Scroll to zoom")
            print("   - Ctrl+Click to move objects")
            
            try:
                while True:
                    sim.step()
                    time.sleep(1/240)
                    
            except KeyboardInterrupt:
                print("\n⏹️  Simulation stopped")
        else:
            # Run for a bit in headless mode
            print("\n▶️  Running headless simulation for 500 steps...")
            for i in range(500):
                sim.step()
                if i % 100 == 0:
                    print(f"   Step {i}/500")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔌 Disconnecting...")
        sim.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main()