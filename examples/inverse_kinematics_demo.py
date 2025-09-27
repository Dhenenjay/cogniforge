"""
Example demonstrating inverse kinematics calculation using calculate_ik().

This script shows how to:
1. Calculate IK for position only
2. Calculate IK with orientation constraints
3. Use null-space control for redundant robots
4. Compare IK solutions with different parameters
5. Validate IK solutions by checking end-effector poses
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


def demo_basic_ik(sim, robot_name):
    """Demonstrate basic IK calculation for position only."""
    
    print(f"\n{'='*60}")
    print(f"BASIC IK CALCULATION - {robot_name}")
    print('='*60)
    
    # Get current EE pose
    current_pos, current_orn = sim.ee_pose(robot_name)
    print(f"\n📍 Current EE position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
    
    # Define target positions
    targets = [
        (0.5, 0.3, 0.4),
        (0.4, -0.3, 0.5),
        (0.6, 0.0, 0.3),
    ]
    
    for i, target_pos in enumerate(targets, 1):
        print(f"\n🎯 Target {i}: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        
        # Calculate IK
        joint_positions = sim.calculate_ik(robot_name, target_pos=target_pos)
        
        print(f"   Joint solution: {[f'{q:.3f}' for q in joint_positions]}")
        
        # Apply the solution
        sim.reset_robot(robot_name, q_default=joint_positions, reset_gripper=False)
        
        # Let robot settle
        for _ in range(50):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.01)
        
        # Verify the result
        achieved_pos, _ = sim.ee_pose(robot_name)
        error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
        print(f"   Achieved: ({achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f})")
        print(f"   Position error: {error:.4f} m")


def demo_ik_with_orientation(sim, robot_name):
    """Demonstrate IK with both position and orientation constraints."""
    
    print(f"\n{'='*60}")
    print("IK WITH ORIENTATION CONSTRAINTS")
    print('='*60)
    
    # Target poses (position + orientation)
    import pybullet as p
    
    targets = [
        {
            "pos": (0.5, 0.2, 0.4),
            "orn": p.getQuaternionFromEuler([0, np.pi/4, 0]),  # 45 deg pitch
            "desc": "45° pitch"
        },
        {
            "pos": (0.4, -0.2, 0.5),
            "orn": p.getQuaternionFromEuler([0, 0, np.pi/2]),  # 90 deg yaw
            "desc": "90° yaw"
        },
        {
            "pos": (0.6, 0.0, 0.35),
            "orn": p.getQuaternionFromEuler([np.pi/6, 0, 0]),  # 30 deg roll
            "desc": "30° roll"
        },
    ]
    
    for i, target in enumerate(targets, 1):
        print(f"\n🎯 Target {i} - {target['desc']}:")
        print(f"   Position: ({target['pos'][0]:.3f}, {target['pos'][1]:.3f}, {target['pos'][2]:.3f})")
        
        # Calculate IK with orientation
        joint_positions = sim.calculate_ik(
            robot_name,
            target_pos=target["pos"],
            target_orn=target["orn"]
        )
        
        print(f"   Joint solution: {[f'{q:.3f}' for q in joint_positions[:7]]}")
        
        # Apply the solution
        sim.reset_robot(robot_name, q_default=joint_positions, reset_gripper=False)
        
        # Let robot settle
        for _ in range(50):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.01)
        
        # Verify the result
        achieved_pos, achieved_orn = sim.ee_pose(robot_name)
        pos_error = np.linalg.norm(np.array(achieved_pos) - np.array(target["pos"]))
        
        # Quaternion dot product for orientation error
        orn_similarity = abs(np.dot(achieved_orn, target["orn"]))
        
        print(f"   Position error: {pos_error:.4f} m")
        print(f"   Orientation similarity: {orn_similarity:.3f} (1.0 = perfect)")


def demo_nullspace_control(sim, robot_name):
    """Demonstrate null-space control for redundant manipulators."""
    
    print(f"\n{'='*60}")
    print("NULL-SPACE CONTROL")
    print('='*60)
    
    target_pos = (0.5, 0.1, 0.4)
    print(f"\n🎯 Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    
    # Different rest poses for null-space
    rest_poses_options = [
        {
            "name": "Elbow up",
            "poses": [0, -0.5, 0, -1.5, 0, 1.0, 0.785],
        },
        {
            "name": "Elbow down",
            "poses": [0, 0.5, 0, -2.0, 0, 1.5, 0.785],
        },
        {
            "name": "Compact",
            "poses": [0, 0, 0, -2.3, 0, 2.3, 0.785],
        },
    ]
    
    for config in rest_poses_options:
        print(f"\n🔧 Configuration: {config['name']}")
        
        # Get joint limits from the robot
        robot = sim.robots[robot_name]
        n_joints = len(robot.joint_indices)
        
        # Calculate IK with null-space
        joint_positions = sim.calculate_ik(
            robot_name,
            target_pos=target_pos,
            use_nullspace=True,
            rest_poses=config["poses"][:n_joints],
            joint_lower_limits=[-2.96] * n_joints,
            joint_upper_limits=[2.96] * n_joints,
            joint_ranges=[5.92] * n_joints,
        )
        
        print(f"   Joint solution: {[f'{q:.3f}' for q in joint_positions[:7]]}")
        
        # Apply the solution
        sim.reset_robot(robot_name, q_default=joint_positions, reset_gripper=False)
        
        # Let robot settle
        for _ in range(50):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.01)
        
        # Verify position is still correct
        achieved_pos, _ = sim.ee_pose(robot_name)
        error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
        print(f"   Position error: {error:.4f} m")


def demo_ik_comparison(sim, robot_name):
    """Compare IK solutions with different parameters."""
    
    print(f"\n{'='*60}")
    print("IK PARAMETER COMPARISON")
    print('='*60)
    
    target_pos = (0.45, 0.15, 0.45)
    print(f"\n🎯 Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    
    # Different IK configurations
    configs = [
        {
            "name": "Default (low damping)",
            "joint_damping": None,  # Will use default
            "max_iterations": 100,
        },
        {
            "name": "High damping (stable)",
            "joint_damping": [0.1] * 7,
            "max_iterations": 100,
        },
        {
            "name": "More iterations",
            "joint_damping": [0.05] * 7,
            "max_iterations": 200,
        },
        {
            "name": "High precision",
            "joint_damping": [0.01] * 7,
            "max_iterations": 150,
            "residual_threshold": 0.0001,
        },
    ]
    
    for config in configs:
        print(f"\n🔧 {config['name']}:")
        
        # Build kwargs
        kwargs = {
            "target_pos": target_pos,
            "max_iterations": config.get("max_iterations", 100),
        }
        
        if config.get("joint_damping"):
            kwargs["joint_damping"] = config["joint_damping"]
        
        if "residual_threshold" in config:
            kwargs["residual_threshold"] = config["residual_threshold"]
        
        # Calculate IK
        try:
            joint_positions = sim.calculate_ik(robot_name, **kwargs)
            
            # Apply and verify
            sim.reset_robot(robot_name, q_default=joint_positions, reset_gripper=False)
            
            # Quick settle
            for _ in range(30):
                sim.step()
            
            achieved_pos, _ = sim.ee_pose(robot_name)
            error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
            
            print(f"   Solution found: {[f'{q:.3f}' for q in joint_positions[:4]]}...")
            print(f"   Position error: {error:.6f} m")
            
        except Exception as e:
            print(f"   Failed: {e}")


def demo_ik_workspace_exploration(sim, robot_name):
    """Explore reachable workspace using IK."""
    
    print(f"\n{'='*60}")
    print("WORKSPACE EXPLORATION")
    print('='*60)
    
    # Test grid of positions
    x_range = np.linspace(0.3, 0.6, 3)
    y_range = np.linspace(-0.3, 0.3, 3)
    z_range = np.linspace(0.2, 0.5, 3)
    
    reachable = 0
    unreachable = 0
    
    print("\nTesting reachability of workspace points...")
    print("  ✅ = Reachable, ❌ = Unreachable\n")
    
    for z in z_range:
        print(f"\n  Z = {z:.2f}:")
        for y in y_range:
            row = "    "
            for x in x_range:
                target_pos = (x, y, z)
                
                try:
                    # Try to calculate IK
                    joint_positions = sim.calculate_ik(
                        robot_name,
                        target_pos=target_pos,
                        max_iterations=50,
                        residual_threshold=0.01
                    )
                    
                    # Apply solution
                    sim.reset_robot(robot_name, q_default=joint_positions, reset_gripper=False)
                    
                    # Check if we reached the target
                    achieved_pos, _ = sim.ee_pose(robot_name)
                    error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
                    
                    if error < 0.01:
                        row += "✅ "
                        reachable += 1
                    else:
                        row += "❌ "
                        unreachable += 1
                        
                except:
                    row += "❌ "
                    unreachable += 1
            
            print(row)
    
    print(f"\n  Summary: {reachable} reachable, {unreachable} unreachable")
    print(f"  Reachability: {100*reachable/(reachable+unreachable):.1f}%")


def demo_trajectory_ik(sim, robot_name):
    """Generate smooth trajectory using IK."""
    
    print(f"\n{'='*60}")
    print("TRAJECTORY GENERATION WITH IK")
    print('='*60)
    
    # Create circular trajectory
    center = np.array([0.5, 0.0, 0.4])
    radius = 0.1
    n_points = 8
    
    print(f"\n📐 Generating circular trajectory:")
    print(f"   Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"   Radius: {radius:.2f} m")
    print(f"   Points: {n_points}")
    
    # Generate trajectory points
    trajectory = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
        trajectory.append(tuple(pos))
    
    # Execute trajectory
    print("\n▶️  Executing trajectory...")
    
    for i, target_pos in enumerate(trajectory):
        print(f"\n  Point {i+1}/{n_points}: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        
        # Calculate IK for this point
        joint_positions = sim.calculate_ik(robot_name, target_pos=target_pos)
        
        # Move to position
        sim.reset_robot(robot_name, q_default=joint_positions, reset_gripper=False)
        
        # Simulate movement
        for _ in range(30):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.01)
        
        # Check accuracy
        achieved_pos, _ = sim.ee_pose(robot_name)
        error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
        print(f"     Error: {error:.4f} m")
    
    print("\n✅ Trajectory complete!")


def main():
    """Run the inverse kinematics demonstration."""
    
    # Create simulator configuration
    config = SimulationConfig(
        gravity=(0.0, 0.0, -9.81),
        time_step=1.0 / 240.0,
        use_real_time=True,
        camera_distance=1.5,
        camera_yaw=45,
        camera_pitch=-30,
        camera_target=(0.5, 0, 0.3),
    )
    
    # Create simulator
    sim = RobotSimulator(config=config)
    
    try:
        print("🚀 Starting Inverse Kinematics Demonstration...")
        sim.connect()
        
        # Load plane
        sim.load_plane()
        
        # Load robots
        print("\n🤖 Loading robots...")
        
        # Load Panda (7-DOF, redundant)
        panda = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="panda",
        )
        
        # Add visual reference
        platform = sim.spawn_platform(
            color_rgb=(0.9, 0.9, 0.9),
            size=0.3,
            position=(0.5, 0, 0.01),
            height=0.005,
        )
        
        # Add target markers (colored blocks)
        sim.spawn_block(
            color_rgb=(1.0, 0.0, 0.0),
            size=0.02,
            position=(0.5, 0.3, 0.4),
            mass=0,  # Static
            block_name="target1"
        )
        
        sim.spawn_block(
            color_rgb=(0.0, 1.0, 0.0),
            size=0.02,
            position=(0.4, -0.3, 0.5),
            mass=0,
            block_name="target2"
        )
        
        sim.spawn_block(
            color_rgb=(0.0, 0.0, 1.0),
            size=0.02,
            position=(0.6, 0.0, 0.3),
            mass=0,
            block_name="target3"
        )
        
        # Run demonstrations
        
        # 1. Basic IK
        demo_basic_ik(sim, "panda")
        
        # 2. IK with orientation
        demo_ik_with_orientation(sim, "panda")
        
        # 3. Null-space control
        demo_nullspace_control(sim, "panda")
        
        # 4. Parameter comparison
        demo_ik_comparison(sim, "panda")
        
        # 5. Workspace exploration
        demo_ik_workspace_exploration(sim, "panda")
        
        # 6. Trajectory generation
        demo_trajectory_ik(sim, "panda")
        
        # Reset to home
        print(f"\n{'='*60}")
        print("FINAL RESET")
        print('='*60)
        
        print("\n🏠 Resetting to home position...")
        sim.reset_robot("panda")
        
        # Keep simulation running if in GUI mode
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n▶️  Simulation running (press Ctrl+C to stop)...")
            print("   Click and drag in the simulation to move camera")
            print("   Scroll to zoom in/out")
            
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