"""
Example demonstrating end-effector control and pose tracking.

This script shows how to:
1. Get end-effector pose using ee_pose()
2. Set end-effector target positions
3. Control gripper
4. Track EE velocity and Jacobian
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


def print_ee_info(sim, robot_name):
    """Print detailed end-effector information."""
    # Get EE pose
    pos, orn = sim.ee_pose(robot_name)
    print(f"\n📍 End-Effector Pose for '{robot_name}':")
    print(f"   Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    print(f"   Orientation (quaternion): {[f'{o:.3f}' for o in orn]}")
    
    # Convert quaternion to Euler angles for readability
    import pybullet as p
    euler = p.getEulerFromQuaternion(orn)
    print(f"   Orientation (euler): roll={np.degrees(euler[0]):.1f}°, "
          f"pitch={np.degrees(euler[1]):.1f}°, yaw={np.degrees(euler[2]):.1f}°")
    
    # Get gripper info
    gripper_info = sim.get_gripper_info(robot_name)
    if gripper_info["has_gripper"]:
        print(f"   Gripper opening: {gripper_info['current_opening']:.3f}")
    
    return pos, orn


def move_to_positions(sim, robot_name, positions, wait_time=2.0):
    """Move end-effector through a series of positions."""
    for i, target_pos in enumerate(positions):
        print(f"\n🎯 Moving to position {i+1}: {target_pos}")
        
        # Set EE position (orientation unconstrained)
        success = sim.set_ee_pose(robot_name, target_pos)
        
        if success:
            print("   ✅ Target reached")
        else:
            print("   ⚠️ Target not fully reached (within tolerance)")
        
        # Get current pose
        pos, orn = sim.ee_pose(robot_name)
        error = np.linalg.norm(np.array(pos) - np.array(target_pos))
        print(f"   Position error: {error:.4f}m")
        
        # Simulate for a bit
        if sim._simulation_mode == SimulationMode.GUI:
            for _ in range(int(wait_time * 240)):  # 240Hz
                sim.step()
                time.sleep(1/240)
        else:
            for _ in range(100):
                sim.step()


def demonstrate_gripper(sim, robot_name):
    """Demonstrate gripper control."""
    if robot_name not in sim.robots:
        return
    
    robot = sim.robots[robot_name]
    if not robot.gripper_indices:
        print(f"\n⚠️ Robot '{robot_name}' has no gripper")
        return
    
    print(f"\n🤖 Demonstrating gripper control for '{robot_name}'")
    
    # Open gripper
    print("   Opening gripper...")
    sim.set_gripper(robot_name, 1.0)  # Fully open
    for _ in range(50):
        sim.step()
    
    # Close gripper
    print("   Closing gripper...")
    sim.set_gripper(robot_name, 0.0)  # Fully closed
    for _ in range(50):
        sim.step()
    
    # Half open
    print("   Half opening gripper...")
    sim.set_gripper(robot_name, 0.5)  # Half open
    for _ in range(50):
        sim.step()


def track_ee_motion(sim, robot_name, duration=2.0):
    """Track end-effector motion and velocities."""
    print(f"\n📊 Tracking end-effector motion for {duration} seconds...")
    
    positions = []
    velocities = []
    timestamps = []
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < duration:
        # Get EE pose
        pos, orn = sim.ee_pose(robot_name)
        positions.append(pos)
        
        # Get EE velocity
        try:
            lin_vel, ang_vel = sim.get_ee_velocity(robot_name)
            velocities.append(lin_vel)
        except:
            velocities.append((0, 0, 0))
        
        timestamps.append(time.time() - start_time)
        
        # Step simulation
        sim.step()
        step_count += 1
        
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(1/240)
    
    # Analyze motion
    positions = np.array(positions)
    if len(positions) > 1:
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        print(f"   Total distance traveled: {total_distance:.4f}m")
        print(f"   Average position: {np.mean(positions, axis=0)}")
        print(f"   Position std dev: {np.std(positions, axis=0)}")
        
        if velocities:
            velocities = np.array(velocities)
            avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
            print(f"   Average speed: {avg_speed:.4f} m/s")


def main():
    """Run the end-effector control demonstration."""
    
    # Create simulator configuration
    config = SimulationConfig(
        gravity=(0.0, 0.0, -9.81),
        time_step=1.0 / 240.0,
        use_real_time=True,
        camera_distance=1.2,
        camera_yaw=45,
        camera_pitch=-30,
        camera_target=(0.3, 0, 0.3),
    )
    
    # Create simulator
    sim = RobotSimulator(config=config)
    
    try:
        print("🚀 Starting End-Effector Control Demo...")
        sim.connect()
        
        # Load plane
        plane_id = sim.load_plane()
        
        # Load KUKA robot
        print("\n🤖 Loading KUKA iiwa robot...")
        kuka = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="kuka",
        )
        print(f"   Robot loaded with tool_link_index: {kuka.tool_link_index}")
        
        # Load Panda robot
        print("\n🤖 Loading Franka Panda robot...")
        panda = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0.8, 0, 0),
            fixed_base=True,
            robot_name="panda",
        )
        print(f"   Robot loaded with tool_link_index: {panda.tool_link_index}")
        
        # Spawn a platform to reach
        platform = sim.spawn_platform(
            color_rgb=(0.7, 0.7, 0.7),
            size=0.15,
            position=(0.4, 0.3, 0.2),
        )
        print("\n📦 Platform spawned as target")
        
        # Spawn blocks to manipulate
        block1 = sim.spawn_block(
            color_rgb=(1.0, 0.0, 0.0),
            size=0.03,
            position=(0.3, 0.2, 0.05),
            block_name="red_block",
        )
        
        block2 = sim.spawn_block(
            color_rgb=(0.0, 0.0, 1.0),
            size=0.03,
            position=(0.5, -0.2, 0.05),
            block_name="blue_block",
        )
        print("🎯 Blocks spawned for manipulation")
        
        # Print initial EE info for both robots
        print("\n" + "="*50)
        print("INITIAL STATE")
        print("="*50)
        print_ee_info(sim, "kuka")
        print_ee_info(sim, "panda")
        
        # Define target positions for KUKA
        kuka_targets = [
            (0.4, 0.0, 0.3),    # Forward
            (0.3, 0.2, 0.4),    # Left and up
            (0.3, -0.2, 0.2),   # Right and down
            (0.4, 0.3, 0.35),   # Above platform
        ]
        
        # Move KUKA through positions
        print("\n" + "="*50)
        print("KUKA MOVEMENT DEMO")
        print("="*50)
        move_to_positions(sim, "kuka", kuka_targets, wait_time=0.5)
        
        # Demonstrate Panda gripper
        print("\n" + "="*50)
        print("PANDA GRIPPER DEMO")
        print("="*50)
        demonstrate_gripper(sim, "panda")
        
        # Get Jacobian for KUKA
        print("\n" + "="*50)
        print("JACOBIAN ANALYSIS")
        print("="*50)
        try:
            lin_jac, ang_jac = sim.get_ee_jacobian("kuka")
            print(f"KUKA Linear Jacobian shape: {lin_jac.shape}")
            print(f"KUKA Angular Jacobian shape: {ang_jac.shape}")
            
            # Analyze manipulability
            manipulability = np.sqrt(np.linalg.det(lin_jac @ lin_jac.T))
            print(f"KUKA Manipulability index: {manipulability:.4f}")
        except Exception as e:
            print(f"Could not compute Jacobian: {e}")
        
        # Track motion while moving
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n" + "="*50)
            print("MOTION TRACKING")
            print("="*50)
            
            # Set a circular motion target
            print("\n🔄 Executing circular motion with KUKA...")
            radius = 0.15
            center = (0.4, 0.0, 0.3)
            
            for angle in np.linspace(0, 2*np.pi, 20):
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = center[2]
                
                sim.set_ee_pose("kuka", (x, y, z))
                
                # Step and display
                for _ in range(10):
                    sim.step()
                    time.sleep(1/240)
                
                # Print current position
                pos, _ = sim.ee_pose("kuka")
                print(f"   Angle {np.degrees(angle):.0f}°: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            
            # Track the motion
            track_ee_motion(sim, "kuka", duration=1.0)
        
        # Final state
        print("\n" + "="*50)
        print("FINAL STATE")
        print("="*50)
        print_ee_info(sim, "kuka")
        print_ee_info(sim, "panda")
        
        # Keep simulation running if in GUI mode
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n▶️  Simulation running (press Ctrl+C to stop)...")
            try:
                while True:
                    sim.step()
                    time.sleep(1/240)
            except KeyboardInterrupt:
                print("\n⏹️  Simulation stopped")
    
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