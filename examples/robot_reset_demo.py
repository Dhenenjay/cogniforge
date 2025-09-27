"""
Example demonstrating robot reset functionality with custom joint configurations.

This script shows how to:
1. Reset robot to default home position
2. Reset robot with custom q_default joint positions
3. Reset only arm joints (not gripper)
4. Save and restore robot configurations
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


def demonstrate_reset_positions(sim, robot_name):
    """Demonstrate different reset positions for a robot."""
    
    print(f"\n{'='*60}")
    print(f"RESET DEMONSTRATIONS FOR {robot_name}")
    print('='*60)
    
    # Get initial position
    initial_q = sim.get_joint_positions(robot_name, include_gripper=False)
    print(f"\n📍 Initial joint positions ({len(initial_q)} joints):")
    print(f"   {[f'{q:.3f}' for q in initial_q]}")
    
    # 1. Reset to default home position
    print("\n1️⃣ Reset to default home position:")
    default_q = sim.reset_robot(robot_name)
    print(f"   Home position: {[f'{q:.3f}' for q in default_q[:7]]}")
    
    # Simulate briefly
    for _ in range(100):
        sim.step()
    
    # 2. Reset to custom position (straight up)
    print("\n2️⃣ Reset to straight-up configuration:")
    if robot_name == "kuka":
        straight_q = [0, 0, 0, 0, 0, 0, 0]  # All zeros for KUKA
    elif robot_name == "panda":
        straight_q = [0, 0, 0, -1.57, 0, 1.57, 0]  # Panda straight
    else:
        straight_q = [0] * len(initial_q)
    
    actual_q = sim.reset_robot(robot_name, q_default=straight_q, reset_gripper=False)
    print(f"   Straight config: {[f'{q:.3f}' for q in actual_q]}")
    
    for _ in range(100):
        sim.step()
    
    # 3. Reset to extended reach position
    print("\n3️⃣ Reset to extended reach position:")
    if robot_name == "kuka":
        reach_q = [0, 0.7, 0, -1.0, 0, 0.5, 0]
    elif robot_name == "panda":
        reach_q = [0, -0.3, 0, -2.0, 0, 1.8, 0.785]
    else:
        reach_q = [0.5] * len(initial_q)
    
    actual_q = sim.reset_robot(robot_name, q_default=reach_q, reset_gripper=False)
    print(f"   Reach position: {[f'{q:.3f}' for q in actual_q]}")
    
    for _ in range(100):
        sim.step()
    
    # 4. Reset to compact position
    print("\n4️⃣ Reset to compact/folded position:")
    if robot_name == "kuka":
        compact_q = [0, 1.5, 0, -2.3, 0, 0.7, 0]
    elif robot_name == "panda":
        compact_q = [0, 0.5, 0, -2.8, 0, 3.0, 0.785]
    else:
        compact_q = [-0.5] * len(initial_q)
    
    actual_q = sim.reset_robot(robot_name, q_default=compact_q, reset_gripper=False)
    print(f"   Compact position: {[f'{q:.3f}' for q in actual_q]}")
    
    # Get end-effector position for each configuration
    pos, orn = sim.ee_pose(robot_name)
    print(f"   End-effector at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")


def demonstrate_configuration_memory(sim, robot_name):
    """Demonstrate saving and restoring robot configurations."""
    
    print(f"\n{'='*60}")
    print("CONFIGURATION SAVE/RESTORE")
    print('='*60)
    
    # Dictionary to store configurations
    saved_configs = {}
    
    # Save current configuration
    print("\n💾 Saving configurations...")
    
    # Configuration 1: Current position
    saved_configs["initial"] = sim.get_joint_positions(robot_name, include_gripper=False)
    print(f"   Saved 'initial': {len(saved_configs['initial'])} joints")
    
    # Move to a new position and save
    sim.reset_robot(robot_name, q_default=[0.3, 0.6, -0.3, -1.2, 0.5, 0.8, 0.2][:len(saved_configs["initial"])])
    for _ in range(50):
        sim.step()
    saved_configs["work_pose"] = sim.get_joint_positions(robot_name, include_gripper=False)
    print(f"   Saved 'work_pose': {len(saved_configs['work_pose'])} joints")
    
    # Another position
    sim.reset_robot(robot_name, q_default=[0, 0, 0, -1.57, 0, 1.57, 0.785][:len(saved_configs["initial"])])
    for _ in range(50):
        sim.step()
    saved_configs["transport"] = sim.get_joint_positions(robot_name, include_gripper=False)
    print(f"   Saved 'transport': {len(saved_configs['transport'])} joints")
    
    # Now restore configurations
    print("\n♻️ Restoring configurations...")
    
    for config_name, q_values in saved_configs.items():
        print(f"\n   Restoring '{config_name}'...")
        sim.reset_robot(robot_name, q_default=q_values)
        
        # Verify restoration
        restored_q = sim.get_joint_positions(robot_name, include_gripper=False)
        error = np.mean(np.abs(np.array(restored_q) - np.array(q_values)))
        print(f"   ✅ Restored with error: {error:.6f} rad")
        
        # Show end-effector position
        pos, orn = sim.ee_pose(robot_name)
        print(f"   EE position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        # Wait a bit
        for _ in range(100):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.01)


def demonstrate_gripper_control(sim, robot_name):
    """Demonstrate gripper control during resets."""
    
    robot = sim.robots[robot_name]
    if not robot.gripper_indices:
        print(f"\n⚠️ Robot '{robot_name}' has no gripper")
        return
    
    print(f"\n{'='*60}")
    print("GRIPPER RESET CONTROL")
    print('='*60)
    
    # Reset with gripper
    print("\n🤖 Reset WITH gripper:")
    all_positions = sim.reset_robot(robot_name, reset_gripper=True)
    print(f"   Total joints reset: {len(all_positions)}")
    gripper_info = sim.get_gripper_info(robot_name)
    print(f"   Gripper opening: {gripper_info['current_opening']:.3f}")
    
    # Reset without gripper
    print("\n🦾 Reset WITHOUT gripper (arm only):")
    arm_positions = sim.reset_robot(robot_name, reset_gripper=False)
    print(f"   Arm joints reset: {len(arm_positions)}")
    
    # Custom arm position with gripper control
    print("\n🎯 Custom arm position + manual gripper control:")
    custom_arm_q = [0, 0.5, 0, -1.5, 0, 1.0, 0][:len(arm_positions)]
    sim.reset_robot(robot_name, q_default=custom_arm_q, reset_gripper=False)
    
    # Manually control gripper
    print("   Opening gripper...")
    sim.set_gripper(robot_name, 1.0)  # Open
    for _ in range(50):
        sim.step()
    
    print("   Closing gripper...")
    sim.set_gripper(robot_name, 0.0)  # Close
    for _ in range(50):
        sim.step()


def demonstrate_velocity_reset(sim, robot_name):
    """Demonstrate resetting joint velocities."""
    
    print(f"\n{'='*60}")
    print("VELOCITY RESET")
    print('='*60)
    
    # Set robot in motion using physics
    print("\n🏃 Setting robot in motion...")
    target_q = [0.5, -0.5, 0.5, -1.0, 0.5, 1.0, 0]
    sim.set_joint_positions(robot_name, target_q, use_physics=True)
    
    # Let it move
    for _ in range(50):
        sim.step()
    
    # Get velocities
    robot = sim.robots[robot_name]
    import pybullet as p
    velocities = []
    for idx in robot.joint_indices[:7]:  # First 7 joints
        state = p.getJointState(robot.robot_id, idx)
        velocities.append(state[1])  # Velocity is at index 1
    
    print(f"   Velocities during motion: {[f'{v:.3f}' for v in velocities]}")
    print(f"   Max velocity: {max(abs(v) for v in velocities):.3f} rad/s")
    
    # Reset velocities to zero
    print("\n🛑 Resetting velocities to zero...")
    sim.reset_robot_velocity(robot_name)
    
    # Check velocities after reset
    velocities_after = []
    for idx in robot.joint_indices[:7]:
        state = p.getJointState(robot.robot_id, idx)
        velocities_after.append(state[1])
    
    print(f"   Velocities after reset: {[f'{v:.3f}' for v in velocities_after]}")
    print(f"   Max velocity: {max(abs(v) for v in velocities_after):.3f} rad/s")


def main():
    """Run the robot reset demonstration."""
    
    # Create simulator configuration
    config = SimulationConfig(
        gravity=(0.0, 0.0, -9.81),
        time_step=1.0 / 240.0,
        use_real_time=True,
        camera_distance=1.5,
        camera_yaw=60,
        camera_pitch=-35,
        camera_target=(0.4, 0, 0.2),
    )
    
    # Create simulator
    sim = RobotSimulator(config=config)
    
    try:
        print("🚀 Starting Robot Reset Demonstration...")
        sim.connect()
        
        # Load plane
        sim.load_plane()
        
        # Load KUKA robot
        print("\n🤖 Loading KUKA iiwa robot...")
        kuka = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="kuka",
        )
        
        # Load Panda robot
        print("🤖 Loading Franka Panda robot...")
        panda = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0.8, 0, 0),
            fixed_base=True,
            robot_name="panda",
        )
        
        # Add some visual elements
        platform = sim.spawn_platform(
            color_rgb=(0.7, 0.7, 0.7),
            size=0.3,
            position=(0.4, 0, 0.01),
            height=0.005,
        )
        
        # Demonstrate reset positions for KUKA
        demonstrate_reset_positions(sim, "kuka")
        
        # Demonstrate reset positions for Panda
        demonstrate_reset_positions(sim, "panda")
        
        # Demonstrate configuration memory for KUKA
        demonstrate_configuration_memory(sim, "kuka")
        
        # Demonstrate gripper control for Panda
        demonstrate_gripper_control(sim, "panda")
        
        # Demonstrate velocity reset
        demonstrate_velocity_reset(sim, "kuka")
        
        # Final reset to home positions
        print(f"\n{'='*60}")
        print("FINAL RESET TO HOME")
        print('='*60)
        
        print("\n🏠 Resetting both robots to home positions...")
        sim.reset_robot("kuka")
        sim.reset_robot("panda")
        
        # Show final EE positions
        kuka_pos, _ = sim.ee_pose("kuka")
        panda_pos, _ = sim.ee_pose("panda")
        print(f"   KUKA EE: ({kuka_pos[0]:.3f}, {kuka_pos[1]:.3f}, {kuka_pos[2]:.3f})")
        print(f"   Panda EE: ({panda_pos[0]:.3f}, {panda_pos[1]:.3f}, {panda_pos[2]:.3f})")
        
        # Keep simulation running if in GUI mode
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n▶️  Simulation running (press Ctrl+C to stop)...")
            print("   Try manually moving robots and then pressing R to reset!")
            
            try:
                import pybullet as p
                while True:
                    # Check for key press (R = reset)
                    keys = p.getKeyboardEvents()
                    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                        print("\n🔄 R key pressed - Resetting robots...")
                        sim.reset_robot("kuka")
                        sim.reset_robot("panda")
                        print("   Robots reset to home positions!")
                    
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