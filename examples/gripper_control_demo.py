"""
Example demonstrating gripper control using open_gripper() and close_gripper() methods.

This script shows how to:
1. Use open_gripper() for fully opening the gripper
2. Use close_gripper() for fully closing the gripper
3. Use custom forces for delicate grasping
4. Coordinate gripper control with arm movements
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


def gripper_basic_control(sim, robot_name):
    """Demonstrate basic open and close gripper functions."""
    
    print(f"\n{'='*60}")
    print(f"BASIC GRIPPER CONTROL - {robot_name}")
    print('='*60)
    
    # Open gripper fully
    print("\n🖐️ Opening gripper fully...")
    sim.open_gripper(robot_name)
    
    # Let the gripper move
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Get gripper info
    gripper_info = sim.get_gripper_info(robot_name)
    print(f"   Gripper opening: {gripper_info['current_opening']:.3f}")
    
    # Close gripper fully
    print("\n✊ Closing gripper fully...")
    sim.close_gripper(robot_name)
    
    # Let the gripper move
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Get gripper info
    gripper_info = sim.get_gripper_info(robot_name)
    print(f"   Gripper opening: {gripper_info['current_opening']:.3f}")


def gripper_force_control(sim, robot_name):
    """Demonstrate gripper control with different forces."""
    
    print(f"\n{'='*60}")
    print("GRIPPER FORCE CONTROL")
    print('='*60)
    
    # Strong force for firm grasp
    print("\n💪 Opening with strong force (150N)...")
    sim.open_gripper(robot_name, force=150.0)
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Gentle closing for delicate objects
    print("\n🌸 Closing gently (20N) for delicate objects...")
    sim.close_gripper(robot_name, force=20.0)
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Medium force for normal grasping
    print("\n📦 Closing with medium force (50N) for normal grasping...")
    sim.open_gripper(robot_name)  # First open
    for _ in range(30):
        sim.step()
    
    sim.close_gripper(robot_name, force=50.0)
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)


def pick_and_place_sequence(sim, robot_name):
    """Demonstrate a pick and place sequence with gripper control."""
    
    print(f"\n{'='*60}")
    print("PICK AND PLACE SEQUENCE")
    print('='*60)
    
    # Starting position - home
    print("\n1️⃣ Moving to home position...")
    sim.reset_robot(robot_name)
    sim.open_gripper(robot_name)  # Start with open gripper
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Move to approach position
    print("\n2️⃣ Moving to approach position...")
    if robot_name == "panda":
        approach_q = [0, -0.3, 0, -2.0, 0, 1.8, 0.785]
    else:  # KUKA
        approach_q = [0, 0.7, 0, -1.0, 0, 0.5, 0]
    
    sim.reset_robot(robot_name, q_default=approach_q, reset_gripper=False)
    
    for _ in range(100):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Get end-effector position
    pos, orn = sim.ee_pose(robot_name)
    print(f"   EE at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    # Close gripper to grasp
    print("\n3️⃣ Closing gripper to grasp object...")
    sim.close_gripper(robot_name, force=40.0)  # Moderate force for secure grip
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Lift object
    print("\n4️⃣ Lifting object (moving to transport position)...")
    if robot_name == "panda":
        lift_q = [0, 0, 0, -1.57, 0, 1.57, 0.785]
    else:  # KUKA
        lift_q = [0, 0.3, 0, -0.5, 0, 0.3, 0]
    
    sim.reset_robot(robot_name, q_default=lift_q, reset_gripper=False)
    
    for _ in range(100):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Move to place position
    print("\n5️⃣ Moving to place position...")
    if robot_name == "panda":
        place_q = [0.5, -0.3, 0, -2.0, 0, 1.8, 0.785]
    else:  # KUKA
        place_q = [0.5, 0.7, 0, -1.0, 0, 0.5, 0]
    
    sim.reset_robot(robot_name, q_default=place_q, reset_gripper=False)
    
    for _ in range(100):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Release object
    print("\n6️⃣ Opening gripper to release object...")
    sim.open_gripper(robot_name)
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Return to home
    print("\n7️⃣ Returning to home position...")
    sim.reset_robot(robot_name)
    
    for _ in range(100):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    print("\n✅ Pick and place sequence complete!")


def gripper_cycle_test(sim, robot_name, cycles=3):
    """Test gripper cycling open and close."""
    
    print(f"\n{'='*60}")
    print(f"GRIPPER CYCLE TEST ({cycles} cycles)")
    print('='*60)
    
    for cycle in range(cycles):
        print(f"\n🔄 Cycle {cycle + 1}/{cycles}")
        
        # Open
        print("   Opening...")
        sim.open_gripper(robot_name, force=80.0)
        for _ in range(30):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.005)
        
        # Get state
        gripper_info = sim.get_gripper_info(robot_name)
        print(f"   Open state: {gripper_info['current_opening']:.3f}")
        
        # Close
        print("   Closing...")
        sim.close_gripper(robot_name, force=40.0)
        for _ in range(30):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.005)
        
        # Get state
        gripper_info = sim.get_gripper_info(robot_name)
        print(f"   Closed state: {gripper_info['current_opening']:.3f}")


def test_grasping_with_blocks(sim, robot_name):
    """Test grasping blocks with the gripper."""
    
    print(f"\n{'='*60}")
    print("BLOCK GRASPING TEST")
    print('='*60)
    
    # Spawn blocks to grasp
    print("\n📦 Spawning blocks...")
    
    # Position blocks based on robot location
    if robot_name == "panda":
        block_x = 0.8 + 0.4  # Panda is at x=0.8
    else:  # KUKA at origin
        block_x = 0.4
    
    # Spawn three blocks
    red_block = sim.spawn_block(
        color_rgb=(1.0, 0.0, 0.0),
        size=0.04,
        position=(block_x, -0.1, 0.02),
        mass=0.05,
        block_name=f"{robot_name}_red_block"
    )
    
    green_block = sim.spawn_block(
        color_rgb=(0.0, 1.0, 0.0),
        size=0.04,
        position=(block_x, 0.0, 0.02),
        mass=0.05,
        block_name=f"{robot_name}_green_block"
    )
    
    blue_block = sim.spawn_block(
        color_rgb=(0.0, 0.0, 1.0),
        size=0.04,
        position=(block_x, 0.1, 0.02),
        mass=0.05,
        block_name=f"{robot_name}_blue_block"
    )
    
    # Let blocks settle
    for _ in range(50):
        sim.step()
    
    # Grasp sequence for middle (green) block
    print("\n🟢 Attempting to grasp green block...")
    
    # Open gripper
    print("   Opening gripper...")
    sim.open_gripper(robot_name)
    
    for _ in range(30):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Move to grasp position
    print("   Moving to grasp position...")
    if robot_name == "panda":
        grasp_q = [0, -0.5, 0, -2.2, 0, 1.7, 0.785]
    else:  # KUKA
        grasp_q = [0, 0.8, 0, -1.2, 0, 0.4, 0]
    
    sim.reset_robot(robot_name, q_default=grasp_q, reset_gripper=False)
    
    for _ in range(100):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Close gripper
    print("   Closing gripper on block...")
    sim.close_gripper(robot_name, force=35.0)  # Gentle but firm
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Lift block
    print("   Lifting block...")
    if robot_name == "panda":
        lift_q = [0, -0.2, 0, -1.8, 0, 1.5, 0.785]
    else:  # KUKA
        lift_q = [0, 0.4, 0, -0.8, 0, 0.4, 0]
    
    sim.reset_robot(robot_name, q_default=lift_q, reset_gripper=False)
    
    for _ in range(100):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    print("   ✅ Block grasped and lifted!")
    
    # Release
    print("   Releasing block...")
    sim.open_gripper(robot_name)
    
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)


def main():
    """Run the gripper control demonstration."""
    
    # Create simulator configuration
    config = SimulationConfig(
        gravity=(0.0, 0.0, -9.81),
        time_step=1.0 / 240.0,
        use_real_time=True,
        camera_distance=1.8,
        camera_yaw=45,
        camera_pitch=-30,
        camera_target=(0.6, 0, 0.1),
    )
    
    # Create simulator
    sim = RobotSimulator(config=config)
    
    try:
        print("🚀 Starting Gripper Control Demonstration...")
        sim.connect()
        
        # Load plane
        sim.load_plane()
        
        # Load Panda robot (has gripper)
        print("\n🤖 Loading Franka Panda robot...")
        panda = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0.8, 0, 0),
            fixed_base=True,
            robot_name="panda",
        )
        
        # Load KUKA robot
        print("🤖 Loading KUKA iiwa robot...")
        kuka = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="kuka",
        )
        
        # Add a platform for visual reference
        platform = sim.spawn_platform(
            color_rgb=(0.8, 0.8, 0.8),
            size=0.4,
            position=(0.4, 0, 0.005),
            height=0.005,
        )
        
        # Run demonstrations
        
        # 1. Basic control
        gripper_basic_control(sim, "panda")
        
        # 2. Force control
        gripper_force_control(sim, "panda")
        
        # 3. Pick and place sequence
        pick_and_place_sequence(sim, "panda")
        
        # 4. Cycle test
        gripper_cycle_test(sim, "panda", cycles=2)
        
        # 5. Block grasping
        test_grasping_with_blocks(sim, "panda")
        
        # Reset robots to home
        print(f"\n{'='*60}")
        print("FINAL RESET")
        print('='*60)
        
        print("\n🏠 Resetting robots to home positions...")
        sim.reset_robot("panda")
        sim.reset_robot("kuka")
        sim.open_gripper("panda")  # Open gripper at end
        
        for _ in range(50):
            sim.step()
        
        # Keep simulation running if in GUI mode
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n▶️  Simulation running (press Ctrl+C to stop)...")
            print("   Keyboard controls:")
            print("   - O: Open gripper")
            print("   - C: Close gripper")
            print("   - G: Gentle close (20N)")
            print("   - F: Firm close (80N)")
            print("   - R: Reset robot to home")
            
            try:
                import pybullet as p
                while True:
                    # Check for key presses
                    keys = p.getKeyboardEvents()
                    
                    if ord('o') in keys and keys[ord('o')] & p.KEY_WAS_TRIGGERED:
                        print("\n🖐️ Opening gripper...")
                        sim.open_gripper("panda")
                    
                    elif ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                        print("\n✊ Closing gripper...")
                        sim.close_gripper("panda")
                    
                    elif ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
                        print("\n🌸 Gentle close (20N)...")
                        sim.close_gripper("panda", force=20.0)
                    
                    elif ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
                        print("\n💪 Firm close (80N)...")
                        sim.close_gripper("panda", force=80.0)
                    
                    elif ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                        print("\n🔄 Resetting robot...")
                        sim.reset_robot("panda")
                    
                    sim.step()
                    time.sleep(1/240)
                    
            except KeyboardInterrupt:
                print("\n⏹️  Simulation stopped")
        else:
            # Run for a bit in headless mode
            print("\n▶️  Running headless simulation for 300 steps...")
            for i in range(300):
                sim.step()
                if i % 100 == 0:
                    print(f"   Step {i}/300")
    
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