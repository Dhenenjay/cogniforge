"""
Example demonstrating robot URDF verification and automatic fallback.

This script shows how to:
1. Verify robot URDF paths and handle missing files
2. Automatically fallback to Panda if KUKA doesn't have 7 DOF
3. Load custom robots with proper validation
4. Handle different PyBullet data installations
"""

import logging
from pathlib import Path
from cogniforge.core import RobotSimulator, RobotType, SimulationMode, SimulationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test_kuka_loading(sim, fallback=True):
    """Test loading KUKA robot with fallback option."""
    
    print(f"\n{'='*60}")
    print(f"TESTING KUKA LOADING (fallback={fallback})")
    print('='*60)
    
    try:
        print("\n🤖 Attempting to load KUKA iiwa...")
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="kuka_test",
            auto_fallback=fallback
        )
        
        print(f"✅ Successfully loaded: {robot_info.robot_type.value}")
        print(f"   Robot ID: {robot_info.robot_id}")
        print(f"   Total joints: {robot_info.num_joints}")
        print(f"   Movable joints: {len(robot_info.joint_indices)}")
        print(f"   End effector index: {robot_info.end_effector_index}")
        print(f"   Gripper indices: {robot_info.gripper_indices}")
        print(f"   Tool link index: {robot_info.tool_link_index}")
        
        # Check if it's actually a 7-DOF robot
        if len(robot_info.joint_indices) >= 7:
            print("   ✅ Has 7+ DOF as expected")
        else:
            print(f"   ⚠️ Only has {len(robot_info.joint_indices)} DOF")
            
        return robot_info
        
    except Exception as e:
        print(f"❌ Failed to load robot: {e}")
        return None


def test_panda_loading(sim):
    """Test loading Franka Panda robot."""
    
    print(f"\n{'='*60}")
    print("TESTING PANDA LOADING")
    print('='*60)
    
    try:
        print("\n🐼 Attempting to load Franka Panda...")
        robot_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0.8, 0, 0),
            fixed_base=True,
            robot_name="panda_test"
        )
        
        print(f"✅ Successfully loaded: {robot_info.robot_type.value}")
        print(f"   Robot ID: {robot_info.robot_id}")
        print(f"   Total joints: {robot_info.num_joints}")
        print(f"   Movable joints: {len(robot_info.joint_indices)}")
        print(f"   End effector index: {robot_info.end_effector_index}")
        print(f"   Gripper indices: {robot_info.gripper_indices}")
        print(f"   Tool link index: {robot_info.tool_link_index}")
        
        # Panda should have 9 movable joints (7 arm + 2 fingers)
        if len(robot_info.joint_indices) >= 9:
            print("   ✅ Has expected joint configuration")
        else:
            print(f"   ⚠️ Unexpected joint count: {len(robot_info.joint_indices)}")
            
        return robot_info
        
    except Exception as e:
        print(f"❌ Failed to load robot: {e}")
        return None


def test_ur5_loading(sim):
    """Test loading UR5 robot."""
    
    print(f"\n{'='*60}")
    print("TESTING UR5 LOADING")
    print('='*60)
    
    try:
        print("\n🦾 Attempting to load UR5...")
        robot_info = sim.load_robot(
            robot_type=RobotType.UR5,
            position=(0, 0.8, 0),
            fixed_base=True,
            robot_name="ur5_test"
        )
        
        print(f"✅ Successfully loaded: {robot_info.robot_type.value}")
        print(f"   Robot ID: {robot_info.robot_id}")
        print(f"   Total joints: {robot_info.num_joints}")
        print(f"   Movable joints: {len(robot_info.joint_indices)}")
        print(f"   End effector index: {robot_info.end_effector_index}")
        print(f"   Gripper indices: {robot_info.gripper_indices}")
        print(f"   Tool link index: {robot_info.tool_link_index}")
        
        # UR5 should have 6 DOF
        if len(robot_info.joint_indices) == 6:
            print("   ✅ Has expected 6 DOF")
        else:
            print(f"   ⚠️ Unexpected DOF count: {len(robot_info.joint_indices)}")
            
        return robot_info
        
    except Exception as e:
        print(f"❌ Failed to load robot: {e}")
        return None


def test_custom_robot(sim, urdf_path):
    """Test loading a custom robot from URDF path."""
    
    print(f"\n{'='*60}")
    print("TESTING CUSTOM ROBOT LOADING")
    print('='*60)
    
    print(f"\n📄 URDF path: {urdf_path}")
    
    try:
        # Check if file exists
        if not Path(urdf_path).exists():
            # Try relative to PyBullet data
            import pybullet_data
            full_path = Path(pybullet_data.getDataPath()) / urdf_path
            if not full_path.exists():
                print(f"❌ URDF file not found: {urdf_path}")
                return None
        
        print("📂 Loading custom robot...")
        robot_info = sim.load_robot(
            robot_type=RobotType.CUSTOM,
            position=(0, -0.8, 0),
            fixed_base=True,
            robot_name="custom_test",
            urdf_path=urdf_path
        )
        
        print(f"✅ Successfully loaded custom robot")
        print(f"   Robot ID: {robot_info.robot_id}")
        print(f"   Total joints: {robot_info.num_joints}")
        print(f"   Movable joints: {len(robot_info.joint_indices)}")
        print(f"   End effector index: {robot_info.end_effector_index}")
        
        return robot_info
        
    except Exception as e:
        print(f"❌ Failed to load custom robot: {e}")
        return None


def test_fallback_scenario(sim):
    """Test automatic fallback from KUKA to Panda."""
    
    print(f"\n{'='*60}")
    print("TESTING AUTOMATIC FALLBACK")
    print('='*60)
    
    print("\n📊 Scenario: KUKA with incorrect DOF")
    print("   Expected: Automatic fallback to Panda")
    
    # First, try without fallback
    print("\n1️⃣ Without fallback (auto_fallback=False):")
    robot1 = test_kuka_loading(sim, fallback=False)
    
    if robot1:
        # Clean up
        import pybullet as p
        p.removeBody(robot1.robot_id)
        del sim.robots["kuka_test"]
    
    # Now try with fallback
    print("\n2️⃣ With fallback (auto_fallback=True):")
    robot2 = test_kuka_loading(sim, fallback=True)
    
    if robot2:
        if robot2.robot_type == RobotType.FRANKA_PANDA:
            print("\n✅ Fallback successful: Loaded Panda instead of KUKA")
        else:
            print("\n⚠️ No fallback occurred: Still loaded KUKA")


def verify_all_robots():
    """Verify all standard robot types can be loaded."""
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE ROBOT VERIFICATION")
    print('='*60)
    
    # Create simulator in DIRECT mode for testing
    config = SimulationConfig(
        time_step=1/240,
        solver_iterations=50,
    )
    
    sim = RobotSimulator(config=config, force_mode=SimulationMode.DIRECT)
    
    try:
        print("\n🚀 Connecting to physics engine...")
        sim.connect()
        sim.load_plane()
        
        # Test each robot type
        robots_loaded = []
        
        # Test KUKA (with fallback)
        kuka_info = test_kuka_loading(sim, fallback=True)
        if kuka_info:
            robots_loaded.append(kuka_info)
        
        # Test Panda
        panda_info = test_panda_loading(sim)
        if panda_info:
            robots_loaded.append(panda_info)
        
        # Test UR5
        ur5_info = test_ur5_loading(sim)
        if ur5_info:
            robots_loaded.append(ur5_info)
        
        # Test custom robot (using a simple URDF)
        custom_info = test_custom_robot(sim, "cube_small.urdf")
        if custom_info:
            robots_loaded.append(custom_info)
        
        # Summary
        print(f"\n{'='*60}")
        print("VERIFICATION SUMMARY")
        print('='*60)
        
        print(f"\n📊 Robots successfully loaded: {len(robots_loaded)}/4")
        for robot in robots_loaded:
            print(f"   ✅ {robot.name}: {robot.robot_type.value} (ID: {robot.robot_id})")
        
        # Test fallback scenario
        test_fallback_scenario(sim)
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔌 Disconnecting...")
        sim.disconnect()


def interactive_robot_selection():
    """Interactive robot selection with GUI mode."""
    
    print(f"\n{'='*60}")
    print("INTERACTIVE ROBOT SELECTION")
    print('='*60)
    
    config = SimulationConfig(
        time_step=1/240,
        camera_distance=2.0,
        camera_yaw=45,
        camera_pitch=-30,
    )
    
    # Try GUI mode if available
    try:
        sim = RobotSimulator(config=config, force_mode=SimulationMode.GUI)
    except:
        print("⚠️ GUI not available, using DIRECT mode")
        sim = RobotSimulator(config=config, force_mode=SimulationMode.DIRECT)
    
    try:
        print("\n🚀 Starting interactive session...")
        sim.connect()
        sim.load_plane()
        
        print("\n🤖 Select robots to load:")
        print("   1. KUKA iiwa (or Panda fallback)")
        print("   2. Franka Panda")
        print("   3. UR5")
        print("   4. All robots")
        print("   5. Exit")
        
        while True:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                test_kuka_loading(sim)
            elif choice == '2':
                test_panda_loading(sim)
            elif choice == '3':
                test_ur5_loading(sim)
            elif choice == '4':
                test_kuka_loading(sim)
                test_panda_loading(sim)
                test_ur5_loading(sim)
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")
            
            if sim._simulation_mode == SimulationMode.GUI:
                print("\n▶️ Simulation running... Press Enter to continue")
                input()
        
    finally:
        print("\n🔌 Disconnecting...")
        sim.disconnect()


def main():
    """Run robot verification demonstrations."""
    
    print("🚀 Starting Robot URDF Verification Demo")
    print("\nThis demo verifies robot URDF paths and demonstrates")
    print("automatic fallback when robots don't meet expectations.\n")
    
    # Run comprehensive verification
    verify_all_robots()
    
    # Optionally run interactive mode
    print(f"\n{'='*60}")
    print("Would you like to try interactive robot selection?")
    response = input("Enter 'y' for yes, any other key to skip: ").strip().lower()
    
    if response == 'y':
        interactive_robot_selection()
    
    print("\n✅ Robot verification demo complete!")
    
    print("\n📝 Key features demonstrated:")
    print("   1. Multiple URDF path attempts for each robot")
    print("   2. Automatic fallback from KUKA to Panda if DOF mismatch")
    print("   3. Custom robot loading with path validation")
    print("   4. Comprehensive robot information display")
    
    print("\n💡 Tips:")
    print("   - Use auto_fallback=True for robust 7-DOF robot loading")
    print("   - Check robot.joint_indices for actual movable joints")
    print("   - Use robot.tool_link_index for end-effector operations")


if __name__ == "__main__":
    main()