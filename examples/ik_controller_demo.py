"""
Demonstration of IK controller with joint limit clamping and fallback to pre-grasp.

This example shows:
1. Setting up IK controller with joint limits
2. Setting pre-grasp pose for fallback
3. Attempting IK with automatic clamping
4. Fallback to pre-grasp on IK failure
5. Safe trajectory generation through waypoints
"""

import time
import numpy as np
import logging
from cogniforge.core import RobotSimulator, RobotType, SimulationMode, SimulationConfig
from cogniforge.control.ik_controller import IKController, IKStatus, create_ik_controller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def demo_joint_limit_clamping(sim, robot_name, ik_controller):
    """Demonstrate joint limit clamping during IK."""
    
    print(f"\n{'='*60}")
    print("JOINT LIMIT CLAMPING DEMO")
    print('='*60)
    
    # Get current joint positions
    current_joints = sim.get_joint_positions(robot_name, include_gripper=False)
    print(f"\nCurrent joint positions: {[f'{q:.3f}' for q in current_joints]}")
    
    # Try to reach a position that would violate joint limits
    extreme_targets = [
        {
            "name": "Far reach (needs clamping)",
            "pos": (0.8, 0.4, 0.6),  # Very far position
        },
        {
            "name": "Low position",
            "pos": (0.4, 0.0, 0.1),  # Very low
        },
        {
            "name": "Behind robot",
            "pos": (-0.2, 0.0, 0.4),  # Behind base
        },
    ]
    
    for target in extreme_targets:
        print(f"\n🎯 Target: {target['name']}")
        print(f"   Position: ({target['pos'][0]:.3f}, {target['pos'][1]:.3f}, {target['pos'][2]:.3f})")
        
        # Compute IK with automatic limit clamping
        joint_solution, status = ik_controller.compute_ik_with_limits(
            target_pos=target["pos"],
            current_joints=current_joints,
            simulator=sim,
            robot_name=robot_name
        )
        
        print(f"   Status: {status.value}")
        print(f"   Solution: {[f'{q:.3f}' for q in joint_solution[:7]]}")
        
        # Check if any joints were clamped
        if ik_controller.joint_limits.is_within_limits(joint_solution):
            print("   ✅ All joints within limits")
        else:
            print("   ⚠️ Some joints were clamped to limits")
        
        # Apply solution
        sim.reset_robot(robot_name, q_default=joint_solution, reset_gripper=False)
        
        # Let simulation settle
        for _ in range(50):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.01)
        
        # Check achieved position
        achieved_pos, _ = sim.ee_pose(robot_name)
        error = np.linalg.norm(np.array(achieved_pos) - np.array(target["pos"]))
        print(f"   Achieved: ({achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f})")
        print(f"   Error: {error:.4f}m")


def demo_pre_grasp_fallback(sim, robot_name, ik_controller):
    """Demonstrate fallback to pre-grasp when IK fails."""
    
    print(f"\n{'='*60}")
    print("PRE-GRASP FALLBACK DEMO")
    print('='*60)
    
    # Move to a good pre-grasp position
    pre_grasp_pos = (0.5, 0.0, 0.5)
    print(f"\n1. Setting pre-grasp position: {pre_grasp_pos}")
    
    # Compute IK for pre-grasp
    current_joints = sim.get_joint_positions(robot_name, include_gripper=False)
    pre_grasp_joints = sim.calculate_ik(robot_name, target_pos=pre_grasp_pos)
    
    # Apply pre-grasp configuration
    sim.reset_robot(robot_name, q_default=pre_grasp_joints, reset_gripper=False)
    
    # Store as fallback position
    ik_controller.set_pre_grasp_pose(
        joint_positions=pre_grasp_joints,
        ee_position=pre_grasp_pos
    )
    print("   ✅ Pre-grasp pose stored for fallback")
    
    # Let robot settle
    for _ in range(50):
        sim.step()
        if sim._simulation_mode == SimulationMode.GUI:
            time.sleep(0.01)
    
    # Try to reach unreachable positions that will trigger fallback
    unreachable_targets = [
        {
            "name": "Too far away",
            "pos": (1.5, 0.0, 0.5),  # Way outside workspace
        },
        {
            "name": "Below ground",
            "pos": (0.5, 0.0, -0.2),  # Below table
        },
        {
            "name": "Impossible orientation",
            "pos": (0.5, 0.0, 0.3),
            "orn": (1.0, 0.0, 0.0, 0.0),  # Invalid quaternion
        },
    ]
    
    for target in unreachable_targets:
        print(f"\n2. Attempting unreachable target: {target['name']}")
        print(f"   Position: {target['pos']}")
        
        # This should trigger fallback
        joint_solution, status = ik_controller.compute_ik_with_limits(
            target_pos=target["pos"],
            target_orn=target.get("orn"),
            current_joints=current_joints,
            simulator=sim,
            robot_name=robot_name
        )
        
        print(f"   Status: {status.value}")
        
        if status == IKStatus.FALLBACK_USED:
            print("   🔄 Fallback to pre-grasp position used")
        elif status == IKStatus.FAILED:
            print("   ❌ IK failed (no fallback available)")
        else:
            print(f"   ✅ Surprisingly succeeded with status: {status.value}")
        
        # Apply solution (will be pre-grasp if fallback was used)
        sim.reset_robot(robot_name, q_default=joint_solution, reset_gripper=False)
        
        # Let simulation settle
        for _ in range(30):
            sim.step()
            if sim._simulation_mode == SimulationMode.GUI:
                time.sleep(0.01)


def demo_safe_trajectory(sim, robot_name, ik_controller):
    """Demonstrate safe trajectory generation with limit checking."""
    
    print(f"\n{'='*60}")
    print("SAFE TRAJECTORY GENERATION")
    print('='*60)
    
    # Define waypoints for pick-and-place task
    waypoints = [
        (0.5, 0.0, 0.5),   # Above object
        (0.5, 0.0, 0.3),   # Approach object
        (0.5, 0.0, 0.25),  # Grasp height
        (0.5, 0.0, 0.4),   # Lift
        (0.3, 0.3, 0.4),   # Move to side
        (0.3, 0.3, 0.25),  # Place position
    ]
    
    print(f"\n📍 Generating trajectory through {len(waypoints)} waypoints")
    
    # Get current joint configuration
    current_joints = sim.get_joint_positions(robot_name, include_gripper=False)
    
    # Generate safe trajectory
    joint_trajectory, status_list = ik_controller.generate_safe_trajectory(
        waypoints=waypoints,
        current_joints=current_joints,
        simulator=sim,
        robot_name=robot_name
    )
    
    # Print trajectory status
    print("\n📊 Trajectory Generation Results:")
    for i, (wp, status) in enumerate(zip(waypoints, status_list)):
        symbol = "✅" if status == IKStatus.SUCCESS else "⚠️"
        print(f"   {symbol} Waypoint {i+1}: {status.value} at {wp}")
    
    # Execute trajectory if at least partially successful
    if any(s != IKStatus.FAILED for s in status_list):
        print("\n▶️ Executing trajectory...")
        
        for i, joint_config in enumerate(joint_trajectory):
            print(f"   Moving to waypoint {i+1}/{len(joint_trajectory)}")
            
            # Move to configuration
            sim.reset_robot(robot_name, q_default=joint_config, reset_gripper=False)
            
            # Simulate smooth motion
            for _ in range(30):
                sim.step()
                if sim._simulation_mode == SimulationMode.GUI:
                    time.sleep(0.01)
            
            # Check position
            pos, _ = sim.ee_pose(robot_name)
            print(f"      Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    else:
        print("\n❌ Trajectory generation failed completely")


def demo_ik_statistics(ik_controller):
    """Display IK controller statistics."""
    
    print(f"\n{'='*60}")
    print("IK CONTROLLER STATISTICS")
    print('='*60)
    
    stats = ik_controller.get_statistics()
    
    print("\n📈 Performance Metrics:")
    print(f"   Total IK attempts:    {stats['total_attempts']}")
    print(f"   Successful:           {stats['successes']}")
    print(f"   Failed:               {stats['failures']}")
    print(f"   Fallbacks used:       {stats['fallbacks']}")
    print(f"   Joint clamps:         {stats['clamps']}")
    print(f"\n   Success rate:         {stats['success_rate']:.1f}%")
    print(f"   Failure rate:         {stats['failure_rate']:.1f}%")
    print(f"   Fallback rate:        {stats['fallback_rate']:.1f}%")


def demo_iteration_limits(sim, robot_name, ik_controller):
    """Demonstrate the effect of IK_MAX_ITERS."""
    
    print(f"\n{'='*60}")
    print("IK ITERATION LIMIT DEMO")
    print('='*60)
    
    target_pos = (0.6, 0.2, 0.4)
    current_joints = sim.get_joint_positions(robot_name, include_gripper=False)
    
    # Test with different iteration limits
    iteration_tests = [10, 50, 100, 150, 200]
    
    print(f"\nTarget position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    print("\nTesting different IK_MAX_ITERS values:")
    
    for max_iters in iteration_tests:
        # Temporarily change max iterations
        original_max_iters = ik_controller.max_iterations
        ik_controller.max_iterations = max_iters
        
        print(f"\n   IK_MAX_ITERS = {max_iters}:")
        
        # Compute IK
        start_time = time.time()
        joint_solution, status = ik_controller.compute_ik_with_limits(
            target_pos=target_pos,
            current_joints=current_joints,
            simulator=sim,
            robot_name=robot_name
        )
        compute_time = time.time() - start_time
        
        # Apply solution to check accuracy
        sim.reset_robot(robot_name, q_default=joint_solution, reset_gripper=False)
        achieved_pos, _ = sim.ee_pose(robot_name)
        error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
        
        print(f"      Status: {status.value}")
        print(f"      Error: {error:.5f}m")
        print(f"      Time: {compute_time:.3f}s")
        
        # Restore original max iterations
        ik_controller.max_iterations = original_max_iters


def main():
    """Run the IK controller demonstration."""
    
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
        print("🚀 Starting IK Controller Demonstration...")
        print(f"   IK_MAX_ITERS = {150}")
        print(f"   IK_CONVERGENCE_THRESHOLD = {0.001}m")
        print(f"   Fallback enabled = True")
        
        sim.connect()
        
        # Load plane and table
        sim.load_plane()
        table = sim.spawn_platform(
            color_rgb=(0.6, 0.4, 0.2),
            size=0.4,
            position=(0.5, 0, 0.2),
            height=0.02
        )
        
        # Load robot
        print("\n🤖 Loading Franka Panda robot...")
        robot = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="panda"
        )
        
        # Create IK controller with limits and fallback
        ik_controller = create_ik_controller(
            robot_type="franka_panda",
            enable_fallback=True
        )
        
        # Add visual markers for targets
        sim.spawn_block(
            color_rgb=(1.0, 0.0, 0.0),
            size=0.02,
            position=(0.5, 0.0, 0.25),
            mass=0.05,
            block_name="target_object"
        )
        
        sim.spawn_block(
            color_rgb=(0.0, 1.0, 0.0),
            size=0.03,
            position=(0.3, 0.3, 0.22),
            mass=0,
            block_name="place_marker"
        )
        
        # Run demonstrations
        
        # 1. Joint limit clamping
        demo_joint_limit_clamping(sim, "panda", ik_controller)
        
        # 2. Pre-grasp fallback
        demo_pre_grasp_fallback(sim, "panda", ik_controller)
        
        # 3. Safe trajectory generation
        demo_safe_trajectory(sim, "panda", ik_controller)
        
        # 4. Iteration limits
        demo_iteration_limits(sim, "panda", ik_controller)
        
        # 5. Show statistics
        demo_ik_statistics(ik_controller)
        
        # Reset to home
        print(f"\n{'='*60}")
        print("FINAL RESET")
        print('='*60)
        
        print("\n🏠 Resetting to home position...")
        sim.reset_robot("panda")
        
        # Keep simulation running if in GUI mode
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n▶️ Simulation running (press Ctrl+C to stop)...")
            
            try:
                while True:
                    sim.step()
                    time.sleep(1/240)
                    
            except KeyboardInterrupt:
                print("\n⏹️ Simulation stopped")
    
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