"""
Example demonstrating deterministic simulations with seeding.

This script shows how to:
1. Set random seeds for reproducible simulations
2. Use deterministic physics engine parameters
3. Compare results between seeded and unseeded runs
4. Demonstrate reproducibility across multiple runs
"""

import time
import numpy as np
import random
import logging
from cogniforge.core import RobotSimulator, RobotType, SimulationMode, SimulationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def run_simulation_with_seed(seed, deterministic=False, verbose=True):
    """Run a simulation with specified seed and return results."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATION WITH SEED: {seed}")
        if deterministic:
            print("(Deterministic mode enabled)")
        print('='*60)
    
    # Create configuration with seed
    config = SimulationConfig(
        seed=seed,
        deterministic=deterministic,
        time_step=1/240,
        solver_iterations=50,
        use_real_time=False,  # Don't use real-time for reproducibility
    )
    
    # Create simulator
    sim = RobotSimulator(config=config, force_mode=SimulationMode.DIRECT)
    sim.connect()
    
    # Load plane
    sim.load_plane()
    
    # Load robot with some randomness in initial position
    # The randomness should be controlled by the seed
    robot_x = 0.0 + np.random.uniform(-0.01, 0.01)
    robot_y = 0.0 + np.random.uniform(-0.01, 0.01)
    
    robot_info = sim.load_robot(
        robot_type=RobotType.KUKA_IIWA,
        position=(robot_x, robot_y, 0),
        fixed_base=True,
        robot_name="kuka",
    )
    
    # Spawn random blocks
    blocks = []
    block_positions = []
    for i in range(3):
        # Random position
        x = 0.4 + np.random.uniform(-0.1, 0.1)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.05 + i * 0.1  # Stack them
        
        # Random color
        color = (
            np.random.rand(),
            np.random.rand(),
            np.random.rand()
        )
        
        block_id = sim.spawn_block(
            color_rgb=color,
            size=0.04,
            position=(x, y, z),
            mass=0.05,
            block_name=f"block_{i}"
        )
        blocks.append(block_id)
        block_positions.append((x, y, z))
    
    # Run simulation for a fixed number of steps
    n_steps = 100
    for step in range(n_steps):
        # Apply some random forces occasionally
        if step % 20 == 0 and blocks:
            block_idx = random.randint(0, len(blocks) - 1)
            force = [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(0, 1.0)
            ]
            import pybullet as p
            p.applyExternalForce(
                blocks[block_idx],
                -1,  # Center of mass
                force,
                [0, 0, 0],
                p.WORLD_FRAME
            )
        
        sim.step()
    
    # Get final block positions
    final_positions = []
    for block_id in blocks:
        import pybullet as p
        pos, _ = p.getBasePositionAndOrientation(block_id)
        final_positions.append(pos)
    
    # Get robot joint positions after random movements
    joint_positions = sim.get_joint_positions("kuka")
    
    # Disconnect
    sim.disconnect()
    
    results = {
        'robot_initial': (robot_x, robot_y),
        'block_initial': block_positions,
        'block_final': final_positions,
        'joint_positions': joint_positions,
    }
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"   Robot initial: ({robot_x:.4f}, {robot_y:.4f})")
        print(f"   Final block positions:")
        for i, pos in enumerate(final_positions):
            print(f"      Block {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    return results


def compare_results(results1, results2, tolerance=1e-6):
    """Compare two simulation results for equality."""
    
    print(f"\n🔍 Comparing results (tolerance={tolerance}):")
    
    # Compare robot initial positions
    robot_diff = np.abs(np.array(results1['robot_initial']) - 
                        np.array(results2['robot_initial']))
    print(f"   Robot initial position diff: {np.max(robot_diff):.8f}")
    
    # Compare block final positions
    max_block_diff = 0
    for i, (pos1, pos2) in enumerate(zip(results1['block_final'], 
                                         results2['block_final'])):
        diff = np.linalg.norm(np.array(pos1) - np.array(pos2))
        max_block_diff = max(max_block_diff, diff)
        print(f"   Block {i} position diff: {diff:.8f}")
    
    # Overall match
    match = (np.max(robot_diff) < tolerance and max_block_diff < tolerance)
    
    if match:
        print("✅ Results match within tolerance!")
    else:
        print("❌ Results differ beyond tolerance!")
    
    return match


def demo_reproducibility():
    """Demonstrate reproducibility with same seed."""
    
    print(f"\n{'='*60}")
    print("REPRODUCIBILITY TEST")
    print('='*60)
    print("\nRunning two simulations with same seed (42)...")
    
    # Run first simulation
    results1 = run_simulation_with_seed(42, deterministic=True, verbose=False)
    print("   First run complete")
    
    # Run second simulation with same seed
    results2 = run_simulation_with_seed(42, deterministic=True, verbose=False)
    print("   Second run complete")
    
    # Compare results
    compare_results(results1, results2)


def demo_different_seeds():
    """Demonstrate different results with different seeds."""
    
    print(f"\n{'='*60}")
    print("DIFFERENT SEEDS TEST")
    print('='*60)
    print("\nRunning simulations with different seeds...")
    
    seeds = [42, 123, 999]
    results = []
    
    for seed in seeds:
        result = run_simulation_with_seed(seed, deterministic=True, verbose=False)
        results.append(result)
        print(f"   Seed {seed} complete")
    
    print("\n🎲 Comparing results across different seeds:")
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            print(f"\n   Seed {seeds[i]} vs Seed {seeds[j]}:")
            match = compare_results(results[i], results[j], tolerance=0.01)
            if match:
                print("      (Unexpectedly similar!)")
            else:
                print("      (Different as expected)")


def demo_deterministic_vs_standard():
    """Compare deterministic mode vs standard physics."""
    
    print(f"\n{'='*60}")
    print("DETERMINISTIC VS STANDARD PHYSICS")
    print('='*60)
    
    seed = 42
    
    # Run with deterministic physics
    print("\n1️⃣ Deterministic physics (fixed timestep, etc.):")
    det_results = []
    for i in range(2):
        result = run_simulation_with_seed(seed, deterministic=True, verbose=False)
        det_results.append(result)
        print(f"   Run {i+1} complete")
    
    print("\n   Comparing deterministic runs:")
    det_match = compare_results(det_results[0], det_results[1], tolerance=1e-10)
    
    # Run with standard physics
    print("\n2️⃣ Standard physics:")
    std_results = []
    for i in range(2):
        result = run_simulation_with_seed(seed, deterministic=False, verbose=False)
        std_results.append(result)
        print(f"   Run {i+1} complete")
    
    print("\n   Comparing standard runs:")
    std_match = compare_results(std_results[0], std_results[1], tolerance=1e-6)
    
    print("\n📊 Summary:")
    print(f"   Deterministic mode reproducibility: {'✅ Perfect' if det_match else '❌ Not perfect'}")
    print(f"   Standard mode reproducibility: {'✅ Good' if std_match else '⚠️ May vary'}")


def demo_random_sampling():
    """Demonstrate controlled random sampling with seeds."""
    
    print(f"\n{'='*60}")
    print("CONTROLLED RANDOM SAMPLING")
    print('='*60)
    
    # Create simulator with seed
    sim = RobotSimulator(seed=42)
    
    print("\n🎲 Random samples with seed 42:")
    print("   NumPy samples:", [np.random.rand() for _ in range(3)])
    print("   Python samples:", [random.random() for _ in range(3)])
    
    # Reset seed
    sim.set_seed(42)
    
    print("\n🔄 After resetting to seed 42:")
    print("   NumPy samples:", [np.random.rand() for _ in range(3)])
    print("   Python samples:", [random.random() for _ in range(3)])
    
    # Reset to time-based seed
    sim.reset_seeds()
    
    print(f"\n⏰ After time-based reset (seed={sim.config.seed}):")
    print("   NumPy samples:", [np.random.rand() for _ in range(3)])
    print("   Python samples:", [random.random() for _ in range(3)])


def demo_waypoint_reproducibility():
    """Demonstrate reproducible waypoint-based motion."""
    
    print(f"\n{'='*60}")
    print("WAYPOINT MOTION REPRODUCIBILITY")
    print('='*60)
    
    def run_waypoint_sim(seed):
        """Run waypoint simulation with given seed."""
        config = SimulationConfig(
            seed=seed,
            deterministic=True,
            time_step=1/240,
            use_real_time=False,
        )
        
        sim = RobotSimulator(config=config, force_mode=SimulationMode.DIRECT)
        sim.connect()
        sim.load_plane()
        
        # Load robot
        sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="panda",
        )
        
        # Generate random waypoints
        waypoints = []
        for _ in range(5):
            waypoint = (
                0.4 + np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.2, 0.2),
                0.3 + np.random.uniform(-0.1, 0.1)
            )
            waypoints.append(waypoint)
        
        # Execute waypoint motion
        trajectory = sim.move_through_waypoints(
            "panda",
            waypoints,
            steps_per_segment=20,
            action_repeat=2,
            return_trajectories=True
        )
        
        sim.disconnect()
        
        return waypoints, trajectory
    
    # Run twice with same seed
    print("\n🤖 Running waypoint motion with seed 123...")
    waypoints1, traj1 = run_waypoint_sim(123)
    print("   First run complete")
    
    waypoints2, traj2 = run_waypoint_sim(123)
    print("   Second run complete")
    
    # Compare waypoints
    print("\n📍 Comparing generated waypoints:")
    for i, (wp1, wp2) in enumerate(zip(waypoints1, waypoints2)):
        diff = np.linalg.norm(np.array(wp1) - np.array(wp2))
        print(f"   Waypoint {i} diff: {diff:.8f}")
    
    # Compare trajectories
    if traj1 and traj2:
        print("\n📈 Comparing trajectory errors:")
        if traj1['ee_errors'] and traj2['ee_errors']:
            for i, (err1, err2) in enumerate(zip(traj1['ee_errors'], 
                                                 traj2['ee_errors'])):
                print(f"   Waypoint {i} error diff: {abs(err1 - err2):.8f}")


def main():
    """Run all deterministic simulation demonstrations."""
    
    print("🚀 Starting Deterministic Simulation Demonstration...")
    print("\nThis demo shows how to achieve reproducible simulations")
    print("using random seeds and deterministic physics settings.")
    
    try:
        # 1. Basic reproducibility
        demo_reproducibility()
        
        # 2. Different seeds
        demo_different_seeds()
        
        # 3. Deterministic vs standard physics
        demo_deterministic_vs_standard()
        
        # 4. Random sampling
        demo_random_sampling()
        
        # 5. Waypoint motion reproducibility
        demo_waypoint_reproducibility()
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        
        print("\n✅ Key takeaways:")
        print("   1. Use seed parameter for reproducible random numbers")
        print("   2. Enable deterministic=True for exact physics reproducibility")
        print("   3. Same seed → same results (with deterministic mode)")
        print("   4. Different seeds → different (but reproducible) results")
        print("   5. Useful for debugging, testing, and experiments")
        
        print("\n📝 Example usage:")
        print("   # For reproducible simulations:")
        print("   config = SimulationConfig(seed=42, deterministic=True)")
        print("   sim = RobotSimulator(config=config)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Deterministic simulation demo complete!")


if __name__ == "__main__":
    main()