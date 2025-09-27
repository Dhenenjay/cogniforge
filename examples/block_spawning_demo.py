"""
Example script demonstrating block spawning in PyBullet simulator.

This script shows how to:
1. Connect to PyBullet
2. Load a plane and robot
3. Spawn colored blocks at specific positions
4. Run the simulation to see blocks falling and interacting
"""

import time
import logging
from cogniforge.core import RobotSimulator, RobotType, SimulationMode, SimulationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    """Run the block spawning demonstration."""
    
    # Create simulator configuration
    config = SimulationConfig(
        gravity=(0.0, 0.0, -9.81),
        time_step=1.0 / 240.0,
        use_real_time=True,  # Real-time simulation for better visualization
        camera_distance=1.5,
        camera_yaw=45,
        camera_pitch=-30,
        camera_target=(0.5, 0, 0),  # Focus on spawn area
    )
    
    # Create simulator (will auto-detect GUI or DIRECT mode)
    sim = RobotSimulator(config=config)
    
    try:
        print("🚀 Starting PyBullet simulator...")
        sim.connect()
        
        # Load plane
        print("📦 Loading plane...")
        plane_id = sim.load_plane()
        
        # Load robot
        print("🤖 Loading KUKA robot...")
        robot = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="kuka",
        )
        
        print(f"✅ Robot loaded: {robot.name} with {robot.num_joints} joints")
        
        # Spawn static platforms
        print("\n🏗️ Spawning platforms...")
        
        # Gray platform
        platform1 = sim.spawn_platform(
            color_rgb=(0.7, 0.7, 0.7),  # Light gray
            size=0.1,
            position=(0.6, 0.2, 0.05),
        )
        print(f"  ⬜ Gray platform spawned (ID: {platform1})")
        
        # Blue platform (smaller and higher)
        platform2 = sim.spawn_platform(
            color_rgb=(0.3, 0.3, 0.8),  # Blue
            size=0.08,
            position=(0.4, -0.2, 0.1),
            platform_name="blue_platform",
            height=0.015,  # Thinner
        )
        print(f"  🟦 Blue platform spawned (ID: {platform2})")
        
        # Spawn a table
        table = sim.spawn_table(
            position=(0.8, 0.0, 0.0),
            table_height=0.35,
            table_size=0.25,
        )
        print(f"  🟫 Table spawned (ID: {table})")
        time.sleep(0.5)
        
        # Spawn blocks with different colors and sizes
        print("\n🎨 Spawning colored blocks...")
        
        # Red block - on the gray platform
        red_block = sim.spawn_block(
            color_rgb=(1.0, 0.0, 0.0),
            size=0.03,
            position=(0.6, 0.2, 0.15),  # Above platform
        )
        print(f"  🔴 Red block spawned on platform (ID: {red_block})")
        time.sleep(0.5)
        
        # Blue block - on the table
        blue_block = sim.spawn_block(
            color_rgb=(0.0, 0.0, 1.0),
            size=0.04,
            position=(0.8, 0.0, 0.45),  # Above table
            block_name="blue_cube",
        )
        print(f"  🔵 Blue block spawned on table (ID: {blue_block})")
        time.sleep(0.5)
        
        # Green block - on the blue platform
        green_block = sim.spawn_block(
            color_rgb=(0.0, 1.0, 0.0),
            size=0.025,
            position=(0.4, -0.2, 0.2),  # Above blue platform
            mass=0.05,
            block_name="green_cube",
        )
        print(f"  🟢 Green block spawned on blue platform (ID: {green_block})")
        time.sleep(0.5)
        
        # Yellow block - dropping from higher
        yellow_block = sim.spawn_block(
            color_rgb=(1.0, 1.0, 0.0),
            size=0.05,
            position=(0.6, 0.2, 0.35),  # High above gray platform
            mass=0.3,
            block_name="yellow_cube",
        )
        print(f"  🟡 Yellow block dropped onto platform (ID: {yellow_block})")
        time.sleep(0.5)
        
        # Purple block - on ground
        purple_block = sim.spawn_block(
            color_rgb=(0.5, 0.0, 1.0),
            size=0.035,
            position=(0.5, 0.0, 0.1),
            block_name="purple_cube",
        )
        print(f"  🟣 Purple block spawned on ground (ID: {purple_block})")
        
        # Spawn random blocks in a circle
        print("\n🎲 Spawning random blocks...")
        random_blocks = sim.spawn_random_blocks(
            num_blocks=5,
            size_range=(0.02, 0.04),
            spawn_height=0.4,
            spawn_radius=0.2,
        )
        print(f"  ✨ Spawned {len(random_blocks)} random blocks")
        
        # Get all object IDs
        all_ids = sim.get_ids()
        print(f"\n📊 Total objects in simulation:")
        print(f"  - Plane: {all_ids['plane_id']}")
        print(f"  - Robots: {len(all_ids['robots'])}")
        print(f"  - Objects: {len(all_ids['objects'])}")
        
        # Run simulation
        if sim._simulation_mode == SimulationMode.GUI:
            print("\n▶️  Running simulation (press Ctrl+C to stop)...")
            print("   Watch the blocks fall and stack!")
            
            try:
                step_count = 0
                while True:
                    sim.step()
                    step_count += 1
                    
                    # Print object states every 240 steps (1 second at 240Hz)
                    if step_count % 240 == 0:
                        # Get state of one block
                        if "blue_cube" in sim.objects:
                            state = sim.get_object_state("blue_cube")
                            pos = state["position"]
                            print(f"   Blue cube position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                    
                    time.sleep(sim.config.time_step)
                    
            except KeyboardInterrupt:
                print("\n⏹️  Simulation stopped by user")
        else:
            print("\n▶️  Running headless simulation for 1000 steps...")
            for i in range(1000):
                sim.step()
                if i % 100 == 0:
                    print(f"   Step {i}/1000")
            
            # Get final positions
            print("\n📍 Final block positions:")
            for name in ["blue_cube", "green_cube", "yellow_cube"]:
                if name in sim.objects:
                    state = sim.get_object_state(name)
                    pos = state["position"]
                    print(f"  - {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        print("\n🔌 Disconnecting from PyBullet...")
        sim.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main()