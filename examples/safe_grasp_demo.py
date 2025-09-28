"""
Demonstration of safe grasp execution with contact queries and vertical lift.

This example shows:
1. Using contact queries to avoid table penetration
2. Automatic vertical lift before lateral movements
3. Safe approach and retreat strategies
4. Contact state monitoring throughout execution
"""

import time
import numpy as np
import logging
import pybullet as p
import pybullet_data
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from cogniforge.control.safe_grasp_execution import (
    SafeGraspExecutor,
    SafeGraspConfig,
    create_safe_grasp_executor,
    ContactState,
    VERTICAL_LIFT_HEIGHT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_simulation():
    """Setup PyBullet simulation environment."""
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set gravity
    p.setGravity(0, 0, -9.81)
    
    # Load plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load table (simple box for demonstration)
    table_height = 0.4
    table_size = [0.8, 0.6, 0.02]  # Table surface dimensions
    
    # Create table visual and collision shapes
    table_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[s/2 for s in table_size],
        rgbaColor=[0.6, 0.4, 0.2, 1]  # Brown color
    )
    
    table_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[s/2 for s in table_size]
    )
    
    # Create table body
    table_id = p.createMultiBody(
        baseMass=0,  # Static table
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=[0.5, 0, table_height]
    )
    
    # Load robot (using Kuka for demonstration)
    robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True
    )
    
    # Get end-effector link (last link)
    num_joints = p.getNumJoints(robot_id)
    ee_link = num_joints - 1
    
    # Add some objects to grasp
    objects = []
    
    # Blue cube
    cube_size = 0.05
    cube_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[cube_size/2] * 3,
        rgbaColor=[0, 0, 1, 1]  # Blue
    )
    cube_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[cube_size/2] * 3
    )
    cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cube_collision,
        baseVisualShapeIndex=cube_visual,
        basePosition=[0.5, 0.1, table_height + cube_size/2 + 0.01]
    )
    objects.append(("blue_cube", cube_id))
    
    # Red cylinder
    cylinder_radius = 0.03
    cylinder_height = 0.08
    cylinder_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=cylinder_radius,
        length=cylinder_height,
        rgbaColor=[1, 0, 0, 1]  # Red
    )
    cylinder_collision = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=cylinder_radius,
        height=cylinder_height
    )
    cylinder_id = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=cylinder_collision,
        baseVisualShapeIndex=cylinder_visual,
        basePosition=[0.4, -0.1, table_height + cylinder_height/2 + 0.01]
    )
    objects.append(("red_cylinder", cylinder_id))
    
    return physics_client, robot_id, ee_link, table_id, table_height, objects


def demonstrate_contact_queries(executor: SafeGraspExecutor, robot_id: int, ee_link: int):
    """Demonstrate contact query functionality."""
    print("\n" + "="*60)
    print("CONTACT QUERY DEMONSTRATION")
    print("="*60)
    
    # Get current EE position
    ee_state = p.getLinkState(robot_id, ee_link)
    current_pos = ee_state[0]
    
    print(f"\nCurrent EE position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
    
    # Check contacts
    contacts = executor.check_contacts(check_table=True, check_all=True)
    
    print(f"\nContact State: {contacts.state.value}")
    print(f"Has contact: {contacts.has_contact}")
    print(f"Distance to nearest: {contacts.distance_to_nearest:.3f}m")
    
    if contacts.has_contact:
        print(f"Contact forces: {contacts.contact_forces}")
        print(f"Max force: {contacts.get_max_force():.2f}N")
    
    print(f"Is safe to proceed: {contacts.is_safe()}")
    
    # Simulate moving closer to table
    print("\n📊 Simulating approach to table...")
    test_positions = [
        current_pos[2],
        current_pos[2] - 0.05,
        current_pos[2] - 0.1,
        executor.config.table_height + 0.02,
        executor.config.table_height + 0.01,
        executor.config.table_height + 0.005
    ]
    
    for z in test_positions:
        # This is a simulation - don't actually move
        simulated_pos = (current_pos[0], current_pos[1], z)
        distance = z - executor.config.table_height
        
        if distance < 0:
            state = ContactState.IN_CONTACT
        elif distance < executor.config.contact_check_distance:
            state = ContactState.NEAR_CONTACT
        else:
            state = ContactState.NO_CONTACT
        
        print(f"   Z={z:.3f}m: distance={distance:.3f}m, state={state.value}")


def demonstrate_vertical_lift(executor: SafeGraspExecutor, object_pos: tuple):
    """Demonstrate vertical lift before lateral movement."""
    print("\n" + "="*60)
    print("VERTICAL LIFT DEMONSTRATION")
    print("="*60)
    
    # Define start and target positions
    start_pos = (0.3, 0.0, executor.config.table_height + 0.1)
    target_pos = object_pos
    
    print(f"\n📍 Movement task:")
    print(f"   Start:  ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
    print(f"   Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    
    # Calculate direct distance
    direct_distance = np.linalg.norm(np.array(target_pos) - np.array(start_pos))
    print(f"   Direct distance: {direct_distance:.3f}m")
    
    # Execute movement with vertical lift
    print(f"\n🔄 Executing with vertical lift (height={VERTICAL_LIFT_HEIGHT}m)...")
    result = executor.move_with_vertical_lift(start_pos, target_pos)
    
    if result['success']:
        print("   ✅ Movement successful!")
        print(f"   Trajectory length: {result['total_distance']:.3f}m")
        print(f"   Efficiency: {direct_distance/result['total_distance']:.1%}")
        
        print("\n   Trajectory waypoints:")
        for i, pos in enumerate(result['trajectory']):
            print(f"      {i+1}. ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        print(f"\n   Contact events: {result['contact_events']}")
    else:
        print(f"   ❌ Movement failed: {result['reason']}")


def demonstrate_safe_grasp(executor: SafeGraspExecutor, object_name: str, object_pos: tuple):
    """Demonstrate complete safe grasp sequence."""
    print("\n" + "="*60)
    print(f"SAFE GRASP DEMONSTRATION - {object_name}")
    print("="*60)
    
    print(f"\n🎯 Target object: {object_name} at {object_pos}")
    
    # Phase 1: Safe approach
    print("\n1️⃣ Safe Approach Phase")
    print("   Using contact queries to avoid table collision...")
    
    approach_result = executor.safe_approach_grasp(
        object_position=object_pos,
        approach_direction=(0, 0, -1)  # From above
    )
    
    if approach_result['success']:
        print(f"   ✅ Approach successful!")
        print(f"   Final position: {approach_result['final_position']}")
        print(f"   Contact state: {approach_result['contact_state']}")
        print(f"   Min table clearance: {approach_result['min_table_clearance']:.3f}m")
        
        # Simulate grasping
        print("\n2️⃣ Grasp Execution")
        print("   Closing gripper...")
        time.sleep(0.5)  # Simulate gripper closing
        print("   ✅ Object grasped!")
        
        # Phase 3: Safe lift and retreat
        print("\n3️⃣ Safe Lift and Retreat")
        print("   Lifting with vertical clearance...")
        
        retreat_result = executor.safe_lift_and_retreat(
            current_pos=approach_result['final_position']
        )
        
        if retreat_result['success']:
            print(f"   ✅ Retreat successful!")
            print(f"   Final height: {retreat_result['final_position'][2]:.3f}m")
            print(f"   Total lift: {retreat_result['total_height_gained']:.3f}m")
        else:
            print(f"   ❌ Retreat failed: {retreat_result['reason']}")
    else:
        print(f"   ❌ Approach failed: {approach_result['reason']}")


def demonstrate_table_avoidance(executor: SafeGraspExecutor):
    """Demonstrate table penetration avoidance."""
    print("\n" + "="*60)
    print("TABLE PENETRATION AVOIDANCE DEMONSTRATION")
    print("="*60)
    
    # Test positions that would penetrate table
    test_cases = [
        {
            "name": "Safe above table",
            "pos": (0.5, 0.0, executor.config.table_height + 0.05),
            "expected": "safe"
        },
        {
            "name": "Near table surface",
            "pos": (0.5, 0.0, executor.config.table_height + 0.002),
            "expected": "near_contact"
        },
        {
            "name": "Below table (would penetrate)",
            "pos": (0.5, 0.0, executor.config.table_height - 0.01),
            "expected": "blocked"
        }
    ]
    
    print("\n🛡️ Testing table collision avoidance...")
    
    for test in test_cases:
        print(f"\n   Test: {test['name']}")
        print(f"   Target Z: {test['pos'][2]:.3f}m (table at {executor.config.table_height:.3f}m)")
        
        # Check if position is safe
        clearance = test['pos'][2] - executor.config.table_height
        
        if clearance < 0:
            print(f"   ❌ BLOCKED - Would penetrate table by {abs(clearance):.3f}m")
        elif clearance < executor.config.min_table_clearance:
            print(f"   ⚠️ WARNING - Only {clearance:.3f}m clearance (min: {executor.config.min_table_clearance:.3f}m)")
        else:
            print(f"   ✅ SAFE - {clearance:.3f}m clearance from table")


def main():
    """Run safe grasp execution demonstration."""
    
    print("🚀 Starting Safe Grasp Execution Demo...")
    print(f"   • Contact queries: ENABLED")
    print(f"   • Vertical lift height: {VERTICAL_LIFT_HEIGHT}m")
    print(f"   • Min table clearance: 5mm")
    
    # Setup simulation
    try:
        physics_client, robot_id, ee_link, table_id, table_height, objects = setup_simulation()
        print("✅ Simulation environment created")
    except Exception as e:
        print(f"❌ Failed to setup simulation: {e}")
        return
    
    # Create safe grasp executor
    executor = create_safe_grasp_executor(
        robot_id=robot_id,
        end_effector_link=ee_link,
        table_id=table_id,
        table_height=table_height,
        enable_contact_queries=True,
        vertical_lift_height=VERTICAL_LIFT_HEIGHT
    )
    
    # Run demonstrations
    try:
        # 1. Contact query demonstration
        demonstrate_contact_queries(executor, robot_id, ee_link)
        time.sleep(1)
        
        # 2. Table avoidance demonstration
        demonstrate_table_avoidance(executor)
        time.sleep(1)
        
        # 3. Vertical lift demonstration
        if objects:
            object_name, object_id = objects[0]
            object_pos_info = p.getBasePositionAndOrientation(object_id)
            object_pos = object_pos_info[0]
            demonstrate_vertical_lift(executor, object_pos)
            time.sleep(1)
        
        # 4. Complete safe grasp demonstration
        for object_name, object_id in objects:
            object_pos_info = p.getBasePositionAndOrientation(object_id)
            object_pos = object_pos_info[0]
            demonstrate_safe_grasp(executor, object_name, object_pos)
            time.sleep(2)
        
        # Display summary
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Contact History:")
        contact_states = [h['state'] for h in executor.contact_history]
        for state in set(contact_states):
            count = contact_states.count(state)
            print(f"   {state}: {count} occurrences")
        
        print(f"\n📍 Movement History:")
        print(f"   Total movements: {len(executor.movement_history)}")
        if executor.movement_history:
            print(f"   First movement: {executor.movement_history[0]['time']}")
            print(f"   Last movement: {executor.movement_history[-1]['time']}")
        
        print("\n✅ Demo completed successfully!")
        
        # Keep simulation running
        print("\nPress Ctrl+C to exit...")
        while True:
            p.stepSimulation()
            time.sleep(1/240)
            
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        p.disconnect()
        print("🔌 Disconnected from simulation")


if __name__ == "__main__":
    main()