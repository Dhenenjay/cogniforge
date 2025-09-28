"""
Skills Library Demo

Demonstrates the use of manipulation skills from the library.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
import time
from skills_library import (
    SkillRegistry, 
    SkillParameters,
    PushSkill,
    SlideSkill, 
    StackSkill,
    PickPlaceSkill
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_scene():
    """Create a demo scene with multiple objects"""
    
    # Create ground
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
    
    # Create robot
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True
    )
    
    # Set initial joint positions
    initial_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
    for i, pos in enumerate(initial_joints):
        p.resetJointState(robot_id, i, pos)
    
    # Create objects
    objects = {}
    
    # Red cube (for pushing)
    red_cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025],
        rgbaColor=[1, 0, 0, 1]
    )
    red_cube_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025]
    )
    red_cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=red_cube_collision,
        baseVisualShapeIndex=red_cube_visual,
        basePosition=[0.4, -0.2, 0.025]
    )
    objects['red_cube'] = red_cube_id
    
    # Blue cube (for sliding)
    blue_cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025],
        rgbaColor=[0, 0, 1, 1]
    )
    blue_cube_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025]
    )
    blue_cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=blue_cube_collision,
        baseVisualShapeIndex=blue_cube_visual,
        basePosition=[0.4, 0, 0.025]
    )
    objects['blue_cube'] = blue_cube_id
    
    # Green cube (for stacking)
    green_cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025],
        rgbaColor=[0, 1, 0, 1]
    )
    green_cube_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025]
    )
    green_cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=green_cube_collision,
        baseVisualShapeIndex=green_cube_visual,
        basePosition=[0.4, 0.2, 0.025]
    )
    objects['green_cube'] = green_cube_id
    
    # Yellow cube (for pick and place)
    yellow_cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025],
        rgbaColor=[1, 1, 0, 1]
    )
    yellow_cube_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025]
    )
    yellow_cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=yellow_cube_collision,
        baseVisualShapeIndex=yellow_cube_visual,
        basePosition=[0.5, -0.1, 0.025]
    )
    objects['yellow_cube'] = yellow_cube_id
    
    # Create target platform
    platform_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.08, 0.08, 0.01],
        rgbaColor=[0.3, 0.3, 0.3, 1]
    )
    platform_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.08, 0.08, 0.01]
    )
    platform_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=platform_collision,
        baseVisualShapeIndex=platform_visual,
        basePosition=[0.3, 0.3, 0.01]
    )
    objects['platform'] = platform_id
    
    return robot_id, objects


def demo_push_skill(robot_id, objects):
    """Demonstrate push skill"""
    print("\n" + "="*60)
    print(" PUSH SKILL DEMO")
    print("="*60)
    print("Pushing red cube to new position...")
    
    # Create push skill
    push_skill = PushSkill(robot_id, objects)
    
    # Set parameters
    params = SkillParameters(
        object_id=objects['red_cube'],
        target_position=np.array([0.5, -0.2, 0.025]),
        speed=0.01
    )
    
    # Execute skill
    result = push_skill.execute(params)
    
    print(f"Result: {result.message}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    return result.status.value == "success"


def demo_slide_skill(robot_id, objects):
    """Demonstrate slide skill"""
    print("\n" + "="*60)
    print(" SLIDE SKILL DEMO")
    print("="*60)
    print("Sliding blue cube across surface...")
    
    # Create slide skill
    slide_skill = SlideSkill(robot_id, objects)
    
    # Set parameters
    params = SkillParameters(
        object_id=objects['blue_cube'],
        target_position=np.array([0.5, 0, 0.025]),
        speed=0.01
    )
    
    # Execute skill
    result = slide_skill.execute(params)
    
    print(f"Result: {result.message}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    return result.status.value == "success"


def demo_stack_skill(robot_id, objects):
    """Demonstrate stack skill"""
    print("\n" + "="*60)
    print(" STACK SKILL DEMO")
    print("="*60)
    print("Stacking green cube on blue cube...")
    
    # Import extended parameters
    from skills_library.stack_skill import StackParameters
    
    # Create stack skill
    stack_skill = StackSkill(robot_id, objects)
    
    # Set parameters to stack green on blue
    params = StackParameters(
        object_id=objects['green_cube'],
        base_object_id=objects['blue_cube'],
        speed=0.008,
        alignment_precision=0.01
    )
    
    # Execute skill
    result = stack_skill.execute(params)
    
    print(f"Result: {result.message}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    if result.data and 'alignment_error' in result.data:
        print(f"Alignment error: {result.data['alignment_error']*1000:.1f}mm")
    
    return result.status.value == "success"


def demo_pick_place_skill(robot_id, objects):
    """Demonstrate pick and place skill"""
    print("\n" + "="*60)
    print(" PICK & PLACE SKILL DEMO")
    print("="*60)
    print("Picking yellow cube and placing on platform...")
    
    # Create pick-place skill
    pick_place_skill = PickPlaceSkill(robot_id, objects)
    
    # Get platform position
    platform_pos, _ = p.getBasePositionAndOrientation(objects['platform'])
    target_pos = np.array(platform_pos)
    target_pos[2] += 0.05  # Place on top of platform
    
    # Set parameters
    params = SkillParameters(
        object_id=objects['yellow_cube'],
        target_position=target_pos,
        speed=0.01,
        precision=0.01
    )
    
    # Execute skill
    result = pick_place_skill.execute(params)
    
    print(f"Result: {result.message}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    if result.data and 'error' in result.data:
        print(f"Placement error: {result.data['error']*1000:.1f}mm")
    
    return result.status.value == "success"


def demo_skill_registry(robot_id, objects):
    """Demonstrate skill registry usage"""
    print("\n" + "="*60)
    print(" SKILL REGISTRY DEMO")
    print("="*60)
    
    # Create and setup registry
    registry = SkillRegistry()
    
    # Register skills
    registry.register('push', PushSkill)
    registry.register('slide', SlideSkill)
    registry.register('stack', StackSkill)
    registry.register('pick_place', PickPlaceSkill)
    
    print(f"Registered skills: {registry.list_skills()}")
    
    # Create skill instance from registry
    push_skill = registry.create_skill('push', robot_id, objects)
    
    if push_skill:
        print("\n✅ Successfully created push skill from registry")
        
        # Use the skill
        params = SkillParameters(
            object_id=objects['red_cube'],
            target_position=np.array([0.45, -0.15, 0.025]),
            speed=0.01
        )
        
        result = push_skill.execute(params)
        print(f"Execution result: {result.message}")
    
    return True


def main():
    """Main demo function"""
    
    print("\n" + "="*80)
    print(" SKILLS LIBRARY DEMONSTRATION")
    print("="*80)
    
    # Connect to PyBullet
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    
    # Set camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0, 0.1]
    )
    
    # Create scene
    robot_id, objects = create_demo_scene()
    
    # Wait for scene to settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1/240)
    
    # Run demos
    demos = [
        ("Push Skill", demo_push_skill),
        ("Slide Skill", demo_slide_skill),
        ("Pick & Place Skill", demo_pick_place_skill),
        ("Stack Skill", demo_stack_skill),
        ("Skill Registry", demo_skill_registry)
    ]
    
    results = []
    
    for name, demo_func in demos:
        print(f"\n📋 Running: {name}")
        input("Press Enter to continue...")
        
        try:
            success = demo_func(robot_id, objects)
            results.append((name, success))
            
            # Wait between demos
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print(" DEMONSTRATION SUMMARY")
    print("="*80)
    
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{name:20} - {status}")
    
    print("\n" + "="*80)
    print(" DEMO COMPLETE")
    print("="*80)
    
    print("\nSimulation will continue running. Press Ctrl+C to exit.")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        print("\nShutting down...")
        p.disconnect()


if __name__ == "__main__":
    main()