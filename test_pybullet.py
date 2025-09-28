#!/usr/bin/env python
"""
Test script for PyBullet installation verification.
Tests all major PyBullet features including physics simulation, 
URDF loading, collision detection, and rendering.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np

def test_pybullet_installation():
    """Comprehensive test of PyBullet installation and features."""
    
    print("=" * 60)
    print("PyBullet Installation Test")
    print("=" * 60)
    
    # Test 1: Import and basic connection
    print("\n1. Testing PyBullet import and connection...")
    try:
        # Connect to physics server (DIRECT mode for headless testing)
        physicsClient = p.connect(p.DIRECT)
        print(f"   ✓ Connected to physics client: {physicsClient}")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        return False
    
    # Test 2: Set up simulation parameters
    print("\n2. Testing simulation setup...")
    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        print("   ✓ Simulation parameters set")
    except Exception as e:
        print(f"   ✗ Failed to set parameters: {e}")
        return False
    
    # Test 3: Load URDF models
    print("\n3. Testing URDF loading...")
    try:
        # Set the search path to find URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load plane
        planeId = p.loadURDF("plane.urdf")
        print(f"   ✓ Loaded plane: ID {planeId}")
        
        # Load robot (R2D2)
        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        robotId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
        print(f"   ✓ Loaded R2D2 robot: ID {robotId}")
    except Exception as e:
        print(f"   ✗ Failed to load URDF: {e}")
        return False
    
    # Test 4: Get robot information
    print("\n4. Testing robot information retrieval...")
    try:
        numJoints = p.getNumJoints(robotId)
        print(f"   ✓ Robot has {numJoints} joints")
        
        # Get joint info for first joint if available
        if numJoints > 0:
            jointInfo = p.getJointInfo(robotId, 0)
            print(f"   ✓ Joint 0 name: {jointInfo[1].decode('utf-8')}")
    except Exception as e:
        print(f"   ✗ Failed to get robot info: {e}")
        return False
    
    # Test 5: Run simulation
    print("\n5. Testing physics simulation...")
    try:
        for i in range(240):  # Simulate 1 second
            p.stepSimulation()
        
        # Get robot position after simulation
        pos, orn = p.getBasePositionAndOrientation(robotId)
        print(f"   ✓ Simulation ran for 1 second")
        print(f"   ✓ Robot position after sim: {[round(p, 3) for p in pos]}")
    except Exception as e:
        print(f"   ✗ Failed to run simulation: {e}")
        return False
    
    # Test 6: Collision detection
    print("\n6. Testing collision detection...")
    try:
        # Check for contacts
        contacts = p.getContactPoints()
        print(f"   ✓ Detected {len(contacts)} contact points")
        
        # Get closest points between robot and plane
        closest = p.getClosestPoints(robotId, planeId, 10.0)
        print(f"   ✓ Found {len(closest)} closest point pairs")
    except Exception as e:
        print(f"   ✗ Failed collision detection: {e}")
        return False
    
    # Test 7: Create additional objects
    print("\n7. Testing object creation...")
    try:
        # Create a box
        boxCollisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.5]
        )
        boxVisualShapeId = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.5],
            rgbaColor=[1, 0, 0, 1]
        )
        boxId = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=boxCollisionShapeId,
            baseVisualShapeIndex=boxVisualShapeId,
            basePosition=[2, 0, 2]
        )
        print(f"   ✓ Created box: ID {boxId}")
    except Exception as e:
        print(f"   ✗ Failed to create objects: {e}")
        return False
    
    # Test 8: Ray casting
    print("\n8. Testing ray casting...")
    try:
        rayFrom = [0, 0, 5]
        rayTo = [0, 0, -5]
        rayResult = p.rayTest(rayFrom, rayTo)
        if rayResult[0][0] != -1:
            print(f"   ✓ Ray hit object ID: {rayResult[0][0]}")
        else:
            print("   ✓ Ray test completed (no hit)")
    except Exception as e:
        print(f"   ✗ Failed ray casting: {e}")
        return False
    
    # Test 9: Camera and rendering (even in DIRECT mode)
    print("\n9. Testing camera setup...")
    try:
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=320, 
            height=240,
            viewMatrix=p.computeViewMatrix(
                cameraEyePosition=[0, -5, 3],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 1]
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60,
                aspect=320/240,
                nearVal=0.1,
                farVal=100
            )
        )
        print(f"   ✓ Camera image captured: {width}x{height}")
        print(f"   ✓ RGB image shape: {rgbImg.shape}")
    except Exception as e:
        print(f"   ✗ Failed camera setup: {e}")
        return False
    
    # Test 10: Clean up
    print("\n10. Testing cleanup...")
    try:
        p.disconnect()
        print("   ✓ Disconnected from physics server")
    except Exception as e:
        print(f"   ✗ Failed to disconnect: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All PyBullet tests passed successfully!")
    print("=" * 60)
    return True

def test_gui_mode():
    """Test PyBullet GUI mode (optional)."""
    print("\nTesting GUI mode (will open a window)...")
    try:
        physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load plane and robot
        planeId = p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 1]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        robotId = p.loadURDF("r2d2.urdf", robotStartPos, robotStartOrientation)
        
        print("GUI window opened. Running simulation for 3 seconds...")
        print("You should see a robot falling onto a plane.")
        
        for i in range(3 * 240):  # 3 seconds at 240Hz
            p.stepSimulation()
            time.sleep(1/240)
        
        p.disconnect()
        print("✓ GUI test completed")
        return True
    except Exception as e:
        print(f"GUI test failed (this is okay if running headless): {e}")
        return False

if __name__ == "__main__":
    # Run comprehensive tests
    success = test_pybullet_installation()
    
    if success:
        print("\n" + "="*60)
        print("PyBullet is fully installed and functional!")
        print("You can now use PyBullet for physics simulations.")
        print("="*60)
        
        # Optional: test GUI mode
        print("\nWould you like to test GUI mode? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                test_gui_mode()
        except:
            print("Skipping GUI test")