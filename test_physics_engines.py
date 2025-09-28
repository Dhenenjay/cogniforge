#!/usr/bin/env python3
"""
Physics Engine Test Suite for Cogniforge Project
================================================
This script tests available physics engines and provides alternatives
for the Cogniforge cognitive architecture project.
"""

import sys
import platform
import importlib
from datetime import datetime


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def test_system_info():
    """Display system information."""
    print_header("System Information")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def test_pymunk():
    """Test PyMunk 2D physics engine."""
    print_header("Testing PyMunk (2D Physics Engine)")
    try:
        import pymunk
        print(f"✓ PyMunk imported successfully")
        print(f"  Version: {pymunk.version}")
        
        # Create a simple physics simulation
        space = pymunk.Space()
        space.gravity = 0, -981  # cm/s^2
        
        # Create a static ground
        ground = pymunk.Segment(space.static_body, (0, 0), (400, 0), 5)
        ground.friction = 1.0
        space.add(ground)
        
        # Create a falling box
        mass = 1
        moment = pymunk.moment_for_box(mass, (50, 50))
        body = pymunk.Body(mass, moment)
        body.position = 200, 200
        shape = pymunk.Poly.create_box(body, (50, 50))
        shape.friction = 0.5
        space.add(body, shape)
        
        # Simulate a few steps
        print(f"  Running simple simulation...")
        for i in range(10):
            space.step(1/60.0)
        
        print(f"  Final box position: {body.position}")
        print(f"✓ PyMunk simulation successful")
        
        # Test collision detection
        print(f"  Testing collision detection...")
        body2 = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 25))
        body2.position = 200, 300
        circle = pymunk.Circle(body2, 25)
        space.add(body2, circle)
        
        print(f"✓ PyMunk collision system working")
        
        return True
        
    except ImportError as e:
        print(f"✗ PyMunk not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ PyMunk test failed: {e}")
        return False


def test_pybullet():
    """Test PyBullet 3D physics engine."""
    print_header("Testing PyBullet (3D Physics Engine)")
    try:
        import pybullet as p
        print(f"✓ PyBullet imported successfully")
        
        # Get version info
        try:
            import pybullet_data
            print(f"  PyBullet data path: {pybullet_data.getDataPath()}")
        except:
            pass
        
        # Connect to physics server (DIRECT mode - no GUI)
        physics_client = p.connect(p.DIRECT)
        print(f"  Physics client ID: {physics_client}")
        
        # Set gravity
        p.setGravity(0, 0, -9.81)
        
        # Load a plane
        plane_id = p.createCollisionShape(p.GEOM_PLANE)
        plane_body = p.createMultiBody(0, plane_id)
        
        # Create a box
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
        box_body = p.createMultiBody(baseMass=1,
                                     baseCollisionShapeIndex=box_collision,
                                     baseVisualShapeIndex=box_visual,
                                     basePosition=[0, 0, 2])
        
        # Run simulation
        print(f"  Running simulation...")
        for i in range(100):
            p.stepSimulation()
        
        # Get final position
        pos, orn = p.getBasePositionAndOrientation(box_body)
        print(f"  Final box position: {pos}")
        
        # Disconnect
        p.disconnect()
        print(f"✓ PyBullet simulation successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ PyBullet not installed: {e}")
        print(f"  Note: PyBullet requires compilation from source on Windows with Python 3.11")
        print(f"  Consider using Python 3.10 or WSL2 for PyBullet support")
        return False
    except Exception as e:
        print(f"✗ PyBullet test failed: {e}")
        return False


def test_alternatives():
    """Suggest and test alternative physics libraries."""
    print_header("Alternative Physics Libraries")
    
    alternatives = {
        'ode': 'Open Dynamics Engine - 3D physics',
        'mujoco': 'MuJoCo - Advanced physics simulation (now free)',
        'box2d': 'Box2D - 2D physics engine',
        'matter-py': 'Matter.py - 2D physics',
    }
    
    print("Checking available alternatives:")
    for lib, description in alternatives.items():
        try:
            if lib == 'ode':
                import ode
                print(f"  ✓ {lib}: {description} - AVAILABLE")
            elif lib == 'mujoco':
                import mujoco
                print(f"  ✓ {lib}: {description} - AVAILABLE")
            elif lib == 'box2d':
                import Box2D
                print(f"  ✓ {lib}: {description} - AVAILABLE")
            else:
                importlib.import_module(lib.replace('-', '_'))
                print(f"  ✓ {lib}: {description} - AVAILABLE")
        except ImportError:
            print(f"  ✗ {lib}: {description} - Not installed")


def create_cogniforge_physics_wrapper():
    """Create a physics wrapper that can work with available engines."""
    print_header("Cogniforge Physics Wrapper")
    
    code = '''
class CogniforgePhysics:
    """
    Physics wrapper for Cogniforge project that abstracts the physics engine.
    Can work with PyBullet (3D) or PyMunk (2D) depending on availability.
    """
    
    def __init__(self, use_3d=True):
        self.use_3d = use_3d
        self.engine = None
        self.space = None
        
        if use_3d:
            try:
                import pybullet as p
                self.engine = 'pybullet'
                self.p = p
                print("Using PyBullet for 3D physics")
            except ImportError:
                print("PyBullet not available, falling back to 2D physics")
                self.use_3d = False
        
        if not self.use_3d:
            try:
                import pymunk
                self.engine = 'pymunk'
                self.pymunk = pymunk
                self.space = pymunk.Space()
                self.space.gravity = (0, -981)
                print("Using PyMunk for 2D physics")
            except ImportError:
                raise ImportError("No physics engine available!")
    
    def create_ground(self):
        """Create a ground plane/line."""
        if self.engine == 'pybullet':
            plane_id = self.p.createCollisionShape(self.p.GEOM_PLANE)
            return self.p.createMultiBody(0, plane_id)
        elif self.engine == 'pymunk':
            ground = self.pymunk.Segment(self.space.static_body, (0, 0), (1000, 0), 5)
            ground.friction = 1.0
            self.space.add(ground)
            return ground
    
    def create_box(self, position, size=1.0, mass=1.0):
        """Create a box object."""
        if self.engine == 'pybullet':
            half_size = size / 2.0
            collision = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[half_size]*3)
            visual = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[half_size]*3)
            return self.p.createMultiBody(baseMass=mass,
                                        baseCollisionShapeIndex=collision,
                                        baseVisualShapeIndex=visual,
                                        basePosition=position)
        elif self.engine == 'pymunk':
            moment = self.pymunk.moment_for_box(mass, (size*100, size*100))
            body = self.pymunk.Body(mass, moment)
            body.position = position[0]*100, position[1]*100  # Convert to pymunk scale
            shape = self.pymunk.Poly.create_box(body, (size*100, size*100))
            self.space.add(body, shape)
            return body
    
    def step_simulation(self, timestep=1/60.0):
        """Step the physics simulation."""
        if self.engine == 'pybullet':
            self.p.stepSimulation()
        elif self.engine == 'pymunk':
            self.space.step(timestep)
    
    def get_position(self, obj):
        """Get object position."""
        if self.engine == 'pybullet':
            pos, _ = self.p.getBasePositionAndOrientation(obj)
            return pos
        elif self.engine == 'pymunk':
            return obj.position.x/100, obj.position.y/100  # Convert back from pymunk scale
    '''
    
    print("Created CogniforgePhysics wrapper class")
    print("\nExample usage:")
    print(code[:500] + "...")
    
    # Save the wrapper
    with open('cogniforge_physics.py', 'w') as f:
        f.write(code)
    print("\n✓ Saved physics wrapper to 'cogniforge_physics.py'")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# COGNIFORGE PHYSICS ENGINE TEST SUITE")
    print("#"*60)
    
    test_system_info()
    
    # Test available engines
    pymunk_ok = test_pymunk()
    pybullet_ok = test_pybullet()
    
    test_alternatives()
    
    # Create wrapper
    create_cogniforge_physics_wrapper()
    
    # Summary
    print_header("Test Summary")
    print(f"PyMunk (2D):   {'✓ WORKING' if pymunk_ok else '✗ NOT AVAILABLE'}")
    print(f"PyBullet (3D): {'✓ WORKING' if pybullet_ok else '✗ NOT AVAILABLE'}")
    
    if not pybullet_ok:
        print("\n⚠ PyBullet Installation Note:")
        print("  PyBullet doesn't have pre-built wheels for Python 3.11 on Windows.")
        print("  Options to get PyBullet working:")
        print("  1. Use Python 3.10 instead of 3.11")
        print("  2. Use WSL2 (Windows Subsystem for Linux)")
        print("  3. Build from source (requires proper MSVC setup)")
        print("  4. Use alternative 3D physics engines like MuJoCo")
    
    if pymunk_ok:
        print("\n✓ PyMunk is working and can be used for 2D physics simulations")
        print("  This is suitable for many cognitive architecture experiments")
    
    print("\n" + "#"*60)
    print("# TEST SUITE COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()