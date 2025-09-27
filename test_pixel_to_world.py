"""
Test script for pixel_to_world coordinate conversion.

Demonstrates camera calibration and pixel-to-world transformations.
"""

import numpy as np
from cogniforge.vision.vision_utils import (
    pixel_to_world,
    world_to_pixel,
    estimate_focal_length,
    get_camera_intrinsics,
    CameraCalibration
)


def test_basic_conversion():
    """Test basic pixel to world conversion."""
    print("="*60)
    print("BASIC PIXEL TO WORLD CONVERSION")
    print("="*60)
    
    # Typical camera parameters
    fx = 600.0  # Focal length in pixels (x-direction)
    fy = 600.0  # Focal length in pixels (y-direction)
    
    # Object at 50cm depth
    depth = 0.5  # meters
    
    # Test cases: pixel offsets from image center
    test_cases = [
        (100, 0, "100px right of center"),
        (0, 100, "100px below center"),
        (-100, 0, "100px left of center"),
        (0, -100, "100px above center"),
        (100, -50, "100px right, 50px up"),
        (200, 150, "200px right, 150px down")
    ]
    
    print(f"\nCamera parameters:")
    print(f"  Focal length: fx={fx}, fy={fy} pixels")
    print(f"  Object depth: {depth} meters")
    
    print("\nConversions:")
    for dx_px, dy_px, description in test_cases:
        dx_m, dy_m = pixel_to_world(dx_px, dy_px, depth, fx, fy)
        print(f"\n  {description}:")
        print(f"    Pixel offset: ({dx_px:4d}, {dy_px:4d}) pixels")
        print(f"    World offset: ({dx_m:7.3f}, {dy_m:7.3f}) meters")
        print(f"                  ({dx_m*100:7.1f}, {dy_m*100:7.1f}) cm")


def test_inverse_conversion():
    """Test that world_to_pixel is inverse of pixel_to_world."""
    print("\n" + "="*60)
    print("INVERSE CONVERSION TEST")
    print("="*60)
    
    fx = 600.0
    fy = 600.0
    depth = 0.5
    
    # Test roundtrip conversion
    original_pixel = (150, -75)
    
    # Convert to world
    dx_m, dy_m = pixel_to_world(original_pixel[0], original_pixel[1], depth, fx, fy)
    
    # Convert back to pixel
    dx_px_back, dy_px_back = world_to_pixel(dx_m, dy_m, depth, fx, fy)
    
    print(f"\nRoundtrip test:")
    print(f"  Original pixel offset: {original_pixel}")
    print(f"  → World offset: ({dx_m:.3f}, {dy_m:.3f}) meters")
    print(f"  → Back to pixel: ({dx_px_back}, {dy_px_back})")
    print(f"  ✓ Match: {(dx_px_back, dy_px_back) == original_pixel}")


def test_different_focal_lengths():
    """Test with different focal lengths (different FOVs)."""
    print("\n" + "="*60)
    print("DIFFERENT FOCAL LENGTHS (FOV)")
    print("="*60)
    
    # Same pixel offset, different focal lengths
    dx_px = 100
    dy_px = 100
    depth = 0.5
    
    # Different camera types
    cameras = [
        ("Wide angle (90° FOV)", estimate_focal_length(640, 90)),
        ("Normal (60° FOV)", estimate_focal_length(640, 60)),
        ("Telephoto (30° FOV)", estimate_focal_length(640, 30)),
        ("Typical webcam", 600.0),
        ("High-res camera", 1200.0)
    ]
    
    print(f"\nFixed pixel offset: ({dx_px}, {dy_px}) pixels")
    print(f"Depth: {depth} meters")
    
    print("\nWorld coordinates for different cameras:")
    for name, focal_length in cameras:
        dx_m, dy_m = pixel_to_world(dx_px, dy_px, depth, focal_length, focal_length)
        print(f"\n  {name} (f={focal_length:.0f}px):")
        print(f"    World offset: ({dx_m:.3f}, {dy_m:.3f}) meters")
        print(f"                  ({dx_m*100:.1f}, {dy_m*100:.1f}) cm")


def test_with_camera_intrinsics():
    """Test using camera intrinsic matrix."""
    print("\n" + "="*60)
    print("USING CAMERA INTRINSIC MATRIX")
    print("="*60)
    
    # Get camera intrinsics from FOV
    intrinsics = get_camera_intrinsics(
        image_width=640,
        image_height=480,
        fov_horizontal=60.0
    )
    
    print("\nCamera intrinsics from 60° FOV:")
    print(f"  Image size: {intrinsics['image_width']}x{intrinsics['image_height']}")
    print(f"  Focal length: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
    print(f"  Principal point: cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
    print("\nIntrinsic matrix K:")
    print(intrinsics['K'])
    
    # Test conversion
    dx_px, dy_px = 100, -50
    depth = 0.5
    
    dx_m, dy_m = pixel_to_world(
        dx_px, dy_px, depth,
        intrinsics['fx'], intrinsics['fy']
    )
    
    print(f"\nPixel offset ({dx_px}, {dy_px}) at {depth}m depth:")
    print(f"  World offset: ({dx_m:.3f}, {dy_m:.3f}) meters")


def test_camera_calibration_class():
    """Test the CameraCalibration helper class."""
    print("\n" + "="*60)
    print("CAMERA CALIBRATION CLASS")
    print("="*60)
    
    # Create calibration from FOV
    calib = CameraCalibration.from_fov(
        image_width=640,
        image_height=480,
        fov_horizontal=60.0
    )
    
    print(f"\n{calib}")
    
    # Test conversions
    test_points = [
        (50, 0, 0.3),   # 50px right at 30cm
        (0, 50, 0.5),   # 50px down at 50cm
        (100, 100, 1.0) # Diagonal at 1m
    ]
    
    print("\nConversion tests:")
    for dx_px, dy_px, depth in test_points:
        dx_m, dy_m = calib.pixel_to_world(dx_px, dy_px, depth)
        print(f"\n  Pixel ({dx_px:3d}, {dy_px:3d}) at {depth:.1f}m:")
        print(f"    → World: ({dx_m:.3f}, {dy_m:.3f}) m")
        print(f"    → World: ({dx_m*100:.1f}, {dy_m*100:.1f}) cm")
        
        # Verify inverse
        dx_px_back, dy_px_back = calib.world_to_pixel(dx_m, dy_m, depth)
        print(f"    ← Back:  ({dx_px_back}, {dy_px_back}) pixels")


def practical_example():
    """Practical example: Robot gripper control."""
    print("\n" + "="*60)
    print("PRACTICAL EXAMPLE: ROBOT GRIPPER CONTROL")
    print("="*60)
    
    # Setup camera (typical RGB-D camera)
    calib = CameraCalibration.from_fov(
        image_width=640,
        image_height=480,
        fov_horizontal=58.0  # Intel RealSense typical FOV
    )
    
    print(f"\nCamera setup: {calib}")
    
    # Simulate object detection
    print("\n1. Object Detection:")
    # Object detected at pixel (420, 180) - right and above center
    object_pixel = (420, 180)
    image_center = (320, 240)
    dx_px = object_pixel[0] - image_center[0]  # +100 (right)
    dy_px = object_pixel[1] - image_center[1]  # -60 (up)
    
    print(f"   Object at pixel: {object_pixel}")
    print(f"   Image center: {image_center}")
    print(f"   Pixel offset: ({dx_px}, {dy_px})")
    
    # Get depth from RGB-D camera
    depth = 0.45  # 45cm from camera
    print(f"   Measured depth: {depth:.2f} meters")
    
    # Convert to world coordinates
    print("\n2. Convert to World Coordinates:")
    dx_m, dy_m = calib.pixel_to_world(dx_px, dy_px, depth)
    
    print(f"   World offset: ({dx_m:.3f}, {dy_m:.3f}) meters")
    print(f"   World offset: ({dx_m*100:.1f}, {dy_m*100:.1f}) cm")
    
    # Generate robot command
    print("\n3. Robot Control Command:")
    print(f"   Move gripper:")
    print(f"     - {abs(dx_m*100):.1f} cm {'RIGHT' if dx_m > 0 else 'LEFT'}")
    print(f"     - {abs(dy_m*100):.1f} cm {'DOWN' if dy_m > 0 else 'UP'}")
    print(f"     - {depth*100:.0f} cm FORWARD")
    
    # Simulate gripper alignment check
    print("\n4. Alignment Check:")
    tolerance = 0.01  # 1cm tolerance
    aligned_x = abs(dx_m) < tolerance
    aligned_y = abs(dy_m) < tolerance
    
    print(f"   X-axis aligned: {'✓' if aligned_x else '✗'} (error: {abs(dx_m)*100:.1f}cm)")
    print(f"   Y-axis aligned: {'✓' if aligned_y else '✗'} (error: {abs(dy_m)*100:.1f}cm)")
    print(f"   Ready to grasp: {'✓ YES' if aligned_x and aligned_y else '✗ NO'}")


def complete_pipeline_example():
    """Complete pipeline from detection to robot control."""
    print("\n" + "="*60)
    print("COMPLETE VISION-TO-CONTROL PIPELINE")
    print("="*60)
    
    from cogniforge.vision.vision_utils import compute_pixel_offset
    
    # 1. Setup camera
    print("\n1. Camera Setup:")
    calib = CameraCalibration(
        fx=600.0,  # Typical value
        fy=600.0,
        image_width=640,
        image_height=480
    )
    print(f"   {calib}")
    
    # 2. Create synthetic image with blue cube
    print("\n2. Object Detection (Blue Cube):")
    height, width = 480, 640
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Place blue cube at specific location
    cube_x, cube_y = 400, 200  # Pixel coordinates
    cube_size = 60
    rgb[cube_y:cube_y+cube_size, cube_x:cube_x+cube_size] = [0, 0, 255]
    
    # Detect cube
    dx_px, dy_px = compute_pixel_offset(rgb, bbox_color=[0, 0, 255])
    print(f"   Detected pixel offset: ({dx_px}, {dy_px})")
    
    # 3. Get depth (simulated)
    print("\n3. Depth Measurement:")
    depth = 0.6  # 60cm
    print(f"   Depth to object: {depth:.2f} meters")
    
    # 4. Convert to world coordinates
    print("\n4. World Coordinates:")
    dx_m, dy_m = pixel_to_world(dx_px, dy_px, depth, calib.fx, calib.fy)
    print(f"   World offset: ({dx_m:.3f}, {dy_m:.3f}) meters")
    print(f"   World offset: ({dx_m*100:.1f}, {dy_m*100:.1f}) cm")
    
    # 5. Generate control commands
    print("\n5. Robot Control Commands:")
    
    # Simple P-controller gains
    kp_linear = 0.5  # Linear velocity gain
    kp_angular = 2.0  # Angular velocity gain
    
    # Compute velocities
    vx = kp_linear * depth  # Forward velocity
    vy = -kp_linear * dx_m  # Lateral velocity (negative for correct direction)
    vz = -kp_linear * dy_m  # Vertical velocity
    omega = kp_angular * np.arctan2(dx_m, depth)  # Angular velocity
    
    print(f"   Linear velocity:")
    print(f"     vx (forward): {vx:.3f} m/s")
    print(f"     vy (lateral): {vy:.3f} m/s")
    print(f"     vz (vertical): {vz:.3f} m/s")
    print(f"   Angular velocity:")
    print(f"     omega (yaw): {np.degrees(omega):.1f} deg/s")
    
    # 6. Safety checks
    print("\n6. Safety Checks:")
    max_vel = 0.3  # m/s
    safe_vx = np.clip(vx, -max_vel, max_vel)
    safe_vy = np.clip(vy, -max_vel, max_vel)
    safe_vz = np.clip(vz, -max_vel, max_vel)
    
    print(f"   Clipped velocities (max {max_vel} m/s):")
    print(f"     vx: {safe_vx:.3f} m/s")
    print(f"     vy: {safe_vy:.3f} m/s")
    print(f"     vz: {safe_vz:.3f} m/s")


if __name__ == "__main__":
    # Run all tests
    print("PIXEL TO WORLD COORDINATE CONVERSION TEST SUITE")
    print("="*60)
    
    # Basic tests
    test_basic_conversion()
    test_inverse_conversion()
    test_different_focal_lengths()
    
    # Advanced tests
    test_with_camera_intrinsics()
    test_camera_calibration_class()
    
    # Practical examples
    practical_example()
    complete_pipeline_example()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    
    # Quick reference
    print("\nQUICK REFERENCE:")
    print("-"*40)
    print("""
# Basic usage:
from cogniforge.vision.vision_utils import pixel_to_world

dx_m, dy_m = pixel_to_world(
    dx_px=100,    # Pixel offset from center
    dy_px=-50,    
    depth=0.5,    # Depth in meters
    fx=600.0,     # Focal length (pixels)
    fy=600.0
)

# With camera calibration:
from cogniforge.vision.vision_utils import CameraCalibration

calib = CameraCalibration.from_fov(640, 480, fov_horizontal=60)
dx_m, dy_m = calib.pixel_to_world(dx_px, dy_px, depth)

# Complete pipeline:
1. Detect object → pixel offset (dx_px, dy_px)
2. Measure depth → depth in meters
3. Convert → world offset (dx_m, dy_m)
4. Control → move robot by (dx_m, dy_m) meters
    """)