"""
Camera Intrinsic Constants
Auto-generated - DO NOT EDIT MANUALLY

Resolution: 1920x1080
Generated at: 2025-09-28T05:57:50
"""

# Camera resolution
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Intrinsic parameters (MUST be updated if resolution changes!)
CAMERA_FX = 1662.768775  # Focal length X (pixels)
CAMERA_FY = 1247.076581  # Focal length Y (pixels)
CAMERA_CX = 960.000000  # Principal point X (pixels)
CAMERA_CY = 540.000000  # Principal point Y (pixels)

# Field of view
CAMERA_FOV_X = 60.00  # Horizontal FOV (degrees)
CAMERA_FOV_Y = 46.83  # Vertical FOV (degrees)

# Distortion coefficients
CAMERA_K1 = 0.000000  # Radial distortion
CAMERA_K2 = 0.000000  # Radial distortion
CAMERA_K3 = 0.000000  # Radial distortion
CAMERA_P1 = 0.000000  # Tangential distortion
CAMERA_P2 = 0.000000  # Tangential distortion

# Scaling functions for resolution changes
def scale_intrinsics(new_width, new_height):
    """Scale intrinsics for new resolution"""
    scale_x = new_width / CAMERA_WIDTH
    scale_y = new_height / CAMERA_HEIGHT
    
    return {
        'fx': CAMERA_FX * scale_x,
        'fy': CAMERA_FY * scale_y,
        'cx': CAMERA_CX * scale_x,
        'cy': CAMERA_CY * scale_y,
        'width': new_width,
        'height': new_height
    }

# Verify these constants match your render resolution!
if __name__ == "__main__":
    print(f"Camera constants for {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"  fx={CAMERA_FX:.1f}, fy={CAMERA_FY:.1f}")
    print(f"  cx={CAMERA_CX:.1f}, cy={CAMERA_CY:.1f}")
    print(f"  FOV: {CAMERA_FOV_X:.1f}° x {CAMERA_FOV_Y:.1f}°")
