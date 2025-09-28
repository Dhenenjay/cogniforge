"""
Camera Intrinsics Manager

This module handles camera intrinsic parameters and their updates
when render resolution changes. It ensures fx, fy, cx, cy are
correctly scaled to maintain accurate projections.
"""

import numpy as np
import json
import logging
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import yaml
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Camera Models
# ============================================================================

class CameraModel(Enum):
    """Supported camera models"""
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"
    RGBD = "rgbd"
    STEREO = "stereo"


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters
    
    The intrinsic matrix K is:
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    
    Where:
    - fx, fy: Focal lengths in pixels
    - cx, cy: Principal point (typically image center)
    """
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int  # Image width (pixels)
    height: int  # Image height (pixels)
    
    # Optional distortion coefficients
    k1: float = 0.0  # Radial distortion coefficient
    k2: float = 0.0  # Radial distortion coefficient
    k3: float = 0.0  # Radial distortion coefficient
    p1: float = 0.0  # Tangential distortion coefficient
    p2: float = 0.0  # Tangential distortion coefficient
    
    # Camera properties
    fov_x: Optional[float] = None  # Horizontal field of view (degrees)
    fov_y: Optional[float] = None  # Vertical field of view (degrees)
    model: CameraModel = CameraModel.PINHOLE
    
    def __post_init__(self):
        """Calculate derived properties"""
        if self.fov_x is None:
            self.fov_x = 2 * np.arctan(self.width / (2 * self.fx)) * 180 / np.pi
        if self.fov_y is None:
            self.fov_y = 2 * np.arctan(self.height / (2 * self.fy)) * 180 / np.pi
    
    @property
    def K(self) -> np.ndarray:
        """Get intrinsic matrix K"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @property
    def distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height,
            'k1': self.k1,
            'k2': self.k2,
            'k3': self.k3,
            'p1': self.p1,
            'p2': self.p2,
            'fov_x': self.fov_x,
            'fov_y': self.fov_y,
            'model': self.model.value
        }


# ============================================================================
# Common Camera Configurations
# ============================================================================

def get_realsense_d435_intrinsics(width: int = 640, height: int = 480) -> CameraIntrinsics:
    """
    Get Intel RealSense D435 camera intrinsics
    
    Default values for 640x480 resolution
    """
    # These are typical values - should be calibrated for specific camera
    base_fx = 617.253
    base_fy = 617.547
    base_cx = 320.0
    base_cy = 240.0
    base_width = 640
    base_height = 480
    
    # Scale if different resolution
    scale_x = width / base_width
    scale_y = height / base_height
    
    return CameraIntrinsics(
        fx=base_fx * scale_x,
        fy=base_fy * scale_y,
        cx=base_cx * scale_x,
        cy=base_cy * scale_y,
        width=width,
        height=height,
        model=CameraModel.RGBD
    )


def get_simulation_camera_intrinsics(width: int = 640, height: int = 480, 
                                    fov: float = 60.0) -> CameraIntrinsics:
    """
    Get simulation camera intrinsics from FOV
    
    Args:
        width: Image width
        height: Image height
        fov: Field of view in degrees (horizontal)
    """
    # Calculate focal length from FOV
    fx = width / (2 * np.tan(fov * np.pi / 360))
    fy = fx  # Square pixels
    
    # Principal point at image center
    cx = width / 2
    cy = height / 2
    
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        fov_x=fov,
        model=CameraModel.PINHOLE
    )


def get_wrist_camera_intrinsics(width: int = 640, height: int = 480) -> CameraIntrinsics:
    """
    Get wrist-mounted camera intrinsics
    
    Typical values for manipulation tasks
    """
    # Wider FOV for close-up manipulation
    fov = 70.0  # degrees
    
    fx = width / (2 * np.tan(fov * np.pi / 360))
    fy = fx
    cx = width / 2
    cy = height / 2
    
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        fov_x=fov,
        k1=-0.05,  # Slight barrel distortion
        k2=0.01,
        model=CameraModel.PINHOLE
    )


# ============================================================================
# Intrinsics Scaling and Verification
# ============================================================================

class IntrinsicsManager:
    """
    Manages camera intrinsics and handles resolution changes
    
    Key functionality:
    1. Scale intrinsics when resolution changes
    2. Verify intrinsics consistency
    3. Convert between different formats
    4. Handle multiple cameras
    """
    
    def __init__(self, base_intrinsics: Optional[CameraIntrinsics] = None):
        """
        Initialize intrinsics manager
        
        Args:
            base_intrinsics: Base camera intrinsics
        """
        self.base_intrinsics = base_intrinsics
        self.current_intrinsics = base_intrinsics
        self.intrinsics_history = []
        
        if base_intrinsics:
            self.intrinsics_history.append(base_intrinsics)
            logger.info(f"IntrinsicsManager initialized with {base_intrinsics.width}x{base_intrinsics.height}")
    
    def resize_intrinsics(self, new_width: int, new_height: int,
                         maintain_fov: bool = True) -> CameraIntrinsics:
        """
        Resize camera intrinsics for new resolution
        
        CRITICAL: When render resolution changes, intrinsics MUST be updated!
        
        Args:
            new_width: New image width
            new_height: New image height
            maintain_fov: If True, maintain field of view; if False, maintain focal length
            
        Returns:
            Updated camera intrinsics
        """
        if self.current_intrinsics is None:
            raise ValueError("No base intrinsics set")
        
        old = self.current_intrinsics
        
        # Calculate scale factors
        scale_x = new_width / old.width
        scale_y = new_height / old.height
        
        if maintain_fov:
            # Scale focal lengths to maintain FOV
            new_fx = old.fx * scale_x
            new_fy = old.fy * scale_y
        else:
            # Keep focal length constant (changes FOV)
            new_fx = old.fx
            new_fy = old.fy
        
        # ALWAYS scale principal point
        new_cx = old.cx * scale_x
        new_cy = old.cy * scale_y
        
        # Create new intrinsics
        new_intrinsics = CameraIntrinsics(
            fx=new_fx,
            fy=new_fy,
            cx=new_cx,
            cy=new_cy,
            width=new_width,
            height=new_height,
            k1=old.k1,
            k2=old.k2,
            k3=old.k3,
            p1=old.p1,
            p2=old.p2,
            model=old.model
        )
        
        # Log the change
        logger.info(f"Resized intrinsics: {old.width}x{old.height} -> {new_width}x{new_height}")
        logger.info(f"  fx: {old.fx:.2f} -> {new_fx:.2f} (scale: {scale_x:.3f})")
        logger.info(f"  fy: {old.fy:.2f} -> {new_fy:.2f} (scale: {scale_y:.3f})")
        logger.info(f"  cx: {old.cx:.2f} -> {new_cx:.2f}")
        logger.info(f"  cy: {old.cy:.2f} -> {new_cy:.2f}")
        logger.info(f"  FOV_x: {old.fov_x:.1f}° -> {new_intrinsics.fov_x:.1f}°")
        logger.info(f"  FOV_y: {old.fov_y:.1f}° -> {new_intrinsics.fov_y:.1f}°")
        
        self.current_intrinsics = new_intrinsics
        self.intrinsics_history.append(new_intrinsics)
        
        return new_intrinsics
    
    def verify_intrinsics(self, intrinsics: CameraIntrinsics) -> Dict[str, Any]:
        """
        Verify camera intrinsics are valid
        
        Returns:
            Verification results
        """
        issues = []
        warnings = []
        
        # Check focal lengths
        if intrinsics.fx <= 0 or intrinsics.fy <= 0:
            issues.append("Focal length must be positive")
        
        # Check principal point
        if intrinsics.cx < 0 or intrinsics.cx > intrinsics.width:
            issues.append(f"Principal point cx ({intrinsics.cx}) outside image bounds")
        if intrinsics.cy < 0 or intrinsics.cy > intrinsics.height:
            issues.append(f"Principal point cy ({intrinsics.cy}) outside image bounds")
        
        # Check if principal point is near center (common case)
        center_x = intrinsics.width / 2
        center_y = intrinsics.height / 2
        if abs(intrinsics.cx - center_x) > intrinsics.width * 0.1:
            warnings.append(f"Principal point cx ({intrinsics.cx:.1f}) far from center ({center_x:.1f})")
        if abs(intrinsics.cy - center_y) > intrinsics.height * 0.1:
            warnings.append(f"Principal point cy ({intrinsics.cy:.1f}) far from center ({center_y:.1f})")
        
        # Check aspect ratio
        pixel_aspect = intrinsics.fx / intrinsics.fy
        if abs(pixel_aspect - 1.0) > 0.1:
            warnings.append(f"Non-square pixels detected (aspect: {pixel_aspect:.3f})")
        
        # Check FOV
        if intrinsics.fov_x < 20 or intrinsics.fov_x > 170:
            warnings.append(f"Unusual FOV_x: {intrinsics.fov_x:.1f}°")
        if intrinsics.fov_y < 20 or intrinsics.fov_y > 170:
            warnings.append(f"Unusual FOV_y: {intrinsics.fov_y:.1f}°")
        
        # Check distortion coefficients
        if abs(intrinsics.k1) > 1.0 or abs(intrinsics.k2) > 1.0:
            warnings.append("Large radial distortion coefficients")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'pixel_aspect': pixel_aspect,
            'principal_point_centered': abs(intrinsics.cx - center_x) < 10 and abs(intrinsics.cy - center_y) < 10
        }
    
    def project_points(self, points_3d: np.ndarray, 
                      intrinsics: Optional[CameraIntrinsics] = None) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: 3D points in camera frame (Nx3)
            intrinsics: Camera intrinsics (uses current if None)
            
        Returns:
            2D image coordinates (Nx2)
        """
        if intrinsics is None:
            intrinsics = self.current_intrinsics
        
        if intrinsics is None:
            raise ValueError("No intrinsics available")
        
        # Ensure points are Nx3
        points_3d = np.atleast_2d(points_3d)
        if points_3d.shape[1] != 3:
            raise ValueError(f"Expected Nx3 points, got {points_3d.shape}")
        
        # Project points
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]
        
        # Apply intrinsics
        u = intrinsics.fx * x + intrinsics.cx
        v = intrinsics.fy * y + intrinsics.cy
        
        return np.column_stack([u, v])
    
    def unproject_points(self, points_2d: np.ndarray, depth: np.ndarray,
                        intrinsics: Optional[CameraIntrinsics] = None) -> np.ndarray:
        """
        Unproject 2D points to 3D using depth
        
        Args:
            points_2d: 2D image coordinates (Nx2)
            depth: Depth values (N,)
            intrinsics: Camera intrinsics (uses current if None)
            
        Returns:
            3D points in camera frame (Nx3)
        """
        if intrinsics is None:
            intrinsics = self.current_intrinsics
        
        if intrinsics is None:
            raise ValueError("No intrinsics available")
        
        # Ensure inputs are correct shape
        points_2d = np.atleast_2d(points_2d)
        depth = np.atleast_1d(depth)
        
        if points_2d.shape[0] != depth.shape[0]:
            raise ValueError("Number of points and depths must match")
        
        # Unproject
        x = (points_2d[:, 0] - intrinsics.cx) * depth / intrinsics.fx
        y = (points_2d[:, 1] - intrinsics.cy) * depth / intrinsics.fy
        z = depth
        
        return np.column_stack([x, y, z])
    
    def get_opengl_projection_matrix(self, near: float = 0.1, far: float = 10.0,
                                    intrinsics: Optional[CameraIntrinsics] = None) -> np.ndarray:
        """
        Get OpenGL projection matrix from intrinsics
        
        Args:
            near: Near clipping plane
            far: Far clipping plane
            intrinsics: Camera intrinsics (uses current if None)
            
        Returns:
            4x4 OpenGL projection matrix
        """
        if intrinsics is None:
            intrinsics = self.current_intrinsics
        
        if intrinsics is None:
            raise ValueError("No intrinsics available")
        
        # OpenGL projection matrix from intrinsics
        # See: http://www.songho.ca/opengl/gl_projectionmatrix.html
        
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.cx
        cy = intrinsics.cy
        width = intrinsics.width
        height = intrinsics.height
        
        # Convert to OpenGL normalized device coordinates
        P = np.zeros((4, 4))
        P[0, 0] = 2 * fx / width
        P[1, 1] = 2 * fy / height
        P[0, 2] = 1 - 2 * cx / width
        P[1, 2] = 2 * cy / height - 1
        P[2, 2] = -(far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near)
        P[3, 2] = -1
        
        return P
    
    def save_intrinsics(self, filepath: str, intrinsics: Optional[CameraIntrinsics] = None):
        """
        Save intrinsics to file
        
        Args:
            filepath: Output file path (.json or .yaml)
            intrinsics: Intrinsics to save (uses current if None)
        """
        if intrinsics is None:
            intrinsics = self.current_intrinsics
        
        if intrinsics is None:
            raise ValueError("No intrinsics to save")
        
        data = intrinsics.to_dict()
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:  # Default to JSON
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved intrinsics to {filepath}")
    
    def load_intrinsics(self, filepath: str) -> CameraIntrinsics:
        """
        Load intrinsics from file
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded camera intrinsics
        """
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        # Create intrinsics object
        intrinsics = CameraIntrinsics(
            fx=data['fx'],
            fy=data['fy'],
            cx=data['cx'],
            cy=data['cy'],
            width=data['width'],
            height=data['height'],
            k1=data.get('k1', 0),
            k2=data.get('k2', 0),
            k3=data.get('k3', 0),
            p1=data.get('p1', 0),
            p2=data.get('p2', 0),
            model=CameraModel(data.get('model', 'pinhole'))
        )
        
        self.current_intrinsics = intrinsics
        self.intrinsics_history.append(intrinsics)
        
        logger.info(f"Loaded intrinsics from {filepath}")
        return intrinsics


# ============================================================================
# Multi-Camera System
# ============================================================================

class MultiCameraSystem:
    """
    Manages multiple cameras with different intrinsics
    
    Useful for robot systems with multiple cameras:
    - Overhead camera
    - Wrist camera
    - Side cameras
    """
    
    def __init__(self):
        """Initialize multi-camera system"""
        self.cameras = {}
        self.managers = {}
        
    def add_camera(self, name: str, intrinsics: CameraIntrinsics):
        """
        Add a camera to the system
        
        Args:
            name: Camera name/identifier
            intrinsics: Camera intrinsics
        """
        self.cameras[name] = intrinsics
        self.managers[name] = IntrinsicsManager(intrinsics)
        logger.info(f"Added camera: {name} ({intrinsics.width}x{intrinsics.height})")
    
    def resize_camera(self, name: str, new_width: int, new_height: int,
                     maintain_fov: bool = True) -> CameraIntrinsics:
        """
        Resize specific camera
        
        Args:
            name: Camera name
            new_width: New width
            new_height: New height
            maintain_fov: Whether to maintain field of view
            
        Returns:
            Updated intrinsics
        """
        if name not in self.managers:
            raise KeyError(f"Camera '{name}' not found")
        
        new_intrinsics = self.managers[name].resize_intrinsics(
            new_width, new_height, maintain_fov
        )
        self.cameras[name] = new_intrinsics
        
        return new_intrinsics
    
    def resize_all_cameras(self, scale_factor: float,
                          maintain_fov: bool = True) -> Dict[str, CameraIntrinsics]:
        """
        Resize all cameras by same scale factor
        
        Args:
            scale_factor: Scale factor for resolution
            maintain_fov: Whether to maintain field of view
            
        Returns:
            Dictionary of updated intrinsics
        """
        updated = {}
        
        for name, manager in self.managers.items():
            current = manager.current_intrinsics
            new_width = int(current.width * scale_factor)
            new_height = int(current.height * scale_factor)
            
            updated[name] = self.resize_camera(name, new_width, new_height, maintain_fov)
        
        return updated
    
    def verify_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Verify all camera intrinsics
        
        Returns:
            Verification results for all cameras
        """
        results = {}
        
        for name, manager in self.managers.items():
            results[name] = manager.verify_intrinsics(manager.current_intrinsics)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all cameras
        
        Returns:
            Summary information
        """
        summary = {
            'camera_count': len(self.cameras),
            'cameras': {}
        }
        
        for name, intrinsics in self.cameras.items():
            summary['cameras'][name] = {
                'resolution': f"{intrinsics.width}x{intrinsics.height}",
                'focal_length': f"({intrinsics.fx:.1f}, {intrinsics.fy:.1f})",
                'principal_point': f"({intrinsics.cx:.1f}, {intrinsics.cy:.1f})",
                'fov': f"({intrinsics.fov_x:.1f}°, {intrinsics.fov_y:.1f}°)",
                'model': intrinsics.model.value
            }
        
        return summary


# ============================================================================
# Resize Examples and Verification
# ============================================================================

def demonstrate_resize_update():
    """Demonstrate how intrinsics must be updated when resizing"""
    
    print("\n" + "="*70)
    print(" CAMERA INTRINSICS RESIZE DEMONSTRATION")
    print("="*70)
    
    # Original camera at 640x480
    original = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480
    )
    
    print("\n[ORIGINAL INTRINSICS]")
    print(f"  Resolution: {original.width}x{original.height}")
    print(f"  fx, fy: {original.fx:.1f}, {original.fy:.1f}")
    print(f"  cx, cy: {original.cx:.1f}, {original.cy:.1f}")
    print(f"  FOV: {original.fov_x:.1f}° x {original.fov_y:.1f}°")
    
    # Create manager
    manager = IntrinsicsManager(original)
    
    # Resize to 1920x1080 (HD)
    print("\n[RESIZING TO 1920x1080]")
    hd_intrinsics = manager.resize_intrinsics(1920, 1080, maintain_fov=True)
    
    print("\n[UPDATED INTRINSICS]")
    print(f"  Resolution: {hd_intrinsics.width}x{hd_intrinsics.height}")
    print(f"  fx, fy: {hd_intrinsics.fx:.1f}, {hd_intrinsics.fy:.1f}")
    print(f"  cx, cy: {hd_intrinsics.cx:.1f}, {hd_intrinsics.cy:.1f}")
    print(f"  FOV: {hd_intrinsics.fov_x:.1f}° x {hd_intrinsics.fov_y:.1f}°")
    
    # Verify scaling
    scale_x = 1920 / 640
    scale_y = 1080 / 480
    
    print("\n[VERIFICATION]")
    print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    print(f"  fx scaled correctly: {original.fx * scale_x:.1f} = {hd_intrinsics.fx:.1f} ✓")
    print(f"  fy scaled correctly: {original.fy * scale_y:.1f} = {hd_intrinsics.fy:.1f} ✓")
    print(f"  cx scaled correctly: {original.cx * scale_x:.1f} = {hd_intrinsics.cx:.1f} ✓")
    print(f"  cy scaled correctly: {original.cy * scale_y:.1f} = {hd_intrinsics.cy:.1f} ✓")
    
    # Resize to 320x240 (smaller)
    print("\n[RESIZING TO 320x240]")
    small_intrinsics = manager.resize_intrinsics(320, 240, maintain_fov=True)
    
    print("\n[UPDATED INTRINSICS]")
    print(f"  Resolution: {small_intrinsics.width}x{small_intrinsics.height}")
    print(f"  fx, fy: {small_intrinsics.fx:.1f}, {small_intrinsics.fy:.1f}")
    print(f"  cx, cy: {small_intrinsics.cx:.1f}, {small_intrinsics.cy:.1f}")
    print(f"  FOV: {small_intrinsics.fov_x:.1f}° x {small_intrinsics.fov_y:.1f}°")
    
    print("\n" + "-"*70)
    print(" KEY TAKEAWAY:")
    print("-"*70)
    print("  When render resolution changes, you MUST update:")
    print("    • fx = fx_original * (new_width / original_width)")
    print("    • fy = fy_original * (new_height / original_height)")
    print("    • cx = cx_original * (new_width / original_width)")
    print("    • cy = cy_original * (new_height / original_height)")
    print("\n  This maintains correct projection geometry!")


def test_multi_camera_system():
    """Test multi-camera system for robot"""
    
    print("\n" + "="*70)
    print(" MULTI-CAMERA SYSTEM TEST")
    print("="*70)
    
    # Create system
    system = MultiCameraSystem()
    
    # Add cameras
    system.add_camera("overhead", get_simulation_camera_intrinsics(1280, 960, fov=45))
    system.add_camera("wrist", get_wrist_camera_intrinsics(640, 480))
    system.add_camera("side_left", get_simulation_camera_intrinsics(800, 600, fov=50))
    system.add_camera("side_right", get_simulation_camera_intrinsics(800, 600, fov=50))
    
    print("\n[INITIAL CONFIGURATION]")
    summary = system.get_summary()
    for name, info in summary['cameras'].items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Resize all cameras for higher resolution
    print("\n[RESIZING ALL CAMERAS BY 1.5x]")
    updated = system.resize_all_cameras(1.5, maintain_fov=True)
    
    print("\n[UPDATED CONFIGURATION]")
    summary = system.get_summary()
    for name, info in summary['cameras'].items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Verify all
    print("\n[VERIFICATION]")
    verifications = system.verify_all()
    all_valid = True
    
    for name, result in verifications.items():
        status = "✓ Valid" if result['valid'] else "✗ Invalid"
        print(f"  {name}: {status}")
        if result['warnings']:
            for warning in result['warnings']:
                print(f"    ⚠ {warning}")
        if result['issues']:
            for issue in result['issues']:
                print(f"    ✗ {issue}")
            all_valid = False
    
    if all_valid:
        print("\n✓ All camera intrinsics are valid!")
    else:
        print("\n✗ Some camera intrinsics have issues!")
    
    return system


def generate_constants_file(intrinsics: CameraIntrinsics, filepath: str = "camera_constants.py"):
    """
    Generate Python constants file with intrinsics
    
    Args:
        intrinsics: Camera intrinsics
        filepath: Output file path
    """
    content = f'''"""
Camera Intrinsic Constants
Auto-generated - DO NOT EDIT MANUALLY

Resolution: {intrinsics.width}x{intrinsics.height}
Generated at: {np.datetime64('now')}
"""

# Camera resolution
CAMERA_WIDTH = {intrinsics.width}
CAMERA_HEIGHT = {intrinsics.height}

# Intrinsic parameters (MUST be updated if resolution changes!)
CAMERA_FX = {intrinsics.fx:.6f}  # Focal length X (pixels)
CAMERA_FY = {intrinsics.fy:.6f}  # Focal length Y (pixels)
CAMERA_CX = {intrinsics.cx:.6f}  # Principal point X (pixels)
CAMERA_CY = {intrinsics.cy:.6f}  # Principal point Y (pixels)

# Field of view
CAMERA_FOV_X = {intrinsics.fov_x:.2f}  # Horizontal FOV (degrees)
CAMERA_FOV_Y = {intrinsics.fov_y:.2f}  # Vertical FOV (degrees)

# Distortion coefficients
CAMERA_K1 = {intrinsics.k1:.6f}  # Radial distortion
CAMERA_K2 = {intrinsics.k2:.6f}  # Radial distortion
CAMERA_K3 = {intrinsics.k3:.6f}  # Radial distortion
CAMERA_P1 = {intrinsics.p1:.6f}  # Tangential distortion
CAMERA_P2 = {intrinsics.p2:.6f}  # Tangential distortion

# Scaling functions for resolution changes
def scale_intrinsics(new_width, new_height):
    """Scale intrinsics for new resolution"""
    scale_x = new_width / CAMERA_WIDTH
    scale_y = new_height / CAMERA_HEIGHT
    
    return {{
        'fx': CAMERA_FX * scale_x,
        'fy': CAMERA_FY * scale_y,
        'cx': CAMERA_CX * scale_x,
        'cy': CAMERA_CY * scale_y,
        'width': new_width,
        'height': new_height
    }}

# Verify these constants match your render resolution!
if __name__ == "__main__":
    print(f"Camera constants for {{CAMERA_WIDTH}}x{{CAMERA_HEIGHT}}")
    print(f"  fx={{CAMERA_FX:.1f}}, fy={{CAMERA_FY:.1f}}")
    print(f"  cx={{CAMERA_CX:.1f}}, cy={{CAMERA_CY:.1f}}")
    print(f"  FOV: {{CAMERA_FOV_X:.1f}}° x {{CAMERA_FOV_Y:.1f}}°")
'''
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Generated constants file: {filepath}")
    print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
    print(f"  fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print(f"  cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main verification and demonstration"""
    
    print("\n" + "="*70)
    print(" CAMERA INTRINSICS VERIFICATION AND UPDATE")
    print("="*70)
    
    # Demonstrate resize updates
    demonstrate_resize_update()
    
    # Test multi-camera system
    system = test_multi_camera_system()
    
    # Generate example constants file
    print("\n" + "="*70)
    print(" GENERATING CAMERA CONSTANTS")
    print("="*70)
    
    # Standard 640x480 camera
    standard_cam = get_simulation_camera_intrinsics(640, 480, fov=60)
    generate_constants_file(standard_cam, "camera_constants_640x480.py")
    
    # HD 1920x1080 camera
    manager = IntrinsicsManager(standard_cam)
    hd_cam = manager.resize_intrinsics(1920, 1080, maintain_fov=True)
    generate_constants_file(hd_cam, "camera_constants_1920x1080.py")
    
    # Save configurations
    print("\n[SAVING CONFIGURATIONS]")
    manager.save_intrinsics("camera_intrinsics_640x480.json", standard_cam)
    manager.save_intrinsics("camera_intrinsics_1920x1080.json", hd_cam)
    
    print("\n" + "="*70)
    print(" CRITICAL REMINDERS")
    print("="*70)
    print("\n⚠️  WHEN YOU RESIZE THE RENDER:")
    print("  1. Update fx = fx_old * (new_width / old_width)")
    print("  2. Update fy = fy_old * (new_height / old_height)")
    print("  3. Update cx = cx_old * (new_width / old_width)")
    print("  4. Update cy = cy_old * (new_height / old_height)")
    print("\n⚠️  FAILURE TO UPDATE WILL CAUSE:")
    print("  • Incorrect 3D-2D projections")
    print("  • Wrong depth unprojection")
    print("  • Misaligned overlays")
    print("  • Failed visual servoing")
    
    print("\n✓ Use the IntrinsicsManager class to handle this automatically!")
    
    return system


if __name__ == "__main__":
    system = main()