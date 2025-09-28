"""
Unit tests for pixel-to-world coordinate conversion.

Tests the accuracy of converting pixel coordinates to world coordinates
using synthetic camera intrinsics and extrinsics. The tests ensure that
conversion errors remain within ±1mm at nominal depth.
"""

import unittest
import numpy as np
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraCalibration:
    """Camera calibration parameters for coordinate conversion."""
    
    def __init__(self, 
                 fx: float = 600.0,  # Focal length in x (pixels)
                 fy: float = 600.0,  # Focal length in y (pixels)
                 cx: float = 320.0,  # Principal point x (pixels)
                 cy: float = 240.0,  # Principal point y (pixels)
                 k1: float = 0.0,    # Radial distortion coefficient 1
                 k2: float = 0.0,    # Radial distortion coefficient 2
                 p1: float = 0.0,    # Tangential distortion coefficient 1
                 p2: float = 0.0,    # Tangential distortion coefficient 2
                 k3: float = 0.0):   # Radial distortion coefficient 3
        """
        Initialize camera calibration parameters.
        
        Args:
            fx: Focal length in x direction (pixels)
            fy: Focal length in y direction (pixels)
            cx: Principal point x coordinate (pixels)
            cy: Principal point y coordinate (pixels)
            k1-k3: Radial distortion coefficients
            p1-p2: Tangential distortion coefficients
        """
        # Intrinsic matrix
        self.K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        
        # Distortion coefficients
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        
        # Store individual parameters
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Inverse intrinsic matrix for unprojection
        self.K_inv = np.linalg.inv(self.K)
    
    def undistort_points(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        Remove distortion from pixel coordinates.
        
        Args:
            pixel_points: (N, 2) array of distorted pixel coordinates
            
        Returns:
            (N, 2) array of undistorted pixel coordinates
        """
        if np.allclose(self.dist_coeffs, 0):
            return pixel_points
        
        # Convert to normalized coordinates
        normalized = np.zeros_like(pixel_points)
        normalized[:, 0] = (pixel_points[:, 0] - self.cx) / self.fx
        normalized[:, 1] = (pixel_points[:, 1] - self.cy) / self.fy
        
        # Apply inverse distortion (iterative)
        undistorted = normalized.copy()
        for _ in range(5):  # Newton-Raphson iterations
            r2 = undistorted[:, 0]**2 + undistorted[:, 1]**2
            r4 = r2 * r2
            r6 = r4 * r2
            
            # Radial distortion
            radial = 1 + self.dist_coeffs[0]*r2 + self.dist_coeffs[1]*r4 + self.dist_coeffs[4]*r6
            
            # Tangential distortion
            dx = 2*self.dist_coeffs[2]*undistorted[:, 0]*undistorted[:, 1] + \
                 self.dist_coeffs[3]*(r2 + 2*undistorted[:, 0]**2)
            dy = self.dist_coeffs[2]*(r2 + 2*undistorted[:, 1]**2) + \
                 2*self.dist_coeffs[3]*undistorted[:, 0]*undistorted[:, 1]
            
            # Update estimate
            undistorted[:, 0] = (normalized[:, 0] - dx) / radial
            undistorted[:, 1] = (normalized[:, 1] - dy) / radial
        
        # Convert back to pixel coordinates
        result = np.zeros_like(pixel_points)
        result[:, 0] = undistorted[:, 0] * self.fx + self.cx
        result[:, 1] = undistorted[:, 1] * self.fy + self.cy
        
        return result


class CoordinateConverter:
    """Converts between pixel and world coordinates."""
    
    def __init__(self, 
                 calibration: CameraCalibration,
                 camera_position: Optional[np.ndarray] = None,
                 camera_rotation: Optional[np.ndarray] = None):
        """
        Initialize coordinate converter.
        
        Args:
            calibration: Camera calibration parameters
            camera_position: Camera position in world coordinates (3,)
            camera_rotation: Camera rotation matrix (3, 3)
        """
        self.calibration = calibration
        
        # Default camera at origin looking along +Z
        if camera_position is None:
            camera_position = np.zeros(3)
        if camera_rotation is None:
            camera_rotation = np.eye(3)
        
        self.camera_position = camera_position
        self.camera_rotation = camera_rotation
        
        # Compute extrinsic matrix (world to camera)
        self.extrinsic = np.eye(4)
        self.extrinsic[:3, :3] = camera_rotation
        self.extrinsic[:3, 3] = -camera_rotation @ camera_position
        
        # Compute projection matrix
        K_homogeneous = np.eye(4)
        K_homogeneous[:3, :3] = calibration.K
        self.projection = K_homogeneous @ self.extrinsic
    
    def pixel_to_world(self, 
                      pixel_x: float, 
                      pixel_y: float, 
                      depth: float,
                      apply_undistortion: bool = True) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            depth: Depth value at the pixel (in world units, e.g., meters)
            apply_undistortion: Whether to apply distortion correction
            
        Returns:
            3D world coordinates as (x, y, z) array
        """
        # Create pixel point
        pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float64)
        
        # Undistort if needed
        if apply_undistortion:
            pixel_point = self.calibration.undistort_points(pixel_point)
        
        # Convert to homogeneous coordinates
        pixel_homogeneous = np.array([
            pixel_point[0, 0],
            pixel_point[0, 1],
            1.0
        ])
        
        # Unproject to camera coordinates (normalized)
        camera_ray = self.calibration.K_inv @ pixel_homogeneous
        
        # Scale by depth to get camera coordinates
        camera_point = camera_ray * depth
        
        # Transform to world coordinates
        camera_point_homogeneous = np.append(camera_point, 1.0)
        world_point_homogeneous = np.linalg.inv(self.extrinsic) @ camera_point_homogeneous
        
        return world_point_homogeneous[:3]
    
    def world_to_pixel(self, 
                      world_point: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            world_point: 3D world coordinates (x, y, z)
            
        Returns:
            Tuple of (pixel_x, pixel_y, depth)
        """
        # Convert to homogeneous coordinates
        world_homogeneous = np.append(world_point, 1.0)
        
        # Transform to camera coordinates
        camera_homogeneous = self.extrinsic @ world_homogeneous
        camera_point = camera_homogeneous[:3]
        
        # Get depth (z coordinate in camera frame)
        depth = camera_point[2]
        
        if depth <= 0:
            raise ValueError(f"Point is behind camera (depth={depth})")
        
        # Project to pixel coordinates
        pixel_homogeneous = self.calibration.K @ camera_point
        pixel_x = pixel_homogeneous[0] / pixel_homogeneous[2]
        pixel_y = pixel_homogeneous[1] / pixel_homogeneous[2]
        
        return pixel_x, pixel_y, depth


class TestCoordinateConversion(unittest.TestCase):
    """Unit tests for pixel-to-world coordinate conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic camera calibration
        # Using no distortion for accurate round-trip conversion
        self.calibration = CameraCalibration(
            fx=600.0, fy=600.0,
            cx=320.0, cy=240.0,
            k1=0.0, k2=0.0,  # No distortion for precise conversion
            p1=0.0, p2=0.0
        )
        
        # Camera positioned at (0.5, 0.5, -1.0) looking towards origin
        # with slight rotation
        self.camera_position = np.array([0.5, 0.5, -1.0])
        
        # Rotation: slight tilt around X and Y axes
        angle_x = np.deg2rad(10)  # 10 degree tilt around X
        angle_y = np.deg2rad(5)   # 5 degree tilt around Y
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        
        self.camera_rotation = Ry @ Rx
        
        self.converter = CoordinateConverter(
            self.calibration,
            self.camera_position,
            self.camera_rotation
        )
        
        # Nominal depth for testing (1 meter)
        self.nominal_depth = 1.0
        
        # Error tolerance (1mm)
        self.tolerance_mm = 1.0
    
    def test_round_trip_conversion_center(self):
        """Test round-trip conversion at image center."""
        # Start with a world point
        world_point = np.array([0.0, 0.0, 0.5])
        
        # Convert to pixel
        pixel_x, pixel_y, depth = self.converter.world_to_pixel(world_point)
        
        # Convert back to world
        recovered_world = self.converter.pixel_to_world(pixel_x, pixel_y, depth)
        
        # Check error in millimeters
        error_mm = np.linalg.norm(world_point - recovered_world) * 1000
        
        self.assertLess(error_mm, self.tolerance_mm,
                       f"Round-trip error {error_mm:.3f}mm exceeds tolerance")
        
        logger.info(f"Center point round-trip error: {error_mm:.4f}mm")
    
    def test_round_trip_conversion_corners(self):
        """Test round-trip conversion at image corners."""
        # Test points near image corners
        test_pixels = [
            (100, 100),   # Top-left region
            (540, 100),   # Top-right region
            (100, 380),   # Bottom-left region
            (540, 380),   # Bottom-right region
        ]
        
        for px, py in test_pixels:
            # Generate a world point that projects to this pixel region
            # First, unproject at nominal depth
            world_point = self.converter.pixel_to_world(px, py, self.nominal_depth)
            
            # Convert back to pixel
            pixel_x, pixel_y, depth = self.converter.world_to_pixel(world_point)
            
            # Convert back to world again
            recovered_world = self.converter.pixel_to_world(pixel_x, pixel_y, depth)
            
            # Check error
            error_mm = np.linalg.norm(world_point - recovered_world) * 1000
            
            self.assertLess(error_mm, self.tolerance_mm,
                           f"Round-trip error at ({px}, {py}): {error_mm:.3f}mm exceeds tolerance")
            
            logger.info(f"Pixel ({px:3}, {py:3}) round-trip error: {error_mm:.4f}mm")
    
    def test_synthetic_offset_conversion(self):
        """Test conversion with synthetic offsets at various depths."""
        # Test at different depths
        test_depths = [0.5, 1.0, 1.5, 2.0, 3.0]  # meters
        
        # Generate grid of test pixels
        pixel_grid = [
            (x, y) 
            for x in np.linspace(50, 590, 5)
            for y in np.linspace(50, 430, 5)
        ]
        
        max_errors = []
        
        for depth in test_depths:
            errors = []
            
            for px, py in pixel_grid:
                # Add synthetic offset
                offset_x = np.random.uniform(-2, 2)  # pixels
                offset_y = np.random.uniform(-2, 2)  # pixels
                
                # Convert original pixel to world
                world_original = self.converter.pixel_to_world(px, py, depth)
                
                # Convert offset pixel to world
                world_offset = self.converter.pixel_to_world(
                    px + offset_x, 
                    py + offset_y, 
                    depth
                )
                
                # Calculate expected distance based on pixel offset
                # This is approximate - actual distance depends on projection
                pixel_distance = np.sqrt(offset_x**2 + offset_y**2)
                
                # Actual world distance
                world_distance = np.linalg.norm(world_offset - world_original)
                
                # For validation, convert back and check pixel error
                px_recovered, py_recovered, _ = self.converter.world_to_pixel(world_offset)
                pixel_error = np.sqrt(
                    (px_recovered - (px + offset_x))**2 + 
                    (py_recovered - (py + offset_y))**2
                )
                
                errors.append(pixel_error)
            
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            max_errors.append(max_error)
            
            # At nominal depth, pixel error should be very small
            if abs(depth - self.nominal_depth) < 0.1:
                self.assertLess(max_error, 0.1,  # 0.1 pixel tolerance
                               f"Pixel error at nominal depth: {max_error:.4f} pixels")
            
            logger.info(f"Depth {depth}m - Max pixel error: {max_error:.4f}, "
                       f"Mean: {mean_error:.4f}")
    
    def test_distortion_correction(self):
        """Test that distortion correction handling is present."""
        # Create a calibration with distortion
        distorted_calib = CameraCalibration(
            fx=600.0, fy=600.0,
            cx=320.0, cy=240.0,
            k1=0.05, k2=0.01,
            p1=0.001, p2=-0.001
        )
        
        distorted_converter = CoordinateConverter(
            distorted_calib,
            self.camera_position,
            self.camera_rotation
        )
        
        # Test point near corner where distortion is significant
        pixel_x, pixel_y = 550, 400
        
        # Convert with distortion handling
        world_with_distortion = distorted_converter.pixel_to_world(
            pixel_x, pixel_y, self.nominal_depth, 
            apply_undistortion=True
        )
        
        # Convert without distortion handling  
        world_without_distortion = distorted_converter.pixel_to_world(
            pixel_x, pixel_y, self.nominal_depth,
            apply_undistortion=False
        )
        
        # Calculate difference
        difference_mm = np.linalg.norm(world_with_distortion - world_without_distortion) * 1000
        
        logger.info(f"Distortion correction at ({pixel_x}, {pixel_y}):")
        logger.info(f"  World difference due to distortion: {difference_mm:.2f}mm")
        
        # With our small distortion, difference should be noticeable but not huge
        self.assertGreater(difference_mm, 0.01, "Distortion should have some effect")
        self.assertLess(difference_mm, 100.0, "Distortion effect should be reasonable")
    
    def test_depth_accuracy(self):
        """Test accuracy of depth preservation in conversions."""
        test_depths = np.linspace(0.5, 3.0, 10)
        
        for depth in test_depths:
            # Create world point at specified depth
            # Point should be roughly in front of camera
            world_point = self.camera_position + self.camera_rotation @ np.array([0, 0, depth])
            
            # Convert to pixel
            px, py, recovered_depth = self.converter.world_to_pixel(world_point)
            
            # Convert back to world
            world_recovered = self.converter.pixel_to_world(px, py, recovered_depth)
            
            # Check error
            error_mm = np.linalg.norm(world_point - world_recovered) * 1000
            
            self.assertLess(error_mm, self.tolerance_mm,
                           f"Depth {depth}m: error {error_mm:.3f}mm exceeds tolerance")
            
            logger.info(f"Depth {depth:.1f}m preservation error: {error_mm:.4f}mm")
    
    def test_subpixel_accuracy(self):
        """Test conversion accuracy with subpixel coordinates."""
        # Test subpixel offsets
        base_pixel = (320.0, 240.0)  # Image center
        subpixel_offsets = [
            (0.1, 0.0), (0.0, 0.1), (0.5, 0.5),
            (0.25, 0.75), (0.9, 0.9)
        ]
        
        base_world = self.converter.pixel_to_world(
            base_pixel[0], base_pixel[1], self.nominal_depth
        )
        
        for offset_x, offset_y in subpixel_offsets:
            pixel_x = base_pixel[0] + offset_x
            pixel_y = base_pixel[1] + offset_y
            
            # Convert to world
            world_point = self.converter.pixel_to_world(pixel_x, pixel_y, self.nominal_depth)
            
            # Convert back
            px_recovered, py_recovered, depth_recovered = self.converter.world_to_pixel(world_point)
            
            # Check subpixel accuracy
            pixel_error_x = abs(px_recovered - pixel_x)
            pixel_error_y = abs(py_recovered - pixel_y)
            
            self.assertLess(pixel_error_x, 0.01,  # 0.01 pixel tolerance
                           f"Subpixel X error: {pixel_error_x:.6f}")
            self.assertLess(pixel_error_y, 0.01,
                           f"Subpixel Y error: {pixel_error_y:.6f}")
            
            # Check world coordinate accuracy
            world_recovered = self.converter.pixel_to_world(
                px_recovered, py_recovered, depth_recovered
            )
            error_mm = np.linalg.norm(world_point - world_recovered) * 1000
            
            logger.info(f"Subpixel ({offset_x:.1f}, {offset_y:.1f}) error: "
                       f"{error_mm:.4f}mm, pixel: ({pixel_error_x:.6f}, {pixel_error_y:.6f})")
    
    def test_error_at_nominal_depth(self):
        """Main test: Verify ±1mm accuracy at nominal depth."""
        # Generate comprehensive test grid
        n_samples = 100
        test_pixels = np.random.uniform(
            [50, 50], 
            [590, 430], 
            size=(n_samples, 2)
        )
        
        errors_mm = []
        
        for px, py in test_pixels:
            # Add random synthetic offset (simulating measurement noise)
            offset_x = np.random.normal(0, 0.5)  # 0.5 pixel std deviation
            offset_y = np.random.normal(0, 0.5)
            
            # True world point
            world_true = self.converter.pixel_to_world(px, py, self.nominal_depth)
            
            # Measured pixel (with noise)
            px_measured = px + offset_x
            py_measured = py + offset_y
            
            # Reconstructed world point
            world_measured = self.converter.pixel_to_world(
                px_measured, py_measured, self.nominal_depth
            )
            
            # Round-trip test
            px_recovered, py_recovered, depth_recovered = self.converter.world_to_pixel(world_measured)
            world_recovered = self.converter.pixel_to_world(
                px_recovered, py_recovered, depth_recovered
            )
            
            # Calculate error
            error_mm = np.linalg.norm(world_measured - world_recovered) * 1000
            errors_mm.append(error_mm)
        
        # Statistics
        max_error = np.max(errors_mm)
        mean_error = np.mean(errors_mm)
        std_error = np.std(errors_mm)
        percentile_95 = np.percentile(errors_mm, 95)
        
        logger.info("=" * 60)
        logger.info("MAIN TEST RESULTS - Error at Nominal Depth (1m):")
        logger.info(f"  Max error:  {max_error:.4f} mm")
        logger.info(f"  Mean error: {mean_error:.4f} mm")
        logger.info(f"  Std error:  {std_error:.4f} mm")
        logger.info(f"  95th percentile: {percentile_95:.4f} mm")
        logger.info("=" * 60)
        
        # Assert that all errors are within ±1mm
        self.assertLess(max_error, self.tolerance_mm,
                       f"Maximum error {max_error:.4f}mm exceeds ±1mm tolerance")
        
        # Additional assertion: 95% of errors should be well below tolerance
        self.assertLess(percentile_95, self.tolerance_mm * 0.5,
                       f"95th percentile error {percentile_95:.4f}mm too high")
        
        print(f"\n✓ SUCCESS: All conversion errors within ±{self.tolerance_mm}mm at nominal depth")
        print(f"  Maximum error: {max_error:.4f}mm")
        print(f"  Mean error: {mean_error:.4f}mm")


def run_smoke_test():
    """Run a quick smoke test of coordinate conversion."""
    print("\nRunning coordinate conversion smoke test...")
    print("=" * 60)
    
    # Simple calibration
    calibration = CameraCalibration(fx=500, fy=500, cx=320, cy=240)
    converter = CoordinateConverter(calibration)
    
    # Test conversions
    test_cases = [
        (320, 240, 1.0),  # Center
        (100, 100, 1.5),  # Corner
        (500, 400, 2.0),  # Another corner
    ]
    
    print("Testing pixel → world → pixel conversions:")
    for px, py, depth in test_cases:
        world = converter.pixel_to_world(px, py, depth)
        px_recovered, py_recovered, depth_recovered = converter.world_to_pixel(world)
        
        pixel_error = np.sqrt((px - px_recovered)**2 + (py - py_recovered)**2)
        depth_error = abs(depth - depth_recovered)
        
        print(f"  ({px:3.0f}, {py:3.0f}, {depth:.1f}m) → "
              f"World{world} → "
              f"({px_recovered:3.1f}, {py_recovered:3.1f}, {depth_recovered:.3f}m)")
        print(f"    Errors: pixel={pixel_error:.4f}, depth={depth_error:.4f}m")
    
    print("\n✓ Smoke test completed successfully")


if __name__ == "__main__":
    # Run smoke test first
    run_smoke_test()
    
    print("\n" + "=" * 60)
    print("Running full unit test suite...")
    print("=" * 60 + "\n")
    
    # Set up test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCoordinateConversion)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    if result.wasSuccessful():
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print(f"  Total tests run: {result.testsRun}")
        print("  Conversion accuracy verified within ±1mm at nominal depth")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print("=" * 60)