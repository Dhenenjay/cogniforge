#!/usr/bin/env python3
"""
Unit Tests for Coordinate Transform Module

Tests pixel-to-world and world-to-pixel conversions with synthetic offsets
and asserts ±1 mm error tolerance at nominal depth.
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniforge.vision.coordinate_transform import (
    CameraIntrinsics, CameraExtrinsics, DistortionCoefficients,
    CoordinateTransform
)


class TestCoordinateTransform(unittest.TestCase):
    """Test cases for coordinate transformation with strict error tolerances."""
    
    def setUp(self):
        """Set up test fixtures with realistic camera parameters."""
        # Typical camera parameters (e.g., Intel RealSense D435)
        self.image_width = 1280
        self.image_height = 720
        
        # Intrinsics (typical values for RGB camera)
        self.fx = 615.0  # focal length in pixels
        self.fy = 615.0
        self.cx = 640.0  # principal point
        self.cy = 360.0
        
        self.intrinsics = CameraIntrinsics(
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            width=self.image_width,
            height=self.image_height
        )
        
        # Nominal depth for testing (1 meter)
        self.nominal_depth = 1.0
        
        # Error tolerance: ±1 mm at nominal depth
        self.world_error_tolerance_m = 0.001  # 1 mm
        self.pixel_error_tolerance = 1.0  # 1 pixel
    
    def test_identity_transform(self):
        """Test pixel to world conversion with identity extrinsics."""
        # Create transform with identity extrinsics (camera at world origin)
        transform = CoordinateTransform(self.intrinsics)
        
        # Test points at image center
        pixel_coords = np.array([[self.cx, self.cy]])
        depth = np.array([self.nominal_depth])
        
        # Convert to world
        world_coords = transform.pixel_to_world(pixel_coords, depth, distorted=False)
        
        # At image center with identity transform, world coords should be (0, 0, depth)
        expected = np.array([[0, 0, self.nominal_depth]])
        error = np.linalg.norm(world_coords - expected)
        
        self.assertLess(error, self.world_error_tolerance_m,
                       f"Identity transform error {error*1000:.3f} mm exceeds tolerance")
    
    def test_synthetic_offset_translations(self):
        """Test with synthetic camera translation offsets."""
        # Test various camera positions
        translations = [
            np.array([0.1, 0, 0]),      # 10cm offset in X
            np.array([0, 0.2, 0]),      # 20cm offset in Y
            np.array([0, 0, 0.5]),      # 50cm offset in Z
            np.array([0.1, -0.1, 0.2]), # Combined offset
        ]
        
        for translation in translations:
            with self.subTest(translation=translation):
                # Create extrinsics with translation only
                extrinsics = CameraExtrinsics(
                    rotation=np.eye(3),
                    translation=translation.reshape(3, 1)
                )
                
                transform = CoordinateTransform(self.intrinsics, extrinsics)
                
                # Generate test points in a grid pattern
                test_pixels, test_depths = self._generate_test_grid()
                
                # Convert to world and back
                world_coords = transform.pixel_to_world(test_pixels, test_depths, distorted=False)
                reprojected_pixels, reprojected_depths = transform.world_to_pixel(
                    world_coords, return_depth=True, apply_distortion=False
                )
                
                # Check pixel reprojection error
                pixel_errors = np.linalg.norm(reprojected_pixels - test_pixels, axis=1)
                max_pixel_error = np.max(pixel_errors)
                
                self.assertLess(max_pixel_error, self.pixel_error_tolerance,
                              f"Pixel reprojection error {max_pixel_error:.3f} exceeds tolerance")
                
                # Check depth consistency
                depth_errors = np.abs(reprojected_depths - test_depths)
                max_depth_error = np.max(depth_errors)
                
                self.assertLess(max_depth_error, self.world_error_tolerance_m,
                              f"Depth error {max_depth_error*1000:.3f} mm exceeds tolerance")
    
    def test_synthetic_rotations(self):
        """Test with synthetic camera rotation offsets."""
        # Test various rotation angles (in degrees)
        rotation_tests = [
            (30, 'x'),   # 30° rotation around X axis
            (45, 'y'),   # 45° rotation around Y axis
            (60, 'z'),   # 60° rotation around Z axis
            (15, 'xyz'), # Combined rotations
        ]
        
        for angle_deg, axis in rotation_tests:
            with self.subTest(angle=angle_deg, axis=axis):
                angle_rad = np.deg2rad(angle_deg)
                
                # Create rotation matrix
                if axis == 'x':
                    R = self._rotation_matrix_x(angle_rad)
                elif axis == 'y':
                    R = self._rotation_matrix_y(angle_rad)
                elif axis == 'z':
                    R = self._rotation_matrix_z(angle_rad)
                else:  # Combined
                    R = self._rotation_matrix_x(angle_rad/3) @ \
                        self._rotation_matrix_y(angle_rad/3) @ \
                        self._rotation_matrix_z(angle_rad/3)
                
                extrinsics = CameraExtrinsics(rotation=R, translation=np.zeros((3, 1)))
                transform = CoordinateTransform(self.intrinsics, extrinsics)
                
                # Test round-trip conversion
                test_pixels, test_depths = self._generate_test_grid(size=5)
                world_coords = transform.pixel_to_world(test_pixels, test_depths, distorted=False)
                reprojected_pixels, _ = transform.world_to_pixel(
                    world_coords, return_depth=True, apply_distortion=False
                )
                
                # Check pixel reprojection error
                pixel_errors = np.linalg.norm(reprojected_pixels - test_pixels, axis=1)
                max_pixel_error = np.max(pixel_errors)
                
                self.assertLess(max_pixel_error, self.pixel_error_tolerance,
                              f"Rotation {axis}={angle_deg}° pixel error {max_pixel_error:.3f} exceeds tolerance")
    
    def test_world_error_at_nominal_depth(self):
        """Test world coordinate error is within ±1mm at nominal depth."""
        # Create transform with some offset
        extrinsics = CameraExtrinsics(
            rotation=self._rotation_matrix_y(np.deg2rad(30)),
            translation=np.array([0.2, -0.1, 0.3]).reshape(3, 1)
        )
        transform = CoordinateTransform(self.intrinsics, extrinsics)
        
        # Generate world points at nominal depth
        world_points = self._generate_world_points_at_depth(self.nominal_depth, count=100)
        
        # Project to pixels
        pixel_coords, depths = transform.world_to_pixel(world_points, return_depth=True)
        
        # Reconstruct world coordinates
        reconstructed_world = transform.pixel_to_world(pixel_coords, depths, distorted=False)
        
        # Calculate errors in mm
        world_errors_mm = np.linalg.norm(reconstructed_world - world_points, axis=1) * 1000
        
        # Check all errors are within ±1mm
        max_error_mm = np.max(world_errors_mm)
        mean_error_mm = np.mean(world_errors_mm)
        
        self.assertLess(max_error_mm, 1.0,
                       f"Max world error {max_error_mm:.3f} mm exceeds 1mm tolerance at nominal depth")
        
        self.assertLess(mean_error_mm, 0.5,
                       f"Mean world error {mean_error_mm:.3f} mm too high at nominal depth")
        
        # Log statistics for debugging
        print(f"\nWorld reconstruction error at {self.nominal_depth}m depth:")
        print(f"  Max error:  {max_error_mm:.3f} mm")
        print(f"  Mean error: {mean_error_mm:.3f} mm")
        print(f"  Std error:  {np.std(world_errors_mm):.3f} mm")
    
    def test_distortion_correction(self):
        """Test distortion correction maintains accuracy."""
        # Create distortion coefficients (typical lens distortion)
        distortion = DistortionCoefficients(
            k1=-0.1,   # Barrel distortion
            k2=0.05,   # Higher order radial
            p1=0.001,  # Tangential
            p2=-0.001
        )
        
        transform = CoordinateTransform(self.intrinsics, distortion=distortion)
        
        # Generate test points
        test_pixels, test_depths = self._generate_test_grid(size=10)
        
        # Apply forward and inverse transformation with distortion
        world_coords = transform.pixel_to_world(test_pixels, test_depths, distorted=True)
        reprojected_pixels, _ = transform.world_to_pixel(
            world_coords, return_depth=True, apply_distortion=True
        )
        
        # Check round-trip error
        pixel_errors = np.linalg.norm(reprojected_pixels - test_pixels, axis=1)
        max_pixel_error = np.max(pixel_errors)
        
        self.assertLess(max_pixel_error, 2.0,  # Slightly higher tolerance with distortion
                       f"Distortion correction pixel error {max_pixel_error:.3f} exceeds tolerance")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        transform = CoordinateTransform(self.intrinsics)
        
        # Test 1: Points at image corners
        corners = np.array([
            [0, 0],
            [self.image_width-1, 0],
            [0, self.image_height-1],
            [self.image_width-1, self.image_height-1]
        ])
        corner_depths = np.ones(4) * self.nominal_depth
        
        world_corners = transform.pixel_to_world(corners, corner_depths, distorted=False)
        reprojected_corners, _ = transform.world_to_pixel(world_corners, return_depth=True)
        
        corner_errors = np.linalg.norm(reprojected_corners - corners, axis=1)
        self.assertTrue(np.all(corner_errors < self.pixel_error_tolerance),
                       "Corner points reprojection failed")
        
        # Test 2: Very close depth (10cm)
        close_depth = np.array([0.1])
        close_pixel = np.array([[self.cx, self.cy]])
        close_world = transform.pixel_to_world(close_pixel, close_depth, distorted=False)
        
        self.assertAlmostEqual(close_world[0, 2], 0.1, places=4,
                              msg="Close depth conversion failed")
        
        # Test 3: Far depth (10m)
        far_depth = np.array([10.0])
        far_pixel = np.array([[self.cx, self.cy]])
        far_world = transform.pixel_to_world(far_pixel, far_depth, distorted=False)
        
        self.assertAlmostEqual(far_world[0, 2], 10.0, places=3,
                              msg="Far depth conversion failed")
    
    def test_multi_depth_consistency(self):
        """Test consistency across multiple depth values."""
        depths = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])  # Various depths in meters
        
        for depth in depths:
            with self.subTest(depth=depth):
                # Create random extrinsics
                extrinsics = CameraExtrinsics(
                    rotation=self._rotation_matrix_y(np.deg2rad(25)),
                    translation=np.array([0.1, -0.05, 0.15]).reshape(3, 1)
                )
                transform = CoordinateTransform(self.intrinsics, extrinsics)
                
                # Generate points at this depth
                world_points = self._generate_world_points_at_depth(depth, count=50)
                
                # Round-trip test
                pixels, recovered_depths = transform.world_to_pixel(world_points, return_depth=True)
                reconstructed = transform.pixel_to_world(pixels, recovered_depths, distorted=False)
                
                # Calculate error in mm
                errors_mm = np.linalg.norm(reconstructed - world_points, axis=1) * 1000
                
                # Scale tolerance with depth (perspective projection uncertainty)
                # At 2x depth, allow 2x error, etc.
                scaled_tolerance_mm = 1.0 * (depth / self.nominal_depth)
                
                max_error = np.max(errors_mm)
                self.assertLess(max_error, scaled_tolerance_mm,
                              f"At depth {depth}m: max error {max_error:.3f}mm exceeds "
                              f"scaled tolerance {scaled_tolerance_mm:.3f}mm")
    
    def test_batch_processing(self):
        """Test batch processing of multiple points."""
        # Create transform with complex configuration
        extrinsics = CameraExtrinsics.from_look_at(
            eye=np.array([1, 1, 1]),
            target=np.array([0, 0, 0]),
            up=np.array([0, 0, 1])
        )
        distortion = DistortionCoefficients(k1=-0.05, k2=0.02)
        transform = CoordinateTransform(self.intrinsics, extrinsics, distortion)
        
        # Generate large batch of points
        n_points = 1000
        pixels = np.random.rand(n_points, 2) * np.array([self.image_width, self.image_height])
        depths = np.random.uniform(0.3, 3.0, n_points)
        
        # Process batch
        world_points = transform.pixel_to_world(pixels, depths)
        reprojected, recovered_depths = transform.world_to_pixel(world_points, return_depth=True)
        
        # Check shapes
        self.assertEqual(world_points.shape, (n_points, 3))
        self.assertEqual(reprojected.shape, (n_points, 2))
        self.assertEqual(recovered_depths.shape, (n_points,))
        
        # Check accuracy for valid points
        valid_mask = ~np.isnan(reprojected[:, 0])
        if np.any(valid_mask):
            pixel_errors = np.linalg.norm(reprojected[valid_mask] - pixels[valid_mask], axis=1)
            self.assertLess(np.median(pixel_errors), 2.0,
                          "Batch processing median error too high")
    
    def test_reprojection_error_metrics(self):
        """Test reprojection error calculation."""
        transform = CoordinateTransform(self.intrinsics)
        
        # Generate test data
        world_points = self._generate_world_points_at_depth(self.nominal_depth, count=50)
        pixel_points, depth_values = transform.world_to_pixel(world_points, return_depth=True)
        
        # Add small noise to simulate measurement error
        pixel_noise = np.random.randn(*pixel_points.shape) * 0.5  # 0.5 pixel std
        noisy_pixels = pixel_points + pixel_noise
        
        # Calculate errors
        stats = transform.get_reprojection_error(world_points, noisy_pixels, depth_values)
        
        # Check statistics are reasonable
        self.assertLess(stats['mean_pixel_error'], 1.0,
                       "Mean pixel error too high")
        self.assertLess(stats['mean_world_error_mm'], 2.0,
                       "Mean world error too high")
        
        # Log statistics
        print("\nReprojection error statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}")
    
    # Helper methods
    
    def _rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Create rotation matrix around X axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    
    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    
    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    
    def _generate_test_grid(self, size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Generate grid of test pixels with varying depths."""
        # Create grid of pixels
        x = np.linspace(100, self.image_width - 100, size)
        y = np.linspace(100, self.image_height - 100, size)
        xx, yy = np.meshgrid(x, y)
        pixels = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Vary depth around nominal value
        depths = self.nominal_depth + np.random.uniform(-0.1, 0.1, len(pixels))
        
        return pixels, depths
    
    def _generate_world_points_at_depth(self, depth: float, count: int = 100) -> np.ndarray:
        """Generate world points approximately at given depth from camera."""
        # Generate points in a plane perpendicular to camera axis
        x_range = depth * np.tan(np.deg2rad(30))  # ±30° field of view
        y_range = depth * np.tan(np.deg2rad(20))  # ±20° field of view
        
        x = np.random.uniform(-x_range, x_range, count)
        y = np.random.uniform(-y_range, y_range, count)
        z = np.full(count, depth) + np.random.uniform(-0.05, 0.05, count)  # Small depth variation
        
        return np.column_stack([x, y, z])


class TestCameraCalibration(unittest.TestCase):
    """Test camera calibration parameter classes."""
    
    def test_intrinsics_from_fov(self):
        """Test creating intrinsics from field of view."""
        fov_x = np.deg2rad(60)  # 60° horizontal FOV
        fov_y = np.deg2rad(45)  # 45° vertical FOV
        width = 1920
        height = 1080
        
        intrinsics = CameraIntrinsics.from_fov(fov_x, fov_y, width, height)
        
        # Check focal lengths
        expected_fx = width / (2 * np.tan(fov_x / 2))
        expected_fy = height / (2 * np.tan(fov_y / 2))
        
        self.assertAlmostEqual(intrinsics.fx, expected_fx, places=2)
        self.assertAlmostEqual(intrinsics.fy, expected_fy, places=2)
        
        # Check principal point (should be image center)
        self.assertAlmostEqual(intrinsics.cx, width / 2)
        self.assertAlmostEqual(intrinsics.cy, height / 2)
    
    def test_extrinsics_from_look_at(self):
        """Test creating extrinsics from look-at parameters."""
        eye = np.array([2, 2, 2])
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        extrinsics = CameraExtrinsics.from_look_at(eye, target, up)
        
        # Check that rotation matrix is orthogonal
        R = extrinsics.rotation
        should_be_identity = R @ R.T
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)
        
        # Check determinant is 1 (proper rotation)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)
    
    def test_extrinsics_inverse(self):
        """Test extrinsics inverse transformation."""
        # Create arbitrary extrinsics
        R = self._create_rotation_matrix(30, 45, 60)
        t = np.array([1, 2, 3]).reshape(3, 1)
        extrinsics = CameraExtrinsics(R, t)
        
        # Get inverse
        inverse = extrinsics.inverse
        
        # Check that applying both gives identity
        combined = extrinsics.matrix @ inverse.matrix
        np.testing.assert_allclose(combined, np.eye(4), atol=1e-10)
    
    def _create_rotation_matrix(self, roll_deg, pitch_deg, yaw_deg):
        """Create rotation matrix from Euler angles."""
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        
        # Roll (X), Pitch (Y), Yaw (Z)
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
        
        return Rz @ Ry @ Rx


def run_tests():
    """Run all coordinate transform tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestCameraCalibration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("COORDINATE TRANSFORM TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Pixel-to-world conversion accurate to ±1mm at nominal depth.")
    else:
        print("\n❌ Some tests failed. Review errors above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)