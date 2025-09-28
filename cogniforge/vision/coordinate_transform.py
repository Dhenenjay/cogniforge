"""
Coordinate Transform Module for CogniForge

Handles pixel-to-world and world-to-pixel coordinate transformations
with camera calibration and depth information.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length in x (pixels)
    fy: float  # Focal length in y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int  # Image width
    height: int  # Image height
    
    @property
    def matrix(self) -> np.ndarray:
        """Get intrinsic matrix K."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> 'CameraIntrinsics':
        """Create from intrinsic matrix."""
        return cls(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=width,
            height=height
        )
    
    @classmethod
    def from_fov(cls, fov_x: float, fov_y: float, width: int, height: int) -> 'CameraIntrinsics':
        """Create from field of view angles (in radians)."""
        fx = width / (2 * np.tan(fov_x / 2))
        fy = height / (2 * np.tan(fov_y / 2))
        cx = width / 2
        cy = height / 2
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (pose in world)."""
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    
    def __post_init__(self):
        """Validate and reshape arrays."""
        self.rotation = np.array(self.rotation).reshape(3, 3)
        self.translation = np.array(self.translation).reshape(3, 1)
        
        # Validate rotation matrix (should be orthogonal)
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-6):
            logger.warning("Rotation matrix is not orthogonal, normalizing...")
            # Use SVD to find nearest orthogonal matrix
            U, _, Vt = np.linalg.svd(self.rotation)
            self.rotation = U @ Vt
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation.flatten()
        return T
    
    @property
    def inverse(self) -> 'CameraExtrinsics':
        """Get inverse transformation (world to camera)."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return CameraExtrinsics(R_inv, t_inv)
    
    @classmethod
    def from_look_at(cls, eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> 'CameraExtrinsics':
        """Create extrinsics from look-at parameters."""
        eye = np.array(eye).reshape(3)
        target = np.array(target).reshape(3)
        
        if up is None:
            up = np.array([0, 0, 1])  # Default up is +Z
        else:
            up = np.array(up).reshape(3)
        
        # Compute camera axes
        z_axis = (eye - target) / np.linalg.norm(eye - target)  # Camera looks along -Z
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Rotation matrix (world to camera)
        R = np.array([x_axis, y_axis, z_axis])
        
        # Translation
        t = -R @ eye
        
        return cls(R, t.reshape(3, 1))


@dataclass
class DistortionCoefficients:
    """Camera distortion coefficients."""
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    k3: float = 0.0  # Radial distortion coefficient 3
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2
    
    @property
    def array(self) -> np.ndarray:
        """Get distortion coefficients as array."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


class CoordinateTransform:
    """
    Handles coordinate transformations between pixel and world coordinates.
    
    Coordinate Systems:
    - Pixel: (u, v) in image plane, origin at top-left
    - Camera: (x, y, z) with camera at origin, z pointing forward
    - World: (X, Y, Z) in world coordinates
    """
    
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: Optional[CameraExtrinsics] = None,
        distortion: Optional[DistortionCoefficients] = None
    ):
        """
        Initialize coordinate transformer.
        
        Args:
            intrinsics: Camera intrinsic parameters
            extrinsics: Camera extrinsic parameters (pose)
            distortion: Distortion coefficients
        """
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics or CameraExtrinsics(np.eye(3), np.zeros((3, 1)))
        self.distortion = distortion or DistortionCoefficients()
        
        # Cache frequently used matrices
        self._K = self.intrinsics.matrix
        self._K_inv = np.linalg.inv(self._K)
        self._update_projection_matrix()
    
    def _update_projection_matrix(self):
        """Update cached projection matrix."""
        if self.extrinsics:
            # P = K * [R | t]
            Rt = np.hstack([self.extrinsics.rotation, self.extrinsics.translation])
            self._P = self._K @ Rt
        else:
            self._P = self._K
    
    def pixel_to_world(
        self,
        pixel_coords: np.ndarray,
        depth: np.ndarray,
        distorted: bool = True
    ) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            pixel_coords: (N, 2) array of pixel coordinates (u, v)
            depth: (N,) array of depth values in meters
            distorted: Whether input pixels are distorted
            
        Returns:
            (N, 3) array of world coordinates (X, Y, Z)
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        pixel_coords = np.atleast_2d(pixel_coords)
        depth = np.atleast_1d(depth)
        
        if pixel_coords.shape[1] != 2:
            raise ValueError(f"pixel_coords must have shape (N, 2), got {pixel_coords.shape}")
        
        if depth.shape[0] != pixel_coords.shape[0]:
            raise ValueError(f"depth must have same length as pixel_coords")
        
        # Undistort pixels if needed
        if distorted and (self.distortion.k1 != 0 or self.distortion.k2 != 0):
            pixel_coords = self.undistort_points(pixel_coords)
        
        # Convert to normalized camera coordinates
        # [u, v, 1]^T = K * [x/z, y/z, 1]^T
        pixels_homogeneous = np.hstack([pixel_coords, np.ones((len(pixel_coords), 1))])
        rays = (self._K_inv @ pixels_homogeneous.T).T
        
        # Scale by depth to get camera coordinates
        camera_coords = rays * depth[:, np.newaxis]
        
        # Transform to world coordinates
        if self.extrinsics:
            # Need to transform from camera to world
            # X_world = R^T * (X_cam - t)
            R_inv = self.extrinsics.rotation.T
            t = self.extrinsics.translation.flatten()
            world_coords = (R_inv @ (camera_coords - t).T).T
        else:
            world_coords = camera_coords
        
        return world_coords
    
    def world_to_pixel(
        self,
        world_coords: np.ndarray,
        return_depth: bool = False,
        apply_distortion: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            world_coords: (N, 3) array of world coordinates (X, Y, Z)
            return_depth: Whether to return depth values
            apply_distortion: Whether to apply distortion
            
        Returns:
            pixel_coords: (N, 2) array of pixel coordinates (u, v)
            depth: (N,) array of depth values (if return_depth=True)
        """
        world_coords = np.atleast_2d(world_coords)
        
        if world_coords.shape[1] != 3:
            raise ValueError(f"world_coords must have shape (N, 3), got {world_coords.shape}")
        
        # Transform to camera coordinates
        if self.extrinsics:
            camera_coords = (self.extrinsics.rotation @ world_coords.T + 
                           self.extrinsics.translation).T
        else:
            camera_coords = world_coords
        
        # Get depth (z-coordinate in camera frame)
        depth = camera_coords[:, 2]
        
        # Avoid division by zero
        valid_mask = depth > 1e-6
        
        # Project to normalized image plane
        normalized_coords = np.zeros((len(camera_coords), 2))
        normalized_coords[valid_mask] = camera_coords[valid_mask, :2] / depth[valid_mask, np.newaxis]
        
        # Apply distortion if needed
        if apply_distortion and (self.distortion.k1 != 0 or self.distortion.k2 != 0):
            normalized_coords = self.distort_points(normalized_coords)
        
        # Convert to pixel coordinates
        pixels_homogeneous = self._K @ np.hstack([normalized_coords, np.ones((len(normalized_coords), 1))]).T
        pixel_coords = pixels_homogeneous[:2, :].T
        
        # Mark invalid pixels
        pixel_coords[~valid_mask] = np.nan
        
        if return_depth:
            return pixel_coords, depth
        return pixel_coords, None
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Remove distortion from pixel coordinates.
        
        Args:
            points: (N, 2) array of distorted pixel coordinates
            
        Returns:
            (N, 2) array of undistorted pixel coordinates
        """
        if self.distortion is None:
            return points
        
        # Convert to normalized coordinates
        normalized = (points - np.array([self.intrinsics.cx, self.intrinsics.cy])) / \
                     np.array([self.intrinsics.fx, self.intrinsics.fy])
        
        # Apply iterative undistortion
        undistorted = normalized.copy()
        for _ in range(5):  # Fixed-point iteration
            r2 = undistorted[:, 0]**2 + undistorted[:, 1]**2
            radial = 1 + self.distortion.k1 * r2 + self.distortion.k2 * r2**2 + self.distortion.k3 * r2**3
            
            tangential_x = 2 * self.distortion.p1 * undistorted[:, 0] * undistorted[:, 1] + \
                          self.distortion.p2 * (r2 + 2 * undistorted[:, 0]**2)
            tangential_y = self.distortion.p1 * (r2 + 2 * undistorted[:, 1]**2) + \
                          2 * self.distortion.p2 * undistorted[:, 0] * undistorted[:, 1]
            
            undistorted[:, 0] = (normalized[:, 0] - tangential_x) / radial
            undistorted[:, 1] = (normalized[:, 1] - tangential_y) / radial
        
        # Convert back to pixel coordinates
        return undistorted * np.array([self.intrinsics.fx, self.intrinsics.fy]) + \
               np.array([self.intrinsics.cx, self.intrinsics.cy])
    
    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply distortion to normalized coordinates.
        
        Args:
            points: (N, 2) array of undistorted normalized coordinates
            
        Returns:
            (N, 2) array of distorted normalized coordinates
        """
        if self.distortion is None:
            return points
        
        r2 = points[:, 0]**2 + points[:, 1]**2
        radial = 1 + self.distortion.k1 * r2 + self.distortion.k2 * r2**2 + self.distortion.k3 * r2**3
        
        tangential_x = 2 * self.distortion.p1 * points[:, 0] * points[:, 1] + \
                      self.distortion.p2 * (r2 + 2 * points[:, 0]**2)
        tangential_y = self.distortion.p1 * (r2 + 2 * points[:, 1]**2) + \
                      2 * self.distortion.p2 * points[:, 0] * points[:, 1]
        
        distorted = points.copy()
        distorted[:, 0] = points[:, 0] * radial + tangential_x
        distorted[:, 1] = points[:, 1] * radial + tangential_y
        
        return distorted
    
    def get_reprojection_error(
        self,
        world_points: np.ndarray,
        pixel_points: np.ndarray,
        depth_values: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate reprojection error statistics.
        
        Args:
            world_points: (N, 3) array of world coordinates
            pixel_points: (N, 2) array of observed pixel coordinates
            depth_values: Optional (N,) array of depth values for reverse check
            
        Returns:
            Dictionary with error statistics
        """
        # Forward projection error (world to pixel)
        projected_pixels, projected_depths = self.world_to_pixel(world_points, return_depth=True)
        pixel_errors = np.linalg.norm(projected_pixels - pixel_points, axis=1)
        
        stats = {
            'mean_pixel_error': np.nanmean(pixel_errors),
            'max_pixel_error': np.nanmax(pixel_errors),
            'std_pixel_error': np.nanstd(pixel_errors),
            'median_pixel_error': np.nanmedian(pixel_errors)
        }
        
        # Reverse projection error if depth is provided
        if depth_values is not None:
            reconstructed_world = self.pixel_to_world(pixel_points, depth_values)
            world_errors = np.linalg.norm(reconstructed_world - world_points, axis=1) * 1000  # Convert to mm
            
            stats.update({
                'mean_world_error_mm': np.mean(world_errors),
                'max_world_error_mm': np.max(world_errors),
                'std_world_error_mm': np.std(world_errors),
                'median_world_error_mm': np.median(world_errors)
            })
        
        return stats