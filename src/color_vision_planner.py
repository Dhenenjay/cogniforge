"""
Color-Based Vision Planner for Multi-Object Manipulation

This module provides:
1. Color detection and segmentation
2. Object selection by color
3. Vision-based planning with multiple colored objects
"""

import numpy as np
import pybullet as p
import cv2
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Color Definitions
# ============================================================================

class ObjectColor(Enum):
    """Standard object colors"""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    MAGENTA = "magenta"
    CYAN = "cyan"
    ORANGE = "orange"
    PURPLE = "purple"


@dataclass
class ColorRange:
    """HSV color range for detection"""
    lower: np.ndarray
    upper: np.ndarray
    rgb: List[float]  # RGB color for rendering
    

# Define HSV color ranges for detection
COLOR_RANGES = {
    ObjectColor.RED: ColorRange(
        lower=np.array([0, 120, 70]),
        upper=np.array([10, 255, 255]),
        rgb=[1.0, 0.0, 0.0, 1.0]
    ),
    ObjectColor.BLUE: ColorRange(
        lower=np.array([100, 120, 70]),
        upper=np.array([130, 255, 255]),
        rgb=[0.0, 0.0, 1.0, 1.0]
    ),
    ObjectColor.GREEN: ColorRange(
        lower=np.array([40, 50, 70]),
        upper=np.array([80, 255, 255]),
        rgb=[0.0, 1.0, 0.0, 1.0]
    ),
    ObjectColor.YELLOW: ColorRange(
        lower=np.array([20, 100, 100]),
        upper=np.array([30, 255, 255]),
        rgb=[1.0, 1.0, 0.0, 1.0]
    ),
    ObjectColor.MAGENTA: ColorRange(
        lower=np.array([140, 100, 100]),
        upper=np.array([170, 255, 255]),
        rgb=[1.0, 0.0, 1.0, 1.0]
    ),
    ObjectColor.CYAN: ColorRange(
        lower=np.array([80, 100, 100]),
        upper=np.array([100, 255, 255]),
        rgb=[0.0, 1.0, 1.0, 1.0]
    ),
    ObjectColor.ORANGE: ColorRange(
        lower=np.array([10, 100, 100]),
        upper=np.array([20, 255, 255]),
        rgb=[1.0, 0.5, 0.0, 1.0]
    ),
    ObjectColor.PURPLE: ColorRange(
        lower=np.array([130, 50, 50]),
        upper=np.array([140, 255, 255]),
        rgb=[0.5, 0.0, 0.5, 1.0]
    )
}


# ============================================================================
# Vision System
# ============================================================================

class ColorVisionSystem:
    """Vision system for color-based object detection"""
    
    def __init__(self, camera_config: Optional[Dict] = None):
        """
        Initialize vision system
        
        Args:
            camera_config: Camera configuration parameters
        """
        self.camera_config = camera_config or self._get_default_camera_config()
        self.last_image = None
        self.last_depth = None
        self.last_segmentation = None
        
    def _get_default_camera_config(self) -> Dict:
        """Get default camera configuration"""
        return {
            'width': 640,
            'height': 480,
            'fov': 60,
            'near': 0.1,
            'far': 10.0,
            'eye_position': [0.5, 0.0, 0.8],
            'target_position': [0.5, 0.0, 0.0],
            'up_vector': [0, 0, 1]
        }
        
    def capture_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture RGB, depth, and segmentation images from PyBullet
        
        Returns:
            (rgb_image, depth_image, segmentation_mask)
        """
        width = self.camera_config['width']
        height = self.camera_config['height']
        
        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_config['eye_position'],
            cameraTargetPosition=self.camera_config['target_position'],
            cameraUpVector=self.camera_config['up_vector']
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_config['fov'],
            aspect=width/height,
            nearVal=self.camera_config['near'],
            farVal=self.camera_config['far']
        )
        
        # Capture images
        _, _, rgb, depth, segmentation = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Process images
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        depth_array = np.array(depth).reshape((height, width))
        segmentation_array = np.array(segmentation).reshape((height, width))
        
        # Store for later use
        self.last_image = rgb_array
        self.last_depth = depth_array
        self.last_segmentation = segmentation_array
        
        return rgb_array, depth_array, segmentation_array
        
    def detect_color_objects(self, target_color: ObjectColor, 
                            min_area: int = 500) -> List[Dict[str, Any]]:
        """
        Detect objects of a specific color
        
        Args:
            target_color: Color to detect
            min_area: Minimum contour area to consider
            
        Returns:
            List of detected objects with properties
        """
        if self.last_image is None:
            self.capture_image()
            
        # Convert to HSV
        hsv = cv2.cvtColor(self.last_image, cv2.COLOR_RGB2HSV)
        
        # Get color range
        color_range = COLOR_RANGES[target_color]
        
        # Create mask
        mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
        
        # Handle red color wrap-around in HSV
        if target_color == ObjectColor.RED:
            # Red wraps around in HSV, so check both ranges
            upper_red = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask, upper_red)
            
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Get depth at center
            depth_value = self.last_depth[center_y, center_x] if self.last_depth is not None else None
            
            # Get 3D position (simplified - would need proper camera calibration)
            position_3d = self._pixel_to_3d(center_x, center_y, depth_value)
            
            detected_objects.append({
                'color': target_color,
                'bbox': (x, y, w, h),
                'center_2d': (center_x, center_y),
                'area': area,
                'depth': depth_value,
                'position_3d': position_3d,
                'confidence': min(1.0, area / 5000)  # Simple confidence based on size
            })
            
        return detected_objects
        
    def _pixel_to_3d(self, pixel_x: int, pixel_y: int, 
                     depth: Optional[float]) -> Optional[np.ndarray]:
        """
        Convert pixel coordinates to 3D position
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            depth: Depth value
            
        Returns:
            3D position or None
        """
        if depth is None:
            return None
            
        # Simplified 3D reconstruction (would need proper calibration)
        width = self.camera_config['width']
        height = self.camera_config['height']
        fov = self.camera_config['fov']
        
        # Normalize pixel coordinates
        nx = (pixel_x - width/2) / (width/2)
        ny = -(pixel_y - height/2) / (height/2)  # Flip Y
        
        # Approximate 3D position
        fov_rad = np.radians(fov)
        x = nx * depth * np.tan(fov_rad/2)
        y = ny * depth * np.tan(fov_rad/2) * (height/width)
        z = -depth  # Negative because depth is along -Z
        
        # Transform to world coordinates (simplified)
        camera_pos = np.array(self.camera_config['eye_position'])
        position_3d = camera_pos + np.array([x, y, z])
        
        return position_3d
        
    def get_color_mask(self, color: ObjectColor) -> np.ndarray:
        """
        Get binary mask for a specific color
        
        Args:
            color: Target color
            
        Returns:
            Binary mask
        """
        if self.last_image is None:
            self.capture_image()
            
        hsv = cv2.cvtColor(self.last_image, cv2.COLOR_RGB2HSV)
        color_range = COLOR_RANGES[color]
        
        mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
        
        # Special handling for red
        if color == ObjectColor.RED:
            upper_red = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask, upper_red)
            
        return mask
        
    def visualize_detection(self, target_colors: List[ObjectColor], 
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize color detection results
        
        Args:
            target_colors: Colors to detect
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        if self.last_image is None:
            self.capture_image()
            
        vis_image = self.last_image.copy()
        
        for color in target_colors:
            objects = self.detect_color_objects(color)
            
            for obj in objects:
                x, y, w, h = obj['bbox']
                center = obj['center_2d']
                
                # Draw bounding box
                color_rgb = COLOR_RANGES[color].rgb[:3]
                color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color_bgr, 2)
                
                # Draw center point
                cv2.circle(vis_image, center, 5, color_bgr, -1)
                
                # Add label
                label = f"{color.value} ({obj['confidence']:.2f})"
                cv2.putText(vis_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image


# ============================================================================
# Multi-Object Scene Manager
# ============================================================================

class MultiColorSceneManager:
    """Manages scenes with multiple colored objects"""
    
    def __init__(self, physics_client: Optional[int] = None):
        """
        Initialize scene manager
        
        Args:
            physics_client: PyBullet physics client
        """
        self.client = physics_client
        self.objects = {}
        self.robot_id = None
        self.platform_id = None
        
    def create_scene_with_colored_cubes(self, 
                                       cube_colors: List[ObjectColor],
                                       cube_positions: Optional[List[List[float]]] = None):
        """
        Create scene with multiple colored cubes
        
        Args:
            cube_colors: List of cube colors
            cube_positions: Optional list of positions
        """
        # Create ground plane
        plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        self.objects['plane'] = plane_id
        
        # Create robot
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Set initial joint positions
        initial_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
        for i, pos in enumerate(initial_joints):
            p.resetJointState(self.robot_id, i, pos)
            
        # Create platform (target)
        platform_size = [0.1, 0.1, 0.02]
        platform_pos = [0.3, 0.3, 0.01]
        
        platform_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=platform_size,
            rgbaColor=[0.2, 0.8, 0.2, 1]  # Green platform
        )
        platform_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=platform_size
        )
        
        self.platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=platform_pos
        )
        self.objects['platform'] = self.platform_id
        
        # Create colored cubes
        if cube_positions is None:
            # Default positions for cubes
            cube_positions = [
                [0.5, -0.1, 0.025],  # First cube
                [0.5, 0.1, 0.025],   # Second cube
            ]
            
        cube_size = 0.025  # 5cm cubes
        
        for i, (color, pos) in enumerate(zip(cube_colors, cube_positions)):
            # Get color RGB
            color_rgb = COLOR_RANGES[color].rgb
            
            # Create cube visual
            cube_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[cube_size]*3,
                rgbaColor=color_rgb
            )
            
            # Create cube collision
            cube_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[cube_size]*3
            )
            
            # Create cube body
            cube_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=cube_collision,
                baseVisualShapeIndex=cube_visual,
                basePosition=pos
            )
            
            # Set friction
            p.changeDynamics(cube_id, -1, lateralFriction=0.5)
            
            # Store cube
            cube_name = f"cube_{color.value}"
            self.objects[cube_name] = cube_id
            
        logger.info(f"Created scene with {len(cube_colors)} colored cubes")
        
    def get_object_by_color(self, color: ObjectColor) -> Optional[int]:
        """
        Get object ID by color
        
        Args:
            color: Target color
            
        Returns:
            Object ID or None
        """
        cube_name = f"cube_{color.value}"
        return self.objects.get(cube_name)


# ============================================================================
# Color-Based Planner
# ============================================================================

class ColorBasedPlanner:
    """Planner that selects and manipulates objects based on color"""
    
    def __init__(self, scene_manager: MultiColorSceneManager,
                 vision_system: ColorVisionSystem):
        """
        Initialize color-based planner
        
        Args:
            scene_manager: Scene manager
            vision_system: Vision system
        """
        self.scene = scene_manager
        self.vision = vision_system
        self.current_target_color = None
        self.grasp_height = 0.1
        self.place_height = 0.15
        
    def select_object_by_color(self, target_color: ObjectColor) -> Optional[Dict[str, Any]]:
        """
        Select an object to manipulate based on color
        
        Args:
            target_color: Target color to pick
            
        Returns:
            Selected object information or None
        """
        logger.info(f"Selecting {target_color.value} object")
        
        # Capture current image
        self.vision.capture_image()
        
        # Detect objects of target color
        detected_objects = self.vision.detect_color_objects(target_color)
        
        if not detected_objects:
            logger.warning(f"No {target_color.value} objects detected")
            return None
            
        # Select the object with highest confidence
        selected = max(detected_objects, key=lambda x: x['confidence'])
        
        # Get PyBullet object ID
        object_id = self.scene.get_object_by_color(target_color)
        
        if object_id is None:
            logger.error(f"Object {target_color.value} not found in scene")
            return None
            
        # Get actual position from PyBullet
        pos, orn = p.getBasePositionAndOrientation(object_id)
        
        selected['object_id'] = object_id
        selected['position'] = pos
        selected['orientation'] = orn
        
        self.current_target_color = target_color
        
        logger.info(f"Selected {target_color.value} object at position {pos}")
        
        return selected
        
    def compute_grasp_offset(self, color: ObjectColor) -> np.ndarray:
        """
        Compute grasp offset based on object color
        Different colors might require different grasp strategies
        
        Args:
            color: Object color
            
        Returns:
            Grasp offset vector
        """
        # Base offset
        base_offset = np.array([0, 0, 0.05])
        
        # Color-specific adjustments
        color_offsets = {
            ObjectColor.RED: np.array([0, 0, 0.01]),     # Slightly higher
            ObjectColor.BLUE: np.array([0, 0, -0.01]),   # Slightly lower
            ObjectColor.GREEN: np.array([0.01, 0, 0]),   # Slight x offset
            ObjectColor.YELLOW: np.array([-0.01, 0, 0]), # Negative x offset
        }
        
        # Apply color-specific offset
        specific_offset = color_offsets.get(color, np.zeros(3))
        total_offset = base_offset + specific_offset
        
        logger.info(f"Grasp offset for {color.value}: {total_offset}")
        
        return total_offset
        
    def plan_pick_and_place(self, target_color: ObjectColor,
                           place_position: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Plan pick and place for object of specific color
        
        Args:
            target_color: Color of object to pick
            place_position: Target placement position
            
        Returns:
            List of waypoints for execution
        """
        # Select object by color
        selected_object = self.select_object_by_color(target_color)
        
        if selected_object is None:
            logger.error("Failed to select object")
            return []
            
        # Get object position
        object_pos = selected_object['position']
        
        # Compute grasp offset based on color
        grasp_offset = self.compute_grasp_offset(target_color)
        
        # Default place position if not provided
        if place_position is None:
            platform_pos, _ = p.getBasePositionAndOrientation(self.scene.platform_id)
            place_position = [platform_pos[0], platform_pos[1], platform_pos[2] + 0.1]
            
        # Generate waypoints
        waypoints = []
        
        # 1. Move above object
        above_object = [
            object_pos[0] + grasp_offset[0],
            object_pos[1] + grasp_offset[1],
            object_pos[2] + self.grasp_height
        ]
        waypoints.append({
            'position': above_object,
            'gripper': 1.0,  # Open
            'description': f'Move above {target_color.value} cube'
        })
        
        # 2. Move down to grasp
        grasp_pos = [
            object_pos[0] + grasp_offset[0],
            object_pos[1] + grasp_offset[1],
            object_pos[2] + grasp_offset[2]
        ]
        waypoints.append({
            'position': grasp_pos,
            'gripper': 1.0,
            'description': f'Approach {target_color.value} cube'
        })
        
        # 3. Close gripper
        waypoints.append({
            'position': grasp_pos,
            'gripper': 0.0,  # Close
            'description': f'Grasp {target_color.value} cube'
        })
        
        # 4. Lift object
        waypoints.append({
            'position': above_object,
            'gripper': 0.0,
            'description': f'Lift {target_color.value} cube'
        })
        
        # 5. Move to place position
        above_place = [
            place_position[0],
            place_position[1],
            place_position[2] + self.place_height
        ]
        waypoints.append({
            'position': above_place,
            'gripper': 0.0,
            'description': f'Move {target_color.value} cube to platform'
        })
        
        # 6. Lower to place
        waypoints.append({
            'position': place_position,
            'gripper': 0.0,
            'description': f'Lower {target_color.value} cube'
        })
        
        # 7. Release
        waypoints.append({
            'position': place_position,
            'gripper': 1.0,
            'description': f'Release {target_color.value} cube'
        })
        
        # 8. Move up
        waypoints.append({
            'position': above_place,
            'gripper': 1.0,
            'description': 'Retract'
        })
        
        logger.info(f"Generated {len(waypoints)} waypoints for {target_color.value} cube")
        
        return waypoints
        
    def execute_plan(self, waypoints: List[Dict[str, Any]], 
                     speed: float = 0.01) -> bool:
        """
        Execute planned waypoints
        
        Args:
            waypoints: List of waypoints
            speed: Execution speed
            
        Returns:
            Success status
        """
        for i, waypoint in enumerate(waypoints):
            logger.info(f"Executing: {waypoint['description']}")
            
            # Move to position
            target_pos = waypoint['position']
            success = self._move_to_position(target_pos, speed)
            
            if not success:
                logger.error(f"Failed at waypoint {i}")
                return False
                
            # Control gripper
            gripper_state = waypoint['gripper']
            self._control_gripper(gripper_state)
            
            # Small delay
            time.sleep(0.5)
            
        logger.info("Plan execution complete")
        return True
        
    def _move_to_position(self, target_pos: List[float], 
                         speed: float = 0.01) -> bool:
        """
        Move end-effector to target position
        
        Args:
            target_pos: Target position
            speed: Movement speed
            
        Returns:
            Success status
        """
        # Simple IK-based movement (placeholder)
        # In practice, use proper IK solver
        
        for _ in range(100):  # Max iterations
            # Get current end-effector position
            ee_state = p.getLinkState(self.scene.robot_id, 7)
            current_pos = ee_state[0]
            
            # Calculate error
            error = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
            
            if error < 0.01:  # Threshold
                return True
                
            # Calculate joint velocities (simplified)
            # In practice, use Jacobian-based control
            joint_velocities = [0] * 7
            for i in range(3):
                diff = target_pos[i] - current_pos[i]
                joint_velocities[i] = np.clip(diff * 5, -speed, speed)
                
            # Apply velocities
            for i in range(7):
                p.setJointMotorControl2(
                    self.scene.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=joint_velocities[i]
                )
                
            p.stepSimulation()
            time.sleep(1/240)
            
        return False
        
    def _control_gripper(self, state: float):
        """
        Control gripper
        
        Args:
            state: 0 for closed, 1 for open
        """
        # Control gripper fingers
        gripper_range = 0.04
        target = state * gripper_range
        
        p.setJointMotorControl2(
            self.scene.robot_id,
            9,  # Left finger
            p.POSITION_CONTROL,
            targetPosition=target
        )
        p.setJointMotorControl2(
            self.scene.robot_id,
            10,  # Right finger
            p.POSITION_CONTROL,
            targetPosition=target
        )


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_color_based_manipulation():
    """Demonstrate color-based object manipulation"""
    
    print("\n" + "="*80)
    print(" COLOR-BASED MANIPULATION DEMO")
    print("="*80)
    
    # Connect to PyBullet
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    
    # Set camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0, 0.1]
    )
    
    # Create scene manager
    scene = MultiColorSceneManager(client)
    
    # Create scene with red and blue cubes
    cube_colors = [ObjectColor.RED, ObjectColor.BLUE]
    cube_positions = [
        [0.5, -0.15, 0.025],  # Red cube
        [0.5, 0.15, 0.025],   # Blue cube
    ]
    
    scene.create_scene_with_colored_cubes(cube_colors, cube_positions)
    
    # Create vision system
    vision = ColorVisionSystem()
    
    # Create planner
    planner = ColorBasedPlanner(scene, vision)
    
    # Wait for scene to settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1/240)
        
    print("\nScene created with RED and BLUE cubes")
    print("\n" + "-"*40)
    
    # Demo sequence
    sequences = [
        (ObjectColor.RED, "Picking RED cube"),
        (ObjectColor.BLUE, "Picking BLUE cube"),
    ]
    
    for color, description in sequences:
        print(f"\n{description}...")
        
        # Capture and visualize
        vision.capture_image()
        vis_image = vision.visualize_detection([ObjectColor.RED, ObjectColor.BLUE], 
                                              f"detection_{color.value}.png")
        print(f"  Detection saved to detection_{color.value}.png")
        
        # Plan for specific color
        waypoints = planner.plan_pick_and_place(color)
        
        if waypoints:
            print(f"  Generated {len(waypoints)} waypoints")
            
            # Execute plan
            success = planner.execute_plan(waypoints)
            
            if success:
                print(f"  ✅ Successfully manipulated {color.value} cube")
            else:
                print(f"  ❌ Failed to manipulate {color.value} cube")
                
        # Wait between tasks
        time.sleep(2)
        
    print("\n" + "="*80)
    print(" DEMO COMPLETE")
    print("="*80)
    
    # Keep simulation running
    print("\nPress Ctrl+C to exit")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        print("\nShutting down...")
        p.disconnect()


def test_color_detection():
    """Test color detection without robot"""
    
    print("\n" + "="*80)
    print(" COLOR DETECTION TEST")
    print("="*80)
    
    # Connect to PyBullet
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    
    # Create scene
    scene = MultiColorSceneManager(client)
    
    # Create multiple colored cubes
    colors = [
        ObjectColor.RED,
        ObjectColor.BLUE,
        ObjectColor.GREEN,
        ObjectColor.YELLOW,
        ObjectColor.MAGENTA,
        ObjectColor.CYAN
    ]
    
    positions = []
    for i, color in enumerate(colors):
        x = 0.3 + (i % 3) * 0.15
        y = -0.2 + (i // 3) * 0.15
        positions.append([x, y, 0.025])
        
    scene.create_scene_with_colored_cubes(colors, positions)
    
    # Create vision system
    vision = ColorVisionSystem()
    
    # Wait for scene to settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1/240)
        
    print(f"\nCreated {len(colors)} colored cubes")
    
    # Test detection for each color
    for color in colors:
        print(f"\nDetecting {color.value} objects...")
        
        vision.capture_image()
        objects = vision.detect_color_objects(color)
        
        if objects:
            for obj in objects:
                print(f"  Found {color.value} object:")
                print(f"    2D Center: {obj['center_2d']}")
                print(f"    Area: {obj['area']}")
                print(f"    Confidence: {obj['confidence']:.2f}")
                if obj['position_3d'] is not None:
                    print(f"    3D Position: {obj['position_3d']}")
        else:
            print(f"  No {color.value} objects detected")
            
    # Save visualization
    vision.visualize_detection(colors, "all_colors_detection.png")
    print("\n✅ Detection visualization saved to all_colors_detection.png")
    
    print("\n" + "="*80)
    print(" TEST COMPLETE")
    print("="*80)
    
    # Keep running
    print("\nPress Ctrl+C to exit")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        print("\nShutting down...")
        p.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Color-based vision and manipulation")
    parser.add_argument('--mode', choices=['demo', 'test', 'both'],
                       default='demo', help='Execution mode')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_color_based_manipulation()
    elif args.mode == 'test':
        test_color_detection()
    else:
        test_color_detection()
        input("\nPress Enter to continue to manipulation demo...")
        demo_color_based_manipulation()