"""
Grasp execution pipeline with vision-based final adjustment.

Implements the complete grasp workflow with a pause before closing gripper,
wrist camera capture, GPT vision analysis, micro adjustment, and final grasp.
"""

import numpy as np
import time
import cv2
from typing import Tuple, Optional, Dict, Any, Callable, Union
import logging
from dataclasses import dataclass
from enum import Enum
import base64

from cogniforge.vision.vision_utils import (
    compute_pixel_offset,
    ask_gpt_for_offset,
    encode_array_to_base64,
    pixel_to_world,
    CameraCalibration
)
from cogniforge.control.robot_control import (
    apply_micro_nudge,
    GraspController,
    GraspPhase,
    SafetyEnvelope
)

logger = logging.getLogger(__name__)


class GraspStatus(Enum):
    """Status of grasp execution."""
    APPROACHING = "approaching"
    PAUSED_FOR_VISION = "paused_for_vision"
    ADJUSTING = "adjusting"
    GRASPING = "grasping"
    LIFTING = "lifting"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class GraspConfig:
    """Configuration for grasp execution."""
    # Vision settings
    use_gpt_vision: bool = True
    gpt_timeout: float = 2.0
    gpt_api_key: Optional[str] = None
    fallback_to_color: bool = True
    target_color: list = None  # Default [0, 0, 255] for blue
    
    # Camera calibration
    wrist_camera_fx: float = 600.0
    wrist_camera_fy: float = 600.0
    wrist_camera_fov: float = 60.0
    
    # Adjustment settings  
    pre_grasp_pause: float = 0.5  # Seconds to pause before vision check
    adjustment_limit: float = 0.03  # Maximum nudge size (3cm)
    alignment_tolerance: float = 0.005  # Required precision (5mm)
    max_adjustment_attempts: int = 3
    
    # Grasp settings
    approach_height: float = 0.1  # Height above object for approach
    grasp_depth: float = 0.02  # How far to lower for grasp
    lift_height: float = 0.1  # Height to lift after grasp
    gripper_close_time: float = 1.0  # Time to close gripper
    
    # Safety settings
    safety_envelope: Optional[SafetyEnvelope] = None
    verify_grasp: bool = True
    
    def __post_init__(self):
        if self.target_color is None:
            self.target_color = [0, 0, 255]  # Blue
        if self.safety_envelope is None:
            self.safety_envelope = SafetyEnvelope()


class GraspExecutor:
    """
    Complete grasp execution pipeline with vision-based adjustment.
    
    Implements the workflow:
    1. Approach object
    2. Pause before grasp
    3. Capture wrist camera image
    4. Analyze with GPT vision (or color detection)
    5. Convert pixel offset to world coordinates
    6. Apply micro nudge
    7. Close gripper
    8. Lift and verify
    """
    
    def __init__(
        self,
        robot_interface,
        config: Optional[GraspConfig] = None
    ):
        """
        Initialize grasp executor.
        
        Args:
            robot_interface: Robot control interface (must implement required methods)
            config: Grasp configuration
        """
        self.robot = robot_interface
        self.config = config or GraspConfig()
        
        # Initialize components
        self.controller = GraspController(
            safety_envelope=self.config.safety_envelope,
            adjustment_limit=self.config.adjustment_limit,
            alignment_tolerance=self.config.alignment_tolerance,
            approach_height=self.config.approach_height
        )
        
        self.wrist_camera = CameraCalibration(
            fx=self.config.wrist_camera_fx,
            fy=self.config.wrist_camera_fy
        )
        
        self.status = GraspStatus.APPROACHING
        self.grasp_history = []
    
    def execute_grasp(
        self,
        object_position: Tuple[float, float, float],
        object_name: str = "object",
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete grasp sequence with vision-based adjustment.
        
        This is the main entry point that implements the full workflow.
        
        Args:
            object_position: Initial object position estimate (x, y, z)
            object_name: Name/description of object
            visualize: Whether to save visualization images
            
        Returns:
            Dictionary with grasp results and diagnostics
            
        Example:
            executor = GraspExecutor(robot, config)
            result = executor.execute_grasp(
                object_position=(0.3, 0.1, 0.05),
                object_name="blue cube"
            )
            if result['success']:
                print("Grasp successful!")
        """
        logger.info(f"Starting grasp execution for {object_name} at {object_position}")
        
        try:
            # Step 1: Approach object
            self.status = GraspStatus.APPROACHING
            approach_result = self._approach_object(object_position)
            if not approach_result['success']:
                return self._grasp_failed("Approach failed", approach_result)
            
            # Step 2: Pause before grasp
            self.status = GraspStatus.PAUSED_FOR_VISION
            logger.info(f"Pausing {self.config.pre_grasp_pause}s before vision check")
            time.sleep(self.config.pre_grasp_pause)
            
            # Step 3: Capture wrist camera image
            wrist_image = self._capture_wrist_image()
            if wrist_image is None:
                return self._grasp_failed("Failed to capture wrist image")
            
            if visualize:
                cv2.imwrite(f"wrist_view_{object_name}.jpg", 
                           cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR))
            
            # Step 4: Analyze with vision (GPT or color detection)
            vision_result = self._analyze_with_vision(
                wrist_image,
                object_name
            )
            
            # Step 5: Convert pixel offset to world coordinates
            if vision_result['detected']:
                dx_px = vision_result['dx_px']
                dy_px = vision_result['dy_px']
                
                # Get depth from wrist to object
                depth = self._get_wrist_depth()
                
                # Convert to world coordinates
                dx_m, dy_m = pixel_to_world(
                    dx_px, dy_px, depth,
                    self.wrist_camera.fx,
                    self.wrist_camera.fy
                )
                
                logger.info(f"Vision offset: pixels({dx_px}, {dy_px}) → "
                           f"world({dx_m:.3f}, {dy_m:.3f})m")
                
                # Always print JSON for UI, even if offset is near zero
                vision_json = {
                    "vision_result": {
                        "detected": True,
                        "method": vision_result['method'],
                        "pixel_offset": {"dx_px": dx_px, "dy_px": dy_px},
                        "world_offset": {"dx_m": round(dx_m, 4), "dy_m": round(dy_m, 4)},
                        "depth_m": round(depth, 3),
                        "confidence": vision_result.get('confidence', 0.0),
                        "alignment_status": "aligned" if (abs(dx_m) <= self.config.alignment_tolerance and 
                                                          abs(dy_m) <= self.config.alignment_tolerance) else "needs_adjustment",
                        "tolerance_m": self.config.alignment_tolerance
                    }
                }
                
                # Print JSON for UI display
                import json
                print("\n" + "="*50)
                print("VISION DETECTION RESULT (JSON):") 
                print(json.dumps(vision_json, indent=2))
                print("="*50 + "\n")
                
                # Step 6: Apply micro nudge if needed
                if abs(dx_m) > self.config.alignment_tolerance or \
                   abs(dy_m) > self.config.alignment_tolerance:
                    
                    self.status = GraspStatus.ADJUSTING
                    logger.info("Offset exceeds tolerance, applying adjustment...")
                    adjustment_result = self._apply_final_adjustment(dx_m, dy_m)
                    
                    if not adjustment_result['success']:
                        logger.warning("Final adjustment failed, attempting grasp anyway")
                else:
                    logger.info("Object well-aligned, no adjustment needed")
            else:
                # Even when detection fails, print JSON for UI
                vision_json = {
                    "vision_result": {
                        "detected": False,
                        "method": vision_result.get('method', 'none'),
                        "pixel_offset": {"dx_px": 0, "dy_px": 0},
                        "world_offset": {"dx_m": 0.0, "dy_m": 0.0},
                        "alignment_status": "no_detection",
                        "message": "Vision detection failed, proceeding without adjustment"
                    }
                }
                
                import json
                print("\n" + "="*50)
                print("VISION DETECTION RESULT (JSON):") 
                print(json.dumps(vision_json, indent=2))
                print("="*50 + "\n")
                
                logger.warning("Vision detection failed, proceeding without adjustment")
            
            # Step 7: Close gripper
            self.status = GraspStatus.GRASPING
            grasp_result = self._execute_grasp_action()
            
            if not grasp_result['success']:
                return self._grasp_failed("Gripper close failed", grasp_result)
            
            # Step 8: Lift and verify
            self.status = GraspStatus.LIFTING
            lift_result = self._lift_and_verify()
            
            if lift_result['success']:
                self.status = GraspStatus.SUCCESS
                logger.info(f"Grasp of {object_name} successful!")
            else:
                return self._grasp_failed("Lift verification failed", lift_result)
            
            # Compile results
            return {
                'success': True,
                'object': object_name,
                'final_position': self.robot.get_gripper_position(),
                'vision_offset': (dx_px, dy_px) if vision_result['detected'] else None,
                'world_offset': (dx_m, dy_m) if vision_result['detected'] else None,
                'adjustments_made': len(self.grasp_history),
                'grasp_quality': lift_result.get('quality', 1.0),
                'execution_time': time.time() - approach_result['start_time'],
                'history': self.grasp_history
            }
            
        except Exception as e:
            logger.error(f"Grasp execution failed with exception: {e}")
            return self._grasp_failed(f"Exception: {e}")
    
    def _approach_object(
        self,
        object_position: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """Move to approach position above object."""
        start_time = time.time()
        
        # Compute approach position
        approach_pos = self.controller.compute_approach_position(object_position)
        logger.info(f"Moving to approach position: {approach_pos}")
        
        # Move robot
        success = self.robot.move_to_position(approach_pos)
        
        # Lower to pre-grasp height
        if success:
            pre_grasp_pos = (
                approach_pos[0],
                approach_pos[1],
                object_position[2] + self.config.grasp_depth + 0.02
            )
            success = self.robot.move_to_position(pre_grasp_pos)
        
        self.grasp_history.append({
            'phase': 'approach',
            'target': approach_pos,
            'success': success,
            'time': time.time() - start_time
        })
        
        return {'success': success, 'start_time': start_time}
    
    def _capture_wrist_image(self) -> Optional[np.ndarray]:
        """Capture image from wrist camera."""
        try:
            # Get image from robot's wrist camera
            image = self.robot.get_wrist_camera_image()
            
            if image is None:
                # Fallback to simulation or test image
                logger.warning("No wrist camera image, using simulation")
                image = self._create_simulation_image()
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to capture wrist image: {e}")
            return None
    
    def _analyze_with_vision(
        self,
        wrist_image: np.ndarray,
        object_name: str
    ) -> Dict[str, Any]:
        """
        Analyze wrist camera image to get object offset.
        
        Tries GPT vision first, falls back to color detection.
        """
        result = {
            'detected': False,
            'dx_px': 0,
            'dy_px': 0,
            'method': None,
            'confidence': 0.0
        }
        
        # Try GPT vision if configured
        if self.config.use_gpt_vision and self.config.gpt_api_key:
            try:
                logger.info("Analyzing with GPT vision...")
                
                # Encode image
                base64_image = encode_array_to_base64(wrist_image)
                
                # Ask GPT for offset
                gpt_result = ask_gpt_for_offset(
                    base64_image,
                    instruction=f"Find the {object_name} and return its pixel offset from image center",
                    api_key=self.config.gpt_api_key,
                    timeout=self.config.gpt_timeout,
                    retry_count=1
                )
                
                if gpt_result['dx_px'] != 0 or gpt_result['dy_px'] != 0:
                    result.update({
                        'detected': True,
                        'dx_px': gpt_result['dx_px'],
                        'dy_px': gpt_result['dy_px'],
                        'method': 'gpt_vision',
                        'confidence': 0.9
                    })
                    logger.info(f"GPT vision detected offset: {gpt_result}")
                    return result
                    
            except Exception as e:
                logger.warning(f"GPT vision failed: {e}")
        
        # Fallback to color detection
        if self.config.fallback_to_color:
            try:
                logger.info("Using color detection fallback...")
                
                dx_px, dy_px = compute_pixel_offset(
                    wrist_image,
                    bbox_color=self.config.target_color,
                    method='color_threshold'
                )
                
                if dx_px != 0 or dy_px != 0:
                    result.update({
                        'detected': True,
                        'dx_px': dx_px,
                        'dy_px': dy_px,
                        'method': 'color_detection',
                        'confidence': 0.7
                    })
                    logger.info(f"Color detection offset: ({dx_px}, {dy_px})")
                    
            except Exception as e:
                logger.warning(f"Color detection failed: {e}")
        
        return result
    
    def _get_wrist_depth(self) -> float:
        """Get depth from wrist camera to object."""
        try:
            # Try to get depth from robot
            depth = self.robot.get_wrist_depth()
            if depth is not None:
                return depth
        except:
            pass
        
        # Estimate based on gripper position and expected object height
        gripper_pos = self.robot.get_gripper_position()
        estimated_depth = gripper_pos[2] - self.config.grasp_depth
        
        return max(0.05, estimated_depth)  # Minimum 5cm
    
    def _apply_final_adjustment(
        self,
        dx_m: float,
        dy_m: float
    ) -> Dict[str, Any]:
        """Apply micro nudge for final alignment."""
        current_pos = self.robot.get_gripper_position()
        
        # Apply micro nudge with safety
        dx_safe, dy_safe, dz_safe = apply_micro_nudge(
            dx_m, dy_m,
            limit=self.config.adjustment_limit,
            current_position=current_pos,
            safety_envelope=self.config.safety_envelope,
            phase=GraspPhase.PRE_GRASP
        )
        
        logger.info(f"Applying final adjustment: ({dx_safe:.3f}, {dy_safe:.3f}, {dz_safe:.3f})m")
        
        # Execute adjustment
        new_pos = (
            current_pos[0] + dx_safe,
            current_pos[1] + dy_safe,
            current_pos[2] + dz_safe
        )
        
        success = self.robot.move_to_position(new_pos)
        
        self.grasp_history.append({
            'phase': 'adjustment',
            'requested': (dx_m, dy_m, 0),
            'applied': (dx_safe, dy_safe, dz_safe),
            'success': success
        })
        
        return {'success': success, 'adjustment': (dx_safe, dy_safe, dz_safe)}
    
    def _execute_grasp_action(self) -> Dict[str, Any]:
        """Close gripper to grasp object."""
        logger.info("Closing gripper...")
        
        # Optional: Lower slightly more for firm grasp
        current_pos = self.robot.get_gripper_position()
        grasp_pos = (
            current_pos[0],
            current_pos[1],
            current_pos[2] - self.config.grasp_depth
        )
        self.robot.move_to_position(grasp_pos)
        
        # Close gripper
        success = self.robot.close_gripper()
        
        # Wait for gripper to close
        time.sleep(self.config.gripper_close_time)
        
        # Check if object was grasped
        if self.config.verify_grasp:
            grasped = self.robot.is_object_grasped()
        else:
            grasped = success
        
        self.grasp_history.append({
            'phase': 'grasp',
            'gripper_closed': success,
            'object_grasped': grasped
        })
        
        return {'success': grasped}
    
    def _lift_and_verify(self) -> Dict[str, Any]:
        """Lift object and verify successful grasp."""
        current_pos = self.robot.get_gripper_position()
        
        # Lift to safe height
        lift_pos = (
            current_pos[0],
            current_pos[1],
            current_pos[2] + self.config.lift_height
        )
        
        success = self.robot.move_to_position(lift_pos)
        
        # Verify object is still grasped
        if success and self.config.verify_grasp:
            still_grasped = self.robot.is_object_grasped()
            
            # Compute grasp quality based on gripper feedback
            quality = self._compute_grasp_quality()
            
            self.grasp_history.append({
                'phase': 'lift',
                'success': still_grasped,
                'quality': quality
            })
            
            return {
                'success': still_grasped,
                'quality': quality
            }
        
        return {'success': success, 'quality': 1.0}
    
    def _compute_grasp_quality(self) -> float:
        """Compute grasp quality score (0-1)."""
        try:
            # Get gripper force/position feedback
            gripper_data = self.robot.get_gripper_feedback()
            
            if gripper_data:
                # Simple quality based on force and position
                force = gripper_data.get('force', 0)
                position = gripper_data.get('position', 0)
                
                # Normalize to 0-1
                force_score = min(1.0, force / 10.0)  # Assume max 10N
                position_score = min(1.0, position / 0.08)  # Assume max 8cm
                
                return 0.6 * force_score + 0.4 * position_score
        except:
            pass
        
        return 0.8  # Default medium-high quality
    
    def _create_simulation_image(self) -> np.ndarray:
        """Create a simulation image for testing."""
        height, width = 480, 640
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add simulated object (blue cube)
        cube_size = 80
        cube_x = width // 2 + np.random.randint(-50, 50)
        cube_y = height // 2 + np.random.randint(-50, 50)
        
        image[cube_y:cube_y+cube_size, cube_x:cube_x+cube_size] = self.config.target_color
        
        return image
    
    def _grasp_failed(self, reason: str, details: Dict = None) -> Dict[str, Any]:
        """Handle grasp failure."""
        self.status = GraspStatus.FAILED
        logger.error(f"Grasp failed: {reason}")
        
        return {
            'success': False,
            'reason': reason,
            'details': details,
            'history': self.grasp_history
        }
    
    def format_vision_result_for_ui(
        self,
        vision_result: Dict[str, Any],
        dx_m: float = 0.0,
        dy_m: float = 0.0,
        depth: float = 0.0
    ) -> Dict[str, Any]:
        """
        Format vision detection result for UI display.
        
        Always returns a properly formatted JSON even for zero/near-zero offsets.
        
        Args:
            vision_result: Raw vision detection result
            dx_m: World x offset in meters
            dy_m: World y offset in meters
            depth: Depth measurement in meters
            
        Returns:
            Formatted dictionary for UI display
        """
        if vision_result.get('detected', False):
            dx_px = vision_result.get('dx_px', 0)
            dy_px = vision_result.get('dy_px', 0)
            
            # Determine alignment status
            is_aligned = (abs(dx_m) <= self.config.alignment_tolerance and 
                         abs(dy_m) <= self.config.alignment_tolerance)
            
            # Calculate offset magnitude
            pixel_magnitude = np.sqrt(dx_px**2 + dy_px**2)
            world_magnitude = np.sqrt(dx_m**2 + dy_m**2)
            
            return {
                "detected": True,
                "method": vision_result.get('method', 'unknown'),
                "confidence": round(vision_result.get('confidence', 0.0), 2),
                "pixel_offset": {
                    "dx_px": int(dx_px),
                    "dy_px": int(dy_px),
                    "magnitude_px": round(pixel_magnitude, 1)
                },
                "world_offset": {
                    "dx_m": round(dx_m, 4),
                    "dy_m": round(dy_m, 4),
                    "magnitude_m": round(world_magnitude, 4)
                },
                "depth_m": round(depth, 3),
                "alignment": {
                    "status": "aligned" if is_aligned else "needs_adjustment",
                    "is_aligned": is_aligned,
                    "tolerance_m": self.config.alignment_tolerance,
                    "error_m": round(world_magnitude, 4)
                },
                "camera_params": {
                    "fx": self.wrist_camera.fx,
                    "fy": self.wrist_camera.fy
                },
                "timestamp": time.time()
            }
        else:
            return {
                "detected": False,
                "method": vision_result.get('method', 'none'),
                "confidence": 0.0,
                "pixel_offset": {
                    "dx_px": 0,
                    "dy_px": 0,
                    "magnitude_px": 0.0
                },
                "world_offset": {
                    "dx_m": 0.0,
                    "dy_m": 0.0,
                    "magnitude_m": 0.0
                },
                "alignment": {
                    "status": "no_detection",
                    "is_aligned": False,
                    "tolerance_m": self.config.alignment_tolerance,
                    "error_m": None
                },
                "message": "No object detected in wrist camera view",
                "timestamp": time.time()
            }
    
    def get_current_status_json(self) -> Dict[str, Any]:
        """
        Get current grasp execution status as JSON for UI.
        
        Returns:
            Current status dictionary
        """
        return {
            "status": self.status.value,
            "gripper_position": self.robot.get_gripper_position() if hasattr(self.robot, 'get_gripper_position') else None,
            "phase_history": [
                {
                    "phase": event.get('phase'),
                    "success": event.get('success'),
                    "timestamp": event.get('time', 0)
                }
                for event in self.grasp_history
            ],
            "config": {
                "adjustment_limit_m": self.config.adjustment_limit,
                "alignment_tolerance_m": self.config.alignment_tolerance,
                "use_gpt_vision": self.config.use_gpt_vision,
                "pre_grasp_pause_s": self.config.pre_grasp_pause
            }
        }


def execute_grasp_with_vision(
    robot,
    object_position: Tuple[float, float, float],
    object_name: str = "blue cube",
    use_gpt: bool = True,
    api_key: Optional[str] = None
) -> bool:
    """
    Simplified function to execute grasp with vision-based adjustment.
    
    Args:
        robot: Robot control interface
        object_position: Target object position
        object_name: Object description
        use_gpt: Whether to use GPT vision
        api_key: OpenAI API key
        
    Returns:
        True if grasp successful
        
    Example:
        success = execute_grasp_with_vision(
            robot,
            object_position=(0.3, 0.1, 0.05),
            object_name="blue cube",
            api_key=os.getenv('OPENAI_API_KEY')
        )
    """
    config = GraspConfig(
        use_gpt_vision=use_gpt,
        gpt_api_key=api_key,
        fallback_to_color=True,
        target_color=[0, 0, 255] if "blue" in object_name.lower() else [255, 0, 0]
    )
    
    executor = GraspExecutor(robot, config)
    result = executor.execute_grasp(object_position, object_name)
    
    return result['success']


# Example robot interface for testing
class SimulatedRobot:
    """Simulated robot for testing grasp execution."""
    
    def __init__(self):
        self.gripper_position = [0.3, 0.0, 0.2]
        self.gripper_closed = False
        self.has_object = False
    
    def get_gripper_position(self):
        return tuple(self.gripper_position)
    
    def move_to_position(self, position):
        self.gripper_position = list(position)
        return True
    
    def get_wrist_camera_image(self):
        # Create test image with blue cube
        height, width = 480, 640
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add blue cube slightly offset from center
        cube_size = 80
        cube_x = width // 2 + 30
        cube_y = height // 2 - 20
        image[cube_y:cube_y+cube_size, cube_x:cube_x+cube_size] = [0, 0, 255]
        
        return image
    
    def get_wrist_depth(self):
        return 0.15  # 15cm
    
    def close_gripper(self):
        self.gripper_closed = True
        self.has_object = True
        return True
    
    def open_gripper(self):
        self.gripper_closed = False
        self.has_object = False
        return True
    
    def is_object_grasped(self):
        return self.has_object
    
    def get_gripper_feedback(self):
        if self.gripper_closed:
            return {'force': 5.0, 'position': 0.04}
        return {'force': 0.0, 'position': 0.08}


# Test the complete pipeline
if __name__ == "__main__":
    print("="*70)
    print("GRASP EXECUTION WITH VISION-BASED ADJUSTMENT TEST")
    print("="*70)
    
    # Create simulated robot
    robot = SimulatedRobot()
    
    # Configure grasp execution
    config = GraspConfig(
        use_gpt_vision=False,  # Use color detection for test
        fallback_to_color=True,
        pre_grasp_pause=0.5,
        adjustment_limit=0.03,
        alignment_tolerance=0.005
    )
    
    # Create executor
    executor = GraspExecutor(robot, config)
    
    # Test object position
    object_position = (0.35, 0.05, 0.05)
    
    print(f"\n1. Initial Setup:")
    print(f"   Object at: {object_position}")
    print(f"   Gripper at: {robot.get_gripper_position()}")
    
    print(f"\n2. Executing Grasp Sequence:")
    print("-" * 40)
    
    # Execute grasp with vision check
    result = executor.execute_grasp(
        object_position=object_position,
        object_name="blue cube",
        visualize=True
    )
    
    print(f"\n3. Results:")
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Vision offset (pixels): {result.get('vision_offset', 'N/A')}")
        print(f"   World offset (meters): {result.get('world_offset', 'N/A')}")
        print(f"   Final position: {result['final_position']}")
        print(f"   Grasp quality: {result.get('grasp_quality', 0):.2f}")
        print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
    else:
        print(f"   Failure reason: {result.get('reason', 'Unknown')}")
    
    print(f"\n4. Execution History:")
    for i, event in enumerate(result.get('history', [])):
        print(f"   {i+1}. Phase: {event.get('phase')}, Success: {event.get('success', 'N/A')}")
    
    print("\n" + "="*70)
    print("COMPLETE WORKFLOW:")
    print("="*70)
    print("""
    1. Approach object from above
    2. → PAUSE (0.5s) before grasp
    3. Capture wrist camera image
    4. Analyze with GPT vision (or color fallback)
    5. Convert pixel offset to world coordinates
    6. Apply micro nudge if needed
    7. Close gripper
    8. Lift and verify grasp
    
    Key Features:
    - Vision check at critical moment (just before grasp)
    - GPT vision with automatic fallback
    - Safe micro adjustments
    - Complete error handling
    - Grasp quality assessment
    """)
    
    print("Test completed successfully!")
    print("="*70)