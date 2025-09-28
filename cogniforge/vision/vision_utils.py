"""
Vision utilities for object detection and tracking.

Provides fast, reliable computer vision methods for detecting objects
by color, template matching, and computing pixel offsets.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any, Union
import logging
from dataclasses import dataclass
from scipy import ndimage
import warnings
import base64
import json
import requests
import os
from openai import OpenAI
import time

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of object detection."""
    center: Tuple[int, int]  # (x, y) pixel coordinates
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # 0-1 confidence score
    mask: Optional[np.ndarray] = None  # Binary mask of detected object
    area: Optional[int] = None  # Pixel area of detection
    
    @property
    def pixel_offset(self) -> Tuple[int, int]:
        """Get pixel offset from image center."""
        return self.center


def ask_gpt_for_offset(
    base64_image: str,
    instruction: str = "Find the blue cube and return its pixel offset from image center",
    api_key: Optional[str] = None,
    model: str = "gpt-5",
    max_tokens: int = 100,
    temperature: float = 0.1,
    timeout: float = 10.0,
    retry_count: int = 2
) -> Dict[str, int]:
    """
    Ask GPT Vision to find object and return pixel offset as JSON.
    
    Uses GPT-4 Vision API to analyze image and return the pixel offset
    of the target object from the image center.
    
    Args:
        base64_image: Base64 encoded image string (without data:image prefix)
        instruction: What to detect (e.g., "Find the blue cube")
        api_key: OpenAI API key (uses env var OPENAI_API_KEY if None)
        model: GPT vision model to use
        max_tokens: Maximum response tokens
        temperature: Sampling temperature (lower = more deterministic)
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure
        
    Returns:
        Dictionary with keys 'dx_px' and 'dy_px' representing pixel offset
        from image center. Returns {"dx_px": 0, "dy_px": 0} on failure.
        
    Example:
        # Encode image to base64
        with open('robot_view.jpg', 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Get offset from GPT
        offset = ask_gpt_for_offset(
            base64_image,
            "Find the blue cube and return its pixel offset"
        )
        print(f"Offset: x={offset['dx_px']}, y={offset['dy_px']}")
    """
    # Get API key
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("No OpenAI API key found, returning zero offset")
            return {"dx_px": 0, "dy_px": 0}
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return {"dx_px": 0, "dy_px": 0}
    
    # Craft the prompt for reliable JSON output
    system_prompt = """You are a precise computer vision system that outputs only JSON.
    Analyze images and return pixel offsets from the image center.
    The image center is at (width/2, height/2).
    Positive dx_px means the object is to the RIGHT of center.
    Positive dy_px means the object is BELOW center.
    Always respond with ONLY a JSON object in this exact format:
    {"dx_px": <integer>, "dy_px": <integer>}
    """
    
    user_prompt = f"""{instruction}
    
    Analyze this image and return the pixel offset of the target object from the image center.
    Output ONLY a JSON object with keys 'dx_px' and 'dy_px' (integers).
    If the object is not found, return {"dx_px": 0, "dy_px": 0}.
    """
    
    # Prepare the API request (Responses API)
    inputs = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]
    
    # Try to get response with retries
    for attempt in range(retry_count):
        try:
            response = client.responses.create(
                model=model,
                input=inputs,
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            # Extract the response text (handle different client versions)
            response_text = None
            if hasattr(response, "output_text"):
                response_text = response.output_text
            else:
                try:
                    # Fallback parsing from structured output
                    chunks = []
                    for item in getattr(response, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            if hasattr(c, "text") and c.text:
                                chunks.append(c.text)
                    response_text = "\n".join(chunks) if chunks else None
                except Exception:
                    response_text = None
            if not response_text and hasattr(response, "choices"):
                # Very defensive fallback if server returned chat-like payload
                try:
                    response_text = response.choices[0].message.content.strip()
                except Exception:
                    response_text = None
            if not response_text:
                raise ValueError("Empty response from model")
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                # Validate the response has required keys
                if 'dx_px' in result and 'dy_px' in result:
                    # Ensure values are integers
                    result['dx_px'] = int(result['dx_px'])
                    result['dy_px'] = int(result['dy_px'])
                    
                    if attempt > 0:
                        logger.info(f"GPT Vision succeeded on attempt {attempt + 1}")
                    
                    return result
                else:
                    logger.warning(f"GPT response missing required keys: {result}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse GPT response as JSON: {response_text}")
                
                # Try to extract numbers from text as fallback
                import re
                numbers = re.findall(r'-?\d+', response_text)
                if len(numbers) >= 2:
                    return {"dx_px": int(numbers[0]), "dy_px": int(numbers[1])}
                
        except requests.Timeout:
            logger.warning(f"GPT Vision request timeout (attempt {attempt + 1}/{retry_count})")
            
        except Exception as e:
            logger.error(f"GPT Vision API error (attempt {attempt + 1}/{retry_count}): {e}")
        
        # Wait before retry
        if attempt < retry_count - 1:
            time.sleep(1.0 * (attempt + 1))  # Exponential backoff
    
    # All attempts failed
    logger.warning("All GPT Vision attempts failed, returning zero offset")
    return {"dx_px": 0, "dy_px": 0}


def ask_gpt_for_offset_simple(
    base64_image: str,
    instruction: str = "Find the blue cube",
    api_key: Optional[str] = None
) -> Dict[str, int]:
    """
    Simplified version of ask_gpt_for_offset with minimal parameters.
    
    Args:
        base64_image: Base64 encoded image
        instruction: What to find
        api_key: Optional API key
        
    Returns:
        {"dx_px": x_offset, "dy_px": y_offset}
        
    Example:
        offset = ask_gpt_for_offset_simple(base64_img, "Find the red ball")
    """
    return ask_gpt_for_offset(
        base64_image,
        instruction,
        api_key,
        model="gpt-5",
        max_tokens=50,
        temperature=0.0,
        timeout=5.0,
        retry_count=1
    )


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string (without data prefix)
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_array_to_base64(image_array: np.ndarray) -> str:
    """
    Encode a numpy array (RGB image) to base64 string.
    
    Args:
        image_array: RGB image as numpy array
        
    Returns:
        Base64 encoded string (without data prefix)
    """
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_array
    
    # Encode to JPEG
    success, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    
    # Convert to base64
    return base64.b64encode(buffer).decode('utf-8')


def hybrid_object_detection(
    rgb: np.ndarray,
    use_gpt: bool = True,
    gpt_timeout: float = 2.0,
    target_color: List[int] = [0, 0, 255],
    instruction: str = "Find the blue cube",
    api_key: Optional[str] = None
) -> Tuple[int, int]:
    """
    Hybrid detection using GPT Vision with fast fallback to color detection.
    
    Tries GPT Vision first for more intelligent detection, but falls back
    to fast color-based detection if GPT is slow or unavailable.
    
    Args:
        rgb: RGB image as numpy array
        use_gpt: Whether to try GPT Vision first
        gpt_timeout: Timeout for GPT Vision in seconds
        target_color: Fallback color for detection
        instruction: Instruction for GPT Vision
        api_key: OpenAI API key
        
    Returns:
        (dx_px, dy_px): Pixel offset from image center
        
    Example:
        # Will try GPT first, then fallback to color detection
        offset_x, offset_y = hybrid_object_detection(
            rgb_image,
            use_gpt=True,
            gpt_timeout=2.0,
            instruction="Find the blue cube"
        )
    """
    if use_gpt and api_key:
        try:
            # Try GPT Vision with timeout
            start_time = time.time()
            
            # Encode image
            base64_image = encode_array_to_base64(rgb)
            
            # Ask GPT with short timeout
            result = ask_gpt_for_offset(
                base64_image,
                instruction,
                api_key,
                timeout=gpt_timeout,
                retry_count=1
            )
            
            elapsed = time.time() - start_time
            
            # Check if we got a valid response
            if result['dx_px'] != 0 or result['dy_px'] != 0:
                logger.info(f"GPT Vision succeeded in {elapsed:.2f}s")
                return (result['dx_px'], result['dy_px'])
            
            logger.info(f"GPT Vision found no object in {elapsed:.2f}s, using color detection")
            
        except Exception as e:
            logger.warning(f"GPT Vision failed: {e}, falling back to color detection")
    
    # Fallback to fast color detection
    offset = compute_pixel_offset(rgb, target_color)
    logger.debug(f"Color detection offset: {offset}")
    
    return offset


def pixel_to_world(
    dx_px: float,
    dy_px: float,
    depth: float,
    fx: float,
    fy: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None
) -> Tuple[float, float]:
    """
    Convert pixel offset to world coordinates using camera intrinsics.
    
    Uses pinhole camera model to convert pixel displacement from image center
    to metric displacement in world coordinates at a given depth.
    
    Args:
        dx_px: Horizontal pixel offset from image center (positive = right)
        dy_px: Vertical pixel offset from image center (positive = down)
        depth: Distance to object in meters
        fx: Focal length in x-direction (pixels)
        fy: Focal length in y-direction (pixels)
        cx: Principal point x-coordinate (defaults to image center if None)
        cy: Principal point y-coordinate (defaults to image center if None)
        
    Returns:
        (dx_m, dy_m): World coordinate offsets in meters
        dx_m: Horizontal offset in meters (positive = right)
        dy_m: Vertical offset in meters (positive = down)
        
    Example:
        # Typical camera parameters
        fx = 600.0  # pixels
        fy = 600.0  # pixels
        depth = 0.5  # 50cm to object
        
        # Convert pixel offset to world
        dx_m, dy_m = pixel_to_world(100, -50, depth, fx, fy)
        print(f"World offset: ({dx_m:.3f}, {dy_m:.3f}) meters")
        
    Note:
        The pinhole camera model relates pixel coordinates to world coordinates:
        x_world = (x_pixel - cx) * depth / fx
        y_world = (y_pixel - cy) * depth / fy
        
        Since we're working with offsets from center, if cx and cy are at
        image center (which they usually are), the formula simplifies to:
        dx_world = dx_pixel * depth / fx
        dy_world = dy_pixel * depth / fy
    """
    # Convert pixel offset to world coordinates
    dx_m = dx_px * depth / fx
    dy_m = dy_px * depth / fy
    
    return (dx_m, dy_m)


def pixel_to_world_with_intrinsics(
    dx_px: float,
    dy_px: float,
    depth: float,
    K: np.ndarray
) -> Tuple[float, float]:
    """
    Convert pixel offset to world coordinates using camera intrinsic matrix.
    
    Args:
        dx_px: Horizontal pixel offset from image center
        dy_px: Vertical pixel offset from image center  
        depth: Distance to object in meters
        K: 3x3 camera intrinsic matrix:
           [[fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]]
            
    Returns:
        (dx_m, dy_m): World coordinate offsets in meters
        
    Example:
        K = np.array([[600, 0, 320],
                      [0, 600, 240],
                      [0, 0, 1]])
        dx_m, dy_m = pixel_to_world_with_intrinsics(100, -50, 0.5, K)
    """
    fx = K[0, 0]
    fy = K[1, 1]
    
    return pixel_to_world(dx_px, dy_px, depth, fx, fy)


def world_to_pixel(
    dx_m: float,
    dy_m: float,
    depth: float,
    fx: float,
    fy: float
) -> Tuple[int, int]:
    """
    Convert world coordinate offset to pixel offset (inverse of pixel_to_world).
    
    Args:
        dx_m: Horizontal offset in meters
        dy_m: Vertical offset in meters
        depth: Distance to object in meters
        fx: Focal length in x-direction (pixels)
        fy: Focal length in y-direction (pixels)
        
    Returns:
        (dx_px, dy_px): Pixel offsets as integers
        
    Example:
        # Convert 10cm right, 5cm down at 0.5m depth
        dx_px, dy_px = world_to_pixel(0.1, 0.05, 0.5, 600, 600)
    """
    dx_px = int(dx_m * fx / depth)
    dy_px = int(dy_m * fy / depth)
    
    return (dx_px, dy_px)


def estimate_focal_length(
    image_width: int,
    fov_degrees: float,
    sensor_width_mm: Optional[float] = None
) -> float:
    """
    Estimate focal length in pixels from field of view.
    
    Args:
        image_width: Image width in pixels
        fov_degrees: Horizontal field of view in degrees
        sensor_width_mm: Optional physical sensor width in mm
        
    Returns:
        Estimated focal length in pixels
        
    Example:
        # Typical webcam with 60° horizontal FOV
        fx = estimate_focal_length(640, 60)
        print(f"Estimated focal length: {fx:.1f} pixels")
    """
    import math
    
    # Convert FOV to radians
    fov_rad = math.radians(fov_degrees)
    
    # Focal length from FOV: f = (w/2) / tan(FOV/2)
    fx = (image_width / 2.0) / math.tan(fov_rad / 2.0)
    
    return fx


def get_camera_intrinsics(
    image_width: int = 640,
    image_height: int = 480,
    fov_horizontal: float = 60.0,
    fov_vertical: Optional[float] = None
) -> Dict[str, float]:
    """
    Get camera intrinsic parameters from image size and FOV.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_horizontal: Horizontal field of view in degrees
        fov_vertical: Vertical FOV (computed from horizontal if None)
        
    Returns:
        Dictionary with keys: 'fx', 'fy', 'cx', 'cy', 'K' (intrinsic matrix)
        
    Example:
        intrinsics = get_camera_intrinsics(640, 480, 60)
        fx = intrinsics['fx']
        K = intrinsics['K']  # 3x3 matrix
    """
    # Estimate focal lengths
    fx = estimate_focal_length(image_width, fov_horizontal)
    
    if fov_vertical is None:
        # Assume square pixels
        fy = fx
    else:
        fy = estimate_focal_length(image_height, fov_vertical)
    
    # Principal point (usually image center)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Build intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'K': K,
        'image_width': image_width,
        'image_height': image_height
    }


class CameraCalibration:
    """
    Camera calibration helper for pixel-to-world conversions.
    
    Stores camera intrinsics and provides convenient conversion methods.
    """
    
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ):
        """
        Initialize camera calibration.
        
        Args:
            fx: Focal length x (pixels)
            fy: Focal length y (pixels)
            cx: Principal point x (defaults to image_width/2)
            cy: Principal point y (defaults to image_height/2)
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height
        
        # Set principal point to image center if not specified
        if self.cx is None and image_width is not None:
            self.cx = image_width / 2.0
        if self.cy is None and image_height is not None:
            self.cy = image_height / 2.0
        
        # Build intrinsic matrix
        self.K = np.array([
            [fx, 0, cx if cx is not None else 0],
            [0, fy, cy if cy is not None else 0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @classmethod
    def from_fov(
        cls,
        image_width: int,
        image_height: int,
        fov_horizontal: float,
        fov_vertical: Optional[float] = None
    ):
        """Create calibration from field of view."""
        params = get_camera_intrinsics(
            image_width, image_height,
            fov_horizontal, fov_vertical
        )
        return cls(
            fx=params['fx'],
            fy=params['fy'],
            cx=params['cx'],
            cy=params['cy'],
            image_width=image_width,
            image_height=image_height
        )
    
    def pixel_to_world(
        self,
        dx_px: float,
        dy_px: float,
        depth: float
    ) -> Tuple[float, float]:
        """Convert pixel offset to world coordinates."""
        return pixel_to_world(dx_px, dy_px, depth, self.fx, self.fy)
    
    def world_to_pixel(
        self,
        dx_m: float,
        dy_m: float,
        depth: float
    ) -> Tuple[int, int]:
        """Convert world offset to pixel coordinates."""
        return world_to_pixel(dx_m, dy_m, depth, self.fx, self.fy)
    
    def get_fov(self) -> Tuple[float, float]:
        """Get field of view in degrees (horizontal, vertical)."""
        import math
        
        if self.image_width is not None:
            fov_h = 2 * math.degrees(math.atan((self.image_width / 2) / self.fx))
        else:
            fov_h = None
            
        if self.image_height is not None:
            fov_v = 2 * math.degrees(math.atan((self.image_height / 2) / self.fy))
        else:
            fov_v = None
            
        return (fov_h, fov_v)
    
    def __repr__(self):
        fov_h, fov_v = self.get_fov()
        return (f"CameraCalibration(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"FOV={fov_h:.1f}°x{fov_v:.1f}°)")


def compute_pixel_offset(
    rgb: np.ndarray,
    bbox_color: List[int] = [0, 0, 255],  # Blue by default
    method: str = 'color_threshold',
    return_debug: bool = False,
    color_tolerance: int = 40,
    min_area: int = 100,
    max_area: Optional[int] = None,
    use_hsv: bool = True,
    morphology: bool = True,
    subpixel: bool = False
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], Dict[str, Any]]]:
    """
    Find the center of a colored object (e.g., blue cube) in an RGB image.
    
    This provides a fast, reliable fallback when GPT vision is slow or unavailable.
    
    Args:
        rgb: RGB image as numpy array (H, W, 3)
        bbox_color: Target color in RGB format [R, G, B]
        method: Detection method ('color_threshold', 'template', 'contour', 'adaptive')
        return_debug: If True, return debug information
        color_tolerance: Tolerance for color matching (higher = more permissive)
        min_area: Minimum area for valid detection (pixels)
        max_area: Maximum area for valid detection (pixels)
        use_hsv: Use HSV color space for better color detection
        morphology: Apply morphological operations to clean up mask
        subpixel: Use subpixel accuracy for center detection
        
    Returns:
        (x, y): Pixel coordinates of object center relative to image center
                Positive x = right, positive y = down
        If return_debug=True: ((x, y), debug_info_dict)
        
    Example:
        # Simple usage
        offset_x, offset_y = compute_pixel_offset(rgb_image)
        
        # With debug info
        (offset_x, offset_y), debug = compute_pixel_offset(
            rgb_image, 
            return_debug=True
        )
        cv2.imshow('mask', debug['mask'])
    """
    if rgb.dtype != np.uint8:
        rgb = np.uint8(np.clip(rgb, 0, 255))
    
    # Get image dimensions
    height, width = rgb.shape[:2]
    image_center = (width // 2, height // 2)
    
    # Choose detection method
    if method == 'color_threshold':
        result = detect_by_color_threshold(
            rgb, bbox_color, color_tolerance, 
            use_hsv, morphology, min_area, max_area
        )
    elif method == 'adaptive':
        result = detect_by_adaptive_threshold(
            rgb, bbox_color, color_tolerance,
            min_area, max_area
        )
    elif method == 'contour':
        result = detect_by_contour_matching(
            rgb, bbox_color, color_tolerance,
            min_area, max_area
        )
    elif method == 'template':
        # Template matching would require a template image
        logger.warning("Template matching requires template image, falling back to color threshold")
        result = detect_by_color_threshold(
            rgb, bbox_color, color_tolerance,
            use_hsv, morphology, min_area, max_area
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if result is None:
        # No detection, return center (no offset)
        if return_debug:
            return (0, 0), {'detected': False, 'method': method}
        return (0, 0)
    
    # Compute pixel offset from image center
    center_x, center_y = result.center
    
    # Use subpixel accuracy if requested
    if subpixel and result.mask is not None:
        center_x, center_y = compute_subpixel_center(result.mask, (center_x, center_y))
    
    offset_x = center_x - image_center[0]
    offset_y = center_y - image_center[1]
    
    if return_debug:
        debug_info = {
            'detected': True,
            'method': method,
            'center': (center_x, center_y),
            'offset': (offset_x, offset_y),
            'bbox': result.bbox,
            'confidence': result.confidence,
            'area': result.area,
            'mask': result.mask,
            'image_size': (width, height),
            'image_center': image_center
        }
        return (offset_x, offset_y), debug_info
    
    return (offset_x, offset_y)


def detect_by_color_threshold(
    rgb: np.ndarray,
    target_color: List[int],
    tolerance: int = 40,
    use_hsv: bool = True,
    morphology: bool = True,
    min_area: int = 100,
    max_area: Optional[int] = None
) -> Optional[DetectionResult]:
    """
    Detect object by color thresholding.
    
    Most reliable method for colored objects like blue cubes.
    """
    if use_hsv:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Convert target color to HSV
        target_hsv = cv2.cvtColor(
            np.uint8([[target_color]]), 
            cv2.COLOR_RGB2HSV
        )[0][0]
        
        # Define HSV range based on target color
        if target_color[2] > 200 and target_color[0] < 100:  # Blue
            # Special handling for blue
            lower = np.array([100, 50, 50])
            upper = np.array([130, 255, 255])
        else:
            # General case
            h_tolerance = tolerance // 2
            s_tolerance = tolerance * 2
            v_tolerance = tolerance * 2
            
            lower = np.array([
                max(0, target_hsv[0] - h_tolerance),
                max(0, target_hsv[1] - s_tolerance),
                max(0, target_hsv[2] - v_tolerance)
            ])
            upper = np.array([
                min(179, target_hsv[0] + h_tolerance),
                min(255, target_hsv[1] + s_tolerance),
                min(255, target_hsv[2] + v_tolerance)
            ])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
    else:
        # RGB thresholding
        lower = np.array([max(0, c - tolerance) for c in target_color])
        upper = np.array([min(255, c + tolerance) for c in target_color])
        mask = cv2.inRange(rgb, lower, upper)
    
    # Apply morphological operations to clean up the mask
    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest valid contour
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            if max_area is None or area <= max_area:
                valid_contours.append((contour, area))
    
    if not valid_contours:
        return None
    
    # Sort by area and get the largest
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    best_contour, area = valid_contours[0]
    
    # Compute center using moments
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(best_contour)
    
    # Compute confidence based on how well the detection fills the bounding box
    bbox_area = w * h
    fill_ratio = area / bbox_area if bbox_area > 0 else 0
    confidence = min(1.0, fill_ratio * 1.2)  # Boost slightly
    
    return DetectionResult(
        center=(cx, cy),
        bbox=(x, y, w, h),
        confidence=confidence,
        mask=mask,
        area=int(area)
    )


def detect_by_adaptive_threshold(
    rgb: np.ndarray,
    target_color: List[int],
    tolerance: int = 40,
    min_area: int = 100,
    max_area: Optional[int] = None
) -> Optional[DetectionResult]:
    """
    Detect object using adaptive color thresholding.
    
    More robust to lighting variations.
    """
    # Convert to LAB color space for perceptual color difference
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(
        np.uint8([[target_color]]), 
        cv2.COLOR_RGB2LAB
    )[0][0]
    
    # Compute color distance in LAB space
    diff = np.sqrt(np.sum((lab - target_lab) ** 2, axis=2))
    
    # Adaptive threshold based on statistics
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    threshold = min(tolerance, mean_diff - 0.5 * std_diff)
    
    # Create binary mask
    mask = (diff < threshold).astype(np.uint8) * 255
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest component
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter and find best contour
    best_contour = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            if max_area is None or area <= max_area:
                if area > best_area:
                    best_area = area
                    best_contour = contour
    
    if best_contour is None:
        return None
    
    # Compute center
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(best_contour)
    
    return DetectionResult(
        center=(cx, cy),
        bbox=(x, y, w, h),
        confidence=0.8,  # Adaptive method confidence
        mask=mask,
        area=int(best_area)
    )


def detect_by_contour_matching(
    rgb: np.ndarray,
    target_color: List[int],
    tolerance: int = 40,
    min_area: int = 100,
    max_area: Optional[int] = None
) -> Optional[DetectionResult]:
    """
    Detect cube-like objects using contour shape analysis.
    
    Good for detecting rectangular/square objects.
    """
    # First get color mask
    result = detect_by_color_threshold(
        rgb, target_color, tolerance,
        use_hsv=True, morphology=True,
        min_area=min_area, max_area=max_area
    )
    
    if result is None:
        return None
    
    # Analyze contour shape
    mask = result.mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_score = 0
    best_result = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Score based on how rectangular the shape is
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Ideal cube should have aspect ratio close to 1
        aspect_score = 1.0 - abs(1.0 - aspect_ratio)
        
        # Check how well contour fills the bounding box
        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        
        # Combined score
        score = aspect_score * 0.5 + fill_ratio * 0.5
        
        if score > best_score:
            best_score = score
            
            # Compute center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                best_result = DetectionResult(
                    center=(cx, cy),
                    bbox=(x, y, w, h),
                    confidence=best_score,
                    mask=mask,
                    area=int(area)
                )
    
    return best_result


def compute_subpixel_center(mask: np.ndarray, initial_center: Tuple[int, int]) -> Tuple[float, float]:
    """
    Compute subpixel-accurate center using weighted centroid.
    """
    # Create weight map (distance from edge)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Use distance as weight
    y_coords, x_coords = np.ogrid[:mask.shape[0], :mask.shape[1]]
    
    # Compute weighted centroid
    total_weight = np.sum(dist)
    if total_weight > 0:
        cx = np.sum(x_coords * dist) / total_weight
        cy = np.sum(y_coords * dist) / total_weight
        return (cx, cy)
    
    return initial_center


def detect_multiple_objects(
    rgb: np.ndarray,
    color_targets: Dict[str, List[int]],
    tolerance: int = 40,
    min_area: int = 100
) -> Dict[str, Optional[DetectionResult]]:
    """
    Detect multiple colored objects in a single pass.
    
    Args:
        rgb: Input RGB image
        color_targets: Dictionary of object_name -> RGB color
        tolerance: Color matching tolerance
        min_area: Minimum area for valid detection
        
    Returns:
        Dictionary of object_name -> DetectionResult (or None if not found)
        
    Example:
        targets = {
            'blue_cube': [0, 0, 255],
            'red_cube': [255, 0, 0],
            'green_cube': [0, 255, 0]
        }
        results = detect_multiple_objects(rgb, targets)
    """
    results = {}
    
    for name, color in color_targets.items():
        result = detect_by_color_threshold(
            rgb, color, tolerance,
            use_hsv=True, morphology=True,
            min_area=min_area
        )
        results[name] = result
    
    return results


def visualize_detection(
    rgb: np.ndarray,
    detection: Optional[DetectionResult],
    object_name: str = "Object",
    show_offset: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Visualize detection result on image.
    
    Args:
        rgb: Original RGB image
        detection: Detection result
        object_name: Name to display
        show_offset: Show pixel offset from center
        show_confidence: Show confidence score
        
    Returns:
        Annotated image
    """
    vis = rgb.copy()
    height, width = rgb.shape[:2]
    image_center = (width // 2, height // 2)
    
    # Draw image center
    cv2.drawMarker(vis, image_center, (0, 255, 0), 
                   cv2.MARKER_CROSS, 20, 2)
    
    if detection is None:
        cv2.putText(vis, f"{object_name}: Not detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 0, 0), 2)
        return vis
    
    # Draw bounding box
    x, y, w, h = detection.bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw center point
    cv2.circle(vis, detection.center, 5, (255, 0, 0), -1)
    
    # Draw line from image center to object center
    if show_offset:
        cv2.line(vis, image_center, detection.center, (255, 255, 0), 2)
        
        # Calculate offset
        offset_x = detection.center[0] - image_center[0]
        offset_y = detection.center[1] - image_center[1]
        
        # Display offset
        offset_text = f"Offset: ({offset_x:+d}, {offset_y:+d})"
        cv2.putText(vis, offset_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Display object name and confidence
    label = object_name
    if show_confidence:
        label += f" ({detection.confidence:.2f})"
    
    cv2.putText(vis, label, (x, y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display area
    cv2.putText(vis, f"Area: {detection.area}px", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return vis


class ObjectTracker:
    """
    Simple object tracker using color detection.
    
    Maintains tracking across frames with Kalman filter for smoothing.
    """
    
    def __init__(
        self,
        target_color: List[int] = [0, 0, 255],
        smooth_factor: float = 0.3,
        use_kalman: bool = True
    ):
        """
        Initialize tracker.
        
        Args:
            target_color: RGB color to track
            smooth_factor: Smoothing factor (0=no smoothing, 1=no update)
            use_kalman: Use Kalman filter for prediction
        """
        self.target_color = target_color
        self.smooth_factor = smooth_factor
        self.use_kalman = use_kalman
        
        self.last_detection = None
        self.last_center = None
        
        if use_kalman:
            self._init_kalman()
    
    def _init_kalman(self):
        """Initialize Kalman filter for tracking."""
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State: [x, y, vx, vy]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
    
    def update(self, rgb: np.ndarray) -> Tuple[int, int]:
        """
        Update tracking with new frame.
        
        Args:
            rgb: Current frame
            
        Returns:
            (x, y) pixel offset from center
        """
        # Detect object
        result = detect_by_color_threshold(
            rgb, self.target_color,
            use_hsv=True, morphology=True
        )
        
        if result is None:
            # Lost tracking, use prediction if available
            if self.use_kalman and self.last_center is not None:
                prediction = self.kf.predict()
                center = (int(prediction[0]), int(prediction[1]))
                
                # Calculate offset
                height, width = rgb.shape[:2]
                offset_x = center[0] - width // 2
                offset_y = center[1] - height // 2
                
                return (offset_x, offset_y)
            
            return (0, 0)
        
        # Update Kalman filter
        if self.use_kalman:
            if self.last_center is None:
                # Initialize Kalman state
                self.kf.statePre = np.array([
                    result.center[0],
                    result.center[1],
                    0,
                    0
                ], dtype=np.float32)
            
            # Correct with measurement
            measurement = np.array([
                [result.center[0]],
                [result.center[1]]
            ], dtype=np.float32)
            
            self.kf.correct(measurement)
            
            # Get filtered position
            state = self.kf.statePost
            center = (int(state[0]), int(state[1]))
        else:
            # Simple exponential smoothing
            if self.last_center is not None:
                center = (
                    int(self.last_center[0] * self.smooth_factor + 
                        result.center[0] * (1 - self.smooth_factor)),
                    int(self.last_center[1] * self.smooth_factor + 
                        result.center[1] * (1 - self.smooth_factor))
                )
            else:
                center = result.center
        
        self.last_detection = result
        self.last_center = center
        
        # Calculate offset from image center
        height, width = rgb.shape[:2]
        offset_x = center[0] - width // 2
        offset_y = center[1] - height // 2
        
        return (offset_x, offset_y)
    
    def reset(self):
        """Reset tracker state."""
        self.last_detection = None
        self.last_center = None
        if self.use_kalman:
            self._init_kalman()


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing Vision Utilities")
    print("="*60)
    
    # Create synthetic test image with blue cube
    print("\n1. Creating test image with blue cube")
    height, width = 480, 640
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 200  # Gray background
    
    # Add blue rectangle (cube)
    cube_size = 80
    cube_x = width // 2 + 50  # Offset from center
    cube_y = height // 2 - 30
    
    rgb[cube_y:cube_y+cube_size, cube_x:cube_x+cube_size] = [0, 0, 255]  # Blue
    
    # Add some noise
    noise = np.random.randint(-20, 20, rgb.shape)
    rgb = np.clip(rgb.astype(int) + noise, 0, 255).astype(np.uint8)
    
    print(f"  Image size: {width}x{height}")
    print(f"  Blue cube at: ({cube_x + cube_size//2}, {cube_y + cube_size//2})")
    print(f"  Expected offset: ({50 + cube_size//2}, {-30 + cube_size//2})")
    
    # Test basic detection
    print("\n2. Testing basic color detection")
    offset = compute_pixel_offset(rgb)
    print(f"  Detected offset: {offset}")
    
    # Test with debug info
    print("\n3. Testing with debug information")
    offset, debug = compute_pixel_offset(rgb, return_debug=True)
    print(f"  Detected: {debug['detected']}")
    print(f"  Center: {debug['center']}")
    print(f"  Offset: {debug['offset']}")
    print(f"  Confidence: {debug['confidence']:.2f}")
    print(f"  Area: {debug['area']} pixels")
    
    # Test different methods
    print("\n4. Testing different detection methods")
    for method in ['color_threshold', 'adaptive', 'contour']:
        offset = compute_pixel_offset(rgb, method=method)
        print(f"  {method}: offset = {offset}")
    
    # Test with no object
    print("\n5. Testing with no blue object")
    rgb_empty = np.ones((height, width, 3), dtype=np.uint8) * 200
    offset = compute_pixel_offset(rgb_empty)
    print(f"  Offset (no detection): {offset}")
    
    # Test multiple objects
    print("\n6. Testing multiple object detection")
    
    # Add red and green cubes
    rgb[100:150, 100:150] = [255, 0, 0]  # Red
    rgb[300:350, 400:450] = [0, 255, 0]  # Green
    
    targets = {
        'blue_cube': [0, 0, 255],
        'red_cube': [255, 0, 0],
        'green_cube': [0, 255, 0]
    }
    
    results = detect_multiple_objects(rgb, targets)
    for name, detection in results.items():
        if detection:
            print(f"  {name}: center={detection.center}, confidence={detection.confidence:.2f}")
        else:
            print(f"  {name}: not detected")
    
    # Test tracker
    print("\n7. Testing object tracker")
    tracker = ObjectTracker(target_color=[0, 0, 255])
    
    # Simulate tracking over multiple frames
    for i in range(5):
        # Simulate slight movement
        rgb_frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        cube_x_moving = cube_x + i * 5
        rgb_frame[cube_y:cube_y+cube_size, cube_x_moving:cube_x_moving+cube_size] = [0, 0, 255]
        
        offset = tracker.update(rgb_frame)
        print(f"  Frame {i+1}: offset = {offset}")
    
    print("\n" + "="*60)
    print("All vision utility tests completed!")
    print("="*60)