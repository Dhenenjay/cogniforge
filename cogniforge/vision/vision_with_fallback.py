"""
Vision system with automatic fallback to color-threshold detection.

When GPT vision API times out or fails, the system falls back to
traditional color-threshold based object detection with offset calculation.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import time
import traceback
from datetime import datetime
from dataclasses import dataclass
import threading

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Vision detection result."""
    success: bool
    method: str  # 'gpt_vision' or 'color_threshold'
    offset_x: float
    offset_y: float
    confidence: float
    objects_detected: List[Dict[str, Any]]
    processing_time: float
    error_message: Optional[str] = None


class ColorThresholdDetector:
    """Fallback color-threshold based object detector."""
    
    # Default color ranges in HSV
    COLOR_RANGES = {
        'red': {
            'lower1': np.array([0, 120, 70]),
            'upper1': np.array([10, 255, 255]),
            'lower2': np.array([170, 120, 70]),  # Red wraps around in HSV
            'upper2': np.array([180, 255, 255])
        },
        'blue': {
            'lower1': np.array([100, 150, 50]),
            'upper1': np.array([130, 255, 255])
        },
        'green': {
            'lower1': np.array([40, 40, 40]),
            'upper1': np.array([80, 255, 255])
        },
        'yellow': {
            'lower1': np.array([20, 100, 100]),
            'upper1': np.array([30, 255, 255])
        },
        'orange': {
            'lower1': np.array([10, 100, 100]),
            'upper1': np.array([20, 255, 255])
        },
        'purple': {
            'lower1': np.array([130, 50, 50]),
            'upper1': np.array([160, 255, 255])
        }
    }
    
    def __init__(self, target_color: str = 'red', min_area: int = 500):
        """
        Initialize color threshold detector.
        
        Args:
            target_color: Primary color to detect
            min_area: Minimum contour area to consider
        """
        self.target_color = target_color.lower()
        self.min_area = min_area
        
    def detect(self, image: np.ndarray, target_position: Optional[Tuple[int, int]] = None) -> VisionResult:
        """
        Detect objects using color thresholding.
        
        Args:
            image: Input image (BGR format)
            target_position: Expected target position for offset calculation
            
        Returns:
            VisionResult with detection details
        """
        start_time = time.time()
        
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Get color range
            if self.target_color not in self.COLOR_RANGES:
                logger.warning(f"Color {self.target_color} not defined, using red")
                color_range = self.COLOR_RANGES['red']
            else:
                color_range = self.COLOR_RANGES[self.target_color]
            
            # Create mask
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            # Apply color threshold
            if 'lower1' in color_range:
                mask1 = cv2.inRange(hsv, color_range['lower1'], color_range['upper1'])
                mask = cv2.bitwise_or(mask, mask1)
            
            if 'lower2' in color_range:  # For colors that wrap around (like red)
                mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
                mask = cv2.bitwise_or(mask, mask2)
            
            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by area
            valid_contours = []
            objects_detected = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    valid_contours.append({
                        'contour': contour,
                        'area': area,
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h)
                    })
                    
                    objects_detected.append({
                        'type': f'{self.target_color}_object',
                        'position': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'confidence': min(area / 5000.0, 1.0)  # Simple confidence based on size
                    })
            
            # Calculate offset
            offset_x, offset_y = 0, 0
            confidence = 0.0
            
            if valid_contours:
                # Use largest contour
                largest = max(valid_contours, key=lambda x: x['area'])
                detected_center = largest['center']
                
                if target_position:
                    offset_x = detected_center[0] - target_position[0]
                    offset_y = detected_center[1] - target_position[1]
                else:
                    # Use image center as reference
                    image_center = (image.shape[1] // 2, image.shape[0] // 2)
                    offset_x = detected_center[0] - image_center[0]
                    offset_y = detected_center[1] - image_center[1]
                
                confidence = min(largest['area'] / 5000.0, 1.0)
            
            processing_time = time.time() - start_time
            
            return VisionResult(
                success=len(valid_contours) > 0,
                method='color_threshold',
                offset_x=offset_x,
                offset_y=offset_y,
                confidence=confidence,
                objects_detected=objects_detected,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Color threshold detection failed: {e}")
            return VisionResult(
                success=False,
                method='color_threshold',
                offset_x=0,
                offset_y=0,
                confidence=0,
                objects_detected=[],
                processing_time=time.time() - start_time,
                error_message=str(e)
            )


class VisionSystemWithFallback:
    """Vision system with automatic GPT fallback to color detection."""
    
    def __init__(
        self,
        gpt_timeout: float = 5.0,
        use_gpt: bool = True,
        fallback_color: str = 'red',
        show_banner: bool = True
    ):
        """
        Initialize vision system with fallback.
        
        Args:
            gpt_timeout: Timeout for GPT vision API in seconds
            use_gpt: Whether to attempt GPT vision first
            fallback_color: Default color for threshold detection
            show_banner: Whether to show fallback banner
        """
        self.gpt_timeout = gpt_timeout
        self.use_gpt = use_gpt
        self.fallback_color = fallback_color
        self.show_banner = show_banner
        self.color_detector = ColorThresholdDetector(fallback_color)
        self.failure_log = []
        self.stats = {
            'gpt_success': 0,
            'gpt_failure': 0,
            'fallback_used': 0,
            'total_calls': 0
        }
        
    def detect_objects(
        self,
        image: np.ndarray,
        prompt: str = "Detect objects and calculate offset",
        target_position: Optional[Tuple[int, int]] = None,
        max_retries: int = 2
    ) -> VisionResult:
        """
        Detect objects with automatic fallback.
        
        Args:
            image: Input image
            prompt: GPT vision prompt
            target_position: Target position for offset calculation
            max_retries: Maximum GPT retry attempts
            
        Returns:
            VisionResult with detection details
        """
        self.stats['total_calls'] += 1
        
        # Try GPT vision first if enabled
        if self.use_gpt:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting GPT vision (attempt {attempt + 1}/{max_retries})")
                    result = self._gpt_vision_detect(image, prompt, target_position)
                    
                    if result.success:
                        self.stats['gpt_success'] += 1
                        logger.info("✅ GPT vision successful")
                        return result
                    else:
                        logger.warning(f"GPT vision failed (attempt {attempt + 1})")
                        
                except TimeoutError as e:
                    logger.error(f"GPT vision timeout (attempt {attempt + 1}): {e}")
                    self._log_failure('timeout', str(e), prompt)
                    
                except Exception as e:
                    logger.error(f"GPT vision error (attempt {attempt + 1}): {e}")
                    self._log_failure('error', str(e), prompt)
                
                if attempt == max_retries - 1:
                    self.stats['gpt_failure'] += 1
                    logger.warning("🔄 All GPT attempts failed, falling back to color threshold")
                    if self.show_banner:
                        self._show_fallback_banner()
        
        # Fallback to color threshold detection
        self.stats['fallback_used'] += 1
        logger.info("📍 Using color threshold detection fallback")
        return self.color_detector.detect(image, target_position)
    
    def _gpt_vision_detect(
        self,
        image: np.ndarray,
        prompt: str,
        target_position: Optional[Tuple[int, int]]
    ) -> VisionResult:
        """
        Detect using GPT vision API (simulated).
        
        Args:
            image: Input image
            prompt: Detection prompt
            target_position: Target position
            
        Returns:
            VisionResult from GPT
            
        Raises:
            TimeoutError: If API times out
        """
        start_time = time.time()
        
        # Create a timeout mechanism
        result_container = {'result': None, 'error': None}
        
        def api_call():
            try:
                # Simulate GPT Vision API call
                time.sleep(0.5)  # Simulate API delay
                
                # Simulate occasional timeout for demonstration
                import random
                if random.random() < 0.3:  # 30% timeout rate for demo
                    time.sleep(self.gpt_timeout + 1)  # Force timeout
                
                # Mock GPT vision response
                # In production, this would be actual OpenAI Vision API call
                mock_detection = {
                    'objects': [
                        {
                            'type': 'cube',
                            'color': 'red',
                            'position': (320 + np.random.randint(-50, 50), 
                                       240 + np.random.randint(-50, 50)),
                            'confidence': 0.95
                        }
                    ]
                }
                
                # Calculate offset
                if mock_detection['objects']:
                    obj = mock_detection['objects'][0]
                    detected_pos = obj['position']
                    
                    if target_position:
                        offset_x = detected_pos[0] - target_position[0]
                        offset_y = detected_pos[1] - target_position[1]
                    else:
                        offset_x = detected_pos[0] - 320  # Assuming 640x480 image
                        offset_y = detected_pos[1] - 240
                    
                    result_container['result'] = VisionResult(
                        success=True,
                        method='gpt_vision',
                        offset_x=offset_x,
                        offset_y=offset_y,
                        confidence=obj['confidence'],
                        objects_detected=[obj],
                        processing_time=time.time() - start_time
                    )
                else:
                    result_container['result'] = VisionResult(
                        success=False,
                        method='gpt_vision',
                        offset_x=0,
                        offset_y=0,
                        confidence=0,
                        objects_detected=[],
                        processing_time=time.time() - start_time,
                        error_message="No objects detected"
                    )
                    
            except Exception as e:
                result_container['error'] = e
        
        # Run API call in thread with timeout
        thread = threading.Thread(target=api_call)
        thread.start()
        thread.join(timeout=self.gpt_timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"GPT vision API timeout after {self.gpt_timeout}s")
        
        if result_container['error']:
            raise result_container['error']
        
        if result_container['result']:
            return result_container['result']
        else:
            raise RuntimeError("GPT vision API returned no result")
    
    def _show_fallback_banner(self):
        """Display fallback banner in console."""
        from cogniforge.ui.console_utils import ConsoleAutoScroller
        
        banner = """
╔══════════════════════════════════════════════════════════╗
║                  ⚠️  VISION FALLBACK IN USE              ║
║                                                          ║
║  GPT Vision unavailable - using color threshold         ║
║  Detection accuracy may be reduced                      ║
╚══════════════════════════════════════════════════════════╝"""
        
        print(banner)
        
        # Also log to scroller if available
        try:
            scroller = ConsoleAutoScroller()
            scroller.print_and_scroll(banner)
        except:
            pass
    
    def _log_failure(self, failure_type: str, error_message: str, prompt: str):
        """Log GPT vision failure."""
        failure_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': failure_type,
            'error': error_message,
            'prompt': prompt,
            'traceback': traceback.format_exc()
        }
        
        self.failure_log.append(failure_entry)
        
        # Save to file
        log_file = f"vision_failures_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            import json
            with open(log_file, 'a') as f:
                json.dump(failure_entry, f)
                f.write('\n')
        except:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vision system statistics."""
        total = self.stats['total_calls']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'gpt_success_rate': self.stats['gpt_success'] / total if total > 0 else 0,
            'fallback_rate': self.stats['fallback_used'] / total if total > 0 else 0,
            'recent_failures': self.failure_log[-5:]
        }
    
    def display_offset_with_status(self, offset_x: float, offset_y: float, method: str):
        """Display offset with method indicator."""
        from cogniforge.ui.vision_display import VisionOffsetDisplay
        
        display = VisionOffsetDisplay()
        
        # Convert to pixels for display
        offset_x_px = int(offset_x)
        offset_y_px = int(offset_y)
        
        # Add method indicator
        if method == 'gpt_vision':
            status_icon = "🤖"
            status_text = "GPT Vision"
            status_color = "\033[92m"  # Green
        else:
            status_icon = "🎨"
            status_text = "Color Threshold"
            status_color = "\033[93m"  # Yellow
        
        print(f"{status_color}{status_icon} {status_text}\033[0m: ", end="")
        display.print_compact_status(offset_x_px, offset_y_px)


def create_robust_vision_system(
    fallback_color: str = 'red',
    gpt_timeout: float = 5.0,
    auto_fallback: bool = True
) -> VisionSystemWithFallback:
    """
    Create a robust vision system with automatic fallback.
    
    Args:
        fallback_color: Default color for threshold detection
        gpt_timeout: GPT API timeout in seconds
        auto_fallback: Enable automatic fallback
        
    Returns:
        Configured vision system
    """
    return VisionSystemWithFallback(
        gpt_timeout=gpt_timeout,
        use_gpt=auto_fallback,
        fallback_color=fallback_color,
        show_banner=True
    )


def test_vision_fallback():
    """Test the vision fallback mechanism."""
    
    print("="*70)
    print("TESTING VISION FALLBACK MECHANISM")
    print("="*70)
    
    # Create test image (640x480 RGB)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a red square in the image
    cv2.rectangle(test_image, (300, 200), (400, 300), (0, 0, 255), -1)
    
    # Add a blue square
    cv2.rectangle(test_image, (100, 100), (150, 150), (255, 0, 0), -1)
    
    # Convert to BGR for OpenCV
    test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    
    # Test with vision system
    vision = VisionSystemWithFallback(
        gpt_timeout=2.0,
        use_gpt=True,
        fallback_color='red'
    )
    
    print("\n1. Testing normal GPT vision (may succeed):")
    print("-"*40)
    
    result = vision.detect_objects(
        test_image_bgr,
        prompt="Detect red cube",
        target_position=(320, 240)
    )
    
    print(f"  Method used: {result.method}")
    print(f"  Success: {result.success}")
    print(f"  Offset: ({result.offset_x:.1f}, {result.offset_y:.1f})")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    
    # Display with status
    vision.display_offset_with_status(result.offset_x, result.offset_y, result.method)
    
    print("\n2. Testing with GPT disabled (always fallback):")
    print("-"*40)
    
    vision_fallback = VisionSystemWithFallback(
        use_gpt=False,
        fallback_color='red'
    )
    
    result = vision_fallback.detect_objects(
        test_image_bgr,
        target_position=(320, 240)
    )
    
    print(f"  Method used: {result.method}")
    print(f"  Success: {result.success}")
    print(f"  Offset: ({result.offset_x:.1f}, {result.offset_y:.1f})")
    print(f"  Objects detected: {len(result.objects_detected)}")
    
    # Display with status
    vision_fallback.display_offset_with_status(result.offset_x, result.offset_y, result.method)
    
    print("\n3. Testing color threshold for blue:")
    print("-"*40)
    
    blue_detector = ColorThresholdDetector('blue')
    result = blue_detector.detect(test_image_bgr, target_position=(125, 125))
    
    print(f"  Success: {result.success}")
    print(f"  Offset: ({result.offset_x:.1f}, {result.offset_y:.1f})")
    
    # Show statistics
    print("\n4. Vision System Statistics:")
    print("-"*40)
    
    stats = vision.get_statistics()
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  GPT successes: {stats['gpt_success']}")
    print(f"  GPT failures: {stats['gpt_failure']}")
    print(f"  Fallback used: {stats['fallback_used']}")
    if stats['total_calls'] > 0:
        print(f"  Fallback rate: {stats['fallback_rate']:.1%}")
    
    print("\n" + "="*70)
    print("✅ Vision fallback test complete!")
    print("="*70)


if __name__ == "__main__":
    # Test without OpenCV dependency for basic functionality
    try:
        test_vision_fallback()
    except ImportError as e:
        print(f"Note: OpenCV required for full functionality: {e}")
        print("\nInstall with: pip install opencv-python")
        
        # Test basic functionality without image processing
        print("\nTesting basic fallback logic without image processing...")
        vision = VisionSystemWithFallback()
        print(f"Vision system initialized with GPT timeout: {vision.gpt_timeout}s")