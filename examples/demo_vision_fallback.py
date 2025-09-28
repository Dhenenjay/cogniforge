"""
Demo: GPT Vision Fallback with Color Threshold

Shows the automatic fallback to color-threshold detection when GPT Vision
times out, with a clear banner indicating fallback is in use.
"""

import sys
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, 'C:/Users/Dhenenjay/cogniforge')

from cogniforge.vision.vision_with_fallback import (
    VisionSystemWithFallback,
    ColorThresholdDetector,
    create_robust_vision_system
)
from cogniforge.ui.vision_display import VisionOffsetDisplay
from cogniforge.ui.console_utils import ConsoleAutoScroller

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class VisionPipelineDemo:
    """Demo pipeline with vision fallback."""
    
    def __init__(self):
        self.vision_system = create_robust_vision_system(
            fallback_color='red',
            gpt_timeout=3.0,
            auto_fallback=True
        )
        self.vision_display = VisionOffsetDisplay()
        self.scroller = ConsoleAutoScroller()
        self.execution_log = []
        
    def process_frame(
        self,
        image: np.ndarray,
        target_position: Tuple[int, int],
        object_name: str = "object",
        simulate_timeout: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single frame with vision detection.
        
        Args:
            image: Input image
            target_position: Expected object position
            object_name: Name of object to detect
            simulate_timeout: Force GPT timeout for demo
            
        Returns:
            Processing results
        """
        print(f"\n📷 Processing frame for '{object_name}'...")
        print("-"*50)
        
        # Temporarily disable GPT if simulating timeout
        original_use_gpt = self.vision_system.use_gpt
        if simulate_timeout:
            print("⚠️ Simulating GPT Vision timeout...")
            # Set very short timeout to force failure
            self.vision_system.gpt_timeout = 0.001
        
        # Detect objects
        result = self.vision_system.detect_objects(
            image,
            prompt=f"Detect {object_name} and calculate offset",
            target_position=target_position,
            max_retries=1 if simulate_timeout else 2
        )
        
        # Display results with method indicator
        self._display_results(result, object_name)
        
        # Restore settings
        self.vision_system.use_gpt = original_use_gpt
        if simulate_timeout:
            self.vision_system.gpt_timeout = 3.0
        
        # Log results
        log_entry = {
            'object': object_name,
            'method': result.method,
            'success': result.success,
            'offset': (result.offset_x, result.offset_y),
            'confidence': result.confidence,
            'processing_time': result.processing_time
        }
        self.execution_log.append(log_entry)
        
        return log_entry
    
    def _display_results(self, result, object_name: str):
        """Display detection results with visual formatting."""
        
        # Method indicator with color
        if result.method == 'gpt_vision':
            method_badge = "🤖 \033[92mGPT Vision\033[0m"
        else:
            method_badge = "🎨 \033[93mColor Threshold\033[0m"
        
        print(f"\nDetection Method: {method_badge}")
        
        if result.success:
            print(f"✅ {object_name.capitalize()} detected successfully")
            
            # Display offset with color coding
            print("Offset: ", end="")
            self.vision_system.display_offset_with_status(
                result.offset_x, 
                result.offset_y,
                result.method
            )
            
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Processing time: {result.processing_time:.3f}s")
            
            # Show detected objects
            if result.objects_detected:
                print(f"Objects found: {len(result.objects_detected)}")
                for i, obj in enumerate(result.objects_detected[:3], 1):  # Show first 3
                    print(f"  {i}. {obj.get('type', 'unknown')} at {obj.get('position', '?')}")
        else:
            print(f"❌ {object_name.capitalize()} not detected")
            if result.error_message:
                print(f"Error: {result.error_message}")
    
    def run_comparison_test(self):
        """Run comparison between GPT and fallback methods."""
        
        print("\n" + "="*70)
        print("🔬 VISION METHOD COMPARISON TEST")
        print("="*70)
        
        # Create synthetic test image
        test_image = self._create_test_image()
        target_pos = (320, 240)
        
        # Test 1: Normal GPT Vision (may succeed)
        print("\n1️⃣ TEST 1: Normal GPT Vision")
        self.process_frame(test_image, target_pos, "red cube", simulate_timeout=False)
        time.sleep(1)
        
        # Test 2: Forced timeout -> Fallback
        print("\n2️⃣ TEST 2: GPT Timeout → Color Fallback")
        self.process_frame(test_image, target_pos, "red cube", simulate_timeout=True)
        time.sleep(1)
        
        # Test 3: Direct color threshold
        print("\n3️⃣ TEST 3: Direct Color Threshold (GPT disabled)")
        self.vision_system.use_gpt = False
        self.process_frame(test_image, target_pos, "red object", simulate_timeout=False)
        self.vision_system.use_gpt = True
        
    def _create_test_image(self) -> np.ndarray:
        """Create a synthetic test image with colored objects."""
        # Create blank image
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Add red rectangle (main target)
        image[200:280, 300:380] = [0, 0, 255]  # Red in BGR
        
        # Add blue circle area
        center = (150, 150)
        for y in range(100, 200):
            for x in range(100, 200):
                if (x - center[0])**2 + (y - center[1])**2 < 30**2:
                    image[y, x] = [255, 0, 0]  # Blue in BGR
        
        # Add green triangle area
        pts = np.array([[500, 100], [450, 180], [550, 180]], np.int32)
        for y in range(100, 181):
            for x in range(450, 551):
                # Simple point-in-triangle test
                if x > 450 and x < 550 and y > 100 and y < 180:
                    image[y, x] = [0, 255, 0]  # Green in BGR
        
        # Add noise for realism
        noise = np.random.randint(-20, 20, image.shape)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def show_statistics(self):
        """Display execution statistics."""
        
        print("\n" + "="*70)
        print("📊 VISION SYSTEM STATISTICS")
        print("="*70)
        
        stats = self.vision_system.get_statistics()
        
        print(f"\nTotal vision calls: {stats['total_calls']}")
        print(f"  GPT Vision successful: {stats['gpt_success']}")
        print(f"  GPT Vision failed: {stats['gpt_failure']}")
        print(f"  Color fallback used: {stats['fallback_used']}")
        
        if stats['total_calls'] > 0:
            print(f"\nSuccess rates:")
            print(f"  GPT success rate: {stats.get('gpt_success_rate', 0):.1%}")
            print(f"  Fallback usage rate: {stats.get('fallback_rate', 0):.1%}")
        
        # Processing time comparison
        if self.execution_log:
            gpt_times = [e['processing_time'] for e in self.execution_log if e['method'] == 'gpt_vision']
            fallback_times = [e['processing_time'] for e in self.execution_log if e['method'] == 'color_threshold']
            
            if gpt_times:
                print(f"\nGPT Vision avg time: {np.mean(gpt_times):.3f}s")
            if fallback_times:
                print(f"Color threshold avg time: {np.mean(fallback_times):.3f}s")
                print(f"Fallback is {np.mean(gpt_times)/np.mean(fallback_times):.1f}x faster" if gpt_times else "")


def main():
    """Run the vision fallback demonstration."""
    
    print("\n" + "="*70)
    print("🔄 GPT VISION FALLBACK DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows automatic fallback to color-threshold")
    print("detection when GPT Vision times out or fails.\n")
    
    # Initialize demo
    demo = VisionPipelineDemo()
    
    # Run comparison tests
    demo.run_comparison_test()
    
    # Show statistics
    demo.show_statistics()
    
    # Test different colors
    print("\n" + "="*70)
    print("🎨 COLOR DETECTION FALLBACK TEST")
    print("="*70)
    
    colors_to_test = ['red', 'blue', 'green', 'yellow']
    
    for color in colors_to_test:
        print(f"\n Testing {color} detection...")
        detector = ColorThresholdDetector(color)
        
        # Create test image
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Add colored region based on test color
        if color == 'red':
            test_img[30:70, 30:70] = [0, 0, 255]  # BGR
        elif color == 'blue':
            test_img[30:70, 30:70] = [255, 0, 0]
        elif color == 'green':
            test_img[30:70, 30:70] = [0, 255, 0]
        elif color == 'yellow':
            test_img[30:70, 30:70] = [0, 255, 255]
        
        result = detector.detect(test_img)
        status = "✅ Detected" if result.success else "❌ Not detected"
        print(f"  {status} - Confidence: {result.confidence:.1%}")
    
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    
    print("\nKey Features Demonstrated:")
    print("  1. Automatic fallback when GPT Vision times out")
    print("  2. Clear banner showing 'VISION FALLBACK IN USE'")
    print("  3. Color-threshold detection for multiple colors")
    print("  4. Offset calculation with both methods")
    print("  5. Performance comparison between methods")


if __name__ == "__main__":
    # Check for OpenCV
    try:
        import cv2
        main()
    except ImportError:
        print("\n⚠️ OpenCV not installed. Installing...")
        print("Run: pip install opencv-python")
        print("\nRunning limited demo without image processing...\n")
        
        # Limited demo without OpenCV
        vision = VisionSystemWithFallback()
        print(f"Vision system initialized")
        print(f"  GPT timeout: {vision.gpt_timeout}s")
        print(f"  Fallback color: {vision.fallback_color}")
        print(f"  Banner enabled: {vision.show_banner}")
        
        # Simulate timeout scenario
        print("\nSimulating GPT timeout scenario...")
        print("When GPT times out, you would see:\n")
        vision._show_fallback_banner()