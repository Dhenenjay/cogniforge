"""
Test script for GPT Vision offset detection.

Demonstrates how to use ask_gpt_for_offset and related functions.
"""

import numpy as np
import cv2
import base64
import json
import os
from cogniforge.vision.vision_utils import (
    ask_gpt_for_offset,
    ask_gpt_for_offset_simple,
    encode_image_to_base64,
    encode_array_to_base64,
    hybrid_object_detection,
    compute_pixel_offset
)


def test_with_synthetic_image():
    """Test with a synthetic image containing a blue cube."""
    print("="*60)
    print("Testing GPT Vision with Synthetic Image")
    print("="*60)
    
    # Create synthetic test image
    height, width = 480, 640
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 200  # Gray background
    
    # Add blue cube offset from center
    cube_size = 80
    cube_x = width // 2 + 100  # 100 pixels right of center
    cube_y = height // 2 - 50   # 50 pixels above center
    
    rgb[cube_y:cube_y+cube_size, cube_x:cube_x+cube_size] = [0, 0, 255]  # Blue
    
    # Expected offset
    expected_dx = 100 + cube_size//2  # Center of cube
    expected_dy = -50 + cube_size//2
    
    print(f"\nCreated {width}x{height} image with blue cube")
    print(f"Expected offset: dx={expected_dx}, dy={expected_dy}")
    
    # Test 1: Color detection (fast, local)
    print("\n1. Testing color detection (fast baseline):")
    offset = compute_pixel_offset(rgb)
    print(f"   Color detection result: dx={offset[0]}, dy={offset[1]}")
    
    # Test 2: GPT Vision (if API key available)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("\n2. Testing GPT Vision API:")
        
        # Encode image to base64
        base64_image = encode_array_to_base64(rgb)
        
        # Ask GPT for offset
        result = ask_gpt_for_offset(
            base64_image,
            instruction="Find the blue square/cube and return its pixel offset from the image center",
            api_key=api_key,
            max_tokens=50,
            temperature=0.0
        )
        
        print(f"   GPT Vision result: {json.dumps(result, indent=2)}")
        print(f"   Offset: dx={result['dx_px']}, dy={result['dy_px']}")
        
        # Test simplified version
        print("\n3. Testing simplified GPT Vision:")
        result_simple = ask_gpt_for_offset_simple(
            base64_image,
            "Find the blue cube",
            api_key
        )
        print(f"   Result: {result_simple}")
        
        # Test hybrid approach
        print("\n4. Testing hybrid detection:")
        hybrid_offset = hybrid_object_detection(
            rgb,
            use_gpt=True,
            gpt_timeout=3.0,
            api_key=api_key
        )
        print(f"   Hybrid result: dx={hybrid_offset[0]}, dy={hybrid_offset[1]}")
        
    else:
        print("\n⚠️ No OPENAI_API_KEY found in environment")
        print("   Set your API key: set OPENAI_API_KEY=your_key_here")
        print("   Skipping GPT Vision tests")
    
    # Save test image for manual inspection
    cv2.imwrite('test_blue_cube.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("\n✅ Saved test image as 'test_blue_cube.jpg'")


def test_with_real_image(image_path: str):
    """Test with a real image file."""
    print("\n" + "="*60)
    print(f"Testing with real image: {image_path}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    
    print(f"\nImage size: {width}x{height}")
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ No API key, using color detection only")
        offset = compute_pixel_offset(rgb)
        print(f"Color detection offset: dx={offset[0]}, dy={offset[1]}")
        return
    
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Test different instructions
    instructions = [
        "Find the blue cube and return its pixel offset from center",
        "Locate the primary blue object in the image",
        "Identify the blue rectangular object"
    ]
    
    for i, instruction in enumerate(instructions, 1):
        print(f"\n{i}. Instruction: '{instruction}'")
        result = ask_gpt_for_offset(
            base64_image,
            instruction,
            api_key,
            max_tokens=50
        )
        print(f"   Result: {result}")


def example_usage():
    """Show example usage patterns."""
    print("\n" + "="*60)
    print("EXAMPLE USAGE PATTERNS")
    print("="*60)
    
    print("""
# 1. Basic Usage with Base64 Image
# ---------------------------------
from cogniforge.vision.vision_utils import ask_gpt_for_offset

# Load and encode image
with open('robot_view.jpg', 'rb') as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

# Get offset (returns JSON with dx_px and dy_px)
offset = ask_gpt_for_offset(base64_image, "Find the blue cube")
print(f"Cube at: ({offset['dx_px']}, {offset['dy_px']})")


# 2. From NumPy Array
# -------------------
import numpy as np
from cogniforge.vision.vision_utils import encode_array_to_base64, ask_gpt_for_offset

# Your RGB image as numpy array
rgb_image = get_camera_image()  # Your function

# Encode and detect
base64_image = encode_array_to_base64(rgb_image)
offset = ask_gpt_for_offset(base64_image, "Find the target object")


# 3. Hybrid Approach (GPT with Fallback)
# ---------------------------------------
from cogniforge.vision.vision_utils import hybrid_object_detection

# Automatically tries GPT first, falls back to color detection
dx, dy = hybrid_object_detection(
    rgb_image,
    use_gpt=True,
    gpt_timeout=2.0,  # Max 2 seconds for GPT
    instruction="Find the blue cube",
    api_key=os.getenv('OPENAI_API_KEY')
)


# 4. Simple One-Liner
# --------------------
from cogniforge.vision.vision_utils import ask_gpt_for_offset_simple

offset = ask_gpt_for_offset_simple(base64_img, "Find blue cube")


# 5. With Error Handling
# -----------------------
try:
    offset = ask_gpt_for_offset(
        base64_image,
        "Find the blue cube",
        api_key=api_key,
        timeout=5.0,
        retry_count=2
    )
    
    if offset['dx_px'] == 0 and offset['dy_px'] == 0:
        print("Object not found")
    else:
        print(f"Found at offset: {offset}")
        
except Exception as e:
    print(f"Detection failed: {e}")
    # Fallback to color detection
    dx, dy = compute_pixel_offset(rgb_image)
    """)


def benchmark_detection_methods():
    """Benchmark different detection methods."""
    print("\n" + "="*60)
    print("BENCHMARKING DETECTION METHODS")
    print("="*60)
    
    import time
    
    # Create test image
    height, width = 480, 640
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 200
    cube_x, cube_y = width // 2 + 75, height // 2 - 40
    rgb[cube_y:cube_y+60, cube_x:cube_x+60] = [0, 0, 255]
    
    print("\nTest image: 640x480 with blue cube at (+75, -40)")
    
    # 1. Color detection benchmark
    print("\n1. Color Detection (Local):")
    times = []
    for _ in range(10):
        start = time.time()
        offset = compute_pixel_offset(rgb)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    print(f"   Average time: {avg_time:.2f} ms")
    print(f"   Result: {offset}")
    print(f"   Speed: {1000/avg_time:.0f} FPS capable")
    
    # 2. GPT Vision benchmark (if API key available)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("\n2. GPT Vision API:")
        base64_image = encode_array_to_base64(rgb)
        
        times = []
        for i in range(3):  # Only 3 tests due to API costs
            start = time.time()
            result = ask_gpt_for_offset(
                base64_image,
                "Find the blue cube",
                api_key,
                retry_count=1
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Test {i+1}: {elapsed:.2f}s, Result: {result}")
        
        avg_time = np.mean(times)
        print(f"   Average time: {avg_time:.2f} seconds")
        print(f"   Speed: {1/avg_time:.2f} FPS capable")
        
        # 3. Hybrid approach
        print("\n3. Hybrid Detection (GPT with 1s timeout):")
        start = time.time()
        offset = hybrid_object_detection(
            rgb,
            use_gpt=True,
            gpt_timeout=1.0,
            api_key=api_key
        )
        elapsed = time.time() - start
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Result: {offset}")
        print(f"   Method used: {'GPT' if elapsed < 1.5 else 'Color (fallback)'}")
    
    print("\n" + "-"*60)
    print("RECOMMENDATION:")
    print("- Real-time control: Use color detection (~1000 FPS)")
    print("- High accuracy needed: Use GPT Vision (~0.3 FPS)")
    print("- Balanced: Use hybrid with 1-2s timeout")


if __name__ == "__main__":
    # Run tests
    print("GPT VISION OFFSET DETECTION TEST SUITE")
    print("="*60)
    
    # Test with synthetic image
    test_with_synthetic_image()
    
    # Show usage examples
    example_usage()
    
    # Benchmark if API key available
    if os.getenv('OPENAI_API_KEY'):
        benchmark_detection_methods()
    else:
        print("\n" + "="*60)
        print("ℹ️ TO ENABLE GPT VISION TESTS:")
        print("="*60)
        print("Set your OpenAI API key as an environment variable:")
        print("\nWindows PowerShell:")
        print('  $env:OPENAI_API_KEY="sk-..."')
        print("\nWindows CMD:")
        print('  set OPENAI_API_KEY=sk-...')
        print("\nLinux/Mac:")
        print('  export OPENAI_API_KEY="sk-..."')
    
    print("\n✅ All tests completed!")