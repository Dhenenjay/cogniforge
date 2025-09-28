"""
Color Calibration for Cube Detection

This module provides adaptive color threshold calibration for detecting
colored cubes under varying lighting conditions, especially when venue
projectors tint the display or environment.
"""

import cv2
import numpy as np
import json
import time
import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Color Space and Cube Types
# ============================================================================

class ColorSpace(Enum):
    """Supported color spaces for detection"""
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"
    YCrCb = "ycrcb"


class CubeColor(Enum):
    """Standard cube colors to detect"""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    WHITE = "white"
    BLACK = "black"


@dataclass
class ColorThresholds:
    """
    Color threshold ranges for detection
    
    For HSV: [H_min, S_min, V_min], [H_max, S_max, V_max]
    H: 0-179, S: 0-255, V: 0-255
    """
    lower: np.ndarray
    upper: np.ndarray
    color_space: ColorSpace = ColorSpace.HSV
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'lower': self.lower.tolist(),
            'upper': self.upper.tolist(),
            'color_space': self.color_space.value,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorThresholds':
        """Create from dictionary"""
        return cls(
            lower=np.array(data['lower']),
            upper=np.array(data['upper']),
            color_space=ColorSpace(data.get('color_space', 'hsv')),
            confidence=data.get('confidence', 1.0)
        )


# ============================================================================
# Default Color Thresholds (No Tint)
# ============================================================================

def get_default_thresholds() -> Dict[CubeColor, ColorThresholds]:
    """
    Get default color thresholds for standard lighting
    
    These are baseline values for typical indoor lighting
    without projector interference.
    """
    return {
        CubeColor.RED: ColorThresholds(
            lower=np.array([0, 100, 100]),
            upper=np.array([10, 255, 255]),
            color_space=ColorSpace.HSV
        ),
        CubeColor.GREEN: ColorThresholds(
            lower=np.array([40, 100, 100]),
            upper=np.array([80, 255, 255]),
            color_space=ColorSpace.HSV
        ),
        CubeColor.BLUE: ColorThresholds(
            lower=np.array([100, 100, 100]),
            upper=np.array([130, 255, 255]),
            color_space=ColorSpace.HSV
        ),
        CubeColor.YELLOW: ColorThresholds(
            lower=np.array([20, 100, 100]),
            upper=np.array([40, 255, 255]),
            color_space=ColorSpace.HSV
        ),
        CubeColor.ORANGE: ColorThresholds(
            lower=np.array([10, 100, 100]),
            upper=np.array([25, 255, 255]),
            color_space=ColorSpace.HSV
        ),
        CubeColor.PURPLE: ColorThresholds(
            lower=np.array([130, 100, 100]),
            upper=np.array([160, 255, 255]),
            color_space=ColorSpace.HSV
        ),
        CubeColor.WHITE: ColorThresholds(
            lower=np.array([0, 0, 200]),
            upper=np.array([179, 30, 255]),
            color_space=ColorSpace.HSV
        ),
        CubeColor.BLACK: ColorThresholds(
            lower=np.array([0, 0, 0]),
            upper=np.array([179, 255, 50]),
            color_space=ColorSpace.HSV
        )
    }


# ============================================================================
# Projector Tint Compensation
# ============================================================================

class ProjectorTintCompensator:
    """
    Compensates for color shifts caused by venue projectors
    
    Common projector issues:
    - Blue/green tint from LCD projectors
    - Yellow/orange tint from older DLP projectors
    - Purple tint from LED projectors
    - Uneven color temperature across display
    """
    
    def __init__(self):
        """Initialize tint compensator"""
        self.tint_profiles = {
            'none': np.array([0, 0, 0]),
            'blue_tint': np.array([10, -10, 20]),     # Shift hue, reduce sat, increase val
            'yellow_tint': np.array([-10, 10, -10]),   # Opposite shifts
            'green_tint': np.array([5, -15, 15]),
            'purple_tint': np.array([15, -5, 10]),
            'warm_light': np.array([-15, 5, -5]),      # Tungsten lighting
            'cool_light': np.array([5, -5, 5]),        # Fluorescent lighting
            'outdoor': np.array([0, -20, 20]),         # Bright outdoor light
            'dim_venue': np.array([0, -10, -30])       # Dim conference room
        }
        
        self.current_profile = 'none'
        self.custom_adjustment = np.array([0, 0, 0])
        
    def detect_tint(self, image: np.ndarray) -> str:
        """
        Auto-detect projector tint from image
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Detected tint profile name
        """
        # Convert to HSV for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate average color in center region
        h, w = image.shape[:2]
        center = hsv[h//4:3*h//4, w//4:3*w//4]
        
        avg_hue = np.mean(center[:, :, 0])
        avg_sat = np.mean(center[:, :, 1])
        avg_val = np.mean(center[:, :, 2])
        
        # Detect dominant tint
        if avg_val < 100:
            return 'dim_venue'
        elif avg_sat < 50:
            return 'outdoor' if avg_val > 200 else 'cool_light'
        elif 90 < avg_hue < 110:  # Blue dominant
            return 'blue_tint'
        elif 20 < avg_hue < 40:  # Yellow dominant
            return 'yellow_tint'
        elif 40 < avg_hue < 80:  # Green dominant
            return 'green_tint'
        elif 130 < avg_hue < 160:  # Purple dominant
            return 'purple_tint'
        elif avg_hue < 20:  # Red/warm dominant
            return 'warm_light'
        
        return 'none'
    
    def compensate_thresholds(self, thresholds: ColorThresholds,
                            profile: Optional[str] = None) -> ColorThresholds:
        """
        Compensate color thresholds for tint
        
        Args:
            thresholds: Original thresholds
            profile: Tint profile to apply (uses current if None)
            
        Returns:
            Compensated thresholds
        """
        if profile is None:
            profile = self.current_profile
        
        if profile not in self.tint_profiles:
            logger.warning(f"Unknown tint profile: {profile}")
            return thresholds
        
        # Get adjustment vector
        adjustment = self.tint_profiles[profile] + self.custom_adjustment
        
        # Apply to HSV thresholds
        if thresholds.color_space == ColorSpace.HSV:
            # Adjust hue (circular)
            lower = thresholds.lower.copy()
            upper = thresholds.upper.copy()
            
            lower[0] = (lower[0] + adjustment[0]) % 180
            upper[0] = (upper[0] + adjustment[0]) % 180
            
            # Adjust saturation
            lower[1] = np.clip(lower[1] + adjustment[1], 0, 255)
            upper[1] = np.clip(upper[1] + adjustment[1], 0, 255)
            
            # Adjust value
            lower[2] = np.clip(lower[2] + adjustment[2], 0, 255)
            upper[2] = np.clip(upper[2] + adjustment[2], 0, 255)
            
            return ColorThresholds(lower, upper, ColorSpace.HSV, thresholds.confidence * 0.9)
        
        return thresholds
    
    def apply_tint_correction(self, image: np.ndarray,
                            profile: Optional[str] = None) -> np.ndarray:
        """
        Apply tint correction to image
        
        Args:
            image: Input image (BGR)
            profile: Tint profile to correct
            
        Returns:
            Corrected image
        """
        if profile is None:
            profile = self.current_profile
        
        if profile == 'none':
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Get correction (opposite of tint)
        correction = -self.tint_profiles.get(profile, np.array([0, 0, 0]))
        
        # Apply correction
        hsv[:, :, 0] = (hsv[:, :, 0] + correction[0]) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + correction[1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + correction[2], 0, 255)
        
        # Convert back to BGR
        hsv = hsv.astype(np.uint8)
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return corrected


# ============================================================================
# Interactive Calibration Tool
# ============================================================================

class ColorCalibrationTool:
    """
    Interactive tool for calibrating color thresholds
    
    Features:
    - Live preview with sliders
    - Auto-detection assistance
    - Save/load calibration profiles
    - Multiple color space support
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize calibration tool
        
        Args:
            camera_index: Camera device index
        """
        self.camera_index = camera_index
        self.cap = None
        self.current_color = CubeColor.RED
        self.thresholds = get_default_thresholds()
        self.tint_compensator = ProjectorTintCompensator()
        self.is_running = False
        
        # Calibration history
        self.calibration_history = []
        self.sample_regions = []
        
    def start_camera(self) -> bool:
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return True
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def sample_color_region(self, image: np.ndarray,
                           region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Sample color from image region
        
        Args:
            image: Input image (BGR)
            region: (x, y, width, height)
            
        Returns:
            Average HSV color
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate average color
        avg_color = np.mean(hsv_roi.reshape(-1, 3), axis=0)
        
        return avg_color
    
    def auto_detect_thresholds(self, image: np.ndarray,
                              sample_point: Tuple[int, int],
                              tolerance: int = 20) -> ColorThresholds:
        """
        Auto-detect color thresholds from sample point
        
        Args:
            image: Input image (BGR)
            sample_point: (x, y) point to sample
            tolerance: Threshold tolerance
            
        Returns:
            Detected thresholds
        """
        # Sample region around point
        x, y = sample_point
        region_size = 20
        region = (
            max(0, x - region_size//2),
            max(0, y - region_size//2),
            region_size,
            region_size
        )
        
        # Get average color
        avg_hsv = self.sample_color_region(image, region)
        
        # Create thresholds with tolerance
        lower = np.array([
            max(0, avg_hsv[0] - tolerance),
            max(0, avg_hsv[1] - tolerance * 2),
            max(0, avg_hsv[2] - tolerance * 2)
        ])
        
        upper = np.array([
            min(179, avg_hsv[0] + tolerance),
            min(255, avg_hsv[1] + tolerance * 2),
            min(255, avg_hsv[2] + tolerance * 2)
        ])
        
        return ColorThresholds(lower, upper, ColorSpace.HSV)
    
    def detect_cubes(self, image: np.ndarray,
                    color: CubeColor) -> List[Dict[str, Any]]:
        """
        Detect colored cubes in image
        
        Args:
            image: Input image (BGR)
            color: Cube color to detect
            
        Returns:
            List of detected cubes with properties
        """
        # Get thresholds (with tint compensation)
        thresholds = self.thresholds[color]
        thresholds = self.tint_compensator.compensate_thresholds(thresholds)
        
        # Convert to appropriate color space
        if thresholds.color_space == ColorSpace.HSV:
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif thresholds.color_space == ColorSpace.LAB:
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            converted = image
        
        # Create mask
        mask = cv2.inRange(converted, thresholds.lower, thresholds.upper)
        
        # Noise reduction
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process detected cubes
        cubes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < 500:  # Too small
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (cubes should be roughly square)
            aspect_ratio = w / h
            if not (0.7 < aspect_ratio < 1.3):
                continue
            
            # Get center
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            cubes.append({
                'color': color.value,
                'bbox': [x, y, w, h],
                'center': [cx, cy],
                'area': area,
                'confidence': thresholds.confidence,
                'contour': contour
            })
        
        return cubes
    
    def save_calibration(self, filename: str):
        """
        Save calibration to file
        
        Args:
            filename: Output filename
        """
        data = {
            'tint_profile': self.tint_compensator.current_profile,
            'custom_adjustment': self.tint_compensator.custom_adjustment.tolist(),
            'thresholds': {
                color.value: thresh.to_dict()
                for color, thresh in self.thresholds.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename: str):
        """
        Load calibration from file
        
        Args:
            filename: Input filename
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Load tint settings
        self.tint_compensator.current_profile = data.get('tint_profile', 'none')
        self.tint_compensator.custom_adjustment = np.array(
            data.get('custom_adjustment', [0, 0, 0])
        )
        
        # Load thresholds
        for color_name, thresh_data in data.get('thresholds', {}).items():
            try:
                color = CubeColor(color_name)
                self.thresholds[color] = ColorThresholds.from_dict(thresh_data)
            except ValueError:
                logger.warning(f"Unknown color: {color_name}")
        
        logger.info(f"Calibration loaded from {filename}")


# ============================================================================
# Calibration GUI
# ============================================================================

class CalibrationGUI:
    """
    Graphical interface for color calibration
    
    Provides sliders and live preview for threshold adjustment
    """
    
    def __init__(self, calibration_tool: ColorCalibrationTool):
        """
        Initialize GUI
        
        Args:
            calibration_tool: Calibration tool instance
        """
        self.tool = calibration_tool
        self.root = tk.Tk()
        self.root.title("Cube Color Calibration")
        self.root.geometry("400x600")
        
        # Variables
        self.current_color_var = tk.StringVar(value="red")
        self.tint_profile_var = tk.StringVar(value="none")
        
        # HSV sliders
        self.h_min = tk.IntVar(value=0)
        self.h_max = tk.IntVar(value=179)
        self.s_min = tk.IntVar(value=0)
        self.s_max = tk.IntVar(value=255)
        self.v_min = tk.IntVar(value=0)
        self.v_max = tk.IntVar(value=255)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create GUI widgets"""
        # Color selection
        color_frame = ttk.LabelFrame(self.root, text="Cube Color", padding=10)
        color_frame.pack(fill="x", padx=10, pady=5)
        
        for color in CubeColor:
            ttk.Radiobutton(
                color_frame, 
                text=color.value.capitalize(),
                variable=self.current_color_var,
                value=color.value,
                command=self.on_color_change
            ).pack(side="left", padx=5)
        
        # Tint compensation
        tint_frame = ttk.LabelFrame(self.root, text="Projector Tint", padding=10)
        tint_frame.pack(fill="x", padx=10, pady=5)
        
        profiles = ['none', 'blue_tint', 'yellow_tint', 'green_tint', 
                   'purple_tint', 'warm_light', 'cool_light', 'dim_venue']
        
        ttk.Combobox(
            tint_frame,
            textvariable=self.tint_profile_var,
            values=profiles,
            state="readonly"
        ).pack(fill="x")
        
        # HSV sliders
        threshold_frame = ttk.LabelFrame(self.root, text="HSV Thresholds", padding=10)
        threshold_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Hue
        ttk.Label(threshold_frame, text="Hue Min:").grid(row=0, column=0, sticky="w")
        ttk.Scale(threshold_frame, from_=0, to=179, variable=self.h_min,
                 orient="horizontal").grid(row=0, column=1, sticky="ew")
        ttk.Label(threshold_frame, textvariable=self.h_min).grid(row=0, column=2)
        
        ttk.Label(threshold_frame, text="Hue Max:").grid(row=1, column=0, sticky="w")
        ttk.Scale(threshold_frame, from_=0, to=179, variable=self.h_max,
                 orient="horizontal").grid(row=1, column=1, sticky="ew")
        ttk.Label(threshold_frame, textvariable=self.h_max).grid(row=1, column=2)
        
        # Saturation
        ttk.Label(threshold_frame, text="Sat Min:").grid(row=2, column=0, sticky="w")
        ttk.Scale(threshold_frame, from_=0, to=255, variable=self.s_min,
                 orient="horizontal").grid(row=2, column=1, sticky="ew")
        ttk.Label(threshold_frame, textvariable=self.s_min).grid(row=2, column=2)
        
        ttk.Label(threshold_frame, text="Sat Max:").grid(row=3, column=0, sticky="w")
        ttk.Scale(threshold_frame, from_=0, to=255, variable=self.s_max,
                 orient="horizontal").grid(row=3, column=1, sticky="ew")
        ttk.Label(threshold_frame, textvariable=self.s_max).grid(row=3, column=2)
        
        # Value
        ttk.Label(threshold_frame, text="Val Min:").grid(row=4, column=0, sticky="w")
        ttk.Scale(threshold_frame, from_=0, to=255, variable=self.v_min,
                 orient="horizontal").grid(row=4, column=1, sticky="ew")
        ttk.Label(threshold_frame, textvariable=self.v_min).grid(row=4, column=2)
        
        ttk.Label(threshold_frame, text="Val Max:").grid(row=5, column=0, sticky="w")
        ttk.Scale(threshold_frame, from_=0, to=255, variable=self.v_max,
                 orient="horizontal").grid(row=5, column=1, sticky="ew")
        ttk.Label(threshold_frame, textvariable=self.v_max).grid(row=5, column=2)
        
        threshold_frame.columnconfigure(1, weight=1)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Auto Detect",
                  command=self.auto_detect).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Apply",
                  command=self.apply_thresholds).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save",
                  command=self.save_config).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load",
                  command=self.load_config).pack(side="left", padx=5)
    
    def on_color_change(self):
        """Handle color selection change"""
        color = CubeColor(self.current_color_var.get())
        thresholds = self.tool.thresholds[color]
        
        # Update sliders
        self.h_min.set(int(thresholds.lower[0]))
        self.h_max.set(int(thresholds.upper[0]))
        self.s_min.set(int(thresholds.lower[1]))
        self.s_max.set(int(thresholds.upper[1]))
        self.v_min.set(int(thresholds.lower[2]))
        self.v_max.set(int(thresholds.upper[2]))
    
    def apply_thresholds(self):
        """Apply current slider values"""
        color = CubeColor(self.current_color_var.get())
        
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
        upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        
        self.tool.thresholds[color] = ColorThresholds(lower, upper, ColorSpace.HSV)
        self.tool.tint_compensator.current_profile = self.tint_profile_var.get()
        
        messagebox.showinfo("Applied", f"Thresholds updated for {color.value}")
    
    def auto_detect(self):
        """Auto-detect thresholds (placeholder for mouse callback)"""
        messagebox.showinfo("Auto Detect", 
                          "Click on a cube in the camera view to auto-detect its color")
    
    def save_config(self):
        """Save calibration configuration"""
        filename = f"calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
        self.tool.save_calibration(filename)
        messagebox.showinfo("Saved", f"Calibration saved to {filename}")
    
    def load_config(self):
        """Load calibration configuration"""
        # For simplicity, load the most recent calibration
        try:
            import glob
            files = glob.glob("calibration_*.json")
            if files:
                latest = max(files)
                self.tool.load_calibration(latest)
                self.on_color_change()  # Update GUI
                messagebox.showinfo("Loaded", f"Calibration loaded from {latest}")
            else:
                messagebox.showwarning("No Files", "No calibration files found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
    
    def run(self):
        """Run GUI"""
        self.root.mainloop()


# ============================================================================
# Live Calibration View
# ============================================================================

def run_live_calibration(camera_index: int = 0):
    """
    Run live calibration with camera feed
    
    Args:
        camera_index: Camera device index
    """
    print("\n" + "="*70)
    print(" CUBE COLOR CALIBRATION")
    print("="*70)
    print("\nStarting live calibration...")
    print("Controls:")
    print("  - Click on cube to auto-detect color")
    print("  - Press 'c' to cycle cube colors")
    print("  - Press 't' to cycle tint profiles")
    print("  - Press 's' to save calibration")
    print("  - Press 'q' to quit")
    print("-"*70)
    
    # Initialize tool
    tool = ColorCalibrationTool(camera_index)
    if not tool.start_camera():
        print("Failed to start camera")
        return
    
    # Current settings
    current_color_idx = 0
    colors = list(CubeColor)
    current_color = colors[current_color_idx]
    
    tint_profiles = list(tool.tint_compensator.tint_profiles.keys())
    current_tint_idx = 0
    
    # Mouse callback for auto-detection
    sample_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal sample_point
        if event == cv2.EVENT_LBUTTONDOWN:
            sample_point = (x, y)
    
    cv2.namedWindow('Original')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Detection')
    cv2.setMouseCallback('Original', mouse_callback)
    
    while True:
        ret, frame = tool.cap.read()
        if not ret:
            continue
        
        # Auto-detect if clicked
        if sample_point:
            thresholds = tool.auto_detect_thresholds(frame, sample_point, tolerance=25)
            tool.thresholds[current_color] = thresholds
            print(f"Auto-detected {current_color.value}: "
                  f"H[{thresholds.lower[0]}-{thresholds.upper[0]}] "
                  f"S[{thresholds.lower[1]}-{thresholds.upper[1]}] "
                  f"V[{thresholds.lower[2]}-{thresholds.upper[2]}]")
            sample_point = None
        
        # Apply tint compensation to display
        tool.tint_compensator.current_profile = tint_profiles[current_tint_idx]
        
        # Detect cubes
        cubes = tool.detect_cubes(frame, current_color)
        
        # Create mask for visualization
        thresholds = tool.tint_compensator.compensate_thresholds(
            tool.thresholds[current_color]
        )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, thresholds.lower, thresholds.upper)
        
        # Draw detections
        detection_frame = frame.copy()
        for cube in cubes:
            x, y, w, h = cube['bbox']
            cv2.rectangle(detection_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(detection_frame, tuple(cube['center']), 5, (0, 0, 255), -1)
            cv2.putText(detection_frame, f"{cube['color']} ({cube['confidence']:.2f})",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add info text
        cv2.putText(detection_frame, f"Color: {current_color.value}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(detection_frame, f"Tint: {tint_profiles[current_tint_idx]}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(detection_frame, f"Detected: {len(cubes)} cubes", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show windows
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Detection', detection_frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Cycle colors
            current_color_idx = (current_color_idx + 1) % len(colors)
            current_color = colors[current_color_idx]
            print(f"Switched to {current_color.value}")
        elif key == ord('t'):
            # Cycle tint profiles
            current_tint_idx = (current_tint_idx + 1) % len(tint_profiles)
            print(f"Switched to {tint_profiles[current_tint_idx]} profile")
        elif key == ord('s'):
            # Save calibration
            filename = f"venue_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
            tool.save_calibration(filename)
            print(f"Saved calibration to {filename}")
    
    # Cleanup
    tool.stop_camera()
    cv2.destroyAllWindows()


# ============================================================================
# Venue-Specific Presets
# ============================================================================

def get_venue_presets() -> Dict[str, Dict[CubeColor, ColorThresholds]]:
    """
    Get pre-calibrated settings for common venue types
    
    Returns:
        Dictionary of venue presets
    """
    return {
        'conference_room_projector': {
            # Typical conference room with overhead projector (blue tint)
            CubeColor.RED: ColorThresholds(
                lower=np.array([170, 80, 80]),  # Shifted due to blue tint
                upper=np.array([10, 255, 255]),
                color_space=ColorSpace.HSV
            ),
            CubeColor.GREEN: ColorThresholds(
                lower=np.array([45, 80, 80]),
                upper=np.array([85, 255, 255]),
                color_space=ColorSpace.HSV
            ),
            CubeColor.BLUE: ColorThresholds(
                lower=np.array([95, 80, 80]),  # Enhanced by blue tint
                upper=np.array([125, 255, 255]),
                color_space=ColorSpace.HSV
            ),
        },
        'auditorium_stage': {
            # Stage lighting with warm spotlights (yellow/orange tint)
            CubeColor.RED: ColorThresholds(
                lower=np.array([0, 120, 100]),
                upper=np.array([15, 255, 255]),
                color_space=ColorSpace.HSV
            ),
            CubeColor.GREEN: ColorThresholds(
                lower=np.array([35, 100, 90]),  # Shifted toward yellow
                upper=np.array([75, 255, 255]),
                color_space=ColorSpace.HSV
            ),
            CubeColor.BLUE: ColorThresholds(
                lower=np.array([105, 100, 90]),  # Diminished by warm light
                upper=np.array([135, 255, 255]),
                color_space=ColorSpace.HSV
            ),
        },
        'outdoor_demo': {
            # Bright outdoor lighting (low saturation)
            CubeColor.RED: ColorThresholds(
                lower=np.array([0, 50, 120]),  # Lower saturation
                upper=np.array([10, 255, 255]),
                color_space=ColorSpace.HSV
            ),
            CubeColor.GREEN: ColorThresholds(
                lower=np.array([40, 50, 120]),
                upper=np.array([80, 255, 255]),
                color_space=ColorSpace.HSV
            ),
            CubeColor.BLUE: ColorThresholds(
                lower=np.array([100, 50, 120]),
                upper=np.array([130, 255, 255]),
                color_space=ColorSpace.HSV
            ),
        }
    }


# ============================================================================
# Quick Calibration Script
# ============================================================================

def quick_calibrate_for_venue():
    """
    Quick calibration script for venue setup
    
    Run this at the venue before demo
    """
    print("\n" + "="*70)
    print(" QUICK VENUE CALIBRATION")
    print("="*70)
    
    print("\nSelect venue type:")
    print("1. Conference room with projector")
    print("2. Auditorium with stage lighting")
    print("3. Outdoor demonstration")
    print("4. Custom calibration")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ['1', '2', '3']:
        # Use preset
        presets = get_venue_presets()
        preset_names = list(presets.keys())
        preset_name = preset_names[int(choice) - 1]
        
        print(f"\nUsing preset: {preset_name}")
        
        # Save preset as current calibration
        tool = ColorCalibrationTool()
        for color, thresholds in presets[preset_name].items():
            tool.thresholds[color] = thresholds
        
        # Detect tint profile
        if 'projector' in preset_name:
            tool.tint_compensator.current_profile = 'blue_tint'
        elif 'stage' in preset_name:
            tool.tint_compensator.current_profile = 'warm_light'
        elif 'outdoor' in preset_name:
            tool.tint_compensator.current_profile = 'outdoor'
        
        # Save
        filename = f"venue_calibration_{preset_name}.json"
        tool.save_calibration(filename)
        print(f"✓ Calibration saved to {filename}")
        
    else:
        # Run live calibration
        print("\nStarting custom calibration...")
        run_live_calibration()
    
    print("\n✓ Venue calibration complete!")
    print("  Use the saved calibration file during your demo")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" COLOR CALIBRATION FOR VENUE PROJECTORS")
    print("="*70)
    
    print("\nThis tool helps compensate for color tinting from venue projectors")
    print("and lighting conditions that affect cube detection.\n")
    
    print("Options:")
    print("1. Quick venue calibration (recommended at venue)")
    print("2. Live calibration with camera")
    print("3. Test calibration files")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        quick_calibrate_for_venue()
    elif choice == '2':
        run_live_calibration()
    else:
        # Test existing calibration
        import glob
        files = glob.glob("*calibration*.json")
        
        if not files:
            print("No calibration files found")
        else:
            print("\nAvailable calibration files:")
            for i, f in enumerate(files):
                print(f"  {i+1}. {f}")
            
            idx = int(input("Select file number: ")) - 1
            if 0 <= idx < len(files):
                tool = ColorCalibrationTool()
                tool.load_calibration(files[idx])
                print(f"\n✓ Loaded calibration from {files[idx]}")
                print(f"  Tint profile: {tool.tint_compensator.current_profile}")
                print(f"  Colors calibrated: {len(tool.thresholds)}")
    
    print("\n" + "="*70)
    print(" CALIBRATION TIPS")
    print("="*70)
    print("\n• Always calibrate at the actual venue if possible")
    print("• Test with different cube positions and angles")
    print("• Account for both projector tint AND ambient lighting")
    print("• Save multiple profiles for different venue sections")
    print("• Use auto-detect by clicking on cubes for quick setup")
    print("\n✓ Good color calibration is critical for reliable cube detection!")