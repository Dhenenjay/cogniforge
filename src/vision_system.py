"""
Vision System with Pixel-to-Meter Conversion and Nudge Logging

This module provides computer vision capabilities for robotic manipulation,
displaying displacements in both pixels and meters, and logging final nudge adjustments.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
from pathlib import Path
from colorama import init, Fore, Style, Back
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow
import threading
from collections import deque
import logging

# Initialize colorama
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_nudge_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('VisionSystem')


@dataclass
class CameraCalibration:
    """Camera calibration parameters"""
    # Intrinsic parameters
    focal_length_x: float = 800.0  # pixels
    focal_length_y: float = 800.0  # pixels
    principal_point_x: float = 320.0  # pixels
    principal_point_y: float = 240.0  # pixels
    
    # Extrinsic parameters
    camera_height: float = 0.5  # meters from work surface
    
    # Pixel to meter conversion (at work surface)
    pixels_per_meter_x: float = 1600.0  # pixels/meter at z=camera_height
    pixels_per_meter_y: float = 1600.0  # pixels/meter at z=camera_height
    
    # Distortion coefficients
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    def pixel_to_meter(self, dx_px: float, dy_px: float, 
                      z_distance: Optional[float] = None) -> Tuple[float, float]:
        """
        Convert pixel displacement to meters
        
        Args:
            dx_px: Horizontal displacement in pixels
            dy_px: Vertical displacement in pixels
            z_distance: Distance from camera (uses camera_height if None)
            
        Returns:
            (dx_m, dy_m): Displacement in meters
        """
        if z_distance is None:
            z_distance = self.camera_height
            
        # Account for perspective (further objects appear smaller)
        scale_factor = z_distance / self.camera_height
        
        dx_m = (dx_px / self.pixels_per_meter_x) * scale_factor
        dy_m = (dy_px / self.pixels_per_meter_y) * scale_factor
        
        return dx_m, dy_m
    
    def meter_to_pixel(self, dx_m: float, dy_m: float,
                      z_distance: Optional[float] = None) -> Tuple[float, float]:
        """
        Convert meter displacement to pixels
        
        Args:
            dx_m: Horizontal displacement in meters
            dy_m: Vertical displacement in meters
            z_distance: Distance from camera
            
        Returns:
            (dx_px, dy_px): Displacement in pixels
        """
        if z_distance is None:
            z_distance = self.camera_height
            
        scale_factor = self.camera_height / z_distance
        
        dx_px = dx_m * self.pixels_per_meter_x * scale_factor
        dy_px = dy_m * self.pixels_per_meter_y * scale_factor
        
        return dx_px, dy_px


@dataclass
class ObjectDetection:
    """Object detection result"""
    object_id: str
    class_name: str
    confidence: float
    bbox_px: Tuple[int, int, int, int]  # x, y, width, height in pixels
    center_px: Tuple[float, float]  # center in pixels
    center_m: Tuple[float, float]  # center in meters
    timestamp: float = field(default_factory=time.time)


@dataclass
class NudgeCommand:
    """Final nudge adjustment command"""
    iteration: int
    dx_px: float  # Displacement in pixels
    dy_px: float
    dx_m: float   # Displacement in meters
    dy_m: float
    distance_px: float  # Total distance
    distance_m: float
    direction_deg: float  # Direction in degrees
    timestamp: datetime
    reason: str  # Why nudge was needed
    success: bool = False


class DisplacementVisualizer:
    """Real-time visualization of displacements and nudges"""
    
    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        """
        Initialize displacement visualizer
        
        Args:
            image_size: (width, height) of visualization
        """
        self.image_size = image_size
        self.fig = None
        self.ax_main = None
        self.ax_info = None
        self.displacement_history = deque(maxlen=50)
        self.nudge_history = []
        
    def create_display(self):
        """Create the visualization display"""
        self.fig = plt.figure(figsize=(14, 8), facecolor='#f5f5f5')
        self.fig.suptitle('🎯 Vision System - Displacement & Nudge Monitor', 
                         fontsize=16, fontweight='bold')
        
        # Main visualization
        self.ax_main = self.fig.add_subplot(121)
        self.ax_main.set_xlim(0, self.image_size[0])
        self.ax_main.set_ylim(self.image_size[1], 0)  # Inverted Y for image coordinates
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlabel('X (pixels)', fontsize=11)
        self.ax_main.set_ylabel('Y (pixels)', fontsize=11)
        self.ax_main.set_title('📹 Camera View with Displacements', fontsize=12, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        self.ax_main.set_facecolor('#ffffff')
        
        # Info panel
        self.ax_info = self.fig.add_subplot(122)
        self.ax_info.axis('off')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.set_title('📊 Displacement Information', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)
        
    def update_displacement(self, current_pos: Tuple[float, float],
                           target_pos: Tuple[float, float],
                           calibration: CameraCalibration):
        """
        Update displacement visualization
        
        Args:
            current_pos: Current position in pixels (x, y)
            target_pos: Target position in pixels (x, y)
            calibration: Camera calibration for conversion
        """
        if self.fig is None:
            self.create_display()
            
        # Clear previous
        self.ax_main.clear()
        self.ax_main.set_xlim(0, self.image_size[0])
        self.ax_main.set_ylim(self.image_size[1], 0)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlabel('X (pixels)', fontsize=11)
        self.ax_main.set_ylabel('Y (pixels)', fontsize=11)
        self.ax_main.set_title('📹 Camera View with Displacements', fontsize=12, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Calculate displacement
        dx_px = target_pos[0] - current_pos[0]
        dy_px = target_pos[1] - current_pos[1]
        distance_px = np.sqrt(dx_px**2 + dy_px**2)
        
        # Convert to meters
        dx_m, dy_m = calibration.pixel_to_meter(dx_px, dy_px)
        distance_m = np.sqrt(dx_m**2 + dy_m**2)
        
        # Direction
        direction_deg = np.degrees(np.arctan2(dy_px, dx_px))
        
        # Draw current position
        current_circle = Circle(current_pos, 10, color='blue', alpha=0.7, label='Current')
        self.ax_main.add_patch(current_circle)
        self.ax_main.text(current_pos[0], current_pos[1] - 20, 'Current', 
                          ha='center', fontsize=9, color='blue', fontweight='bold')
        
        # Draw target position
        target_circle = Circle(target_pos, 10, color='red', alpha=0.7, label='Target')
        self.ax_main.add_patch(target_circle)
        self.ax_main.text(target_pos[0], target_pos[1] - 20, 'Target', 
                          ha='center', fontsize=9, color='red', fontweight='bold')
        
        # Draw displacement vector
        if distance_px > 0:
            arrow = plt.Arrow(current_pos[0], current_pos[1], 
                            dx_px * 0.9, dy_px * 0.9,
                            width=20, color='green', alpha=0.6)
            self.ax_main.add_patch(arrow)
            
            # Add displacement text at midpoint
            mid_x = (current_pos[0] + target_pos[0]) / 2
            mid_y = (current_pos[1] + target_pos[1]) / 2
            self.ax_main.text(mid_x, mid_y - 10, 
                            f'{distance_px:.1f}px\n{distance_m*1000:.1f}mm',
                            ha='center', fontsize=10, color='green', 
                            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                                                        facecolor='white', 
                                                        edgecolor='green',
                                                        alpha=0.8))
        
        # Draw history trail
        self.displacement_history.append((current_pos, target_pos, time.time()))
        for i, (c_pos, t_pos, _) in enumerate(self.displacement_history):
            alpha = (i + 1) / len(self.displacement_history) * 0.3
            self.ax_main.plot([c_pos[0], t_pos[0]], [c_pos[1], t_pos[1]], 
                            'g--', alpha=alpha, linewidth=1)
        
        # Update info panel
        self._update_info_panel(dx_px, dy_px, dx_m, dy_m, distance_px, 
                               distance_m, direction_deg)
        
        # Log displacement
        print(f"{Fore.CYAN}Displacement: "
              f"dx={dx_px:.1f}px ({dx_m*1000:.1f}mm), "
              f"dy={dy_px:.1f}px ({dy_m*1000:.1f}mm), "
              f"distance={distance_px:.1f}px ({distance_m*1000:.1f}mm){Style.RESET_ALL}")
        
        plt.draw()
        plt.pause(0.001)
        
    def _update_info_panel(self, dx_px: float, dy_px: float, 
                          dx_m: float, dy_m: float,
                          distance_px: float, distance_m: float, 
                          direction_deg: float):
        """Update the information panel"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        
        # Title
        self.ax_info.text(0.5, 0.95, 'Displacement Measurements', 
                         ha='center', fontsize=13, fontweight='bold')
        
        # Pixel measurements
        self.ax_info.text(0.1, 0.85, 'Pixels:', fontsize=11, fontweight='bold')
        self.ax_info.text(0.15, 0.80, f'dx: {dx_px:+.1f} px', fontsize=10)
        self.ax_info.text(0.15, 0.75, f'dy: {dy_px:+.1f} px', fontsize=10)
        self.ax_info.text(0.15, 0.70, f'distance: {distance_px:.1f} px', fontsize=10)
        
        # Meter measurements
        self.ax_info.text(0.1, 0.60, 'Meters:', fontsize=11, fontweight='bold', color='green')
        self.ax_info.text(0.15, 0.55, f'dx: {dx_m*1000:+.2f} mm', fontsize=10, color='green')
        self.ax_info.text(0.15, 0.50, f'dy: {dy_m*1000:+.2f} mm', fontsize=10, color='green')
        self.ax_info.text(0.15, 0.45, f'distance: {distance_m*1000:.2f} mm', fontsize=10, color='green')
        
        # Direction
        self.ax_info.text(0.1, 0.35, f'Direction: {direction_deg:+.1f}°', 
                         fontsize=11, fontweight='bold', color='blue')
        
        # Visual representation
        self.ax_info.text(0.1, 0.25, 'Visual Scale:', fontsize=11, fontweight='bold')
        
        # Draw scale reference
        scale_x = 0.15
        scale_y = 0.15
        scale_width = 0.3
        
        # 10cm reference bar
        ref_10cm_px = 10 * 1600 / 100  # 10cm in pixels at standard distance
        scale_factor = scale_width / ref_10cm_px
        
        # Draw scale bar
        rect = Rectangle((scale_x, scale_y), scale_width, 0.02, 
                        facecolor='black', alpha=0.8)
        self.ax_info.add_patch(rect)
        self.ax_info.text(scale_x + scale_width/2, scale_y - 0.02, 
                         '10 cm', ha='center', fontsize=9)
        
        # Draw current displacement on scale
        if distance_m > 0:
            disp_width = min(distance_m * 10 * scale_width, 0.7)  # Scale to 10cm reference
            rect_disp = Rectangle((scale_x, scale_y + 0.04), disp_width, 0.02,
                                 facecolor='green', alpha=0.6)
            self.ax_info.add_patch(rect_disp)
            self.ax_info.text(scale_x + disp_width/2, scale_y + 0.07,
                            f'{distance_m*100:.1f} cm', ha='center', fontsize=9, color='green')
        
        # Nudge history summary
        if self.nudge_history:
            self.ax_info.text(0.6, 0.85, 'Recent Nudges:', fontsize=11, fontweight='bold')
            for i, nudge in enumerate(self.nudge_history[-3:]):
                y_pos = 0.80 - i * 0.1
                color = 'green' if nudge.success else 'red'
                self.ax_info.text(0.65, y_pos, 
                                f'#{nudge.iteration}: {nudge.distance_m*1000:.1f}mm',
                                fontsize=9, color=color)
    
    def log_nudge(self, nudge: NudgeCommand):
        """Log a nudge command"""
        self.nudge_history.append(nudge)


class VisionSystem:
    """Main vision system for robotic manipulation"""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None):
        """
        Initialize vision system
        
        Args:
            calibration: Camera calibration parameters
        """
        self.calibration = calibration or CameraCalibration()
        self.visualizer = DisplacementVisualizer()
        self.current_frame = None
        self.detections = []
        self.nudge_log = []
        self.nudge_iteration = 0
        
        # Create log directory
        self.log_dir = Path("vision_logs")
        self.log_dir.mkdir(exist_ok=True)
        
    def process_frame(self, frame: np.ndarray) -> List[ObjectDetection]:
        """
        Process a camera frame and detect objects
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected objects
        """
        self.current_frame = frame.copy()
        
        # Simulate object detection (replace with actual detection)
        # In real implementation, this would use YOLO, Detectron2, etc.
        detections = self._simulate_detection(frame)
        
        self.detections = detections
        return detections
    
    def _simulate_detection(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Simulate object detection for demo"""
        h, w = frame.shape[:2]
        
        # Simulate detecting an object
        detection = ObjectDetection(
            object_id="obj_001",
            class_name="cube",
            confidence=0.95,
            bbox_px=(w//2 - 50, h//2 - 50, 100, 100),
            center_px=(w//2, h//2),
            center_m=self.calibration.pixel_to_meter(w//2, h//2)
        )
        
        return [detection]
    
    def calculate_displacement(self, current_pos: Tuple[float, float],
                              target_pos: Tuple[float, float],
                              show_visualization: bool = True) -> Dict[str, float]:
        """
        Calculate displacement between current and target positions
        
        Args:
            current_pos: Current position in pixels (x, y)
            target_pos: Target position in pixels (x, y)
            show_visualization: Whether to show visualization
            
        Returns:
            Dictionary with displacement information
        """
        # Calculate in pixels
        dx_px = target_pos[0] - current_pos[0]
        dy_px = target_pos[1] - current_pos[1]
        distance_px = np.sqrt(dx_px**2 + dy_px**2)
        
        # Convert to meters
        dx_m, dy_m = self.calibration.pixel_to_meter(dx_px, dy_px)
        distance_m = np.sqrt(dx_m**2 + dy_m**2)
        
        # Calculate direction
        direction_rad = np.arctan2(dy_px, dx_px)
        direction_deg = np.degrees(direction_rad)
        
        # Create result dictionary
        result = {
            'dx_px': dx_px,
            'dy_px': dy_px,
            'distance_px': distance_px,
            'dx_m': dx_m,
            'dy_m': dy_m,
            'distance_m': distance_m,
            'direction_deg': direction_deg
        }
        
        # Display in console
        self._print_displacement(result)
        
        # Update visualization
        if show_visualization:
            self.visualizer.update_displacement(current_pos, target_pos, self.calibration)
        
        return result
    
    def _print_displacement(self, displacement: Dict[str, float]):
        """Print displacement information to console"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Displacement Calculation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}📏 Pixels:{Style.RESET_ALL}")
        print(f"  dx: {displacement['dx_px']:+.1f} px")
        print(f"  dy: {displacement['dy_px']:+.1f} px")
        print(f"  distance: {displacement['distance_px']:.1f} px")
        
        print(f"\n{Fore.GREEN}📐 Meters:{Style.RESET_ALL}")
        print(f"  dx: {displacement['dx_m']*1000:+.2f} mm")
        print(f"  dy: {displacement['dy_m']*1000:+.2f} mm")
        print(f"  distance: {displacement['distance_m']*1000:.2f} mm")
        
        print(f"\n{Fore.BLUE}🧭 Direction: {displacement['direction_deg']:+.1f}°{Style.RESET_ALL}")
    
    def apply_nudge(self, dx_px: float, dy_px: float, 
                   reason: str = "Fine adjustment") -> NudgeCommand:
        """
        Apply a final nudge adjustment and log it
        
        Args:
            dx_px: Nudge displacement in pixels (x)
            dy_px: Nudge displacement in pixels (y)
            reason: Reason for the nudge
            
        Returns:
            NudgeCommand with details
        """
        self.nudge_iteration += 1
        
        # Convert to meters
        dx_m, dy_m = self.calibration.pixel_to_meter(dx_px, dy_px)
        distance_px = np.sqrt(dx_px**2 + dy_px**2)
        distance_m = np.sqrt(dx_m**2 + dy_m**2)
        direction_deg = np.degrees(np.arctan2(dy_px, dx_px))
        
        # Create nudge command
        nudge = NudgeCommand(
            iteration=self.nudge_iteration,
            dx_px=dx_px,
            dy_px=dy_px,
            dx_m=dx_m,
            dy_m=dy_m,
            distance_px=distance_px,
            distance_m=distance_m,
            direction_deg=direction_deg,
            timestamp=datetime.now(),
            reason=reason,
            success=False  # Will be updated after execution
        )
        
        # Log the nudge
        self._log_nudge(nudge)
        
        # Add to visualizer
        self.visualizer.log_nudge(nudge)
        
        return nudge
    
    def _log_nudge(self, nudge: NudgeCommand):
        """Log nudge command to file and console"""
        # Console output with color
        print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🎯 FINAL NUDGE #{nudge.iteration}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        
        print(f"📍 Displacement:")
        print(f"  • Pixels: dx={nudge.dx_px:+.1f}px, dy={nudge.dy_px:+.1f}px")
        print(f"  • Meters: dx={nudge.dx_m*1000:+.2f}mm, dy={nudge.dy_m*1000:+.2f}mm")
        print(f"  • Distance: {nudge.distance_px:.1f}px ({nudge.distance_m*1000:.2f}mm)")
        print(f"  • Direction: {nudge.direction_deg:+.1f}°")
        print(f"📝 Reason: {nudge.reason}")
        print(f"⏰ Timestamp: {nudge.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        # Log to file
        log_entry = {
            'iteration': nudge.iteration,
            'timestamp': nudge.timestamp.isoformat(),
            'displacement_px': {'dx': nudge.dx_px, 'dy': nudge.dy_px, 'distance': nudge.distance_px},
            'displacement_m': {'dx': nudge.dx_m, 'dy': nudge.dy_m, 'distance': nudge.distance_m},
            'direction_deg': nudge.direction_deg,
            'reason': nudge.reason,
            'success': nudge.success
        }
        
        # Append to log file
        log_file = self.log_dir / f"nudge_log_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            # Read existing log
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new entry
            logs.append(log_entry)
            
            # Write updated log
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
                
            logger.info(f"Nudge #{nudge.iteration} logged: {nudge.distance_m*1000:.2f}mm @ {nudge.direction_deg:.1f}°")
            
        except Exception as e:
            logger.error(f"Failed to log nudge: {e}")
        
        # Store in memory
        self.nudge_log.append(nudge)
    
    def confirm_nudge_success(self, nudge_iteration: int, success: bool = True):
        """
        Confirm whether a nudge was successful
        
        Args:
            nudge_iteration: Iteration number of the nudge
            success: Whether the nudge was successful
        """
        # Find the nudge
        for nudge in self.nudge_log:
            if nudge.iteration == nudge_iteration:
                nudge.success = success
                
                # Update log file
                self._update_nudge_log(nudge)
                
                # Print confirmation
                if success:
                    print(f"{Fore.GREEN}✅ Nudge #{nudge_iteration} confirmed successful{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Nudge #{nudge_iteration} marked as failed{Style.RESET_ALL}")
                    
                break
    
    def _update_nudge_log(self, nudge: NudgeCommand):
        """Update nudge log file with success status"""
        log_file = self.log_dir / f"nudge_log_{nudge.timestamp.strftime('%Y%m%d')}.json"
        
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    
                # Update the specific nudge
                for log in logs:
                    if log['iteration'] == nudge.iteration:
                        log['success'] = nudge.success
                        break
                        
                # Write back
                with open(log_file, 'w') as f:
                    json.dump(logs, f, indent=2, default=str)
                    
        except Exception as e:
            logger.error(f"Failed to update nudge log: {e}")
    
    def get_nudge_summary(self) -> Dict[str, Any]:
        """Get summary of all nudges"""
        if not self.nudge_log:
            return {}
            
        successful = sum(1 for n in self.nudge_log if n.success)
        total = len(self.nudge_log)
        
        avg_distance_px = np.mean([n.distance_px for n in self.nudge_log])
        avg_distance_m = np.mean([n.distance_m for n in self.nudge_log])
        
        summary = {
            'total_nudges': total,
            'successful': successful,
            'success_rate': successful / total * 100 if total > 0 else 0,
            'avg_distance_px': avg_distance_px,
            'avg_distance_mm': avg_distance_m * 1000,
            'total_distance_mm': sum(n.distance_m for n in self.nudge_log) * 1000
        }
        
        return summary


def demo_vision_system():
    """Demonstrate vision system with displacement and nudge logging"""
    
    print(f"{Fore.CYAN}{'='*70}")
    print(f" VISION SYSTEM DEMO")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # Create vision system
    calibration = CameraCalibration(
        pixels_per_meter_x=2000,  # 2000 pixels per meter
        pixels_per_meter_y=2000,
        camera_height=0.5  # 0.5m above work surface
    )
    
    vision = VisionSystem(calibration)
    
    # Simulate frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print(f"{Fore.YELLOW}Processing camera frame...{Style.RESET_ALL}")
    detections = vision.process_frame(frame)
    
    # Test displacement calculation
    print(f"\n{Fore.YELLOW}Calculating displacement to target...{Style.RESET_ALL}")
    current_pos = (320, 240)  # Center of image
    target_pos = (450, 180)   # Target position
    
    displacement = vision.calculate_displacement(current_pos, target_pos, show_visualization=True)
    
    time.sleep(2)
    
    # Test nudge commands
    print(f"\n{Fore.YELLOW}Applying nudges for fine positioning...{Style.RESET_ALL}")
    
    # First nudge
    nudge1 = vision.apply_nudge(10, -5, "Initial alignment correction")
    time.sleep(1)
    vision.confirm_nudge_success(nudge1.iteration, True)
    
    # Second nudge
    nudge2 = vision.apply_nudge(-3, 2, "Fine tuning after grasp")
    time.sleep(1)
    vision.confirm_nudge_success(nudge2.iteration, True)
    
    # Third nudge
    nudge3 = vision.apply_nudge(1, -1, "Final precision adjustment")
    time.sleep(1)
    vision.confirm_nudge_success(nudge3.iteration, False)  # This one failed
    
    # Retry nudge
    nudge4 = vision.apply_nudge(2, -1.5, "Retry after failed nudge")
    time.sleep(1)
    vision.confirm_nudge_success(nudge4.iteration, True)
    
    # Get summary
    summary = vision.get_nudge_summary()
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Nudge Summary{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"Total nudges: {summary['total_nudges']}")
    print(f"Successful: {summary['successful']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Average distance: {summary['avg_distance_mm']:.2f}mm")
    print(f"Total distance moved: {summary['total_distance_mm']:.2f}mm")
    
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")
    print(f"\n📁 Nudge logs saved to: {vision.log_dir}")
    
    print("\nPress Enter to close visualization...")
    input()
    plt.close('all')


if __name__ == "__main__":
    demo_vision_system()