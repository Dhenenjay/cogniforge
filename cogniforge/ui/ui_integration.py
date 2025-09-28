"""
UI integration utilities for displaying vision and grasp results.

Provides formatted JSON output for UI components to display
vision detection and grasp execution status.
"""

import json
import sys
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows color support
init(autoreset=True)


class VisionUIFormatter:
    """
    Formats vision detection results for UI display.
    
    Ensures all results are printed as JSON, even when offsets are zero.
    Colors (dx, dy) text green when |dx|,|dy| < 3 px; otherwise amber.
    """
    
    @staticmethod
    def format_offset_with_color(dx_px: int, dy_px: int, use_colors: bool = True) -> str:
        """
        Format pixel offset with color coding.
        Green when |dx|,|dy| < 3 px (well-aligned), amber otherwise.
        
        Args:
            dx_px: X pixel offset
            dy_px: Y pixel offset
            use_colors: Whether to apply colors
            
        Returns:
            Formatted offset string with color
        """
        # Check if well-aligned (both offsets < 3 pixels)
        is_aligned = abs(dx_px) < 3 and abs(dy_px) < 3
        
        # Format the offset string
        offset_str = f"({dx_px:+3d}, {dy_px:+3d})"
        
        if use_colors:
            if is_aligned:
                # Green for well-aligned
                return f"{Fore.GREEN}{offset_str}{Style.RESET_ALL}"
            else:
                # Amber/Yellow for needs adjustment
                return f"{Fore.YELLOW}{offset_str}{Style.RESET_ALL}"
        
        return offset_str
    
    @staticmethod
    def print_vision_result(
        dx_px: int,
        dy_px: int,
        dx_m: float,
        dy_m: float,
        depth: float,
        method: str = "unknown",
        confidence: float = 0.0,
        tolerance_m: float = 0.005,
        force_print: bool = True,
        stream=sys.stdout
    ) -> Dict[str, Any]:
        """
        Print formatted vision detection result for UI.
        
        Always prints JSON, even when offset is approximately zero.
        
        Args:
            dx_px: Pixel offset in x direction
            dy_px: Pixel offset in y direction
            dx_m: World offset in x direction (meters)
            dy_m: World offset in y direction (meters)
            depth: Depth to object (meters)
            method: Detection method used
            confidence: Detection confidence (0-1)
            tolerance_m: Alignment tolerance in meters
            force_print: Always print even if aligned
            stream: Output stream for printing
            
        Returns:
            Formatted result dictionary
            
        Example:
            result = VisionUIFormatter.print_vision_result(
                dx_px=5, dy_px=-3,
                dx_m=0.001, dy_m=-0.0006,
                depth=0.15,
                method="gpt_vision"
            )
        """
        # Calculate magnitudes
        pixel_magnitude = np.sqrt(dx_px**2 + dy_px**2)
        world_magnitude = np.sqrt(dx_m**2 + dy_m**2)
        
        # Determine alignment status
        is_aligned = world_magnitude <= tolerance_m
        
        # Determine visual indicators
        if is_aligned:
            status_icon = "✓"
            status_color = "green"
            status_text = "ALIGNED"
            action_required = "No adjustment needed"
        else:
            status_icon = "→"
            status_color = "yellow"
            status_text = "ADJUST"
            action_required = f"Move {abs(dx_m*1000):.1f}mm {('right' if dx_m > 0 else 'left')}, {abs(dy_m*1000):.1f}mm {('down' if dy_m > 0 else 'up')}"
        
        # Build result dictionary
        result = {
            "vision_detection": {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "confidence": round(confidence, 2),
                "detected": True,
                
                "pixel_offset": {
                    "dx": dx_px,
                    "dy": dy_px,
                    "magnitude": round(pixel_magnitude, 1),
                    "units": "pixels"
                },
                
                "world_offset": {
                    "dx": round(dx_m, 4),
                    "dy": round(dy_m, 4),
                    "magnitude": round(world_magnitude, 4),
                    "units": "meters"
                },
                
                "world_offset_mm": {
                    "dx": round(dx_m * 1000, 1),
                    "dy": round(dy_m * 1000, 1),
                    "magnitude": round(world_magnitude * 1000, 1),
                    "units": "millimeters"
                },
                
                "depth": {
                    "value": round(depth, 3),
                    "units": "meters"
                },
                
                "alignment": {
                    "is_aligned": is_aligned,
                    "status": status_text,
                    "icon": status_icon,
                    "color": status_color,
                    "tolerance_m": tolerance_m,
                    "tolerance_mm": tolerance_m * 1000,
                    "error_m": round(world_magnitude, 4),
                    "error_mm": round(world_magnitude * 1000, 1),
                    "action_required": action_required
                },
                
                "ui_display": {
                    "title": f"{status_icon} Vision Detection: {status_text}",
                    "subtitle": f"Offset: {world_magnitude*1000:.1f}mm (tolerance: {tolerance_m*1000}mm)",
                    "details": [
                        f"Pixel offset: ({dx_px}, {dy_px})",
                        f"World offset: ({dx_m*1000:.1f}, {dy_m*1000:.1f}) mm",
                        f"Detection method: {method}",
                        f"Confidence: {confidence*100:.0f}%"
                    ]
                }
            }
        }
        
        # Always print for UI (even if aligned)
        if force_print or not is_aligned:
            # Print header with colored offset
            print("\n" + "="*60, file=stream)
            
            # Format offset with color
            colored_offset = VisionUIFormatter.format_offset_with_color(dx_px, dy_px)
            
            # Alignment indicator
            if abs(dx_px) < 3 and abs(dy_px) < 3:
                align_status = f"{Fore.GREEN}✓ ALIGNED{Style.RESET_ALL}"
            else:
                align_status = f"{Fore.YELLOW}⚡ ADJUSTMENT NEEDED{Style.RESET_ALL}"
            
            print(f"VISION DETECTION RESULT {colored_offset} {align_status}", file=stream)
            print(json.dumps(result, indent=2), file=stream)
            print("="*60 + "\n", file=stream)
        
        return result
    
    @staticmethod
    def print_no_detection(
        reason: str = "No object detected",
        method: str = "none",
        stream=sys.stdout
    ) -> Dict[str, Any]:
        """
        Print JSON for cases where vision detection fails.
        
        Args:
            reason: Reason for detection failure
            method: Method that was attempted
            stream: Output stream
            
        Returns:
            Formatted failure result
        """
        result = {
            "vision_detection": {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "confidence": 0.0,
                "detected": False,
                
                "pixel_offset": {
                    "dx": 0,
                    "dy": 0,
                    "magnitude": 0.0,
                    "units": "pixels"
                },
                
                "world_offset": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "magnitude": 0.0,
                    "units": "meters"
                },
                
                "alignment": {
                    "is_aligned": False,
                    "status": "NO_DETECTION",
                    "icon": "✗",
                    "color": "red",
                    "action_required": "Manual positioning required"
                },
                
                "ui_display": {
                    "title": "✗ Vision Detection: FAILED",
                    "subtitle": reason,
                    "details": [
                        f"Method attempted: {method}",
                        "No object detected in view",
                        "Proceeding without vision adjustment"
                    ]
                },
                
                "error": {
                    "message": reason,
                    "recovery": "Using last known position or manual control"
                }
            }
        }
        
        print("\n" + "="*60, file=stream)
        print("VISION DETECTION RESULT (UI JSON):", file=stream)
        print(json.dumps(result, indent=2), file=stream)
        print("="*60 + "\n", file=stream)
        
        return result


class GraspUIDisplay:
    """
    Displays grasp execution status for UI.
    """
    
    @staticmethod
    def print_grasp_status(
        phase: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        stream=sys.stdout
    ) -> Dict[str, Any]:
        """
        Print current grasp execution status as JSON.
        
        Args:
            phase: Current phase of execution
            status: Status of the phase
            details: Additional details
            stream: Output stream
            
        Returns:
            Status dictionary
        """
        phase_icons = {
            "approaching": "⤵",
            "paused_for_vision": "👁",
            "adjusting": "🎯",
            "grasping": "🤏",
            "lifting": "⤴",
            "success": "✅",
            "failed": "❌"
        }
        
        result = {
            "grasp_status": {
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "status": status,
                "icon": phase_icons.get(phase, "●"),
                "details": details or {},
                
                "ui_display": {
                    "title": f"{phase_icons.get(phase, '●')} {phase.upper().replace('_', ' ')}",
                    "subtitle": status,
                    "progress_percent": _get_phase_progress(phase)
                }
            }
        }
        
        print(json.dumps(result, indent=2), file=stream)
        return result


def _get_phase_progress(phase: str) -> int:
    """Get progress percentage based on phase."""
    progress_map = {
        "approaching": 20,
        "paused_for_vision": 40,
        "adjusting": 60,
        "grasping": 80,
        "lifting": 90,
        "success": 100,
        "failed": 0
    }
    return progress_map.get(phase, 0)


# Convenience functions for common cases
def print_aligned_result(
    dx_px: int = 0,
    dy_px: int = 0,
    method: str = "color_detection",
    confidence: float = 0.9
) -> None:
    """
    Print result when object is already well-aligned.
    
    Even with near-zero offsets, this ensures JSON is printed for UI.
    """
    VisionUIFormatter.print_vision_result(
        dx_px=dx_px,
        dy_px=dy_px,
        dx_m=0.0,
        dy_m=0.0,
        depth=0.15,  # Typical wrist camera depth
        method=method,
        confidence=confidence,
        tolerance_m=0.005,
        force_print=True  # Always print for UI
    )


def print_adjustment_needed(
    dx_px: int,
    dy_px: int,
    dx_m: float,
    dy_m: float,
    depth: float,
    method: str = "gpt_vision"
) -> None:
    """
    Print result when adjustment is needed.
    """
    VisionUIFormatter.print_vision_result(
        dx_px=dx_px,
        dy_px=dy_px,
        dx_m=dx_m,
        dy_m=dy_m,
        depth=depth,
        method=method,
        confidence=0.85,
        tolerance_m=0.005
    )


# Example usage for UI testing
if __name__ == "__main__":
    print("="*70)
    print("UI INTEGRATION TEST - Vision Results Display")
    print("="*70)
    
    print("\n1. Testing ALIGNED case (near-zero offset):")
    print("-"*40)
    # Even with tiny offsets, JSON is printed
    result1 = VisionUIFormatter.print_vision_result(
        dx_px=2,
        dy_px=-1,
        dx_m=0.0003,  # 0.3mm - well within tolerance
        dy_m=-0.0002,  # 0.2mm
        depth=0.15,
        method="gpt_vision",
        confidence=0.92
    )
    
    print("\n2. Testing NEEDS ADJUSTMENT case:")
    print("-"*40)
    result2 = VisionUIFormatter.print_vision_result(
        dx_px=45,
        dy_px=-30,
        dx_m=0.012,  # 12mm - exceeds 5mm tolerance
        dy_m=-0.008,  # 8mm
        depth=0.15,
        method="color_detection",
        confidence=0.75
    )
    
    print("\n3. Testing NO DETECTION case:")
    print("-"*40)
    result3 = VisionUIFormatter.print_no_detection(
        reason="Blue cube not found in wrist camera view",
        method="gpt_vision"
    )
    
    print("\n4. Testing grasp status updates:")
    print("-"*40)
    phases = [
        ("approaching", "Moving to pre-grasp position"),
        ("paused_for_vision", "Capturing wrist camera image"),
        ("adjusting", "Applying micro nudge"),
        ("grasping", "Closing gripper"),
        ("success", "Object grasped successfully")
    ]
    
    for phase, status in phases:
        GraspUIDisplay.print_grasp_status(phase, status)
        print()  # Space between updates
    
    print("="*70)
    print("All UI display tests completed!")
    print("="*70)