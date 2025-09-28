"""
Enhanced Vision Display Module

Provides colored visualization of vision detection offsets.
Green when |dx|,|dy| < 3 px (well-aligned), amber otherwise.
"""

import sys
from typing import Tuple, Optional
from colorama import init, Fore, Back, Style
import numpy as np

# Initialize colorama for Windows
init(autoreset=True)


class VisionOffsetDisplay:
    """Display vision offsets with color coding based on alignment."""
    
    # Thresholds
    ALIGNED_THRESHOLD_PX = 3  # Pixels
    ALIGNED_THRESHOLD_MM = 1.0  # Millimeters
    
    @staticmethod
    def get_alignment_color(dx_px: int, dy_px: int) -> str:
        """
        Get color based on alignment quality.
        
        Args:
            dx_px: X offset in pixels
            dy_px: Y offset in pixels
            
        Returns:
            Colorama color code
        """
        if abs(dx_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX and \
           abs(dy_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX:
            return Fore.GREEN  # Well-aligned
        else:
            return Fore.YELLOW  # Needs adjustment (amber)
    
    @staticmethod
    def format_pixel_offset(dx_px: int, dy_px: int, 
                          show_labels: bool = True,
                          show_magnitude: bool = True) -> str:
        """
        Format pixel offset with color coding.
        
        Args:
            dx_px: X offset in pixels
            dy_px: Y offset in pixels
            show_labels: Whether to show dx/dy labels
            show_magnitude: Whether to show magnitude
            
        Returns:
            Formatted colored string
        """
        color = VisionOffsetDisplay.get_alignment_color(dx_px, dy_px)
        
        if show_labels:
            offset_str = f"dx={dx_px:+3d}px, dy={dy_px:+3d}px"
        else:
            offset_str = f"({dx_px:+3d}, {dy_px:+3d})"
        
        colored_str = f"{color}{offset_str}{Style.RESET_ALL}"
        
        if show_magnitude:
            magnitude = np.sqrt(dx_px**2 + dy_px**2)
            mag_color = Fore.GREEN if magnitude < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX else Fore.YELLOW
            colored_str += f" │ {mag_color}|r|={magnitude:.1f}px{Style.RESET_ALL}"
        
        return colored_str
    
    @staticmethod
    def format_world_offset(dx_m: float, dy_m: float, 
                          show_mm: bool = True) -> str:
        """
        Format world offset with color coding.
        
        Args:
            dx_m: X offset in meters
            dy_m: Y offset in meters
            show_mm: Whether to show in millimeters
            
        Returns:
            Formatted colored string
        """
        # Convert to mm for display
        dx_mm = dx_m * 1000
        dy_mm = dy_m * 1000
        magnitude_mm = np.sqrt(dx_mm**2 + dy_mm**2)
        
        # Color based on mm threshold
        if magnitude_mm < VisionOffsetDisplay.ALIGNED_THRESHOLD_MM:
            color = Fore.GREEN
        else:
            color = Fore.YELLOW
        
        if show_mm:
            offset_str = f"({dx_mm:+.1f}, {dy_mm:+.1f}) mm"
        else:
            offset_str = f"({dx_m:+.4f}, {dy_m:+.4f}) m"
        
        return f"{color}{offset_str}{Style.RESET_ALL}"
    
    @staticmethod
    def print_alignment_status(dx_px: int, dy_px: int, 
                              dx_m: float, dy_m: float,
                              show_guidance: bool = True):
        """
        Print comprehensive alignment status with colors.
        
        Args:
            dx_px: X offset in pixels
            dy_px: Y offset in pixels
            dx_m: X offset in meters
            dy_m: Y offset in meters
            show_guidance: Whether to show adjustment guidance
        """
        # Header
        print("\n" + "═" * 60)
        
        # Check alignment
        pixel_aligned = abs(dx_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX and \
                       abs(dy_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX
        
        # Status icon and title
        if pixel_aligned:
            status_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}"
            status_text = f"{Fore.GREEN}VISION ALIGNED{Style.RESET_ALL}"
        else:
            status_icon = f"{Fore.YELLOW}⚡{Style.RESET_ALL}"
            status_text = f"{Fore.YELLOW}ADJUSTMENT REQUIRED{Style.RESET_ALL}"
        
        print(f"{status_icon} {status_text}")
        print("─" * 60)
        
        # Pixel offsets
        pixel_str = VisionOffsetDisplay.format_pixel_offset(dx_px, dy_px, show_labels=True)
        print(f"  Pixel Offset: {pixel_str}")
        
        # World offsets
        world_str = VisionOffsetDisplay.format_world_offset(dx_m, dy_m)
        print(f"  World Offset: {world_str}")
        
        # Visual indicator
        print("\n  Visual Alignment:")
        VisionOffsetDisplay._print_alignment_grid(dx_px, dy_px)
        
        # Guidance
        if show_guidance and not pixel_aligned:
            print("\n  Adjustment Guidance:")
            if abs(dx_px) >= VisionOffsetDisplay.ALIGNED_THRESHOLD_PX:
                direction = "RIGHT" if dx_px > 0 else "LEFT"
                print(f"    → Move {abs(dx_px)}px {direction}")
            if abs(dy_px) >= VisionOffsetDisplay.ALIGNED_THRESHOLD_PX:
                direction = "DOWN" if dy_px > 0 else "UP"
                print(f"    → Move {abs(dy_px)}px {direction}")
        
        print("═" * 60)
    
    @staticmethod
    def _print_alignment_grid(dx_px: int, dy_px: int, grid_size: int = 21):
        """
        Print a visual alignment grid showing current offset.
        
        Args:
            dx_px: X offset in pixels
            dy_px: Y offset in pixels
            grid_size: Size of the grid (should be odd)
        """
        # Ensure grid size is odd
        if grid_size % 2 == 0:
            grid_size += 1
        
        center = grid_size // 2
        
        # Scale offsets to fit in grid
        scale = 3  # pixels per grid cell
        grid_dx = min(max(dx_px // scale, -center), center)
        grid_dy = min(max(dy_px // scale, -center), center)
        
        # Print grid
        print("    ", end="")
        for x in range(grid_size):
            if x == center:
                print(f"{Fore.CYAN}│{Style.RESET_ALL}", end="")
            else:
                print(" ", end="")
        print()
        
        for y in range(grid_size):
            print("    ", end="")
            for x in range(grid_size):
                # Current position
                if x == center + grid_dx and y == center + grid_dy:
                    if grid_dx == 0 and grid_dy == 0:
                        print(f"{Fore.GREEN}◉{Style.RESET_ALL}", end="")  # Aligned
                    else:
                        print(f"{Fore.YELLOW}●{Style.RESET_ALL}", end="")  # Offset
                # Center crosshair
                elif x == center and y == center:
                    print(f"{Fore.CYAN}┼{Style.RESET_ALL}", end="")
                elif x == center:
                    print(f"{Fore.CYAN}│{Style.RESET_ALL}", end="")
                elif y == center:
                    print(f"{Fore.CYAN}─{Style.RESET_ALL}", end="")
                else:
                    print("·", end="")
            print()
    
    @staticmethod
    def print_compact_status(dx_px: int, dy_px: int):
        """
        Print a compact single-line status.
        
        Args:
            dx_px: X offset in pixels
            dy_px: Y offset in pixels
        """
        offset_str = VisionOffsetDisplay.format_pixel_offset(dx_px, dy_px, show_labels=False)
        
        if abs(dx_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX and \
           abs(dy_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX:
            status = f"{Fore.GREEN}✓ ALIGNED{Style.RESET_ALL}"
        else:
            status = f"{Fore.YELLOW}→ ADJUST{Style.RESET_ALL}"
        
        print(f"Vision: {offset_str} {status}")
    
    @staticmethod
    def create_status_badge(dx_px: int, dy_px: int, style: str = "default") -> str:
        """
        Create a status badge for the vision alignment.
        
        Args:
            dx_px: X offset in pixels
            dy_px: Y offset in pixels
            style: Badge style ("default", "box", "minimal")
            
        Returns:
            Formatted badge string
        """
        is_aligned = abs(dx_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX and \
                    abs(dy_px) < VisionOffsetDisplay.ALIGNED_THRESHOLD_PX
        
        offset_str = f"({dx_px:+3d}, {dy_px:+3d})"
        
        if style == "box":
            if is_aligned:
                badge = f"""┌───────────────────────┐
│ {Fore.GREEN}✓ Vision: {offset_str}{Style.RESET_ALL} │
└───────────────────────┘"""
            else:
                badge = f"""┌───────────────────────┐
│ {Fore.YELLOW}⚡ Vision: {offset_str}{Style.RESET_ALL} │
└───────────────────────┘"""
        elif style == "minimal":
            color = Fore.GREEN if is_aligned else Fore.YELLOW
            icon = "●" if is_aligned else "○"
            badge = f"{color}{icon} {offset_str}{Style.RESET_ALL}"
        else:  # default
            color = Fore.GREEN if is_aligned else Fore.YELLOW
            status = "ALIGNED" if is_aligned else "OFFSET"
            badge = f"{color}Vision {offset_str}: {status}{Style.RESET_ALL}"
        
        return badge


class VisionAlignmentMonitor:
    """Monitor and display vision alignment in real-time."""
    
    def __init__(self, threshold_px: int = 3):
        """
        Initialize alignment monitor.
        
        Args:
            threshold_px: Pixel threshold for alignment
        """
        self.threshold_px = threshold_px
        self.history = []
        self.display = VisionOffsetDisplay()
    
    def update(self, dx_px: int, dy_px: int, dx_m: float = 0.0, dy_m: float = 0.0):
        """
        Update with new vision offset and display status.
        
        Args:
            dx_px: X offset in pixels
            dy_px: Y offset in pixels
            dx_m: X offset in meters
            dy_m: Y offset in meters
        """
        # Store in history
        self.history.append({
            'dx_px': dx_px,
            'dy_px': dy_px,
            'dx_m': dx_m,
            'dy_m': dy_m
        })
        
        # Display compact status
        self.display.print_compact_status(dx_px, dy_px)
    
    def show_trend(self):
        """Show alignment trend over recent updates."""
        if len(self.history) < 2:
            print("Not enough data for trend analysis")
            return
        
        recent = self.history[-5:]  # Last 5 updates
        
        print("\n" + "─" * 40)
        print("ALIGNMENT TREND (last 5 updates):")
        print("─" * 40)
        
        for i, data in enumerate(recent, 1):
            dx, dy = data['dx_px'], data['dy_px']
            offset_str = self.display.format_pixel_offset(dx, dy, show_labels=False, show_magnitude=False)
            
            # Trend indicator
            if i > 1:
                prev = recent[i-2]
                prev_mag = np.sqrt(prev['dx_px']**2 + prev['dy_px']**2)
                curr_mag = np.sqrt(dx**2 + dy**2)
                
                if curr_mag < prev_mag:
                    trend = f"{Fore.GREEN}↓{Style.RESET_ALL}"  # Improving
                elif curr_mag > prev_mag:
                    trend = f"{Fore.RED}↑{Style.RESET_ALL}"  # Worsening
                else:
                    trend = "→"  # Same
            else:
                trend = " "
            
            print(f"  {i}. {offset_str} {trend}")
        
        # Overall trend
        first_mag = np.sqrt(recent[0]['dx_px']**2 + recent[0]['dy_px']**2)
        last_mag = np.sqrt(recent[-1]['dx_px']**2 + recent[-1]['dy_px']**2)
        
        if last_mag < self.threshold_px:
            overall = f"{Fore.GREEN}✓ Converged to alignment{Style.RESET_ALL}"
        elif last_mag < first_mag:
            overall = f"{Fore.YELLOW}↓ Improving{Style.RESET_ALL}"
        else:
            overall = f"{Fore.RED}⚠ Not converging{Style.RESET_ALL}"
        
        print(f"\nOverall: {overall}")
        print("─" * 40)


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("VISION OFFSET COLOR DISPLAY DEMO")
    print("="*70)
    
    display = VisionOffsetDisplay()
    
    # Test cases
    test_cases = [
        (1, 2, 0.0002, 0.0004, "Well-aligned (green)"),
        (2, -1, 0.0004, -0.0002, "Well-aligned (green)"),
        (5, 2, 0.001, 0.0004, "X needs adjustment (amber)"),
        (1, 7, 0.0002, 0.0014, "Y needs adjustment (amber)"),
        (10, -15, 0.002, -0.003, "Both need adjustment (amber)"),
    ]
    
    print("\n1. OFFSET COLOR CODING:")
    print("-" * 40)
    for dx, dy, dx_m, dy_m, description in test_cases:
        offset_str = display.format_pixel_offset(dx, dy)
        print(f"  {description}:")
        print(f"    {offset_str}")
    
    print("\n2. FULL ALIGNMENT STATUS:")
    print("-" * 40)
    # Show detailed status for one case
    display.print_alignment_status(10, -5, 0.002, -0.001)
    
    print("\n3. COMPACT STATUS LINE:")
    print("-" * 40)
    for dx, dy, _, _, _ in test_cases:
        display.print_compact_status(dx, dy)
    
    print("\n4. STATUS BADGES:")
    print("-" * 40)
    
    # Different badge styles
    styles = ["default", "box", "minimal"]
    for style in styles:
        print(f"\n{style.upper()} style:")
        badge = display.create_status_badge(2, 1, style)
        print(badge)
    
    print("\n5. REAL-TIME MONITORING:")
    print("-" * 40)
    
    monitor = VisionAlignmentMonitor()
    
    # Simulate alignment convergence
    offsets = [
        (15, 10),
        (10, 7),
        (5, 4),
        (2, 2),
        (1, 1)
    ]
    
    print("Simulating alignment convergence:")
    for dx, dy in offsets:
        monitor.update(dx, dy)
        import time
        time.sleep(0.5)
    
    # Show trend
    monitor.show_trend()
    
    print("\n" + "="*70)
    print("✅ Vision offset color display demo complete!")
    print("="*70)