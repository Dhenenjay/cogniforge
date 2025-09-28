#!/usr/bin/env python3
"""
Vision Offset Color Coding Example

Demonstrates how vision (dx, dy) text is colored:
- GREEN when |dx|,|dy| < 3 px (well-aligned)
- AMBER when |dx| >= 3 or |dy| >= 3 px (needs adjustment)
"""

from cogniforge.ui.vision_display import VisionOffsetDisplay, VisionAlignmentMonitor
from cogniforge.ui.ui_integration import VisionUIFormatter
from colorama import init, Fore, Style
import time

# Initialize colorama
init(autoreset=True)

def show_color_rules():
    """Show the color coding rules."""
    print("\n" + "="*70)
    print("VISION OFFSET COLOR CODING RULES")
    print("="*70)
    
    print(f"""
    The (dx, dy) offset text is colored based on alignment:
    
    {Fore.GREEN}● GREEN{Style.RESET_ALL}:  |dx| < 3 px AND |dy| < 3 px  (Well-aligned)
    {Fore.YELLOW}● AMBER{Style.RESET_ALL}:  |dx| >= 3 px OR |dy| >= 3 px  (Needs adjustment)
    
    Threshold: ±3 pixels in each direction
    """)

def demonstrate_offsets():
    """Demonstrate various offset scenarios with colors."""
    print("\n" + "="*70)
    print("OFFSET EXAMPLES")
    print("="*70)
    
    display = VisionOffsetDisplay()
    
    # Test scenarios
    scenarios = [
        # Well-aligned (GREEN)
        ("Perfect alignment", 0, 0),
        ("Minimal offset", 1, 1),
        ("Just under threshold", 2, -2),
        ("X aligned, Y small", 0, 2),
        ("Y aligned, X small", -2, 0),
        
        # Needs adjustment (AMBER)
        ("X needs adjustment", 5, 1),
        ("Y needs adjustment", 2, 7),
        ("Both need adjustment", 10, -8),
        ("Just over threshold", 3, 2),
        ("Large offset", 15, 20),
    ]
    
    print("\nWELL-ALIGNED (GREEN):")
    print("-" * 40)
    for desc, dx, dy in scenarios[:5]:
        offset_str = display.format_pixel_offset(dx, dy, show_labels=False, show_magnitude=False)
        print(f"  {desc:20} {offset_str}")
    
    print("\nNEEDS ADJUSTMENT (AMBER):")
    print("-" * 40)
    for desc, dx, dy in scenarios[5:]:
        offset_str = display.format_pixel_offset(dx, dy, show_labels=False, show_magnitude=False)
        print(f"  {desc:20} {offset_str}")

def simulate_vision_alignment():
    """Simulate a vision alignment process."""
    print("\n" + "="*70)
    print("VISION ALIGNMENT SIMULATION")
    print("="*70)
    
    display = VisionOffsetDisplay()
    
    print("\nStarting vision-guided alignment...")
    print("-" * 40)
    
    # Simulation sequence
    alignment_sequence = [
        (20, 15, "Initial detection - large offset"),
        (15, 10, "First adjustment"),
        (10, 7, "Getting closer"),
        (5, 4, "Almost there"),
        (3, 3, "Near threshold"),
        (2, 2, "Within threshold!"),
        (1, 1, "Precise alignment"),
        (0, 0, "Perfect alignment achieved!")
    ]
    
    for dx, dy, status in alignment_sequence:
        # Format offset with color
        offset_str = display.format_pixel_offset(dx, dy, show_labels=False, show_magnitude=False)
        
        # Alignment indicator
        if abs(dx) < 3 and abs(dy) < 3:
            indicator = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        else:
            indicator = f"{Fore.YELLOW}→{Style.RESET_ALL}"
        
        print(f"{indicator} {offset_str} - {status}")
        time.sleep(0.5)
    
    print("\n" + f"{Fore.GREEN}✅ Alignment complete!{Style.RESET_ALL}")

def show_ui_integration():
    """Show how it integrates with the UI formatter."""
    print("\n" + "="*70)
    print("UI INTEGRATION")
    print("="*70)
    
    # Test with UI formatter
    test_cases = [
        (1, 2, 0.0002, 0.0004, "Aligned case"),
        (5, -3, 0.001, -0.0006, "X adjustment needed"),
        (2, 8, 0.0004, 0.0016, "Y adjustment needed"),
    ]
    
    for dx, dy, dx_m, dy_m, desc in test_cases:
        print(f"\n{desc}:")
        result = VisionUIFormatter.print_vision_result(
            dx_px=dx,
            dy_px=dy,
            dx_m=dx_m,
            dy_m=dy_m,
            depth=0.15,
            method="test",
            confidence=0.9,
            force_print=True
        )

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    VISION (dx, dy) COLOR CODING DEMO                       ║
║                                                                            ║
║   Shows how vision offset text is colored based on alignment quality:      ║
║   • GREEN when |dx|,|dy| < 3 px (aligned)                                 ║
║   • AMBER when |dx| >= 3 or |dy| >= 3 px (needs adjustment)              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Show color rules
    show_color_rules()
    input("\nPress Enter to see offset examples...")
    
    # Demonstrate different offsets
    demonstrate_offsets()
    input("\nPress Enter to see alignment simulation...")
    
    # Simulate alignment process
    simulate_vision_alignment()
    input("\nPress Enter to see UI integration...")
    
    # Show UI integration
    show_ui_integration()
    
    print("\n" + "="*70)
    print("✅ Demo complete! Vision offsets are now color-coded for clarity.")
    print("="*70)

if __name__ == "__main__":
    main()