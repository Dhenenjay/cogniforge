"""
Integrated UI Demo
Demonstrates vision offset color coding and console auto-scrolling together.
"""

from cogniforge.ui.vision_display import VisionOffsetDisplay, VisionAlignmentMonitor
from cogniforge.ui.console_utils import ConsoleAutoScroller, ProgressTracker
import time
import random


def main():
    """Run integrated UI demo."""
    
    print("\n" + "="*70)
    print("🎯 INTEGRATED UI FEATURES DEMO")
    print("="*70)
    
    # Initialize components
    display = VisionOffsetDisplay()
    monitor = VisionAlignmentMonitor()
    scroller = ConsoleAutoScroller()
    
    print("\n📊 FEATURE 1: Vision Offset Color Coding")
    print("-"*40)
    print("Green = Well-aligned (|dx|, |dy| < 3px)")
    print("Amber = Needs adjustment (≥ 3px)")
    print()
    
    # Show static examples
    test_offsets = [
        (1, 2, "Nearly perfect"),
        (0, 0, "Perfect alignment"),
        (5, -3, "Slight drift"),
        (10, 8, "Major offset"),
    ]
    
    for dx, dy, desc in test_offsets:
        scroller.print_and_scroll(f"  {desc}:")
        display.print_compact_status(dx, dy)
        time.sleep(0.5)
    
    print("\n📊 FEATURE 2: Auto-Scrolling Console Output")
    print("-"*40)
    
    # Simulate real-time monitoring with auto-scroll
    print("\nSimulating vision alignment convergence...")
    print("(Console auto-scrolls to keep latest visible)\n")
    
    # Start with large offset, converge to aligned
    convergence_path = [
        (15, 12, "Initial detection"),
        (12, 9, "Adjusting..."),
        (9, 6, "Getting closer..."),
        (6, 4, "Almost there..."),
        (3, 2, "Fine tuning..."),
        (1, 1, "Locked on target!"),
    ]
    
    for dx, dy, status in convergence_path:
        # Print status with auto-scroll
        scroller.print_and_scroll(f"\n⚡ {status}")
        
        # Show colored offset
        display.print_compact_status(dx, dy)
        
        # Visual alignment grid
        if dx <= 3 and dy <= 3:
            badge = display.create_status_badge(dx, dy, "minimal")
            scroller.print_and_scroll(f"  {badge}")
        
        time.sleep(0.8)
    
    print("\n📊 FEATURE 3: Live Training Progress with Colors")
    print("-"*40)
    
    # Simulate training with progress
    tracker = ProgressTracker(20, "Training BC Model")
    
    for epoch in range(20):
        # Update progress
        tracker.update(1)
        
        # Simulate vision checks during training
        if epoch % 5 == 0:
            # Random vision offset check
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            
            # Print inline status
            offset_str = display.format_pixel_offset(dx, dy, show_labels=False, show_magnitude=False)
            scroller.print_and_scroll(f"\n  Vision check: {offset_str}")
        
        time.sleep(0.15)
    
    tracker.finish()
    
    print("\n📊 COMBINED DEMO: Real-time Monitoring")
    print("-"*40)
    
    print("\nMonitoring vision alignment with trend analysis...")
    
    # Simulate noisy convergence
    for i in range(10):
        # Add noise to simulate real conditions
        if i < 5:
            base_offset = 10 - (i * 2)
        else:
            base_offset = 2
        
        dx = base_offset + random.randint(-2, 2)
        dy = base_offset + random.randint(-2, 2)
        
        # Update monitor
        monitor.update(dx, dy, dx * 0.0002, dy * 0.0002)
        
        # Auto-scroll to keep visible
        scroller.scroll_to_bottom()
        
        time.sleep(0.5)
    
    # Show final trend
    monitor.show_trend()
    
    print("\n" + "="*70)
    print("✅ INTEGRATED UI DEMO COMPLETE!")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  • Vision offsets with green/amber color coding")
    print("  • Console auto-scrolling for real-time visibility")
    print("  • Progress tracking with integrated vision checks")
    print("  • Trend monitoring with alignment convergence")


if __name__ == "__main__":
    main()