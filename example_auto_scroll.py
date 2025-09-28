#!/usr/bin/env python3
"""
Console Auto-Scroll Example

Demonstrates automatic scrolling to the latest console output,
ensuring important updates are always visible.
"""

import time
import random
from cogniforge.ui.console_utils import (
    ConsoleAutoScroller, 
    ProgressTracker,
    LiveLogger,
    ScrollingOutput,
    enable_auto_scroll,
    disable_auto_scroll
)
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)


def demo_basic_scrolling():
    """Basic auto-scrolling demonstration."""
    print("\n" + "="*60)
    print("BASIC AUTO-SCROLLING")
    print("="*60)
    
    scroller = ConsoleAutoScroller()
    
    print("\nGenerating output with auto-scroll...")
    print("(Console automatically scrolls to show latest line)")
    print("-" * 40)
    
    for i in range(20):
        message = f"Output line {i+1:2d}: Processing data..."
        if i % 5 == 0:
            message += f" {Fore.GREEN}✓ Checkpoint{Style.RESET_ALL}"
        
        scroller.print_and_scroll(message)
        time.sleep(0.1)
    
    print("-" * 40)
    print("✅ Auto-scrolling keeps the latest output visible!")


def demo_training_progress():
    """Simulate training progress with auto-scroll."""
    print("\n" + "="*60)
    print("TRAINING PROGRESS WITH AUTO-SCROLL")
    print("="*60)
    
    tracker = ProgressTracker(100, "Training Model")
    
    print("\nSimulating neural network training...")
    print("(Progress bar auto-scrolls to stay visible)")
    print("-" * 40)
    
    for epoch in range(100):
        # Simulate varying training speed
        delay = random.uniform(0.02, 0.08)
        
        # Update with loss value
        loss = 10.0 * (1.0 - epoch/100) + random.uniform(-0.5, 0.5)
        tracker.update(1, f"Loss: {loss:.4f}")
        
        time.sleep(delay)
    
    tracker.finish("Training Complete!")
    print("\n✅ Progress stayed visible throughout training!")


def demo_live_logging():
    """Demonstrate live logging with auto-scroll."""
    print("\n" + "="*60)
    print("LIVE LOGGING WITH AUTO-SCROLL")
    print("="*60)
    
    logger = LiveLogger("System", auto_scroll=True)
    
    print("\nSimulating system monitoring...")
    print("(Log entries auto-scroll to show latest)")
    print("-" * 40)
    
    events = [
        ("info", "System initialized"),
        ("info", "Loading configuration..."),
        ("success", "Configuration loaded successfully"),
        ("info", "Connecting to database..."),
        ("warn", "Connection slow, retrying..."),
        ("success", "Database connected"),
        ("info", "Starting data processing..."),
        ("info", "Processing batch 1/5..."),
        ("info", "Processing batch 2/5..."),
        ("warn", "High memory usage detected"),
        ("info", "Processing batch 3/5..."),
        ("info", "Processing batch 4/5..."),
        ("info", "Processing batch 5/5..."),
        ("success", "All batches processed successfully"),
        ("info", "Generating report..."),
        ("success", "Report saved to output/report.pdf"),
        ("error", "Failed to send email notification"),
        ("info", "Retrying email send..."),
        ("success", "Email sent successfully"),
        ("success", "All tasks completed!")
    ]
    
    for level, message in events:
        if level == "info":
            logger.info(message)
        elif level == "warn":
            logger.warn(message)
        elif level == "success":
            logger.success(message)
        elif level == "error":
            logger.error(message)
        
        time.sleep(0.3)
    
    print("\n✅ Latest log entries always visible!")


def demo_multi_stream_output():
    """Demonstrate multiple output streams with auto-scroll."""
    print("\n" + "="*60)
    print("MULTI-STREAM OUTPUT WITH AUTO-SCROLL")
    print("="*60)
    
    output = ScrollingOutput(max_lines=10, auto_scroll=True)
    
    print("\nSimulating multiple data streams...")
    print("(All streams auto-scroll together)")
    print("-" * 40)
    
    for i in range(30):
        stream_id = random.randint(1, 3)
        value = random.uniform(0, 100)
        
        if stream_id == 1:
            output.add_line(f"[Sensor A] Reading: {value:.2f} units", force_scroll=True)
        elif stream_id == 2:
            output.add_line(f"[Sensor B] Temperature: {value:.1f}°C", force_scroll=True)
        else:
            output.add_line(f"[Sensor C] Pressure: {value:.0f} kPa", force_scroll=True)
        
        time.sleep(0.15)
    
    print("\n✅ Multiple streams stayed synchronized!")


def demo_global_auto_scroll():
    """Demonstrate global auto-scroll for all print statements."""
    print("\n" + "="*60)
    print("GLOBAL AUTO-SCROLL")
    print("="*60)
    
    print("\nEnabling global auto-scroll...")
    enable_auto_scroll()
    
    print("\nAll print statements now auto-scroll:")
    for i in range(10):
        print(f"  Regular print {i+1} - automatically scrolled")
        time.sleep(0.2)
    
    print("\nDisabling global auto-scroll...")
    disable_auto_scroll()
    
    print("✅ Global auto-scroll demonstrated!")


def simulate_real_scenario():
    """Simulate a real training/processing scenario."""
    print("\n" + "="*60)
    print("REAL-WORLD SCENARIO: ML TRAINING")
    print("="*60)
    
    scroller = ConsoleAutoScroller()
    logger = LiveLogger("Trainer")
    
    # Initialize
    logger.info("Initializing training environment...")
    time.sleep(0.5)
    
    # Load data
    logger.info("Loading dataset...")
    tracker = ProgressTracker(1000, "Loading samples")
    for i in range(100):
        tracker.update(10)
        time.sleep(0.01)
    tracker.finish("Dataset loaded")
    
    # Training loop
    logger.info("Starting training...")
    print()
    
    for epoch in range(1, 6):
        print(f"\n{Fore.CYAN}═══ Epoch {epoch}/5 ═══{Style.RESET_ALL}")
        
        # Training progress
        tracker = ProgressTracker(50, f"Epoch {epoch}")
        for batch in range(50):
            loss = 2.0 / epoch + random.uniform(-0.1, 0.1)
            acc = min(0.99, 0.7 + 0.05 * epoch + random.uniform(-0.02, 0.02))
            
            tracker.update(1, f"Loss: {loss:.3f}, Acc: {acc:.1%}")
            time.sleep(0.03)
        
        tracker.finish(f"Loss: {loss:.3f}, Acc: {acc:.1%}")
        
        # Log epoch results
        logger.success(f"Epoch {epoch} complete - Loss: {loss:.3f}, Accuracy: {acc:.1%}")
    
    print()
    logger.success("Training completed successfully!")
    logger.info("Model saved to models/trained_model.pt")
    
    print("\n✅ Real scenario with auto-scroll complete!")


def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                      CONSOLE AUTO-SCROLL DEMONSTRATION                     ║
║                                                                            ║
║  Shows how auto-scrolling keeps the latest output visible in the console  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    demos = [
        ("Basic Auto-Scrolling", demo_basic_scrolling),
        ("Training Progress", demo_training_progress),
        ("Live Logging", demo_live_logging),
        ("Multi-Stream Output", demo_multi_stream_output),
        ("Global Auto-Scroll", demo_global_auto_scroll),
        ("Real ML Training Scenario", simulate_real_scenario),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. Run all demos")
    
    choice = input("\nSelect demo (0-6): ").strip()
    
    if choice == "0":
        for name, demo_func in demos:
            print(f"\n\n{'='*60}")
            print(f"Running: {name}")
            print('='*60)
            demo_func()
            input("\nPress Enter to continue...")
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        demos[int(choice)-1][1]()
    else:
        print("Running all demos...")
        for name, demo_func in demos:
            demo_func()
            time.sleep(1)
    
    print("\n" + "="*60)
    print("🎉 Auto-scroll demonstration complete!")
    print("Console always shows the latest output line.")
    print("="*60)


if __name__ == "__main__":
    main()