"""
Console Utilities Module

Provides auto-scrolling and enhanced console output management
for real-time displays and progress tracking.
"""

import sys
import os
import time
import ctypes
from typing import Optional, List, Any
from dataclasses import dataclass
import threading
from queue import Queue
import shutil


class ConsoleAutoScroller:
    """
    Automatically scrolls console to show latest output.
    Works on Windows, Linux, and macOS.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize auto-scroller.
        
        Args:
            enabled: Whether auto-scrolling is enabled
        """
        self.enabled = enabled
        self.platform = sys.platform
        self._setup_platform_specific()
    
    def _setup_platform_specific(self):
        """Setup platform-specific console handling."""
        if self.platform == "win32":
            # Windows-specific setup
            self.kernel32 = ctypes.windll.kernel32
            self.STD_OUTPUT_HANDLE = -11
            self.handle = self.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
            
            # Console info structure
            class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
                _fields_ = [
                    ("dwSize", ctypes.c_uint32),
                    ("dwCursorPosition", ctypes.c_uint32),
                    ("wAttributes", ctypes.c_uint16),
                    ("srWindow", ctypes.c_uint64),
                    ("dwMaximumWindowSize", ctypes.c_uint32)
                ]
            
            self.CONSOLE_SCREEN_BUFFER_INFO = CONSOLE_SCREEN_BUFFER_INFO
    
    def scroll_to_bottom(self):
        """Scroll console to the bottom (latest output)."""
        if not self.enabled:
            return
        
        if self.platform == "win32":
            self._scroll_windows()
        else:
            self._scroll_unix()
    
    def _scroll_windows(self):
        """Windows-specific scrolling."""
        try:
            # Get console info
            csbi = self.CONSOLE_SCREEN_BUFFER_INFO()
            self.kernel32.GetConsoleScreenBufferInfo(self.handle, ctypes.byref(csbi))
            
            # Move cursor to end
            cursor_pos = csbi.dwSize - 1
            self.kernel32.SetConsoleCursorPosition(self.handle, cursor_pos)
            
            # Force console to scroll
            sys.stdout.write("\n")
            sys.stdout.flush()
        except:
            # Fallback method
            self._scroll_fallback()
    
    def _scroll_unix(self):
        """Unix/Linux/macOS scrolling."""
        try:
            # ANSI escape sequences
            sys.stdout.write("\033[J")  # Clear from cursor to end
            sys.stdout.flush()
        except:
            self._scroll_fallback()
    
    def _scroll_fallback(self):
        """Fallback scrolling method."""
        sys.stdout.write("\n")
        sys.stdout.flush()
    
    def print_and_scroll(self, text: str, end: str = "\n"):
        """
        Print text and auto-scroll to show it.
        
        Args:
            text: Text to print
            end: Line ending
        """
        print(text, end=end)
        sys.stdout.flush()
        if self.enabled:
            self.scroll_to_bottom()
    
    def enable(self):
        """Enable auto-scrolling."""
        self.enabled = True
    
    def disable(self):
        """Disable auto-scrolling."""
        self.enabled = False


class ScrollingOutput:
    """
    Manages scrolling output with buffering and rate limiting.
    """
    
    def __init__(self, 
                 max_lines: int = 100,
                 auto_scroll: bool = True,
                 rate_limit: float = 0.033):  # ~30 FPS
        """
        Initialize scrolling output.
        
        Args:
            max_lines: Maximum lines to keep in buffer
            auto_scroll: Whether to auto-scroll
            rate_limit: Minimum time between updates (seconds)
        """
        self.max_lines = max_lines
        self.auto_scroll = auto_scroll
        self.rate_limit = rate_limit
        self.last_update = 0
        self.buffer = []
        self.scroller = ConsoleAutoScroller(auto_scroll)
    
    def add_line(self, text: str, force_scroll: bool = False):
        """
        Add a line to output with auto-scrolling.
        
        Args:
            text: Text to add
            force_scroll: Force immediate scroll
        """
        # Add to buffer
        self.buffer.append(text)
        if len(self.buffer) > self.max_lines:
            self.buffer.pop(0)
        
        # Check rate limiting
        current_time = time.time()
        if force_scroll or (current_time - self.last_update) >= self.rate_limit:
            self.scroller.print_and_scroll(text)
            self.last_update = current_time
        else:
            print(text)
    
    def clear(self):
        """Clear the output buffer."""
        self.buffer.clear()
        if sys.platform == "win32":
            os.system("cls")
        else:
            os.system("clear")
    
    def refresh(self):
        """Refresh the entire display."""
        self.clear()
        for line in self.buffer:
            print(line)
        if self.auto_scroll:
            self.scroller.scroll_to_bottom()


class ProgressTracker:
    """
    Track and display progress with auto-scrolling.
    """
    
    def __init__(self, 
                 total: int,
                 description: str = "Progress",
                 auto_scroll: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Progress description
            auto_scroll: Whether to auto-scroll
        """
        self.total = total
        self.current = 0
        self.description = description
        self.scroller = ConsoleAutoScroller(auto_scroll)
        self.start_time = time.time()
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """
        Update progress and auto-scroll.
        
        Args:
            increment: Amount to increment
            message: Optional message to display
        """
        self.current = min(self.current + increment, self.total)
        
        # Calculate progress
        progress = self.current / self.total if self.total > 0 else 0
        bar_width = 30
        filled = int(progress * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        # Build progress line
        line = f"\r{self.description}: [{bar}] {self.current}/{self.total} ({progress*100:.1f}%) {eta_str}"
        
        if message:
            line += f" - {message}"
        
        # Print and scroll
        self.scroller.print_and_scroll(line, end="")
    
    def finish(self, message: str = "Complete"):
        """
        Finish progress tracking.
        
        Args:
            message: Completion message
        """
        self.current = self.total
        elapsed = time.time() - self.start_time
        
        line = f"\r{self.description}: [{'█' * 30}] {self.total}/{self.total} (100.0%) - {message} ({elapsed:.1f}s)"
        self.scroller.print_and_scroll(line)


class LiveLogger:
    """
    Live logging with auto-scroll support.
    """
    
    def __init__(self, 
                 name: str = "Log",
                 auto_scroll: bool = True,
                 show_timestamp: bool = True):
        """
        Initialize live logger.
        
        Args:
            name: Logger name
            auto_scroll: Whether to auto-scroll
            show_timestamp: Whether to show timestamps
        """
        self.name = name
        self.show_timestamp = show_timestamp
        self.scroller = ConsoleAutoScroller(auto_scroll)
        self.output = ScrollingOutput(auto_scroll=auto_scroll)
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with auto-scroll.
        
        Args:
            message: Message to log
            level: Log level
        """
        if self.show_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            line = f"[{timestamp}] [{level:5}] {self.name}: {message}"
        else:
            line = f"[{level:5}] {self.name}: {message}"
        
        # Color based on level
        if level == "ERROR":
            from colorama import Fore, Style
            line = f"{Fore.RED}{line}{Style.RESET_ALL}"
        elif level == "WARN":
            from colorama import Fore, Style
            line = f"{Fore.YELLOW}{line}{Style.RESET_ALL}"
        elif level == "SUCCESS":
            from colorama import Fore, Style
            line = f"{Fore.GREEN}{line}{Style.RESET_ALL}"
        
        self.output.add_line(line, force_scroll=True)
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")
    
    def warn(self, message: str):
        """Log warning message."""
        self.log(message, "WARN")
    
    def success(self, message: str):
        """Log success message."""
        self.log(message, "SUCCESS")


def enable_auto_scroll():
    """Enable global auto-scrolling for print statements."""
    global _auto_scroll_enabled
    _auto_scroll_enabled = True
    
    # Monkey-patch print function
    original_print = print
    scroller = ConsoleAutoScroller(True)
    
    def auto_scroll_print(*args, **kwargs):
        original_print(*args, **kwargs)
        if _auto_scroll_enabled:
            scroller.scroll_to_bottom()
    
    import builtins
    builtins.print = auto_scroll_print
    
    print("✅ Auto-scroll enabled for console output")


def disable_auto_scroll():
    """Disable global auto-scrolling."""
    global _auto_scroll_enabled
    _auto_scroll_enabled = False
    print("❌ Auto-scroll disabled")


# Global flag
_auto_scroll_enabled = False


class ScrollingMenu:
    """
    Interactive menu with auto-scrolling.
    """
    
    def __init__(self, 
                 title: str,
                 options: List[str],
                 auto_scroll: bool = True):
        """
        Initialize scrolling menu.
        
        Args:
            title: Menu title
            options: List of menu options
            auto_scroll: Whether to auto-scroll
        """
        self.title = title
        self.options = options
        self.scroller = ConsoleAutoScroller(auto_scroll)
    
    def display(self) -> int:
        """
        Display menu and get selection.
        
        Returns:
            Selected option index (0-based)
        """
        while True:
            # Clear and display menu
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")
            
            print("=" * 60)
            print(self.title.center(60))
            print("=" * 60)
            
            for i, option in enumerate(self.options, 1):
                print(f"{i}. {option}")
            
            print("0. Exit")
            print("=" * 60)
            
            # Auto-scroll to bottom
            self.scroller.scroll_to_bottom()
            
            # Get selection
            try:
                choice = int(input("\nSelect option: "))
                if 0 <= choice <= len(self.options):
                    return choice - 1
                else:
                    self.scroller.print_and_scroll("Invalid choice. Please try again.")
                    time.sleep(1)
            except ValueError:
                self.scroller.print_and_scroll("Invalid input. Please enter a number.")
                time.sleep(1)


# Demo functions
def demo_auto_scroll():
    """Demonstrate auto-scrolling functionality."""
    print("\n" + "="*60)
    print("AUTO-SCROLL DEMONSTRATION")
    print("="*60)
    
    scroller = ConsoleAutoScroller()
    
    print("\n1. Basic auto-scrolling:")
    for i in range(10):
        scroller.print_and_scroll(f"  Line {i+1} - Auto-scrolling to show this...")
        time.sleep(0.2)
    
    print("\n2. Progress tracking with auto-scroll:")
    tracker = ProgressTracker(50, "Processing")
    for i in range(50):
        tracker.update(1, f"Item {i+1}")
        time.sleep(0.05)
    tracker.finish()
    
    print("\n3. Live logging with auto-scroll:")
    logger = LiveLogger("Demo")
    logger.info("Starting process...")
    time.sleep(0.5)
    logger.warn("This is a warning")
    time.sleep(0.5)
    logger.success("Task completed successfully!")
    time.sleep(0.5)
    logger.error("This would be an error message")
    
    print("\n" + "="*60)
    print("✅ Auto-scroll demo complete!")


if __name__ == "__main__":
    from colorama import init
    init(autoreset=True)
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                       CONSOLE AUTO-SCROLL UTILITIES                        ║
║                                                                            ║
║  Automatically scrolls console to show the latest output line,            ║
║  ensuring important updates are always visible.                           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run demo
    demo_auto_scroll()
    
    print("\n4. Testing global auto-scroll:")
    enable_auto_scroll()
    
    for i in range(5):
        print(f"  Auto-scrolled line {i+1}")
        time.sleep(0.3)
    
    disable_auto_scroll()
    print("\n✨ All demos complete!")