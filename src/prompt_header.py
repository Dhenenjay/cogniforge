"""
Prompt Header Utility

Displays the current task/prompt as a formatted header in the console.
This helps track what the system is currently working on.
"""

import sys
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from colorama import init, Fore, Back, Style
import textwrap
import json

# Initialize colorama for cross-platform color support
init(autoreset=True)


class PromptHeader:
    """Displays formatted headers for prompts and tasks"""
    
    # Header styles
    STYLES = {
        'default': {
            'border': '=',
            'width': 80,
            'color': Fore.CYAN,
            'bg': '',
            'align': 'center'
        },
        'task': {
            'border': '─',
            'width': 80,
            'color': Fore.GREEN,
            'bg': '',
            'align': 'left'
        },
        'warning': {
            'border': '!',
            'width': 80,
            'color': Fore.YELLOW,
            'bg': '',
            'align': 'center'
        },
        'error': {
            'border': '#',
            'width': 80,
            'color': Fore.RED,
            'bg': '',
            'align': 'center'
        },
        'info': {
            'border': '·',
            'width': 80,
            'color': Fore.BLUE,
            'bg': '',
            'align': 'left'
        },
        'success': {
            'border': '✓',
            'width': 80,
            'color': Fore.GREEN + Style.BRIGHT,
            'bg': '',
            'align': 'center'
        }
    }
    
    @staticmethod
    def print_header(prompt: str, style: str = 'default', 
                    subtitle: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    show_timestamp: bool = True):
        """
        Print a formatted header with the prompt
        
        Args:
            prompt: The main prompt/task text
            style: Style name from STYLES
            subtitle: Optional subtitle
            metadata: Optional metadata to display
            show_timestamp: Whether to show timestamp
        """
        config = PromptHeader.STYLES.get(style, PromptHeader.STYLES['default'])
        
        width = config['width']
        border = config['border']
        color = config['color']
        bg = config['bg']
        align = config['align']
        
        # Create border line
        border_line = border * width
        
        # Format main prompt
        if len(prompt) > width - 4:
            # Wrap long prompts
            wrapped = textwrap.wrap(prompt, width=width - 4)
        else:
            wrapped = [prompt]
            
        # Print header
        print()
        print(f"{color}{bg}{border_line}{Style.RESET_ALL}")
        
        # Print wrapped prompt lines
        for line in wrapped:
            if align == 'center':
                formatted = line.center(width - 2)
            elif align == 'right':
                formatted = line.rjust(width - 2)
            else:  # left
                formatted = line.ljust(width - 2)
            print(f"{color}{bg}{border} {formatted} {border}{Style.RESET_ALL}")
            
        # Print subtitle if provided
        if subtitle:
            sub_formatted = subtitle.center(width - 2) if align == 'center' else subtitle.ljust(width - 2)
            print(f"{color}{bg}{border} {sub_formatted} {border}{Style.RESET_ALL}")
            
        # Print timestamp if requested
        if show_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ts_formatted = timestamp.center(width - 2) if align == 'center' else timestamp.rjust(width - 2)
            print(f"{color}{bg}{border} {ts_formatted} {border}{Style.RESET_ALL}")
            
        # Print metadata if provided
        if metadata:
            print(f"{color}{bg}{border}{' ' * (width - 2)}{border}{Style.RESET_ALL}")
            for key, value in metadata.items():
                meta_line = f"{key}: {value}"
                if len(meta_line) > width - 4:
                    meta_line = meta_line[:width - 7] + "..."
                meta_formatted = meta_line.ljust(width - 2)
                print(f"{color}{bg}{border} {meta_formatted} {border}{Style.RESET_ALL}")
                
        print(f"{color}{bg}{border_line}{Style.RESET_ALL}")
        print()
        
    @staticmethod
    def print_compact(prompt: str, prefix: str = "►", 
                     color: str = Fore.CYAN):
        """
        Print a compact one-line header
        
        Args:
            prompt: The prompt text
            prefix: Prefix symbol
            color: Color to use
        """
        print(f"\n{color}{prefix} {prompt}{Style.RESET_ALL}")
        
    @staticmethod
    def print_section(title: str, items: List[str], 
                      style: str = 'info'):
        """
        Print a section with title and items
        
        Args:
            title: Section title
            items: List of items to display
            style: Style to use
        """
        config = PromptHeader.STYLES.get(style, PromptHeader.STYLES['default'])
        color = config['color']
        
        print(f"\n{color}┌── {title} ──")
        for item in items:
            print(f"{color}│  • {item}")
        print(f"{color}└{'─' * (len(title) + 6)}{Style.RESET_ALL}\n")
        
    @staticmethod
    def print_progress(prompt: str, current: int, total: int,
                      bar_width: int = 40):
        """
        Print a progress bar with prompt
        
        Args:
            prompt: The prompt text
            current: Current progress
            total: Total items
            bar_width: Width of progress bar
        """
        percent = current / total if total > 0 else 0
        filled = int(bar_width * percent)
        
        bar = '█' * filled + '░' * (bar_width - filled)
        
        color = Fore.GREEN if percent >= 1 else Fore.YELLOW if percent >= 0.5 else Fore.CYAN
        
        print(f"\r{color}► {prompt}: [{bar}] {current}/{total} ({percent*100:.1f}%){Style.RESET_ALL}", 
              end='', flush=True)
        
        if current >= total:
            print()  # New line when complete
            
    @staticmethod
    def print_tree(prompt: str, tree_data: Dict, indent: str = ""):
        """
        Print a tree structure with the prompt as root
        
        Args:
            prompt: Root prompt
            tree_data: Tree structure as nested dict
            indent: Current indentation
        """
        if not indent:  # Root level
            print(f"\n{Fore.CYAN}🌳 {prompt}{Style.RESET_ALL}")
            
        for i, (key, value) in enumerate(tree_data.items()):
            is_last = i == len(tree_data) - 1
            
            # Choose connector
            connector = "└── " if is_last else "├── "
            
            # Print current level
            print(f"{Fore.GREEN}{indent}{connector}{key}{Style.RESET_ALL}")
            
            # Recursively print children if dict
            if isinstance(value, dict):
                extension = "    " if is_last else "│   "
                PromptHeader.print_tree("", value, indent + extension)
            elif isinstance(value, list):
                extension = "    " if is_last else "│   "
                for item in value:
                    print(f"{Fore.BLUE}{indent}{extension}• {item}{Style.RESET_ALL}")
                    
    @staticmethod
    def print_comparison(prompt: str, before: Dict, after: Dict):
        """
        Print a before/after comparison
        
        Args:
            prompt: The prompt describing the comparison
            before: Before state
            after: After state
        """
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f" {prompt}")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        # Find all keys
        all_keys = set(before.keys()) | set(after.keys())
        
        print(f"{'Metric':<20} {'Before':>15} {'After':>15} {'Change':>15}")
        print("-" * 65)
        
        for key in sorted(all_keys):
            before_val = before.get(key, 'N/A')
            after_val = after.get(key, 'N/A')
            
            # Calculate change if both are numbers
            if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                change = after_val - before_val
                change_pct = (change / before_val * 100) if before_val != 0 else 0
                
                # Color based on change
                if change > 0:
                    change_color = Fore.GREEN
                    change_str = f"+{change:.2f} ({change_pct:+.1f}%)"
                elif change < 0:
                    change_color = Fore.RED
                    change_str = f"{change:.2f} ({change_pct:.1f}%)"
                else:
                    change_color = Fore.YELLOW
                    change_str = "No change"
                    
                print(f"{key:<20} {before_val:>15.2f} {after_val:>15.2f} "
                      f"{change_color}{change_str:>15}{Style.RESET_ALL}")
            else:
                print(f"{key:<20} {str(before_val):>15} {str(after_val):>15} {'---':>15}")


class PromptLogger:
    """Logs prompts to file for history tracking"""
    
    def __init__(self, log_file: str = "prompts.log"):
        """
        Initialize prompt logger
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        
    def log(self, prompt: str, response: Optional[str] = None,
            metadata: Optional[Dict] = None):
        """
        Log a prompt and optional response
        
        Args:
            prompt: The prompt text
            response: Optional response text
            metadata: Optional metadata
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response,
            'metadata': metadata
        }
        
        # Append to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    def read_last(self, n: int = 10) -> List[Dict]:
        """
        Read last n prompts from log
        
        Args:
            n: Number of prompts to read
            
        Returns:
            List of prompt entries
        """
        if not os.path.exists(self.log_file):
            return []
            
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
                
        return entries


def auto_print_prompt(func):
    """
    Decorator to automatically print prompt from function docstring
    
    Usage:
        @auto_print_prompt
        def my_function():
            '''Process user data and optimize performance'''
            ...
    """
    def wrapper(*args, **kwargs):
        if func.__doc__:
            prompt = func.__doc__.strip()
            PromptHeader.print_header(prompt, style='task')
        return func(*args, **kwargs)
    return wrapper


# Global prompt display function for easy access
def show_prompt(prompt: str, **kwargs):
    """
    Quick function to display a prompt
    
    Args:
        prompt: The prompt text
        **kwargs: Additional arguments for print_header
    """
    PromptHeader.print_header(prompt, **kwargs)


# Demo and examples
def demo_prompt_headers():
    """Demonstrate various prompt header styles"""
    
    # Basic header
    PromptHeader.print_header(
        "Create a robotic manipulation system with PPO and BC",
        style='default'
    )
    
    # Task header with subtitle
    PromptHeader.print_header(
        "Implementing Waypoint Optimization",
        style='task',
        subtitle="Reducing trajectory complexity by 50%",
        metadata={'Algorithm': 'A*', 'Complexity': 'O(n log n)', 'Status': 'In Progress'}
    )
    
    # Warning header
    PromptHeader.print_header(
        "BC Model Not Loaded - Using Default Configuration",
        style='warning'
    )
    
    # Success header
    PromptHeader.print_header(
        "✅ Optimization Complete!",
        style='success',
        subtitle="Reduced 16 waypoints to 8 (-50%)"
    )
    
    # Compact headers
    PromptHeader.print_compact("Loading BC model...", "⚡", Fore.YELLOW)
    PromptHeader.print_compact("Executing optimized trajectory", "🚀", Fore.GREEN)
    
    # Section display
    PromptHeader.print_section(
        "Available Skills",
        ["Push", "Pull", "Stack", "Pick & Place"],
        style='info'
    )
    
    # Progress bar
    import time
    for i in range(11):
        PromptHeader.print_progress("Training BC Model", i, 10)
        time.sleep(0.1)
        
    # Tree structure
    tree = {
        "Motion Controller": {
            "BC Model": ["9 waypoints", "8.15s execution"],
            "Optimizer": ["5 waypoints", "4.23s execution"],
            "History": {
                "Past States": "8",
                "Future States": "2"
            }
        }
    }
    PromptHeader.print_tree("System Architecture", tree)
    
    # Comparison
    before = {'waypoints': 16, 'distance': 0.926, 'time': 8.15}
    after = {'waypoints': 8, 'distance': 0.623, 'time': 5.32}
    PromptHeader.print_comparison("Optimization Results", before, after)


@auto_print_prompt
def example_function():
    """Analyze sensor data and generate control signals"""
    print("Function is running...")
    return "Complete"


if __name__ == "__main__":
    print(f"{Fore.CYAN}{'='*80}")
    print(" PROMPT HEADER DEMONSTRATION")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    demo_prompt_headers()
    
    print("\n" + "="*80)
    print(" DECORATOR EXAMPLE")
    print("="*80)
    
    result = example_function()
    print(f"Result: {result}")
    
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")