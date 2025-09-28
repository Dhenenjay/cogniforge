"""
Code Generator with File Path Display and Editor Integration

This module provides code generation capabilities that automatically display
the generated file path and open it in a read-only editor pane.
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import shutil
import platform
import webbrowser
from colorama import init, Fore, Style, Back
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import ast
import black
import autopep8
import isort

# Initialize colorama
init(autoreset=True)


@dataclass
class CodeGenConfig:
    """Configuration for code generation"""
    output_dir: str = "generated_code"
    template_dir: str = "templates"
    auto_format: bool = True
    add_comments: bool = True
    add_docstrings: bool = True
    open_in_editor: bool = True
    editor_read_only: bool = True
    show_line_numbers: bool = True
    syntax_highlighting: bool = True
    auto_save: bool = True
    backup_existing: bool = True


class EditorWindow:
    """Read-only editor window for viewing generated code"""
    
    def __init__(self, filepath: Path, title: str = None, read_only: bool = True):
        """
        Initialize editor window
        
        Args:
            filepath: Path to file to display
            title: Window title
            read_only: Whether editor is read-only
        """
        self.filepath = filepath
        self.read_only = read_only
        self.root = tk.Tk()
        
        # Set window title
        if title:
            self.root.title(title)
        else:
            self.root.title(f"Code Viewer - {filepath.name} (Read-Only)" if read_only else f"Code Editor - {filepath.name}")
        
        # Set window size and position
        self.root.geometry("1000x700")
        self.center_window()
        
        # Create UI
        self.create_widgets()
        
        # Load file content
        self.load_file()
        
        # Set read-only if specified
        if self.read_only:
            self.text_editor.config(state=tk.DISABLED)
            
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def create_widgets(self):
        """Create UI widgets"""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Copy Path", command=self.copy_path)
        file_menu.add_command(label="Open in System Editor", command=self.open_in_system_editor)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Line Numbers", command=self.toggle_line_numbers)
        view_menu.add_command(label="Toggle Word Wrap", command=self.toggle_word_wrap)
        
        # Create toolbar frame
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File path label
        self.path_label = ttk.Label(toolbar_frame, text=f"📁 {self.filepath}", 
                                   font=("Consolas", 10))
        self.path_label.pack(side=tk.LEFT, padx=5)
        
        # Copy path button
        ttk.Button(toolbar_frame, text="📋 Copy Path", 
                  command=self.copy_path).pack(side=tk.LEFT, padx=2)
        
        # Open in editor button
        ttk.Button(toolbar_frame, text="🔗 Open in Editor", 
                  command=self.open_in_system_editor).pack(side=tk.LEFT, padx=2)
        
        # Read-only indicator
        if self.read_only:
            ro_label = ttk.Label(toolbar_frame, text="🔒 READ-ONLY", 
                               font=("Arial", 10, "bold"), foreground="red")
            ro_label.pack(side=tk.RIGHT, padx=5)
        
        # Create main frame with line numbers
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Line numbers text widget
        self.line_numbers = tk.Text(main_frame, width=4, padx=3, 
                                   takefocus=0, border=0, state='disabled',
                                   background='lightgray', foreground='black',
                                   font=("Consolas", 11))
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create text editor with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Text editor
        self.text_editor = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.NONE,
            font=("Consolas", 11),
            undo=True,
            background='#1e1e1e',
            foreground='#d4d4d4',
            insertbackground='white',
            selectbackground='#264f78',
            selectforeground='white'
        )
        self.text_editor.pack(fill=tk.BOTH, expand=True)
        
        # Configure syntax highlighting tags
        self.configure_syntax_highlighting()
        
        # Bind events
        self.text_editor.bind('<KeyRelease>', self.on_content_changed)
        self.text_editor.bind('<MouseWheel>', self.sync_scroll)
        self.text_editor.bind('<Button-4>', self.sync_scroll)
        self.text_editor.bind('<Button-5>', self.sync_scroll)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def configure_syntax_highlighting(self):
        """Configure syntax highlighting for Python code"""
        # Keywords
        self.text_editor.tag_config('keyword', foreground='#569cd6')
        self.text_editor.tag_config('builtin', foreground='#4ec9b0')
        self.text_editor.tag_config('string', foreground='#ce9178')
        self.text_editor.tag_config('comment', foreground='#6a9955')
        self.text_editor.tag_config('number', foreground='#b5cea8')
        self.text_editor.tag_config('function', foreground='#dcdcaa')
        self.text_editor.tag_config('class', foreground='#4ec9b0')
        self.text_editor.tag_config('decorator', foreground='#ffd700')
        
    def load_file(self):
        """Load file content into editor"""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Enable temporarily to insert content
            if self.read_only:
                self.text_editor.config(state=tk.NORMAL)
                
            self.text_editor.delete(1.0, tk.END)
            self.text_editor.insert(1.0, content)
            
            # Apply syntax highlighting
            self.apply_syntax_highlighting()
            
            # Update line numbers
            self.update_line_numbers()
            
            # Set back to read-only
            if self.read_only:
                self.text_editor.config(state=tk.DISABLED)
                
            # Update status
            file_size = os.path.getsize(self.filepath)
            line_count = content.count('\n') + 1
            self.status_bar.config(text=f"Loaded: {self.filepath.name} | Lines: {line_count} | Size: {file_size} bytes")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            
    def apply_syntax_highlighting(self):
        """Apply syntax highlighting to Python code"""
        if not self.filepath.suffix == '.py':
            return
            
        content = self.text_editor.get(1.0, tk.END)
        
        # Python keywords
        keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 
                   'try', 'except', 'finally', 'with', 'as', 'import', 'from',
                   'return', 'yield', 'lambda', 'async', 'await', 'pass',
                   'break', 'continue', 'raise', 'assert', 'del', 'global',
                   'nonlocal', 'in', 'is', 'and', 'or', 'not']
        
        # Apply keyword highlighting
        for keyword in keywords:
            start_idx = '1.0'
            while True:
                pos = self.text_editor.search(rf'\b{keyword}\b', start_idx, tk.END, regexp=True)
                if not pos:
                    break
                end_idx = f'{pos}+{len(keyword)}c'
                self.text_editor.tag_add('keyword', pos, end_idx)
                start_idx = end_idx
                
        # Highlight strings (simple approach)
        for quote in ['"', "'"]:
            start_idx = '1.0'
            while True:
                pos = self.text_editor.search(quote, start_idx, tk.END)
                if not pos:
                    break
                end_pos = self.text_editor.search(quote, f'{pos}+1c', tk.END)
                if end_pos:
                    self.text_editor.tag_add('string', pos, f'{end_pos}+1c')
                    start_idx = f'{end_pos}+1c'
                else:
                    break
                    
        # Highlight comments
        start_idx = '1.0'
        while True:
            pos = self.text_editor.search('#', start_idx, tk.END)
            if not pos:
                break
            line_end = self.text_editor.index(f'{pos} lineend')
            self.text_editor.tag_add('comment', pos, line_end)
            start_idx = f'{line_end}+1c'
            
    def update_line_numbers(self):
        """Update line numbers display"""
        self.line_numbers.config(state=tk.NORMAL)
        self.line_numbers.delete(1.0, tk.END)
        
        content = self.text_editor.get(1.0, tk.END)
        lines = content.split('\n')
        line_numbers_text = '\n'.join(str(i) for i in range(1, len(lines)))
        
        self.line_numbers.insert(1.0, line_numbers_text)
        self.line_numbers.config(state=tk.DISABLED)
        
    def on_content_changed(self, event=None):
        """Handle content changes"""
        self.update_line_numbers()
        
    def sync_scroll(self, event=None):
        """Synchronize scrolling between line numbers and text editor"""
        self.line_numbers.yview_moveto(self.text_editor.yview()[0])
        
    def toggle_line_numbers(self):
        """Toggle line numbers visibility"""
        if self.line_numbers.winfo_viewable():
            self.line_numbers.pack_forget()
        else:
            self.line_numbers.pack(side=tk.LEFT, fill=tk.Y, before=self.text_editor.master)
            
    def toggle_word_wrap(self):
        """Toggle word wrap"""
        if self.text_editor.cget('wrap') == tk.NONE:
            self.text_editor.config(wrap=tk.WORD)
        else:
            self.text_editor.config(wrap=tk.NONE)
            
    def copy_path(self):
        """Copy file path to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(str(self.filepath))
        self.status_bar.config(text=f"Path copied: {self.filepath}")
        
    def open_in_system_editor(self):
        """Open file in system default editor"""
        try:
            if platform.system() == 'Windows':
                os.startfile(self.filepath)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', self.filepath])
            else:  # Linux
                subprocess.call(['xdg-open', self.filepath])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open in system editor: {e}")
            
    def show(self):
        """Show the editor window"""
        self.root.mainloop()


class CodeGenerator:
    """Main code generator class"""
    
    def __init__(self, config: Optional[CodeGenConfig] = None):
        """
        Initialize code generator
        
        Args:
            config: Configuration for code generation
        """
        self.config = config or CodeGenConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_files = []
        
    def generate_python_file(self, 
                            filename: str,
                            code_content: str,
                            description: str = "",
                            auto_open: bool = True) -> Path:
        """
        Generate a Python file and open it in editor
        
        Args:
            filename: Name of the file to generate
            code_content: Python code content
            description: Description of the generated code
            auto_open: Whether to auto-open in editor
            
        Returns:
            Path to generated file
        """
        # Ensure .py extension
        if not filename.endswith('.py'):
            filename += '.py'
            
        filepath = self.output_dir / filename
        
        # Backup existing file if needed
        if filepath.exists() and self.config.backup_existing:
            backup_path = filepath.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak')
            shutil.copy2(filepath, backup_path)
            print(f"{Fore.YELLOW}📦 Backed up existing file to: {backup_path}{Style.RESET_ALL}")
        
        # Format code if enabled
        if self.config.auto_format:
            code_content = self.format_python_code(code_content)
        
        # Add header comment
        if self.config.add_comments:
            header = self.generate_header(filename, description)
            code_content = header + "\n\n" + code_content
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        # Print file path with formatting
        self._print_file_generated(filepath, description)
        
        # Record generated file
        self.generated_files.append(filepath)
        
        # Open in editor if requested
        if auto_open and self.config.open_in_editor:
            self.open_in_editor(filepath)
        
        return filepath
    
    def generate_header(self, filename: str, description: str) -> str:
        """Generate file header comment"""
        header = f'"""\n'
        header += f'Generated File: {filename}\n'
        header += f'Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        if description:
            header += f'Description: {description}\n'
        header += f'Generator: CodeGenerator v1.0\n'
        header += f'"""\n'
        return header
    
    def format_python_code(self, code: str) -> str:
        """Format Python code using black and isort"""
        try:
            # Format imports with isort
            code = isort.code(code)
            
            # Format with black
            code = black.format_str(code, mode=black.Mode())
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Could not auto-format code: {e}{Style.RESET_ALL}")
            
            # Fallback to autopep8
            try:
                code = autopep8.fix_code(code)
            except:
                pass
                
        return code
    
    def _print_file_generated(self, filepath: Path, description: str = ""):
        """Print formatted file generation message"""
        print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ CODE GENERATED SUCCESSFULLY{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")
        
        # File details
        print(f"📄 {Fore.CYAN}File Name:{Style.RESET_ALL} {filepath.name}")
        print(f"📁 {Fore.CYAN}Full Path:{Style.RESET_ALL} {filepath.absolute()}")
        print(f"📂 {Fore.CYAN}Directory:{Style.RESET_ALL} {filepath.parent}")
        
        if description:
            print(f"📝 {Fore.CYAN}Description:{Style.RESET_ALL} {description}")
        
        # File stats
        if filepath.exists():
            size = filepath.stat().st_size
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            print(f"📊 {Fore.CYAN}File Size:{Style.RESET_ALL} {size:,} bytes")
            print(f"📏 {Fore.CYAN}Lines of Code:{Style.RESET_ALL} {lines:,}")
        
        # Copy instruction
        print(f"\n💡 {Fore.YELLOW}To copy path:{Style.RESET_ALL} Click 'Copy Path' in the editor window")
        print(f"💡 {Fore.YELLOW}To open in IDE:{Style.RESET_ALL} Click 'Open in Editor' or press Ctrl+O")
        
    def open_in_editor(self, filepath: Path, read_only: Optional[bool] = None):
        """
        Open generated file in editor pane
        
        Args:
            filepath: Path to file
            read_only: Whether to open as read-only
        """
        if read_only is None:
            read_only = self.config.editor_read_only
        
        print(f"\n{Fore.CYAN}🔓 Opening in {'read-only' if read_only else 'editable'} editor...{Style.RESET_ALL}")
        
        # Create editor window in separate thread to avoid blocking
        def open_editor():
            editor = EditorWindow(filepath, read_only=read_only)
            editor.show()
        
        editor_thread = threading.Thread(target=open_editor, daemon=True)
        editor_thread.start()
        
        # Also try to open in system editor if available
        if not read_only:
            self.try_open_in_ide(filepath)
    
    def try_open_in_ide(self, filepath: Path):
        """Try to open file in common IDEs"""
        # Check for common IDEs
        ide_commands = [
            ('code', [str(filepath)]),  # VS Code
            ('subl', [str(filepath)]),   # Sublime Text
            ('atom', [str(filepath)]),   # Atom
            ('pycharm', [str(filepath)]), # PyCharm
            ('notepad++', [str(filepath)]) # Notepad++
        ]
        
        for ide_name, cmd in ide_commands:
            try:
                # Check if IDE is available
                result = subprocess.run(['where', ide_name], 
                                      capture_output=True, 
                                      text=True,
                                      shell=True)
                if result.returncode == 0:
                    # IDE found, try to open
                    subprocess.Popen([ide_name] + cmd, shell=True)
                    print(f"📝 Also opened in {ide_name}")
                    break
            except:
                continue
    
    def generate_class(self, class_name: str, 
                      methods: List[Dict[str, Any]],
                      base_class: Optional[str] = None,
                      description: str = "") -> Path:
        """
        Generate a Python class
        
        Args:
            class_name: Name of the class
            methods: List of method definitions
            base_class: Optional base class
            description: Class description
            
        Returns:
            Path to generated file
        """
        code = f"class {class_name}"
        if base_class:
            code += f"({base_class})"
        code += ":\n"
        
        if description and self.config.add_docstrings:
            code += f'    """{description}"""\n\n'
        
        # Add __init__ if not provided
        has_init = any(m.get('name') == '__init__' for m in methods)
        if not has_init:
            code += "    def __init__(self):\n"
            code += "        pass\n\n"
        
        # Add methods
        for method in methods:
            code += self._generate_method(method)
            code += "\n"
        
        filename = f"{class_name.lower()}.py"
        return self.generate_python_file(filename, code, description)
    
    def _generate_method(self, method: Dict[str, Any]) -> str:
        """Generate a method definition"""
        name = method.get('name', 'method')
        params = method.get('params', ['self'])
        returns = method.get('returns', 'None')
        docstring = method.get('docstring', '')
        body = method.get('body', 'pass')
        
        # Build method signature
        if isinstance(params, list):
            params_str = ', '.join(params)
        else:
            params_str = params
            
        code = f"    def {name}({params_str}) -> {returns}:\n"
        
        # Add docstring
        if docstring and self.config.add_docstrings:
            code += f'        """{docstring}"""\n'
            
        # Add body
        body_lines = body.split('\n') if isinstance(body, str) else body
        for line in body_lines:
            code += f"        {line}\n"
            
        return code
    
    def list_generated_files(self) -> List[Path]:
        """List all generated files"""
        print(f"\n{Fore.CYAN}📋 Generated Files:{Style.RESET_ALL}")
        for i, filepath in enumerate(self.generated_files, 1):
            print(f"  {i}. {filepath}")
        return self.generated_files


def demo_code_generator():
    """Demonstrate code generator with automatic editor opening"""
    
    print(f"{Fore.CYAN}{'='*70}")
    print(f" CODE GENERATOR DEMONSTRATION")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # Create generator with config
    config = CodeGenConfig(
        output_dir="generated_code",
        auto_format=True,
        add_comments=True,
        add_docstrings=True,
        open_in_editor=True,
        editor_read_only=True  # Open as read-only
    )
    
    generator = CodeGenerator(config)
    
    # Example 1: Generate a simple function
    print(f"{Fore.YELLOW}Generating example function...{Style.RESET_ALL}")
    
    code = '''
import numpy as np
from typing import List, Tuple

def calculate_trajectory(waypoints: List[Tuple[float, float]], 
                        timestep: float = 0.1) -> np.ndarray:
    """Calculate trajectory from waypoints"""
    trajectory = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i + 1])
        
        # Linear interpolation
        num_steps = int(np.linalg.norm(end - start) / timestep)
        for t in np.linspace(0, 1, num_steps):
            point = start + t * (end - start)
            trajectory.append(point)
    
    return np.array(trajectory)

def optimize_waypoints(waypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Optimize waypoints by removing redundant points"""
    if len(waypoints) <= 2:
        return waypoints
    
    optimized = [waypoints[0]]
    
    for i in range(1, len(waypoints) - 1):
        prev = np.array(waypoints[i - 1])
        curr = np.array(waypoints[i])
        next = np.array(waypoints[i + 1])
        
        # Check if current point is on the line between prev and next
        v1 = curr - prev
        v2 = next - prev
        
        # Calculate cross product
        cross = np.abs(np.cross(v1, v2))
        
        # Keep point if not collinear
        if cross > 0.01:
            optimized.append(waypoints[i])
    
    optimized.append(waypoints[-1])
    return optimized
'''
    
    filepath1 = generator.generate_python_file(
        "trajectory_utils.py",
        code,
        "Trajectory calculation and optimization utilities"
    )
    
    time.sleep(2)
    
    # Example 2: Generate a class
    print(f"\n{Fore.YELLOW}Generating robot controller class...{Style.RESET_ALL}")
    
    methods = [
        {
            'name': '__init__',
            'params': ['self', 'robot_name: str = "Robot"'],
            'returns': 'None',
            'docstring': 'Initialize robot controller',
            'body': 'self.robot_name = robot_name\nself.position = np.zeros(3)\nself.is_moving = False'
        },
        {
            'name': 'move_to',
            'params': ['self', 'target: np.ndarray'],
            'returns': 'bool',
            'docstring': 'Move robot to target position',
            'body': '''self.is_moving = True
        print(f"Moving {self.robot_name} to {target}")
        self.position = target
        self.is_moving = False
        return True'''
        },
        {
            'name': 'get_position',
            'params': ['self'],
            'returns': 'np.ndarray',
            'docstring': 'Get current robot position',
            'body': 'return self.position.copy()'
        }
    ]
    
    filepath2 = generator.generate_class(
        "RobotController",
        methods,
        base_class=None,
        description="Simple robot controller for demonstration"
    )
    
    time.sleep(2)
    
    # Example 3: Generate config file
    print(f"\n{Fore.YELLOW}Generating configuration file...{Style.RESET_ALL}")
    
    config_code = '''
from dataclasses import dataclass
from typing import Optional

@dataclass
class SystemConfig:
    """System configuration parameters"""
    
    # Robot parameters
    robot_name: str = "CogniForge"
    max_velocity: float = 1.0  # m/s
    max_acceleration: float = 2.0  # m/s^2
    
    # Vision parameters
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 30
    detection_threshold: float = 0.8
    
    # Control parameters
    control_frequency: int = 100  # Hz
    position_tolerance: float = 0.001  # meters
    orientation_tolerance: float = 0.01  # radians
    
    # Safety parameters
    emergency_stop_distance: float = 0.05  # meters
    max_force: float = 100.0  # Newtons
    collision_detection: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.max_velocity <= 0:
            raise ValueError("max_velocity must be positive")
        if self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        return True
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'robot_name': self.robot_name,
            'max_velocity': self.max_velocity,
            'max_acceleration': self.max_acceleration,
            'camera_resolution': self.camera_resolution,
            'camera_fps': self.camera_fps,
            'detection_threshold': self.detection_threshold,
            'control_frequency': self.control_frequency,
            'position_tolerance': self.position_tolerance,
            'orientation_tolerance': self.orientation_tolerance,
            'emergency_stop_distance': self.emergency_stop_distance,
            'max_force': self.max_force,
            'collision_detection': self.collision_detection
        }

# Default configuration instance
default_config = SystemConfig()
'''
    
    filepath3 = generator.generate_python_file(
        "system_config.py",
        config_code,
        "System configuration with default parameters"
    )
    
    # List all generated files
    print(f"\n{Fore.GREEN}Summary:{Style.RESET_ALL}")
    generator.list_generated_files()
    
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")
    print(f"\n💡 All files opened in read-only editor windows")
    print(f"💡 File paths have been printed and can be copied from the editor")
    
    # Keep program running to maintain editor windows
    print(f"\n{Fore.YELLOW}Press Enter to exit and close all editor windows...{Style.RESET_ALL}")
    input()


if __name__ == "__main__":
    demo_code_generator()