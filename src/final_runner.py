"""
Final Runner with Demo Mode Toggle

This module provides a clean execution environment for generated scripts
with the ability to toggle between demo mode and production mode.
"""

import os
import sys
import subprocess
import importlib
import importlib.util
import traceback
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from colorama import init, Fore, Style, Back
import threading
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import logging

# Initialize colorama
init(autoreset=True)

# Global demo mode flag
DEMO_MODE = True


@dataclass
class RunConfig:
    """Configuration for script execution"""
    demo_mode: bool = True
    verbose: bool = True
    show_output: bool = True
    capture_output: bool = False
    measure_time: bool = True
    log_execution: bool = True
    suppress_warnings: bool = False
    clean_imports: bool = True
    reset_environment: bool = True
    output_dir: str = "execution_logs"
    max_execution_time: float = 300.0  # 5 minutes max


class DemoModeContext:
    """Context manager for demo mode settings"""
    
    _instance = None
    _demo_mode = True
    _original_print = print
    _suppressed_outputs = [
        "Press Enter to",
        "Demo complete",
        "Press any key",
        "Close window",
        "Exit to continue"
    ]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def is_demo_mode(cls) -> bool:
        """Check if demo mode is enabled"""
        return cls._demo_mode
    
    @classmethod
    def set_demo_mode(cls, enabled: bool):
        """Set demo mode state"""
        cls._demo_mode = enabled
        if enabled:
            print(f"{Fore.YELLOW}🎭 Demo Mode: ENABLED{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}⚡ Demo Mode: DISABLED (Production Mode){Style.RESET_ALL}")
    
    @classmethod
    def toggle_demo_mode(cls):
        """Toggle demo mode on/off"""
        cls.set_demo_mode(not cls._demo_mode)
        return cls._demo_mode
    
    @classmethod
    def should_suppress(cls, text: str) -> bool:
        """Check if output should be suppressed in production mode"""
        if cls._demo_mode:
            return False
        
        # Suppress demo-specific outputs in production mode
        text_lower = str(text).lower()
        return any(suppress.lower() in text_lower for suppress in cls._suppressed_outputs)


# Override print function to respect demo mode
_original_print = print

def demo_aware_print(*args, **kwargs):
    """Print function that respects demo mode settings"""
    # Convert args to string to check content
    output = ' '.join(str(arg) for arg in args)
    
    # Check if should suppress
    if not DemoModeContext.is_demo_mode() and DemoModeContext.should_suppress(output):
        return
    
    # Otherwise print normally
    _original_print(*args, **kwargs)

# Install demo-aware print
print = demo_aware_print


class ScriptRunner:
    """Runs Python scripts with demo mode control"""
    
    def __init__(self, config: Optional[RunConfig] = None):
        """
        Initialize script runner
        
        Args:
            config: Execution configuration
        """
        self.config = config or RunConfig()
        self.execution_history = []
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set demo mode
        DemoModeContext.set_demo_mode(self.config.demo_mode)
        
    def run_script(self, script_path: Union[str, Path], 
                  args: Optional[List[str]] = None,
                  env_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run a Python script with demo mode control
        
        Args:
            script_path: Path to Python script
            args: Command line arguments
            env_vars: Environment variables
            
        Returns:
            Execution results
        """
        script_path = Path(script_path)
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Print header
        self._print_execution_header(script_path)
        
        # Prepare environment
        env = self._prepare_environment(env_vars)
        
        # Choose execution method based on demo mode
        if DemoModeContext.is_demo_mode():
            result = self._run_with_demo_features(script_path, args, env)
        else:
            result = self._run_production_mode(script_path, args, env)
        
        # Log execution
        if self.config.log_execution:
            self._log_execution(script_path, result)
        
        # Print summary
        self._print_execution_summary(result)
        
        return result
    
    def _print_execution_header(self, script_path: Path):
        """Print execution header"""
        mode = "DEMO" if DemoModeContext.is_demo_mode() else "PRODUCTION"
        color = Fore.YELLOW if DemoModeContext.is_demo_mode() else Fore.GREEN
        
        print(f"\n{color}{'='*70}{Style.RESET_ALL}")
        print(f"{color}🚀 EXECUTING SCRIPT ({mode} MODE){Style.RESET_ALL}")
        print(f"{color}{'='*70}{Style.RESET_ALL}\n")
        print(f"📄 Script: {script_path.name}")
        print(f"📁 Path: {script_path.absolute()}")
        print(f"🎭 Mode: {mode}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _prepare_environment(self, env_vars: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Prepare environment variables"""
        env = os.environ.copy()
        
        # Add demo mode flag
        env['DEMO_MODE'] = '1' if DemoModeContext.is_demo_mode() else '0'
        env['PYTHONPATH'] = str(Path.cwd())
        
        # Force UTF-8 for child processes to avoid Windows cp1252 issues with emojis/colors
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Add custom env vars
        if env_vars:
            env.update(env_vars)
            
        return env
    
    def _run_with_demo_features(self, script_path: Path, 
                                args: Optional[List[str]], 
                                env: Dict[str, str]) -> Dict[str, Any]:
        """Run script with demo features enabled"""
        print(f"{Fore.YELLOW}🎭 Running with demo features...{Style.RESET_ALL}\n")
        
        start_time = time.time()
        
        # Build command
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        # Run with subprocess
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            output_lines = []
            error_lines = []
            
            # Read output with timeout
            import select
            import platform
            
            if platform.system() != 'Windows':
                # Unix-like systems
                while True:
                    reads = [process.stdout.fileno(), process.stderr.fileno()]
                    ret = select.select(reads, [], [], 0.1)
                    
                    for fd in ret[0]:
                        if fd == process.stdout.fileno():
                            line = process.stdout.readline()
                            if line:
                                print(f"  {line}", end='')
                                output_lines.append(line)
                        if fd == process.stderr.fileno():
                            line = process.stderr.readline()
                            if line:
                                print(f"  {Fore.RED}{line}{Style.RESET_ALL}", end='')
                                error_lines.append(line)
                    
                    if process.poll() is not None:
                        break
            else:
                # Windows - simpler approach
                stdout, stderr = process.communicate(timeout=self.config.max_execution_time)
                output_lines = stdout.splitlines(keepends=True) if stdout else []
                error_lines = stderr.splitlines(keepends=True) if stderr else []
                
                for line in output_lines:
                    print(f"  {line}", end='')
                for line in error_lines:
                    print(f"  {Fore.RED}{line}{Style.RESET_ALL}", end='')
            
            return_code = process.returncode
            execution_time = time.time() - start_time
            
            return {
                'success': return_code == 0,
                'return_code': return_code,
                'stdout': ''.join(output_lines),
                'stderr': ''.join(error_lines),
                'execution_time': execution_time,
                'mode': 'demo'
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Execution timeout ({self.config.max_execution_time}s)',
                'execution_time': self.config.max_execution_time,
                'mode': 'demo'
            }
        except Exception as e:
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': time.time() - start_time,
                'mode': 'demo'
            }
    
    def _run_production_mode(self, script_path: Path,
                            args: Optional[List[str]],
                            env: Dict[str, str]) -> Dict[str, Any]:
        """Run script in production mode (no demo artifacts)"""
        print(f"{Fore.GREEN}⚡ Running in production mode (clean output)...{Style.RESET_ALL}\n")
        
        start_time = time.time()
        
        # Modify script to skip demo sections
        modified_script = self._create_production_script(script_path)
        
        # Build command
        cmd = [sys.executable, str(modified_script)]
        if args:
            cmd.extend(args)
        
        # Run cleanly
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.config.max_execution_time
            )
            
            # Only show essential output
            if result.stdout:
                essential_output = self._filter_essential_output(result.stdout)
                if essential_output:
                    print(essential_output)
            
            if result.stderr and result.returncode != 0:
                print(f"{Fore.RED}Errors:{Style.RESET_ALL}")
                print(result.stderr)
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'mode': 'production'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Execution timeout ({self.config.max_execution_time}s)',
                'execution_time': self.config.max_execution_time,
                'mode': 'production'
            }
        except Exception as e:
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': time.time() - start_time,
                'mode': 'production'
            }
        finally:
            # Clean up modified script
            if 'modified_script' in locals() and modified_script.exists():
                modified_script.unlink()
    
    def _create_production_script(self, script_path: Path) -> Path:
        """Create modified script for production mode"""
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Inject demo mode flag
        production_header = """
import os
os.environ['DEMO_MODE'] = '0'

# Production mode - skip demo artifacts
def is_demo_mode():
    return False

"""
        
        # Replace common demo patterns
        replacements = [
            ('if __name__ == "__main__":', 
             'if __name__ == "__main__":\n    _demo_mode = False'),
            ('input()', '# input() - skipped in production'),
            ('plt.show()', 'plt.show(block=False) # Non-blocking in production'),
            ('time.sleep(', 'pass # time.sleep(')
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        # Write modified script
        modified_path = script_path.parent / f".{script_path.stem}_production.py"
        with open(modified_path, 'w', encoding='utf-8') as f:
            f.write(production_header)
            f.write(content)
        
        return modified_path
    
    def _filter_essential_output(self, output: str) -> str:
        """Filter output to show only essential information"""
        lines = output.splitlines()
        essential = []
        
        skip_patterns = [
            'Demo', 'demo',
            'Press Enter', 'press enter',
            'Example', 'example',
            '===', '---',
            'Complete!', 'complete!'
        ]
        
        for line in lines:
            # Skip demo-related lines
            if any(pattern in line for pattern in skip_patterns):
                continue
            
            # Skip empty lines
            if not line.strip():
                continue
                
            # Keep essential output
            if any(word in line.lower() for word in ['error', 'warning', 'result', 'output']):
                essential.append(line)
            elif line.startswith(('✅', '❌', '⚠️', '📊')):  # Keep status indicators
                essential.append(line)
        
        return '\n'.join(essential)
    
    def _print_execution_summary(self, result: Dict[str, Any]):
        """Print execution summary"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Execution Summary{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        color = Fore.GREEN if result['success'] else Fore.RED
        
        print(f"Status: {color}{status}{Style.RESET_ALL}")
        print(f"Mode: {result['mode'].upper()}")
        print(f"Return Code: {result['return_code']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        
        if not result['success'] and result['stderr']:
            print(f"\n{Fore.RED}Error Output:{Style.RESET_ALL}")
            print(result['stderr'][:500])  # First 500 chars of error
    
    def _log_execution(self, script_path: Path, result: Dict[str, Any]):
        """Log execution details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'script': str(script_path),
            'mode': result['mode'],
            'success': result['success'],
            'return_code': result['return_code'],
            'execution_time': result['execution_time']
        }
        
        log_file = self.output_dir / f"execution_log_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Read existing log
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new entry
        logs.append(log_entry)
        
        # Write back
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)


class FinalRunner:
    """Main runner with demo mode toggle"""
    
    def __init__(self):
        self.runner = None
        self.demo_mode = True
        
    def toggle_demo_mode(self) -> bool:
        """Toggle demo mode and return new state"""
        self.demo_mode = DemoModeContext.toggle_demo_mode()
        
        # Update runner config
        if self.runner:
            self.runner.config.demo_mode = self.demo_mode
            
        return self.demo_mode
    
    def run(self, script_path: Union[str, Path], 
           args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a script with current demo mode setting
        
        Args:
            script_path: Path to script
            args: Command line arguments
            
        Returns:
            Execution results
        """
        # Create runner with current config
        config = RunConfig(demo_mode=self.demo_mode)
        self.runner = ScriptRunner(config)
        
        # Run script
        return self.runner.run_script(script_path, args)
    
    def run_all_demos(self, demo_scripts: Optional[List[Path]] = None):
        """Run all demo scripts"""
        if demo_scripts is None:
            # Find all demo scripts in src directory
            src_dir = Path("src")
            demo_scripts = [
                src_dir / "motion_controller.py",
                src_dir / "waypoint_optimizer.py",
                src_dir / "bc_trainer_enhanced.py",
                src_dir / "behavior_tree_json.py",
                src_dir / "vision_system.py",
                src_dir / "expert_collector.py"
            ]
            demo_scripts = [s for s in demo_scripts if s.exists()]
        
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}RUNNING ALL DEMOS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        print(f"Found {len(demo_scripts)} demo scripts")
        print(f"Demo Mode: {'ON' if self.demo_mode else 'OFF'}\n")
        
        results = []
        for script in demo_scripts:
            print(f"\n{Fore.YELLOW}Running: {script.name}{Style.RESET_ALL}")
            try:
                result = self.run(script)
                results.append((script, result))
            except Exception as e:
                print(f"{Fore.RED}Failed to run {script.name}: {e}{Style.RESET_ALL}")
                results.append((script, {'success': False, 'error': str(e)}))
        
        # Print final summary
        self._print_final_summary(results)
        
    def _print_final_summary(self, results: List[tuple]):
        """Print summary of all executions"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}FINAL SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        successful = sum(1 for _, r in results if r.get('success', False))
        total = len(results)
        
        print(f"Total Scripts: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success Rate: {successful/total*100:.1f}%\n")
        
        print("Results:")
        for script, result in results:
            status = "✅" if result.get('success', False) else "❌"
            time_str = f"{result.get('execution_time', 0):.2f}s" if 'execution_time' in result else "N/A"
            print(f"  {status} {script.name:<30} ({time_str})")


def interactive_mode():
    """Interactive mode with demo toggle"""
    runner = FinalRunner()
    
    while True:
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}FINAL RUNNER - INTERACTIVE MODE{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        mode_status = "🎭 DEMO" if runner.demo_mode else "⚡ PRODUCTION"
        print(f"Current Mode: {mode_status}\n")
        
        print("Options:")
        print("  1. Toggle Demo Mode")
        print("  2. Run specific script")
        print("  3. Run all demos")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            new_mode = runner.toggle_demo_mode()
            mode_name = "DEMO" if new_mode else "PRODUCTION"
            print(f"\nMode changed to: {mode_name}")
            
        elif choice == '2':
            script_path = input("Enter script path: ").strip()
            if script_path:
                try:
                    runner.run(script_path)
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                    
        elif choice == '3':
            runner.run_all_demos()
            
        elif choice == '4':
            print("\nExiting...")
            break
            
        else:
            print(f"{Fore.RED}Invalid option{Style.RESET_ALL}")


def main():
    """Main entry point with command line support"""
    parser = argparse.ArgumentParser(description='Final Runner with Demo Mode Toggle')
    parser.add_argument('script', nargs='?', help='Script to run')
    parser.add_argument('--demo', action='store_true', help='Enable demo mode')
    parser.add_argument('--production', action='store_true', help='Enable production mode')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--all', action='store_true', help='Run all demo scripts')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.production:
        demo_mode = False
    elif args.demo:
        demo_mode = True
    else:
        demo_mode = True  # Default to demo mode
    
    runner = FinalRunner()
    runner.demo_mode = demo_mode
    DemoModeContext.set_demo_mode(demo_mode)
    
    # Execute based on arguments
    if args.interactive:
        interactive_mode()
    elif args.all:
        runner.run_all_demos()
    elif args.script:
        runner.run(args.script)
    else:
        # Default to interactive mode
        interactive_mode()


def demo_final_runner():
    """Demonstrate the final runner"""
    print(f"{Fore.CYAN}{'='*70}")
    print(f" FINAL RUNNER DEMONSTRATION")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    runner = FinalRunner()
    
    # Create a simple test script
    test_script = Path("test_script.py")
    test_script.write_text("""
import time

print("Starting script execution...")
print("Performing calculations...")
result = sum(range(100))
print(f"Result: {result}")

# Demo artifacts (will be skipped in production mode)
if __name__ == "__main__":
    print("\\n=== Demo Section ===")
    print("This is demo output that should be hidden in production mode")
    time.sleep(1)
    print("Press Enter to continue...")  # This will be suppressed
    # input()  # This will be skipped in production
    print("Demo complete!")
""")
    
    # Run in demo mode
    print(f"{Fore.YELLOW}1. Running in DEMO mode:{Style.RESET_ALL}")
    runner.run(test_script)
    
    time.sleep(2)
    
    # Toggle to production mode
    print(f"\n{Fore.YELLOW}2. Toggling to PRODUCTION mode:{Style.RESET_ALL}")
    runner.toggle_demo_mode()
    
    # Run in production mode
    print(f"\n{Fore.YELLOW}3. Running in PRODUCTION mode:{Style.RESET_ALL}")
    runner.run(test_script)
    
    # Clean up
    test_script.unlink()
    
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")
    print(f"\nNotice how production mode:")
    print(f"  • Skips demo artifacts")
    print(f"  • Removes input() calls")
    print(f"  • Shows only essential output")
    print(f"  • Runs faster without delays")


if __name__ == "__main__":
    # Check if running as script
    if len(sys.argv) > 1:
        main()
    else:
        demo_final_runner()