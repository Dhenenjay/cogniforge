"""
Recovery System with Failsafe Execution

This module provides a recovery mechanism with a prominent recovery button
that appears when failures occur, allowing users to skip to the final script
with last-known good parameters.
"""

import os
import sys
import json
import time
import pickle
import traceback
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from colorama import init, Fore, Style, Back
import numpy as np

# Initialize colorama
init(autoreset=True)

# Ensure UTF-8 output on Windows consoles that default to cp1252
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

@dataclass
class LastKnownGood:
    """Last known good configuration"""
    timestamp: str
    script_path: str
    parameters: Dict[str, Any]
    output: Any
    execution_time: float
    success: bool = True
    
    def save(self, filepath: Path):
        """Save configuration to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: Path) -> 'LastKnownGood':
        """Load configuration from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class RecoveryButton(tk.Tk):
    """Large recovery button GUI"""
    
    def __init__(self, on_recovery: Callable, error_info: str = ""):
        super().__init__()
        
        self.on_recovery = on_recovery
        self.error_info = error_info
        self.recovery_triggered = False
        
        # Window setup
        self.title("⚠️ SYSTEM RECOVERY")
        self.geometry("600x400")
        self.configure(bg='#2c2c2c')
        
        # Center window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
        # Make window stay on top
        self.attributes('-topmost', True)
        
        # Create UI
        self.create_widgets()
        
        # Flash animation
        self.flash_button()
    
    def create_widgets(self):
        """Create recovery UI"""
        # Error message
        error_frame = tk.Frame(self, bg='#2c2c2c')
        error_frame.pack(pady=20, padx=20, fill=tk.X)
        
        error_label = tk.Label(
            error_frame,
            text="❌ EXECUTION FAILED",
            font=("Arial", 20, "bold"),
            fg='#ff4444',
            bg='#2c2c2c'
        )
        error_label.pack()
        
        # Error details
        if self.error_info:
            details_text = tk.Text(
                error_frame,
                height=6,
                width=60,
                bg='#1a1a1a',
                fg='#ffffff',
                font=("Consolas", 10),
                wrap=tk.WORD
            )
            details_text.pack(pady=10)
            details_text.insert(1.0, self.error_info[:500])  # First 500 chars
            details_text.config(state=tk.DISABLED)
        
        # Recovery button - LARGE AND PROMINENT
        self.recovery_btn = tk.Button(
            self,
            text="🔧 RECOVER",
            command=self.trigger_recovery,
            font=("Arial", 24, "bold"),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            activeforeground='white',
            relief=tk.RAISED,
            bd=5,
            width=20,
            height=2,
            cursor='hand2'
        )
        self.recovery_btn.pack(pady=30)
        
        # Additional options
        options_frame = tk.Frame(self, bg='#2c2c2c')
        options_frame.pack()
        
        # Skip button
        skip_btn = tk.Button(
            options_frame,
            text="⏭️ Skip to Final",
            command=self.skip_to_final,
            font=("Arial", 12),
            bg='#2196F3',
            fg='white',
            width=15,
            cursor='hand2'
        )
        skip_btn.pack(side=tk.LEFT, padx=5)
        
        # Retry button
        retry_btn = tk.Button(
            options_frame,
            text="🔄 Retry",
            command=self.retry_current,
            font=("Arial", 12),
            bg='#FFC107',
            fg='black',
            width=15,
            cursor='hand2'
        )
        retry_btn.pack(side=tk.LEFT, padx=5)
        
        # Abort button
        abort_btn = tk.Button(
            options_frame,
            text="❌ Abort",
            command=self.abort_execution,
            font=("Arial", 12),
            bg='#f44336',
            fg='white',
            width=15,
            cursor='hand2'
        )
        abort_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            self,
            text="Click RECOVER to use last known good configuration",
            font=("Arial", 11),
            fg='#aaaaaa',
            bg='#2c2c2c'
        )
        self.status_label.pack(pady=10)
        
        # Bind keyboard shortcuts
        self.bind('<Return>', lambda e: self.trigger_recovery())
        self.bind('<Escape>', lambda e: self.abort_execution())
        self.bind('<F5>', lambda e: self.retry_current())
    
    def flash_button(self):
        """Flash the recovery button to draw attention"""
        colors = ['#4CAF50', '#81C784', '#4CAF50']
        
        def flash(index=0):
            if index < len(colors) * 2:
                self.recovery_btn.config(bg=colors[index % len(colors)])
                self.after(200, lambda: flash(index + 1))
        
        flash()
    
    def trigger_recovery(self):
        """Trigger recovery process"""
        self.recovery_triggered = True
        self.status_label.config(text="🔄 Initiating recovery...", fg='#4CAF50')
        self.recovery_btn.config(state=tk.DISABLED, text="RECOVERING...")
        
        # Call recovery callback
        self.after(500, self._execute_recovery)
    
    def _execute_recovery(self):
        """Execute recovery callback"""
        try:
            self.on_recovery('recover')
            self.status_label.config(text="✅ Recovery initiated successfully", fg='#4CAF50')
            self.after(1000, self.destroy)
        except Exception as e:
            self.status_label.config(text=f"❌ Recovery failed: {str(e)}", fg='#ff4444')
            self.recovery_btn.config(state=tk.NORMAL, text="🔧 RECOVER")
    
    def skip_to_final(self):
        """Skip to final script"""
        self.on_recovery('skip')
        self.destroy()
    
    def retry_current(self):
        """Retry current script"""
        self.on_recovery('retry')
        self.destroy()
    
    def abort_execution(self):
        """Abort all execution"""
        self.on_recovery('abort')
        self.destroy()


class RecoverySystem:
    """Main recovery system with failsafe mechanisms"""
    
    def __init__(self, checkpoint_dir: str = "recovery_checkpoints"):
        """
        Initialize recovery system
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_known_good_file = self.checkpoint_dir / "last_known_good.json"
        self.recovery_log_file = self.checkpoint_dir / "recovery_log.json"
        self.final_script_file = self.checkpoint_dir / "final_script_config.json"
        
        self.last_known_good: Optional[LastKnownGood] = None
        self.recovery_count = 0
        self.max_recovery_attempts = 3
        
        # Load existing last known good if available
        self._load_last_known_good()
        
    def _load_last_known_good(self):
        """Load last known good configuration"""
        if self.last_known_good_file.exists():
            try:
                self.last_known_good = LastKnownGood.load(self.last_known_good_file)
                print(f"{Fore.GREEN}✅ Loaded last known good configuration from {self.last_known_good.timestamp}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Could not load last known good: {e}{Style.RESET_ALL}")
    
    def save_checkpoint(self, script_path: str, parameters: Dict[str, Any], 
                       output: Any = None, execution_time: float = 0.0):
        """
        Save a successful checkpoint
        
        Args:
            script_path: Path to the script
            parameters: Parameters used
            output: Output from the script
            execution_time: Time taken to execute
        """
        checkpoint = LastKnownGood(
            timestamp=datetime.now().isoformat(),
            script_path=str(script_path),
            parameters=parameters,
            output=output if output else {},
            execution_time=execution_time,
            success=True
        )
        
        # Save as last known good
        checkpoint.save(self.last_known_good_file)
        self.last_known_good = checkpoint
        
        # Also save timestamped backup
        backup_file = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        checkpoint.save(backup_file)
        
        print(f"{Fore.GREEN}💾 Checkpoint saved{Style.RESET_ALL}")
    
    def show_recovery_button(self, error_info: str = "") -> str:
        """
        Show the recovery button and wait for user action
        
        Args:
            error_info: Error information to display
            
        Returns:
            User's choice: 'recover', 'skip', 'retry', or 'abort'
        """
        user_choice = {'action': None}
        
        def on_recovery(action: str):
            user_choice['action'] = action
        
        # Create and show recovery button
        recovery_gui = RecoveryButton(on_recovery, error_info)
        
        # Play alert sound (Windows)
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except:
            pass
        
        recovery_gui.mainloop()
        
        return user_choice['action'] or 'abort'
    
    def execute_with_recovery(self, script_path: Union[str, Path], 
                             parameters: Optional[Dict[str, Any]] = None,
                             final_script_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Execute a script with recovery mechanism
        
        Args:
            script_path: Path to the script to execute
            parameters: Parameters for the script
            final_script_path: Path to final script to skip to on recovery
            
        Returns:
            Execution result
        """
        script_path = Path(script_path)
        parameters = parameters or {}
        
        # Print execution header
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🛡️ PROTECTED EXECUTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        print(f"📄 Script: {script_path.name}")
        print(f"🔧 Recovery: Enabled")
        if self.last_known_good:
            print(f"💾 Last Good: {self.last_known_good.timestamp}")
        print()
        
        try:
            # Attempt to execute the script
            result = self._execute_script(script_path, parameters)
            
            if result['success']:
                # Save as new checkpoint
                self.save_checkpoint(
                    str(script_path),
                    parameters,
                    result.get('output'),
                    result.get('execution_time', 0)
                )
                print(f"{Fore.GREEN}✅ Execution successful{Style.RESET_ALL}")
                return result
            else:
                raise Exception(result.get('error', 'Unknown error'))
                
        except Exception as e:
            # Execution failed
            error_info = f"Script: {script_path.name}\nError: {str(e)}\n{traceback.format_exc()}"
            
            print(f"\n{Fore.RED}{'='*70}{Style.RESET_ALL}")
            print(f"{Fore.RED}❌ EXECUTION FAILED{Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*70}{Style.RESET_ALL}")
            print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}\n")
            
            # Show recovery button
            user_choice = self.show_recovery_button(error_info)
            
            if user_choice == 'recover':
                return self._recover_with_last_known_good()
            elif user_choice == 'skip':
                return self._skip_to_final(final_script_path, parameters)
            elif user_choice == 'retry':
                return self.execute_with_recovery(script_path, parameters, final_script_path)
            else:  # abort
                print(f"{Fore.RED}❌ Execution aborted by user{Style.RESET_ALL}")
                return {'success': False, 'error': 'User aborted', 'aborted': True}
    
    def _execute_script(self, script_path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a script with parameters
        
        Args:
            script_path: Path to script
            parameters: Script parameters
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        # Build command
        cmd = [sys.executable, str(script_path)]
        
        # Add parameters as JSON argument
        if parameters:
            cmd.extend(['--params', json.dumps(parameters)])
        
        try:
            # Execute script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout,
                    'execution_time': execution_time
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Script failed',
                    'output': result.stdout,
                    'execution_time': execution_time
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Script timeout (60s)',
                'execution_time': 60.0
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _recover_with_last_known_good(self) -> Dict[str, Any]:
        """Recover using last known good configuration"""
        print(f"\n{Fore.YELLOW}🔧 INITIATING RECOVERY{Style.RESET_ALL}")
        
        if not self.last_known_good:
            print(f"{Fore.RED}❌ No last known good configuration available{Style.RESET_ALL}")
            return {'success': False, 'error': 'No recovery checkpoint available'}
        
        print(f"📍 Using checkpoint from: {self.last_known_good.timestamp}")
        print(f"📄 Script: {self.last_known_good.script_path}")
        print(f"⚙️ Parameters: {json.dumps(self.last_known_good.parameters, indent=2)}")
        
        # Re-run with last known good
        script_path = Path(self.last_known_good.script_path)
        
        if not script_path.exists():
            print(f"{Fore.RED}❌ Recovery script not found: {script_path}{Style.RESET_ALL}")
            return {'success': False, 'error': 'Recovery script not found'}
        
        print(f"\n{Fore.GREEN}▶️ Executing recovery...{Style.RESET_ALL}")
        
        result = self._execute_script(script_path, self.last_known_good.parameters)
        
        if result['success']:
            print(f"{Fore.GREEN}✅ Recovery successful!{Style.RESET_ALL}")
            self._log_recovery('success', script_path)
        else:
            print(f"{Fore.RED}❌ Recovery failed: {result.get('error')}{Style.RESET_ALL}")
            self._log_recovery('failed', script_path, result.get('error'))
        
        return result
    
    def _skip_to_final(self, final_script_path: Optional[Path], 
                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Skip to final script with safe parameters"""
        print(f"\n{Fore.YELLOW}⏭️ SKIPPING TO FINAL SCRIPT{Style.RESET_ALL}")
        
        if not final_script_path:
            # Use default final script
            final_script_path = Path("src/final_runner.py")
        
        final_script_path = Path(final_script_path)
        
        if not final_script_path.exists():
            print(f"{Fore.RED}❌ Final script not found: {final_script_path}{Style.RESET_ALL}")
            return {'success': False, 'error': 'Final script not found'}
        
        # Use safe default parameters
        safe_params = self._get_safe_parameters(parameters)
        
        print(f"📄 Final Script: {final_script_path.name}")
        print(f"⚙️ Safe Parameters: {json.dumps(safe_params, indent=2)}")
        print(f"\n{Fore.GREEN}▶️ Executing final script...{Style.RESET_ALL}")
        
        result = self._execute_script(final_script_path, safe_params)
        
        if result['success']:
            print(f"{Fore.GREEN}✅ Final script executed successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Final script failed: {result.get('error')}{Style.RESET_ALL}")
        
        return result
    
    def _get_safe_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get safe default parameters"""
        # Start with provided parameters
        safe_params = parameters.copy()
        
        # Override with safe defaults
        safe_defaults = {
            'demo_mode': False,  # Disable demo mode
            'max_iterations': 10,  # Limit iterations
            'timeout': 30,  # Set timeout
            'verbose': False,  # Reduce output
            'fail_fast': True,  # Fail fast on errors
            'skip_validation': False,  # Don't skip validation
            'use_defaults': True  # Use default values
        }
        
        safe_params.update(safe_defaults)
        
        # If we have last known good params, merge them
        if self.last_known_good:
            for key, value in self.last_known_good.parameters.items():
                if key not in safe_params:
                    safe_params[key] = value
        
        return safe_params
    
    def _log_recovery(self, status: str, script_path: Path, error: Optional[str] = None):
        """Log recovery attempt"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'script': str(script_path),
            'error': error,
            'recovery_count': self.recovery_count
        }
        
        # Load existing log
        logs = []
        if self.recovery_log_file.exists():
            with open(self.recovery_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        
        # Append new entry
        logs.append(log_entry)
        
        # Save log
        with open(self.recovery_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
        
        self.recovery_count += 1


def demo_recovery_system():
    """Demonstrate the recovery system"""
    print(f"{Fore.CYAN}{'='*70}")
    print(f" RECOVERY SYSTEM DEMONSTRATION")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # Create recovery system
    recovery = RecoverySystem()
    
    # Create test scripts
    good_script = Path("good_script.py")
    good_script.write_text("""
import sys
import json

print("Executing good script...")
print("Result: SUCCESS")
sys.exit(0)
""", encoding='utf-8')
    
    bad_script = Path("bad_script.py")
    bad_script.write_text("""
import sys

print("Executing bad script...")
print("This will fail...")
raise Exception("Simulated failure!")
sys.exit(1)
""", encoding='utf-8')
    
    final_script = Path("final_script.py")
    final_script.write_text("""
import sys
import json

print("=== FINAL SCRIPT ===")
print("Running with safe parameters")
print("Cleaning up...")
print("✅ Final script complete!")
sys.exit(0)
""", encoding='utf-8')
    
    # Test 1: Execute good script (will become last known good)
    print(f"{Fore.YELLOW}1. Running good script to establish checkpoint...{Style.RESET_ALL}")
    result1 = recovery.execute_with_recovery(
        good_script,
        {'param1': 'value1', 'param2': 42}
    )
    time.sleep(2)
    
    # Test 2: Execute bad script (will trigger recovery)
    print(f"\n{Fore.YELLOW}2. Running bad script (will fail and show recovery)...{Style.RESET_ALL}")
    result2 = recovery.execute_with_recovery(
        bad_script,
        {'param1': 'bad_value'},
        final_script_path=final_script
    )
    
    # Clean up
    good_script.unlink()
    bad_script.unlink()
    final_script.unlink()
    
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")
    print(f"\nRecovery features demonstrated:")
    print(f"  • Checkpoint saving on success")
    print(f"  • Large recovery button on failure")
    print(f"  • Recovery with last known good")
    print(f"  • Skip to final script option")
    print(f"  • Safe parameter fallbacks")


if __name__ == "__main__":
    demo_recovery_system()