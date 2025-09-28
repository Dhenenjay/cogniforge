"""
Expert Trajectory Collector with Progress Spinner

This module handles expert trajectory collection for imitation learning,
with visual feedback through animated spinners and progress indicators.
"""

import time
import threading
import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path
from colorama import init, Fore, Style, Back
import itertools

# Initialize colorama
init(autoreset=True)

# Try to import prompt_header for nice displays
try:
    from prompt_header import PromptHeader
except ImportError:
    class PromptHeader:
        @staticmethod
        def print_header(msg, **kwargs):
            print(f"\n=== {msg} ===\n")
        @staticmethod
        def print_progress(msg, current, total, **kwargs):
            print(f"{msg}: {current}/{total}")


@dataclass
class Trajectory:
    """Represents a single expert trajectory"""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    duration: float = 0.0


class Spinner:
    """Animated spinner for visual feedback"""
    
    # Different spinner styles
    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'dots2': ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
        'dots3': ['⠋', '⠙', '⠚', '⠞', '⠖', '⠦', '⠴', '⠲', '⠳', '⠓'],
        'line': ['|', '/', '-', '\\'],
        'line2': ['⠂', '-', '–', '—', '–', '-'],
        'pipe': ['┤', '┘', '┴', '└', '├', '┌', '┬', '┐'],
        'star': ['✶', '✸', '✹', '✺', '✹', '✸'],
        'grow': ['▁', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃'],
        'balloon': [' ', '.', 'o', 'O', '@', '*', ' '],
        'noise': ['▓', '▒', '░'],
        'bounce': ['⠁', '⠂', '⠄', '⠂'],
        'boxBounce': ['▖', '▘', '▝', '▗'],
        'triangle': ['◢', '◣', '◤', '◥'],
        'arc': ['◜', '◠', '◝', '◞', '◡', '◟'],
        'circle': ['◡', '⊙', '◠'],
        'square': ['◰', '◳', '◲', '◱'],
        'squareCorners': ['◶', '◷', '◵', '◴'],
        'circleQuarters': ['◴', '◷', '◶', '◵'],
        'circleHalves': ['◐', '◓', '◑', '◒'],
        'arrow': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'arrow2': ['⬆️ ', '↗️ ', '➡️ ', '↘️ ', '⬇️ ', '↙️ ', '⬅️ ', '↖️ '],
        'bouncingBar': ['[    ]', '[=   ]', '[==  ]', '[=== ]', '[ ===]', '[  ==]', '[   =]', '[    ]'],
        'bouncingBall': ['( ●    )', '(  ●   )', '(   ●  )', '(    ● )', '(     ●)', '(    ● )', '(   ●  )', '(  ●   )', '( ●    )', '(●     )'],
        'pong': ['▐⠂       ▌', '▐⠈       ▌', '▐ ⠂      ▌', '▐ ⠠      ▌', '▐  ⡀     ▌', '▐  ⠠     ▌', '▐   ⠂    ▌', '▐   ⠈    ▌', '▐    ⠂   ▌', '▐    ⠠   ▌', '▐     ⡀  ▌', '▐     ⠠  ▌', '▐      ⠂ ▌', '▐      ⠈ ▌', '▐       ⠂▌', '▐       ⠠▌', '▐       ⡀▌', '▐      ⠠ ▌', '▐      ⠂ ▌', '▐     ⠈  ▌', '▐     ⠂  ▌', '▐    ⠠   ▌', '▐    ⡀   ▌', '▐   ⠠    ▌', '▐   ⠂    ▌', '▐  ⠈     ▌', '▐  ⠂     ▌', '▐ ⠠      ▌', '▐ ⡀      ▌', '▐⠠       ▌'],
        'robot': ['🤖', '🤖', '🤖', '🤖'],
        'earth': ['🌍', '🌎', '🌏'],
        'moon': ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'],
        'clock': ['🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛']
    }
    
    def __init__(self, message: str = "Processing", 
                 style: str = "dots",
                 color: str = Fore.CYAN,
                 success_mark: str = "✅",
                 fail_mark: str = "❌"):
        """
        Initialize spinner
        
        Args:
            message: Message to display
            style: Spinner style from SPINNERS
            color: Color for spinner
            success_mark: Symbol for success
            fail_mark: Symbol for failure
        """
        self.message = message
        self.frames = self.SPINNERS.get(style, self.SPINNERS['dots'])
        self.color = color
        self.success_mark = success_mark
        self.fail_mark = fail_mark
        self._stop_event = threading.Event()
        self._thread = None
        self._status = None
        
    def _spin(self):
        """Internal spinning animation"""
        for frame in itertools.cycle(self.frames):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f'\r{self.color}{frame} {self.message}{Style.RESET_ALL}')
            sys.stdout.flush()
            time.sleep(0.1)
            
    def start(self):
        """Start spinner animation"""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()
        return self
        
    def stop(self, success: bool = True, message: Optional[str] = None):
        """
        Stop spinner with status
        
        Args:
            success: Whether operation succeeded
            message: Optional completion message
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            
        # Clear line and show result
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        
        mark = self.success_mark if success else self.fail_mark
        color = Fore.GREEN if success else Fore.RED
        final_message = message or self.message
        
        print(f'{mark} {color}{final_message}{Style.RESET_ALL}')
        sys.stdout.flush()
        
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        success = exc_type is None
        self.stop(success=success)


class ProgressSpinner:
    """Spinner with progress counter"""
    
    def __init__(self, total: int, 
                 message: str = "Processing",
                 style: str = "dots",
                 show_percentage: bool = True,
                 show_count: bool = True):
        """
        Initialize progress spinner
        
        Args:
            total: Total items to process
            message: Base message
            style: Spinner style
            show_percentage: Show percentage complete
            show_count: Show count (current/total)
        """
        self.total = total
        self.current = 0
        self.message = message
        self.frames = Spinner.SPINNERS.get(style, Spinner.SPINNERS['dots'])
        self.show_percentage = show_percentage
        self.show_count = show_count
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        
    def _spin(self):
        """Internal spinning with progress"""
        for frame in itertools.cycle(self.frames):
            if self._stop_event.is_set():
                break
                
            with self._lock:
                progress_parts = [f'{Fore.CYAN}{frame}', self.message]
                
                if self.show_count:
                    progress_parts.append(f'[{self.current}/{self.total}]')
                    
                if self.show_percentage and self.total > 0:
                    percentage = (self.current / self.total) * 100
                    progress_parts.append(f'{percentage:.1f}%')
                    
                progress_text = ' '.join(progress_parts)
                
            sys.stdout.write(f'\r{progress_text}{Style.RESET_ALL}')
            sys.stdout.flush()
            time.sleep(0.1)
            
    def update(self, increment: int = 1):
        """Update progress counter"""
        with self._lock:
            self.current = min(self.current + increment, self.total)
            
    def start(self):
        """Start progress spinner"""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()
        return self
        
    def stop(self, success: bool = True):
        """Stop progress spinner"""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            
        # Clear line
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        
        mark = "✅" if success else "❌"
        color = Fore.GREEN if success else Fore.RED
        
        status = "Complete" if success else "Failed"
        print(f'{mark} {color}{self.message} - {status} [{self.current}/{self.total}]{Style.RESET_ALL}')


class ExpertCollector:
    """Collects expert trajectories with visual feedback"""
    
    def __init__(self, 
                 env_name: str = "Manipulation",
                 save_dir: str = "expert_trajectories"):
        """
        Initialize expert collector
        
        Args:
            env_name: Environment name
            save_dir: Directory to save trajectories
        """
        self.env_name = env_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories: List[Trajectory] = []
        
    def collect_trajectories(self, 
                            num_trajectories: int,
                            expert_policy: Optional[Callable] = None,
                            max_steps: int = 100,
                            render: bool = False) -> List[Trajectory]:
        """
        Collect expert trajectories with spinner animation
        
        Args:
            num_trajectories: Number of trajectories to collect
            expert_policy: Expert policy function
            max_steps: Maximum steps per trajectory
            render: Whether to render environment
            
        Returns:
            List of collected trajectories
        """
        # Show header
        PromptHeader.print_header(
            "Expert Trajectory Collection",
            style='task',
            subtitle=f"Collecting {num_trajectories} trajectories",
            metadata={
                'Environment': self.env_name,
                'Max Steps': max_steps,
                'Policy': 'Expert' if expert_policy else 'Random'
            }
        )
        
        # Main collection spinner
        print(f"\n{Fore.YELLOW}Initializing collection environment...{Style.RESET_ALL}")
        time.sleep(0.5)
        
        # Show collection spinner
        with Spinner("Collecting expert trajectories", style="dots", color=Fore.CYAN):
            time.sleep(1.5)  # Simulate initialization
            
        print()  # New line after spinner
        
        # Progress spinner for trajectory collection
        progress = ProgressSpinner(
            num_trajectories, 
            "Collecting trajectories",
            style="dots2",
            show_percentage=True
        )
        progress.start()
        
        collected_trajectories = []
        
        try:
            for i in range(num_trajectories):
                # Collect single trajectory
                trajectory = self._collect_single_trajectory(
                    expert_policy,
                    max_steps,
                    render,
                    trajectory_id=i
                )
                
                collected_trajectories.append(trajectory)
                self.trajectories.append(trajectory)
                
                # Update progress
                progress.update(1)
                
                # Small delay between trajectories
                time.sleep(0.1)
                
            progress.stop(success=True)
            
        except Exception as e:
            progress.stop(success=False)
            print(f"{Fore.RED}Error during collection: {e}{Style.RESET_ALL}")
            raise
            
        # Show collection summary
        self._show_collection_summary(collected_trajectories)
        
        # Save trajectories with spinner
        self._save_trajectories_with_spinner(collected_trajectories)
        
        return collected_trajectories
        
    def _collect_single_trajectory(self,
                                  expert_policy: Optional[Callable],
                                  max_steps: int,
                                  render: bool,
                                  trajectory_id: int) -> Trajectory:
        """
        Collect a single trajectory
        
        Args:
            expert_policy: Policy to use
            max_steps: Maximum steps
            render: Whether to render
            trajectory_id: Trajectory identifier
            
        Returns:
            Collected trajectory
        """
        states = []
        actions = []
        rewards = []
        
        # Simulate trajectory collection
        start_time = time.time()
        
        # Initial state
        state = np.random.randn(10)  # Example state
        
        for step in range(max_steps):
            states.append(state.copy())
            
            # Get action from policy
            if expert_policy:
                action = expert_policy(state)
            else:
                # Random action for demo (typically 4-dim for robot control)
                action = np.random.randn(4)
                
            # Ensure action is a numpy array
            action = np.asarray(action).flatten()
            
            actions.append(action)
            
            # Simulate environment step
            state = state + 0.1 * action + 0.01 * np.random.randn(10)
            reward = -np.sum(state**2) * 0.01  # Example reward
            rewards.append(reward)
            
            # Check termination
            if np.random.random() < 0.01:  # 1% chance to terminate
                break
                
        duration = time.time() - start_time
        
        return Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            metadata={
                'trajectory_id': trajectory_id,
                'num_steps': len(states),
                'total_reward': sum(rewards),
                'env_name': self.env_name
            },
            success=True,
            duration=duration
        )
        
    def _show_collection_summary(self, trajectories: List[Trajectory]):
        """Show summary of collected trajectories"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Collection Summary{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        total_steps = sum(len(t.states) for t in trajectories)
        total_reward = sum(sum(t.rewards) for t in trajectories)
        avg_length = total_steps / len(trajectories) if trajectories else 0
        avg_reward = total_reward / len(trajectories) if trajectories else 0
        
        print(f"📊 Trajectories Collected: {len(trajectories)}")
        print(f"📈 Total Steps: {total_steps}")
        print(f"📏 Average Length: {avg_length:.1f} steps")
        print(f"🎯 Average Reward: {avg_reward:.2f}")
        print(f"✅ Success Rate: {sum(t.success for t in trajectories)/len(trajectories)*100:.1f}%")
        
    def _save_trajectories_with_spinner(self, trajectories: List[Trajectory]):
        """Save trajectories with progress spinner"""
        print(f"\n{Fore.YELLOW}Saving trajectories...{Style.RESET_ALL}\n")
        
        # Saving spinner for different formats
        save_tasks = [
            ("Saving as pickle", "pkl"),
            ("Saving as JSON", "json"),
            ("Creating summary", "txt")
        ]
        
        for task_name, extension in save_tasks:
            with Spinner(task_name, style="dots3", color=Fore.BLUE) as spinner:
                time.sleep(0.5)  # Simulate save time
                
                if extension == "pkl":
                    self._save_pickle(trajectories)
                elif extension == "json":
                    self._save_json(trajectories)
                elif extension == "txt":
                    self._save_summary(trajectories)
                    
                spinner.stop(success=True, message=f"{task_name} - Complete")
                
        print(f"\n{Fore.GREEN}✅ All trajectories saved to {self.save_dir}{Style.RESET_ALL}")
        
    def _save_pickle(self, trajectories: List[Trajectory]):
        """Save trajectories as pickle"""
        filepath = self.save_dir / f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(trajectories, f)
            
    def _save_json(self, trajectories: List[Trajectory]):
        """Save trajectories as JSON"""
        filepath = self.save_dir / f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to JSON-serializable format
        data = []
        for traj in trajectories:
            data.append({
                'states': [s.tolist() for s in traj.states],
                'actions': [a.tolist() for a in traj.actions],
                'rewards': traj.rewards,
                'metadata': traj.metadata,
                'timestamp': traj.timestamp,
                'success': traj.success,
                'duration': traj.duration
            })
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _save_summary(self, trajectories: List[Trajectory]):
        """Save summary of trajectories"""
        filepath = self.save_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filepath, 'w') as f:
            f.write("Expert Trajectory Collection Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Number of Trajectories: {len(trajectories)}\n")
            f.write(f"Collection Time: {datetime.now().isoformat()}\n\n")
            
            for i, traj in enumerate(trajectories):
                f.write(f"Trajectory {i+1}:\n")
                f.write(f"  - Steps: {len(traj.states)}\n")
                f.write(f"  - Total Reward: {sum(traj.rewards):.2f}\n")
                f.write(f"  - Success: {traj.success}\n")
                f.write(f"  - Duration: {traj.duration:.2f}s\n\n")


class MultiTaskCollector:
    """Collect trajectories for multiple tasks with visual feedback"""
    
    def __init__(self):
        self.collectors = {}
        
    def collect_all_tasks(self, tasks: List[str], 
                         trajectories_per_task: int = 10):
        """
        Collect trajectories for multiple tasks
        
        Args:
            tasks: List of task names
            trajectories_per_task: Trajectories per task
        """
        print(f"{Fore.CYAN}{'='*80}")
        print(f" MULTI-TASK EXPERT COLLECTION")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        print(f"Tasks to collect: {', '.join(tasks)}")
        print(f"Trajectories per task: {trajectories_per_task}\n")
        
        # Overall progress
        total_trajectories = len(tasks) * trajectories_per_task
        
        with ProgressSpinner(
            len(tasks),
            "Collecting all tasks",
            style="arrow",
            show_percentage=True
        ) as overall_progress:
            
            for task in tasks:
                print(f"\n{Fore.YELLOW}Task: {task}{Style.RESET_ALL}")
                
                # Create collector for this task
                collector = ExpertCollector(env_name=task)
                self.collectors[task] = collector
                
                # Collect trajectories
                collector.collect_trajectories(
                    trajectories_per_task,
                    expert_policy=None,  # Would use task-specific policy
                    max_steps=50
                )
                
                overall_progress.update(1)
                
        print(f"\n{Fore.GREEN}✅ All tasks complete!{Style.RESET_ALL}")


def demo_expert_collection():
    """Demonstrate expert trajectory collection with spinners"""
    
    print(f"{Fore.CYAN}{'='*80}")
    print(f" EXPERT TRAJECTORY COLLECTION DEMO")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    # Test different spinner styles
    print(f"{Fore.YELLOW}Testing spinner styles:{Style.RESET_ALL}\n")
    
    spinner_styles = ['dots', 'dots2', 'line', 'arc', 'bounce', 'arrow']
    
    for style in spinner_styles:
        with Spinner(f"Testing {style} style", style=style, color=Fore.CYAN):
            time.sleep(1)
            
    print()
    
    # Collect expert trajectories
    print(f"{Fore.YELLOW}Starting expert trajectory collection...{Style.RESET_ALL}\n")
    
    collector = ExpertCollector(
        env_name="RoboticManipulation",
        save_dir="expert_data"
    )
    
    # Define a simple expert policy
    def expert_policy(state):
        """Simple expert policy for demonstration"""
        return -0.1 * state[:4] + 0.01 * np.random.randn(4)
    
    # Collect trajectories with visual feedback
    trajectories = collector.collect_trajectories(
        num_trajectories=20,
        expert_policy=expert_policy,
        max_steps=100,
        render=False
    )
    
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Test multi-task collection
    print(f"\n{Fore.YELLOW}Testing multi-task collection...{Style.RESET_ALL}\n")
    
    multi_collector = MultiTaskCollector()
    tasks = ["Pick", "Place", "Push", "Stack"]
    
    # Show task spinner
    with Spinner("Preparing multi-task collection", style="robot", color=Fore.MAGENTA):
        time.sleep(2)
        
    print()
    
    # Collect for all tasks (reduced for demo)
    for task in tasks:
        print(f"\n{Fore.CYAN}Collecting for task: {task}{Style.RESET_ALL}")
        with ProgressSpinner(5, f"Collecting {task} trajectories", style="circle"):
            time.sleep(1)  # Simulate collection
            
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")


if __name__ == "__main__":
    demo_expert_collection()