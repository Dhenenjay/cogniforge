"""
Motion Controller with Undo Support

Allows switching between BC-only and optimized motion for A/B testing.
Supports undo/redo functionality and motion history tracking.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import pickle
from datetime import datetime
import logging
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionType(Enum):
    """Types of motion control"""
    BC_ONLY = "bc_only"           # Original behavioral cloning
    OPTIMIZED = "optimized"        # Optimized trajectory
    HYBRID = "hybrid"              # Mix of BC and optimization
    MANUAL = "manual"              # Manual control
    

class MotionSource(Enum):
    """Source of motion data"""
    EXPERT_DEMO = "expert_demonstration"
    BC_MODEL = "bc_model"
    OPTIMIZER = "optimizer"
    USER_INPUT = "user_input"
    

@dataclass
class MotionState:
    """Represents a motion state that can be saved/restored"""
    motion_type: MotionType
    waypoints: List[Dict[str, Any]]
    timestamp: float
    source: MotionSource
    metrics: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    
    def summary(self) -> str:
        """Get summary of motion state"""
        return (f"Type: {self.motion_type.value}, "
                f"Waypoints: {len(self.waypoints)}, "
                f"Source: {self.source.value}")
                

class MotionHistory:
    """Manages history of motion states for undo/redo"""
    
    def __init__(self, max_history: int = 50):
        """
        Initialize motion history
        
        Args:
            max_history: Maximum number of states to keep
        """
        self.max_history = max_history
        self.past_states: deque = deque(maxlen=max_history)
        self.future_states: deque = deque(maxlen=max_history)
        self.current_state: Optional[MotionState] = None
        
    def push(self, state: MotionState):
        """
        Push new state to history
        
        Args:
            state: Motion state to save
        """
        if self.current_state:
            self.past_states.append(self.current_state)
            
        self.current_state = state
        self.future_states.clear()  # Clear redo stack on new action
        
    def undo(self) -> Optional[MotionState]:
        """
        Undo to previous state
        
        Returns:
            Previous state or None
        """
        if not self.past_states:
            return None
            
        if self.current_state:
            self.future_states.appendleft(self.current_state)
            
        self.current_state = self.past_states.pop()
        return self.current_state
        
    def redo(self) -> Optional[MotionState]:
        """
        Redo to next state
        
        Returns:
            Next state or None
        """
        if not self.future_states:
            return None
            
        if self.current_state:
            self.past_states.append(self.current_state)
            
        self.current_state = self.future_states.popleft()
        return self.current_state
        
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return len(self.past_states) > 0
        
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return len(self.future_states) > 0
        
    def clear(self):
        """Clear all history"""
        self.past_states.clear()
        self.future_states.clear()
        self.current_state = None
        
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of history state"""
        return {
            'past_count': len(self.past_states),
            'future_count': len(self.future_states),
            'has_current': self.current_state is not None,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo()
        }


class MotionControllerWithUndo:
    """Motion controller with undo/redo and A/B testing support"""
    
    def __init__(self, robot_id: int, scene_objects: Dict[str, int]):
        """
        Initialize motion controller
        
        Args:
            robot_id: Robot ID in simulation
            scene_objects: Dictionary of scene objects
        """
        self.robot_id = robot_id
        self.scene_objects = scene_objects
        self.history = MotionHistory()
        
        # Motion sources
        self.bc_model = None
        self.optimizer = None
        
        # A/B testing
        self.ab_mode = False
        self.ab_results = []
        
        # Current execution
        self.current_waypoints = []
        self.is_executing = False
        
    def set_bc_model(self, bc_model: Any):
        """Set BC model for motion generation"""
        self.bc_model = bc_model
        logger.info("BC model loaded")
        
    def set_optimizer(self, optimizer: Any):
        """Set optimizer for trajectory optimization"""
        self.optimizer = optimizer
        logger.info("Optimizer loaded")
        
    def generate_bc_motion(self, task_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate motion using BC model
        
        Args:
            task_params: Task parameters
            
        Returns:
            BC waypoints
        """
        if not self.bc_model:
            logger.warning("No BC model loaded, using dummy waypoints")
            return self._generate_dummy_bc_waypoints(task_params)
            
        # Generate waypoints from BC model
        # This would call actual BC inference
        waypoints = self.bc_model.generate_waypoints(task_params)
        
        return waypoints
        
    def generate_optimized_motion(self, bc_waypoints: List[Dict[str, Any]], 
                                 optimization_level: float = 0.5) -> List[Dict[str, Any]]:
        """
        Generate optimized motion from BC waypoints
        
        Args:
            bc_waypoints: Original BC waypoints
            optimization_level: Level of optimization (0=none, 1=maximum)
            
        Returns:
            Optimized waypoints
        """
        if not self.optimizer:
            logger.warning("No optimizer loaded, returning original waypoints")
            return bc_waypoints
            
        # Apply optimization based on level
        if optimization_level <= 0:
            return bc_waypoints
        elif optimization_level >= 1:
            # Full optimization
            return self.optimizer.optimize(bc_waypoints)
        else:
            # Partial optimization (blend)
            optimized = self.optimizer.optimize(bc_waypoints)
            return self._blend_waypoints(bc_waypoints, optimized, optimization_level)
            
    def _blend_waypoints(self, waypoints1: List[Dict[str, Any]], 
                        waypoints2: List[Dict[str, Any]], 
                        alpha: float) -> List[Dict[str, Any]]:
        """Blend two waypoint trajectories"""
        blended = []
        
        # Simple linear interpolation
        max_len = max(len(waypoints1), len(waypoints2))
        
        for i in range(max_len):
            if i < len(waypoints1) and i < len(waypoints2):
                # Blend positions
                pos1 = np.array(waypoints1[i]['position'])
                pos2 = np.array(waypoints2[i]['position'])
                blended_pos = (1 - alpha) * pos1 + alpha * pos2
                
                blended.append({
                    'position': blended_pos.tolist(),
                    'gripper': waypoints1[i].get('gripper', 1.0),
                    'description': f"Blended ({alpha:.1f})"
                })
            elif i < len(waypoints1):
                blended.append(waypoints1[i])
            else:
                blended.append(waypoints2[i])
                
        return blended
        
    def execute_bc_only(self, task_params: Dict[str, Any]) -> MotionState:
        """
        Execute BC-only motion
        
        Args:
            task_params: Task parameters
            
        Returns:
            Motion state
        """
        logger.info(f"{Fore.CYAN}Executing BC-only motion{Style.RESET_ALL}")
        
        # Generate BC waypoints
        bc_waypoints = self.generate_bc_motion(task_params)
        
        # Create motion state
        state = MotionState(
            motion_type=MotionType.BC_ONLY,
            waypoints=bc_waypoints,
            timestamp=time.time(),
            source=MotionSource.BC_MODEL,
            description="BC-only motion from model"
        )
        
        # Save to history
        self.history.push(state)
        
        # Execute
        self._execute_waypoints(bc_waypoints)
        
        # Compute metrics
        state.metrics = self._compute_metrics(bc_waypoints)
        
        return state
        
    def execute_optimized(self, task_params: Dict[str, Any], 
                         optimization_level: float = 0.5) -> MotionState:
        """
        Execute optimized motion
        
        Args:
            task_params: Task parameters
            optimization_level: Optimization level
            
        Returns:
            Motion state
        """
        logger.info(f"{Fore.GREEN}Executing optimized motion (level: {optimization_level:.1f}){Style.RESET_ALL}")
        
        # Generate BC waypoints first
        bc_waypoints = self.generate_bc_motion(task_params)
        
        # Optimize
        optimized_waypoints = self.generate_optimized_motion(bc_waypoints, optimization_level)
        
        # Create motion state
        state = MotionState(
            motion_type=MotionType.OPTIMIZED,
            waypoints=optimized_waypoints,
            timestamp=time.time(),
            source=MotionSource.OPTIMIZER,
            description=f"Optimized motion (level: {optimization_level:.1f})"
        )
        
        # Save to history
        self.history.push(state)
        
        # Execute
        self._execute_waypoints(optimized_waypoints)
        
        # Compute metrics
        state.metrics = self._compute_metrics(optimized_waypoints)
        
        return state
        
    def undo(self) -> bool:
        """
        Undo to previous motion
        
        Returns:
            Success status
        """
        if not self.history.can_undo():
            logger.warning("No previous state to undo to")
            return False
            
        previous_state = self.history.undo()
        
        if previous_state:
            logger.info(f"{Fore.YELLOW}Undoing to: {previous_state.summary()}{Style.RESET_ALL}")
            
            # Re-execute previous motion
            self._execute_waypoints(previous_state.waypoints)
            
            # Show comparison
            self._show_state_comparison(self.history.current_state, previous_state)
            
            return True
            
        return False
        
    def redo(self) -> bool:
        """
        Redo to next motion
        
        Returns:
            Success status
        """
        if not self.history.can_redo():
            logger.warning("No next state to redo to")
            return False
            
        next_state = self.history.redo()
        
        if next_state:
            logger.info(f"{Fore.YELLOW}Redoing to: {next_state.summary()}{Style.RESET_ALL}")
            
            # Re-execute next motion
            self._execute_waypoints(next_state.waypoints)
            
            # Show comparison
            self._show_state_comparison(self.history.current_state, next_state)
            
            return True
            
        return False
        
    def toggle_motion_type(self) -> MotionState:
        """
        Toggle between BC-only and optimized motion
        
        Returns:
            New motion state
        """
        if not self.history.current_state:
            logger.warning("No current state to toggle")
            return None
            
        current = self.history.current_state
        
        if current.motion_type == MotionType.BC_ONLY:
            # Switch to optimized
            logger.info(f"{Fore.CYAN}Switching from BC-only to Optimized{Style.RESET_ALL}")
            
            # Generate optimized from current BC waypoints
            optimized_waypoints = self.generate_optimized_motion(current.waypoints)
            
            new_state = MotionState(
                motion_type=MotionType.OPTIMIZED,
                waypoints=optimized_waypoints,
                timestamp=time.time(),
                source=MotionSource.OPTIMIZER,
                description="Toggled to optimized"
            )
            
        else:
            # Switch to BC-only
            logger.info(f"{Fore.GREEN}Switching from Optimized to BC-only{Style.RESET_ALL}")
            
            # Need to regenerate BC waypoints or retrieve from history
            bc_state = self._find_last_bc_state()
            
            if bc_state:
                new_state = MotionState(
                    motion_type=MotionType.BC_ONLY,
                    waypoints=bc_state.waypoints,
                    timestamp=time.time(),
                    source=MotionSource.BC_MODEL,
                    description="Toggled to BC-only"
                )
            else:
                logger.warning("No BC state found in history")
                return current
                
        # Save and execute
        self.history.push(new_state)
        self._execute_waypoints(new_state.waypoints)
        new_state.metrics = self._compute_metrics(new_state.waypoints)
        
        # Show comparison
        self._show_state_comparison(current, new_state)
        
        return new_state
        
    def start_ab_test(self, task_params: Dict[str, Any], num_trials: int = 5):
        """
        Start A/B testing between BC-only and optimized
        
        Args:
            task_params: Task parameters
            num_trials: Number of trials for each type
        """
        logger.info(f"\n{Fore.CYAN}{'='*60}")
        logger.info(" A/B TESTING MODE")
        logger.info(f"{'='*60}{Style.RESET_ALL}")
        
        self.ab_mode = True
        self.ab_results = []
        
        for trial in range(num_trials):
            logger.info(f"\n{Fore.YELLOW}Trial {trial + 1}/{num_trials}{Style.RESET_ALL}")
            
            # Test BC-only
            logger.info("  Testing BC-only...")
            bc_state = self.execute_bc_only(task_params)
            self.ab_results.append({
                'trial': trial + 1,
                'type': 'BC-only',
                'metrics': bc_state.metrics,
                'waypoints': len(bc_state.waypoints)
            })
            
            time.sleep(1)  # Pause between tests
            
            # Test Optimized
            logger.info("  Testing Optimized...")
            opt_state = self.execute_optimized(task_params, optimization_level=0.8)
            self.ab_results.append({
                'trial': trial + 1,
                'type': 'Optimized',
                'metrics': opt_state.metrics,
                'waypoints': len(opt_state.waypoints)
            })
            
            time.sleep(1)
            
        # Show results
        self._show_ab_results()
        
        self.ab_mode = False
        
    def _show_ab_results(self):
        """Display A/B testing results"""
        
        if not self.ab_results:
            return
            
        # Separate results by type
        bc_results = [r for r in self.ab_results if r['type'] == 'BC-only']
        opt_results = [r for r in self.ab_results if r['type'] == 'Optimized']
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(" A/B TEST RESULTS")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        # Calculate averages
        bc_avg_time = np.mean([r['metrics'].get('execution_time', 0) for r in bc_results])
        opt_avg_time = np.mean([r['metrics'].get('execution_time', 0) for r in opt_results])
        
        bc_avg_distance = np.mean([r['metrics'].get('total_distance', 0) for r in bc_results])
        opt_avg_distance = np.mean([r['metrics'].get('total_distance', 0) for r in opt_results])
        
        bc_avg_waypoints = np.mean([r['waypoints'] for r in bc_results])
        opt_avg_waypoints = np.mean([r['waypoints'] for r in opt_results])
        
        # Display comparison
        print(f"\n{'Metric':<20} {'BC-only':>15} {'Optimized':>15} {'Improvement':>15}")
        print("-" * 65)
        
        # Time
        time_improvement = (bc_avg_time - opt_avg_time) / bc_avg_time * 100
        time_color = Fore.GREEN if time_improvement > 0 else Fore.RED
        print(f"{'Execution Time':<20} {bc_avg_time:>14.3f}s {opt_avg_time:>14.3f}s "
              f"{time_color}{time_improvement:>14.1f}%{Style.RESET_ALL}")
        
        # Distance
        dist_improvement = (bc_avg_distance - opt_avg_distance) / bc_avg_distance * 100
        dist_color = Fore.GREEN if dist_improvement > 0 else Fore.RED
        print(f"{'Total Distance':<20} {bc_avg_distance:>14.3f}m {opt_avg_distance:>14.3f}m "
              f"{dist_color}{dist_improvement:>14.1f}%{Style.RESET_ALL}")
        
        # Waypoints
        wp_reduction = (bc_avg_waypoints - opt_avg_waypoints) / bc_avg_waypoints * 100
        wp_color = Fore.GREEN if wp_reduction > 0 else Fore.RED
        print(f"{'Waypoints':<20} {bc_avg_waypoints:>14.1f} {opt_avg_waypoints:>14.1f} "
              f"{wp_color}{wp_reduction:>14.1f}%{Style.RESET_ALL}")
        
        # Winner determination
        print(f"\n{Fore.YELLOW}WINNER: ", end="")
        if time_improvement > 10 and dist_improvement > 10:
            print(f"{Fore.GREEN}Optimized Motion (Significant improvement){Style.RESET_ALL}")
        elif time_improvement > 0 or dist_improvement > 0:
            print(f"{Fore.GREEN}Optimized Motion (Marginal improvement){Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}BC-only Motion (More reliable){Style.RESET_ALL}")
            
    def _find_last_bc_state(self) -> Optional[MotionState]:
        """Find last BC-only state in history"""
        
        # Check current state
        if self.history.current_state and self.history.current_state.motion_type == MotionType.BC_ONLY:
            return self.history.current_state
            
        # Check past states
        for state in reversed(self.history.past_states):
            if state.motion_type == MotionType.BC_ONLY:
                return state
                
        return None
        
    def _show_state_comparison(self, old_state: MotionState, new_state: MotionState):
        """Show comparison between two states"""
        
        print(f"\n{Fore.YELLOW}State Comparison:{Style.RESET_ALL}")
        print(f"  From: {old_state.motion_type.value} ({len(old_state.waypoints)} waypoints)")
        print(f"  To:   {new_state.motion_type.value} ({len(new_state.waypoints)} waypoints)")
        
        if old_state.metrics and new_state.metrics:
            time_diff = new_state.metrics.get('execution_time', 0) - old_state.metrics.get('execution_time', 0)
            dist_diff = new_state.metrics.get('total_distance', 0) - old_state.metrics.get('total_distance', 0)
            
            time_color = Fore.GREEN if time_diff < 0 else Fore.RED if time_diff > 0 else Fore.YELLOW
            dist_color = Fore.GREEN if dist_diff < 0 else Fore.RED if dist_diff > 0 else Fore.YELLOW
            
            print(f"  Time change: {time_color}{time_diff:+.3f}s{Style.RESET_ALL}")
            print(f"  Distance change: {dist_color}{dist_diff:+.3f}m{Style.RESET_ALL}")
            
    def _execute_waypoints(self, waypoints: List[Dict[str, Any]]):
        """Execute waypoints (placeholder)"""
        self.current_waypoints = waypoints
        self.is_executing = True
        
        # Simulate execution
        for i, waypoint in enumerate(waypoints):
            if 'position' in waypoint:
                # Move robot to waypoint position
                # This would call actual robot control
                pass
                
            if 'gripper' in waypoint:
                # Set gripper state
                pass
                
            # Small delay for simulation
            time.sleep(0.01)
            
        self.is_executing = False
        
    def _compute_metrics(self, waypoints: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute motion metrics"""
        
        metrics = {
            'execution_time': 0.0,
            'total_distance': 0.0,
            'num_waypoints': len(waypoints)
        }
        
        if len(waypoints) < 2:
            return metrics
            
        # Calculate total distance
        for i in range(len(waypoints) - 1):
            pos1 = np.array(waypoints[i]['position'])
            pos2 = np.array(waypoints[i + 1]['position'])
            metrics['total_distance'] += np.linalg.norm(pos2 - pos1)
            
        # Estimate execution time
        metrics['execution_time'] = metrics['total_distance'] / 0.1  # Assume 0.1 m/s
        
        return metrics
        
    def _generate_dummy_bc_waypoints(self, task_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dummy BC waypoints for testing"""
        
        # Simple pick and place trajectory
        waypoints = [
            {'position': [0.3, 0.0, 0.1], 'gripper': 1.0, 'description': 'Start'},
            {'position': [0.4, 0.0, 0.2], 'gripper': 1.0, 'description': 'Approach'},
            {'position': [0.4, 0.0, 0.05], 'gripper': 1.0, 'description': 'Pre-grasp'},
            {'position': [0.4, 0.0, 0.05], 'gripper': 0.0, 'description': 'Grasp'},
            {'position': [0.4, 0.0, 0.2], 'gripper': 0.0, 'description': 'Lift'},
            {'position': [0.3, 0.2, 0.2], 'gripper': 0.0, 'description': 'Transport'},
            {'position': [0.3, 0.2, 0.1], 'gripper': 0.0, 'description': 'Lower'},
            {'position': [0.3, 0.2, 0.1], 'gripper': 1.0, 'description': 'Release'},
            {'position': [0.3, 0.2, 0.15], 'gripper': 1.0, 'description': 'Retract'}
        ]
        
        return waypoints
        
    def save_history(self, filepath: str):
        """Save motion history to file"""
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'history': self.history,
                'ab_results': self.ab_results
            }, f)
            
        logger.info(f"History saved to {filepath}")
        
    def load_history(self, filepath: str):
        """Load motion history from file"""
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.history = data['history']
        self.ab_results = data.get('ab_results', [])
        
        logger.info(f"History loaded from {filepath}")
        
    def print_status(self):
        """Print current controller status"""
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(" MOTION CONTROLLER STATUS")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        # Current state
        if self.history.current_state:
            print(f"\nCurrent Motion: {self.history.current_state.motion_type.value}")
            print(f"Waypoints: {len(self.history.current_state.waypoints)}")
            print(f"Source: {self.history.current_state.source.value}")
            
            if self.history.current_state.metrics:
                print(f"Execution Time: {self.history.current_state.metrics.get('execution_time', 0):.3f}s")
                print(f"Total Distance: {self.history.current_state.metrics.get('total_distance', 0):.3f}m")
        else:
            print("\nNo current motion state")
            
        # History status
        history_summary = self.history.get_history_summary()
        print(f"\nHistory:")
        print(f"  Past states: {history_summary['past_count']}")
        print(f"  Future states: {history_summary['future_count']}")
        print(f"  Can undo: {'Yes' if history_summary['can_undo'] else 'No'}")
        print(f"  Can redo: {'Yes' if history_summary['can_redo'] else 'No'}")
        
        # A/B test status
        if self.ab_mode:
            print(f"\n{Fore.YELLOW}A/B Testing Active{Style.RESET_ALL}")
            print(f"Results collected: {len(self.ab_results)}")
            
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


# Demo and testing
def demo_motion_controller():
    """Demonstrate motion controller with undo"""
    
    print(f"\n{Fore.CYAN}{'='*80}")
    print(" MOTION CONTROLLER WITH UNDO DEMONSTRATION")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    # Create dummy robot and scene
    robot_id = 0
    scene_objects = {'cube': 1, 'platform': 2}
    
    # Create controller
    controller = MotionControllerWithUndo(robot_id, scene_objects)
    
    # Task parameters
    task_params = {
        'object_id': 1,
        'target_position': [0.3, 0.2, 0.1]
    }
    
    print(f"\n{Fore.YELLOW}1. Execute BC-only motion:{Style.RESET_ALL}")
    bc_state = controller.execute_bc_only(task_params)
    print(f"   Generated {len(bc_state.waypoints)} waypoints")
    print(f"   Execution time: {bc_state.metrics['execution_time']:.3f}s")
    
    time.sleep(1)
    
    print(f"\n{Fore.YELLOW}2. Execute Optimized motion:{Style.RESET_ALL}")
    opt_state = controller.execute_optimized(task_params, optimization_level=0.7)
    print(f"   Generated {len(opt_state.waypoints)} waypoints")
    print(f"   Execution time: {opt_state.metrics['execution_time']:.3f}s")
    
    time.sleep(1)
    
    print(f"\n{Fore.YELLOW}3. Undo to BC-only:{Style.RESET_ALL}")
    success = controller.undo()
    if success:
        print("   ✅ Successfully undone to BC-only motion")
    
    time.sleep(1)
    
    print(f"\n{Fore.YELLOW}4. Redo to Optimized:{Style.RESET_ALL}")
    success = controller.redo()
    if success:
        print("   ✅ Successfully redone to optimized motion")
    
    time.sleep(1)
    
    print(f"\n{Fore.YELLOW}5. Toggle motion type:{Style.RESET_ALL}")
    new_state = controller.toggle_motion_type()
    print(f"   Toggled to: {new_state.motion_type.value}")
    
    time.sleep(1)
    
    print(f"\n{Fore.YELLOW}6. A/B Testing (3 trials):{Style.RESET_ALL}")
    controller.start_ab_test(task_params, num_trials=3)
    
    # Print final status
    controller.print_status()
    
    return controller


def interactive_demo():
    """Interactive demo with user commands"""
    
    print(f"\n{Fore.CYAN}{'='*80}")
    print(" INTERACTIVE MOTION CONTROLLER")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    # Setup
    robot_id = 0
    scene_objects = {'cube': 1, 'platform': 2}
    controller = MotionControllerWithUndo(robot_id, scene_objects)
    
    task_params = {
        'object_id': 1,
        'target_position': [0.3, 0.2, 0.1]
    }
    
    # Commands
    commands = {
        'b': ('Execute BC-only motion', lambda: controller.execute_bc_only(task_params)),
        'o': ('Execute Optimized motion', lambda: controller.execute_optimized(task_params)),
        'u': ('Undo', lambda: controller.undo()),
        'r': ('Redo', lambda: controller.redo()),
        't': ('Toggle motion type', lambda: controller.toggle_motion_type()),
        'a': ('Run A/B test', lambda: controller.start_ab_test(task_params, num_trials=3)),
        's': ('Show status', lambda: controller.print_status()),
        'q': ('Quit', lambda: 'quit')
    }
    
    print("\nCommands:")
    for key, (desc, _) in commands.items():
        print(f"  {key}: {desc}")
    
    while True:
        try:
            cmd = input(f"\n{Fore.YELLOW}Enter command: {Style.RESET_ALL}").strip().lower()
            
            if cmd not in commands:
                print(f"{Fore.RED}Invalid command{Style.RESET_ALL}")
                continue
                
            if cmd == 'q':
                print("Exiting...")
                break
                
            # Execute command
            result = commands[cmd][1]()
            
            if result == 'quit':
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
            
    return controller


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        controller = interactive_demo()
    else:
        controller = demo_motion_controller()
    
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")