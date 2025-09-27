"""
Reward function templates for reinforcement learning in robotic manipulation.

This module provides reward function templates that can be customized by
specifying weight parameters for different reward components.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import logging

# Configure logging
logger = logging.getLogger(__name__)


def compute_reward(
    state: Dict[str, Any],
    action: Union[np.ndarray, List[float]],
    next_state: Dict[str, Any],
    info: Dict[str, Any],
    w: Dict[str, float]
) -> float:
    """
    Compute reward for robotic manipulation task using weighted components.
    
    This function calculates a scalar reward value based on multiple weighted
    components. The weights in 'w' determine the relative importance of each
    reward component.
    
    GPT INSTRUCTIONS FOR WEIGHT SPECIFICATION:
    ==========================================
    When asked to design a reward function, ONLY modify the float values in the
    weight dictionary w. DO NOT change the function structure or add new keys.
    
    Fill in the following weights with appropriate float values:
    
    w = {
        'dist': _____,        # Distance to target (negative for penalty, e.g., -1.0)
        'grasp': _____,       # Successful grasp bonus (positive, e.g., 10.0)
        'collision': _____,   # Collision penalty (negative, e.g., -5.0)
        'success': _____,     # Task completion bonus (positive, e.g., 100.0)
        'time': _____,        # Time penalty per step (negative, e.g., -0.1)
        'smooth': _____,      # Action smoothness bonus (positive, e.g., 0.5)
        'orientation': _____  # Orientation alignment bonus (positive, e.g., 2.0)
    }
    
    Example good response:
    w = {
        'dist': -1.0,
        'grasp': 10.0,
        'collision': -5.0,
        'success': 100.0,
        'time': -0.1,
        'smooth': 0.5,
        'orientation': 2.0
    }
    
    Guidelines for weight selection:
    - Use negative values for penalties (things to minimize)
    - Use positive values for bonuses (things to maximize)
    - Scale weights based on relative importance
    - Typical ranges: penalties [-10, 0], bonuses [0, 100]
    - Time penalty should be small to avoid rushing
    ==========================================
    
    Args:
        state: Current state dictionary containing:
            - 'robot_pose': End-effector position and orientation
            - 'object_positions': Dictionary of object positions
            - 'gripper_state': Gripper open/closed state
            - 'timestep': Current simulation timestep
            
        action: Action taken (joint velocities or end-effector displacement)
        
        next_state: Resulting state after action
        
        info: Additional information dictionary containing:
            - 'collision': Boolean, whether collision occurred
            - 'grasp_success': Boolean, whether object was grasped
            - 'task_success': Boolean, whether task completed
            - 'distance_to_target': Float, distance to target
            
        w: Weight dictionary with keys:
            - 'dist': Weight for distance penalty
            - 'grasp': Weight for grasp success bonus
            - 'collision': Weight for collision penalty
            - 'success': Weight for task completion bonus
            - 'time': Weight for time penalty
            - 'smooth': Weight for action smoothness
            - 'orientation': Weight for orientation alignment
            
    Returns:
        Float reward value (sum of weighted components)
        
    Example:
        # Define weights for a grasping task
        weights = {
            'dist': -1.0,      # Penalize distance to object
            'grasp': 10.0,     # Reward successful grasp
            'collision': -5.0,  # Penalize collisions
            'success': 100.0,  # Large bonus for task completion
            'time': -0.1,      # Small time penalty
            'smooth': 0.5,     # Encourage smooth motion
            'orientation': 2.0  # Reward proper gripper orientation
        }
        
        reward = compute_reward(state, action, next_state, info, weights)
    """
    # Initialize reward
    reward = 0.0
    
    # Distance component: Penalize distance to target
    if 'dist' in w and w['dist'] != 0:
        distance = info.get('distance_to_target', 0.0)
        reward += w['dist'] * distance
    
    # Grasp component: Bonus for successful grasp
    if 'grasp' in w and w['grasp'] != 0:
        if info.get('grasp_success', False):
            reward += w['grasp']
    
    # Collision component: Penalty for collisions
    if 'collision' in w and w['collision'] != 0:
        if info.get('collision', False):
            reward += w['collision']
    
    # Success component: Bonus for task completion
    if 'success' in w and w['success'] != 0:
        if info.get('task_success', False):
            reward += w['success']
    
    # Time component: Penalty for each timestep
    if 'time' in w and w['time'] != 0:
        reward += w['time']
    
    # Smoothness component: Reward smooth actions
    if 'smooth' in w and w['smooth'] != 0:
        if isinstance(action, (list, np.ndarray)):
            # Penalize large actions (encourage smoothness)
            action_magnitude = np.linalg.norm(action)
            reward += w['smooth'] * (1.0 - min(action_magnitude, 1.0))
    
    # Orientation component: Reward proper alignment
    if 'orientation' in w and w['orientation'] != 0:
        alignment = info.get('orientation_alignment', 0.0)  # Value between 0 and 1
        reward += w['orientation'] * alignment
    
    return reward


def create_weight_template() -> Dict[str, float]:
    """
    Create an empty weight template for the reward function.
    
    Returns a dictionary with all weight keys initialized to 0.0.
    This template should be filled with appropriate values for the specific task.
    
    Returns:
        Dictionary with weight keys and zero values
        
    Example:
        w = create_weight_template()
        w['dist'] = -1.0      # Set distance penalty
        w['grasp'] = 10.0     # Set grasp bonus
        w['collision'] = -5.0  # Set collision penalty
    """
    return {
        'dist': 0.0,        # Distance to target weight
        'grasp': 0.0,       # Grasp success weight
        'collision': 0.0,   # Collision penalty weight
        'success': 0.0,     # Task success weight
        'time': 0.0,        # Time penalty weight
        'smooth': 0.0,      # Action smoothness weight
        'orientation': 0.0  # Orientation alignment weight
    }


def design_reward_weights(task_description: str) -> Dict[str, float]:
    """
    Template function for GPT to fill in reward weights based on task description.
    
    GPT SHOULD ONLY MODIFY THE FLOAT VALUES IN THE RETURNED DICTIONARY.
    
    Args:
        task_description: Natural language description of the task
        
    Returns:
        Weight dictionary with appropriate values for the task
        
    Example GPT Response for "Pick up the red cube carefully":
        return {
            'dist': -2.0,       # Strong penalty for distance
            'grasp': 20.0,      # High reward for grasping
            'collision': -10.0, # High penalty for collisions (careful)
            'success': 100.0,   # Large success bonus
            'time': -0.05,      # Small time penalty (careful > fast)
            'smooth': 1.0,      # Reward smooth motion (careful)
            'orientation': 3.0  # Important to align gripper properly
        }
    """
    # GPT SHOULD FILL THESE VALUES BASED ON THE TASK
    return {
        'dist': 0.0,        # FILL THIS: Distance penalty (negative)
        'grasp': 0.0,       # FILL THIS: Grasp bonus (positive)
        'collision': 0.0,   # FILL THIS: Collision penalty (negative)
        'success': 0.0,     # FILL THIS: Success bonus (positive)
        'time': 0.0,        # FILL THIS: Time penalty (negative)
        'smooth': 0.0,      # FILL THIS: Smoothness bonus (positive)
        'orientation': 0.0  # FILL THIS: Orientation bonus (positive)
    }


class RewardFunction:
    """
    Configurable reward function class for robotic tasks.
    
    This class allows creating custom reward functions by specifying weights
    for different reward components.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize reward function with weights.
        
        Args:
            weights: Dictionary of component weights. If None, uses zeros.
        """
        self.weights = weights if weights is not None else create_weight_template()
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate weight dictionary has expected keys."""
        expected_keys = {'dist', 'grasp', 'collision', 'success', 
                        'time', 'smooth', 'orientation'}
        
        for key in expected_keys:
            if key not in self.weights:
                logger.warning(f"Missing weight key '{key}', setting to 0.0")
                self.weights[key] = 0.0
    
    def __call__(
        self,
        state: Dict[str, Any],
        action: Union[np.ndarray, List[float]],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """
        Compute reward using stored weights.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            info: Additional information
            
        Returns:
            Reward value
        """
        return compute_reward(state, action, next_state, info, self.weights)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update reward weights.
        
        Args:
            new_weights: Dictionary with weight updates
        """
        self.weights.update(new_weights)
        self._validate_weights()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weight values."""
        return self.weights.copy()
    
    def describe(self) -> str:
        """Get human-readable description of reward function."""
        lines = ["Reward Function Weights:"]
        for key, value in self.weights.items():
            if value != 0:
                sign = "+" if value > 0 else ""
                lines.append(f"  {key}: {sign}{value:.2f}")
        return "\n".join(lines)


# Preset reward functions for common tasks
PRESET_REWARDS = {
    'grasping': {
        'dist': -1.0,
        'grasp': 50.0,
        'collision': -10.0,
        'success': 100.0,
        'time': -0.1,
        'smooth': 0.5,
        'orientation': 5.0
    },
    'pick_and_place': {
        'dist': -2.0,
        'grasp': 20.0,
        'collision': -5.0,
        'success': 150.0,
        'time': -0.1,
        'smooth': 1.0,
        'orientation': 3.0
    },
    'stacking': {
        'dist': -3.0,
        'grasp': 10.0,
        'collision': -20.0,  # High penalty for knocking stack
        'success': 200.0,
        'time': -0.05,       # Allow more time for precision
        'smooth': 2.0,       # Emphasize smooth motion
        'orientation': 10.0  # Critical for stacking
    },
    'pushing': {
        'dist': -0.5,        # Less critical than grasping
        'grasp': 0.0,        # Not relevant for pushing
        'collision': -2.0,   # Some collision expected
        'success': 100.0,
        'time': -0.1,
        'smooth': 0.2,
        'orientation': 1.0
    },
    'careful_manipulation': {
        'dist': -2.0,
        'grasp': 30.0,
        'collision': -50.0,  # Very high collision penalty
        'success': 100.0,
        'time': -0.02,       # Very low time pressure
        'smooth': 5.0,       # High smoothness reward
        'orientation': 8.0
    }
}


def get_preset_reward(task_type: str) -> RewardFunction:
    """
    Get a preset reward function for a common task type.
    
    Args:
        task_type: Type of task ('grasping', 'pick_and_place', 'stacking', etc.)
        
    Returns:
        RewardFunction with appropriate weights
        
    Raises:
        ValueError: If task_type not recognized
        
    Example:
        reward_fn = get_preset_reward('grasping')
        reward = reward_fn(state, action, next_state, info)
    """
    if task_type not in PRESET_REWARDS:
        available = ', '.join(PRESET_REWARDS.keys())
        raise ValueError(f"Unknown task type '{task_type}'. Available: {available}")
    
    weights = PRESET_REWARDS[task_type]
    return RewardFunction(weights)


def interpolate_weights(
    w1: Dict[str, float],
    w2: Dict[str, float],
    alpha: float
) -> Dict[str, float]:
    """
    Interpolate between two weight dictionaries.
    
    Useful for curriculum learning or smooth weight transitions.
    
    Args:
        w1: First weight dictionary
        w2: Second weight dictionary
        alpha: Interpolation factor (0 = w1, 1 = w2)
        
    Returns:
        Interpolated weight dictionary
        
    Example:
        # Gradually transition from exploration to exploitation
        early_weights = {'dist': -0.1, 'success': 10.0, ...}
        late_weights = {'dist': -2.0, 'success': 100.0, ...}
        current_weights = interpolate_weights(early_weights, late_weights, 0.5)
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    result = {}
    
    all_keys = set(w1.keys()) | set(w2.keys())
    for key in all_keys:
        v1 = w1.get(key, 0.0)
        v2 = w2.get(key, 0.0)
        result[key] = (1 - alpha) * v1 + alpha * v2
    
    return result


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Using compute_reward directly
    print("Example 1: Direct reward computation")
    print("-" * 50)
    
    state = {
        'robot_pose': ([0.5, 0.0, 0.3], [0, 0, 0, 1]),
        'object_positions': {'red_cube': [0.6, 0.1, 0.1]},
        'gripper_state': 0.0,  # Closed
        'timestep': 10
    }
    
    action = [0.1, 0.0, -0.05, 0.0, 0.0, 0.0]  # Small movement
    
    next_state = {
        'robot_pose': ([0.55, 0.0, 0.25], [0, 0, 0, 1]),
        'object_positions': {'red_cube': [0.6, 0.1, 0.1]},
        'gripper_state': 0.0,
        'timestep': 11
    }
    
    info = {
        'collision': False,
        'grasp_success': False,
        'task_success': False,
        'distance_to_target': 0.15,
        'orientation_alignment': 0.9
    }
    
    # Define weights for a grasping task
    weights = {
        'dist': -1.0,
        'grasp': 10.0,
        'collision': -5.0,
        'success': 100.0,
        'time': -0.1,
        'smooth': 0.5,
        'orientation': 2.0
    }
    
    reward = compute_reward(state, action, next_state, info, weights)
    print(f"Reward: {reward:.3f}")
    print()
    
    # Example 2: Using RewardFunction class
    print("Example 2: RewardFunction class")
    print("-" * 50)
    
    reward_fn = RewardFunction(weights)
    print(reward_fn.describe())
    print(f"Reward: {reward_fn(state, action, next_state, info):.3f}")
    print()
    
    # Example 3: Using preset rewards
    print("Example 3: Preset reward functions")
    print("-" * 50)
    
    for task_type in ['grasping', 'stacking', 'careful_manipulation']:
        preset = get_preset_reward(task_type)
        print(f"\n{task_type.upper()}:")
        print(preset.describe())
    
    print()
    
    # Example 4: Template for GPT
    print("Example 4: Template for GPT to fill")
    print("-" * 50)
    print("""
    Task: "Pick up the fragile glass carefully and place it on the shelf"
    
    Please fill in the weight values:
    
    w = {
        'dist': ___,        # Distance penalty
        'grasp': ___,       # Grasp bonus
        'collision': ___,   # Collision penalty (should be high for fragile object)
        'success': ___,     # Success bonus
        'time': ___,        # Time penalty (should be low for careful task)
        'smooth': ___,      # Smoothness bonus (should be high)
        'orientation': ___  # Orientation bonus
    }
    """)