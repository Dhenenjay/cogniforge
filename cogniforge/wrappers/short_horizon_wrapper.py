"""
Short-horizon Gymnasium wrapper for end-effector delta control.

This wrapper provides:
- End-effector delta position commands
- Action repeat (default 10 steps)
- Short horizon episodes
- Automatic coordinate frame handling
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ShortHorizonDeltaWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for short-horizon control with EE delta commands.
    
    This wrapper:
    1. Converts actions to end-effector delta commands
    2. Repeats each action for multiple steps (action_repeat)
    3. Manages short horizon episodes
    4. Accumulates rewards over repeated steps
    5. Handles early termination within action repeats
    
    Args:
        env: Base Gymnasium environment (should support EE control)
        action_repeat: Number of times to repeat each action (default 10)
        delta_scale: Scaling factor for delta commands (default 0.05)
        max_delta: Maximum allowed delta per command (default 0.1)
        horizon: Maximum episode length in macro-steps (default 50)
        use_gripper: Whether to include gripper commands (default True)
        frame: Coordinate frame for deltas ('world' or 'ee', default 'world')
        accumulate_info: Whether to accumulate info dicts (default False)
        
    Example:
        env = gym.make('FrankaPickPlace-v0')
        wrapped_env = ShortHorizonDeltaWrapper(
            env,
            action_repeat=10,
            delta_scale=0.05,
            horizon=50
        )
        
        obs, info = wrapped_env.reset()
        # Action is now [dx, dy, dz, gripper] in [-1, 1]
        action = np.array([0.1, 0.0, -0.2, 1.0])
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
    """
    
    def __init__(
        self,
        env: gym.Env,
        action_repeat: int = 10,
        delta_scale: float = 0.05,
        max_delta: float = 0.1,
        horizon: int = 50,
        use_gripper: bool = True,
        frame: str = 'world',
        accumulate_info: bool = False,
        clip_actions: bool = True,
        relative_to_current: bool = True,
        smooth_actions: bool = False,
        smoothing_alpha: float = 0.3,
        return_ee_pos: bool = True,
        ee_key: str = 'ee_pos',
        verbose: bool = False
    ):
        super().__init__(env)
        
        self.action_repeat = action_repeat
        self.delta_scale = delta_scale
        self.max_delta = max_delta
        self.horizon = horizon
        self.use_gripper = use_gripper
        self.frame = frame
        self.accumulate_info = accumulate_info
        self.clip_actions = clip_actions
        self.relative_to_current = relative_to_current
        self.smooth_actions = smooth_actions
        self.smoothing_alpha = smoothing_alpha
        self.return_ee_pos = return_ee_pos
        self.ee_key = ee_key
        self.verbose = verbose
        
        # Track episode state
        self.macro_step_count = 0
        self.total_step_count = 0
        self.current_ee_pos = None
        self.target_ee_pos = None
        self.action_history = deque(maxlen=5)
        
        # Redefine action space for delta commands
        action_dim = 3  # dx, dy, dz
        if self.use_gripper:
            action_dim += 1  # Add gripper dimension
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Store original action space for reference
        self._original_action_space = env.action_space
        
        if verbose:
            logger.info(f"ShortHorizonDeltaWrapper initialized:")
            logger.info(f"  Action repeat: {action_repeat}")
            logger.info(f"  Delta scale: {delta_scale}")
            logger.info(f"  Horizon: {horizon} macro-steps")
            logger.info(f"  Frame: {frame}")
            logger.info(f"  New action space: {self.action_space}")
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment and wrapper state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset wrapper state
        self.macro_step_count = 0
        self.total_step_count = 0
        self.action_history.clear()
        
        # Get initial EE position
        self.current_ee_pos = self._get_ee_position(obs, info)
        self.target_ee_pos = self.current_ee_pos.copy() if self.current_ee_pos is not None else None
        
        # Add EE position to observation if requested
        if self.return_ee_pos and self.current_ee_pos is not None:
            obs = self._add_ee_to_obs(obs, self.current_ee_pos)
        
        # Add wrapper info
        info['wrapper'] = {
            'macro_steps': 0,
            'total_steps': 0,
            'ee_pos': self.current_ee_pos.copy() if self.current_ee_pos is not None else None
        }
        
        if self.verbose:
            logger.info(f"Environment reset. Initial EE pos: {self.current_ee_pos}")
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute delta action with action repeat.
        
        Args:
            action: Delta command [dx, dy, dz, (gripper)] in [-1, 1]
            
        Returns:
            Standard Gymnasium step returns with accumulated reward
        """
        # Validate and process action
        action = np.array(action, dtype=np.float32)
        
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        
        # Smooth actions if enabled
        if self.smooth_actions and len(self.action_history) > 0:
            prev_action = self.action_history[-1]
            action = self.smoothing_alpha * action + (1 - self.smoothing_alpha) * prev_action
        
        self.action_history.append(action.copy())
        
        # Extract delta position and gripper
        if self.use_gripper:
            delta_pos = action[:3] * self.delta_scale
            gripper_cmd = action[3]
        else:
            delta_pos = action * self.delta_scale
            gripper_cmd = None
        
        # Clip delta to maximum
        delta_norm = np.linalg.norm(delta_pos)
        if delta_norm > self.max_delta:
            delta_pos = delta_pos * (self.max_delta / delta_norm)
        
        # Compute target position
        if self.relative_to_current and self.current_ee_pos is not None:
            self.target_ee_pos = self.current_ee_pos + delta_pos
        elif self.target_ee_pos is not None:
            self.target_ee_pos = self.target_ee_pos + delta_pos
        else:
            # Fallback: use delta directly
            self.target_ee_pos = delta_pos
        
        # Execute action repeat
        accumulated_reward = 0.0
        accumulated_info = {}
        last_obs = None
        terminated = False
        truncated = False
        
        for repeat_idx in range(self.action_repeat):
            # Convert delta to environment action
            env_action = self._delta_to_env_action(
                delta_pos if repeat_idx == 0 else np.zeros(3),  # Only apply delta on first repeat
                gripper_cmd,
                last_obs
            )
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            
            # Update state
            last_obs = obs
            accumulated_reward += reward
            self.total_step_count += 1
            
            # Update current EE position
            new_ee_pos = self._get_ee_position(obs, info)
            if new_ee_pos is not None:
                self.current_ee_pos = new_ee_pos
            
            # Accumulate info if requested
            if self.accumulate_info:
                self._accumulate_info(accumulated_info, info, repeat_idx)
            else:
                accumulated_info = info  # Keep only last info
            
            # Check for early termination
            if terminated or truncated:
                if self.verbose:
                    logger.info(f"Early termination at repeat {repeat_idx + 1}/{self.action_repeat}")
                break
        
        # Update macro step count
        self.macro_step_count += 1
        
        # Check horizon truncation
        if self.macro_step_count >= self.horizon:
            truncated = True
            if self.verbose:
                logger.info(f"Horizon reached: {self.macro_step_count} macro-steps")
        
        # Add EE position to observation
        if self.return_ee_pos and self.current_ee_pos is not None:
            last_obs = self._add_ee_to_obs(last_obs, self.current_ee_pos)
        
        # Add wrapper info
        accumulated_info['wrapper'] = {
            'macro_steps': self.macro_step_count,
            'total_steps': self.total_step_count,
            'ee_pos': self.current_ee_pos.copy() if self.current_ee_pos is not None else None,
            'target_ee_pos': self.target_ee_pos.copy() if self.target_ee_pos is not None else None,
            'delta_applied': delta_pos,
            'action_repeats_executed': repeat_idx + 1
        }
        
        return last_obs, accumulated_reward, terminated, truncated, accumulated_info
    
    def _get_ee_position(self, obs: Any, info: Dict) -> Optional[np.ndarray]:
        """Extract end-effector position from observation or info."""
        # Try info dict first
        if self.ee_key in info:
            return np.array(info[self.ee_key])
        
        # Try nested robot state in info
        if 'robot_state' in info and self.ee_key in info['robot_state']:
            return np.array(info['robot_state'][self.ee_key])
        
        # Try observation dict
        if isinstance(obs, dict):
            if self.ee_key in obs:
                return np.array(obs[self.ee_key])
            if 'robot' in obs and isinstance(obs['robot'], dict):
                if self.ee_key in obs['robot']:
                    return np.array(obs['robot'][self.ee_key])
        
        # Try observation array (assume first 3 values are EE pos)
        if isinstance(obs, np.ndarray) and len(obs) >= 3:
            # This is a heuristic - may need adjustment for specific envs
            return obs[:3].copy()
        
        if self.verbose:
            logger.warning(f"Could not extract EE position from obs/info")
        
        return None
    
    def _add_ee_to_obs(self, obs: Any, ee_pos: np.ndarray) -> Any:
        """Add EE position to observation."""
        if isinstance(obs, dict):
            obs = obs.copy()
            obs['ee_position'] = ee_pos
        elif isinstance(obs, np.ndarray):
            # For array observations, concatenate EE position
            # This changes observation space - use with caution
            pass  # Keep original for now
        
        return obs
    
    def _delta_to_env_action(
        self,
        delta_pos: np.ndarray,
        gripper_cmd: Optional[float],
        obs: Any
    ) -> Any:
        """
        Convert delta command to environment action.
        
        This method needs to be adapted based on the specific environment's
        action interface.
        """
        # Get environment's expected action format
        if isinstance(self._original_action_space, gym.spaces.Box):
            action_dim = self._original_action_space.shape[0]
            env_action = np.zeros(action_dim, dtype=np.float32)
            
            # Common formats:
            # 1. Position control: [x, y, z, gripper]
            # 2. Velocity control: [vx, vy, vz, gripper]
            # 3. Joint control: [j1, j2, ..., jn]
            
            # Assuming position or velocity control for now
            if action_dim >= 3:
                if self.target_ee_pos is not None:
                    # Position control mode
                    env_action[:3] = self.target_ee_pos
                else:
                    # Velocity/delta control mode
                    env_action[:3] = delta_pos / self.delta_scale
                
                # Add gripper if available
                if self.use_gripper and gripper_cmd is not None and action_dim >= 4:
                    env_action[3] = gripper_cmd
            
            # Clip to action space bounds
            env_action = np.clip(
                env_action,
                self._original_action_space.low,
                self._original_action_space.high
            )
            
        elif isinstance(self._original_action_space, gym.spaces.Dict):
            # Dictionary action space
            env_action = {}
            
            if 'position' in self._original_action_space.spaces:
                if self.target_ee_pos is not None:
                    env_action['position'] = self.target_ee_pos
                else:
                    env_action['position'] = delta_pos
            
            if 'gripper' in self._original_action_space.spaces and gripper_cmd is not None:
                env_action['gripper'] = gripper_cmd
        
        else:
            # Fallback: return delta directly
            env_action = delta_pos
        
        return env_action
    
    def _accumulate_info(self, accumulated: Dict, new_info: Dict, step: int):
        """Accumulate info dictionaries across action repeats."""
        for key, value in new_info.items():
            if key not in accumulated:
                if isinstance(value, (int, float)):
                    accumulated[key] = value
                elif isinstance(value, np.ndarray):
                    accumulated[key] = [value]
                elif isinstance(value, list):
                    accumulated[key] = [value]
                elif isinstance(value, dict):
                    accumulated[key] = value.copy()
                else:
                    accumulated[key] = value
            else:
                if isinstance(value, (int, float)):
                    # Sum numeric values
                    accumulated[key] += value
                elif isinstance(value, np.ndarray):
                    # Collect arrays
                    if isinstance(accumulated[key], list):
                        accumulated[key].append(value)
                elif isinstance(value, list):
                    # Extend lists
                    if isinstance(accumulated[key], list):
                        accumulated[key].extend(value)
                elif isinstance(value, dict):
                    # Merge dicts recursively
                    if isinstance(accumulated[key], dict):
                        self._accumulate_info(accumulated[key], value, step)
    
    @property
    def unwrapped(self):
        """Get the unwrapped environment."""
        return self.env.unwrapped
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def render(self):
        """Render the environment."""
        return self.env.render()


class AdaptiveDeltaWrapper(ShortHorizonDeltaWrapper):
    """
    Extended wrapper with adaptive delta scaling based on task progress.
    
    This wrapper adjusts the delta scale based on:
    - Distance to goal
    - Task phase (approach vs fine manipulation)
    - Success rate history
    
    Args:
        env: Base environment
        initial_scale: Initial delta scale (default 0.05)
        min_scale: Minimum delta scale (default 0.01)
        max_scale: Maximum delta scale (default 0.1)
        adaptation_rate: How quickly to adapt scale (default 0.1)
        **kwargs: Additional arguments for base wrapper
    """
    
    def __init__(
        self,
        env: gym.Env,
        initial_scale: float = 0.05,
        min_scale: float = 0.01,
        max_scale: float = 0.1,
        adaptation_rate: float = 0.1,
        distance_threshold: float = 0.2,
        fine_control_distance: float = 0.05,
        **kwargs
    ):
        # Set initial scale
        kwargs['delta_scale'] = initial_scale
        super().__init__(env, **kwargs)
        
        self.initial_scale = initial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.adaptation_rate = adaptation_rate
        self.distance_threshold = distance_threshold
        self.fine_control_distance = fine_control_distance
        
        # Tracking for adaptation
        self.goal_position = None
        self.phase = 'approach'  # 'approach' or 'fine'
        self.success_history = deque(maxlen=10)
        self.distance_history = deque(maxlen=20)
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset with adaptive scale reset."""
        obs, info = super().reset(**kwargs)
        
        # Reset adaptive parameters
        self.delta_scale = self.initial_scale
        self.phase = 'approach'
        self.distance_history.clear()
        
        # Extract goal position if available
        self.goal_position = self._extract_goal_position(obs, info)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """Step with adaptive delta scaling."""
        # Adapt scale before stepping
        self._adapt_scale()
        
        # Execute step with current scale
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update tracking
        if self.goal_position is not None and self.current_ee_pos is not None:
            distance = np.linalg.norm(self.goal_position - self.current_ee_pos)
            self.distance_history.append(distance)
            
            # Update phase
            if distance < self.fine_control_distance:
                self.phase = 'fine'
            elif distance > self.distance_threshold:
                self.phase = 'approach'
        
        # Track success
        if terminated:
            success = info.get('success', False) or info.get('is_success', False)
            self.success_history.append(success)
        
        return obs, reward, terminated, truncated, info
    
    def _adapt_scale(self):
        """Adapt delta scale based on current state."""
        if len(self.distance_history) < 2:
            return
        
        # Compute distance trend
        recent_distances = list(self.distance_history)[-5:]
        distance_decreasing = all(
            recent_distances[i] >= recent_distances[i+1] 
            for i in range(len(recent_distances)-1)
        )
        
        # Adapt based on phase and progress
        if self.phase == 'fine':
            # Use smaller scale for fine control
            target_scale = self.min_scale * 1.5
        elif self.phase == 'approach':
            if distance_decreasing:
                # Making progress, can use larger scale
                target_scale = self.max_scale * 0.8
            else:
                # Not making progress, reduce scale
                target_scale = self.initial_scale
        else:
            target_scale = self.initial_scale
        
        # Smooth adaptation
        self.delta_scale = (
            self.adaptation_rate * target_scale + 
            (1 - self.adaptation_rate) * self.delta_scale
        )
        
        # Clip to bounds
        self.delta_scale = np.clip(self.delta_scale, self.min_scale, self.max_scale)
    
    def _extract_goal_position(self, obs: Any, info: Dict) -> Optional[np.ndarray]:
        """Extract goal position from observation or info."""
        # Try info dict
        for key in ['goal_pos', 'goal_position', 'target_pos', 'target_position']:
            if key in info:
                return np.array(info[key])
        
        # Try observation dict
        if isinstance(obs, dict):
            for key in ['goal', 'desired_goal', 'goal_pos']:
                if key in obs:
                    goal = obs[key]
                    if isinstance(goal, np.ndarray) and len(goal) >= 3:
                        return goal[:3]
        
        return None


class MultiStepPlannerWrapper(ShortHorizonDeltaWrapper):
    """
    Wrapper that allows planning multiple delta steps ahead.
    
    Instead of single delta commands, accepts a sequence of deltas
    and executes them in order with action repeat.
    
    Args:
        env: Base environment
        plan_horizon: Number of delta steps to plan (default 5)
        replan_frequency: How often to accept new plans (default 1)
        **kwargs: Additional arguments for base wrapper
    """
    
    def __init__(
        self,
        env: gym.Env,
        plan_horizon: int = 5,
        replan_frequency: int = 1,
        interpolate_plan: bool = True,
        **kwargs
    ):
        super().__init__(env, **kwargs)
        
        self.plan_horizon = plan_horizon
        self.replan_frequency = replan_frequency
        self.interpolate_plan = interpolate_plan
        
        # Redefine action space for multi-step plans
        action_dim = 3  # dx, dy, dz per step
        if self.use_gripper:
            action_dim += 1
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(plan_horizon, action_dim),
            dtype=np.float32
        )
        
        # Planning state
        self.current_plan = None
        self.plan_step = 0
        self.steps_since_replan = 0
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset with planning state reset."""
        obs, info = super().reset(**kwargs)
        
        self.current_plan = None
        self.plan_step = 0
        self.steps_since_replan = 0
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute multi-step plan.
        
        Args:
            action: Multi-step delta plan of shape (plan_horizon, action_dim)
        """
        # Accept new plan if it's time to replan
        if self.steps_since_replan == 0:
            self.current_plan = np.array(action, dtype=np.float32)
            self.plan_step = 0
            
            if self.interpolate_plan:
                self.current_plan = self._interpolate_plan(self.current_plan)
        
        # Execute current step of plan
        if self.current_plan is not None and self.plan_step < len(self.current_plan):
            current_action = self.current_plan[self.plan_step]
        else:
            # Fallback: stop moving
            current_action = np.zeros(self.action_space.shape[1])
        
        # Execute with base wrapper
        obs, reward, terminated, truncated, info = super().step(current_action)
        
        # Update planning state
        self.plan_step += 1
        self.steps_since_replan += 1
        
        if self.steps_since_replan >= self.replan_frequency:
            self.steps_since_replan = 0
        
        # Add planning info
        info['wrapper']['plan_step'] = self.plan_step
        info['wrapper']['steps_since_replan'] = self.steps_since_replan
        
        return obs, reward, terminated, truncated, info
    
    def _interpolate_plan(self, plan: np.ndarray) -> np.ndarray:
        """Smooth interpolation between plan waypoints."""
        from scipy.interpolate import interp1d
        
        n_steps = len(plan)
        if n_steps <= 2:
            return plan
        
        # Create smooth interpolation
        t_original = np.linspace(0, 1, n_steps)
        t_smooth = np.linspace(0, 1, n_steps)
        
        smoothed_plan = np.zeros_like(plan)
        for dim in range(plan.shape[1]):
            f = interp1d(t_original, plan[:, dim], kind='cubic', 
                        fill_value='extrapolate')
            smoothed_plan[:, dim] = f(t_smooth)
        
        return smoothed_plan


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Short-Horizon Delta Wrapper")
    print("=" * 60)
    
    # Create a dummy environment for testing
    class DummyRobotEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-2, high=2, shape=(10,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32
            )
            self.ee_pos = np.array([0.5, 0.0, 0.3])
            self.goal_pos = np.array([0.8, 0.2, 0.3])
            self.steps = 0
        
        def reset(self, **kwargs):
            self.ee_pos = np.array([0.5, 0.0, 0.3])
            self.steps = 0
            obs = np.concatenate([self.ee_pos, self.goal_pos, np.zeros(4)])
            info = {
                'ee_pos': self.ee_pos.copy(),
                'goal_pos': self.goal_pos.copy()
            }
            return obs, info
        
        def step(self, action):
            # Simple dynamics: move EE by action
            self.ee_pos += action[:3] * 0.01
            self.steps += 1
            
            obs = np.concatenate([self.ee_pos, self.goal_pos, np.zeros(4)])
            
            # Compute reward
            distance = np.linalg.norm(self.ee_pos - self.goal_pos)
            reward = -distance
            
            # Check termination
            terminated = distance < 0.05
            truncated = self.steps >= 200
            
            info = {
                'ee_pos': self.ee_pos.copy(),
                'goal_pos': self.goal_pos.copy(),
                'distance': distance,
                'success': terminated
            }
            
            return obs, reward, terminated, truncated, info
        
        def render(self):
            pass
        
        def close(self):
            pass
    
    # Test 1: Basic wrapper
    print("\n1. Testing Basic ShortHorizonDeltaWrapper")
    print("-" * 40)
    
    env = DummyRobotEnv()
    wrapped_env = ShortHorizonDeltaWrapper(
        env,
        action_repeat=10,
        delta_scale=0.05,
        horizon=50,
        verbose=True
    )
    
    obs, info = wrapped_env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {wrapped_env.action_space}")
    print(f"Initial EE position: {info['wrapper']['ee_pos']}")
    
    # Take a few steps
    for i in range(3):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  EE pos: {info['wrapper']['ee_pos']}")
        print(f"  Macro steps: {info['wrapper']['macro_steps']}")
        print(f"  Total steps: {info['wrapper']['total_steps']}")
    
    # Test 2: Adaptive wrapper
    print("\n\n2. Testing AdaptiveDeltaWrapper")
    print("-" * 40)
    
    env = DummyRobotEnv()
    adaptive_env = AdaptiveDeltaWrapper(
        env,
        initial_scale=0.05,
        min_scale=0.01,
        max_scale=0.1,
        action_repeat=5,
        horizon=30
    )
    
    obs, info = adaptive_env.reset()
    print(f"Initial delta scale: {adaptive_env.delta_scale:.3f}")
    
    # Simulate approaching goal
    for i in range(5):
        # Move toward goal
        action = np.array([0.3, 0.2, 0.0, 0.0])  # Move in x-y plane
        obs, reward, terminated, truncated, info = adaptive_env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Delta scale: {adaptive_env.delta_scale:.3f}")
        print(f"  Phase: {adaptive_env.phase}")
        print(f"  Distance to goal: {info.get('distance', -1):.3f}")
        
        if terminated:
            print("  Goal reached!")
            break
    
    # Test 3: Multi-step planner
    print("\n\n3. Testing MultiStepPlannerWrapper")
    print("-" * 40)
    
    env = DummyRobotEnv()
    planner_env = MultiStepPlannerWrapper(
        env,
        plan_horizon=5,
        replan_frequency=5,
        action_repeat=3,
        horizon=20
    )
    
    obs, info = planner_env.reset()
    print(f"Action space (multi-step): {planner_env.action_space}")
    
    # Create a multi-step plan
    plan = np.array([
        [0.2, 0.0, 0.0, 0.0],   # Move right
        [0.2, 0.1, 0.0, 0.0],   # Move right and forward
        [0.1, 0.2, 0.0, 0.0],   # Move more forward
        [0.0, 0.1, 0.0, 0.0],   # Move forward
        [0.0, 0.0, 0.0, 1.0],   # Close gripper
    ])
    
    print(f"\nExecuting plan with shape {plan.shape}")
    
    for i in range(3):
        obs, reward, terminated, truncated, info = planner_env.step(plan)
        print(f"\nExecution step {i+1}:")
        print(f"  Plan step: {info['wrapper']['plan_step']}")
        print(f"  Steps since replan: {info['wrapper']['steps_since_replan']}")
        print(f"  EE position: {info['wrapper']['ee_pos']}")
    
    print("\n" + "=" * 60)
    print("All wrapper tests completed!")
    print("=" * 60)