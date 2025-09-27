"""
Policy evaluation and rollout utilities.

This module provides functions for evaluating trained policies in environments,
recording videos/gifs, and computing performance metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
import logging
import time
from pathlib import Path
import imageio
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
import io

# Configure logging
logger = logging.getLogger(__name__)


def rollout_policy(
    policy: Union[nn.Module, Callable],
    env: Any,
    max_steps: int = 1000,
    action_repeat: int = 1,
    render: bool = True,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_format: str = 'gif',
    fps: int = 30,
    deterministic: bool = True,
    device: Optional[str] = None,
    verbose: bool = True,
    return_trajectory: bool = False,
    early_termination: bool = True
) -> Union[Tuple[bool, Optional[str]], Tuple[bool, Optional[str], Dict[str, Any]]]:
    """
    Rollout a policy in an environment and optionally record video.
    
    This function executes a trained policy in the environment for evaluation,
    optionally recording a video/gif of the execution.
    
    Args:
        policy: Trained policy (nn.Module or callable)
        env: Environment with gym-like interface
        max_steps: Maximum number of steps
        action_repeat: Number of times to repeat each action
        render: Whether to render the environment
        record_video: Whether to record video/gif
        video_path: Path to save video (auto-generated if None)
        video_format: Output format ('gif', 'mp4', 'avi', 'webm')
        fps: Frames per second for video
        deterministic: Whether to use deterministic policy actions
        device: Device for policy computation
        verbose: Whether to print progress
        return_trajectory: Whether to return full trajectory data
        early_termination: Whether to stop on done signal
        
    Returns:
        If return_trajectory=False:
            Tuple of (success_flag, video_path)
        If return_trajectory=True:
            Tuple of (success_flag, video_path, trajectory_dict)
            
    Example:
        # Simple evaluation
        success, video_path = rollout_policy(
            policy, env, 
            max_steps=500,
            record_video=True,
            video_format='gif'
        )
        
        # With trajectory data
        success, video_path, trajectory = rollout_policy(
            policy, env,
            return_trajectory=True
        )
    """
    # Setup device
    if device is None and isinstance(policy, nn.Module):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy = policy.to(device)
        policy.eval()
    
    # Initialize recording
    frames = []
    recording = record_video or render
    
    # Initialize trajectory storage
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'infos': [],
        'frames': []
    }
    
    # Reset environment
    try:
        state = env.reset()
        if hasattr(env, 'render') and recording:
            frame = env.render(mode='rgb_array' if record_video else 'human')
            if record_video and frame is not None:
                frames.append(frame)
    except Exception as e:
        logger.error(f"Failed to reset environment: {e}")
        return (False, None) if not return_trajectory else (False, None, trajectory)
    
    # Initialize episode variables
    total_reward = 0.0
    success = False
    done = False
    steps = 0
    
    if verbose:
        logger.info(f"Starting policy rollout for max {max_steps} steps")
    
    # Main rollout loop
    while steps < max_steps and (not done or not early_termination):
        # Store state
        trajectory['states'].append(_serialize_state(state))
        
        # Get action from policy
        try:
            action = _get_policy_action(
                policy, state, deterministic, device
            )
        except Exception as e:
            logger.error(f"Policy failed at step {steps}: {e}")
            break
        
        # Store action
        trajectory['actions'].append(_serialize_action(action))
        
        # Repeat action if specified
        step_reward = 0.0
        for _ in range(action_repeat):
            try:
                # Step environment
                next_state, reward, done, info = env.step(action)
                step_reward += reward
                
                # Render if needed
                if recording and hasattr(env, 'render'):
                    frame = env.render(mode='rgb_array' if record_video else 'human')
                    if record_video and frame is not None:
                        frames.append(frame)
                
                # Update state
                state = next_state
                
                # Check if we should stop
                if done and early_termination:
                    break
                    
            except Exception as e:
                logger.error(f"Environment step failed at step {steps}: {e}")
                done = True
                break
        
        # Accumulate reward and info
        total_reward += step_reward
        trajectory['rewards'].append(step_reward)
        trajectory['infos'].append(info)
        
        # Check for success
        if 'success' in info and info['success']:
            success = True
        
        # Increment step counter
        steps += 1
        
        # Progress logging
        if verbose and steps % 100 == 0:
            logger.info(f"Step {steps}/{max_steps}, Reward: {total_reward:.2f}")
        
        # Check termination
        if done and early_termination:
            break
    
    # Final state
    trajectory['states'].append(_serialize_state(state))
    
    # Determine success
    if not success:
        # Alternative success criteria
        if total_reward > 0 or (done and steps < max_steps):
            success = True
    
    # Save video if recorded
    video_file_path = None
    if record_video and frames:
        video_file_path = _save_video(
            frames, video_path, video_format, fps, verbose
        )
    
    # Summary
    if verbose:
        logger.info(f"Rollout complete: Steps={steps}, Success={success}, "
                   f"Total Reward={total_reward:.2f}")
        if video_file_path:
            logger.info(f"Video saved to: {video_file_path}")
    
    # Add summary to trajectory
    trajectory['summary'] = {
        'success': success,
        'total_reward': total_reward,
        'steps': steps,
        'done': done
    }
    
    if return_trajectory:
        return success, video_file_path, trajectory
    else:
        return success, video_file_path


def _get_policy_action(
    policy: Union[nn.Module, Callable],
    state: Any,
    deterministic: bool,
    device: Optional[str]
) -> Any:
    """
    Get action from policy.
    
    Args:
        policy: Policy model or callable
        state: Current state
        deterministic: Whether to use deterministic action
        device: Computation device
        
    Returns:
        Action to take
    """
    if isinstance(policy, nn.Module):
        # Neural network policy
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        else:
            state_tensor = torch.FloatTensor([state])
        
        # Move to device
        if device:
            state_tensor = state_tensor.to(device)
        
        # Get action
        with torch.no_grad():
            if hasattr(policy, 'get_action'):
                action = policy.get_action(state_tensor, deterministic=deterministic)
            else:
                action, _ = policy(state_tensor, deterministic=deterministic)
                action = action.squeeze(0)
        
        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
    elif callable(policy):
        # Function-based policy
        action = policy(state)
    else:
        raise ValueError(f"Unsupported policy type: {type(policy)}")
    
    return action


def _serialize_state(state: Any) -> Any:
    """Serialize state for storage."""
    if isinstance(state, np.ndarray):
        return state.copy()
    elif isinstance(state, torch.Tensor):
        return state.cpu().numpy()
    elif isinstance(state, dict):
        return {k: _serialize_state(v) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        return type(state)(_serialize_state(s) for s in state)
    else:
        return state


def _serialize_action(action: Any) -> Any:
    """Serialize action for storage."""
    if isinstance(action, np.ndarray):
        return action.copy()
    elif isinstance(action, torch.Tensor):
        return action.cpu().numpy()
    elif isinstance(action, dict):
        return {k: _serialize_action(v) for k, v in action.items()}
    elif isinstance(action, (list, tuple)):
        return type(action)(_serialize_action(a) for a in action)
    else:
        return action


def _save_video(
    frames: List[np.ndarray],
    video_path: Optional[str],
    video_format: str,
    fps: int,
    verbose: bool
) -> str:
    """
    Save frames as video file.
    
    Args:
        frames: List of RGB frames
        video_path: Output path
        video_format: Output format
        fps: Frames per second
        verbose: Whether to log progress
        
    Returns:
        Path to saved video
    """
    if not frames:
        return None
    
    # Generate path if not provided
    if video_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = f"rollout_{timestamp}.{video_format}"
    else:
        # Ensure correct extension
        path = Path(video_path)
        video_path = str(path.with_suffix(f'.{video_format}'))
    
    try:
        if video_format == 'gif':
            # Save as GIF using imageio
            imageio.mimsave(video_path, frames, fps=fps, loop=0)
            
        elif video_format == 'mp4':
            # Save as MP4 using imageio-ffmpeg
            writer = imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            
        elif video_format == 'webm':
            # Save as WebM
            writer = imageio.get_writer(video_path, fps=fps, codec='libvpx-vp9', quality=8)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            
        elif video_format == 'avi':
            # Save as AVI using OpenCV
            if len(frames) > 0:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                
                out.release()
        else:
            raise ValueError(f"Unsupported video format: {video_format}")
        
        if verbose:
            file_size = Path(video_path).stat().st_size / 1024 / 1024  # MB
            logger.info(f"Saved {len(frames)} frames to {video_path} ({file_size:.2f} MB)")
        
        return video_path
        
    except Exception as e:
        logger.error(f"Failed to save video: {e}")
        
        # Fallback to GIF if other format fails
        if video_format != 'gif':
            logger.info("Falling back to GIF format...")
            gif_path = str(Path(video_path).with_suffix('.gif'))
            try:
                imageio.mimsave(gif_path, frames, fps=fps, loop=0)
                return gif_path
            except Exception as e2:
                logger.error(f"GIF fallback also failed: {e2}")
        
        return None


def rollout_multiple(
    policy: Union[nn.Module, Callable],
    env: Any,
    n_rollouts: int = 10,
    max_steps: int = 1000,
    action_repeat: int = 1,
    record_best: bool = True,
    deterministic: bool = True,
    parallel: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform multiple rollouts and compute statistics.
    
    Args:
        policy: Policy to evaluate
        env: Environment
        n_rollouts: Number of rollouts
        max_steps: Maximum steps per rollout
        action_repeat: Action repeat factor
        record_best: Whether to record video of best rollout
        deterministic: Whether to use deterministic policy
        parallel: Whether to run rollouts in parallel
        verbose: Whether to print progress
        
    Returns:
        Dictionary with statistics and optional video path
        
    Example:
        results = rollout_multiple(
            policy, env,
            n_rollouts=20,
            record_best=True
        )
        print(f"Success rate: {results['success_rate']:.2%}")
    """
    successes = []
    rewards = []
    steps_list = []
    trajectories = []
    
    best_reward = -float('inf')
    best_trajectory = None
    best_idx = -1
    
    if verbose:
        logger.info(f"Starting {n_rollouts} rollouts")
    
    for i in range(n_rollouts):
        # Run rollout
        success, _, trajectory = rollout_policy(
            policy, env,
            max_steps=max_steps,
            action_repeat=action_repeat,
            render=False,
            record_video=False,
            deterministic=deterministic,
            verbose=False,
            return_trajectory=True
        )
        
        # Collect statistics
        successes.append(success)
        total_reward = trajectory['summary']['total_reward']
        rewards.append(total_reward)
        steps_list.append(trajectory['summary']['steps'])
        trajectories.append(trajectory)
        
        # Track best
        if total_reward > best_reward:
            best_reward = total_reward
            best_trajectory = trajectory
            best_idx = i
        
        if verbose and (i + 1) % 5 == 0:
            logger.info(f"Completed {i+1}/{n_rollouts} rollouts")
    
    # Compute statistics
    results = {
        'n_rollouts': n_rollouts,
        'success_rate': np.mean(successes),
        'success_count': sum(successes),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_steps': np.mean(steps_list),
        'std_steps': np.std(steps_list),
        'best_rollout_idx': best_idx,
        'best_reward': best_reward,
        'all_rewards': rewards,
        'all_successes': successes,
        'all_steps': steps_list
    }
    
    # Record best rollout if requested
    if record_best and best_trajectory is not None:
        if verbose:
            logger.info(f"Recording video of best rollout (reward: {best_reward:.2f})")
        
        # Replay best trajectory
        success, video_path = replay_trajectory(
            env, best_trajectory,
            record_video=True,
            video_format='gif',
            fps=30
        )
        results['best_video_path'] = video_path
    
    if verbose:
        logger.info(f"Rollout statistics:")
        logger.info(f"  Success rate: {results['success_rate']:.2%}")
        logger.info(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"  Mean steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    
    return results


def replay_trajectory(
    env: Any,
    trajectory: Dict[str, Any],
    record_video: bool = True,
    video_path: Optional[str] = None,
    video_format: str = 'gif',
    fps: int = 30,
    verbose: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Replay a recorded trajectory in the environment.
    
    Args:
        env: Environment
        trajectory: Recorded trajectory data
        record_video: Whether to record video
        video_path: Output video path
        video_format: Video format
        fps: Frames per second
        verbose: Whether to print progress
        
    Returns:
        Tuple of (success, video_path)
    """
    frames = []
    
    # Reset environment
    state = env.reset()
    
    if record_video and hasattr(env, 'render'):
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    
    # Replay actions
    actions = trajectory.get('actions', [])
    
    for i, action in enumerate(actions):
        try:
            state, reward, done, info = env.step(action)
            
            if record_video and hasattr(env, 'render'):
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
            
            if done:
                break
                
        except Exception as e:
            logger.error(f"Failed to replay action {i}: {e}")
            break
    
    # Save video
    video_file_path = None
    if record_video and frames:
        video_file_path = _save_video(
            frames, video_path, video_format, fps, verbose
        )
    
    success = trajectory.get('summary', {}).get('success', False)
    return success, video_file_path


def create_comparison_video(
    policies: List[Union[nn.Module, Callable]],
    env: Any,
    labels: Optional[List[str]] = None,
    max_steps: int = 500,
    video_path: str = 'comparison.gif',
    fps: int = 30,
    side_by_side: bool = True
) -> str:
    """
    Create comparison video of multiple policies.
    
    Args:
        policies: List of policies to compare
        env: Environment
        labels: Labels for each policy
        max_steps: Maximum steps per rollout
        video_path: Output video path
        fps: Frames per second
        side_by_side: Whether to show side-by-side (vs sequential)
        
    Returns:
        Path to saved video
        
    Example:
        video_path = create_comparison_video(
            [expert_policy, learned_policy],
            env,
            labels=['Expert', 'Learned'],
            side_by_side=True
        )
    """
    n_policies = len(policies)
    if labels is None:
        labels = [f'Policy {i+1}' for i in range(n_policies)]
    
    all_frames = []
    
    for i, policy in enumerate(policies):
        logger.info(f"Recording {labels[i]}...")
        
        # Rollout policy
        success, _, trajectory = rollout_policy(
            policy, env,
            max_steps=max_steps,
            render=True,
            record_video=True,
            verbose=False,
            return_trajectory=True
        )
        
        frames = trajectory.get('frames', [])
        
        # Add label to frames
        labeled_frames = []
        for frame in frames:
            labeled_frame = _add_text_to_frame(
                frame, labels[i], 
                position='top',
                color=(255, 255, 0)
            )
            labeled_frames.append(labeled_frame)
        
        all_frames.append(labeled_frames)
    
    # Combine frames
    if side_by_side:
        combined_frames = _combine_frames_side_by_side(all_frames)
    else:
        combined_frames = _combine_frames_sequential(all_frames)
    
    # Save video
    video_file_path = _save_video(
        combined_frames, video_path, 'gif', fps, True
    )
    
    return video_file_path


def _add_text_to_frame(
    frame: np.ndarray,
    text: str,
    position: str = 'top',
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.7
) -> np.ndarray:
    """Add text overlay to frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Position text
    if position == 'top':
        org = (10, 30)
    elif position == 'bottom':
        org = (10, h - 10)
    else:
        org = (10, 30)
    
    # Add text with OpenCV
    cv2.putText(
        frame, text, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, 2, cv2.LINE_AA
    )
    
    return frame


def _combine_frames_side_by_side(
    frame_lists: List[List[np.ndarray]]
) -> List[np.ndarray]:
    """Combine multiple frame sequences side by side."""
    max_frames = max(len(frames) for frames in frame_lists)
    combined = []
    
    for i in range(max_frames):
        row_frames = []
        for frames in frame_lists:
            if i < len(frames):
                row_frames.append(frames[i])
            else:
                # Use last frame if this sequence is shorter
                row_frames.append(frames[-1] if frames else np.zeros_like(frame_lists[0][0]))
        
        # Concatenate horizontally
        combined_frame = np.hstack(row_frames)
        combined.append(combined_frame)
    
    return combined


def _combine_frames_sequential(
    frame_lists: List[List[np.ndarray]]
) -> List[np.ndarray]:
    """Combine multiple frame sequences sequentially."""
    combined = []
    for frames in frame_lists:
        combined.extend(frames)
    return combined


def visualize_rollout(
    trajectory: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize rollout trajectory with plots.
    
    Args:
        trajectory: Trajectory data from rollout
        figsize: Figure size
        save_path: Optional path to save figure
    """
    rewards = trajectory.get('rewards', [])
    states = trajectory.get('states', [])
    actions = trajectory.get('actions', [])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Rewards per Step')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Plot cumulative reward
    cumulative_rewards = np.cumsum(rewards)
    axes[0, 1].plot(cumulative_rewards)
    axes[0, 1].set_title('Cumulative Reward')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(True)
    
    # Plot state trajectories (first few dimensions)
    if states and len(states) > 0:
        states_array = np.array(states)
        if states_array.ndim > 1:
            n_dims = min(3, states_array.shape[1])
            for i in range(n_dims):
                axes[1, 0].plot(states_array[:, i], label=f'State dim {i}')
            axes[1, 0].set_title('State Trajectory')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
    
    # Plot action magnitudes
    if actions and len(actions) > 0:
        actions_array = np.array(actions)
        if actions_array.ndim > 1:
            action_norms = np.linalg.norm(actions_array, axis=1)
        else:
            action_norms = np.abs(actions_array)
        
        axes[1, 1].plot(action_norms)
        axes[1, 1].set_title('Action Magnitude')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create dummy environment for testing
    class DummyEnv:
        def __init__(self):
            self.state_dim = 4
            self.action_dim = 2
            self.step_count = 0
            self.max_steps = 100
            
        def reset(self):
            self.step_count = 0
            return np.random.randn(self.state_dim)
        
        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(self.state_dim)
            reward = -np.linalg.norm(action) + np.random.normal(0, 0.1)
            done = self.step_count >= self.max_steps
            info = {'success': done and reward > -1}
            return next_state, reward, done, info
        
        def render(self, mode='rgb_array'):
            # Create simple visualization
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            # Add step counter
            img[10:30, 10:50] = 255
            return img
    
    # Create dummy policy
    class DummyPolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim)
            )
        
        def forward(self, state, deterministic=True):
            action = self.net(state)
            return action, {}
        
        def get_action(self, state, deterministic=True):
            action, _ = self.forward(state, deterministic)
            return action.squeeze(0)
    
    # Test rollout_policy
    print("=" * 60)
    print("Testing rollout_policy")
    print("=" * 60)
    
    env = DummyEnv()
    policy = DummyPolicy(env.state_dim, env.action_dim)
    
    # Single rollout with video
    success, video_path = rollout_policy(
        policy, env,
        max_steps=50,
        action_repeat=1,
        record_video=True,
        video_format='gif',
        verbose=True
    )
    
    print(f"\nRollout success: {success}")
    print(f"Video saved to: {video_path}")
    
    # Test multiple rollouts
    print("\n" + "=" * 60)
    print("Testing rollout_multiple")
    print("=" * 60)
    
    results = rollout_multiple(
        policy, env,
        n_rollouts=10,
        max_steps=50,
        record_best=True,
        verbose=True
    )
    
    print(f"\nSuccess rate: {results['success_rate']:.2%}")
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    # Test with trajectory return
    print("\n" + "=" * 60)
    print("Testing with trajectory return")
    print("=" * 60)
    
    success, video_path, trajectory = rollout_policy(
        policy, env,
        max_steps=30,
        return_trajectory=True,
        record_video=False,
        verbose=True
    )
    
    print(f"\nTrajectory summary:")
    print(f"  States collected: {len(trajectory['states'])}")
    print(f"  Actions taken: {len(trajectory['actions'])}")
    print(f"  Total reward: {trajectory['summary']['total_reward']:.2f}")
    
    # Visualize trajectory
    if len(trajectory['rewards']) > 0:
        visualize_rollout(trajectory, save_path='trajectory_plot.png')
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)