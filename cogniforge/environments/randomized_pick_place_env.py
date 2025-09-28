"""
Randomized Pick and Place Environment with configurable cube spawn randomization.

This environment provides a pick-and-place task with randomized cube initial positions
to improve robustness and generalization of learned policies.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from cogniforge.core.simulator import RobotSimulator, RobotType, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class RandomizedEnvConfig:
    """Configuration for randomized pick-and-place environment."""
    
    # Cube randomization parameters
    cube_offset_range: float = 0.02  # ±2cm randomization
    cube_base_position: Tuple[float, float, float] = (0.5, 0.0, 0.05)
    cube_size: float = 0.03
    cube_color: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # Blue
    randomize_xy_only: bool = True  # Only randomize X,Y, not Z
    
    # Platform parameters
    platform_position: Tuple[float, float, float] = (0.3, 0.3, 0.01)
    platform_size: Tuple[float, float, float] = (0.1, 0.1, 0.02)
    platform_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
    
    # Table parameters
    table_position: Tuple[float, float, float] = (0.6, 0.0, 0.0)
    table_height: float = 0.4
    table_size: float = 0.3
    
    # Environment parameters
    max_episode_steps: int = 200
    reward_shaping: bool = True
    success_reward: float = 10.0
    grasp_reward: float = 2.0
    distance_penalty_weight: float = 0.1
    
    # Observation parameters
    include_cube_position: bool = True
    include_target_position: bool = True
    include_gripper_state: bool = True
    
    # Action parameters
    action_type: str = "delta"  # "delta" or "absolute"
    delta_scale: float = 0.05
    
    # Random seed
    seed: Optional[int] = None


class RandomizedPickPlaceEnv(gym.Env):
    """
    Gymnasium-compatible environment for pick-and-place with randomized cube spawning.
    
    The cube's initial position is randomized by ±2cm (configurable) at each reset
    to improve policy robustness and generalization.
    
    Observation Space:
        - Robot joint positions (7D)
        - End-effector position (3D)
        - Gripper state (1D)
        - Cube position (3D) - if enabled
        - Target position (3D) - if enabled
        
    Action Space:
        - Delta end-effector position (3D)
        - Gripper command (1D): 0=close, 1=open
    
    Rewards:
        - Distance-based shaping reward
        - Grasp success reward
        - Task completion reward
    """
    
    def __init__(
        self,
        config: Optional[RandomizedEnvConfig] = None,
        render_mode: Optional[str] = None,
        force_gui: bool = False
    ):
        """
        Initialize the randomized pick-and-place environment.
        
        Args:
            config: Environment configuration. Uses defaults if None.
            render_mode: Rendering mode ('human' for GUI, None for headless).
            force_gui: Force GUI mode regardless of render_mode.
        """
        super().__init__()
        
        self.config = config or RandomizedEnvConfig()
        self.render_mode = render_mode
        
        # Setup simulator
        sim_config = SimulationConfig(
            gravity=(0.0, 0.0, -9.81),
            time_step=1.0 / 240.0,
            use_real_time=False,
            seed=self.config.seed
        )
        
        self.simulator = RobotSimulator(config=sim_config)
        self._is_connected = False
        
        # Environment state
        self.robot_name = "robot"
        self.cube_id = None
        self.platform_id = None
        self.table_id = None
        self.current_cube_position = None
        self.randomized_cube_offset = None
        
        # Episode tracking
        self.steps = 0
        self.episode_reward = 0.0
        self.object_grasped = False
        self.task_completed = False
        
        # Random number generator
        self.rng = np.random.RandomState(self.config.seed)
        
        # Define observation space
        obs_dim = self._calculate_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space (3D position delta + gripper)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        logger.info(f"RandomizedPickPlaceEnv initialized with ±{self.config.cube_offset_range*100:.1f}cm randomization")
    
    def _calculate_obs_dim(self) -> int:
        """Calculate observation space dimension based on configuration."""
        dim = 7 + 3 + 1  # Joint positions + EE position + gripper
        if self.config.include_cube_position:
            dim += 3
        if self.config.include_target_position:
            dim += 3
        return dim
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment with randomized cube position.
        
        The cube position is randomized by ±2cm (or configured range) from the base position.
        
        Args:
            seed: Random seed for this episode.
            options: Additional reset options.
            
        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        
        # Set random seed if provided
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Connect to simulator if needed
        if not self._is_connected:
            self.simulator.connect()
            self._is_connected = True
            
            # Load robot
            robot_info = self.simulator.load_robot(
                robot_type=RobotType.KUKA_IIWA,
                position=(0, 0, 0),
                fixed_base=True,
                robot_name=self.robot_name
            )
            logger.info(f"Robot loaded: {robot_info.name}")
        
        # Reset simulator state
        self.simulator.reset_simulation()
        
        # Spawn scene objects
        self._spawn_scene()
        
        # Reset robot to home position
        home_joints = [0.0, 0.4, 0.0, -1.5, 0.0, 1.0, 0.0]
        self.simulator.set_robot_joints(self.robot_name, home_joints, use_physics=False)
        
        # Open gripper
        self.simulator.open_gripper(self.robot_name)
        
        # Reset episode state
        self.steps = 0
        self.episode_reward = 0.0
        self.object_grasped = False
        self.task_completed = False
        
        # Run simulation for a few steps to stabilize
        for _ in range(50):
            self.simulator.step()
        
        # Get initial observation
        obs = self._get_observation()
        
        # Create info dict
        info = {
            "cube_position": self.current_cube_position,
            "cube_offset": self.randomized_cube_offset,
            "platform_position": list(self.config.platform_position),
            "randomization_range": self.config.cube_offset_range
        }
        
        logger.info(f"Environment reset with cube offset: {self.randomized_cube_offset}")
        
        return obs, info
    
    def _spawn_scene(self):
        """Spawn scene objects with randomized cube position."""
        
        # Spawn table
        self.table_id = self.simulator.spawn_table(
            position=self.config.table_position,
            table_height=self.config.table_height,
            table_size=self.config.table_size
        )
        
        # Generate random offset for cube position (±2cm by default)
        if self.config.randomize_xy_only:
            # Only randomize X and Y coordinates
            offset_x = self.rng.uniform(
                -self.config.cube_offset_range, 
                self.config.cube_offset_range
            )
            offset_y = self.rng.uniform(
                -self.config.cube_offset_range, 
                self.config.cube_offset_range
            )
            offset_z = 0.0
        else:
            # Randomize all three coordinates
            offset_x = self.rng.uniform(
                -self.config.cube_offset_range, 
                self.config.cube_offset_range
            )
            offset_y = self.rng.uniform(
                -self.config.cube_offset_range, 
                self.config.cube_offset_range
            )
            offset_z = self.rng.uniform(
                -self.config.cube_offset_range/2,  # Less Z randomization
                self.config.cube_offset_range/2
            )
        
        self.randomized_cube_offset = [offset_x, offset_y, offset_z]
        
        # Calculate actual cube spawn position
        base_pos = self.config.cube_base_position
        self.current_cube_position = [
            base_pos[0] + offset_x,
            base_pos[1] + offset_y,
            base_pos[2] + offset_z
        ]
        
        # Spawn cube at randomized position
        self.cube_id = self.simulator.spawn_block(
            color_rgb=self.config.cube_color,
            size=self.config.cube_size,
            position=tuple(self.current_cube_position),
            mass=0.05,
            block_name="target_cube"
        )
        
        logger.info(
            f"Cube spawned at {self.current_cube_position} "
            f"(offset: [{offset_x:.3f}, {offset_y:.3f}, {offset_z:.3f}])"
        )
        
        # Spawn target platform
        self.platform_id = self.simulator.spawn_platform(
            color_rgb=self.config.platform_color,
            size=self.config.platform_size[0],
            position=self.config.platform_position,
            platform_name="target_platform",
            height=self.config.platform_size[2]
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []
        
        # Get robot state
        robot_state = self.simulator.get_robot_state(self.robot_name)
        
        # Joint positions (7D for KUKA)
        joint_positions = robot_state["joint_positions"][:7]
        obs.extend(joint_positions)
        
        # End-effector position (3D)
        ee_pos, _ = self.simulator.ee_pose(self.robot_name)
        obs.extend(ee_pos)
        
        # Gripper state (1D) - simplified as open/closed
        gripper_info = self.simulator.get_gripper_state(self.robot_name)
        gripper_opening = gripper_info.get("current_opening", 0.0)
        obs.append(gripper_opening)
        
        # Cube position (3D) if enabled
        if self.config.include_cube_position:
            cube_state = self.simulator.get_object_state("target_cube")
            cube_pos = cube_state["position"]
            obs.extend(cube_pos)
        
        # Target platform position (3D) if enabled
        if self.config.include_target_position:
            obs.extend(self.config.platform_position)
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: [dx, dy, dz, gripper] where dx,dy,dz are delta positions
                   and gripper is 0=close, 1=open.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Parse action
        delta_pos = action[:3] * self.config.delta_scale
        gripper_cmd = action[3]
        
        # Get current end-effector position
        ee_pos, ee_orn = self.simulator.ee_pose(self.robot_name)
        
        # Calculate target position
        target_pos = [
            ee_pos[0] + delta_pos[0],
            ee_pos[1] + delta_pos[1],
            ee_pos[2] + delta_pos[2]
        ]
        
        # Execute movement (simplified - in practice would use IK)
        # For now, just set a target position
        success = self._move_to_position(target_pos)
        
        # Set gripper
        if gripper_cmd > 0.5:
            self.simulator.open_gripper(self.robot_name)
        else:
            self.simulator.close_gripper(self.robot_name)
        
        # Step simulation
        for _ in range(10):  # Multiple physics steps per action
            self.simulator.step()
        
        # Update step count
        self.steps += 1
        
        # Check grasp status
        if not self.object_grasped and gripper_cmd <= 0.5:
            self.object_grasped = self._check_grasp()
        
        # Check task completion
        if self.object_grasped:
            self.task_completed = self._check_placement()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check termination
        terminated = self.task_completed
        truncated = self.steps >= self.config.max_episode_steps
        
        # Get observation
        obs = self._get_observation()
        
        # Create info dict
        info = {
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "object_grasped": self.object_grasped,
            "task_completed": self.task_completed,
            "cube_position": self.simulator.get_object_state("target_cube")["position"],
            "ee_position": ee_pos
        }
        
        return obs, reward, terminated, truncated, info
    
    def _move_to_position(self, target_pos: List[float]) -> bool:
        """
        Move end-effector to target position (simplified).
        
        In practice, this would use the IK controller.
        For now, we'll use a simple position-based approach.
        """
        # This is a placeholder - actual implementation would use IK
        # For demonstration, we'll just report the target
        logger.debug(f"Moving to position: {target_pos}")
        return True
    
    def _check_grasp(self) -> bool:
        """Check if cube is successfully grasped."""
        # Get positions
        ee_pos, _ = self.simulator.ee_pose(self.robot_name)
        cube_state = self.simulator.get_object_state("target_cube")
        cube_pos = cube_state["position"]
        
        # Check if cube is close to gripper
        distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        
        # Check gripper state
        gripper_info = self.simulator.get_gripper_state(self.robot_name)
        gripper_closed = gripper_info.get("current_opening", 1.0) < 0.02
        
        # Consider grasped if close and gripper closed
        grasped = distance < 0.05 and gripper_closed
        
        if grasped:
            logger.info("Object successfully grasped!")
        
        return grasped
    
    def _check_placement(self) -> bool:
        """Check if cube is successfully placed on target platform."""
        cube_state = self.simulator.get_object_state("target_cube")
        cube_pos = cube_state["position"]
        
        platform_pos = self.config.platform_position
        
        # Check if cube is above platform
        dx = abs(cube_pos[0] - platform_pos[0])
        dy = abs(cube_pos[1] - platform_pos[1])
        dz = cube_pos[2] - platform_pos[2]
        
        # Success if cube is on platform (within tolerance)
        on_platform = dx < 0.05 and dy < 0.05 and 0.02 < dz < 0.1
        
        if on_platform:
            logger.info("Task completed! Cube placed on platform.")
        
        return on_platform
    
    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        reward = 0.0
        
        # Task completion reward
        if self.task_completed:
            reward += self.config.success_reward
            return reward
        
        # Grasp reward
        if self.object_grasped and not hasattr(self, '_grasp_rewarded'):
            reward += self.config.grasp_reward
            self._grasp_rewarded = True
        
        # Distance-based shaping (if enabled)
        if self.config.reward_shaping:
            ee_pos, _ = self.simulator.ee_pose(self.robot_name)
            
            if not self.object_grasped:
                # Distance to cube
                cube_state = self.simulator.get_object_state("target_cube")
                cube_pos = cube_state["position"]
                distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
            else:
                # Distance to platform
                platform_pos = self.config.platform_position
                distance = np.linalg.norm(np.array(ee_pos) - np.array(platform_pos))
            
            # Negative distance as penalty
            reward -= self.config.distance_penalty_weight * distance
        
        return reward
    
    def render(self):
        """Render environment (if in GUI mode)."""
        if self.render_mode == "human":
            # GUI rendering is handled by PyBullet automatically
            pass
        return None
    
    def close(self):
        """Clean up environment."""
        if self._is_connected:
            self.simulator.disconnect()
            self._is_connected = False
        logger.info("Environment closed")
    
    def get_cube_randomization_info(self) -> Dict[str, Any]:
        """
        Get information about current cube randomization.
        
        Returns:
            Dictionary with randomization details.
        """
        return {
            "base_position": list(self.config.cube_base_position),
            "current_position": self.current_cube_position,
            "offset_applied": self.randomized_cube_offset,
            "offset_range": self.config.cube_offset_range,
            "randomize_xy_only": self.config.randomize_xy_only
        }


def create_randomized_env(
    cube_offset_cm: float = 2.0,
    seed: Optional[int] = None,
    render: bool = False
) -> RandomizedPickPlaceEnv:
    """
    Convenience function to create a randomized environment.
    
    Args:
        cube_offset_cm: Randomization range in centimeters (default ±2cm).
        seed: Random seed for reproducibility.
        render: Whether to enable GUI rendering.
    
    Returns:
        Configured RandomizedPickPlaceEnv instance.
    
    Example:
        >>> env = create_randomized_env(cube_offset_cm=2.0, seed=42)
        >>> obs, info = env.reset()
        >>> print(f"Cube spawned at: {info['cube_position']}")
        >>> print(f"Offset applied: {info['cube_offset']}")
    """
    config = RandomizedEnvConfig(
        cube_offset_range=cube_offset_cm / 100.0,  # Convert cm to meters
        seed=seed
    )
    
    render_mode = "human" if render else None
    
    env = RandomizedPickPlaceEnv(
        config=config,
        render_mode=render_mode
    )
    
    logger.info(f"Created randomized environment with ±{cube_offset_cm}cm cube randomization")
    
    return env


if __name__ == "__main__":
    """Demonstration of randomized pick-and-place environment."""
    
    import matplotlib.pyplot as plt
    
    # Create environment with ±2cm randomization
    print("Creating randomized pick-and-place environment...")
    env = create_randomized_env(cube_offset_cm=2.0, seed=None, render=False)
    
    # Collect cube positions over multiple resets
    positions = []
    offsets = []
    
    print("\nResetting environment 10 times to show randomization:")
    print("-" * 50)
    
    for i in range(10):
        obs, info = env.reset(seed=i)
        
        cube_pos = info["cube_position"]
        cube_offset = info["cube_offset"]
        
        positions.append(cube_pos[:2])  # Store X,Y positions
        offsets.append(cube_offset[:2])
        
        print(f"Reset {i+1:2d}: Cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}), "
              f"Offset: ({cube_offset[0]:.3f}, {cube_offset[1]:.3f})")
    
    # Visualize randomization pattern
    print("\nGenerating visualization...")
    
    positions = np.array(positions)
    offsets = np.array(offsets)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Cube positions in XY plane
    ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, alpha=0.6, label='Cube positions')
    ax1.scatter([0.5], [0.0], c='red', s=200, marker='x', linewidth=2, label='Base position')
    
    # Add 2cm boundary box
    rect = plt.Rectangle((0.48, -0.02), 0.04, 0.04, 
                         fill=False, edgecolor='gray', linestyle='--', linewidth=1)
    ax1.add_patch(rect)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Randomized Cube Spawn Positions (Top View)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # Plot 2: Distribution of offsets
    ax2.hist2d(offsets[:, 0], offsets[:, 1], bins=5, cmap='Blues')
    ax2.set_xlabel('X Offset (m)')
    ax2.set_ylabel('Y Offset (m)')
    ax2.set_title('Offset Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Add ±2cm boundaries
    ax2.axhline(y=0.02, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.02, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.02, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=-0.02, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle('Cube Position Randomization (±2cm)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('cube_randomization.png', dpi=100, bbox_inches='tight')
    print("Visualization saved to 'cube_randomization.png'")
    
    # Clean up
    env.close()
    
    print("\n✅ Randomization demonstration complete!")
    print(f"   - Cube base position: (0.5, 0.0, 0.05)")
    print(f"   - Randomization range: ±2cm in X and Y")
    print(f"   - Z-coordinate: Fixed (no randomization by default)")