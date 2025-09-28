"""
Reset System with Hotkey and API Support

Provides quick scene reset functionality via:
- Keyboard hotkeys (R for reset, S for reseed)
- REST API endpoint (/reset)
- Python API for programmatic control
"""

import pybullet as p
import numpy as np
import time
import threading
import random
import keyboard
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import queue
from pathlib import Path
import logging
import platform
import psutil
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Reset Configuration
# ============================================================================

class ResetMode(Enum):
    """Reset modes available"""
    SOFT = "soft"       # Reset positions only
    HARD = "hard"       # Recreate all objects
    RESEED = "reseed"   # New random seed + reset
    CUSTOM = "custom"   # User-defined reset


@dataclass
class SceneConfig:
    """Scene configuration for reset"""
    seed: Optional[int] = None
    
    # Robot config
    robot_base: List[float] = None
    robot_joints: List[float] = None
    
    # Object positions (randomization ranges)
    cube_pos_range: Dict[str, Tuple[float, float]] = None
    platform_pos_range: Dict[str, Tuple[float, float]] = None
    obstacle_count_range: Tuple[int, int] = None
    
    # Object properties
    cube_size_range: Tuple[float, float] = None
    cube_mass_range: Tuple[float, float] = None
    cube_friction_range: Tuple[float, float] = None
    num_cubes: int = 2  # Number of cubes to create
    
    # Colors
    cube_colors: List[List[float]] = None
    platform_colors: List[List[float]] = None
    
    def __post_init__(self):
        """Set defaults if not provided"""
        if self.robot_base is None:
            self.robot_base = [0, 0, 0]
            
        if self.robot_joints is None:
            # Panda default home position
            self.robot_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
            
        if self.cube_pos_range is None:
            self.cube_pos_range = {
                'x': (0.3, 0.6),
                'y': (-0.2, 0.2),
                'z': (0.02, 0.02)
            }
            
        if self.platform_pos_range is None:
            self.platform_pos_range = {
                'x': (0.2, 0.4),
                'y': (0.2, 0.4),
                'z': (0.01, 0.01)
            }
            
        if self.obstacle_count_range is None:
            self.obstacle_count_range = (0, 3)
            
        if self.cube_size_range is None:
            self.cube_size_range = (0.03, 0.06)
            
        if self.cube_mass_range is None:
            self.cube_mass_range = (0.05, 0.2)
            
        if self.cube_friction_range is None:
            self.cube_friction_range = (0.3, 0.8)
            
        if self.cube_colors is None:
            self.cube_colors = [
                [1, 0, 0, 1],    # Red
                [0, 1, 0, 1],    # Green
                [0, 0, 1, 1],    # Blue
                [1, 1, 0, 1],    # Yellow
                [1, 0, 1, 1],    # Magenta
                [0, 1, 1, 1],    # Cyan
            ]
            
        if self.platform_colors is None:
            self.platform_colors = [
                [0.2, 0.8, 0.2, 1],  # Green
                [0.8, 0.2, 0.2, 1],  # Red
                [0.2, 0.2, 0.8, 1],  # Blue
            ]


# ============================================================================
# Scene Manager
# ============================================================================

class SceneManager:
    """Manages scene state and reset operations"""
    
    def __init__(self, physics_client: Optional[int] = None):
        """
        Initialize scene manager
        
        Args:
            physics_client: PyBullet physics client ID
        """
        self.client = physics_client
        self.config = SceneConfig()
        self.current_seed = None
        
        # Object tracking
        self.robot_id = None
        self.objects = {}  # name -> id mapping
        self.initial_states = {}  # Store initial states for soft reset
        
        # Reset callbacks
        self.pre_reset_callbacks = []
        self.post_reset_callbacks = []
        
        # Statistics
        self.reset_count = 0
        self.last_reset_time = 0
        self.reset_history = []
        
    def set_seed(self, seed: Optional[int] = None):
        """
        Set random seed for reproducibility
        
        Args:
            seed: Random seed (None for random)
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        self.current_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Seed set to: {seed}")
        return seed
        
    def save_initial_state(self):
        """Save initial state of all objects for soft reset"""
        self.initial_states = {}
        
        # Save robot state
        if self.robot_id is not None:
            joint_states = []
            for i in range(p.getNumJoints(self.robot_id)):
                state = p.getJointState(self.robot_id, i)
                joint_states.append(state[0])  # Position
            self.initial_states['robot'] = joint_states
            
        # Save object states
        for name, obj_id in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            lin_vel, ang_vel = p.getBaseVelocity(obj_id)
            self.initial_states[name] = {
                'position': pos,
                'orientation': orn,
                'linear_velocity': lin_vel,
                'angular_velocity': ang_vel
            }
            
    def soft_reset(self):
        """Soft reset: restore positions and velocities"""
        logger.info("Performing soft reset...")
        
        # Reset robot
        if self.robot_id is not None and 'robot' in self.initial_states:
            for i, pos in enumerate(self.initial_states['robot']):
                p.resetJointState(self.robot_id, i, pos, 0)
                
        # Reset objects
        for name, obj_id in self.objects.items():
            if name in self.initial_states:
                state = self.initial_states[name]
                p.resetBasePositionAndOrientation(
                    obj_id,
                    state['position'],
                    state['orientation']
                )
                p.resetBaseVelocity(
                    obj_id,
                    state['linear_velocity'],
                    state['angular_velocity']
                )
                
    def hard_reset(self):
        """Hard reset: remove and recreate all objects"""
        logger.info("Performing hard reset...")
        
        # Remove all objects
        for obj_id in self.objects.values():
            p.removeBody(obj_id)
        self.objects.clear()
        
        # Recreate scene
        self.create_scene()
        
    def reseed_reset(self):
        """Reseed and reset with new randomization"""
        logger.info("Performing reseed reset...")
        
        # Generate new seed
        new_seed = self.set_seed(None)
        
        # Hard reset with new positions
        self.hard_reset()
        
        return new_seed
        
    def create_scene(self):
        """Create/recreate the scene with current config"""
        
        # Create ground plane
        if 'plane' not in self.objects:
            plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
            self.objects['plane'] = plane_id
            
        # Create robot
        if self.robot_id is None:
            self.robot_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=self.config.robot_base,
                useFixedBase=True
            )
            
            # Set initial joint positions
            for i, pos in enumerate(self.config.robot_joints):
                p.resetJointState(self.robot_id, i, pos)
                
        # Create multiple colored cubes
        num_cubes = self.config.num_cubes if hasattr(self.config, 'num_cubes') else 2
        
        # Define specific colors for first two cubes (red and blue)
        primary_colors = [
            [1, 0, 0, 1],    # Red
            [0, 0, 1, 1],    # Blue
        ]
        
        for i in range(num_cubes):
            cube_size = np.random.uniform(*self.config.cube_size_range)
            
            # Position cubes side by side
            if i == 0:  # Red cube
                cube_pos = [
                    np.random.uniform(*self.config.cube_pos_range['x']),
                    np.random.uniform(*self.config.cube_pos_range['y']) - 0.1,
                    self.config.cube_pos_range['z'][0] + cube_size/2
                ]
                cube_color = primary_colors[0]  # Red
                cube_name = 'cube_red'
            elif i == 1:  # Blue cube
                cube_pos = [
                    np.random.uniform(*self.config.cube_pos_range['x']),
                    np.random.uniform(*self.config.cube_pos_range['y']) + 0.1,
                    self.config.cube_pos_range['z'][0] + cube_size/2
                ]
                cube_color = primary_colors[1]  # Blue
                cube_name = 'cube_blue'
            else:
                # Additional random cubes
                cube_pos = [
                    np.random.uniform(*self.config.cube_pos_range['x']),
                    np.random.uniform(*self.config.cube_pos_range['y']),
                    self.config.cube_pos_range['z'][0] + cube_size/2
                ]
                cube_color = random.choice(self.config.cube_colors)
                cube_name = f'cube_{i}'
            
            cube_mass = np.random.uniform(*self.config.cube_mass_range)
            
            cube_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[cube_size/2]*3,
                rgbaColor=cube_color
            )
            cube_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[cube_size/2]*3
            )
            
            cube_id = p.createMultiBody(
                baseMass=cube_mass,
                baseCollisionShapeIndex=cube_collision,
                baseVisualShapeIndex=cube_visual,
                basePosition=cube_pos
            )
            
            # Set friction
            friction = np.random.uniform(*self.config.cube_friction_range)
            p.changeDynamics(cube_id, -1, lateralFriction=friction)
            
            self.objects[cube_name] = cube_id
            
        # Keep backward compatibility
        if 'cube_red' in self.objects:
            self.objects['cube'] = self.objects['cube_red']
        
        # Create platform
        platform_pos = [
            np.random.uniform(*self.config.platform_pos_range['x']),
            np.random.uniform(*self.config.platform_pos_range['y']),
            self.config.platform_pos_range['z'][0]
        ]
        platform_color = random.choice(self.config.platform_colors)
        
        platform_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.08, 0.08, 0.01],
            rgbaColor=platform_color
        )
        platform_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.08, 0.08, 0.01]
        )
        
        platform_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=platform_pos
        )
        
        self.objects['platform'] = platform_id
        
        # Create random obstacles
        num_obstacles = np.random.randint(*self.config.obstacle_count_range)
        for i in range(num_obstacles):
            self._create_random_obstacle(i)
            
        # Save initial state
        self.save_initial_state()
        
        logger.info(f"Scene created with {len(self.objects)} objects")
        
    def _create_random_obstacle(self, index: int):
        """Create a random obstacle"""
        
        # Random size and position
        size = np.random.uniform(0.02, 0.04)
        pos = [
            np.random.uniform(0.2, 0.6),
            np.random.uniform(-0.3, 0.3),
            size/2
        ]
        
        # Random color (darker)
        color = [np.random.uniform(0.2, 0.5) for _ in range(3)] + [1]
        
        obstacle_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size/2]*3,
            rgbaColor=color
        )
        obstacle_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size/2]*3
        )
        
        obstacle_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=obstacle_collision,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=pos
        )
        
        self.objects[f'obstacle_{index}'] = obstacle_id
        
    def reset(self, mode: ResetMode = ResetMode.SOFT, **kwargs):
        """
        Main reset function
        
        Args:
            mode: Reset mode
            **kwargs: Additional parameters for custom reset
            
        Returns:
            Reset information dict
        """
        start_time = time.time()
        
        # Run pre-reset callbacks
        for callback in self.pre_reset_callbacks:
            callback()
            
        # Execute reset based on mode
        if mode == ResetMode.SOFT:
            self.soft_reset()
        elif mode == ResetMode.HARD:
            self.hard_reset()
        elif mode == ResetMode.RESEED:
            new_seed = self.reseed_reset()
            kwargs['new_seed'] = new_seed
        elif mode == ResetMode.CUSTOM:
            self.custom_reset(**kwargs)
            
        # Update statistics
        reset_time = time.time() - start_time
        self.reset_count += 1
        self.last_reset_time = reset_time
        
        reset_info = {
            'mode': mode.value,
            'reset_count': self.reset_count,
            'reset_time': reset_time,
            'seed': self.current_seed,
            'num_objects': len(self.objects),
            **kwargs
        }
        
        self.reset_history.append(reset_info)
        
        # Run post-reset callbacks
        for callback in self.post_reset_callbacks:
            callback(reset_info)
            
        logger.info(f"Reset complete ({mode.value}) in {reset_time:.3f}s")
        
        return reset_info
        
    def custom_reset(self, **kwargs):
        """Custom reset with user-defined parameters"""
        
        # Update config with provided parameters
        if 'cube_position' in kwargs:
            pos = kwargs['cube_position']
            if 'cube' in self.objects:
                p.resetBasePositionAndOrientation(
                    self.objects['cube'],
                    pos,
                    [0, 0, 0, 1]
                )
                
        if 'robot_joints' in kwargs:
            joints = kwargs['robot_joints']
            if self.robot_id is not None:
                for i, pos in enumerate(joints):
                    p.resetJointState(self.robot_id, i, pos)
                    
        # Add more custom reset options as needed
        
    def register_callback(self, callback: Callable, 
                         phase: str = 'post'):
        """
        Register reset callback
        
        Args:
            callback: Function to call during reset
            phase: 'pre' or 'post' reset
        """
        if phase == 'pre':
            self.pre_reset_callbacks.append(callback)
        else:
            self.post_reset_callbacks.append(callback)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get reset statistics"""
        return {
            'total_resets': self.reset_count,
            'last_reset_time': self.last_reset_time,
            'average_reset_time': np.mean([r['reset_time'] for r in self.reset_history]) if self.reset_history else 0,
            'reset_modes_used': [r['mode'] for r in self.reset_history[-10:]],  # Last 10
            'current_seed': self.current_seed
        }


# ============================================================================
# Hotkey Handler
# ============================================================================

class HotkeyHandler:
    """Handles keyboard hotkeys for reset"""
    
    def __init__(self, scene_manager: SceneManager):
        """
        Initialize hotkey handler
        
        Args:
            scene_manager: SceneManager instance
        """
        self.scene_manager = scene_manager
        self.enabled = False
        self.hotkeys = {
            'r': (self._reset_soft, "Soft reset"),
            'shift+r': (self._reset_hard, "Hard reset"),
            's': (self._reseed, "Reseed and reset"),
            'ctrl+r': (self._reset_with_menu, "Reset with options"),
            'h': (self._show_help, "Show hotkey help")
        }
        
    def _reset_soft(self):
        """Soft reset handler"""
        print("\n[HOTKEY] Soft reset triggered")
        info = self.scene_manager.reset(ResetMode.SOFT)
        print(f"  Reset #{info['reset_count']} completed in {info['reset_time']:.3f}s")
        
    def _reset_hard(self):
        """Hard reset handler"""
        print("\n[HOTKEY] Hard reset triggered")
        info = self.scene_manager.reset(ResetMode.HARD)
        print(f"  Reset #{info['reset_count']} completed in {info['reset_time']:.3f}s")
        
    def _reseed(self):
        """Reseed handler"""
        print("\n[HOTKEY] Reseed triggered")
        info = self.scene_manager.reset(ResetMode.RESEED)
        print(f"  New seed: {info.get('new_seed', 'N/A')}")
        print(f"  Reset #{info['reset_count']} completed in {info['reset_time']:.3f}s")
        
    def _reset_with_menu(self):
        """Show reset menu"""
        print("\n" + "="*50)
        print(" RESET OPTIONS")
        print("="*50)
        print("1. Soft reset (positions only)")
        print("2. Hard reset (recreate objects)")
        print("3. Reseed with new random seed")
        print("4. Custom reset")
        print("0. Cancel")
        
        try:
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self.scene_manager.reset(ResetMode.SOFT)
            elif choice == '2':
                self.scene_manager.reset(ResetMode.HARD)
            elif choice == '3':
                self.scene_manager.reset(ResetMode.RESEED)
            elif choice == '4':
                # Custom reset example
                cube_x = float(input("Cube X position (0.3-0.6): "))
                cube_y = float(input("Cube Y position (-0.2-0.2): "))
                self.scene_manager.reset(
                    ResetMode.CUSTOM,
                    cube_position=[cube_x, cube_y, 0.02]
                )
            else:
                print("Reset cancelled")
                
        except Exception as e:
            print(f"Error during reset: {e}")
            
    def _show_help(self):
        """Show hotkey help"""
        print("\n" + "="*50)
        print(" HOTKEY HELP")
        print("="*50)
        for key, (_, desc) in self.hotkeys.items():
            print(f"  {key:10} - {desc}")
        print("="*50)
        
    def enable(self):
        """Enable hotkey listening"""
        if self.enabled:
            return
            
        for key, (handler, _) in self.hotkeys.items():
            keyboard.add_hotkey(key, handler)
            
        self.enabled = True
        logger.info("Hotkeys enabled")
        print("\nHotkeys enabled. Press 'h' for help.")
        
    def disable(self):
        """Disable hotkey listening"""
        if not self.enabled:
            return
            
        keyboard.unhook_all()
        self.enabled = False
        logger.info("Hotkeys disabled")


# ============================================================================
# Benchmark Manager
# ============================================================================

class BenchmarkManager:
    """Manages BC benchmark runs and metrics"""
    
    def __init__(self, scene_manager: SceneManager):
        """
        Initialize benchmark manager
        
        Args:
            scene_manager: SceneManager instance
        """
        self.scene_manager = scene_manager
        self.benchmark_results = []
        self.current_benchmark = None
        self.is_running = False
        
        # BC model placeholder
        self.bc_model = None
        self.bc_loaded = False
        
        # Task completion criteria
        self.success_threshold = 0.05  # 5cm from target
        self.max_episode_steps = 1000
        self.num_episodes = 10  # Episodes per benchmark
        
    def load_bc_model(self, model_path: str = None):
        """Load BC model for benchmarking"""
        try:
            if model_path is None:
                # Look for default BC model
                possible_paths = [
                    Path("models/bc_model.pth"),
                    Path("models/bc_policy.pkl"),
                    Path("trained_models/bc_latest.pth")
                ]
                
                for path in possible_paths:
                    if path.exists():
                        model_path = str(path)
                        break
                        
            if model_path and Path(model_path).exists():
                # Placeholder for actual model loading
                # In practice, load your trained BC model here
                logger.info(f"Loading BC model from {model_path}")
                self.bc_model = self._create_dummy_bc_model()
                self.bc_loaded = True
                return True
            else:
                logger.warning("No BC model found, using random policy")
                self.bc_model = self._create_dummy_bc_model()
                self.bc_loaded = False
                return False
                
        except Exception as e:
            logger.error(f"Failed to load BC model: {e}")
            self.bc_model = self._create_dummy_bc_model()
            self.bc_loaded = False
            return False
            
    def _create_dummy_bc_model(self):
        """Create dummy BC model for testing"""
        class DummyBCModel:
            def predict(self, obs):
                # Random actions for testing
                # Replace with actual BC inference
                return np.random.uniform(-1, 1, size=7)  # 7 DOF for Panda
                
        return DummyBCModel()
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation from environment"""
        obs = []
        
        # Get robot state
        if self.scene_manager.robot_id is not None:
            # Joint positions and velocities
            for i in range(7):  # 7 DOF arm
                state = p.getJointState(self.scene_manager.robot_id, i)
                obs.extend([state[0], state[1]])  # Position, velocity
                
            # End-effector position
            ee_state = p.getLinkState(self.scene_manager.robot_id, 7)
            obs.extend(ee_state[0])  # Position
            obs.extend(ee_state[1])  # Orientation
            
        # Get cube position
        if 'cube' in self.scene_manager.objects:
            cube_pos, cube_orn = p.getBasePositionAndOrientation(
                self.scene_manager.objects['cube']
            )
            obs.extend(cube_pos)
            obs.extend(cube_orn)
            
        # Get platform position
        if 'platform' in self.scene_manager.objects:
            platform_pos, _ = p.getBasePositionAndOrientation(
                self.scene_manager.objects['platform']
            )
            obs.extend(platform_pos)
            
        return np.array(obs, dtype=np.float32)
        
    def _compute_reward(self) -> Tuple[float, bool]:
        """
        Compute reward and check task completion
        
        Returns:
            (reward, done) tuple
        """
        if 'cube' not in self.scene_manager.objects:
            return 0.0, False
            
        if 'platform' not in self.scene_manager.objects:
            return 0.0, False
            
        # Get positions
        cube_pos, _ = p.getBasePositionAndOrientation(
            self.scene_manager.objects['cube']
        )
        platform_pos, _ = p.getBasePositionAndOrientation(
            self.scene_manager.objects['platform']
        )
        
        # Calculate distance
        distance = np.linalg.norm(
            np.array(cube_pos[:2]) - np.array(platform_pos[:2])
        )
        
        # Check if cube is on platform
        height_diff = abs(cube_pos[2] - (platform_pos[2] + 0.03))
        
        # Reward shaping
        reward = -distance  # Negative distance as reward
        
        # Check success
        success = distance < self.success_threshold and height_diff < 0.02
        
        return reward, success
        
    def _execute_action(self, action: np.ndarray):
        """Execute action on robot"""
        if self.scene_manager.robot_id is None:
            return
            
        # Apply action to robot joints
        for i in range(min(len(action), 7)):  # 7 DOF
            p.setJointMotorControl2(
                self.scene_manager.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=100
            )
            
    def run_episode(self) -> Dict[str, Any]:
        """
        Run single BC episode
        
        Returns:
            Episode statistics
        """
        # Reset environment
        self.scene_manager.reset(ResetMode.SOFT)
        
        episode_start = time.time()
        total_reward = 0
        steps = 0
        success = False
        
        for step in range(self.max_episode_steps):
            # Get observation
            obs = self._get_observation()
            
            # Get BC action
            action = self.bc_model.predict(obs)
            
            # Execute action
            self._execute_action(action)
            
            # Step simulation
            p.stepSimulation()
            
            # Compute reward
            reward, done = self._compute_reward()
            total_reward += reward
            steps += 1
            
            if done:
                success = True
                break
                
        episode_time = time.time() - episode_start
        
        return {
            'success': success,
            'steps': steps,
            'time': episode_time,
            'reward': total_reward,
            'steps_to_success': steps if success else None
        }
        
    def run_benchmark(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run full benchmark suite
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            Benchmark results
        """
        if self.is_running:
            return {'error': 'Benchmark already running'}
            
        self.is_running = True
        
        if num_episodes is None:
            num_episodes = self.num_episodes
            
        # Load BC model if not loaded
        if not self.bc_loaded:
            self.load_bc_model()
            
        benchmark_start = time.time()
        episodes_results = []
        
        logger.info(f"Starting BC benchmark with {num_episodes} episodes")
        
        for episode_num in range(num_episodes):
            logger.info(f"Running episode {episode_num + 1}/{num_episodes}")
            
            # Run episode
            result = self.run_episode()
            result['episode'] = episode_num + 1
            episodes_results.append(result)
            
            # Log progress
            if result['success']:
                logger.info(f"  Episode {episode_num + 1}: SUCCESS in {result['steps']} steps ({result['time']:.2f}s)")
            else:
                logger.info(f"  Episode {episode_num + 1}: FAILED after {result['steps']} steps ({result['time']:.2f}s)")
                
        benchmark_time = time.time() - benchmark_start
        
        # Compute statistics
        successes = [r for r in episodes_results if r['success']]
        success_rate = len(successes) / num_episodes
        
        avg_steps = np.mean([r['steps'] for r in episodes_results])
        avg_time = np.mean([r['time'] for r in episodes_results])
        avg_reward = np.mean([r['reward'] for r in episodes_results])
        
        if successes:
            avg_steps_to_success = np.mean([r['steps'] for r in successes])
            avg_time_to_success = np.mean([r['time'] for r in successes])
        else:
            avg_steps_to_success = None
            avg_time_to_success = None
            
        # Overall results
        benchmark_results = {
            'timestamp': time.time(),
            'num_episodes': num_episodes,
            'total_time': benchmark_time,
            'success_rate': success_rate,
            'successful_episodes': len(successes),
            'failed_episodes': num_episodes - len(successes),
            'avg_steps': avg_steps,
            'avg_time': avg_time,
            'avg_reward': avg_reward,
            'avg_steps_to_success': avg_steps_to_success,
            'avg_time_to_success': avg_time_to_success,
            'bc_model_loaded': self.bc_loaded,
            'episodes': episodes_results
        }
        
        # Store results
        self.benchmark_results.append(benchmark_results)
        self.current_benchmark = benchmark_results
        self.is_running = False
        
        # Log summary
        logger.info("="*60)
        logger.info(" BC BENCHMARK COMPLETE")
        logger.info("="*60)
        logger.info(f"Success Rate: {success_rate*100:.1f}%")
        logger.info(f"Total Time: {benchmark_time:.2f}s")
        if avg_time_to_success:
            logger.info(f"Avg Time to Success: {avg_time_to_success:.2f}s")
            logger.info(f"Avg Steps to Success: {avg_steps_to_success:.0f}")
        logger.info("="*60)
        
        return benchmark_results
        
    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Get latest benchmark results"""
        return self.current_benchmark if self.current_benchmark else None
        
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all benchmark results"""
        return self.benchmark_results
        
    def clear_results(self):
        """Clear benchmark results"""
        self.benchmark_results = []
        self.current_benchmark = None


# ============================================================================
# REST API Server
# ============================================================================

class ResetAPIServer:
    """REST API server for reset control"""
    
    def __init__(self, scene_manager: SceneManager, 
                 host: str = '127.0.0.1', port: int = 5000):
        """
        Initialize API server
        
        Args:
            scene_manager: SceneManager instance
            host: Server host
            port: Server port
        """
        self.scene_manager = scene_manager
        self.host = host
        self.port = port
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web clients
        
        # Benchmark manager
        self.benchmark_manager = BenchmarkManager(scene_manager)
        
        # Register routes
        self._register_routes()
        
        # Server thread
        self.server_thread = None
        self.is_running = False
        self.startup_time = time.time()
        
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            """Reset endpoint"""
            try:
                data = request.get_json() or {}
                mode = data.get('mode', 'soft')
                
                # Map string to enum
                mode_map = {
                    'soft': ResetMode.SOFT,
                    'hard': ResetMode.HARD,
                    'reseed': ResetMode.RESEED,
                    'custom': ResetMode.CUSTOM
                }
                
                reset_mode = mode_map.get(mode, ResetMode.SOFT)
                
                # Perform reset
                info = self.scene_manager.reset(reset_mode, **data)
                
                return jsonify({
                    'success': True,
                    'info': info
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @self.app.route('/reset/soft', methods=['POST'])
        def reset_soft():
            """Soft reset endpoint"""
            info = self.scene_manager.reset(ResetMode.SOFT)
            return jsonify({'success': True, 'info': info})
            
        @self.app.route('/reset/hard', methods=['POST'])
        def reset_hard():
            """Hard reset endpoint"""
            info = self.scene_manager.reset(ResetMode.HARD)
            return jsonify({'success': True, 'info': info})
            
        @self.app.route('/reset/reseed', methods=['POST'])
        def reset_reseed():
            """Reseed endpoint"""
            info = self.scene_manager.reset(ResetMode.RESEED)
            return jsonify({'success': True, 'info': info})
            
        @self.app.route('/seed', methods=['GET', 'POST'])
        def seed():
            """Get or set seed"""
            if request.method == 'POST':
                data = request.get_json() or {}
                seed_value = data.get('seed', None)
                new_seed = self.scene_manager.set_seed(seed_value)
                return jsonify({'success': True, 'seed': new_seed})
            else:
                return jsonify({'seed': self.scene_manager.current_seed})
                
        @self.app.route('/stats', methods=['GET'])
        def stats():
            """Get reset statistics"""
            return jsonify(self.scene_manager.get_statistics())
            
        @self.app.route('/config', methods=['GET', 'POST'])
        def config():
            """Get or update configuration"""
            if request.method == 'POST':
                data = request.get_json() or {}
                # Update config fields
                for key, value in data.items():
                    if hasattr(self.scene_manager.config, key):
                        setattr(self.scene_manager.config, key, value)
                return jsonify({'success': True})
            else:
                # Return current config
                return jsonify({
                    'robot_joints': self.scene_manager.config.robot_joints,
                    'cube_pos_range': self.scene_manager.config.cube_pos_range,
                    'platform_pos_range': self.scene_manager.config.platform_pos_range,
                    'seed': self.scene_manager.current_seed
                })
                
        @self.app.route('/health', methods=['GET'])
        def health():
            """Basic health check"""
            return jsonify({'status': 'healthy', 'running': True})
            
        @self.app.route('/healthz', methods=['GET'])
        def healthz():
            """Comprehensive health check with system info"""
            try:
                health_data = {
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'uptime': time.time() - self.startup_time if hasattr(self, 'startup_time') else None,
                    
                    # PyBullet Connection Status
                    'pybullet': self._get_pybullet_status(),
                    
                    # Model Status
                    'models': self._get_model_status(),
                    
                    # Device Information
                    'device': self._get_device_info(),
                    
                    # System Resources
                    'system': self._get_system_info(),
                    
                    # Scene Status
                    'scene': self._get_scene_status(),
                    
                    # API Server Status
                    'api': {
                        'running': self.is_running,
                        'host': self.host,
                        'port': self.port,
                        'endpoints_available': self._get_available_endpoints()
                    },
                    
                    # Benchmark Status
                    'benchmark': {
                        'is_running': self.benchmark_manager.is_running,
                        'total_runs': len(self.benchmark_manager.benchmark_results),
                        'last_run': self.benchmark_manager.get_latest_results() is not None
                    }
                }
                
                # Determine overall health
                if not health_data['pybullet']['connected']:
                    health_data['status'] = 'degraded'
                    health_data['warnings'] = health_data.get('warnings', []) + ['PyBullet not connected']
                    
                if not health_data['models']['bc']['loaded']:
                    health_data['warnings'] = health_data.get('warnings', []) + ['BC model not loaded']
                    
                return jsonify(health_data)
                
            except Exception as e:
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': time.time()
                }), 500
            
        @self.app.route('/benchmark', methods=['POST'])
        def benchmark():
            """Run BC benchmark"""
            try:
                data = request.get_json() or {}
                num_episodes = data.get('num_episodes', 10)
                model_path = data.get('model_path', None)
                
                # Load model if specified
                if model_path:
                    self.benchmark_manager.load_bc_model(model_path)
                    
                # Run benchmark
                results = self.benchmark_manager.run_benchmark(num_episodes)
                
                return jsonify({
                    'success': True,
                    'results': results
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @self.app.route('/benchmark/status', methods=['GET'])
        def benchmark_status():
            """Get benchmark status"""
            return jsonify({
                'is_running': self.benchmark_manager.is_running,
                'bc_model_loaded': self.benchmark_manager.bc_loaded,
                'latest_results': self.benchmark_manager.get_latest_results()
            })
            
        @self.app.route('/benchmark/results', methods=['GET'])
        def benchmark_results():
            """Get all benchmark results"""
            return jsonify({
                'results': self.benchmark_manager.get_all_results()
            })
            
        @self.app.route('/benchmark/clear', methods=['POST'])
        def benchmark_clear():
            """Clear benchmark results"""
            self.benchmark_manager.clear_results()
            return jsonify({'success': True})
            
        @self.app.route('/benchmark/load_model', methods=['POST'])
        def load_bc_model():
            """Load BC model"""
            try:
                data = request.get_json() or {}
                model_path = data.get('model_path', None)
                
                success = self.benchmark_manager.load_bc_model(model_path)
                
                return jsonify({
                    'success': success,
                    'model_loaded': self.benchmark_manager.bc_loaded
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
            
    def start(self):
        """Start API server in background thread"""
        if self.is_running:
            return
            
        def run_server():
            self.app.run(host=self.host, port=self.port, debug=False)
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        logger.info(f"API server started at http://{self.host}:{self.port}")
        print(f"\nREST API available at: http://{self.host}:{self.port}/reset")
        print(f"Health check available at: http://{self.host}:{self.port}/healthz")
        
    def _get_pybullet_status(self) -> Dict[str, Any]:
        """Get PyBullet connection status"""
        try:
            if self.scene_manager.client is not None:
                # Test connection by getting simulation info
                connection_info = p.getConnectionInfo(self.scene_manager.client)
                
                return {
                    'connected': connection_info['isConnected'],
                    'connection_method': ['DIRECT', 'GUI', 'SHARED_MEMORY', 'UDP', 'TCP'][connection_info['connectionMethod']],
                    'num_bodies': p.getNumBodies(),
                    'num_joints': p.getNumJoints(self.scene_manager.robot_id) if self.scene_manager.robot_id else 0,
                    'gravity': p.getGravity(),
                    'time_step': p.getPhysicsEngineParameters()['fixedTimeStep'],
                    'real_time_simulation': p.getPhysicsEngineParameters()['useRealTimeSimulation']
                }
            else:
                return {
                    'connected': False,
                    'connection_method': None,
                    'error': 'No physics client initialized'
                }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
            
    def _get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        model_status = {
            'bc': {
                'loaded': self.benchmark_manager.bc_loaded,
                'model_path': None,
                'parameters': None
            }
        }
        
        # Get BC model details if loaded
        if self.benchmark_manager.bc_loaded and self.benchmark_manager.bc_model:
            try:
                if hasattr(self.benchmark_manager.bc_model, 'model'):
                    # For actual trained models
                    if TORCH_AVAILABLE:
                        model = self.benchmark_manager.bc_model.model
                        model_status['bc']['parameters'] = sum(p.numel() for p in model.parameters())
                        model_status['bc']['device'] = str(next(model.parameters()).device)
                else:
                    # For dummy models
                    model_status['bc']['type'] = 'dummy/random'
            except:
                pass
                
        return model_status
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device and hardware information"""
        device_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            }
        }
        
        # Add GPU info if available
        if TORCH_AVAILABLE:
            device_info['torch_available'] = True
            device_info['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                device_info['cuda'] = {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(0),
                    'capability': torch.cuda.get_device_capability(0),
                    'memory_allocated': torch.cuda.memory_allocated(0),
                    'memory_reserved': torch.cuda.memory_reserved(0)
                }
        else:
            device_info['torch_available'] = False
            device_info['cuda_available'] = False
            
        return device_info
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        try:
            process = psutil.Process()
            
            return {
                'process_id': process.pid,
                'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                'process_cpu_percent': process.cpu_percent(),
                'threads': threading.active_count(),
                'disk_usage': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
        except Exception as e:
            return {'error': str(e)}
            
    def _get_scene_status(self) -> Dict[str, Any]:
        """Get current scene status"""
        return {
            'objects_count': len(self.scene_manager.objects),
            'robot_id': self.scene_manager.robot_id is not None,
            'current_seed': self.scene_manager.current_seed,
            'reset_count': self.scene_manager.reset_count,
            'last_reset_time': self.scene_manager.last_reset_time,
            'scene_objects': list(self.scene_manager.objects.keys())
        }
        
    def _get_available_endpoints(self) -> List[str]:
        """Get list of available API endpoints"""
        endpoints = []
        for rule in self.app.url_map.iter_rules():
            if rule.endpoint != 'static':
                endpoints.append({
                    'path': str(rule),
                    'methods': list(rule.methods - {'HEAD', 'OPTIONS'})
                })
        return endpoints
        
    def stop(self):
        """Stop API server"""
        # Flask doesn't have a clean shutdown in thread
        # This is a limitation, server will stop when main program exits
        self.is_running = False
        logger.info("API server stop requested")


# ============================================================================
# Main Reset System
# ============================================================================

class ResetSystem:
    """Main reset system combining all components"""
    
    def __init__(self, physics_client: Optional[int] = None):
        """
        Initialize complete reset system
        
        Args:
            physics_client: PyBullet physics client
        """
        # Core components
        self.scene_manager = SceneManager(physics_client)
        self.hotkey_handler = HotkeyHandler(self.scene_manager)
        self.api_server = ResetAPIServer(self.scene_manager)
        
        # State
        self.initialized = False
        
    def initialize(self, enable_hotkeys: bool = True,
                  enable_api: bool = True,
                  api_port: int = 5000):
        """
        Initialize reset system
        
        Args:
            enable_hotkeys: Enable keyboard hotkeys
            enable_api: Enable REST API
            api_port: API server port
        """
        if self.initialized:
            return
            
        # Create initial scene
        self.scene_manager.create_scene()
        
        # Enable hotkeys
        if enable_hotkeys:
            self.hotkey_handler.enable()
            
        # Start API server
        if enable_api:
            self.api_server.port = api_port
            self.api_server.start()
            
        self.initialized = True
        
        print("\n" + "="*60)
        print(" RESET SYSTEM INITIALIZED")
        print("="*60)
        
        if enable_hotkeys:
            print("\nHotkeys:")
            print("  R       - Soft reset")
            print("  Shift+R - Hard reset")
            print("  S       - Reseed")
            print("  Ctrl+R  - Reset menu")
            print("  H       - Help")
            
        if enable_api:
            print(f"\nAPI Endpoints:")
            print(f"  POST http://localhost:{api_port}/reset")
            print(f"  POST http://localhost:{api_port}/reset/soft")
            print(f"  POST http://localhost:{api_port}/reset/hard")
            print(f"  POST http://localhost:{api_port}/reset/reseed")
            print(f"  POST http://localhost:{api_port}/benchmark")
            print(f"  GET  http://localhost:{api_port}/benchmark/status")
            print(f"  GET  http://localhost:{api_port}/healthz")
            print(f"  GET  http://localhost:{api_port}/stats")
            
        print("="*60)
        
    def reset(self, mode: str = "soft", **kwargs) -> Dict[str, Any]:
        """
        Programmatic reset
        
        Args:
            mode: Reset mode string
            **kwargs: Additional parameters
            
        Returns:
            Reset information
        """
        mode_map = {
            'soft': ResetMode.SOFT,
            'hard': ResetMode.HARD,
            'reseed': ResetMode.RESEED,
            'custom': ResetMode.CUSTOM
        }
        
        reset_mode = mode_map.get(mode, ResetMode.SOFT)
        return self.scene_manager.reset(reset_mode, **kwargs)
        
    def shutdown(self):
        """Shutdown reset system"""
        self.hotkey_handler.disable()
        self.api_server.stop()
        logger.info("Reset system shutdown")


# ============================================================================
# Example Usage
# ============================================================================

def example_with_pybullet():
    """Example integration with PyBullet simulation"""
    import pybullet as p
    import time
    
    # Connect to PyBullet
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    
    # Initialize reset system
    reset_system = ResetSystem(client)
    reset_system.initialize(enable_hotkeys=True, enable_api=True)
    
    # Simulation loop
    print("\nSimulation running. Use hotkeys or API to reset.")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
            
            # Optional: programmatic reset every 1000 steps
            # if step_count % 1000 == 0:
            #     reset_system.reset("soft")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        reset_system.shutdown()
        p.disconnect()


def example_api_client():
    """Example API client usage"""
    import requests
    import time
    
    base_url = "http://localhost:5000"
    
    # Wait for server to be ready
    time.sleep(1)
    
    # Soft reset
    response = requests.post(f"{base_url}/reset/soft")
    print("Soft reset:", response.json())
    
    # Hard reset
    response = requests.post(f"{base_url}/reset/hard")
    print("Hard reset:", response.json())
    
    # Reseed
    response = requests.post(f"{base_url}/reset/reseed")
    print("Reseed:", response.json())
    
    # Custom reset with parameters
    response = requests.post(f"{base_url}/reset", json={
        'mode': 'custom',
        'cube_position': [0.5, 0.1, 0.02]
    })
    print("Custom reset:", response.json())
    
    # Get statistics
    response = requests.get(f"{base_url}/stats")
    print("Statistics:", response.json())


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" RESET SYSTEM DEMO")
    print("="*70)
    
    print("\nOptions:")
    print("1. Run with PyBullet GUI (hotkeys + API)")
    print("2. Test API client")
    print("3. Test without GUI")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        example_with_pybullet()
    elif choice == '2':
        # Start server first
        reset_system = ResetSystem()
        reset_system.initialize(enable_hotkeys=False, enable_api=True)
        time.sleep(1)
        example_api_client()
    else:
        # Headless test
        client = p.connect(p.DIRECT)
        reset_system = ResetSystem(client)
        reset_system.initialize(enable_hotkeys=False, enable_api=True)
        
        # Test resets
        for mode in ['soft', 'hard', 'reseed']:
            print(f"\nTesting {mode} reset...")
            info = reset_system.reset(mode)
            print(f"  Result: {info}")
            time.sleep(0.5)
            
        reset_system.shutdown()
        p.disconnect()