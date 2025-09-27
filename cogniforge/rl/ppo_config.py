"""
PPO configuration for lightweight CPU training.

This module provides optimized PPO configurations for small-scale training
with compact networks and CPU-friendly batch sizes.
"""

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    StopTrainingOnRewardThreshold,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from typing import Dict, Any, Optional, Union, Callable, Type, List
import torch.nn as nn
import logging
import os
import json
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)


class CompactMLP(BaseFeaturesExtractor):
    """
    Compact MLP feature extractor for PPO.
    
    Small 2-layer network with 64 units per layer for CPU-friendly training.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        hidden_dim: int = 64,
        activation_fn: Type[nn.Module] = nn.ReLU
    ):
        super().__init__(observation_space, features_dim)
        
        n_input_features = get_flattened_obs_dim(observation_space)
        
        self.net = nn.Sequential(
            nn.Linear(n_input_features, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, features_dim),
            activation_fn(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def create_lightweight_ppo(
    env: Union[gym.Env, DummyVecEnv],
    policy_kwargs: Optional[Dict[str, Any]] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 128,          # Small buffer for low memory usage
    batch_size: int = 16,        # Tiny batch size
    n_epochs: int = 8,           # 5-10 mini-epochs
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False,       # State-dependent exploration
    normalize_advantage: bool = True,
    target_kl: Optional[float] = 0.02,
    tensorboard_log: Optional[str] = None,
    device: str = "auto",
    verbose: int = 1
) -> PPO:
    """
    Create a lightweight PPO model optimized for CPU training.
    
    Args:
        env: Training environment
        policy_kwargs: Custom policy network architecture
        learning_rate: Learning rate (default 3e-4)
        n_steps: Number of steps per environment per update (default 128)
        batch_size: Minibatch size (default 16)
        n_epochs: Number of epochs for PPO updates (default 8)
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        use_sde: Use state-dependent exploration
        normalize_advantage: Normalize advantages
        target_kl: Target KL divergence for early stopping
        tensorboard_log: Directory for tensorboard logs
        device: Device for training ('cpu', 'cuda', or 'auto')
        verbose: Verbosity level
        
    Returns:
        Configured PPO model
        
    Example:
        env = gym.make('CartPole-v1')
        model = create_lightweight_ppo(
            env,
            n_epochs=8,
            batch_size=16,
            device='cpu'
        )
        model.learn(total_timesteps=5000)
    """
    # Default compact policy architecture if not provided
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": dict(
                pi=[64, 64],  # Actor network: 2 layers × 64 units
                vf=[64, 64]   # Critic network: 2 layers × 64 units
            ),
            "activation_fn": nn.ReLU,
            "features_extractor_class": CompactMLP,
            "features_extractor_kwargs": {
                "features_dim": 64,
                "hidden_dim": 64
            }
        }
    
    # Check device and adjust if needed
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.info("CPU detected, using CPU-optimized settings")
    
    # Create PPO model with compact configuration
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=target_kl,
        normalize_advantage=normalize_advantage,
        stats_window_size=100,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=verbose,
        seed=None,
        _init_setup_model=True
    )
    
    if verbose:
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        logger.info(f"PPO Model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Buffer size: {n_steps}")
        logger.info(f"  Epochs: {n_epochs}")
    
    return model


def train_lightweight_ppo(
    env_fn: Callable[[], gym.Env],
    total_timesteps: int = None,
    n_envs: int = 1,
    save_path: str = "ppo_model",
    eval_env_fn: Optional[Callable[[], gym.Env]] = None,
    eval_freq: int = 1000,
    n_eval_episodes: int = 5,
    checkpoint_freq: int = 2000,
    log_dir: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    auto_adjust_timesteps: bool = True,
    cpu_limit_gb: float = 4.0,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Complete training pipeline for lightweight PPO.
    
    Automatically adjusts timesteps based on available CPU/memory.
    
    Args:
        env_fn: Function to create environment
        total_timesteps: Total training timesteps (auto-adjusted if None)
        n_envs: Number of parallel environments (1 for CPU)
        save_path: Path to save trained model
        eval_env_fn: Function to create evaluation environment
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        checkpoint_freq: Checkpoint save frequency
        log_dir: Directory for logs
        hyperparameters: Custom hyperparameters for PPO
        auto_adjust_timesteps: Automatically adjust timesteps based on resources
        cpu_limit_gb: Memory limit for auto-adjustment (GB)
        verbose: Verbosity level
        
    Returns:
        Dictionary containing:
        - 'model': Trained PPO model
        - 'env': Training environment
        - 'results': Training results
        - 'config': Configuration used
        
    Example:
        def make_env():
            env = gym.make('CartPole-v1')
            return ShortHorizonDeltaWrapper(env, action_repeat=5)
        
        results = train_lightweight_ppo(
            env_fn=make_env,
            total_timesteps=10000,
            n_envs=1,
            save_path='models/ppo_lightweight'
        )
    """
    # Setup logging directory
    if log_dir is None:
        log_dir = f"logs/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # Check system resources and auto-adjust if needed
    if auto_adjust_timesteps and total_timesteps is None:
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        cpu_count = psutil.cpu_count()
        
        if available_memory < cpu_limit_gb or cpu_count <= 4:
            total_timesteps = 5000  # Conservative for limited resources
            if verbose:
                logger.info(f"Limited resources detected (RAM: {available_memory:.1f}GB, CPUs: {cpu_count})")
                logger.info(f"Using conservative timesteps: {total_timesteps}")
        else:
            total_timesteps = 15000  # Moderate for decent resources
            if verbose:
                logger.info(f"Adequate resources (RAM: {available_memory:.1f}GB, CPUs: {cpu_count})")
                logger.info(f"Using moderate timesteps: {total_timesteps}")
    elif total_timesteps is None:
        total_timesteps = 10000  # Default
    
    # Create training environment(s)
    if n_envs == 1:
        env = DummyVecEnv([lambda: Monitor(env_fn(), log_dir)])
    else:
        # Use SubprocVecEnv for multiple environments
        env = SubprocVecEnv([lambda: Monitor(env_fn(), log_dir) for _ in range(n_envs)])
    
    # Optionally add normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Setup evaluation environment if provided
    eval_env = None
    eval_callback = None
    if eval_env_fn is not None:
        eval_env = DummyVecEnv([lambda: Monitor(eval_env_fn(), log_dir)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, norm_obs_keys=None)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{save_path}_best",
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose
        )
    
    # Default hyperparameters for lightweight training
    default_hyperparams = {
        "learning_rate": 3e-4,
        "n_steps": 128,        # Small buffer
        "batch_size": 16,      # Tiny batches
        "n_epochs": 8,         # 5-10 epochs
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "normalize_advantage": True,
        "target_kl": 0.02
    }
    
    # Update with custom hyperparameters
    if hyperparameters:
        default_hyperparams.update(hyperparameters)
    
    # Create PPO model
    model = create_lightweight_ppo(
        env=env,
        tensorboard_log=f"{log_dir}/tensorboard",
        verbose=verbose,
        **default_hyperparams
    )
    
    # Setup callbacks
    callbacks = []
    
    if eval_callback:
        callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=f"{save_path}_checkpoints",
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping on no improvement
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=10,
        verbose=verbose
    )
    
    if eval_callback:
        eval_callback.callback = stop_callback
    
    # Combine callbacks
    callback_list = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0] if callbacks else None
    
    # Save configuration
    config = {
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "hyperparameters": default_hyperparams,
        "model_architecture": {
            "pi": model.policy_kwargs.get("net_arch", {}).get("pi", [64, 64]),
            "vf": model.policy_kwargs.get("net_arch", {}).get("vf", [64, 64])
        },
        "device": str(model.device),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    if verbose:
        print("\n" + "="*60)
        print("Starting PPO Training")
        print("="*60)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Number of environments: {n_envs}")
        print(f"Batch size: {default_hyperparams['batch_size']}")
        print(f"Buffer size: {default_hyperparams['n_steps']}")
        print(f"Epochs per update: {default_hyperparams['n_epochs']}")
        print(f"Learning rate: {default_hyperparams['learning_rate']}")
        print(f"Device: {model.device}")
        print("="*60 + "\n")
    
    # Train model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=10,
            tb_log_name="ppo",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        # Save final model
        model.save(save_path)
        env.save(f"{save_path}_vecnormalize.pkl")
        
        if verbose:
            print(f"\nModel saved to {save_path}")
        
    except KeyboardInterrupt:
        if verbose:
            print("\nTraining interrupted by user")
        model.save(f"{save_path}_interrupted")
        env.save(f"{save_path}_interrupted_vecnormalize.pkl")
    
    # Cleanup
    if eval_env:
        eval_env.close()
    
    return {
        "model": model,
        "env": env,
        "config": config,
        "log_dir": log_dir,
        "save_path": save_path
    }


def quick_ppo_train(
    env: gym.Env,
    timesteps: Optional[int] = None,
    auto_scale: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> PPO:
    """
    Quick PPO training with automatic configuration.
    
    Simplest interface for training PPO with CPU-optimized settings.
    
    Args:
        env: Gymnasium environment
        timesteps: Training timesteps (auto-scaled if None)
        auto_scale: Automatically scale based on resources
        save_path: Path to save model
        verbose: Print progress
        
    Returns:
        Trained PPO model
        
    Example:
        env = gym.make('CartPole-v1')
        model = quick_ppo_train(env, timesteps=5000)
        
        # Test the model
        obs, _ = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            if done:
                break
    """
    # Auto-scale timesteps based on system resources
    if timesteps is None and auto_scale:
        mem_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = psutil.cpu_count()
        
        if mem_gb < 4 or cpu_count <= 2:
            timesteps = 3000  # Very limited resources
            batch_size = 8
            n_steps = 64
            n_epochs = 5
        elif mem_gb < 8 or cpu_count <= 4:
            timesteps = 5000  # Limited resources
            batch_size = 16
            n_steps = 128
            n_epochs = 8
        else:
            timesteps = 15000  # Decent resources
            batch_size = 32
            n_steps = 256
            n_epochs = 10
        
        if verbose:
            print(f"Auto-scaled configuration:")
            print(f"  Timesteps: {timesteps}")
            print(f"  Batch size: {batch_size}")
            print(f"  Buffer size: {n_steps}")
            print(f"  Epochs: {n_epochs}")
    else:
        timesteps = timesteps or 10000
        batch_size = 16
        n_steps = 128
        n_epochs = 8
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Create model with compact architecture
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64])  # Small 2×64 network
        },
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1 if verbose else 0,
        device="cpu"  # Force CPU for compatibility
    )
    
    # Train
    if verbose:
        print(f"\nTraining PPO for {timesteps} timesteps...")
    
    model.learn(total_timesteps=timesteps, progress_bar=verbose)
    
    # Save if path provided
    if save_path:
        model.save(save_path)
        vec_env.save(f"{save_path}_vecnormalize.pkl")
        if verbose:
            print(f"Model saved to {save_path}")
    
    return model


class PPOTrainer:
    """
    High-level PPO trainer with resource management.
    
    Provides easy interface for training PPO with automatic resource
    adaptation and experiment tracking.
    """
    
    def __init__(
        self,
        env_name: str,
        env_fn: Callable[[], gym.Env],
        experiment_name: Optional[str] = None,
        base_dir: str = "experiments",
        device: str = "auto"
    ):
        """
        Initialize PPO trainer.
        
        Args:
            env_name: Name of the environment
            env_fn: Function to create environment
            experiment_name: Name for this experiment
            base_dir: Base directory for saving
            device: Device for training
        """
        self.env_name = env_name
        self.env_fn = env_fn
        self.experiment_name = experiment_name or f"{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = base_dir
        self.device = device
        
        # Setup directories
        self.exp_dir = os.path.join(base_dir, self.experiment_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.model = None
        self.env = None
        self.results = []
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        n_runs: int = 1,
        hyperparameter_search: bool = False,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train PPO model with optional hyperparameter search.
        
        Args:
            total_timesteps: Training timesteps per run
            n_runs: Number of training runs
            hyperparameter_search: Perform hyperparameter search
            save_best: Save best model
            verbose: Print progress
            
        Returns:
            Training results
        """
        best_reward = -float('inf')
        best_model = None
        best_config = None
        
        # Determine timesteps
        if total_timesteps is None:
            mem_gb = psutil.virtual_memory().available / (1024**3)
            total_timesteps = 5000 if mem_gb < 4 else 10000 if mem_gb < 8 else 15000
            if verbose:
                print(f"Auto-selected timesteps: {total_timesteps} (based on {mem_gb:.1f}GB available RAM)")
        
        configs = self._generate_configs() if hyperparameter_search else [self._get_default_config()]
        
        for run_idx in range(n_runs):
            config = configs[run_idx % len(configs)]
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Run {run_idx + 1}/{n_runs}")
                print(f"Config: batch_size={config['batch_size']}, "
                      f"n_epochs={config['n_epochs']}, lr={config['learning_rate']}")
                print(f"{'='*60}")
            
            # Create environment
            env = DummyVecEnv([self.env_fn])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            
            # Create model
            model = create_lightweight_ppo(
                env=env,
                tensorboard_log=os.path.join(self.log_dir, f"run_{run_idx}"),
                verbose=1 if verbose else 0,
                **config
            )
            
            # Train
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=verbose,
                reset_num_timesteps=True
            )
            
            # Evaluate
            eval_env = DummyVecEnv([self.env_fn])
            mean_reward, std_reward = self._evaluate_model(model, eval_env, n_episodes=10)
            
            if verbose:
                print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Track results
            result = {
                "run": run_idx,
                "config": config,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "timesteps": total_timesteps
            }
            self.results.append(result)
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_model = model
                best_config = config
                
                if save_best:
                    model.save(os.path.join(self.model_dir, "best_model"))
                    env.save(os.path.join(self.model_dir, "best_model_vecnormalize.pkl"))
            
            # Clean up
            env.close()
            eval_env.close()
        
        # Save results
        self._save_results()
        
        self.model = best_model
        
        return {
            "best_model": best_model,
            "best_config": best_config,
            "best_reward": best_reward,
            "all_results": self.results
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default lightweight configuration."""
        return {
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 16,
            "n_epochs": 8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01
        }
    
    def _generate_configs(self) -> List[Dict[str, Any]]:
        """Generate configurations for hyperparameter search."""
        configs = []
        
        # Small search space for CPU
        batch_sizes = [8, 16, 32]
        n_epochs = [5, 8, 10]
        learning_rates = [1e-4, 3e-4, 5e-4]
        
        for bs in batch_sizes:
            for ne in n_epochs:
                for lr in learning_rates:
                    config = self._get_default_config()
                    config.update({
                        "batch_size": bs,
                        "n_epochs": ne,
                        "learning_rate": lr,
                        "n_steps": bs * 8  # Scale buffer with batch size
                    })
                    configs.append(config)
        
        return configs
    
    def _evaluate_model(
        self,
        model: PPO,
        env: VecNormalize,
        n_episodes: int = 10
    ) -> tuple:
        """Evaluate model performance."""
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]
            
            episode_rewards.append(total_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def _save_results(self):
        """Save training results to file."""
        results_path = os.path.join(self.exp_dir, "results.json")
        
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def load_best(self) -> PPO:
        """Load the best saved model."""
        model_path = os.path.join(self.model_dir, "best_model")
        vec_norm_path = os.path.join(self.model_dir, "best_model_vecnormalize.pkl")
        
        env = DummyVecEnv([self.env_fn])
        env = VecNormalize.load(vec_norm_path, env)
        
        model = PPO.load(model_path, env=env)
        
        return model


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Lightweight PPO Configuration")
    print("=" * 60)
    
    # Check system resources
    mem_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = psutil.cpu_count()
    print(f"\nSystem Resources:")
    print(f"  Available RAM: {mem_gb:.1f} GB")
    print(f"  CPU cores: {cpu_count}")
    print(f"  PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Test 1: Create lightweight PPO model
    print("\n1. Creating Lightweight PPO Model")
    print("-" * 40)
    
    env = gym.make("CartPole-v1")
    
    model = create_lightweight_ppo(
        env,
        n_steps=128,
        batch_size=16,
        n_epochs=8,
        device="cpu",
        verbose=1
    )
    
    print(f"Model created successfully!")
    print(f"Policy network: {model.policy}")
    
    # Test 2: Quick training
    print("\n2. Quick Training Test")
    print("-" * 40)
    
    env = gym.make("CartPole-v1")
    
    # Determine timesteps based on resources
    if mem_gb < 4:
        test_timesteps = 3000
        print(f"Using minimal timesteps: {test_timesteps} (low memory)")
    elif mem_gb < 8:
        test_timesteps = 5000
        print(f"Using reduced timesteps: {test_timesteps} (moderate memory)")
    else:
        test_timesteps = 10000
        print(f"Using standard timesteps: {test_timesteps} (adequate memory)")
    
    model = quick_ppo_train(
        env,
        timesteps=test_timesteps,
        auto_scale=True,
        verbose=True
    )
    
    # Test 3: Evaluate trained model
    print("\n3. Evaluating Trained Model")
    print("-" * 40)
    
    total_reward = 0
    n_test_episodes = 5
    
    for episode in range(n_test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.0f}")
    
    avg_reward = total_reward / n_test_episodes
    print(f"\nAverage reward over {n_test_episodes} episodes: {avg_reward:.2f}")
    
    # Test 4: Resource usage summary
    print("\n4. Training Configuration Summary")
    print("-" * 40)
    
    configs = [
        ("Minimal (3-5k steps)", 3000, 8, 5),
        ("Light (5-10k steps)", 5000, 16, 8),
        ("Standard (10-20k steps)", 10000, 32, 10)
    ]
    
    print("\nRecommended configurations based on resources:")
    for name, timesteps, batch, epochs in configs:
        print(f"\n{name}:")
        print(f"  Total timesteps: {timesteps}")
        print(f"  Batch size: {batch}")
        print(f"  Epochs: {epochs}")
        print(f"  Estimated time: ~{timesteps/100:.0f} seconds on CPU")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)