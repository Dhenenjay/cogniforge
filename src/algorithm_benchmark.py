"""
Algorithm Benchmark: PPO vs CMA-ES

This module benchmarks PPO (Proximal Policy Optimization) vs CMA-ES
(Covariance Matrix Adaptation Evolution Strategy) to determine which
is better suited for your machine and task requirements.
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from collections import deque
import warnings
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set single thread for PyTorch (as per your optimization)
torch.set_num_threads(1)


# ============================================================================
# Test Problem: Simple Manipulation Task
# ============================================================================

class ManipulationEnvironment:
    """
    Simplified manipulation environment for benchmarking
    Represents a reaching/grasping task
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 7):
        """
        Initialize environment
        
        Args:
            state_dim: Observation space dimension
            action_dim: Action space dimension (7 DOF for Panda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = 50
        self.current_step = 0
        
        # Target position for reaching
        self.target = np.random.randn(3) * 0.1 + np.array([0.5, 0.0, 0.3])
        self.state = None
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        # Initial state: joint positions + target position + gripper state
        self.state = np.random.randn(self.state_dim) * 0.1
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in environment
        
        Args:
            action: Joint velocities or torques
            
        Returns:
            next_state, reward, done, info
        """
        # Apply action
        # Pad or truncate action to match state dimension
        if len(action) < self.state_dim:
            action_padded = np.zeros(self.state_dim)
            action_padded[:len(action)] = action
        else:
            action_padded = action[:self.state_dim]
        
        self.state += action_padded * 0.01  # Simple integration
        self.state = np.clip(self.state, -2, 2)
        
        # Calculate reward (negative distance to target)
        end_effector_pos = self.state[:3]  # Simplified FK
        distance = np.linalg.norm(end_effector_pos - self.target)
        reward = -distance - 0.01 * np.linalg.norm(action)  # Distance + action penalty
        
        # Check termination
        self.current_step += 1
        done = self.current_step >= self.max_steps or distance < 0.05
        
        info = {'distance': distance, 'success': distance < 0.05}
        
        return self.state, reward, done, info


# ============================================================================
# PPO Implementation
# ============================================================================

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        # Actor output
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_std).expand_as(action_mean)
        
        # Critic output
        value = self.critic(x)
        
        return action_mean, action_std, value


class PPOAgent:
    """PPO agent for continuous control"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 3e-4, gamma: float = 0.99):
        """
        Initialize PPO agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
        """
        self.device = torch.device("cpu")  # Use CPU for single-thread
        self.gamma = gamma
        
        # Network
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Timing stats
        self.inference_times = []
        self.update_times = []
    
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select action using policy"""
        start_time = time.perf_counter()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_mean, action_std, value = self.network(state_tensor)
            
            # Sample action from distribution
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        self.inference_times.append(time.perf_counter() - start_time)
        
        return action.squeeze().cpu().numpy(), log_prob, value
    
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """
        Update policy using collected trajectories
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Training statistics
        """
        start_time = time.perf_counter()
        
        # Prepare batch data
        states = torch.FloatTensor([t['state'] for t in trajectories]).to(self.device)
        actions = torch.FloatTensor([t['action'] for t in trajectories]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in trajectories]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in trajectories]).to(self.device)
        
        # Calculate returns
        returns = self._calculate_returns(rewards)
        
        # PPO update
        action_mean, action_std, values = self.network(states)
        dist = Normal(action_mean, action_std)
        
        # Calculate new log probabilities
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Calculate advantages
        advantages = returns - values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = self.value_coef * nn.MSELoss()(values.squeeze(), returns)
        entropy = self.entropy_coef * dist.entropy().mean()
        
        total_loss = policy_loss + value_loss - entropy
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.update_times.append(time.perf_counter() - start_time)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def _calculate_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns


# ============================================================================
# CMA-ES Implementation
# ============================================================================

class CMAESOptimizer:
    """
    CMA-ES optimizer for black-box optimization
    Simplified implementation for benchmarking
    """
    
    def __init__(self, dim: int, population_size: Optional[int] = None,
                 sigma: float = 0.5):
        """
        Initialize CMA-ES
        
        Args:
            dim: Problem dimension
            population_size: Population size (default: 4 + 3*log(dim))
            sigma: Initial step size
        """
        self.dim = dim
        self.sigma = sigma
        
        # Population size
        if population_size is None:
            self.population_size = int(4 + 3 * np.log(dim))
        else:
            self.population_size = population_size
        
        # Selection parameters
        self.mu = self.population_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1 / np.sum(self.weights ** 2)
        
        # Adaptation parameters
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        
        # Initialize state
        self.mean = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.invsqrtC = np.eye(dim)
        
        # Timing stats
        self.generation_times = []
        self.evaluation_times = []
        self.generation = 0
    
    def ask(self) -> np.ndarray:
        """Generate new population"""
        start_time = time.perf_counter()
        
        # Sample new population
        z = np.random.randn(self.population_size, self.dim)
        y = z @ self.invsqrtC.T
        population = self.mean + self.sigma * y
        
        self.generation_times.append(time.perf_counter() - start_time)
        
        return population
    
    def tell(self, population: np.ndarray, fitness: np.ndarray):
        """Update distribution based on fitness"""
        start_time = time.perf_counter()
        
        # Sort by fitness
        indices = np.argsort(fitness)
        population = population[indices]
        fitness = fitness[indices]
        
        # Select best individuals
        selected = population[:self.mu]
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = self.weights @ selected
        
        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  (self.mean - old_mean) / self.sigma @ self.invsqrtC
        
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < \
               1.4 + 2 / (self.dim + 1) * np.sqrt(self.dim)
        
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  (self.mean - old_mean) / self.sigma
        
        # Update covariance matrix
        artmp = (selected - old_mean) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * artmp.T @ np.diag(self.weights) @ artmp
        
        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        
        # Decomposition of C
        self.C = (self.C + self.C.T) / 2  # Enforce symmetry
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
        self.invsqrtC = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T
        
        self.evaluation_times.append(time.perf_counter() - start_time)
        
        self.generation = getattr(self, 'generation', 0) + 1


class CMAESAgent:
    """CMA-ES agent for manipulation tasks"""
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize CMA-ES agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy parameters (simple linear policy for benchmarking)
        self.param_dim = (state_dim + 1) * action_dim  # Weights + biases
        self.optimizer = CMAESOptimizer(self.param_dim)
        
        # Current policy parameters
        self.params = None
        
    def set_params(self, params: np.ndarray):
        """Set policy parameters"""
        self.params = params
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """Get action from current policy"""
        if self.params is None:
            return np.zeros(self.action_dim)
        
        # Reshape parameters
        W = self.params[:self.state_dim * self.action_dim].reshape(self.action_dim, self.state_dim)
        b = self.params[self.state_dim * self.action_dim:]
        
        # Linear policy
        action = np.tanh(W @ state + b)
        
        return action
    
    def evaluate_policy(self, env: ManipulationEnvironment, params: np.ndarray) -> float:
        """
        Evaluate a policy
        
        Args:
            env: Environment
            params: Policy parameters
            
        Returns:
            Total reward
        """
        self.set_params(params)
        
        total_reward = 0
        state = env.reset()
        
        for _ in range(env.max_steps):
            action = self.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return total_reward


# ============================================================================
# Benchmarking Framework
# ============================================================================

@dataclass
class BenchmarkResults:
    """Results from algorithm benchmark"""
    algorithm: str
    total_time: float
    iterations: int
    avg_iteration_time: float
    avg_inference_time: float
    avg_update_time: float
    final_reward: float
    success_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'algorithm': self.algorithm,
            'total_time': self.total_time,
            'iterations': self.iterations,
            'avg_iteration_time': self.avg_iteration_time,
            'avg_inference_time': self.avg_inference_time,
            'avg_update_time': self.avg_update_time,
            'final_reward': self.final_reward,
            'success_rate': self.success_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent
        }


def benchmark_ppo(num_iterations: int = 100, 
                  num_episodes_per_iter: int = 10) -> BenchmarkResults:
    """
    Benchmark PPO algorithm
    
    Args:
        num_iterations: Number of training iterations
        num_episodes_per_iter: Episodes per iteration
        
    Returns:
        Benchmark results
    """
    print("\n[BENCHMARKING PPO]")
    print("-" * 50)
    
    # Initialize
    env = ManipulationEnvironment()
    agent = PPOAgent(env.state_dim, env.action_dim)
    
    # Track metrics
    start_time = time.perf_counter()
    rewards = []
    successes = []
    process = psutil.Process(os.getpid())
    
    # Training loop
    for iteration in range(num_iterations):
        iter_start = time.perf_counter()
        trajectories = []
        
        # Collect trajectories
        for _ in range(num_episodes_per_iter):
            state = env.reset()
            episode_reward = 0
            
            for _ in range(env.max_steps):
                action, log_prob, value = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                trajectories.append({
                    'state': state,
                    'action': action,
                    'log_prob': log_prob.item(),
                    'reward': reward,
                    'value': value.item()
                })
                
                state = next_state
                episode_reward += reward
                
                if done:
                    successes.append(info['success'])
                    break
            
            rewards.append(episode_reward)
        
        # Update policy
        if len(trajectories) > 0:
            agent.update(trajectories)
        
        # Progress
        if iteration % 20 == 0:
            avg_reward = np.mean(rewards[-20:]) if rewards else 0
            print(f"  Iteration {iteration}: Avg Reward = {avg_reward:.3f}")
    
    # Calculate results
    total_time = time.perf_counter() - start_time
    
    results = BenchmarkResults(
        algorithm="PPO",
        total_time=total_time,
        iterations=num_iterations,
        avg_iteration_time=total_time / num_iterations,
        avg_inference_time=np.mean(agent.inference_times) * 1000 if agent.inference_times else 0,
        avg_update_time=np.mean(agent.update_times) * 1000 if agent.update_times else 0,
        final_reward=np.mean(rewards[-10:]) if rewards else 0,
        success_rate=np.mean(successes) if successes else 0,
        memory_usage_mb=process.memory_info().rss / 1024 / 1024,
        cpu_usage_percent=process.cpu_percent()
    )
    
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Avg iteration time: {results.avg_iteration_time*1000:.2f}ms")
    print(f"  Avg inference time: {results.avg_inference_time:.3f}ms")
    print(f"  Final reward: {results.final_reward:.3f}")
    
    return results


def benchmark_cmaes(num_generations: int = 50,
                    num_evaluations_per_gen: int = 5) -> BenchmarkResults:
    """
    Benchmark CMA-ES algorithm
    
    Args:
        num_generations: Number of generations
        num_evaluations_per_gen: Evaluations per individual
        
    Returns:
        Benchmark results
    """
    print("\n[BENCHMARKING CMA-ES]")
    print("-" * 50)
    
    # Initialize
    env = ManipulationEnvironment()
    agent = CMAESAgent(env.state_dim, env.action_dim)
    
    # Track metrics
    start_time = time.perf_counter()
    rewards = []
    successes = []
    process = psutil.Process(os.getpid())
    
    # Evolution loop
    for generation in range(num_generations):
        gen_start = time.perf_counter()
        
        # Generate population
        population = agent.optimizer.ask()
        fitness = []
        
        # Evaluate population
        for individual in population:
            # Multiple evaluations per individual
            individual_fitness = []
            for _ in range(num_evaluations_per_gen):
                reward = agent.evaluate_policy(env, individual)
                individual_fitness.append(reward)
            
            # Use mean fitness
            mean_fitness = np.mean(individual_fitness)
            fitness.append(-mean_fitness)  # CMA-ES minimizes
            rewards.append(mean_fitness)
            
            # Check success
            agent.set_params(individual)
            state = env.reset()
            for _ in range(env.max_steps):
                action = agent.act(state)
                state, _, done, info = env.step(action)
                if done:
                    successes.append(info['success'])
                    break
        
        # Update distribution
        agent.optimizer.tell(population, np.array(fitness))
        
        # Progress
        if generation % 10 == 0:
            best_fitness = -np.min(fitness)
            print(f"  Generation {generation}: Best Fitness = {best_fitness:.3f}")
    
    # Calculate results
    total_time = time.perf_counter() - start_time
    
    results = BenchmarkResults(
        algorithm="CMA-ES",
        total_time=total_time,
        iterations=num_generations,
        avg_iteration_time=total_time / num_generations,
        avg_inference_time=np.mean(agent.optimizer.generation_times) * 1000 if agent.optimizer.generation_times else 0,
        avg_update_time=np.mean(agent.optimizer.evaluation_times) * 1000 if agent.optimizer.evaluation_times else 0,
        final_reward=np.mean(rewards[-10:]) if rewards else 0,
        success_rate=np.mean(successes) if successes else 0,
        memory_usage_mb=process.memory_info().rss / 1024 / 1024,
        cpu_usage_percent=process.cpu_percent()
    )
    
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Avg generation time: {results.avg_iteration_time*1000:.2f}ms")
    print(f"  Avg inference time: {results.avg_inference_time:.3f}ms")
    print(f"  Final reward: {results.final_reward:.3f}")
    
    return results


def compare_algorithms() -> Dict[str, Any]:
    """
    Compare PPO and CMA-ES algorithms
    
    Returns:
        Comparison results and recommendation
    """
    print("\n" + "="*70)
    print(" PPO vs CMA-ES BENCHMARK")
    print("="*70)
    
    # System info
    print("\n[SYSTEM INFO]")
    print(f"  CPU cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical CPUs: {psutil.cpu_count(logical=True)}")
    print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    
    # Run benchmarks
    print("\n[RUNNING BENCHMARKS]")
    print("This will take approximately 1-2 minutes...")
    
    # Quick warm-up
    print("\nWarming up...")
    env = ManipulationEnvironment()
    for _ in range(10):
        env.reset()
        for _ in range(10):
            env.step(np.random.randn(env.action_dim))
    
    # Benchmark PPO
    ppo_results = benchmark_ppo(num_iterations=100, num_episodes_per_iter=5)
    
    # Benchmark CMA-ES  
    cmaes_results = benchmark_cmaes(num_generations=50, num_evaluations_per_gen=3)
    
    # Compare results
    print("\n" + "="*70)
    print(" COMPARISON RESULTS")
    print("="*70)
    
    comparison = {
        'PPO': ppo_results.to_dict(),
        'CMA-ES': cmaes_results.to_dict()
    }
    
    print(f"\n{'Metric':<25} {'PPO':<20} {'CMA-ES':<20}")
    print("-"*65)
    print(f"{'Total Time (s)':<25} {ppo_results.total_time:<20.2f} {cmaes_results.total_time:<20.2f}")
    print(f"{'Avg Iteration (ms)':<25} {ppo_results.avg_iteration_time*1000:<20.2f} {cmaes_results.avg_iteration_time*1000:<20.2f}")
    print(f"{'Avg Inference (ms)':<25} {ppo_results.avg_inference_time:<20.3f} {cmaes_results.avg_inference_time:<20.3f}")
    print(f"{'Avg Update (ms)':<25} {ppo_results.avg_update_time:<20.3f} {cmaes_results.avg_update_time:<20.3f}")
    print(f"{'Final Reward':<25} {ppo_results.final_reward:<20.3f} {cmaes_results.final_reward:<20.3f}")
    print(f"{'Success Rate':<25} {ppo_results.success_rate:<20.2%} {cmaes_results.success_rate:<20.2%}")
    print(f"{'Memory Usage (MB)':<25} {ppo_results.memory_usage_mb:<20.1f} {cmaes_results.memory_usage_mb:<20.1f}")
    
    # Make recommendation
    print("\n" + "="*70)
    print(" RECOMMENDATION")
    print("="*70)
    
    # Score each algorithm
    ppo_score = 0
    cmaes_score = 0
    
    # Speed comparison
    if ppo_results.avg_iteration_time < cmaes_results.avg_iteration_time:
        ppo_score += 1
        speed_winner = "PPO"
    else:
        cmaes_score += 1
        speed_winner = "CMA-ES"
    
    # Memory comparison
    if ppo_results.memory_usage_mb < cmaes_results.memory_usage_mb:
        ppo_score += 1
        memory_winner = "PPO"
    else:
        cmaes_score += 1
        memory_winner = "CMA-ES"
    
    # Performance comparison
    if ppo_results.final_reward > cmaes_results.final_reward:
        ppo_score += 1
        performance_winner = "PPO"
    else:
        cmaes_score += 1
        performance_winner = "CMA-ES"
    
    winner = "PPO" if ppo_score > cmaes_score else "CMA-ES"
    
    print(f"\n  Speed winner: {speed_winner}")
    print(f"  Memory winner: {memory_winner}")
    print(f"  Performance winner: {performance_winner}")
    print(f"\n  OVERALL WINNER: {winner}")
    
    # Specific recommendations
    print("\n[DETAILED ANALYSIS]")
    
    if winner == "PPO":
        print("\n✓ PPO is recommended for your system because:")
        print("  • Better suited for high-dimensional continuous control")
        print("  • More sample-efficient for manipulation tasks")
        print("  • Provides stable learning with value function")
        print("  • Better real-time performance with neural networks")
        print(f"  • {ppo_results.avg_inference_time:.3f}ms inference time fits 1kHz control")
        
        print("\n  Suggested PPO configuration:")
        print("    • Hidden dimensions: 64 (small network)")
        print("    • Learning rate: 3e-4")
        print("    • Batch size: 64")
        print("    • Clip epsilon: 0.2")
        print("    • Single CPU thread (already optimized)")
        
    else:
        print("\n✓ CMA-ES is recommended for your system because:")
        print("  • Gradient-free optimization (no backprop)")
        print("  • More robust to local optima")
        print("  • Better for sim-to-real transfer")
        print("  • Parallelizable population evaluation")
        print(f"  • {cmaes_results.avg_iteration_time*1000:.2f}ms per generation")
        
        print("\n  Suggested CMA-ES configuration:")
        print("    • Population size: 20")
        print("    • Initial sigma: 0.5")
        print("    • Elite ratio: 0.5")
        print("    • Restart strategy: IPOP")
    
    # Task-specific recommendations
    print("\n[TASK-SPECIFIC RECOMMENDATIONS]")
    
    print("\n  For MANIPULATION tasks (grasping, placing):")
    if ppo_results.avg_inference_time < 1.0:  # Less than 1ms
        print("    → PPO (fast inference for reactive control)")
    else:
        print("    → CMA-ES (more robust for contact-rich tasks)")
    
    print("\n  For NAVIGATION tasks (path planning):")
    print("    → CMA-ES (better global optimization)")
    
    print("\n  For HYBRID systems:")
    print("    → Use PPO for Align/Grasp (learnable)")
    print("    → Use scripted for MoveTo/Place")
    
    # Save results
    with open('algorithm_benchmark_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\n✓ Results saved to 'algorithm_benchmark_results.json'")
    
    return {
        'comparison': comparison,
        'winner': winner,
        'recommendation': {
            'algorithm': winner,
            'reason': f"{speed_winner} for speed, {memory_winner} for memory, {performance_winner} for performance",
            'use_case': 'manipulation' if winner == 'PPO' else 'optimization'
        }
    }


# ============================================================================
# Quick Test for Immediate Decision
# ============================================================================

def quick_test() -> str:
    """
    Quick 30-second test to choose algorithm
    
    Returns:
        Recommended algorithm
    """
    print("\n" + "="*70)
    print(" QUICK ALGORITHM SELECTION TEST (30 seconds)")
    print("="*70)
    
    print("\nRunning quick performance test...")
    
    # Test PPO
    print("\n[Testing PPO...]")
    env = ManipulationEnvironment()
    ppo_agent = PPOAgent(env.state_dim, env.action_dim)
    
    ppo_start = time.perf_counter()
    ppo_inferences = []
    
    for _ in range(100):
        state = np.random.randn(env.state_dim)
        t_start = time.perf_counter()
        action, _, _ = ppo_agent.act(state)
        ppo_inferences.append(time.perf_counter() - t_start)
    
    ppo_time = time.perf_counter() - ppo_start
    ppo_avg = np.mean(ppo_inferences) * 1000
    
    print(f"  PPO: {ppo_avg:.3f}ms per inference")
    
    # Test CMA-ES
    print("\n[Testing CMA-ES...]")
    cmaes_agent = CMAESAgent(env.state_dim, env.action_dim)
    cmaes_optimizer = CMAESOptimizer(cmaes_agent.param_dim)
    
    cmaes_start = time.perf_counter()
    cmaes_generations = []
    
    for _ in range(10):
        t_start = time.perf_counter()
        population = cmaes_optimizer.ask()
        fitness = np.random.randn(len(population))
        cmaes_optimizer.tell(population, fitness)
        cmaes_generations.append(time.perf_counter() - t_start)
    
    cmaes_time = time.perf_counter() - cmaes_start
    cmaes_avg = np.mean(cmaes_generations) * 1000
    
    print(f"  CMA-ES: {cmaes_avg:.3f}ms per generation")
    
    # Decision
    print("\n" + "="*70)
    print(" QUICK TEST RESULTS")
    print("="*70)
    
    if ppo_avg < 1.0 and ppo_avg < cmaes_avg / 10:
        recommendation = "PPO"
        reason = f"PPO inference ({ppo_avg:.3f}ms) is fast enough for 1kHz control"
    elif cmaes_avg < 100:
        recommendation = "CMA-ES"
        reason = f"CMA-ES ({cmaes_avg:.3f}ms/gen) better for robustness"
    else:
        recommendation = "PPO"
        reason = "PPO more suitable for real-time control"
    
    print(f"\n✓ RECOMMENDATION: {recommendation}")
    print(f"  Reason: {reason}")
    
    return recommendation


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" ALGORITHM SELECTION: PPO vs CMA-ES")
    print("="*70)
    
    # Quick test first
    quick_recommendation = quick_test()
    
    # Ask if user wants full benchmark
    print("\n" + "-"*70)
    print("Quick test complete!")
    print(f"Quick recommendation: {quick_recommendation}")
    print("\nRun full benchmark? (Takes ~2 minutes)")
    print("Press Enter to run full benchmark, or Ctrl+C to use quick recommendation")
    
    try:
        input()
        # Run full benchmark
        results = compare_algorithms()
        
        print("\n" + "="*70)
        print(" FINAL RECOMMENDATION")
        print("="*70)
        print(f"\n✓ Use {results['recommendation']['algorithm']} for your manipulation system")
        print(f"  Reason: {results['recommendation']['reason']}")
        
    except KeyboardInterrupt:
        print("\n\nUsing quick recommendation...")
        print(f"✓ Selected: {quick_recommendation}")
    
    print("\n" + "="*70)
    print(" IMPLEMENTATION NOTES")
    print("="*70)
    
    if quick_recommendation == "PPO":
        print("\nPPO Implementation Guide:")
        print("  1. Use the PPONetwork class with 64 hidden units")
        print("  2. Set torch.set_num_threads(1) for CPU optimization")
        print("  3. Use batch size of 64 for updates")
        print("  4. Train Align and Grasp behaviors only")
        print("  5. Keep MoveTo and Place scripted")
    else:
        print("\nCMA-ES Implementation Guide:")
        print("  1. Use population size of 20")
        print("  2. Parameterize only critical behaviors")
        print("  3. Use parallel evaluation if possible")
        print("  4. Apply to Align and Grasp optimization")
        print("  5. Keep navigation deterministic")
    
    print("\n✓ Algorithm selection complete!")