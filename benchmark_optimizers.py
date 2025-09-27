"""
Benchmark script to compare PPO training time vs CMA-ES optimization.

This script tests both approaches on a simple task to help decide
which optimization path is more suitable for your system.
"""

import time
import gymnasium as gym
import numpy as np
import psutil
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import PPO components
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import CMA-ES components
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("CMA-ES not installed. Install with: pip install cma")


def print_system_info():
    """Print system specifications."""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    
    print(f"CPU Cores: {cpu_count} physical, {cpu_count_logical} logical")
    if cpu_freq:
        print(f"CPU Frequency: {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)")
    
    # Memory info
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    print(f"Memory Usage: {mem.percent}%")
    
    # PyTorch info
    print(f"PyTorch Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print("="*70 + "\n")


def create_test_env():
    """Create a simple test environment."""
    env = gym.make('CartPole-v1')
    return env


def benchmark_ppo_minimal(timesteps=1000, verbose=False):
    """
    Benchmark minimal PPO configuration.
    
    Ultra-light config for fastest possible training.
    """
    print("\n" + "-"*70)
    print("BENCHMARKING PPO (Minimal Config)")
    print("-"*70)
    
    env = create_test_env()
    vec_env = DummyVecEnv([lambda: env])
    
    # Ultra-minimal configuration
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs={
            "net_arch": dict(pi=[32, 32], vf=[32, 32])  # Even smaller network
        },
        learning_rate=3e-4,
        n_steps=64,        # Very small buffer
        batch_size=8,      # Tiny batch
        n_epochs=4,        # Few epochs
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1 if verbose else 0,
        device="cpu"
    )
    
    print(f"Configuration:")
    print(f"  Network: 2x32 units")
    print(f"  Buffer size: 64 steps")
    print(f"  Batch size: 8")
    print(f"  Epochs: 4")
    print(f"  Timesteps: {timesteps}")
    
    # Time the training
    start_time = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    training_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Timesteps/second: {timesteps/training_time:.0f}")
    
    # Quick evaluation
    obs = vec_env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards[0]
        if dones[0]:
            break
    
    print(f"  Sample episode reward: {total_reward:.0f}")
    
    vec_env.close()
    
    return training_time, timesteps/training_time


def benchmark_ppo_standard(timesteps=3000, verbose=False):
    """
    Benchmark standard lightweight PPO configuration.
    """
    print("\n" + "-"*70)
    print("BENCHMARKING PPO (Standard Lightweight)")
    print("-"*70)
    
    env = create_test_env()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Standard lightweight configuration
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        },
        learning_rate=3e-4,
        n_steps=128,
        batch_size=16,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1 if verbose else 0,
        device="cpu"
    )
    
    print(f"Configuration:")
    print(f"  Network: 2x64 units")
    print(f"  Buffer size: 128 steps")
    print(f"  Batch size: 16")
    print(f"  Epochs: 8")
    print(f"  Timesteps: {timesteps}")
    
    # Time the training
    start_time = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    training_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Timesteps/second: {timesteps/training_time:.0f}")
    
    # Quick evaluation
    obs = vec_env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards[0]
        if dones[0]:
            break
    
    print(f"  Sample episode reward: {total_reward:.0f}")
    
    vec_env.close()
    
    return training_time, timesteps/training_time


def benchmark_cmaes(n_iterations=50, popsize=8):
    """
    Benchmark CMA-ES optimization.
    """
    if not CMA_AVAILABLE:
        print("\nCMA-ES not available for benchmarking")
        return None, None
    
    print("\n" + "-"*70)
    print("BENCHMARKING CMA-ES")
    print("-"*70)
    
    # Simple optimization problem (4D waypoint optimization)
    dim = 12  # 4 waypoints × 3 dimensions
    
    def dummy_cost_function(x):
        """Simple quadratic cost for benchmarking."""
        # Simulate trajectory evaluation time
        time.sleep(0.001)  # 1ms per evaluation
        return np.sum(x**2) + np.random.normal(0, 0.1)
    
    print(f"Configuration:")
    print(f"  Dimensions: {dim}")
    print(f"  Population size: {popsize}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Total evaluations: ~{n_iterations * popsize}")
    
    # Setup CMA-ES
    x0 = np.random.randn(dim) * 0.5
    sigma0 = 0.5
    
    opts = {
        'maxiter': n_iterations,
        'popsize': popsize,
        'verb_disp': 0,
        'verb_log': 0,
        'verbose': -9
    }
    
    # Time the optimization
    start_time = time.time()
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    iteration = 0
    while not es.stop() and iteration < n_iterations:
        X = es.ask()
        fitness = [dummy_cost_function(x) for x in X]
        es.tell(X, fitness)
        iteration += 1
    
    optimization_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Optimization time: {optimization_time:.2f} seconds")
    print(f"  Iterations completed: {iteration}")
    print(f"  Time per iteration: {optimization_time/iteration:.3f} seconds")
    print(f"  Final best cost: {es.result.fbest:.4f}")
    
    return optimization_time, iteration


def benchmark_realistic_task():
    """
    Benchmark on a more realistic robotic control task.
    """
    print("\n" + "="*70)
    print("REALISTIC TASK BENCHMARK")
    print("="*70)
    
    # Create a pendulum environment (continuous control)
    env = gym.make('Pendulum-v1')
    
    # Test 1: PPO with very short training
    print("\n1. PPO Quick Training (1000 steps)")
    vec_env = DummyVecEnv([lambda: gym.make('Pendulum-v1')])
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
        n_steps=64,
        batch_size=16,
        n_epochs=5,
        verbose=0,
        device="cpu"
    )
    
    start = time.time()
    model.learn(total_timesteps=1000, progress_bar=False)
    ppo_time = time.time() - start
    
    print(f"  Time: {ppo_time:.2f}s")
    
    # Test 2: CMA-ES for policy optimization
    if CMA_AVAILABLE:
        print("\n2. CMA-ES Policy Optimization (10 iterations)")
        
        def policy_cost(params):
            """Evaluate a policy with given parameters."""
            # Simulate policy evaluation
            time.sleep(0.01)  # 10ms per evaluation
            return np.random.random()  # Random cost for demo
        
        dim = 64 * 2 + 64 * 2  # Approximate parameter count
        x0 = np.zeros(dim)
        
        es = cma.CMAEvolutionStrategy(x0, 0.5, {
            'maxiter': 10,
            'popsize': 6,
            'verbose': -9
        })
        
        start = time.time()
        for _ in range(10):
            X = es.ask()
            fitness = [policy_cost(x) for x in X]
            es.tell(X, fitness)
            if es.stop():
                break
        cma_time = time.time() - start
        
        print(f"  Time: {cma_time:.2f}s")
    
    vec_env.close()
    
    return ppo_time, cma_time if CMA_AVAILABLE else None


def make_recommendation(ppo_times, cma_times):
    """
    Make recommendation based on benchmark results.
    """
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    avg_ppo_time = np.mean([t for t in ppo_times if t is not None])
    
    print(f"\nPPO Performance:")
    print(f"  Average time for minimal training: {avg_ppo_time:.2f}s")
    
    if avg_ppo_time > 30:
        print(f"  ⚠️ PPO training takes >{30}s on your system")
        recommendation = "CMA-ES"
    elif avg_ppo_time > 20:
        print(f"  ⚡ PPO training is moderate ({avg_ppo_time:.1f}s)")
        recommendation = "HYBRID"
    else:
        print(f"  ✅ PPO training is fast ({avg_ppo_time:.1f}s)")
        recommendation = "PPO"
    
    if CMA_AVAILABLE and cma_times:
        avg_cma_time = np.mean([t for t in cma_times if t is not None])
        print(f"\nCMA-ES Performance:")
        print(f"  Average optimization time: {avg_cma_time:.2f}s")
        
        if avg_cma_time < avg_ppo_time * 0.5:
            print(f"  ✅ CMA-ES is significantly faster")
            recommendation = "CMA-ES"
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED APPROACH: {recommendation}")
    print(f"{'='*70}")
    
    if recommendation == "CMA-ES":
        print("""
Recommendation: Use CMA-ES for trajectory optimization
- CMA-ES is faster on your system for small-scale optimization
- Good for waypoint optimization and trajectory planning
- Consider using the optimization.py module with budget_iters=20-50
        """)
    elif recommendation == "PPO":
        print("""
Recommendation: Use PPO for policy learning
- PPO runs efficiently on your system
- Good for learning reactive policies
- Use the lightweight configuration with 3-5k timesteps
        """)
    else:  # HYBRID
        print("""
Recommendation: Use hybrid approach
- Start with CMA-ES for trajectory optimization (fast initial solution)
- Refine with behavior cloning or short PPO training
- Combines benefits of both approaches
        """)
    
    return recommendation


def run_full_benchmark():
    """
    Run complete benchmark suite.
    """
    print("="*70)
    print("COGNIFORGE OPTIMIZER BENCHMARK")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Print system info
    print_system_info()
    
    ppo_times = []
    cma_times = []
    
    # Benchmark 1: Minimal PPO
    print("\n📊 TEST 1: Minimal PPO Configuration")
    try:
        time_taken, throughput = benchmark_ppo_minimal(timesteps=1000)
        ppo_times.append(time_taken)
        
        if time_taken > 30:
            print("\n⚠️ PPO is slow on this system. Testing smaller config...")
            time_taken, throughput = benchmark_ppo_minimal(timesteps=500)
            ppo_times.append(time_taken * 2)  # Extrapolate to 1000 steps
    except Exception as e:
        print(f"Error in PPO minimal benchmark: {e}")
    
    # Benchmark 2: Standard PPO (if minimal was fast enough)
    if ppo_times and ppo_times[-1] < 20:
        print("\n📊 TEST 2: Standard Lightweight PPO")
        try:
            time_taken, throughput = benchmark_ppo_standard(timesteps=3000)
            ppo_times.append(time_taken)
        except Exception as e:
            print(f"Error in PPO standard benchmark: {e}")
    else:
        print("\n⏭️ Skipping standard PPO (system too slow)")
    
    # Benchmark 3: CMA-ES
    if CMA_AVAILABLE:
        print("\n📊 TEST 3: CMA-ES Optimization")
        try:
            time_taken, iterations = benchmark_cmaes(n_iterations=30, popsize=8)
            if time_taken:
                cma_times.append(time_taken)
        except Exception as e:
            print(f"Error in CMA-ES benchmark: {e}")
    
    # Benchmark 4: Realistic task comparison
    print("\n📊 TEST 4: Realistic Task Comparison")
    try:
        ppo_time, cma_time = benchmark_realistic_task()
        if ppo_time:
            ppo_times.append(ppo_time)
        if cma_time:
            cma_times.append(cma_time)
    except Exception as e:
        print(f"Error in realistic benchmark: {e}")
    
    # Make recommendation
    recommendation = make_recommendation(ppo_times, cma_times)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'cpu_cores': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'ppo_times': ppo_times,
        'cma_times': cma_times,
        'recommendation': recommendation
    }
    
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to benchmark_results.json")
    print(f"⏱️ Total benchmark time: {sum(ppo_times + cma_times):.1f} seconds")
    
    return recommendation


def quick_test():
    """
    Quick 10-second test to get immediate feedback.
    """
    print("\n" + "="*70)
    print("QUICK PPO TEST (10 seconds max)")
    print("="*70)
    
    env = gym.make('CartPole-v1')
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs={"net_arch": dict(pi=[32, 32], vf=[32, 32])},
        n_steps=32,
        batch_size=8,
        n_epochs=2,
        verbose=0,
        device="cpu"
    )
    
    print("Training PPO for 500 timesteps...")
    start = time.time()
    model.learn(total_timesteps=500, progress_bar=True)
    elapsed = time.time() - start
    
    print(f"\nTime taken: {elapsed:.2f} seconds")
    print(f"Throughput: {500/elapsed:.0f} timesteps/second")
    
    if elapsed > 10:
        print("\n⚠️ PPO is SLOW on your system!")
        print("✅ RECOMMENDATION: Use CMA-ES for optimization")
    elif elapsed > 5:
        print("\n⚡ PPO speed is MODERATE")
        print("✅ RECOMMENDATION: Use hybrid approach (CMA-ES + BC refinement)")
    else:
        print("\n🚀 PPO is FAST on your system!")
        print("✅ RECOMMENDATION: You can use either PPO or CMA-ES effectively")
    
    vec_env.close()
    return elapsed


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run quick test only
        quick_test()
    else:
        # Run full benchmark
        print("Starting benchmark... This will take 1-2 minutes.\n")
        print("For a quick 10-second test, run: python benchmark_optimizers.py --quick\n")
        
        try:
            recommendation = run_full_benchmark()
            
            print("\n" + "="*70)
            print("BENCHMARK COMPLETE!")
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user.")
            print("Running quick test instead...")
            quick_test()
        except Exception as e:
            print(f"\nError during benchmark: {e}")
            print("Running quick test as fallback...")
            quick_test()