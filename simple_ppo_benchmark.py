"""
Simple PPO timing benchmark without Gym dependency.

This script simulates PPO training to measure performance on your system.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
from datetime import datetime


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
        print(f"CPU Frequency: {cpu_freq.current:.2f} MHz")
    
    # Memory info
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    print(f"Memory Usage: {mem.percent}%")
    
    # PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print("="*70 + "\n")


class SimplePPONetwork(nn.Module):
    """Minimal PPO network for benchmarking."""
    
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.actor(x), self.critic(x)


def simulate_ppo_training(
    timesteps=1000,
    batch_size=16,
    n_epochs=8,
    hidden_dim=64,
    device='cpu'
):
    """
    Simulate PPO training loop to measure performance.
    
    This mimics the computational cost of PPO without requiring an environment.
    """
    print(f"\nConfiguration:")
    print(f"  Network size: 2x{hidden_dim} units")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per update: {n_epochs}")
    print(f"  Total timesteps: {timesteps}")
    print(f"  Device: {device}")
    
    # Create network
    model = SimplePPONetwork(input_dim=4, hidden_dim=hidden_dim, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Simulate training data
    n_updates = timesteps // batch_size
    
    start_time = time.time()
    
    for update in range(n_updates):
        # Simulate collecting rollout data
        observations = torch.randn(batch_size, 4, device=device)
        actions = torch.randint(0, 2, (batch_size,), device=device)
        rewards = torch.randn(batch_size, device=device)
        advantages = torch.randn(batch_size, device=device)
        
        # PPO update loop
        for epoch in range(n_epochs):
            # Forward pass
            action_probs, values = model(observations)
            
            # Simulate loss calculation
            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
            ratio = torch.exp(action_log_probs - action_log_probs.detach())
            
            # Policy loss (clipped PPO objective)
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values.squeeze(), rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    training_time = time.time() - start_time
    
    return training_time, timesteps / training_time


def benchmark_minimal_ppo():
    """Benchmark minimal PPO configuration."""
    print("\n" + "="*70)
    print("BENCHMARKING MINIMAL PPO (Ultra-light)")
    print("="*70)
    
    time_taken, throughput = simulate_ppo_training(
        timesteps=500,
        batch_size=8,
        n_epochs=4,
        hidden_dim=32,
        device='cpu'
    )
    
    print(f"\nResults:")
    print(f"  Training time: {time_taken:.2f} seconds")
    print(f"  Throughput: {throughput:.0f} timesteps/second")
    
    return time_taken


def benchmark_standard_ppo():
    """Benchmark standard lightweight PPO."""
    print("\n" + "="*70)
    print("BENCHMARKING STANDARD LIGHTWEIGHT PPO")
    print("="*70)
    
    time_taken, throughput = simulate_ppo_training(
        timesteps=1000,
        batch_size=16,
        n_epochs=8,
        hidden_dim=64,
        device='cpu'
    )
    
    print(f"\nResults:")
    print(f"  Training time: {time_taken:.2f} seconds")
    print(f"  Throughput: {throughput:.0f} timesteps/second")
    
    return time_taken


def benchmark_cmaes_simulation():
    """Simulate CMA-ES optimization timing."""
    print("\n" + "="*70)
    print("SIMULATING CMA-ES OPTIMIZATION")
    print("="*70)
    
    try:
        import cma
        
        # Simple optimization problem
        dim = 12  # 4 waypoints × 3 dimensions
        
        def cost_function(x):
            time.sleep(0.001)  # Simulate evaluation time
            return np.sum(x**2)
        
        print(f"Configuration:")
        print(f"  Dimensions: {dim}")
        print(f"  Population size: 8")
        print(f"  Iterations: 20")
        
        x0 = np.random.randn(dim) * 0.5
        es = cma.CMAEvolutionStrategy(x0, 0.5, {
            'maxiter': 20,
            'popsize': 8,
            'verbose': -9
        })
        
        start_time = time.time()
        
        for _ in range(20):
            X = es.ask()
            fitness = [cost_function(x) for x in X]
            es.tell(X, fitness)
            if es.stop():
                break
        
        cma_time = time.time() - start_time
        
        print(f"\nResults:")
        print(f"  Optimization time: {cma_time:.2f} seconds")
        print(f"  Time per iteration: {cma_time/20:.3f} seconds")
        
        return cma_time
        
    except ImportError:
        print("CMA-ES not installed. Simulating based on typical performance...")
        print(f"  Estimated time: ~2-5 seconds for 20 iterations")
        return 3.0  # Estimated


def make_recommendation(ppo_minimal_time, ppo_standard_time, cma_time):
    """Make optimization recommendation based on timings."""
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    print(f"\nTiming Summary:")
    print(f"  PPO Minimal (500 steps): {ppo_minimal_time:.2f}s")
    print(f"  PPO Standard (1000 steps): {ppo_standard_time:.2f}s")
    print(f"  PPO Extrapolated (5000 steps): ~{ppo_standard_time * 5:.1f}s")
    
    if cma_time:
        print(f"  CMA-ES (20 iterations): {cma_time:.2f}s")
    
    # Decision logic
    ppo_5k_estimate = ppo_standard_time * 5
    
    if ppo_5k_estimate > 30:
        print(f"\n⚠️ PPO training would take >{30}s for meaningful results")
        print("✅ RECOMMENDATION: Use CMA-ES for optimization")
        recommendation = "CMA-ES"
        
        print("""
Suggested approach:
1. Use CMA-ES for waypoint/trajectory optimization
2. Use the optimization.py module with:
   - budget_iters=20-50
   - popsize=8-16
3. Refine with behavior cloning if needed
        """)
        
    elif ppo_5k_estimate > 15:
        print(f"\n⚡ PPO training is moderate (~{ppo_5k_estimate:.0f}s for 5k steps)")
        print("✅ RECOMMENDATION: Hybrid approach")
        recommendation = "HYBRID"
        
        print("""
Suggested approach:
1. Start with CMA-ES for quick trajectory optimization (5-10 iterations)
2. Generate expert demonstrations from optimized trajectory
3. Use behavior cloning for fast policy learning
4. Optional: Brief PPO fine-tuning (1-2k steps)
        """)
        
    else:
        print(f"\n🚀 PPO is fast on your system (~{ppo_5k_estimate:.0f}s for 5k steps)")
        print("✅ RECOMMENDATION: You can use either approach")
        recommendation = "FLEXIBLE"
        
        print("""
Suggested approaches:
- For trajectory optimization: CMA-ES (fast convergence)
- For policy learning: PPO (good exploration)
- For best results: Combine both methods
        """)
    
    return recommendation


def main():
    """Run the complete benchmark."""
    print("="*70)
    print("COGNIFORGE OPTIMIZATION BENCHMARK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Print system info
    print_system_info()
    
    # Run benchmarks
    print("Starting benchmarks...\n")
    
    # Test 1: Minimal PPO
    ppo_minimal_time = benchmark_minimal_ppo()
    
    # Test 2: Standard PPO
    ppo_standard_time = benchmark_standard_ppo()
    
    # Test 3: CMA-ES
    cma_time = benchmark_cmaes_simulation()
    
    # Make recommendation
    recommendation = make_recommendation(ppo_minimal_time, ppo_standard_time, cma_time)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'cpu_cores': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'pytorch_device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'timings': {
            'ppo_minimal_500': ppo_minimal_time,
            'ppo_standard_1000': ppo_standard_time,
            'ppo_estimated_5000': ppo_standard_time * 5,
            'cma_20_iterations': cma_time
        },
        'recommendation': recommendation
    }
    
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to benchmark_results.json")
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    
    return recommendation


if __name__ == "__main__":
    try:
        recommendation = main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        print("\nQuick assessment: If this script takes >10s to run, use CMA-ES instead of PPO.")