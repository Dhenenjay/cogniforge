#!/usr/bin/env python3
"""Quick algorithm timing test for PPO vs CMA-ES"""

import sys
sys.path.append('src')

from algorithm_benchmark import quick_test

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" QUICK PPO vs CMA-ES TIMING TEST")
    print("="*70)
    
    recommendation = quick_test()
    
    print("\n" + "="*70)
    print(" RESULTS ON YOUR MACHINE")
    print("="*70)
    print(f"\n✓ SELECTED: {recommendation}")
    
    if recommendation == "PPO":
        print("\nWhy PPO was chosen:")
        print("  • Sub-millisecond inference latency")
        print("  • Suitable for 1kHz control loop")
        print("  • Better for reactive manipulation")
        print("  • Efficient with single-thread PyTorch")
        
        print("\nConfiguration for your system:")
        print("  • Network: 2 layers, 64 hidden units")
        print("  • Batch size: 64")
        print("  • Learning rate: 3e-4")
        print("  • Update frequency: Every 100 steps")
    else:
        print("\nWhy CMA-ES was chosen:")
        print("  • More robust to local optima")
        print("  • Better for contact-rich tasks")
        print("  • No gradient computation needed")
        print("  • Good for parameter optimization")
        
        print("\nConfiguration for your system:")
        print("  • Population size: 20")
        print("  • Initial sigma: 0.5")
        print("  • Evaluations per generation: 5")
    
    print("\n✓ Algorithm selection complete!")