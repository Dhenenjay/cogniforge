#!/usr/bin/env python3
"""
Simple Behavioral Cloning Training Badge Example

Shows the 'Live Training: ON' badge when BC training starts.
"""

import time
import numpy as np
from cogniforge.learning.behavioral_cloning import (
    BCConfig, BCPolicy, BCTrainer, TrainingBadge,
    create_linear_toy_dataset
)

def show_badge_styles():
    """Show all available badge styles."""
    print("\n" + "="*70)
    print("AVAILABLE BADGE STYLES")
    print("="*70)
    
    print("\n1. DEFAULT STYLE:")
    badge = TrainingBadge(style="default")
    badge.show("on")
    print()
    time.sleep(1)
    
    print("\n2. MINIMAL STYLE:")
    badge = TrainingBadge(style="minimal")
    badge.show("on")
    print()
    time.sleep(1)
    
    print("\n3. BOX STYLE:")
    badge = TrainingBadge(style="box")
    badge.show("on")
    time.sleep(1)
    
    print("\n4. ANIMATED STYLE:")
    badge = TrainingBadge(style="animated")
    badge.show("on")
    print("\nAnimating", end="")
    for _ in range(10):
        print(".", end="", flush=True)
        time.sleep(0.3)
    badge.hide()
    print(" Done!")

def quick_training_demo():
    """Quick demo of BC training with live badge."""
    print("\n" + "="*70)
    print("QUICK BC TRAINING WITH LIVE BADGE")
    print("="*70)
    
    # Create simple configuration
    config = BCConfig(
        input_dim=5,
        output_dim=2,
        hidden_dims=[16, 16],
        epochs=10,
        batch_size=16,
        learning_rate=0.01,
        log_interval=5,
        verbose=False  # Suppress verbose logs to see badge clearly
    )
    
    # Create toy dataset
    print("\nPreparing dataset...")
    train_data, _ = create_linear_toy_dataset(
        n_samples=100,
        input_dim=config.input_dim,
        output_dim=config.output_dim
    )
    
    # Create model and trainer
    policy = BCPolicy(config)
    trainer = BCTrainer(policy, config, show_badge=True)
    
    print("\n" + "-"*70)
    print("TRAINING STARTING - WATCH FOR THE BADGE!")
    print("-"*70)
    print()
    
    # Train with badge
    history = trainer.train(train_data, show_live_badge=True)
    
    # Show summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✅ Successfully trained for {len(history)} epochs")
    print(f"📉 Final loss: {history[-1].train_loss:.4f}")

def simulate_training_progress():
    """Simulate training with progress updates."""
    print("\n" + "="*70)
    print("SIMULATED TRAINING WITH PROGRESS")
    print("="*70)
    
    badge = TrainingBadge(style="default")
    
    # Show badge when training starts
    print("\n🚀 Starting Behavioral Cloning training...")
    badge.show("on", "Initializing model...")
    time.sleep(2)
    
    # Simulate training epochs
    total_epochs = 20
    for epoch in range(1, total_epochs + 1):
        # Simulate decreasing loss
        loss = 1.0 * np.exp(-epoch/10) + np.random.uniform(-0.01, 0.01)
        
        # Update badge with progress
        badge.update_progress(epoch, total_epochs, loss)
        
        # Simulate training time
        time.sleep(0.2)
    
    # Training complete
    print()  # New line after progress bar
    badge.show("complete", f"Model trained successfully! Final loss: {loss:.4f}")
    time.sleep(2)
    
    print("\n✨ Training simulation complete!")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                   🟢 BEHAVIORAL CLONING TRAINING BADGE 🟢                  ║
║                                                                            ║
║  This example demonstrates the 'Live Training: ON' badge that appears     ║
║  when Behavioral Cloning training starts.                                 ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nSelect demo:")
    print("1. Show all badge styles")
    print("2. Quick BC training with live badge")
    print("3. Simulated training with progress")
    print("4. Run all demos")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        show_badge_styles()
    elif choice == "2":
        quick_training_demo()
    elif choice == "3":
        simulate_training_progress()
    elif choice == "4":
        show_badge_styles()
        input("\nPress Enter to continue...")
        quick_training_demo()
        input("\nPress Enter to continue...")
        simulate_training_progress()
    else:
        # Default: show simulated progress
        simulate_training_progress()
    
    print("\n" + "="*70)
    print("✅ Demo complete! The badge appears whenever BC training starts.")
    print("="*70)