#!/usr/bin/env python3
"""
Behavioral Cloning Live Training Badge Demo

Demonstrates the live training badge that appears when BC training starts,
showing real-time progress updates during training.
"""

import torch
import numpy as np
import time
import logging
from cogniforge.learning.behavioral_cloning import (
    BCConfig, BCPolicy, BCDataset, BCTrainer, TrainingBadge,
    create_linear_toy_dataset
)

# Configure logging to see training updates
logging.basicConfig(level=logging.INFO, format='%(message)s')


def demo_badge_styles():
    """Demonstrate different badge styles."""
    print("\n" + "="*70)
    print("BADGE STYLE SHOWCASE")
    print("="*70)
    
    styles = ["default", "minimal", "box", "animated"]
    
    for style in styles:
        print(f"\n{style.upper()} Style:")
        print("-" * 40)
        
        badge = TrainingBadge(style=style, position="inline")
        
        # Show different states
        states = ["on", "paused", "complete", "error", "off"]
        for state in states:
            badge.show(state)
            time.sleep(0.8)
            print()  # New line for next state
    
    print("\n✅ Badge showcase complete!")


def demo_live_training():
    """Demonstrate live training with badge updates."""
    print("\n" + "="*70)
    print("LIVE TRAINING WITH BADGE DEMO")
    print("="*70)
    
    # Configuration for quick demo
    config = BCConfig(
        input_dim=10,
        output_dim=3,
        hidden_dims=[32, 32],
        batch_size=32,
        learning_rate=1e-2,
        epochs=20,
        log_interval=5,
        seed=42,
        verbose=True
    )
    
    # Create toy dataset
    print("\n📊 Creating toy dataset...")
    train_data, val_data = create_linear_toy_dataset(
        n_samples=500,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        seed=42
    )
    
    # Create policy
    policy = BCPolicy(config)
    
    # Create trainer with badge enabled
    print("\n🚀 Initializing trainer with live badge...")
    trainer = BCTrainer(policy, config, show_badge=True)
    
    # Train with live badge
    print("\n" + "="*70)
    print("Watch for the 'Live Training: ON' badge above!")
    print("="*70)
    time.sleep(1)
    
    history = trainer.train(train_data, val_data, show_live_badge=True)
    
    # Show results
    print("\n" + "="*70)
    print("TRAINING RESULTS")
    print("="*70)
    print(f"Initial loss: {history[0].train_loss:.6f}")
    print(f"Final loss: {history[-1].train_loss:.6f}")
    print(f"Improvement: {(1 - history[-1].train_loss/history[0].train_loss)*100:.1f}%")


def demo_animated_badge():
    """Show animated badge during simulated training."""
    print("\n" + "="*70)
    print("ANIMATED BADGE DEMO")
    print("="*70)
    
    badge = TrainingBadge(style="animated", position="top")
    
    print("\nSimulating training with animated badge...")
    badge.show("on")
    
    # Simulate training progress
    for epoch in range(1, 11):
        loss = 10.0 * np.exp(-epoch/5) + np.random.normal(0, 0.1)
        badge.update_progress(epoch, 10, loss)
        time.sleep(0.5)
    
    print()  # New line after progress
    badge.show("complete", "Training simulation complete!")
    time.sleep(2)
    badge.hide()
    
    print("\n✅ Animation demo complete!")


def demo_custom_badge_integration():
    """Show how to integrate badge with custom training loops."""
    print("\n" + "="*70)
    print("CUSTOM TRAINING LOOP WITH BADGE")
    print("="*70)
    
    # Create badge
    badge = TrainingBadge(style="box", position="top")
    
    # Show initial state
    badge.show("on")
    time.sleep(1)
    
    print("\nCustom training loop running...")
    
    # Simulate custom training
    for epoch in range(1, 6):
        # Training phase
        print(f"\nEpoch {epoch}/5:")
        
        # Simulate batch training
        for batch in range(3):
            print(f"  Processing batch {batch+1}/3...", end="")
            time.sleep(0.3)
            print(" ✓")
        
        # Calculate metrics
        loss = 5.0 / epoch + np.random.normal(0, 0.1)
        
        # Update badge
        badge.update_progress(epoch, 5, loss)
        time.sleep(0.5)
    
    print()
    badge.show("complete")
    time.sleep(2)
    
    print("\n✅ Custom integration complete!")


def demo_error_handling():
    """Demonstrate badge behavior during training errors."""
    print("\n" + "="*70)
    print("ERROR HANDLING WITH BADGE")
    print("="*70)
    
    badge = TrainingBadge(style="default", position="top")
    
    try:
        badge.show("on", "Starting training...")
        time.sleep(1)
        
        # Simulate training
        for epoch in range(1, 4):
            badge.update_progress(epoch, 10, 2.5 - epoch * 0.3)
            time.sleep(0.5)
            
            if epoch == 3:
                # Simulate error
                raise ValueError("Simulated training error!")
        
    except Exception as e:
        print()  # New line
        badge.show("error", f"Training failed: {e}")
        time.sleep(2)
    
    print("\n✅ Error handling demo complete!")


def main():
    """Run all demos."""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║              🟢 BEHAVIORAL CLONING LIVE TRAINING BADGE 🟢          ║
    ║                                                                    ║
    ║    This demo showcases the live training badge that appears       ║
    ║    when Behavioral Cloning training starts, providing real-time   ║
    ║    status updates during the training process.                    ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\nSelect a demo:")
        print("1. Badge Style Showcase")
        print("2. Live Training with Badge")
        print("3. Animated Badge Demo")
        print("4. Custom Training Integration")
        print("5. Error Handling Demo")
        print("6. Run All Demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "1":
            demo_badge_styles()
        elif choice == "2":
            demo_live_training()
        elif choice == "3":
            demo_animated_badge()
        elif choice == "4":
            demo_custom_badge_integration()
        elif choice == "5":
            demo_error_handling()
        elif choice == "6":
            demo_badge_styles()
            input("\nPress Enter to continue...")
            demo_live_training()
            input("\nPress Enter to continue...")
            demo_animated_badge()
            input("\nPress Enter to continue...")
            demo_custom_badge_integration()
            input("\nPress Enter to continue...")
            demo_error_handling()
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()