"""
Behavioral Cloning (BC) Training and Benchmarking

This script provides:
1. BC model training from demonstrations
2. Integration with reset system for benchmarking
3. Performance logging and metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pybullet as p
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# BC Model Architecture
# ============================================================================

class BCPolicy(nn.Module):
    """Behavioral Cloning policy network"""
    
    def __init__(self, input_dim: int = 30, output_dim: int = 7, 
                 hidden_dims: List[int] = [128, 64]):
        """
        Initialize BC policy
        
        Args:
            input_dim: Observation dimension
            output_dim: Action dimension (7 for Panda robot)
            hidden_dims: Hidden layer dimensions
        """
        super(BCPolicy, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
        

class DemonstrationDataset(Dataset):
    """Dataset for demonstration data"""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        """
        Initialize dataset
        
        Args:
            observations: Array of observations
            actions: Array of actions
        """
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.observations)
        
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


# ============================================================================
# Demonstration Collector
# ============================================================================

class DemonstrationCollector:
    """Collects expert demonstrations for BC training"""
    
    def __init__(self, physics_client: Optional[int] = None):
        """
        Initialize demonstration collector
        
        Args:
            physics_client: PyBullet client
        """
        self.client = physics_client
        self.demonstrations = []
        self.current_trajectory = []
        
    def collect_manual_demonstration(self) -> Dict[str, Any]:
        """
        Collect demonstration via manual control
        
        Returns:
            Demonstration data
        """
        print("\n" + "="*60)
        print(" MANUAL DEMONSTRATION MODE")
        print("="*60)
        print("Use mouse to control robot")
        print("Press SPACE to start/stop recording")
        print("Press R to reset scene")
        print("Press Q to quit")
        print("="*60)
        
        recording = False
        trajectory = []
        
        while True:
            keys = p.getKeyboardEvents()
            
            # Toggle recording
            if ord(' ') in keys and keys[ord(' ')] == p.KEY_WAS_TRIGGERED:
                recording = not recording
                if recording:
                    print("\n[RECORDING STARTED]")
                    trajectory = []
                else:
                    print("[RECORDING STOPPED]")
                    if trajectory:
                        self.demonstrations.append(trajectory)
                        print(f"Saved trajectory with {len(trajectory)} steps")
                        
            # Reset scene
            if ord('r') in keys and keys[ord('r')] == p.KEY_WAS_TRIGGERED:
                # Reset logic here
                print("[SCENE RESET]")
                
            # Quit
            if ord('q') in keys and keys[ord('q')] == p.KEY_WAS_TRIGGERED:
                print("[EXITING DEMONSTRATION MODE]")
                break
                
            # Record state if recording
            if recording:
                obs = self._get_observation()
                action = self._get_current_action()
                trajectory.append({'observation': obs, 'action': action})
                
            p.stepSimulation()
            time.sleep(1/240)
            
        return {
            'num_demonstrations': len(self.demonstrations),
            'total_steps': sum(len(d) for d in self.demonstrations)
        }
        
    def generate_synthetic_demonstrations(self, num_episodes: int = 50) -> Dict[str, Any]:
        """
        Generate synthetic expert demonstrations
        
        Args:
            num_episodes: Number of demonstration episodes
            
        Returns:
            Demonstration statistics
        """
        logger.info(f"Generating {num_episodes} synthetic demonstrations")
        
        for episode in range(num_episodes):
            trajectory = []
            steps = np.random.randint(50, 200)
            
            for step in range(steps):
                # Generate synthetic observation
                obs = np.random.randn(30).astype(np.float32)
                
                # Generate synthetic expert action (smooth trajectory)
                if step == 0:
                    action = np.random.uniform(-0.5, 0.5, 7)
                else:
                    # Smooth action generation
                    prev_action = trajectory[-1]['action']
                    action = prev_action + np.random.uniform(-0.1, 0.1, 7)
                    action = np.clip(action, -1, 1)
                    
                trajectory.append({
                    'observation': obs,
                    'action': action
                })
                
            self.demonstrations.append(trajectory)
            
            if (episode + 1) % 10 == 0:
                logger.info(f"  Generated {episode + 1}/{num_episodes} demonstrations")
                
        return {
            'num_demonstrations': len(self.demonstrations),
            'total_steps': sum(len(d) for d in self.demonstrations),
            'avg_trajectory_length': np.mean([len(d) for d in self.demonstrations])
        }
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation from environment"""
        # Placeholder - implement actual observation extraction
        return np.random.randn(30).astype(np.float32)
        
    def _get_current_action(self) -> np.ndarray:
        """Get current action from manual control"""
        # Placeholder - implement actual action extraction
        return np.random.uniform(-1, 1, 7).astype(np.float32)
        
    def save_demonstrations(self, path: str = "demonstrations.pkl"):
        """Save demonstrations to file"""
        with open(path, 'wb') as f:
            pickle.dump(self.demonstrations, f)
        logger.info(f"Saved {len(self.demonstrations)} demonstrations to {path}")
        
    def load_demonstrations(self, path: str = "demonstrations.pkl"):
        """Load demonstrations from file"""
        with open(path, 'rb') as f:
            self.demonstrations = pickle.load(f)
        logger.info(f"Loaded {len(self.demonstrations)} demonstrations from {path}")
        
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare demonstrations for training
        
        Returns:
            (observations, actions) arrays
        """
        observations = []
        actions = []
        
        for trajectory in self.demonstrations:
            for step in trajectory:
                observations.append(step['observation'])
                actions.append(step['action'])
                
        return np.array(observations), np.array(actions)


# ============================================================================
# BC Trainer
# ============================================================================

class BCTrainer:
    """Trains BC policy from demonstrations"""
    
    def __init__(self, model: BCPolicy, device: str = 'cpu'):
        """
        Initialize BC trainer
        
        Args:
            model: BC policy model
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 100, early_stopping_patience: int = 10) -> Dict[str, Any]:
        """
        Train BC model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training results
        """
        logger.info("Starting BC training...")
        
        patience_counter = 0
        training_start = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, (obs, actions) in enumerate(train_loader):
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                predictions = self.model(obs)
                loss = self.criterion(predictions, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for obs, actions in val_loader:
                        obs = obs.to(self.device)
                        actions = actions.to(self.device)
                        
                        predictions = self.model(obs)
                        loss = self.criterion(predictions, actions)
                        val_loss += loss.item()
                        
                avg_val_loss = val_loss / len(val_loader)
                self.val_losses.append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.save_checkpoint("best_bc_model.pth")
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            # Logging
            if (epoch + 1) % 10 == 0:
                if val_loader:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}")
                    
        training_time = time.time() - training_start
        
        # Final results
        results = {
            'epochs_trained': epoch + 1,
            'training_time': training_time,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': self.best_val_loss if val_loader else None
        }
        
        logger.info("="*60)
        logger.info(" TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Time: {training_time:.2f}s")
        logger.info(f"Final Train Loss: {results['final_train_loss']:.4f}")
        if results['final_val_loss']:
            logger.info(f"Final Val Loss: {results['final_val_loss']:.4f}")
        logger.info("="*60)
        
        return results
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if len(self.train_losses) > 10:
            # Moving average
            window = 10
            ma_train = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            plt.plot(ma_train, label=f'Train MA ({window})')
            
            if self.val_losses:
                ma_val = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
                plt.plot(ma_val, label=f'Val MA ({window})')
                
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Smoothed Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training curves to {save_path}")
        else:
            plt.show()


# ============================================================================
# BC Inference Wrapper
# ============================================================================

class BCInferenceModel:
    """Wrapper for BC model inference"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize inference model
        
        Args:
            model_path: Path to trained model
            device: Inference device
        """
        self.device = device
        
        # Load model
        self.model = BCPolicy().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Loaded BC model from {model_path}")
        
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action from observation
        
        Args:
            observation: Current observation
            
        Returns:
            Predicted action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.model(obs_tensor).cpu().numpy().squeeze()
            
        return action
        
    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict actions for batch of observations
        
        Args:
            observations: Batch of observations
            
        Returns:
            Predicted actions
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            actions = self.model(obs_tensor).cpu().numpy()
            
        return actions


# ============================================================================
# Benchmark Integration
# ============================================================================

class BCBenchmarkRunner:
    """Runs BC benchmarks with the reset system"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        """
        Initialize benchmark runner
        
        Args:
            api_url: Reset system API URL
        """
        self.api_url = api_url
        self.results_history = []
        
    def run_benchmark(self, model_path: str, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Run BC benchmark via API
        
        Args:
            model_path: Path to BC model
            num_episodes: Number of test episodes
            
        Returns:
            Benchmark results
        """
        import requests
        
        logger.info(f"Running BC benchmark with {num_episodes} episodes")
        
        # Trigger benchmark via API
        response = requests.post(f"{self.api_url}/benchmark", json={
            'num_episodes': num_episodes,
            'model_path': model_path
        })
        
        if response.status_code == 200:
            results = response.json()['results']
            self.results_history.append(results)
            
            # Log summary
            self._log_results(results)
            
            return results
        else:
            logger.error(f"Benchmark failed: {response.text}")
            return None
            
    def _log_results(self, results: Dict[str, Any]):
        """Log benchmark results"""
        logger.info("\n" + "="*60)
        logger.info(" BC BENCHMARK RESULTS")
        logger.info("="*60)
        logger.info(f"Success Rate: {results['success_rate']*100:.1f}%")
        logger.info(f"Total Time: {results['total_time']:.2f}s")
        
        if results.get('avg_time_to_success'):
            logger.info(f"Avg Time to Success: {results['avg_time_to_success']:.2f}s")
            logger.info(f"Avg Steps to Success: {results['avg_steps_to_success']:.0f}")
        else:
            logger.info("No successful episodes")
            
        logger.info(f"Avg Steps: {results['avg_steps']:.0f}")
        logger.info(f"Avg Reward: {results['avg_reward']:.2f}")
        logger.info("="*60)
        
    def compare_results(self, baseline_results: Dict[str, Any], 
                       new_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two benchmark results
        
        Args:
            baseline_results: Baseline benchmark results
            new_results: New benchmark results
            
        Returns:
            Comparison statistics
        """
        comparison = {
            'success_rate_diff': new_results['success_rate'] - baseline_results['success_rate'],
            'time_improvement': baseline_results.get('avg_time_to_success', 0) - new_results.get('avg_time_to_success', 0),
            'steps_improvement': baseline_results.get('avg_steps_to_success', 0) - new_results.get('avg_steps_to_success', 0),
            'reward_improvement': new_results['avg_reward'] - baseline_results['avg_reward']
        }
        
        # Calculate percentage improvements
        if baseline_results['success_rate'] > 0:
            comparison['success_rate_improvement_pct'] = (
                comparison['success_rate_diff'] / baseline_results['success_rate'] * 100
            )
            
        logger.info("\n" + "="*60)
        logger.info(" BENCHMARK COMPARISON")
        logger.info("="*60)
        logger.info(f"Success Rate Change: {comparison['success_rate_diff']*100:+.1f}%")
        
        if comparison.get('time_improvement'):
            logger.info(f"Time Improvement: {comparison['time_improvement']:+.2f}s")
            
        if comparison.get('steps_improvement'):
            logger.info(f"Steps Improvement: {comparison['steps_improvement']:+.0f}")
            
        logger.info(f"Reward Improvement: {comparison['reward_improvement']:+.2f}")
        logger.info("="*60)
        
        return comparison


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main_training_pipeline():
    """Main BC training and benchmarking pipeline"""
    
    print("\n" + "="*70)
    print(" BC TRAINING AND BENCHMARKING PIPELINE")
    print("="*70)
    
    # 1. Collect/Generate Demonstrations
    print("\n[1/4] Generating Demonstrations...")
    collector = DemonstrationCollector()
    demo_stats = collector.generate_synthetic_demonstrations(num_episodes=100)
    print(f"  Generated {demo_stats['num_demonstrations']} demonstrations")
    
    # 2. Prepare Training Data
    print("\n[2/4] Preparing Training Data...")
    observations, actions = collector.prepare_training_data()
    
    # Split data
    split_idx = int(0.8 * len(observations))
    train_obs, val_obs = observations[:split_idx], observations[split_idx:]
    train_act, val_act = actions[:split_idx], actions[split_idx:]
    
    # Create datasets
    train_dataset = DemonstrationDataset(train_obs, train_act)
    val_dataset = DemonstrationDataset(val_obs, val_act)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # 3. Train BC Model
    print("\n[3/4] Training BC Model...")
    model = BCPolicy()
    trainer = BCTrainer(model)
    
    train_results = trainer.train(
        train_loader, 
        val_loader,
        epochs=50,
        early_stopping_patience=10
    )
    
    # Save model
    trainer.save_checkpoint("models/bc_model.pth")
    print(f"  Training complete in {train_results['training_time']:.2f}s")
    
    # 4. Run Benchmark
    print("\n[4/4] Running Benchmark...")
    benchmark_runner = BCBenchmarkRunner()
    
    # Run baseline (random policy)
    print("\n  Running baseline benchmark (random policy)...")
    baseline_results = benchmark_runner.run_benchmark(None, num_episodes=10)
    
    # Run BC benchmark
    print("\n  Running BC benchmark...")
    bc_results = benchmark_runner.run_benchmark("models/bc_model.pth", num_episodes=10)
    
    # Compare results
    if baseline_results and bc_results:
        comparison = benchmark_runner.compare_results(baseline_results, bc_results)
    
    # Plot training curves
    trainer.plot_training_curves("training_curves.png")
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BC Training and Benchmarking")
    parser.add_argument('--mode', choices=['train', 'benchmark', 'demo'], 
                       default='train', help='Execution mode')
    parser.add_argument('--model', type=str, default='models/bc_model.pth',
                       help='Model path')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of benchmark episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main_training_pipeline()
        
    elif args.mode == 'benchmark':
        # Run benchmark only
        runner = BCBenchmarkRunner()
        results = runner.run_benchmark(args.model, args.episodes)
        
    elif args.mode == 'demo':
        # Collect demonstrations
        collector = DemonstrationCollector()
        
        print("\nDemo collection modes:")
        print("1. Manual (GUI required)")
        print("2. Synthetic")
        
        choice = input("\nSelect mode (1-2): ").strip()
        
        if choice == '1':
            stats = collector.collect_manual_demonstration()
        else:
            num = int(input("Number of synthetic demos: "))
            stats = collector.generate_synthetic_demonstrations(num)
            
        # Save demonstrations
        collector.save_demonstrations("demonstrations.pkl")
        print(f"\nSaved {stats['num_demonstrations']} demonstrations")