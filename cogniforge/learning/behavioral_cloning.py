"""
Behavioral Cloning (BC) Module for CogniForge

Implements imitation learning via behavioral cloning with support for
various network architectures and training configurations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import time
import logging
from pathlib import Path
import sys
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BCConfig:
    """Configuration for Behavioral Cloning training."""
    # Model architecture
    input_dim: int = 10
    output_dim: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = 'relu'  # 'relu', 'tanh', 'elu'
    dropout_rate: float = 0.0
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    
    # Optimization
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    scheduler: Optional[str] = None  # 'cosine', 'step', 'exponential'
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss and regularization
    loss_type: str = 'mse'  # 'mse', 'l1', 'huber'
    l2_reg: float = 0.0
    gradient_clip: Optional[float] = None
    
    # Device and precision
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp: bool = False  # Automatic mixed precision
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 10
    verbose: bool = True
    seed: Optional[int] = None


@dataclass
class BCMetrics:
    """Metrics tracked during BC training."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    time_elapsed: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'learning_rate': self.learning_rate,
            'time_elapsed': self.time_elapsed
        }


class BCDataset(Dataset):
    """Dataset for Behavioral Cloning."""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 transform=None, normalize=True):
        """
        Initialize BC dataset.
        
        Args:
            observations: (N, obs_dim) array of observations
            actions: (N, act_dim) array of expert actions
            transform: Optional data transformation
            normalize: Whether to normalize inputs
        """
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        self.transform = transform
        
        if normalize:
            self.obs_mean = self.observations.mean(dim=0)
            self.obs_std = self.observations.std(dim=0) + 1e-6
            self.observations = (self.observations - self.obs_mean) / self.obs_std
            
            self.act_mean = self.actions.mean(dim=0)
            self.act_std = self.actions.std(dim=0) + 1e-6
        else:
            self.obs_mean = torch.zeros(observations.shape[1])
            self.obs_std = torch.ones(observations.shape[1])
            self.act_mean = torch.zeros(actions.shape[1])
            self.act_std = torch.ones(actions.shape[1])
        
        assert len(self.observations) == len(self.actions), \
            f"Observation and action counts must match: {len(self.observations)} != {len(self.actions)}"
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self.observations[idx]
        act = self.actions[idx]
        
        if self.transform:
            obs = self.transform(obs)
        
        return obs, act
    
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions to original scale."""
        return actions * self.act_std + self.act_mean


class BCPolicy(nn.Module):
    """Neural network policy for Behavioral Cloning."""
    
    def __init__(self, config: BCConfig):
        """
        Initialize BC policy network.
        
        Args:
            config: BC configuration
        """
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        input_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Activation
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'tanh':
                layers.append(nn.Tanh())
            elif config.activation == 'elu':
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {config.activation}")
            
            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Predicted actions
        """
        return self.network(obs)
    
    def get_action(self, obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get action for a single observation (inference mode).
        
        Args:
            obs: Single observation
            
        Returns:
            Action as numpy array
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            action = self.forward(obs)
        
        return action.cpu().numpy().squeeze()


class TrainingBadge:
    """Displays live training status badge in the UI."""
    
    # Badge styles
    STYLES = {
        "default": {
            "on": "🟢 Live Training: ON",
            "off": "🔴 Live Training: OFF",
            "paused": "🟡 Live Training: PAUSED",
            "complete": "✅ Training: COMPLETE",
            "error": "❌ Training: ERROR"
        },
        "minimal": {
            "on": "● Training",
            "off": "○ Idle",
            "paused": "◐ Paused",
            "complete": "✓ Done",
            "error": "✗ Error"
        },
        "box": {
            "on": "┌─────────────────────┐\n│ 🟢 LIVE TRAINING: ON │\n└─────────────────────┘",
            "off": "┌──────────────────────┐\n│ 🔴 LIVE TRAINING: OFF │\n└──────────────────────┘",
            "paused": "┌────────────────────────┐\n│ 🟡 LIVE TRAINING: PAUSED│\n└────────────────────────┘",
            "complete": "┌───────────────────────┐\n│ ✅ TRAINING: COMPLETE │\n└───────────────────────┘",
            "error": "┌──────────────────┐\n│ ❌ TRAINING: ERROR│\n└──────────────────┘"
        },
        "animated": {
            "on": ["⟳ Training...", "⟲ Training...", "◉ Training...", "◎ Training..."],
            "off": "○ Idle",
            "paused": "◐ Paused",
            "complete": "✓ Complete",
            "error": "✗ Error"
        }
    }
    
    def __init__(self, style: str = "default", position: str = "top", 
                 auto_hide: bool = False, update_interval: float = 0.5):
        """
        Initialize training badge.
        
        Args:
            style: Badge style ("default", "minimal", "box", "animated")
            position: Display position ("top", "bottom", "inline")
            auto_hide: Automatically hide when training stops
            update_interval: Update interval for animated badges
        """
        self.style = style
        self.position = position
        self.auto_hide = auto_hide
        self.update_interval = update_interval
        self.status = "off"
        self.animation_frame = 0
        self.last_update = time.time()
        self._stop_animation = False
        self._animation_thread = None
        
    def show(self, status: str = "on", message: Optional[str] = None):
        """Display the badge with given status."""
        self.status = status
        badge_text = self._get_badge_text(status)
        
        if self.position == "top":
            # Clear line and print at top
            print("\r" + " " * 80 + "\r", end="")  # Clear line
            print(badge_text, end="\n" if "\n" in badge_text else "\r")
        elif self.position == "bottom":
            print("\n" + badge_text)
        else:  # inline
            print(badge_text, end=" ")
        
        if message:
            print(f" {message}")
        
        # Start animation if needed
        if status == "on" and self.style == "animated":
            self._start_animation()
        else:
            self._stop_animation_thread()
        
        sys.stdout.flush()
    
    def _get_badge_text(self, status: str) -> str:
        """Get badge text for current status and style."""
        style_dict = self.STYLES.get(self.style, self.STYLES["default"])
        
        if self.style == "animated" and status == "on":
            # Return current animation frame
            frames = style_dict["on"]
            return frames[self.animation_frame % len(frames)]
        else:
            return style_dict.get(status, style_dict["off"])
    
    def _start_animation(self):
        """Start animation thread for animated badges."""
        if self._animation_thread and self._animation_thread.is_alive():
            return
        
        self._stop_animation = False
        self._animation_thread = threading.Thread(target=self._animate)
        self._animation_thread.daemon = True
        self._animation_thread.start()
    
    def _animate(self):
        """Animation loop for animated badges."""
        while not self._stop_animation:
            time.sleep(self.update_interval)
            if self._stop_animation:
                break
            
            self.animation_frame += 1
            badge_text = self._get_badge_text("on")
            
            # Update display
            print("\r" + " " * 40 + "\r", end="")  # Clear line
            print(badge_text, end="\r")
            sys.stdout.flush()
    
    def _stop_animation_thread(self):
        """Stop animation thread."""
        self._stop_animation = True
        if self._animation_thread:
            self._animation_thread.join(timeout=0.5)
    
    def hide(self):
        """Hide the badge."""
        self._stop_animation_thread()
        if self.position == "top":
            print("\r" + " " * 80 + "\r", end="")
        sys.stdout.flush()
    
    def update_progress(self, epoch: int, total_epochs: int, loss: float):
        """Update badge with training progress."""
        progress = epoch / total_epochs * 100
        progress_bar = self._create_progress_bar(progress, width=10)
        
        badge = self._get_badge_text("on")
        status_line = f"{badge} │ Epoch {epoch}/{total_epochs} │ {progress_bar} {progress:.0f}% │ Loss: {loss:.4f}"
        
        print("\r" + " " * 100 + "\r", end="")  # Clear line
        print(status_line, end="\r")
        sys.stdout.flush()
    
    def _create_progress_bar(self, progress: float, width: int = 10) -> str:
        """Create a mini progress bar."""
        filled = int(progress / 100 * width)
        empty = width - filled
        return "█" * filled + "░" * empty


class BCTrainer:
    """Trainer for Behavioral Cloning."""
    
    def __init__(self, policy: BCPolicy, config: BCConfig, show_badge: bool = True):
        """
        Initialize BC trainer.
        
        Args:
            policy: Policy network to train
            config: Training configuration
        """
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.show_badge = show_badge
        
        # Initialize training badge
        self.badge = TrainingBadge(style="default", position="top") if show_badge else None
        
        # Set seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.seed)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Training history
        self.history: List[BCMetrics] = []
        self.best_val_loss = float('inf')
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        params = self.policy.parameters()
        
        if self.config.optimizer == 'adam':
            return optim.Adam(params, lr=self.config.learning_rate,
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(params, lr=self.config.learning_rate,
                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(params, lr=self.config.learning_rate,
                           weight_decay=self.config.weight_decay,
                           momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler if specified."""
        if self.config.scheduler is None:
            return None
        
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                **self.config.scheduler_params
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_params.get('step_size', 30),
                gamma=self.config.scheduler_params.get('gamma', 0.1)
            )
        elif self.config.scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.scheduler_params.get('gamma', 0.95)
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        if self.config.loss_type == 'mse':
            return nn.MSELoss()
        elif self.config.loss_type == 'l1':
            return nn.L1Loss()
        elif self.config.loss_type == 'huber':
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.policy.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (obs, act) in enumerate(train_loader):
            obs = obs.to(self.device)
            act = act.to(self.device)
            
            # Mixed precision training
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.policy(obs)
                    loss = self.criterion(pred, act)
                    
                    # Add L2 regularization if specified
                    if self.config.l2_reg > 0:
                        l2_loss = sum(p.pow(2).sum() for p in self.policy.parameters())
                        loss = loss + self.config.l2_reg * l2_loss
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                pred = self.policy(obs)
                loss = self.criterion(pred, act)
                
                # Add L2 regularization if specified
                if self.config.l2_reg > 0:
                    l2_loss = sum(p.pow(2).sum() for p in self.policy.parameters())
                    loss = loss + self.config.l2_reg * l2_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate on validation data.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for obs, act in val_loader:
                obs = obs.to(self.device)
                act = act.to(self.device)
                
                pred = self.policy(obs)
                loss = self.criterion(pred, act)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataset: BCDataset, 
              val_dataset: Optional[BCDataset] = None,
              early_stopping: int = 0,
              show_live_badge: Optional[bool] = None) -> List[BCMetrics]:
        """
        Train the BC policy.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            early_stopping: Patience for early stopping (0 = disabled)
            
        Returns:
            Training history
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        start_time = time.time()
        patience_counter = 0
        
        # Show training badge
        if show_live_badge is None:
            show_live_badge = self.show_badge
        
        if show_live_badge and self.badge:
            self.badge.show("on", f"Starting BC training for {self.config.epochs} epochs...")
            time.sleep(0.5)  # Brief pause to show the message
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                
                # Early stopping
                if early_stopping > 0:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping:
                            if self.config.verbose:
                                logger.info(f"Early stopping at epoch {epoch+1}")
                            break
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config.learning_rate
            
            # Record metrics
            metrics = BCMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                time_elapsed=time.time() - epoch_start
            )
            self.history.append(metrics)
            
            # Update badge with progress
            if show_live_badge and self.badge:
                self.badge.update_progress(epoch + 1, self.config.epochs, train_loss)
            
            # Logging
            if self.config.verbose and (epoch + 1) % self.config.log_interval == 0:
                # Clear badge line before logging
                if show_live_badge and self.badge:
                    print()  # New line after badge
                
                msg = f"Epoch {epoch+1}/{self.config.epochs}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                msg += f", lr={current_lr:.6f}, time={metrics.time_elapsed:.2f}s"
                logger.info(msg)
        
        total_time = time.time() - start_time
        
        # Update badge to complete
        if show_live_badge and self.badge:
            print()  # New line after progress
            self.badge.show("complete", f"Training completed in {total_time:.2f} seconds")
            time.sleep(1)  # Show completion message
            if self.badge.auto_hide:
                self.badge.hide()
        
        if self.config.verbose:
            logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        if self.config.verbose:
            logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.config.verbose:
            logger.info(f"Checkpoint loaded from {filepath}")


def create_linear_toy_dataset(n_samples: int = 1000, 
                             input_dim: int = 10, 
                             output_dim: int = 3,
                             noise_std: float = 0.01,
                             seed: Optional[int] = None) -> Tuple[BCDataset, BCDataset]:
    """
    Create a linear toy dataset for testing BC.
    
    The true relationship is: y = Wx + b + noise
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Dimension of observations
        output_dim: Dimension of actions
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        Training and validation datasets
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate true linear mapping
    W = np.random.randn(output_dim, input_dim) * 0.5
    b = np.random.randn(output_dim) * 0.1
    
    # Generate observations
    X = np.random.randn(n_samples, input_dim)
    
    # Generate actions (expert demonstrations)
    Y = X @ W.T + b + np.random.randn(n_samples, output_dim) * noise_std
    
    # Split into train and validation
    split = int(0.8 * n_samples)
    
    train_dataset = BCDataset(X[:split], Y[:split], normalize=True)
    val_dataset = BCDataset(X[split:], Y[split:], normalize=True)
    
    return train_dataset, val_dataset


# Example usage
if __name__ == "__main__":
    # Quick smoke test
    print("Running BC smoke test on linear toy dataset...")
    print("=" * 60)
    
    # Configuration
    config = BCConfig(
        input_dim=10,
        output_dim=3,
        hidden_dims=[32, 32],
        batch_size=64,
        learning_rate=1e-2,
        epochs=20,
        log_interval=5,
        seed=42
    )
    
    # Create toy dataset
    train_data, val_data = create_linear_toy_dataset(
        n_samples=1000,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        seed=42
    )
    
    # Create and train policy
    policy = BCPolicy(config)
    trainer = BCTrainer(policy, config)
    
    start = time.time()
    history = trainer.train(train_data, val_data)
    elapsed = time.time() - start
    
    # Check results
    initial_loss = history[0].train_loss
    final_loss = history[-1].train_loss
    
    print(f"\nResults:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Improvement: {(1 - final_loss/initial_loss)*100:.1f}%")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Success: {'✓' if final_loss < initial_loss * 0.1 and elapsed < 2 else '✗'}")