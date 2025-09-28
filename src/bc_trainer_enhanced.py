"""
Enhanced Behavior Cloning Trainer with Real-time Loss Visualization

Features:
- Real-time loss curve with epoch numbers
- Strict 15-second training time cap
- Adaptive training for time efficiency
- Live matplotlib visualization
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import threading
import warnings
from pathlib import Path
from colorama import init, Fore, Style
import json
import sys

# Initialize colorama
init(autoreset=True)

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set matplotlib backend for better performance
plt.switch_backend('TkAgg')

@dataclass
class BCConfig:
    """Configuration for BC training with time constraints"""
    # Time constraints
    max_time_seconds: float = 15.0  # Hard cap at 15 seconds
    
    # Training parameters
    learning_rate: float = 3e-3  # Higher LR for faster convergence
    batch_size: int = 128  # Larger batch for efficiency
    max_epochs: int = 200  # Allow more epochs (time will cap it)
    
    # Early stopping
    early_stop_patience: int = 8
    early_stop_threshold: float = 1e-4
    validation_split: float = 0.2
    
    # Model architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    dropout: float = 0.1
    
    # Optimization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    lr_scheduler: bool = True
    lr_decay_factor: float = 0.95
    
    # Display
    plot_update_interval: float = 0.2  # Update plot every 200ms
    verbose_interval: int = 1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FastBCDataset(Dataset):
    """Optimized dataset for fast BC training"""
    
    def __init__(self, trajectories):
        self.states = []
        self.actions = []
        
        for traj in trajectories:
            if hasattr(traj, 'states'):
                self.states.extend(traj.states)
                self.actions.extend(traj.actions)
        
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        
        # Pre-compute on GPU if available
        if torch.cuda.is_available():
            self.states = self.states.cuda()
            self.actions = self.actions.cuda()
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BCNetwork(nn.Module):
    """Efficient BC network"""
    
    def __init__(self, state_dim: int, action_dim: int, config: BCConfig):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        
        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
                nn.Dropout(config.dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for faster convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class LiveLossPlotter:
    """Real-time loss curve visualization with epoch numbers"""
    
    def __init__(self, max_time: float = 15.0):
        self.max_time = max_time
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.times = []
        self.start_time = None
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(14, 6), facecolor='#f0f0f0')
        self.fig.suptitle('🤖 Behavior Cloning Training Monitor', 
                          fontsize=16, fontweight='bold', y=0.98)
        
        # Create grid for subplots
        gs = GridSpec(2, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main loss curve (takes 2 columns)
        self.ax_loss = self.fig.add_subplot(gs[:, :2])
        self._setup_loss_plot()
        
        # Time progress
        self.ax_time = self.fig.add_subplot(gs[0, 2])
        self._setup_time_plot()
        
        # Statistics
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self._setup_stats_plot()
        
        # Initialize plot elements
        self.train_line, = self.ax_loss.plot([], [], 'b-', linewidth=2.5, 
                                             label='Train Loss', marker='o', markersize=4)
        self.val_line, = self.ax_loss.plot([], [], 'r--', linewidth=2.5, 
                                           label='Val Loss', marker='s', markersize=4)
        
        # Time progress bar
        self.time_bar = self.ax_time.barh(0, 0, height=0.6, color='#4CAF50')
        self.time_text = self.ax_time.text(self.max_time/2, 0, '', 
                                           ha='center', va='center', fontsize=11, 
                                           fontweight='bold', color='white')
        
        # Stats text elements
        self.stats_texts = []
        stats_labels = ['Current Epoch:', 'Train Loss:', 'Val Loss:', 
                       'Best Val Loss:', 'Time Elapsed:', 'Time Remaining:']
        for i, label in enumerate(stats_labels):
            y_pos = 0.85 - i * 0.15
            self.ax_stats.text(0.05, y_pos, label, fontsize=10, fontweight='bold')
            text = self.ax_stats.text(0.55, y_pos, '--', fontsize=10)
            self.stats_texts.append(text)
        
        plt.tight_layout()
        
        # Threading setup
        self.lock = threading.Lock()
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Show plot
        plt.ion()
        plt.show(block=False)
    
    def _setup_loss_plot(self):
        """Setup loss curve plot"""
        self.ax_loss.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        self.ax_loss.set_ylabel('Loss', fontsize=12, fontweight='bold')
        self.ax_loss.set_title('📈 Training & Validation Loss', fontsize=12, fontweight='bold')
        self.ax_loss.grid(True, alpha=0.3, linestyle='--')
        self.ax_loss.set_facecolor('#ffffff')
        self.ax_loss.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    def _setup_time_plot(self):
        """Setup time progress plot"""
        self.ax_time.set_xlim(0, self.max_time)
        self.ax_time.set_ylim(-0.5, 0.5)
        self.ax_time.set_title('⏱️ Time Progress', fontsize=12, fontweight='bold')
        self.ax_time.set_xlabel('Seconds', fontsize=10)
        self.ax_time.set_yticks([])
        self.ax_time.axvline(x=self.max_time, color='red', linestyle='--', 
                            linewidth=2, alpha=0.7)
        self.ax_time.text(self.max_time, -0.35, f'{self.max_time}s cap', 
                         ha='right', fontsize=9, color='red')
        self.ax_time.set_facecolor('#ffffff')
    
    def _setup_stats_plot(self):
        """Setup statistics display"""
        self.ax_stats.set_title('📊 Training Statistics', fontsize=12, fontweight='bold')
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor('#ffffff')
    
    def start(self):
        """Start the plotter"""
        self.start_time = time.time()
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop(self):
        """Stop the plotter"""
        self.stop_event.set()
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def add_data(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """Add new training data point"""
        with self.lock:
            self.epochs.append(epoch)
            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)
            self.times.append(time.time() - self.start_time)
    
    def _update_loop(self):
        """Background update loop"""
        while not self.stop_event.is_set():
            try:
                self._update_plot()
                time.sleep(0.1)  # 100ms update rate
            except Exception as e:
                print(f"Plot update error: {e}")
                break
    
    def _update_plot(self):
        """Update all plot elements"""
        with self.lock:
            if not self.epochs:
                return
            
            # Update loss curves
            self.train_line.set_data(self.epochs, self.train_losses)
            if self.val_losses:
                val_epochs = self.epochs[:len(self.val_losses)]
                self.val_line.set_data(val_epochs, self.val_losses)
            
            # Auto-scale loss plot
            self.ax_loss.set_xlim(0, max(self.epochs[-1] + 1, 10))
            all_losses = self.train_losses + (self.val_losses if self.val_losses else [])
            if all_losses:
                loss_margin = 0.1 * (max(all_losses) - min(all_losses))
                self.ax_loss.set_ylim(max(0, min(all_losses) - loss_margin),
                                     max(all_losses) + loss_margin)
            
            # Set epoch ticks (show all if ≤15, otherwise every 5)
            if len(self.epochs) <= 15:
                self.ax_loss.set_xticks(self.epochs)
            else:
                step = 5
                ticks = [e for e in self.epochs if e % step == 0]
                if self.epochs[-1] not in ticks:
                    ticks.append(self.epochs[-1])
                self.ax_loss.set_xticks(ticks)
            
            # Update time progress
            elapsed = time.time() - self.start_time if self.start_time else 0
            remaining = max(0, self.max_time - elapsed)
            
            # Update time bar color
            if elapsed < self.max_time * 0.5:
                color = '#4CAF50'  # Green
            elif elapsed < self.max_time * 0.8:
                color = '#FFC107'  # Yellow
            else:
                color = '#F44336'  # Red
            
            # Update time bar
            self.time_bar.remove()
            self.time_bar = self.ax_time.barh(0, min(elapsed, self.max_time), 
                                              height=0.6, color=color, alpha=0.8)
            self.time_text.set_text(f'{elapsed:.1f}s')
            self.time_text.set_position((min(elapsed/2, self.max_time/2), 0))
            
            # Update statistics
            stats_values = [
                f"{self.epochs[-1]}",
                f"{self.train_losses[-1]:.4f}",
                f"{self.val_losses[-1]:.4f}" if self.val_losses else "N/A",
                f"{min(self.val_losses):.4f}" if self.val_losses else "N/A",
                f"{elapsed:.1f}s",
                f"{remaining:.1f}s"
            ]
            
            for text, value in zip(self.stats_texts, stats_values):
                text.set_text(value)
                # Color code based on value
                if 'N/A' not in value:
                    if 's' in value:  # Time values
                        if remaining < 3:
                            text.set_color('red')
                        else:
                            text.set_color('green')
                    else:
                        text.set_color('black')
        
        # Refresh display
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except:
            pass


class EnhancedBCTrainer:
    """BC Trainer with real-time visualization and 15s time cap"""
    
    def __init__(self, config: BCConfig = None):
        self.config = config or BCConfig()
        self.device = torch.device(self.config.device)
        self.model = None
        self.training_complete = False
    
    def train(self, trajectories, show_plot: bool = True) -> Dict[str, Any]:
        """
        Train BC model with strict 15-second cap and live visualization
        
        Args:
            trajectories: Expert trajectories
            show_plot: Show real-time loss curve
            
        Returns:
            Training results
        """
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 Behavior Cloning Training (≤15s cap){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        start_time = time.time()
        
        # Prepare data
        dataset = FastBCDataset(trajectories)
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Use larger batch size for efficiency
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                 shuffle=True, pin_memory=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size * 2,
                               shuffle=False, pin_memory=True, num_workers=0)
        
        # Initialize model
        state_dim = dataset.states.shape[1]
        action_dim = dataset.actions.shape[1]
        self.model = BCNetwork(state_dim, action_dim, self.config).to(self.device)
        
        # Optimizer with adaptive learning rate
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=self.config.learning_rate,
                               weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler for faster convergence
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                                                     gamma=self.config.lr_decay_factor)
        
        criterion = nn.MSELoss()
        
        # Initialize plotter
        plotter = None
        if show_plot:
            plotter = LiveLossPlotter(self.config.max_time_seconds)
            plotter.start()
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train': [], 'val': [], 'epochs': []}
        
        print(f"📊 Training on {len(train_dataset)} samples")
        print(f"🎯 Validation on {len(val_dataset)} samples")
        print(f"⏰ Time cap: {self.config.max_time_seconds}s\n")
        
        epoch = 0
        while epoch < self.config.max_epochs:
            epoch_start = time.time()
            elapsed = epoch_start - start_time
            
            # Check time cap
            if elapsed >= self.config.max_time_seconds:
                print(f"\n{Fore.YELLOW}⏰ Time cap reached! ({elapsed:.1f}s){Style.RESET_ALL}")
                break
            
            # Adaptive training based on remaining time
            time_remaining = self.config.max_time_seconds - elapsed
            if time_remaining < 3 and epoch > 5:
                # Speed up: skip some validation
                validate_this_epoch = epoch % 3 == 0
            else:
                validate_this_epoch = True
            
            # Training
            self.model.train()
            train_losses = []
            
            for batch_idx, (states, actions) in enumerate(train_loader):
                # Check time during training
                if time.time() - start_time >= self.config.max_time_seconds:
                    break
                
                optimizer.zero_grad()
                predictions = self.model(states)
                loss = criterion(predictions, actions)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.gradient_clip)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses) if train_losses else 0
            
            # Validation
            avg_val_loss = None
            if validate_this_epoch and val_loader:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for states, actions in val_loader:
                        predictions = self.model(states)
                        loss = criterion(predictions, actions)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses) if val_losses else None
                
                # Early stopping
                if avg_val_loss and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stop_patience:
                    print(f"\n{Fore.YELLOW}Early stopping at epoch {epoch}{Style.RESET_ALL}")
                    break
            
            # Update scheduler
            if self.config.lr_scheduler and epoch > 0 and epoch % 5 == 0:
                scheduler.step()
            
            # Record history
            history['epochs'].append(epoch)
            history['train'].append(avg_train_loss)
            if avg_val_loss is not None:
                history['val'].append(avg_val_loss)
            
            # Update plot
            if plotter:
                plotter.add_data(epoch, avg_train_loss, avg_val_loss)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            if epoch % self.config.verbose_interval == 0:
                msg = f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f}"
                if avg_val_loss:
                    msg += f" | Val: {avg_val_loss:.4f}"
                msg += f" | Time: {elapsed:.1f}s ({epoch_time:.2f}s/epoch)"
                
                # Color based on time
                if elapsed < self.config.max_time_seconds * 0.6:
                    print(f"{Fore.GREEN}{msg}{Style.RESET_ALL}")
                elif elapsed < self.config.max_time_seconds * 0.9:
                    print(f"{Fore.YELLOW}{msg}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{msg}{Style.RESET_ALL}")
            
            epoch += 1
        
        # Training complete
        total_time = time.time() - start_time
        self.training_complete = True
        
        if plotter:
            plotter.stop()
        
        # Print summary
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Training Complete!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        print(f"✅ Epochs completed: {epoch}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        if total_time > self.config.max_time_seconds:
            print(f"⚠️  {Fore.YELLOW}Exceeded time cap by {total_time - self.config.max_time_seconds:.2f}s{Style.RESET_ALL}")
        else:
            print(f"✨ {Fore.GREEN}Completed within time cap!{Style.RESET_ALL}")
        
        if history['val']:
            print(f"📉 Best validation loss: {min(history['val']):.4f}")
        print(f"📈 Final training loss: {history['train'][-1]:.4f}")
        
        return {
            'history': history,
            'total_time': total_time,
            'epochs': epoch,
            'within_time_cap': total_time <= self.config.max_time_seconds
        }


def demo():
    """Demo BC training with visualization"""
    
    # Generate dummy trajectories
    print(f"{Fore.YELLOW}Generating demo trajectories...{Style.RESET_ALL}")
    
    class DemoTrajectory:
        def __init__(self):
            self.states = [np.random.randn(10) for _ in range(50)]
            self.actions = [np.tanh(s[:4]) * 0.5 for s in self.states]
    
    trajectories = [DemoTrajectory() for _ in range(20)]
    
    # Configure and train
    config = BCConfig(
        max_time_seconds=15.0,
        max_epochs=100,
        batch_size=64,
        learning_rate=3e-3
    )
    
    trainer = EnhancedBCTrainer(config)
    results = trainer.train(trajectories, show_plot=True)
    
    print(f"\n{Fore.GREEN}✅ Demo complete!{Style.RESET_ALL}")
    print("\nPress Enter to close plot and exit...")
    input()
    plt.close('all')


if __name__ == "__main__":
    demo()