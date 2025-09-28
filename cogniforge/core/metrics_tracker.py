"""
Metrics tracking module for CogniForge pipeline.
Tracks BC loss, CMA-ES cost, PPO rewards, and vision offsets.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import deque

from .logging_utils import log_event, EventPhase, LogLevel


@dataclass
class BCMetrics:
    """Behavior Cloning training metrics."""
    epoch: int
    loss: float
    learning_rate: float
    batch_size: int = 32
    accuracy: Optional[float] = None
    validation_loss: Optional[float] = None
    gradient_norm: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CMAESMetrics:
    """CMA-ES optimization metrics."""
    iteration: int
    best_cost: float
    mean_cost: float
    std_cost: float
    population_size: int
    sigma: float  # Step size
    converged: bool = False
    improvement: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PPOMetrics:
    """PPO training metrics."""
    episode: int
    average_reward: float
    min_reward: float
    max_reward: float
    value_loss: float
    policy_loss: float
    entropy: float
    explained_variance: Optional[float] = None
    kl_divergence: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class VisionOffsets:
    """Vision detection offsets."""
    # Pixel space offsets
    dx_pixel: int
    dy_pixel: int
    
    # World space offsets (in mm)
    dx_world: float
    dy_world: float
    
    # Additional metadata
    confidence: float
    detection_method: str = "template_matching"
    depth_mm: Optional[float] = None
    rotation_deg: Optional[float] = None
    scale_factor: Optional[float] = None
    processing_time_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def pixel_magnitude(self) -> float:
        """Calculate pixel offset magnitude."""
        return np.sqrt(self.dx_pixel**2 + self.dy_pixel**2)
    
    @property
    def world_magnitude(self) -> float:
        """Calculate world offset magnitude in mm."""
        return np.sqrt(self.dx_world**2 + self.dy_world**2)
    
    def is_aligned(self, tolerance_mm: float = 5.0) -> bool:
        """Check if object is aligned within tolerance."""
        return self.world_magnitude <= tolerance_mm


class MetricsTracker:
    """
    Centralized metrics tracking for the CogniForge pipeline.
    Tracks BC, CMA-ES, PPO, and vision metrics with history.
    """
    
    def __init__(self, request_id: Optional[str] = None,
                 save_dir: Optional[Path] = None,
                 event_logger: Optional[Callable[..., Any]] = None):
        """
        Initialize metrics tracker.
        
        Args:
            request_id: Unique request identifier
            save_dir: Directory to save metrics history
            event_logger: Optional logging callback
        """
        self.request_id = request_id or f"metrics_{int(time.time())}"
        self.save_dir = Path(save_dir) if save_dir else Path("metrics")
        self.save_dir.mkdir(exist_ok=True)

        if event_logger is None:
            def _default_logger(phase, message, **payload):
                level_value = payload.pop("level", LogLevel.INFO)
                if isinstance(level_value, LogLevel):
                    level_obj = level_value
                else:
                    try:
                        level_obj = LogLevel(level_value)
                    except Exception:
                        level_obj = LogLevel.INFO
                log_event(phase, message, level=level_obj, **payload)
            self.event_logger = _default_logger
            self._uses_default_logger = True
        else:
            self.event_logger = event_logger
            self._uses_default_logger = False

        # Metrics storage
        self.bc_history: List[BCMetrics] = []
        self.cmaes_history: List[CMAESMetrics] = []
        self.ppo_history: List[PPOMetrics] = []
        self.vision_history: List[VisionOffsets] = []
        
        # Current best values
        self.best_bc_loss = float('inf')
        self.best_cmaes_cost = float('inf')
        self.best_ppo_reward = float('-inf')
        self.smallest_vision_offset = float('inf')
        
        # Moving averages for smoothing
        self.bc_loss_ma = deque(maxlen=10)
        self.ppo_reward_ma = deque(maxlen=20)
        self.vision_offset_ma = deque(maxlen=5)
        
        # Start time for duration tracking
        self.start_time = time.time()

    def _emit_metric_event(self, phase, message, level=LogLevel.METRIC, **metrics):
        """Emit metric event through configured logger."""
        payload = dict(metrics)
        if self.request_id and "request_id" not in payload:
            payload["request_id"] = self.request_id

        if getattr(self, '_uses_default_logger', False):
            self.event_logger(phase, message, level=level, **payload)
        else:
            level_value = level.value if isinstance(level, LogLevel) else level
            payload.setdefault("level", level_value)
            self.event_logger(phase, message, **payload)

    
    def track_bc_epoch(self, epoch: int, loss: float, 
                       learning_rate: float = 0.001,
                       batch_size: int = 32,
                       accuracy: Optional[float] = None,
                       validation_loss: Optional[float] = None,
                       gradient_norm: Optional[float] = None) -> BCMetrics:
        """
        Track behavior cloning training epoch.
        
        Args:
            epoch: Epoch number
            loss: Training loss
            learning_rate: Learning rate
            batch_size: Batch size
            accuracy: Optional accuracy metric
            validation_loss: Optional validation loss
            gradient_norm: Optional gradient norm
        
        Returns:
            BCMetrics object
        """
        metrics = BCMetrics(
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            accuracy=accuracy,
            validation_loss=validation_loss,
            gradient_norm=gradient_norm
        )
        
        self.bc_history.append(metrics)
        self.bc_loss_ma.append(loss)
        
        # Update best
        if loss < self.best_bc_loss:
            self.best_bc_loss = loss
            improvement = True
        else:
            improvement = False
        
        # Log to unified system
        self._emit_metric_event(
            EventPhase.BEHAVIOR_CLONING,
            f"BC Epoch {epoch} completed",
            level=LogLevel.METRIC,
            epoch=epoch,
            loss=round(loss, 6),
            learning_rate=learning_rate,
            best_loss=round(self.best_bc_loss, 6),
            moving_avg_loss=round(np.mean(self.bc_loss_ma), 6),
            improved=improvement,
            accuracy=accuracy,
        )
        
        # Save if improved
        if improvement:
            self._save_checkpoint('bc', metrics)
        
        return metrics
    
    def track_cmaes_iteration(self, iteration: int, best_cost: float,
                              mean_cost: float, std_cost: float,
                              population_size: int, sigma: float,
                              converged: bool = False) -> CMAESMetrics:
        """
        Track CMA-ES optimization iteration.
        
        Args:
            iteration: Iteration number
            best_cost: Best cost in current population
            mean_cost: Mean cost of population
            std_cost: Standard deviation of costs
            population_size: Size of population
            sigma: Current step size
            converged: Whether optimization has converged
        
        Returns:
            CMAESMetrics object
        """
        # Calculate improvement
        improvement = None
        if self.cmaes_history:
            improvement = self.cmaes_history[-1].best_cost - best_cost
        
        metrics = CMAESMetrics(
            iteration=iteration,
            best_cost=best_cost,
            mean_cost=mean_cost,
            std_cost=std_cost,
            population_size=population_size,
            sigma=sigma,
            converged=converged,
            improvement=improvement
        )
        
        self.cmaes_history.append(metrics)
        
        # Update best
        if best_cost < self.best_cmaes_cost:
            self.best_cmaes_cost = best_cost
            improved = True
        else:
            improved = False
        
        # Log to unified system
        self._emit_metric_event(
            EventPhase.OPTIMIZATION,
            f"CMA-ES iteration {iteration}",
            level=LogLevel.METRIC,
            iteration=iteration,
            best_cost=round(best_cost, 6),
            mean_cost=round(mean_cost, 6),
            std_cost=round(std_cost, 6),
            sigma=round(sigma, 6),
            population_size=population_size,
            global_best=round(self.best_cmaes_cost, 6),
            improved=improved,
            converged=converged,
        )
        
        # Save if improved or converged
        if improved or converged:
            self._save_checkpoint('cmaes', metrics)
        
        return metrics
    
    def track_ppo_episode(self, episode: int, average_reward: float,
                         min_reward: float, max_reward: float,
                         value_loss: float, policy_loss: float,
                         entropy: float,
                         explained_variance: Optional[float] = None,
                         kl_divergence: Optional[float] = None) -> PPOMetrics:
        """
        Track PPO training episode.
        
        Args:
            episode: Episode number
            average_reward: Average reward
            min_reward: Minimum reward
            max_reward: Maximum reward
            value_loss: Value function loss
            policy_loss: Policy loss
            entropy: Policy entropy
            explained_variance: Optional explained variance
            kl_divergence: Optional KL divergence
        
        Returns:
            PPOMetrics object
        """
        metrics = PPOMetrics(
            episode=episode,
            average_reward=average_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            value_loss=value_loss,
            policy_loss=policy_loss,
            entropy=entropy,
            explained_variance=explained_variance,
            kl_divergence=kl_divergence
        )
        
        self.ppo_history.append(metrics)
        self.ppo_reward_ma.append(average_reward)
        
        # Update best
        if average_reward > self.best_ppo_reward:
            self.best_ppo_reward = average_reward
            improved = True
        else:
            improved = False
        
        # Log to unified system
        self._emit_metric_event(
            EventPhase.OPTIMIZATION,
            f"PPO episode {episode}",
            level=LogLevel.METRIC,
            episode=episode,
            avg_reward=round(average_reward, 4),
            min_reward=round(min_reward, 4),
            max_reward=round(max_reward, 4),
            value_loss=round(value_loss, 6),
            policy_loss=round(policy_loss, 6),
            entropy=round(entropy, 6),
            best_reward=round(self.best_ppo_reward, 4),
            moving_avg_reward=round(np.mean(self.ppo_reward_ma), 4),
            improved=improved,
        )
        
        # Save if improved
        if improved:
            self._save_checkpoint('ppo', metrics)
        
        return metrics
    
    def track_vision_offset(self, dx_pixel: int, dy_pixel: int,
                           dx_world: float, dy_world: float,
                           confidence: float,
                           detection_method: str = "template_matching",
                           depth_mm: Optional[float] = None,
                           rotation_deg: Optional[float] = None,
                           scale_factor: Optional[float] = None,
                           processing_time_ms: Optional[float] = None) -> VisionOffsets:
        """
        Track vision detection offsets.
        
        Args:
            dx_pixel: Horizontal offset in pixels
            dy_pixel: Vertical offset in pixels
            dx_world: Horizontal offset in mm
            dy_world: Vertical offset in mm
            confidence: Detection confidence (0-1)
            detection_method: Method used for detection
            depth_mm: Optional depth measurement
            rotation_deg: Optional rotation offset
            scale_factor: Optional scale difference
            processing_time_ms: Processing time in milliseconds
        
        Returns:
            VisionOffsets object
        """
        offsets = VisionOffsets(
            dx_pixel=dx_pixel,
            dy_pixel=dy_pixel,
            dx_world=dx_world,
            dy_world=dy_world,
            confidence=confidence,
            detection_method=detection_method,
            depth_mm=depth_mm,
            rotation_deg=rotation_deg,
            scale_factor=scale_factor,
            processing_time_ms=processing_time_ms
        )
        
        self.vision_history.append(offsets)
        self.vision_offset_ma.append(offsets.world_magnitude)
        
        # Update smallest offset
        if offsets.world_magnitude < self.smallest_vision_offset:
            self.smallest_vision_offset = offsets.world_magnitude
            improved = True
        else:
            improved = False
        
        # Log to unified system
        self._emit_metric_event(
            EventPhase.VISION_REFINEMENT,
            f"Vision offset detected",
            level=LogLevel.METRIC,
            dx_pixel=dx_pixel,
            dy_pixel=dy_pixel,
            dx_world=round(dx_world, 2),
            dy_world=round(dy_world, 2),
            pixel_magnitude=round(offsets.pixel_magnitude, 2),
            world_magnitude=round(offsets.world_magnitude, 2),
            confidence=round(confidence, 3),
            method=detection_method,
            aligned=offsets.is_aligned(),
            best_offset=round(self.smallest_vision_offset, 2),
            moving_avg_offset=round(np.mean(self.vision_offset_ma), 2),
            improved=improved,
        )
        
        return offsets
    
    def get_bc_summary(self) -> Dict[str, Any]:
        """Get behavior cloning metrics summary."""
        if not self.bc_history:
            return {}
        
        losses = [m.loss for m in self.bc_history]
        accuracies = [m.accuracy for m in self.bc_history if m.accuracy]
        
        return {
            'total_epochs': len(self.bc_history),
            'best_loss': self.best_bc_loss,
            'final_loss': losses[-1],
            'average_loss': np.mean(losses),
            'loss_reduction': losses[0] - losses[-1] if len(losses) > 1 else 0,
            'final_accuracy': accuracies[-1] if accuracies else None,
            'learning_rate': self.bc_history[-1].learning_rate,
            'convergence_rate': self._calculate_convergence_rate(losses)
        }
    
    def get_cmaes_summary(self) -> Dict[str, Any]:
        """Get CMA-ES metrics summary."""
        if not self.cmaes_history:
            return {}
        
        costs = [m.best_cost for m in self.cmaes_history]
        
        return {
            'total_iterations': len(self.cmaes_history),
            'best_cost': self.best_cmaes_cost,
            'final_cost': costs[-1],
            'average_cost': np.mean(costs),
            'cost_reduction': costs[0] - costs[-1] if len(costs) > 1 else 0,
            'final_sigma': self.cmaes_history[-1].sigma,
            'converged': self.cmaes_history[-1].converged,
            'convergence_rate': self._calculate_convergence_rate(costs)
        }
    
    def get_ppo_summary(self) -> Dict[str, Any]:
        """Get PPO metrics summary."""
        if not self.ppo_history:
            return {}
        
        rewards = [m.average_reward for m in self.ppo_history]
        
        return {
            'total_episodes': len(self.ppo_history),
            'best_reward': self.best_ppo_reward,
            'final_reward': rewards[-1],
            'average_reward': np.mean(rewards),
            'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
            'final_entropy': self.ppo_history[-1].entropy,
            'convergence_rate': self._calculate_convergence_rate(rewards, minimize=False)
        }
    
    def get_vision_summary(self) -> Dict[str, Any]:
        """Get vision offsets summary."""
        if not self.vision_history:
            return {}
        
        pixel_offsets = [(v.dx_pixel, v.dy_pixel) for v in self.vision_history]
        world_offsets = [(v.dx_world, v.dy_world) for v in self.vision_history]
        magnitudes = [v.world_magnitude for v in self.vision_history]
        confidences = [v.confidence for v in self.vision_history]
        
        return {
            'total_detections': len(self.vision_history),
            'smallest_offset_mm': self.smallest_vision_offset,
            'average_offset_mm': np.mean(magnitudes),
            'std_offset_mm': np.std(magnitudes),
            'average_confidence': np.mean(confidences),
            'aligned_count': sum(1 for v in self.vision_history if v.is_aligned()),
            'alignment_rate': sum(1 for v in self.vision_history if v.is_aligned()) / len(self.vision_history),
            'average_dx_pixel': np.mean([p[0] for p in pixel_offsets]),
            'average_dy_pixel': np.mean([p[1] for p in pixel_offsets]),
            'average_dx_world': np.mean([w[0] for w in world_offsets]),
            'average_dy_world': np.mean([w[1] for w in world_offsets])
        }
    
    def get_full_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        duration = time.time() - self.start_time
        
        return {
            'request_id': self.request_id,
            'duration_seconds': duration,
            'bc_metrics': self.get_bc_summary(),
            'cmaes_metrics': self.get_cmaes_summary(),
            'ppo_metrics': self.get_ppo_summary(),
            'vision_metrics': self.get_vision_summary(),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_metrics(self, filename: Optional[str] = None):
        """
        Save all metrics to JSON file.
        
        Args:
            filename: Optional filename, defaults to request_id.json
        """
        if filename is None:
            filename = f"{self.request_id}_metrics.json"
        
        filepath = self.save_dir / filename
        
        data = {
            'summary': self.get_full_summary(),
            'bc_history': [asdict(m) for m in self.bc_history],
            'cmaes_history': [asdict(m) for m in self.cmaes_history],
            'ppo_history': [asdict(m) for m in self.ppo_history],
            'vision_history': [asdict(m) for m in self.vision_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self._emit_metric_event(
            EventPhase.COMPLETED,
            f"Metrics saved to {filepath}",
            level=LogLevel.INFO,
            total_bc_epochs=len(self.bc_history),
            total_cmaes_iterations=len(self.cmaes_history),
            total_ppo_episodes=len(self.ppo_history),
            total_vision_detections=len(self.vision_history),
        )
        
        return filepath
    
    def _calculate_convergence_rate(self, values: List[float], 
                                   minimize: bool = True) -> float:
        """
        Calculate convergence rate.
        
        Args:
            values: List of metric values
            minimize: Whether lower is better
        
        Returns:
            Convergence rate (0-1, higher is better)
        """
        if len(values) < 2:
            return 0.0
        
        # Calculate improvement
        if minimize:
            improvement = values[0] - values[-1]
            max_possible = values[0]
        else:
            improvement = values[-1] - values[0]
            max_possible = abs(values[0])
        
        if max_possible == 0:
            return 0.0
        
        rate = improvement / max_possible
        return max(0.0, min(1.0, rate))
    
    def _save_checkpoint(self, metric_type: str, metrics: Any):
        """Save checkpoint for improved metrics."""
        checkpoint_dir = self.save_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        filepath = checkpoint_dir / f"{self.request_id}_{metric_type}_best.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def plot_metrics(self, save_plots: bool = True) -> Optional[Path]:
        """
        Generate plots for metrics (requires matplotlib).
        
        Args:
            save_plots: Whether to save plots to file
        
        Returns:
            Path to saved plots directory if save_plots is True
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            self._emit_metric_event(
                EventPhase.COMPLETED,
                "Matplotlib not installed, skipping plots",
                level=LogLevel.WARNING,
            )
            return None
        
        if not any([self.bc_history, self.cmaes_history, 
                   self.ppo_history, self.vision_history]):
            return None
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # BC Loss plot
        if self.bc_history:
            ax1 = fig.add_subplot(gs[0, 0])
            epochs = [m.epoch for m in self.bc_history]
            losses = [m.loss for m in self.bc_history]
            ax1.plot(epochs, losses, 'b-', label='BC Loss')
            ax1.axhline(y=self.best_bc_loss, color='r', linestyle='--', 
                       label=f'Best: {self.best_bc_loss:.6f}')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Behavior Cloning Loss per Epoch')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # CMA-ES Cost plot
        if self.cmaes_history:
            ax2 = fig.add_subplot(gs[0, 1])
            iterations = [m.iteration for m in self.cmaes_history]
            costs = [m.best_cost for m in self.cmaes_history]
            mean_costs = [m.mean_cost for m in self.cmaes_history]
            ax2.plot(iterations, costs, 'g-', label='Best Cost')
            ax2.plot(iterations, mean_costs, 'g--', alpha=0.5, label='Mean Cost')
            ax2.axhline(y=self.best_cmaes_cost, color='r', linestyle='--',
                       label=f'Global Best: {self.best_cmaes_cost:.6f}')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Cost')
            ax2.set_title('CMA-ES Best Cost per Iteration')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # PPO Reward plot
        if self.ppo_history:
            ax3 = fig.add_subplot(gs[1, 0])
            episodes = [m.episode for m in self.ppo_history]
            avg_rewards = [m.average_reward for m in self.ppo_history]
            min_rewards = [m.min_reward for m in self.ppo_history]
            max_rewards = [m.max_reward for m in self.ppo_history]
            
            ax3.plot(episodes, avg_rewards, 'b-', label='Average Reward')
            ax3.fill_between(episodes, min_rewards, max_rewards, alpha=0.3)
            ax3.axhline(y=self.best_ppo_reward, color='r', linestyle='--',
                       label=f'Best: {self.best_ppo_reward:.4f}')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')
            ax3.set_title('PPO Average Reward per Episode')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Vision Offsets plot
        if self.vision_history:
            ax4 = fig.add_subplot(gs[1, 1])
            indices = range(len(self.vision_history))
            pixel_mags = [v.pixel_magnitude for v in self.vision_history]
            world_mags = [v.world_magnitude for v in self.vision_history]
            
            ax4.plot(indices, pixel_mags, 'c-', label='Pixel Magnitude', alpha=0.7)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(indices, world_mags, 'orange', label='World Magnitude (mm)')
            
            ax4.set_xlabel('Detection')
            ax4.set_ylabel('Pixel Offset', color='c')
            ax4_twin.set_ylabel('World Offset (mm)', color='orange')
            ax4.set_title('Vision Detection Offsets')
            ax4.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Vision scatter plot
        if self.vision_history:
            ax5 = fig.add_subplot(gs[2, :])
            dx_pixels = [v.dx_pixel for v in self.vision_history]
            dy_pixels = [v.dy_pixel for v in self.vision_history]
            confidences = [v.confidence for v in self.vision_history]
            
            scatter = ax5.scatter(dx_pixels, dy_pixels, c=confidences, 
                                 cmap='viridis', s=50, alpha=0.7)
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax5.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax5.set_xlabel('dx (pixels)')
            ax5.set_ylabel('dy (pixels)')
            ax5.set_title('Vision Pixel Offsets Distribution')
            ax5.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label='Confidence')
            
            # Add circle for tolerance
            circle = plt.Circle((0, 0), 10, fill=False, 
                              edgecolor='g', linestyle='--', 
                              label='10px tolerance')
            ax5.add_patch(circle)
            ax5.legend()
            ax5.set_aspect('equal', adjustable='box')
        
        plt.suptitle(f'Metrics for Request: {self.request_id}', fontsize=14)
        plt.tight_layout()
        
        if save_plots:
            plot_dir = self.save_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            plot_path = plot_dir / f"{self.request_id}_metrics.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            self._emit_metric_event(
                EventPhase.COMPLETED,
                f"Plots saved to {plot_path}",
                level=LogLevel.INFO,
                plot_path=str(plot_path),
            )
            
            return plot_path
        else:
            plt.show()
            return None


# Global tracker instance
_global_tracker: Optional[MetricsTracker] = None

def get_global_tracker(request_id: Optional[str] = None) -> MetricsTracker:
    """Get or create global metrics tracker."""
    global _global_tracker
    if _global_tracker is None or (request_id and request_id != _global_tracker.request_id):
        _global_tracker = MetricsTracker(request_id)
    return _global_tracker

# Convenience functions
def track_bc_loss(epoch: int, loss: float, **kwargs) -> BCMetrics:
    """Track BC loss for an epoch."""
    tracker = get_global_tracker()
    return tracker.track_bc_epoch(epoch, loss, **kwargs)

def track_cmaes_cost(iteration: int, best_cost: float, mean_cost: float, 
                     std_cost: float, population_size: int, sigma: float, **kwargs) -> CMAESMetrics:
    """Track CMA-ES cost for an iteration."""
    tracker = get_global_tracker()
    return tracker.track_cmaes_iteration(iteration, best_cost, mean_cost, 
                                         std_cost, population_size, sigma, **kwargs)

def track_ppo_reward(episode: int, average_reward: float, min_reward: float,
                    max_reward: float, value_loss: float, policy_loss: float,
                    entropy: float, **kwargs) -> PPOMetrics:
    """Track PPO reward for an episode."""
    tracker = get_global_tracker()
    return tracker.track_ppo_episode(episode, average_reward, min_reward,
                                     max_reward, value_loss, policy_loss,
                                     entropy, **kwargs)

def track_vision_offset(dx_pixel: int, dy_pixel: int, dx_world: float, 
                       dy_world: float, confidence: float, **kwargs) -> VisionOffsets:
    """Track vision detection offsets."""
    tracker = get_global_tracker()
    return tracker.track_vision_offset(dx_pixel, dy_pixel, dx_world, 
                                       dy_world, confidence, **kwargs)
