"""
CMA-ES Cost Visualization Module

Provides real-time sparkline and progress visualization for CMA-ES optimization
in the UI, showing cost over iterations with various display formats.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time
from dataclasses import dataclass, field, asdict
from collections import deque
import math


@dataclass
class CMAESMetrics:
    """Data class for CMA-ES optimization metrics."""
    iteration: int
    best_cost: float
    mean_cost: float
    std_cost: float
    population_size: int
    sigma: float
    converged: bool = False
    improvement: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SparklineGenerator:
    """Generate ASCII sparkline visualizations for numerical data."""
    
    # Unicode block characters for smooth sparklines
    BLOCKS = " ▁▂▃▄▅▆▇█"
    
    # ASCII alternatives for compatibility
    ASCII_BLOCKS = " .-:=+*#@"
    
    # Braille patterns for ultra-smooth sparklines
    BRAILLE_OFFSET = 0x2800
    BRAILLE_MATRIX = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80]
    ]
    
    @classmethod
    def generate_sparkline(cls, values: List[float], width: int = 50, 
                          style: str = "blocks", show_values: bool = True) -> str:
        """
        Generate a sparkline visualization of the data.
        
        Args:
            values: List of numerical values to visualize
            width: Width of the sparkline in characters
            style: Visualization style ("blocks", "ascii", "braille", "dots")
            show_values: Whether to show min/max values
            
        Returns:
            Formatted sparkline string
        """
        if len(values) == 0:
            return "No data"
        
        # Ensure we have enough values
        if len(values) == 1:
            values = values * 2  # Duplicate single value
        
        # Resample to fit width
        if len(values) > width:
            # Downsample
            indices = np.linspace(0, len(values) - 1, width)
            resampled = [values[int(i)] for i in indices]
        else:
            resampled = values
        
        # Normalize values
        min_val = min(resampled)
        max_val = max(resampled)
        
        if max_val == min_val:
            # All values are the same
            normalized = [0.5] * len(resampled)
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in resampled]
        
        # Generate sparkline based on style
        if style == "blocks":
            sparkline = cls._generate_block_sparkline(normalized)
        elif style == "ascii":
            sparkline = cls._generate_ascii_sparkline(normalized)
        elif style == "braille":
            sparkline = cls._generate_braille_sparkline(normalized)
        elif style == "dots":
            sparkline = cls._generate_dot_sparkline(normalized)
        else:
            sparkline = cls._generate_block_sparkline(normalized)
        
        # Add values if requested
        if show_values:
            min_str = f"{min_val:.2f}"
            max_str = f"{max_val:.2f}"
            current_str = f"{values[-1]:.2f}" if values else "N/A"
            return f"[{min_str}] {sparkline} [{max_str}] Current: {current_str}"
        
        return sparkline
    
    @classmethod
    def _generate_block_sparkline(cls, normalized: List[float]) -> str:
        """Generate sparkline using Unicode block characters."""
        blocks = []
        for val in normalized:
            index = int(val * (len(cls.BLOCKS) - 1))
            blocks.append(cls.BLOCKS[index])
        return "".join(blocks)
    
    @classmethod
    def _generate_ascii_sparkline(cls, normalized: List[float]) -> str:
        """Generate sparkline using ASCII characters."""
        chars = []
        for val in normalized:
            index = int(val * (len(cls.ASCII_BLOCKS) - 1))
            chars.append(cls.ASCII_BLOCKS[index])
        return "".join(chars)
    
    @classmethod
    def _generate_dot_sparkline(cls, normalized: List[float]) -> str:
        """Generate sparkline using dots and lines."""
        chars = []
        levels = "⣀⣤⣶⣿"
        for val in normalized:
            index = int(val * (len(levels) - 1))
            chars.append(levels[index])
        return "".join(chars)
    
    @classmethod
    def _generate_braille_sparkline(cls, normalized: List[float]) -> str:
        """Generate sparkline using Braille characters for smooth curves."""
        # Group values into pairs for Braille encoding
        chars = []
        for i in range(0, len(normalized), 2):
            if i + 1 < len(normalized):
                v1, v2 = normalized[i], normalized[i + 1]
            else:
                v1, v2 = normalized[i], normalized[i]
            
            # Map to Braille pattern
            y1 = int(v1 * 3)
            y2 = int(v2 * 3)
            pattern = cls.BRAILLE_OFFSET
            pattern |= cls.BRAILLE_MATRIX[3 - y1][0]
            pattern |= cls.BRAILLE_MATRIX[3 - y2][1]
            chars.append(chr(pattern))
        
        return "".join(chars)


class CMAESVisualizer:
    """Main visualizer for CMA-ES optimization progress."""
    
    def __init__(self, history_size: int = 100, update_interval: float = 0.1):
        """
        Initialize the CMA-ES visualizer.
        
        Args:
            history_size: Maximum number of iterations to keep in history
            update_interval: Minimum time between UI updates (seconds)
        """
        self.history_size = history_size
        self.update_interval = update_interval
        self.last_update_time = 0
        
        # Cost history tracking
        self.cost_history: deque = deque(maxlen=history_size)
        self.best_cost_history: deque = deque(maxlen=history_size)
        self.mean_cost_history: deque = deque(maxlen=history_size)
        self.std_history: deque = deque(maxlen=history_size)
        self.sigma_history: deque = deque(maxlen=history_size)
        
        # Metrics tracking
        self.iterations: List[int] = []
        self.start_time = time.time()
        self.last_improvement_iter = 0
        self.total_improvements = 0
        self.best_ever_cost = float('inf')
        
        # Sparkline generator
        self.sparkline = SparklineGenerator()
    
    def update(self, metrics: CMAESMetrics, force_update: bool = False) -> Optional[Dict[str, Any]]:
        """
        Update visualization with new metrics.
        
        Args:
            metrics: New CMA-ES metrics
            force_update: Force UI update even if within update interval
            
        Returns:
            UI update dictionary if update performed, None otherwise
        """
        # Check if we should update
        current_time = time.time()
        if not force_update and (current_time - self.last_update_time) < self.update_interval:
            return None
        
        # Update history
        self.iterations.append(metrics.iteration)
        self.best_cost_history.append(metrics.best_cost)
        self.mean_cost_history.append(metrics.mean_cost)
        self.std_history.append(metrics.std_cost)
        self.sigma_history.append(metrics.sigma)
        
        # Track improvements
        if metrics.best_cost < self.best_ever_cost:
            self.best_ever_cost = metrics.best_cost
            self.last_improvement_iter = metrics.iteration
            self.total_improvements += 1
        
        # Generate visualization
        ui_update = self._generate_ui_update(metrics)
        
        self.last_update_time = current_time
        return ui_update
    
    def _generate_ui_update(self, metrics: CMAESMetrics) -> Dict[str, Any]:
        """Generate UI update dictionary with visualizations."""
        # Calculate derived metrics
        elapsed_time = time.time() - self.start_time
        iterations_per_sec = metrics.iteration / elapsed_time if elapsed_time > 0 else 0
        stagnation = metrics.iteration - self.last_improvement_iter
        convergence_rate = self._calculate_convergence_rate()
        
        # Generate sparklines
        cost_sparkline = self.sparkline.generate_sparkline(
            list(self.best_cost_history), width=40, style="blocks"
        )
        mean_sparkline = self.sparkline.generate_sparkline(
            list(self.mean_cost_history), width=30, style="ascii", show_values=False
        )
        sigma_sparkline = self.sparkline.generate_sparkline(
            list(self.sigma_history), width=20, style="dots", show_values=False
        )
        
        # Create progress bar
        progress_bar = self._create_progress_bar(convergence_rate)
        
        # Generate trend indicator
        trend = self._get_trend_indicator()
        
        return {
            "cmaes_visualization": {
                "timestamp": datetime.now().isoformat(),
                "iteration": metrics.iteration,
                
                "current_metrics": {
                    "best_cost": round(metrics.best_cost, 6),
                    "mean_cost": round(metrics.mean_cost, 6),
                    "std_cost": round(metrics.std_cost, 6),
                    "sigma": round(metrics.sigma, 6),
                    "population_size": metrics.population_size,
                    "converged": metrics.converged
                },
                
                "sparklines": {
                    "best_cost": {
                        "visualization": cost_sparkline,
                        "width": 40,
                        "min": round(min(self.best_cost_history), 4),
                        "max": round(max(self.best_cost_history), 4),
                        "current": round(metrics.best_cost, 4)
                    },
                    "mean_cost": {
                        "visualization": mean_sparkline,
                        "width": 30
                    },
                    "sigma": {
                        "visualization": sigma_sparkline,
                        "width": 20,
                        "current": round(metrics.sigma, 6)
                    }
                },
                
                "progress": {
                    "bar": progress_bar,
                    "convergence_rate": round(convergence_rate * 100, 1),
                    "stagnation_iterations": stagnation,
                    "total_improvements": self.total_improvements,
                    "best_ever_cost": round(self.best_ever_cost, 6),
                    "trend": trend
                },
                
                "performance": {
                    "elapsed_time_sec": round(elapsed_time, 1),
                    "iterations_per_sec": round(iterations_per_sec, 2),
                    "estimated_remaining_iter": self._estimate_remaining_iterations(),
                    "estimated_time_remaining_sec": self._estimate_time_remaining()
                },
                
                "display": {
                    "main_chart": self._generate_main_chart(),
                    "status_line": self._generate_status_line(metrics),
                    "compact_view": self._generate_compact_view(metrics)
                }
            }
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on recent improvements."""
        if len(self.best_cost_history) < 10:
            return 0.0
        
        # Compare last 10 iterations to previous 10
        recent = list(self.best_cost_history)[-10:]
        if len(self.best_cost_history) >= 20:
            previous = list(self.best_cost_history)[-20:-10]
            
            recent_avg = sum(recent) / len(recent)
            previous_avg = sum(previous) / len(previous)
            
            if previous_avg > 0:
                improvement = (previous_avg - recent_avg) / previous_avg
                return max(0, min(1, improvement))
        
        return 0.0
    
    def _create_progress_bar(self, progress: float, width: int = 30) -> str:
        """Create a progress bar visualization."""
        filled = int(progress * width)
        empty = width - filled
        
        bar = "█" * filled + "▒" * empty
        percentage = int(progress * 100)
        
        return f"[{bar}] {percentage}%"
    
    def _get_trend_indicator(self) -> str:
        """Get trend indicator based on recent cost changes."""
        if len(self.best_cost_history) < 3:
            return "→ Stable"
        
        recent = list(self.best_cost_history)[-3:]
        
        # Calculate trend
        improvements = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        
        if improvements >= 2:
            return "↓ Improving"
        elif improvements == 0:
            return "→ Stagnant"
        else:
            return "↔ Fluctuating"
    
    def _estimate_remaining_iterations(self) -> int:
        """Estimate remaining iterations based on convergence rate."""
        if len(self.sigma_history) < 2:
            return -1
        
        # Use sigma decay to estimate convergence
        current_sigma = self.sigma_history[-1]
        target_sigma = 1e-8  # Typical convergence threshold
        
        if current_sigma <= target_sigma:
            return 0
        
        # Estimate based on recent sigma decay rate
        if len(self.sigma_history) >= 10:
            recent_sigmas = list(self.sigma_history)[-10:]
            decay_rate = (recent_sigmas[0] - recent_sigmas[-1]) / (10 * recent_sigmas[0])
            
            if decay_rate > 0:
                remaining = math.log(target_sigma / current_sigma) / math.log(1 - decay_rate)
                return max(0, int(remaining))
        
        return -1  # Cannot estimate
    
    def _estimate_time_remaining(self) -> float:
        """Estimate time remaining based on current rate."""
        remaining_iter = self._estimate_remaining_iterations()
        if remaining_iter <= 0:
            return -1
        
        elapsed = time.time() - self.start_time
        if self.iterations and elapsed > 0:
            rate = self.iterations[-1] / elapsed
            return remaining_iter / rate if rate > 0 else -1
        
        return -1
    
    def _generate_main_chart(self) -> str:
        """Generate the main cost chart visualization."""
        if len(self.best_cost_history) < 2:
            return "Gathering data..."
        
        # Create multi-line chart
        chart_lines = []
        chart_height = 6
        chart_width = 50
        
        # Header
        chart_lines.append("┌" + "─" * (chart_width + 2) + "┐")
        chart_lines.append(f"│ {'Cost over Iterations':^{chart_width}} │")
        chart_lines.append("├" + "─" * (chart_width + 2) + "┤")
        
        # Generate chart area
        best_costs = list(self.best_cost_history)
        sparkline = self.sparkline.generate_sparkline(
            best_costs, width=chart_width, style="blocks", show_values=False
        )
        
        # Add sparkline with padding
        chart_lines.append(f"│ {sparkline} │")
        
        # Add min/max labels
        min_cost = min(best_costs)
        max_cost = max(best_costs)
        current = best_costs[-1]
        
        chart_lines.append("├" + "─" * (chart_width + 2) + "┤")
        chart_lines.append(f"│ Min: {min_cost:8.4f}  Current: {current:8.4f}  Max: {max_cost:8.4f} │")
        
        # Footer
        chart_lines.append("└" + "─" * (chart_width + 2) + "┘")
        
        return "\n".join(chart_lines)
    
    def _generate_status_line(self, metrics: CMAESMetrics) -> str:
        """Generate a single status line for compact display."""
        trend = "↓" if len(self.best_cost_history) > 1 and self.best_cost_history[-1] < self.best_cost_history[-2] else "→"
        
        return (f"Iter {metrics.iteration:4d} │ "
                f"Cost: {metrics.best_cost:8.4f} {trend} │ "
                f"σ: {metrics.sigma:.3e} │ "
                f"{'CONVERGED' if metrics.converged else 'OPTIMIZING'}")
    
    def _generate_compact_view(self, metrics: CMAESMetrics) -> str:
        """Generate a compact single-line view with sparkline."""
        sparkline = self.sparkline.generate_sparkline(
            list(self.best_cost_history)[-20:],  # Last 20 values
            width=15,
            style="blocks",
            show_values=False
        )
        
        return f"[{metrics.iteration:3d}] {sparkline} {metrics.best_cost:.4f}"
    
    def print_update(self, metrics: CMAESMetrics):
        """Print formatted update to console."""
        update = self.update(metrics, force_update=True)
        if update:
            viz = update["cmaes_visualization"]
            
            # Print main chart
            print("\n" + viz["display"]["main_chart"])
            
            # Print sparklines section
            print("\n📊 SPARKLINES:")
            print(f"  Best Cost: {viz['sparklines']['best_cost']['visualization']}")
            print(f"  Mean Cost: {viz['sparklines']['mean_cost']['visualization']}")
            print(f"  Sigma:     {viz['sparklines']['sigma']['visualization']}")
            
            # Print progress
            print(f"\n📈 PROGRESS:")
            print(f"  {viz['progress']['bar']}")
            print(f"  Trend: {viz['progress']['trend']}")
            print(f"  Improvements: {viz['progress']['total_improvements']}")
            
            # Print status
            print(f"\n📋 STATUS:")
            print(f"  {viz['display']['status_line']}")
            
            # Print JSON for UI
            print("\n" + "="*60)
            print("UI JSON UPDATE:")
            # Convert numpy types to native Python types for JSON serialization
            json_safe = json.loads(json.dumps(update, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
            print(json.dumps(json_safe, indent=2))
            print("="*60)
    
    def save_history(self, filepath: str):
        """Save optimization history to file."""
        history = {
            "iterations": self.iterations,
            "best_cost_history": list(self.best_cost_history),
            "mean_cost_history": list(self.mean_cost_history),
            "std_history": list(self.std_history),
            "sigma_history": list(self.sigma_history),
            "best_ever_cost": self.best_ever_cost,
            "total_improvements": self.total_improvements,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_history(self, filepath: str):
        """Load optimization history from file."""
        with open(filepath, 'r') as f:
            history = json.load(f)
        
        self.iterations = history.get("iterations", [])
        self.best_cost_history = deque(history.get("best_cost_history", []), maxlen=self.history_size)
        self.mean_cost_history = deque(history.get("mean_cost_history", []), maxlen=self.history_size)
        self.std_history = deque(history.get("std_history", []), maxlen=self.history_size)
        self.sigma_history = deque(history.get("sigma_history", []), maxlen=self.history_size)
        self.best_ever_cost = history.get("best_ever_cost", float('inf'))
        self.total_improvements = history.get("total_improvements", 0)


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("CMA-ES SPARKLINE VISUALIZATION DEMO")
    print("="*70)
    
    # Create visualizer
    visualizer = CMAESVisualizer()
    
    # Simulate optimization progress
    np.random.seed(42)
    best_cost = 100.0
    sigma = 1.0
    
    for iteration in range(30):
        # Simulate cost improvement with some noise
        improvement = np.random.exponential(2.0) * (1.0 - iteration / 30)
        best_cost = max(0.1, best_cost - improvement)
        
        # Add noise for mean cost
        mean_cost = best_cost + abs(np.random.normal(5, 2))
        std_cost = abs(np.random.normal(2, 0.5))
        
        # Decay sigma
        sigma *= 0.95
        
        # Create metrics
        metrics = CMAESMetrics(
            iteration=iteration + 1,
            best_cost=best_cost,
            mean_cost=mean_cost,
            std_cost=std_cost,
            population_size=20,
            sigma=sigma,
            converged=(sigma < 1e-6),
            improvement=improvement
        )
        
        # Print update every 5 iterations
        if (iteration + 1) % 5 == 0:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*70}")
            visualizer.print_update(metrics)
            time.sleep(0.1)  # Small delay for visualization
    
    # Save history
    visualizer.save_history("cmaes_history_demo.json")
    print("\n✅ History saved to cmaes_history_demo.json")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)