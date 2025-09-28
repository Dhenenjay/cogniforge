"""
Waypoint Optimizer with Diff Visualization

Shows differences between original expert waypoints and optimized versions.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
from enum import Enum
import logging
from colorama import init, Fore, Back, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization applied"""
    SMOOTH = "smooth"           # Trajectory smoothing
    SHORTCUT = "shortcut"       # Remove redundant waypoints
    VELOCITY = "velocity"       # Optimize velocity profile
    COLLISION = "collision"     # Collision avoidance
    ENERGY = "energy"          # Energy minimization


@dataclass
class Waypoint:
    """Single waypoint representation"""
    position: np.ndarray
    gripper: float
    timestamp: Optional[float] = None
    velocity: Optional[np.ndarray] = None
    description: str = ""
    
    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate distance to another waypoint"""
        return np.linalg.norm(self.position - other.position)
        
    def __str__(self) -> str:
        pos_str = f"[{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}]"
        grip_str = f"grip={self.gripper:.1f}"
        return f"{pos_str} {grip_str}"


class WaypointOptimizer:
    """Optimizes waypoint trajectories"""
    
    def __init__(self, optimization_params: Optional[Dict] = None, method: str = 'default'):
        """
        Initialize optimizer
        
        Args:
            optimization_params: Optimization parameters
            method: Optimization method hint (e.g., 'spline', 'default')
        """
        self.params = optimization_params or self._get_default_params()
        self.method = method
        self.optimization_history = []
        
    def _get_default_params(self) -> Dict:
        """Get default optimization parameters"""
        return {
            'smoothing_window': 3,
            'min_waypoint_distance': 0.02,  # 2cm
            'max_velocity': 0.5,            # m/s
            'max_acceleration': 1.0,        # m/s²
            'corner_radius': 0.05,          # 5cm corner smoothing
            'redundancy_threshold': 0.01    # 1cm for redundant points
        }
        
    def optimize(self, waypoints: List[Waypoint], 
                optimization_types: List[OptimizationType] = None) -> List[Waypoint]:
        """
        Optimize waypoint trajectory
        
        Args:
            waypoints: Original waypoints
            optimization_types: Types of optimization to apply
            
        Returns:
            Optimized waypoints
        """
        if optimization_types is None:
            optimization_types = [
                OptimizationType.SHORTCUT,
                OptimizationType.SMOOTH,
                OptimizationType.VELOCITY
            ]
            
        optimized = waypoints.copy()
        
        for opt_type in optimization_types:
            if opt_type == OptimizationType.SHORTCUT:
                optimized = self._remove_redundant_waypoints(optimized)
            elif opt_type == OptimizationType.SMOOTH:
                optimized = self._smooth_trajectory(optimized)
            elif opt_type == OptimizationType.VELOCITY:
                optimized = self._optimize_velocity_profile(optimized)
            elif opt_type == OptimizationType.COLLISION:
                optimized = self._add_collision_avoidance(optimized)
            elif opt_type == OptimizationType.ENERGY:
                optimized = self._minimize_energy(optimized)
                
        return optimized
        
    def _remove_redundant_waypoints(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Remove redundant waypoints that are colinear or too close"""
        if len(waypoints) <= 2:
            return waypoints
            
        optimized = [waypoints[0]]
        
        for i in range(1, len(waypoints) - 1):
            prev_wp = optimized[-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]
            
            # Check if current point is redundant
            if self._is_redundant(prev_wp, curr_wp, next_wp):
                continue  # Skip redundant point
                
            optimized.append(curr_wp)
            
        optimized.append(waypoints[-1])  # Always keep last waypoint
        
        return optimized
        
    def _is_redundant(self, prev: Waypoint, curr: Waypoint, next: Waypoint) -> bool:
        """Check if middle waypoint is redundant"""
        
        # Check if too close to previous
        if curr.distance_to(prev) < self.params['redundancy_threshold']:
            return True
            
        # Check if colinear
        v1 = curr.position - prev.position
        v2 = next.position - curr.position
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # If vectors are nearly parallel (dot product close to 1)
            if np.dot(v1_norm, v2_norm) > 0.98:
                return True
                
        return False
        
    def _smooth_trajectory(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Smooth trajectory using moving average"""
        if len(waypoints) <= 2:
            return waypoints
            
        window = self.params['smoothing_window']
        smoothed = []
        
        for i, wp in enumerate(waypoints):
            if i == 0 or i == len(waypoints) - 1:
                # Keep first and last waypoints unchanged
                smoothed.append(wp)
            else:
                # Apply smoothing
                start_idx = max(0, i - window // 2)
                end_idx = min(len(waypoints), i + window // 2 + 1)
                
                window_positions = [waypoints[j].position for j in range(start_idx, end_idx)]
                avg_position = np.mean(window_positions, axis=0)
                
                smoothed_wp = Waypoint(
                    position=avg_position,
                    gripper=wp.gripper,
                    description=wp.description + " (smoothed)"
                )
                smoothed.append(smoothed_wp)
                
        return smoothed
        
    def _optimize_velocity_profile(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Optimize velocity profile for smooth motion"""
        if len(waypoints) <= 1:
            return waypoints
            
        optimized = []
        total_distance = sum(waypoints[i].distance_to(waypoints[i+1]) 
                           for i in range(len(waypoints)-1))
        
        for i, wp in enumerate(waypoints):
            opt_wp = Waypoint(
                position=wp.position.copy(),
                gripper=wp.gripper,
                description=wp.description
            )
            
            # Calculate velocity based on position in trajectory
            if i == 0 or i == len(waypoints) - 1:
                # Zero velocity at start/end
                opt_wp.velocity = np.zeros(3)
            else:
                # Trapezoidal velocity profile
                progress = sum(waypoints[j].distance_to(waypoints[j+1]) 
                             for j in range(i)) / total_distance
                
                if progress < 0.2:  # Acceleration phase
                    speed = self.params['max_velocity'] * (progress / 0.2)
                elif progress > 0.8:  # Deceleration phase
                    speed = self.params['max_velocity'] * ((1 - progress) / 0.2)
                else:  # Constant velocity phase
                    speed = self.params['max_velocity']
                    
                direction = waypoints[i+1].position - wp.position
                if np.linalg.norm(direction) > 0:
                    opt_wp.velocity = direction / np.linalg.norm(direction) * speed
                else:
                    opt_wp.velocity = np.zeros(3)
                    
            optimized.append(opt_wp)
            
        return optimized
        
    def _add_collision_avoidance(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Add waypoints for collision avoidance"""
        # Simplified - would need actual obstacle information
        optimized = []
        
        for i, wp in enumerate(waypoints):
            optimized.append(wp)
            
            # Add intermediate waypoint if large height change
            if i < len(waypoints) - 1:
                height_diff = abs(waypoints[i+1].position[2] - wp.position[2])
                if height_diff > 0.1:  # 10cm height difference
                    # Add intermediate waypoint at higher altitude
                    mid_pos = (wp.position + waypoints[i+1].position) / 2
                    mid_pos[2] = max(wp.position[2], waypoints[i+1].position[2]) + 0.05
                    
                    intermediate = Waypoint(
                        position=mid_pos,
                        gripper=wp.gripper,
                        description="Collision avoidance"
                    )
                    optimized.append(intermediate)
                    
        return optimized
        
    def _minimize_energy(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Minimize energy consumption"""
        # Prefer movements that minimize vertical displacement
        optimized = []
        
        for i, wp in enumerate(waypoints):
            opt_wp = Waypoint(
                position=wp.position.copy(),
                gripper=wp.gripper,
                description=wp.description
            )
            
            # Reduce unnecessary vertical movements in middle waypoints
            if 0 < i < len(waypoints) - 1:
                prev_z = waypoints[i-1].position[2]
                next_z = waypoints[i+1].position[2]
                
                # If current height is unnecessarily high
                if wp.position[2] > max(prev_z, next_z) + 0.05:
                    opt_wp.position[2] = max(prev_z, next_z) + 0.05
                    opt_wp.description += " (height optimized)"
                    
            optimized.append(opt_wp)
            
        return optimized

    # Backward-compatible API used by older tests/demos
    def optimize_trajectory(self, waypoints: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """Optimize a raw waypoint array and return an optimized array.
        Accepts (N,3) numpy array and returns optimized (M,3) array.
        """
        # Convert array to Waypoint objects
        wp_objs = [Waypoint(position=np.array(wp, dtype=float), gripper=1.0) for wp in waypoints]
        optimized = self.optimize(wp_objs)
        # Convert back to numpy array
        return np.stack([wp.position for wp in optimized], axis=0)
        
    def compute_metrics(self, waypoints: List[Waypoint]) -> Dict[str, float]:
        """Compute trajectory metrics"""
        metrics = {
            'total_distance': 0.0,
            'max_segment_distance': 0.0,
            'avg_segment_distance': 0.0,
            'total_time': 0.0,
            'smoothness': 0.0,
            'num_waypoints': len(waypoints)
        }
        
        if len(waypoints) <= 1:
            return metrics
            
        # Calculate distances
        distances = []
        for i in range(len(waypoints) - 1):
            dist = waypoints[i].distance_to(waypoints[i + 1])
            distances.append(dist)
            
        metrics['total_distance'] = sum(distances)
        metrics['max_segment_distance'] = max(distances) if distances else 0
        metrics['avg_segment_distance'] = np.mean(distances) if distances else 0
        
        # Estimate time (assuming constant velocity between waypoints)
        if self.params['max_velocity'] > 0:
            metrics['total_time'] = metrics['total_distance'] / self.params['max_velocity']
            
        # Calculate smoothness (sum of angle changes)
        if len(waypoints) >= 3:
            angle_changes = []
            for i in range(1, len(waypoints) - 1):
                v1 = waypoints[i].position - waypoints[i-1].position
                v2 = waypoints[i+1].position - waypoints[i].position
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))
                    angle_changes.append(angle)
                    
            metrics['smoothness'] = sum(angle_changes) if angle_changes else 0
            
        return metrics


class WaypointDiffViewer:
    """Visualizes differences between original and optimized waypoints"""
    
    @staticmethod
    def show_diff(original: List[Waypoint], optimized: List[Waypoint],
                  show_metrics: bool = True, show_details: bool = False):
        """
        Show diff between original and optimized waypoints
        
        Args:
            original: Original expert waypoints
            optimized: Optimized waypoints
            show_metrics: Show trajectory metrics
            show_details: Show detailed waypoint comparison
        """
        print("\n" + "="*80)
        print(f"{Fore.CYAN} WAYPOINT OPTIMIZATION DIFF VIEW")
        print("="*80)
        
        # Summary
        print(f"\n{Fore.YELLOW}📊 SUMMARY:")
        print(f"  Original:  {len(original)} waypoints")
        print(f"  Optimized: {len(optimized)} waypoints")
        
        reduction = len(original) - len(optimized)
        if reduction > 0:
            pct = (reduction / len(original)) * 100
            print(f"  {Fore.GREEN}Reduction: {reduction} waypoints ({pct:.1f}%)")
        elif reduction < 0:
            print(f"  {Fore.RED}Increase:  {-reduction} waypoints")
        else:
            print(f"  {Fore.YELLOW}No change in waypoint count")
            
        # Compute metrics
        if show_metrics:
            optimizer = WaypointOptimizer()
            orig_metrics = optimizer.compute_metrics(original)
            opt_metrics = optimizer.compute_metrics(optimized)
            
            print(f"\n{Fore.YELLOW}📈 METRICS COMPARISON:")
            print(f"  {'Metric':<20} {'Original':>12} {'Optimized':>12} {'Change':>12}")
            print("  " + "-"*58)
            
            for key in orig_metrics:
                orig_val = orig_metrics[key]
                opt_val = opt_metrics[key]
                
                if isinstance(orig_val, int):
                    change = opt_val - orig_val
                    change_str = f"{change:+d}"
                    print(f"  {key:<20} {orig_val:>12d} {opt_val:>12d} {change_str:>12}")
                else:
                    change = opt_val - orig_val
                    pct_change = (change / orig_val * 100) if orig_val != 0 else 0
                    
                    # Color code improvements
                    if key in ['total_distance', 'total_time', 'smoothness']:
                        if change < 0:
                            change_str = f"{Fore.GREEN}{change:+.3f} ({pct_change:+.1f}%){Style.RESET_ALL}"
                        elif change > 0:
                            change_str = f"{Fore.RED}{change:+.3f} ({pct_change:+.1f}%){Style.RESET_ALL}"
                        else:
                            change_str = f"{change:+.3f}"
                    else:
                        change_str = f"{change:+.3f}"
                        
                    print(f"  {key:<20} {orig_val:>12.3f} {opt_val:>12.3f} {change_str:>12}")
                    
        # Detailed waypoint comparison
        if show_details:
            print(f"\n{Fore.YELLOW}📍 WAYPOINT DETAILS:")
            
            # Create alignment between original and optimized
            WaypointDiffViewer._show_aligned_diff(original, optimized)
            
    @staticmethod
    def _show_aligned_diff(original: List[Waypoint], optimized: List[Waypoint]):
        """Show aligned diff of waypoints"""
        
        # Simple diff visualization
        max_len = max(len(original), len(optimized))
        
        print(f"\n  {'#':<3} {'Original':<35} {'':^5} {'Optimized':<35}")
        print("  " + "-"*80)
        
        for i in range(max_len):
            orig_str = ""
            opt_str = ""
            symbol = "  "
            
            if i < len(original):
                orig_str = str(original[i])
                
            if i < len(optimized):
                opt_str = str(optimized[i])
                
            # Determine change type
            if i >= len(original):
                symbol = f"{Fore.GREEN}++{Style.RESET_ALL}"  # Added
                opt_str = f"{Fore.GREEN}{opt_str}{Style.RESET_ALL}"
            elif i >= len(optimized):
                symbol = f"{Fore.RED}--{Style.RESET_ALL}"  # Removed
                orig_str = f"{Fore.RED}{orig_str}{Style.RESET_ALL}"
            elif original[i].distance_to(optimized[i]) > 0.001:
                symbol = f"{Fore.YELLOW}~~{Style.RESET_ALL}"  # Modified
                orig_str = f"{Fore.YELLOW}{orig_str}{Style.RESET_ALL}"
                opt_str = f"{Fore.YELLOW}{opt_str}{Style.RESET_ALL}"
            else:
                symbol = "  "  # Unchanged
                
            print(f"  {i:>2} {orig_str:<35} {symbol:^5} {opt_str:<35}")
            
    @staticmethod
    def show_compact_diff(original: List[Waypoint], optimized: List[Waypoint]):
        """Show compact diff summary"""
        
        optimizer = WaypointOptimizer()
        orig_metrics = optimizer.compute_metrics(original)
        opt_metrics = optimizer.compute_metrics(optimized)
        
        # Compact one-line summary
        dist_change = opt_metrics['total_distance'] - orig_metrics['total_distance']
        wp_change = len(optimized) - len(original)
        
        print(f"\n{Fore.CYAN}[OPTIMIZATION RESULT]{Style.RESET_ALL}")
        
        # Distance
        if dist_change < 0:
            print(f"  ✅ Distance: {orig_metrics['total_distance']:.3f}m → "
                  f"{opt_metrics['total_distance']:.3f}m "
                  f"{Fore.GREEN}({dist_change:.3f}m, {dist_change/orig_metrics['total_distance']*100:+.1f}%){Style.RESET_ALL}")
        else:
            print(f"  📊 Distance: {orig_metrics['total_distance']:.3f}m → "
                  f"{opt_metrics['total_distance']:.3f}m ({dist_change:+.3f}m)")
            
        # Waypoints
        if wp_change < 0:
            print(f"  ✅ Waypoints: {len(original)} → {len(optimized)} "
                  f"{Fore.GREEN}({wp_change} waypoints){Style.RESET_ALL}")
        elif wp_change > 0:
            print(f"  ⚠️  Waypoints: {len(original)} → {len(optimized)} "
                  f"{Fore.YELLOW}(+{wp_change} for safety){Style.RESET_ALL}")
        else:
            print(f"  📊 Waypoints: {len(original)} (unchanged)")
            
        # Smoothness
        smooth_change = opt_metrics['smoothness'] - orig_metrics['smoothness']
        if smooth_change < 0:
            print(f"  ✅ Smoothness: {Fore.GREEN}{abs(smooth_change):.2f} rad smoother{Style.RESET_ALL}")
        elif smooth_change > 0:
            print(f"  ⚠️  Smoothness: {Fore.YELLOW}{smooth_change:.2f} rad less smooth{Style.RESET_ALL}")


# Demo function
def demo_waypoint_optimization():
    """Demonstrate waypoint optimization with diff view"""
    
    print(f"\n{Fore.CYAN}{'='*80}")
    print(" WAYPOINT OPTIMIZATION DEMONSTRATION")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    # Create example expert waypoints (pick and place trajectory)
    expert_waypoints = [
        Waypoint(np.array([0.3, 0.0, 0.1]), 1.0, description="Start"),
        Waypoint(np.array([0.35, 0.0, 0.15]), 1.0, description="Approach high"),
        Waypoint(np.array([0.4, 0.0, 0.2]), 1.0, description="Above object"),
        Waypoint(np.array([0.4, 0.01, 0.2]), 1.0, description="Slight adjustment"),  # Redundant
        Waypoint(np.array([0.4, 0.0, 0.15]), 1.0, description="Lower"),
        Waypoint(np.array([0.4, 0.0, 0.1]), 1.0, description="Pre-grasp"),
        Waypoint(np.array([0.4, 0.0, 0.05]), 1.0, description="Grasp position"),
        Waypoint(np.array([0.4, 0.0, 0.05]), 0.0, description="Close gripper"),
        Waypoint(np.array([0.4, 0.0, 0.15]), 0.0, description="Lift"),
        Waypoint(np.array([0.4, 0.0, 0.25]), 0.0, description="Lift high"),  # Unnecessarily high
        Waypoint(np.array([0.3, 0.2, 0.25]), 0.0, description="Move to target"),
        Waypoint(np.array([0.3, 0.2, 0.2]), 0.0, description="Above target"),
        Waypoint(np.array([0.3, 0.2, 0.15]), 0.0, description="Lower"),
        Waypoint(np.array([0.3, 0.2, 0.1]), 0.0, description="Place position"),
        Waypoint(np.array([0.3, 0.2, 0.1]), 1.0, description="Release"),
        Waypoint(np.array([0.3, 0.2, 0.15]), 1.0, description="Retract")
    ]
    
    # Create optimizer
    optimizer = WaypointOptimizer()
    
    # Test different optimization strategies
    print(f"\n{Fore.YELLOW}1. SHORTCUT OPTIMIZATION (Remove redundant points):{Style.RESET_ALL}")
    optimized_shortcut = optimizer.optimize(expert_waypoints, [OptimizationType.SHORTCUT])
    WaypointDiffViewer.show_compact_diff(expert_waypoints, optimized_shortcut)
    
    print(f"\n{Fore.YELLOW}2. SMOOTHING OPTIMIZATION (Smooth trajectory):{Style.RESET_ALL}")
    optimized_smooth = optimizer.optimize(expert_waypoints, [OptimizationType.SMOOTH])
    WaypointDiffViewer.show_compact_diff(expert_waypoints, optimized_smooth)
    
    print(f"\n{Fore.YELLOW}3. ENERGY OPTIMIZATION (Minimize vertical movement):{Style.RESET_ALL}")
    optimized_energy = optimizer.optimize(expert_waypoints, [OptimizationType.ENERGY])
    WaypointDiffViewer.show_compact_diff(expert_waypoints, optimized_energy)
    
    print(f"\n{Fore.YELLOW}4. COMBINED OPTIMIZATION (All strategies):{Style.RESET_ALL}")
    optimized_all = optimizer.optimize(expert_waypoints)
    
    # Show detailed diff for combined optimization
    WaypointDiffViewer.show_diff(expert_waypoints, optimized_all, 
                                 show_metrics=True, show_details=True)
    
    print(f"\n{Fore.GREEN}✅ Optimization complete!{Style.RESET_ALL}")
    
    return expert_waypoints, optimized_all


if __name__ == "__main__":
    # Run demonstration
    original, optimized = demo_waypoint_optimization()
    
    # Additional examples
    print(f"\n{Fore.CYAN}{'='*80}")
    print(" ADDITIONAL OPTIMIZATION EXAMPLES")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    # Example 1: Simple push trajectory
    print(f"\n{Fore.YELLOW}Example 1: Push Trajectory Optimization{Style.RESET_ALL}")
    push_waypoints = [
        Waypoint(np.array([0.4, -0.1, 0.05]), 1.0),
        Waypoint(np.array([0.42, -0.1, 0.05]), 1.0),
        Waypoint(np.array([0.44, -0.1, 0.05]), 1.0),
        Waypoint(np.array([0.46, -0.1, 0.05]), 1.0),
        Waypoint(np.array([0.48, -0.1, 0.05]), 1.0),
        Waypoint(np.array([0.5, -0.1, 0.05]), 1.0),
    ]
    
    optimizer = WaypointOptimizer()
    optimized_push = optimizer.optimize(push_waypoints, [OptimizationType.SHORTCUT])
    WaypointDiffViewer.show_compact_diff(push_waypoints, optimized_push)
    
    # Example 2: Complex trajectory with unnecessary movements
    print(f"\n{Fore.YELLOW}Example 2: Complex Trajectory Simplification{Style.RESET_ALL}")
    complex_waypoints = [
        Waypoint(np.array([0.3, 0.0, 0.1]), 1.0),
        Waypoint(np.array([0.3, 0.1, 0.1]), 1.0),
        Waypoint(np.array([0.3, 0.1, 0.2]), 1.0),  # Up
        Waypoint(np.array([0.4, 0.1, 0.2]), 1.0),  # Forward
        Waypoint(np.array([0.4, 0.1, 0.1]), 1.0),  # Down
        Waypoint(np.array([0.4, 0.2, 0.1]), 1.0),
    ]
    
    optimized_complex = optimizer.optimize(complex_waypoints, 
                                          [OptimizationType.SHORTCUT, 
                                           OptimizationType.ENERGY])
    WaypointDiffViewer.show_compact_diff(complex_waypoints, optimized_complex)