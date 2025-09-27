"""
Robot control utilities for safe gripper positioning and micro-adjustments.

Provides functions for applying small, safe adjustments to pre-grasp poses
with proper safety envelopes and constraints.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GraspPhase(Enum):
    """Phases of grasping operation."""
    APPROACH = "approach"
    PRE_GRASP = "pre_grasp"
    GRASP = "grasp"
    LIFT = "lift"
    RETREAT = "retreat"


@dataclass
class SafetyEnvelope:
    """Safety limits for robot workspace."""
    x_min: float = -0.5  # meters
    x_max: float = 0.5
    y_min: float = -0.5
    y_max: float = 0.5
    z_min: float = 0.0   # Don't go below table
    z_max: float = 0.8
    
    # Maximum velocities (m/s)
    max_linear_vel: float = 0.1
    max_angular_vel: float = 0.5  # rad/s
    
    # Maximum accelerations (m/s²)
    max_linear_acc: float = 0.5
    max_angular_acc: float = 1.0
    
    def is_within_bounds(self, x: float, y: float, z: float = None) -> bool:
        """Check if position is within safety envelope."""
        if not (self.x_min <= x <= self.x_max):
            return False
        if not (self.y_min <= y <= self.y_max):
            return False
        if z is not None and not (self.z_min <= z <= self.z_max):
            return False
        return True


def apply_micro_nudge(
    dx_m: float,
    dy_m: float,
    limit: float = 0.03,
    dz_m: float = 0.0,
    safety_envelope: Optional[SafetyEnvelope] = None,
    current_position: Optional[Tuple[float, float, float]] = None,
    adaptive_limit: bool = True,
    phase: GraspPhase = GraspPhase.PRE_GRASP,
    smoothing: float = 0.0,
    previous_nudge: Optional[Tuple[float, float, float]] = None
) -> Tuple[float, float, float]:
    """
    Apply micro-adjustment to pre-grasp pose with safety clamping.
    
    Safely adjusts the gripper position by small amounts to improve
    grasp alignment, with proper safety envelope enforcement.
    
    Args:
        dx_m: Desired adjustment in x-direction (meters)
        dy_m: Desired adjustment in y-direction (meters)
        limit: Maximum adjustment magnitude (default 0.03m = 3cm)
        dz_m: Desired adjustment in z-direction (default 0)
        safety_envelope: Safety bounds for robot workspace
        current_position: Current gripper position (x, y, z) in meters
        adaptive_limit: Adapt limit based on grasp phase
        phase: Current grasping phase
        smoothing: Smoothing factor [0,1] to blend with previous nudge
        previous_nudge: Previous nudge for smoothing
        
    Returns:
        (dx_clamped, dy_clamped, dz_clamped): Clamped adjustments in meters
        
    Example:
        # Apply small adjustment for final alignment
        dx_safe, dy_safe, dz_safe = apply_micro_nudge(
            dx_m=0.05,    # Want to move 5cm right
            dy_m=-0.02,   # Want to move 2cm up
            limit=0.03    # But limit to 3cm max
        )
        # Result will be clamped to stay within limit
    """
    # Use default safety envelope if none provided
    if safety_envelope is None:
        safety_envelope = SafetyEnvelope()
    
    # Adapt limit based on grasp phase
    if adaptive_limit:
        phase_limits = {
            GraspPhase.APPROACH: limit * 2.0,      # More freedom during approach
            GraspPhase.PRE_GRASP: limit,           # Standard limit
            GraspPhase.GRASP: limit * 0.5,         # Very precise during grasp
            GraspPhase.LIFT: limit * 0.3,          # Minimal adjustments during lift
            GraspPhase.RETREAT: limit * 1.5        # Relaxed during retreat
        }
        limit = phase_limits.get(phase, limit)
    
    # Compute magnitude of requested adjustment
    requested_magnitude = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
    
    # Clamp to limit while preserving direction
    if requested_magnitude > limit:
        scale = limit / requested_magnitude
        dx_clamped = dx_m * scale
        dy_clamped = dy_m * scale
        dz_clamped = dz_m * scale
        
        logger.info(f"Nudge clamped from {requested_magnitude:.3f}m to {limit:.3f}m")
    else:
        dx_clamped = dx_m
        dy_clamped = dy_m
        dz_clamped = dz_m
    
    # Apply smoothing if previous nudge provided
    if smoothing > 0 and previous_nudge is not None:
        dx_prev, dy_prev, dz_prev = previous_nudge
        dx_clamped = smoothing * dx_prev + (1 - smoothing) * dx_clamped
        dy_clamped = smoothing * dy_prev + (1 - smoothing) * dy_clamped
        dz_clamped = smoothing * dz_prev + (1 - smoothing) * dz_clamped
    
    # Check safety envelope if current position is known
    if current_position is not None:
        x_current, y_current, z_current = current_position
        x_new = x_current + dx_clamped
        y_new = y_current + dy_clamped
        z_new = z_current + dz_clamped
        
        # Clamp to stay within safety envelope
        if x_new < safety_envelope.x_min:
            dx_clamped = safety_envelope.x_min - x_current
            logger.warning(f"X adjustment clamped to stay within bounds")
        elif x_new > safety_envelope.x_max:
            dx_clamped = safety_envelope.x_max - x_current
            logger.warning(f"X adjustment clamped to stay within bounds")
        
        if y_new < safety_envelope.y_min:
            dy_clamped = safety_envelope.y_min - y_current
            logger.warning(f"Y adjustment clamped to stay within bounds")
        elif y_new > safety_envelope.y_max:
            dy_clamped = safety_envelope.y_max - y_current
            logger.warning(f"Y adjustment clamped to stay within bounds")
        
        if z_new < safety_envelope.z_min:
            dz_clamped = safety_envelope.z_min - z_current
            logger.warning(f"Z adjustment clamped to stay above minimum")
        elif z_new > safety_envelope.z_max:
            dz_clamped = safety_envelope.z_max - z_current
            logger.warning(f"Z adjustment clamped to stay below maximum")
    
    # Final safety check: ensure non-NaN and finite
    dx_clamped = np.clip(dx_clamped, -limit, limit) if np.isfinite(dx_clamped) else 0.0
    dy_clamped = np.clip(dy_clamped, -limit, limit) if np.isfinite(dy_clamped) else 0.0
    dz_clamped = np.clip(dz_clamped, -limit, limit) if np.isfinite(dz_clamped) else 0.0
    
    return (dx_clamped, dy_clamped, dz_clamped)


def apply_micro_nudge_simple(
    dx_m: float,
    dy_m: float,
    limit: float = 0.03
) -> Tuple[float, float]:
    """
    Simple version of micro nudge for 2D adjustments.
    
    Args:
        dx_m: Desired x adjustment in meters
        dy_m: Desired y adjustment in meters  
        limit: Maximum adjustment magnitude (default 3cm)
        
    Returns:
        (dx_clamped, dy_clamped): Clamped adjustments
        
    Example:
        dx_safe, dy_safe = apply_micro_nudge_simple(0.05, -0.02, 0.03)
    """
    result = apply_micro_nudge(dx_m, dy_m, limit)
    return (result[0], result[1])


def compute_grasp_adjustment(
    object_center: Tuple[float, float, float],
    gripper_position: Tuple[float, float, float],
    max_adjustment: float = 0.05,
    alignment_tolerance: float = 0.005,
    approach_vector: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute optimal grasp adjustment based on object and gripper positions.
    
    Args:
        object_center: Object center position (x, y, z) in meters
        gripper_position: Current gripper position (x, y, z) in meters
        max_adjustment: Maximum allowed adjustment
        alignment_tolerance: Tolerance for considering alignment achieved
        approach_vector: Preferred approach direction (default: from above)
        
    Returns:
        Dictionary containing:
        - 'adjustment': (dx, dy, dz) adjustment vector
        - 'distance': Current distance to object
        - 'aligned': Boolean indicating if within tolerance
        - 'adjustment_magnitude': Magnitude of adjustment
        
    Example:
        result = compute_grasp_adjustment(
            object_center=(0.3, 0.1, 0.05),
            gripper_position=(0.32, 0.08, 0.15)
        )
        dx, dy, dz = result['adjustment']
    """
    # Convert to numpy arrays
    obj_pos = np.array(object_center)
    grip_pos = np.array(gripper_position)
    
    # Compute offset vector
    offset = obj_pos - grip_pos
    distance = np.linalg.norm(offset)
    
    # Check if already aligned
    aligned = distance < alignment_tolerance
    
    if aligned:
        return {
            'adjustment': (0.0, 0.0, 0.0),
            'distance': distance,
            'aligned': True,
            'adjustment_magnitude': 0.0
        }
    
    # Compute adjustment
    if approach_vector is not None:
        # Project offset onto approach plane
        approach = np.array(approach_vector) / np.linalg.norm(approach_vector)
        offset_perp = offset - np.dot(offset, approach) * approach
        adjustment = offset_perp
    else:
        # Default: adjust in x-y plane only (approach from above)
        adjustment = offset.copy()
        adjustment[2] = 0  # No z adjustment
    
    # Apply micro nudge clamping
    dx, dy, dz = apply_micro_nudge(
        adjustment[0], 
        adjustment[1],
        limit=max_adjustment,
        dz_m=adjustment[2] if len(adjustment) > 2 else 0
    )
    
    adjustment_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return {
        'adjustment': (dx, dy, dz),
        'distance': distance,
        'aligned': False,
        'adjustment_magnitude': adjustment_magnitude
    }


def iterative_grasp_alignment(
    get_object_position_fn,
    get_gripper_position_fn,
    apply_adjustment_fn,
    max_iterations: int = 10,
    tolerance: float = 0.005,
    adjustment_limit: float = 0.03,
    convergence_rate: float = 0.8
) -> Dict[str, Any]:
    """
    Iteratively align gripper with object using micro nudges.
    
    Args:
        get_object_position_fn: Function returning object position (x, y, z)
        get_gripper_position_fn: Function returning gripper position (x, y, z)
        apply_adjustment_fn: Function to apply adjustment (dx, dy, dz) -> success
        max_iterations: Maximum alignment iterations
        tolerance: Alignment tolerance in meters
        adjustment_limit: Limit for each micro nudge
        convergence_rate: Rate of adjustment reduction per iteration
        
    Returns:
        Dictionary with alignment results
        
    Example:
        result = iterative_grasp_alignment(
            get_object_position_fn=lambda: detect_object_position(),
            get_gripper_position_fn=lambda: robot.get_gripper_position(),
            apply_adjustment_fn=lambda adj: robot.move_gripper_relative(adj)
        )
    """
    history = []
    current_limit = adjustment_limit
    
    for iteration in range(max_iterations):
        # Get current positions
        obj_pos = get_object_position_fn()
        grip_pos = get_gripper_position_fn()
        
        # Compute required adjustment
        result = compute_grasp_adjustment(
            obj_pos, grip_pos,
            max_adjustment=current_limit,
            alignment_tolerance=tolerance
        )
        
        # Check if aligned
        if result['aligned']:
            logger.info(f"Alignment achieved in {iteration} iterations")
            return {
                'success': True,
                'iterations': iteration,
                'final_distance': result['distance'],
                'history': history
            }
        
        # Apply micro nudge
        dx, dy, dz = result['adjustment']
        dx_safe, dy_safe, dz_safe = apply_micro_nudge(
            dx, dy,
            limit=current_limit,
            dz_m=dz,
            phase=GraspPhase.PRE_GRASP
        )
        
        # Apply adjustment
        success = apply_adjustment_fn((dx_safe, dy_safe, dz_safe))
        
        if not success:
            logger.warning(f"Failed to apply adjustment at iteration {iteration}")
            return {
                'success': False,
                'iterations': iteration,
                'final_distance': result['distance'],
                'history': history,
                'error': 'adjustment_failed'
            }
        
        # Record history
        history.append({
            'iteration': iteration,
            'distance': result['distance'],
            'adjustment': (dx_safe, dy_safe, dz_safe),
            'adjustment_magnitude': result['adjustment_magnitude']
        })
        
        # Reduce adjustment limit for finer control
        current_limit *= convergence_rate
        
        # Check for convergence
        if len(history) >= 3:
            recent_distances = [h['distance'] for h in history[-3:]]
            if max(recent_distances) - min(recent_distances) < tolerance * 0.1:
                logger.info(f"Converged after {iteration + 1} iterations")
                return {
                    'success': True,
                    'iterations': iteration + 1,
                    'final_distance': result['distance'],
                    'history': history
                }
    
    logger.warning(f"Max iterations reached without convergence")
    return {
        'success': False,
        'iterations': max_iterations,
        'final_distance': result['distance'] if 'result' in locals() else None,
        'history': history,
        'error': 'max_iterations'
    }


class GraspController:
    """
    High-level grasp controller with micro-adjustment capabilities.
    
    Manages the complete grasping pipeline with safety and precision.
    """
    
    def __init__(
        self,
        safety_envelope: Optional[SafetyEnvelope] = None,
        adjustment_limit: float = 0.03,
        alignment_tolerance: float = 0.005,
        approach_height: float = 0.1
    ):
        """
        Initialize grasp controller.
        
        Args:
            safety_envelope: Robot workspace limits
            adjustment_limit: Maximum micro nudge size
            alignment_tolerance: Required alignment precision
            approach_height: Height above object for approach
        """
        self.safety_envelope = safety_envelope or SafetyEnvelope()
        self.adjustment_limit = adjustment_limit
        self.alignment_tolerance = alignment_tolerance
        self.approach_height = approach_height
        
        self.current_phase = GraspPhase.APPROACH
        self.nudge_history = []
        self.last_nudge = None
    
    def compute_approach_position(
        self,
        object_position: Tuple[float, float, float],
        approach_offset: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, float, float]:
        """
        Compute safe approach position above object.
        
        Args:
            object_position: Target object position
            approach_offset: Optional offset from default approach
            
        Returns:
            Safe approach position (x, y, z)
        """
        x, y, z = object_position
        
        if approach_offset:
            dx, dy, dz = approach_offset
        else:
            dx, dy, dz = 0, 0, self.approach_height
        
        # Apply safety clamping
        approach_x = np.clip(x + dx, self.safety_envelope.x_min, self.safety_envelope.x_max)
        approach_y = np.clip(y + dy, self.safety_envelope.y_min, self.safety_envelope.y_max)
        approach_z = np.clip(z + dz, self.safety_envelope.z_min, self.safety_envelope.z_max)
        
        return (approach_x, approach_y, approach_z)
    
    def apply_pre_grasp_adjustment(
        self,
        desired_adjustment: Tuple[float, float, float],
        current_position: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Apply micro nudge for pre-grasp alignment.
        
        Args:
            desired_adjustment: Desired adjustment (dx, dy, dz)
            current_position: Current gripper position
            
        Returns:
            Safe adjustment to apply
        """
        self.current_phase = GraspPhase.PRE_GRASP
        
        # Apply micro nudge with all safety features
        dx, dy, dz = apply_micro_nudge(
            desired_adjustment[0],
            desired_adjustment[1],
            limit=self.adjustment_limit,
            dz_m=desired_adjustment[2] if len(desired_adjustment) > 2 else 0,
            safety_envelope=self.safety_envelope,
            current_position=current_position,
            phase=self.current_phase,
            smoothing=0.3 if self.last_nudge else 0,
            previous_nudge=self.last_nudge
        )
        
        # Update history
        self.last_nudge = (dx, dy, dz)
        self.nudge_history.append({
            'timestamp': np.datetime64('now'),
            'adjustment': (dx, dy, dz),
            'phase': self.current_phase
        })
        
        return (dx, dy, dz)
    
    def validate_grasp_pose(
        self,
        gripper_position: Tuple[float, float, float],
        object_position: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """
        Validate if current pose is suitable for grasping.
        
        Args:
            gripper_position: Current gripper position
            object_position: Target object position
            
        Returns:
            Validation results dictionary
        """
        # Calculate offset
        offset = np.array(object_position) - np.array(gripper_position)
        distance = np.linalg.norm(offset)
        xy_distance = np.linalg.norm(offset[:2])
        
        # Check criteria
        aligned = distance < self.alignment_tolerance
        xy_aligned = xy_distance < self.alignment_tolerance
        within_bounds = self.safety_envelope.is_within_bounds(*gripper_position)
        
        # Compute quality score (0-1)
        distance_score = np.exp(-distance / self.alignment_tolerance)
        safety_margin = min(
            gripper_position[2] - self.safety_envelope.z_min,
            self.safety_envelope.z_max - gripper_position[2]
        )
        safety_score = np.clip(safety_margin / 0.1, 0, 1)
        
        quality = 0.7 * distance_score + 0.3 * safety_score
        
        return {
            'valid': aligned and within_bounds,
            'aligned': aligned,
            'xy_aligned': xy_aligned,
            'within_bounds': within_bounds,
            'distance': distance,
            'xy_distance': xy_distance,
            'quality': quality,
            'offset': offset.tolist()
        }
    
    def reset(self):
        """Reset controller state."""
        self.current_phase = GraspPhase.APPROACH
        self.nudge_history.clear()
        self.last_nudge = None


def safe_velocity_command(
    dx_m: float,
    dy_m: float,
    dz_m: float = 0,
    time_horizon: float = 1.0,
    max_velocity: float = 0.1,
    max_acceleration: float = 0.5
) -> Tuple[float, float, float]:
    """
    Convert position adjustment to safe velocity command.
    
    Args:
        dx_m, dy_m, dz_m: Desired position change in meters
        time_horizon: Time to achieve position change
        max_velocity: Maximum allowed velocity (m/s)
        max_acceleration: Maximum allowed acceleration (m/s²)
        
    Returns:
        (vx, vy, vz): Safe velocity commands in m/s
        
    Example:
        vx, vy, vz = safe_velocity_command(0.05, -0.02, 0, time_horizon=2.0)
    """
    # Compute required velocities
    vx_req = dx_m / time_horizon
    vy_req = dy_m / time_horizon
    vz_req = dz_m / time_horizon
    
    # Compute magnitude
    v_mag = np.sqrt(vx_req**2 + vy_req**2 + vz_req**2)
    
    # Apply velocity limit
    if v_mag > max_velocity:
        scale = max_velocity / v_mag
        vx_req *= scale
        vy_req *= scale
        vz_req *= scale
    
    # Apply acceleration limit (assuming starting from rest)
    max_v_from_acc = max_acceleration * time_horizon
    if v_mag > max_v_from_acc:
        scale = max_v_from_acc / v_mag
        vx_req *= scale
        vy_req *= scale
        vz_req *= scale
    
    return (vx_req, vy_req, vz_req)


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("MICRO NUDGE AND GRASP CONTROL TEST")
    print("="*60)
    
    # Test 1: Basic micro nudge
    print("\n1. Basic Micro Nudge:")
    dx, dy, dz = apply_micro_nudge(0.05, -0.02, limit=0.03)
    print(f"   Requested: (0.050, -0.020, 0.000) m")
    print(f"   Clamped:   ({dx:.3f}, {dy:.3f}, {dz:.3f}) m")
    
    # Test 2: With safety envelope
    print("\n2. With Safety Envelope:")
    safety = SafetyEnvelope(x_max=0.4)
    current_pos = (0.38, 0.0, 0.2)
    dx, dy, dz = apply_micro_nudge(
        0.05, 0, 
        current_position=current_pos,
        safety_envelope=safety
    )
    print(f"   Current position: {current_pos}")
    print(f"   Requested: 0.05m right")
    print(f"   Clamped to boundary: ({dx:.3f}, {dy:.3f}, {dz:.3f}) m")
    
    # Test 3: Grasp controller
    print("\n3. Grasp Controller:")
    controller = GraspController()
    
    # Compute approach position
    object_pos = (0.3, 0.1, 0.05)
    approach_pos = controller.compute_approach_position(object_pos)
    print(f"   Object at: {object_pos}")
    print(f"   Approach: {approach_pos}")
    
    # Apply pre-grasp adjustment
    adjustment = controller.apply_pre_grasp_adjustment(
        desired_adjustment=(0.02, -0.01, 0),
        current_position=approach_pos
    )
    print(f"   Pre-grasp adjustment: {adjustment}")
    
    # Validate grasp pose
    validation = controller.validate_grasp_pose(
        gripper_position=(0.31, 0.095, 0.15),
        object_position=object_pos
    )
    print(f"   Grasp validation:")
    print(f"     Valid: {validation['valid']}")
    print(f"     Distance: {validation['distance']:.3f} m")
    print(f"     Quality: {validation['quality']:.2f}")
    
    # Test 4: Velocity command
    print("\n4. Safe Velocity Command:")
    vx, vy, vz = safe_velocity_command(0.05, -0.02, 0, time_horizon=2.0)
    print(f"   Position change: (0.05, -0.02, 0) m")
    print(f"   Time horizon: 2.0 s")
    print(f"   Velocity command: ({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)