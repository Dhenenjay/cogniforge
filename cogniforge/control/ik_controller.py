"""
Inverse Kinematics controller with joint limit clamping and failure handling.

This module provides robust IK computation with:
- Joint limit enforcement
- Maximum iteration limits
- Fallback to pre-grasp positions on IK failure
- Smooth trajectory generation with safety checks
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# IK configuration constants
IK_MAX_ITERS = 150  # Maximum iterations for IK solver
IK_CONVERGENCE_THRESHOLD = 0.001  # Position error threshold in meters
IK_DAMPING_DEFAULT = 0.01  # Default joint damping for stability
IK_FALLBACK_ITERS = 50  # Reduced iterations for fallback attempt


class IKStatus(Enum):
    """Status of IK computation."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    FALLBACK_USED = "fallback_used"


@dataclass
class JointLimits:
    """Joint limits for a robot."""
    lower: List[float]  # Lower limits in radians
    upper: List[float]  # Upper limits in radians
    velocities: List[float]  # Maximum velocities in rad/s
    efforts: List[float]  # Maximum efforts/torques in Nm
    
    def clamp(self, positions: List[float]) -> List[float]:
        """
        Clamp joint positions to physical limits.
        
        Args:
            positions: Joint positions to clamp
            
        Returns:
            Clamped joint positions
        """
        clamped = []
        for i, pos in enumerate(positions):
            if i < len(self.lower) and i < len(self.upper):
                clamped_pos = np.clip(pos, self.lower[i], self.upper[i])
                if abs(clamped_pos - pos) > 0.01:  # More than 0.01 rad difference
                    logger.warning(
                        f"Joint {i} clamped from {pos:.3f} to {clamped_pos:.3f} rad "
                        f"(limits: [{self.lower[i]:.3f}, {self.upper[i]:.3f}])"
                    )
                clamped.append(clamped_pos)
            else:
                clamped.append(pos)
        return clamped
    
    def is_within_limits(self, positions: List[float]) -> bool:
        """Check if all joints are within limits."""
        for i, pos in enumerate(positions):
            if i < len(self.lower) and i < len(self.upper):
                if not (self.lower[i] <= pos <= self.upper[i]):
                    return False
        return True


# Standard joint limits for common robots
ROBOT_JOINT_LIMITS = {
    "franka_panda": JointLimits(
        lower=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        upper=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        velocities=[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
        efforts=[87, 87, 87, 87, 12, 12, 12]
    ),
    "kuka_iiwa": JointLimits(
        lower=[-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05],
        upper=[2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05],
        velocities=[1.48, 1.48, 1.74, 2.26, 2.35, 2.35, 2.35],
        efforts=[320, 320, 176, 176, 110, 40, 40]
    ),
    "ur5": JointLimits(
        lower=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
        upper=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
        velocities=[3.15, 3.15, 3.15, 3.2, 3.2, 3.2],
        efforts=[150, 150, 150, 28, 28, 28]
    )
}


@dataclass
class PreGraspPose:
    """Pre-grasp configuration for fallback."""
    joint_positions: List[float]
    ee_position: Tuple[float, float, float]
    ee_orientation: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 1.0
    
    def is_valid(self) -> bool:
        """Check if pre-grasp pose is valid."""
        return (
            self.joint_positions is not None and 
            len(self.joint_positions) > 0 and
            all(np.isfinite(pos) for pos in self.joint_positions)
        )


class IKController:
    """
    Robust IK controller with joint limit enforcement and failure handling.
    """
    
    def __init__(
        self,
        robot_type: str = "franka_panda",
        custom_limits: Optional[JointLimits] = None,
        max_iterations: int = IK_MAX_ITERS,
        convergence_threshold: float = IK_CONVERGENCE_THRESHOLD,
        enable_fallback: bool = True
    ):
        """
        Initialize IK controller.
        
        Args:
            robot_type: Type of robot (franka_panda, kuka_iiwa, ur5)
            custom_limits: Custom joint limits (overrides robot_type)
            max_iterations: Maximum IK iterations
            convergence_threshold: Convergence threshold in meters
            enable_fallback: Enable fallback to pre-grasp on failure
        """
        self.robot_type = robot_type
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.enable_fallback = enable_fallback
        
        # Set joint limits
        if custom_limits:
            self.joint_limits = custom_limits
        elif robot_type in ROBOT_JOINT_LIMITS:
            self.joint_limits = ROBOT_JOINT_LIMITS[robot_type]
        else:
            logger.warning(f"Unknown robot type {robot_type}, using default limits")
            # Default conservative limits
            self.joint_limits = JointLimits(
                lower=[-3.14] * 7,
                upper=[3.14] * 7,
                velocities=[2.0] * 7,
                efforts=[100] * 7
            )
        
        # Store pre-grasp configuration for fallback
        self.pre_grasp_pose: Optional[PreGraspPose] = None
        
        # IK statistics
        self.stats = {
            "total_attempts": 0,
            "successes": 0,
            "failures": 0,
            "fallbacks": 0,
            "clamps": 0
        }
        
        logger.info(
            f"IK Controller initialized for {robot_type} with "
            f"max_iters={max_iterations}, threshold={convergence_threshold}m"
        )
    
    def set_pre_grasp_pose(
        self,
        joint_positions: List[float],
        ee_position: Tuple[float, float, float],
        ee_orientation: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Set pre-grasp pose for fallback.
        
        Args:
            joint_positions: Joint configuration at pre-grasp
            ee_position: End-effector position at pre-grasp
            ee_orientation: End-effector orientation at pre-grasp
        """
        # Clamp to joint limits before storing
        clamped_joints = self.joint_limits.clamp(joint_positions)
        
        self.pre_grasp_pose = PreGraspPose(
            joint_positions=clamped_joints,
            ee_position=ee_position,
            ee_orientation=ee_orientation,
            confidence=1.0
        )
        
        logger.info(
            f"Pre-grasp pose set: EE at ({ee_position[0]:.3f}, "
            f"{ee_position[1]:.3f}, {ee_position[2]:.3f})"
        )
    
    def compute_ik_with_limits(
        self,
        target_pos: Tuple[float, float, float],
        target_orn: Optional[Tuple[float, float, float, float]] = None,
        current_joints: Optional[List[float]] = None,
        simulator = None,
        robot_name: str = None,
        use_nullspace: bool = True,
        joint_damping: Optional[List[float]] = None
    ) -> Tuple[List[float], IKStatus]:
        """
        Compute IK with joint limit clamping and failure handling.
        
        Args:
            target_pos: Target end-effector position
            target_orn: Target end-effector orientation (quaternion)
            current_joints: Current joint configuration
            simulator: Robot simulator instance (must have calculate_ik method)
            robot_name: Name of robot in simulator
            use_nullspace: Use null-space optimization
            joint_damping: Joint damping coefficients
            
        Returns:
            Tuple of (joint_positions, status) where:
            - joint_positions: Clamped joint configuration
            - status: IKStatus indicating success/failure/fallback
            
        Raises:
            ValueError: If simulator or robot_name not provided
        """
        if simulator is None or robot_name is None:
            raise ValueError("Simulator and robot_name must be provided")
        
        self.stats["total_attempts"] += 1
        
        # Setup joint damping
        if joint_damping is None:
            joint_damping = [IK_DAMPING_DEFAULT] * len(self.joint_limits.lower)
        
        try:
            # First attempt with full iterations
            logger.debug(
                f"Computing IK for target ({target_pos[0]:.3f}, "
                f"{target_pos[1]:.3f}, {target_pos[2]:.3f}) with "
                f"max_iters={self.max_iterations}"
            )
            
            # Calculate IK using simulator
            if use_nullspace:
                joint_positions = simulator.calculate_ik(
                    robot_name,
                    target_pos=target_pos,
                    target_orn=target_orn,
                    current_q=current_joints,
                    max_iterations=self.max_iterations,
                    residual_threshold=self.convergence_threshold,
                    joint_damping=joint_damping,
                    use_nullspace=True,
                    joint_lower_limits=self.joint_limits.lower,
                    joint_upper_limits=self.joint_limits.upper,
                    joint_ranges=[
                        u - l for l, u in 
                        zip(self.joint_limits.lower, self.joint_limits.upper)
                    ],
                    rest_poses=current_joints if current_joints else [
                        (l + u) / 2 for l, u in 
                        zip(self.joint_limits.lower, self.joint_limits.upper)
                    ]
                )
            else:
                joint_positions = simulator.calculate_ik(
                    robot_name,
                    target_pos=target_pos,
                    target_orn=target_orn,
                    current_q=current_joints,
                    max_iterations=self.max_iterations,
                    residual_threshold=self.convergence_threshold,
                    joint_damping=joint_damping
                )
            
            # Clamp to joint limits
            original_positions = joint_positions.copy()
            clamped_positions = self.joint_limits.clamp(joint_positions)
            
            if original_positions != clamped_positions:
                self.stats["clamps"] += 1
                logger.info("Joint positions clamped to physical limits")
            
            # Verify solution by checking achieved position
            if self._verify_ik_solution(
                clamped_positions, target_pos, simulator, robot_name
            ):
                self.stats["successes"] += 1
                return clamped_positions, IKStatus.SUCCESS
            else:
                logger.warning("IK solution verification failed, attempting fallback")
                
        except Exception as e:
            logger.error(f"IK computation failed: {e}")
        
        # IK failed or didn't converge well
        self.stats["failures"] += 1
        
        # Try fallback to pre-grasp if enabled
        if self.enable_fallback and self.pre_grasp_pose and self.pre_grasp_pose.is_valid():
            logger.info("IK failed, falling back to pre-grasp configuration")
            self.stats["fallbacks"] += 1
            
            # Try simplified IK from pre-grasp position
            try:
                fallback_positions = simulator.calculate_ik(
                    robot_name,
                    target_pos=self.pre_grasp_pose.ee_position,
                    target_orn=self.pre_grasp_pose.ee_orientation,
                    current_q=current_joints,
                    max_iterations=IK_FALLBACK_ITERS,  # Fewer iterations
                    residual_threshold=self.convergence_threshold * 2,  # Relaxed threshold
                    joint_damping=[d * 2 for d in joint_damping]  # More damping
                )
                
                # Clamp fallback solution
                clamped_fallback = self.joint_limits.clamp(fallback_positions)
                
                return clamped_fallback, IKStatus.FALLBACK_USED
                
            except Exception as e:
                logger.error(f"Fallback IK also failed: {e}")
                # Return pre-grasp joint configuration directly
                return self.pre_grasp_pose.joint_positions, IKStatus.FALLBACK_USED
        
        # No fallback available, return clamped current position
        if current_joints:
            logger.warning("IK failed with no fallback, returning clamped current position")
            return self.joint_limits.clamp(current_joints), IKStatus.FAILED
        
        # Last resort: return middle of joint ranges
        logger.error("IK completely failed, returning neutral position")
        neutral = [
            (l + u) / 2 for l, u in 
            zip(self.joint_limits.lower, self.joint_limits.upper)
        ]
        return neutral, IKStatus.FAILED
    
    def _verify_ik_solution(
        self,
        joint_positions: List[float],
        target_pos: Tuple[float, float, float],
        simulator,
        robot_name: str,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Verify IK solution achieves target position.
        
        Args:
            joint_positions: Joint configuration to verify
            target_pos: Target position
            simulator: Simulator instance
            robot_name: Robot name
            tolerance: Position error tolerance (uses convergence_threshold if None)
            
        Returns:
            True if solution is within tolerance
        """
        if tolerance is None:
            tolerance = self.convergence_threshold * 2  # Slightly relaxed for verification
        
        try:
            # Set robot to computed configuration
            simulator.reset_robot(robot_name, q_default=joint_positions, reset_gripper=False)
            
            # Get achieved end-effector position
            achieved_pos, _ = simulator.ee_pose(robot_name)
            
            # Calculate position error
            error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
            
            if error <= tolerance:
                logger.debug(f"IK solution verified: error = {error:.4f}m")
                return True
            else:
                logger.warning(
                    f"IK solution error {error:.4f}m exceeds tolerance {tolerance:.4f}m"
                )
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify IK solution: {e}")
            return False
    
    def generate_safe_trajectory(
        self,
        waypoints: List[Tuple[float, float, float]],
        current_joints: List[float],
        simulator,
        robot_name: str,
        use_orientation: bool = False,
        orientations: Optional[List[Tuple[float, float, float, float]]] = None
    ) -> Tuple[List[List[float]], List[IKStatus]]:
        """
        Generate safe trajectory through waypoints with IK and limit checking.
        
        Args:
            waypoints: List of target positions
            current_joints: Starting joint configuration
            simulator: Simulator instance
            robot_name: Robot name
            use_orientation: Whether to use orientation constraints
            orientations: List of target orientations (if use_orientation=True)
            
        Returns:
            Tuple of (joint_trajectory, status_list) where:
            - joint_trajectory: List of joint configurations
            - status_list: IK status for each waypoint
        """
        joint_trajectory = [current_joints]
        status_list = []
        
        for i, target_pos in enumerate(waypoints):
            target_orn = None
            if use_orientation and orientations and i < len(orientations):
                target_orn = orientations[i]
            
            # Compute IK from previous configuration
            joints, status = self.compute_ik_with_limits(
                target_pos=target_pos,
                target_orn=target_orn,
                current_joints=joint_trajectory[-1],
                simulator=simulator,
                robot_name=robot_name
            )
            
            joint_trajectory.append(joints)
            status_list.append(status)
            
            # Log waypoint status
            logger.info(
                f"Waypoint {i+1}/{len(waypoints)}: {status.value} "
                f"at ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
            )
            
            # Stop if complete failure
            if status == IKStatus.FAILED:
                logger.error(f"Trajectory generation stopped at waypoint {i+1}")
                break
        
        return joint_trajectory[1:], status_list  # Exclude initial configuration
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get IK computation statistics.
        
        Returns:
            Dictionary with statistics
        """
        total = self.stats["total_attempts"]
        if total > 0:
            success_rate = self.stats["successes"] / total * 100
            failure_rate = self.stats["failures"] / total * 100
            fallback_rate = self.stats["fallbacks"] / total * 100
        else:
            success_rate = failure_rate = fallback_rate = 0
        
        return {
            "total_attempts": total,
            "successes": self.stats["successes"],
            "failures": self.stats["failures"],
            "fallbacks": self.stats["fallbacks"],
            "clamps": self.stats["clamps"],
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "fallback_rate": fallback_rate
        }
    
    def reset_statistics(self):
        """Reset IK statistics."""
        self.stats = {
            "total_attempts": 0,
            "successes": 0,
            "failures": 0,
            "fallbacks": 0,
            "clamps": 0
        }
        logger.info("IK statistics reset")


def create_ik_controller(
    robot_type: str = "franka_panda",
    enable_fallback: bool = True
) -> IKController:
    """
    Factory function to create IK controller with default settings.
    
    Args:
        robot_type: Type of robot
        enable_fallback: Enable pre-grasp fallback
        
    Returns:
        Configured IK controller
    """
    return IKController(
        robot_type=robot_type,
        max_iterations=IK_MAX_ITERS,
        convergence_threshold=IK_CONVERGENCE_THRESHOLD,
        enable_fallback=enable_fallback
    )