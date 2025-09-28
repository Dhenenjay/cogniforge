"""Cogniforge Control Module - Robot control and manipulation."""

# Import from robot_control what's actually defined there
from .robot_control import apply_micro_nudge, apply_micro_nudge_simple, GraspPhase, SafetyEnvelope
from .ik_controller import IKController, IKStatus, create_ik_controller
from .safe_grasp_execution import SafeGraspExecutor

# Create a placeholder RobotController for backwards compatibility
class RobotController:
    """Placeholder for RobotController - use IKController instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RobotController is deprecated. Please use IKController or the functions in robot_control module."
        )

__all__ = [
    'RobotController',  # For backwards compatibility
    'IKController',
    'IKStatus',
    'create_ik_controller',
    'SafeGraspExecutor',
    'apply_micro_nudge',
    'apply_micro_nudge_simple',
    'GraspPhase',
    'SafetyEnvelope',
]
