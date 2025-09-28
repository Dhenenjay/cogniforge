"""Core module for CogniForge."""

from .config import settings

# Conditional imports to avoid dependency issues
try:
    from .simulator import RobotSimulator, RobotType, SimulationMode, SimulationConfig
    _simulator_available = True
except ImportError:
    _simulator_available = False
    RobotSimulator = None
    RobotType = None
    SimulationMode = None
    SimulationConfig = None

__all__ = [
    "settings",
    "RobotSimulator",
    "RobotType",
    "SimulationMode",
    "SimulationConfig",
]
