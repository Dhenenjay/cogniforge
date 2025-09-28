"""
Base Skill Class

Defines the common interface for all manipulation skills.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """Status of skill execution"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class SkillResult:
    """Result of skill execution"""
    status: SkillStatus
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    waypoints_executed: int = 0
    

@dataclass
class SkillParameters:
    """Base parameters for skills"""
    object_id: Optional[int] = None
    target_position: Optional[np.ndarray] = None
    speed: float = 0.01
    max_force: float = 100.0
    timeout: float = 30.0
    precision: float = 0.01  # Position precision in meters
    

class BaseSkill(ABC):
    """
    Abstract base class for manipulation skills
    
    All skills must implement:
    - validate_preconditions: Check if skill can be executed
    - plan: Generate plan for skill execution
    - execute: Execute the planned skill
    """
    
    def __init__(self, robot_id: int, scene_objects: Dict[str, int]):
        """
        Initialize base skill
        
        Args:
            robot_id: PyBullet robot ID
            scene_objects: Dictionary of scene object IDs
        """
        self.robot_id = robot_id
        self.scene_objects = scene_objects
        self.current_plan = []
        self.status = SkillStatus.NOT_STARTED
        
    @abstractmethod
    def validate_preconditions(self, params: SkillParameters) -> Tuple[bool, str]:
        """
        Validate that skill preconditions are met
        
        Args:
            params: Skill parameters
            
        Returns:
            (is_valid, message) tuple
        """
        pass
        
    @abstractmethod
    def plan(self, params: SkillParameters) -> List[Dict[str, Any]]:
        """
        Generate plan for skill execution
        
        Args:
            params: Skill parameters
            
        Returns:
            List of waypoints/actions
        """
        pass
        
    @abstractmethod
    def execute(self, params: SkillParameters) -> SkillResult:
        """
        Execute the skill
        
        Args:
            params: Skill parameters
            
        Returns:
            Skill execution result
        """
        pass
        
    def reset(self):
        """Reset skill state"""
        self.current_plan = []
        self.status = SkillStatus.NOT_STARTED
        
    def get_end_effector_position(self) -> np.ndarray:
        """Get current end-effector position"""
        import pybullet as p
        ee_state = p.getLinkState(self.robot_id, 7)  # Assuming link 7 is end-effector
        return np.array(ee_state[0])
        
    def get_object_position(self, object_id: int) -> np.ndarray:
        """Get object position"""
        import pybullet as p
        pos, _ = p.getBasePositionAndOrientation(object_id)
        return np.array(pos)
        
    def move_to_position(self, target_pos: np.ndarray, 
                        speed: float = 0.01,
                        timeout: float = 10.0) -> bool:
        """
        Move end-effector to target position
        
        Args:
            target_pos: Target 3D position
            speed: Movement speed
            timeout: Maximum time to reach position
            
        Returns:
            Success status
        """
        import pybullet as p
        import time
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            current_pos = self.get_end_effector_position()
            error = np.linalg.norm(target_pos - current_pos)
            
            if error < 0.01:  # Within 1cm
                return True
                
            # Simple proportional control (replace with proper IK)
            direction = (target_pos - current_pos) / (error + 1e-6)
            velocity = direction * min(speed, error * 5)
            
            # Apply joint velocities (simplified)
            for i in range(min(3, 7)):  # First 3 joints for position
                p.setJointMotorControl2(
                    self.robot_id, i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=velocity[i] if i < 3 else 0
                )
                
            p.stepSimulation()
            time.sleep(1/240)
            
        return False
        
    def control_gripper(self, state: float):
        """
        Control gripper
        
        Args:
            state: 0 for closed, 1 for open
        """
        import pybullet as p
        
        # Assuming joints 9 and 10 are gripper fingers
        gripper_range = 0.04
        target = state * gripper_range
        
        p.setJointMotorControl2(
            self.robot_id, 9,
            p.POSITION_CONTROL,
            targetPosition=target
        )
        p.setJointMotorControl2(
            self.robot_id, 10,
            p.POSITION_CONTROL,
            targetPosition=target
        )
        
    def compute_approach_position(self, object_pos: np.ndarray,
                                 approach_height: float = 0.1) -> np.ndarray:
        """
        Compute approach position above object
        
        Args:
            object_pos: Object position
            approach_height: Height above object
            
        Returns:
            Approach position
        """
        approach_pos = object_pos.copy()
        approach_pos[2] += approach_height
        return approach_pos
        
    def wait_for_stability(self, object_id: int, 
                          threshold: float = 0.001,
                          max_wait: float = 2.0) -> bool:
        """
        Wait for object to stabilize
        
        Args:
            object_id: Object to monitor
            threshold: Velocity threshold for stability
            max_wait: Maximum wait time
            
        Returns:
            Whether object stabilized
        """
        import pybullet as p
        import time
        
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            linear_vel, angular_vel = p.getBaseVelocity(object_id)
            
            if (np.linalg.norm(linear_vel) < threshold and 
                np.linalg.norm(angular_vel) < threshold):
                return True
                
            p.stepSimulation()
            time.sleep(1/240)
            
        return False