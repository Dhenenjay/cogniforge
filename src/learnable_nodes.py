"""
Learnable Behavior Tree Nodes with BC/Optimization

This module implements learnable parameters for Align and Grasp nodes,
while keeping MoveTo and Place as scripted behaviors. The learning system
targets only the critical manipulation parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import json
import time
from enum import Enum

from behavior_tree import (
    BehaviorNode, 
    NodeStatus, 
    Blackboard,
    MoveTo,
    Place,
    Align as BaseAlign,
    Grasp as BaseGrasp
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Learnable Parameter Classes
# ============================================================================

@dataclass
class AlignParameters:
    """
    Learnable parameters for the Align node
    """
    # Approach angles for different object types
    approach_angles: Dict[str, np.ndarray] = field(default_factory=lambda: {
        'default': np.array([0.0, 0.0, -np.pi/2]),
        'cube': np.array([0.0, 0.0, -np.pi/2]),
        'cylinder': np.array([0.0, 0.0, 0.0]),
        'sphere': np.array([0.0, 0.0, -np.pi/2])
    })
    
    # Alignment speed profiles
    rotation_speed: float = 0.5  # rad/s
    interpolation_steps: int = 10
    
    # Adaptive alignment offsets based on object size
    size_offset_factor: float = 0.1
    
    # Tolerance adjustments
    alignment_tolerance: float = 0.01  # radians
    
    # Learned corrections from BC
    learned_corrections: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Optimization bounds
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'rotation_speed': (0.1, 2.0),
        'alignment_tolerance': (0.001, 0.1),
        'size_offset_factor': (0.0, 0.5)
    })
    
    def to_vector(self) -> np.ndarray:
        """Convert parameters to optimization vector"""
        vector = []
        
        # Flatten approach angles
        for key in sorted(self.approach_angles.keys()):
            vector.extend(self.approach_angles[key].tolist())
        
        # Add scalar parameters
        vector.append(self.rotation_speed)
        vector.append(float(self.interpolation_steps))
        vector.append(self.size_offset_factor)
        vector.append(self.alignment_tolerance)
        
        # Add learned corrections
        vector.extend(self.learned_corrections.tolist())
        
        return np.array(vector)
    
    def from_vector(self, vector: np.ndarray) -> None:
        """Update parameters from optimization vector"""
        idx = 0
        
        # Update approach angles
        for key in sorted(self.approach_angles.keys()):
            self.approach_angles[key] = vector[idx:idx+3]
            idx += 3
        
        # Update scalar parameters
        self.rotation_speed = np.clip(vector[idx], *self.bounds['rotation_speed'])
        idx += 1
        
        self.interpolation_steps = int(np.clip(vector[idx], 5, 20))
        idx += 1
        
        self.size_offset_factor = np.clip(vector[idx], *self.bounds['size_offset_factor'])
        idx += 1
        
        self.alignment_tolerance = np.clip(vector[idx], *self.bounds['alignment_tolerance'])
        idx += 1
        
        # Update learned corrections
        self.learned_corrections = vector[idx:idx+3]


@dataclass
class GraspParameters:
    """
    Learnable parameters for the Grasp node
    """
    # Pre-grasp offsets for different object types
    pre_grasp_offsets: Dict[str, np.ndarray] = field(default_factory=lambda: {
        'default': np.array([0.0, 0.0, 0.05]),
        'cube': np.array([0.0, 0.0, 0.05]),
        'cylinder': np.array([0.0, 0.0, 0.08]),
        'sphere': np.array([0.0, 0.0, 0.06])
    })
    
    # Approach parameters
    approach_speed: float = 0.05  # m/s
    approach_steps: int = 10
    
    # Grasp force profile
    min_grasp_force: float = 2.0  # N
    max_grasp_force: float = 10.0  # N
    force_ramp_time: float = 0.5  # seconds
    
    # Gripper timing
    gripper_close_time: float = 0.5  # seconds
    gripper_settle_time: float = 0.2  # seconds
    
    # Lift parameters after grasp
    post_grasp_lift: float = 0.02  # meters
    lift_speed: float = 0.03  # m/s
    
    # Contact detection threshold
    contact_force_threshold: float = 0.5  # N
    
    # Retry strategy parameters
    retry_offset_adjustments: np.ndarray = field(default_factory=lambda: 
        np.array([[0.0, 0.0, 0.0],      # No adjustment
                  [0.01, 0.0, 0.0],      # Slight x offset
                  [-0.01, 0.0, 0.0],     # Opposite x offset
                  [0.0, 0.01, 0.0]]))    # Slight y offset
    
    # Learned corrections from BC
    learned_force_profile: np.ndarray = field(default_factory=lambda: np.ones(5))
    learned_position_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Optimization bounds
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'approach_speed': (0.01, 0.2),
        'min_grasp_force': (1.0, 5.0),
        'max_grasp_force': (5.0, 20.0),
        'post_grasp_lift': (0.01, 0.1),
        'contact_force_threshold': (0.1, 2.0)
    })
    
    def to_vector(self) -> np.ndarray:
        """Convert parameters to optimization vector"""
        vector = []
        
        # Flatten pre-grasp offsets
        for key in sorted(self.pre_grasp_offsets.keys()):
            vector.extend(self.pre_grasp_offsets[key].tolist())
        
        # Add scalar parameters
        vector.extend([
            self.approach_speed,
            float(self.approach_steps),
            self.min_grasp_force,
            self.max_grasp_force,
            self.force_ramp_time,
            self.gripper_close_time,
            self.gripper_settle_time,
            self.post_grasp_lift,
            self.lift_speed,
            self.contact_force_threshold
        ])
        
        # Add retry adjustments (flatten)
        vector.extend(self.retry_offset_adjustments.flatten().tolist())
        
        # Add learned components
        vector.extend(self.learned_force_profile.tolist())
        vector.extend(self.learned_position_offset.tolist())
        
        return np.array(vector)
    
    def from_vector(self, vector: np.ndarray) -> None:
        """Update parameters from optimization vector"""
        idx = 0
        
        # Update pre-grasp offsets
        for key in sorted(self.pre_grasp_offsets.keys()):
            self.pre_grasp_offsets[key] = vector[idx:idx+3]
            idx += 3
        
        # Update scalar parameters
        self.approach_speed = np.clip(vector[idx], *self.bounds['approach_speed'])
        idx += 1
        
        self.approach_steps = int(np.clip(vector[idx], 5, 30))
        idx += 1
        
        self.min_grasp_force = np.clip(vector[idx], *self.bounds['min_grasp_force'])
        idx += 1
        
        self.max_grasp_force = np.clip(vector[idx], *self.bounds['max_grasp_force'])
        idx += 1
        
        self.force_ramp_time = np.clip(vector[idx], 0.1, 2.0)
        idx += 1
        
        self.gripper_close_time = np.clip(vector[idx], 0.1, 2.0)
        idx += 1
        
        self.gripper_settle_time = np.clip(vector[idx], 0.05, 1.0)
        idx += 1
        
        self.post_grasp_lift = np.clip(vector[idx], *self.bounds['post_grasp_lift'])
        idx += 1
        
        self.lift_speed = np.clip(vector[idx], 0.01, 0.1)
        idx += 1
        
        self.contact_force_threshold = np.clip(vector[idx], *self.bounds['contact_force_threshold'])
        idx += 1
        
        # Update retry adjustments
        retry_size = 4 * 3  # 4 strategies × 3 dimensions
        self.retry_offset_adjustments = vector[idx:idx+retry_size].reshape(4, 3)
        idx += retry_size
        
        # Update learned components
        self.learned_force_profile = vector[idx:idx+5]
        idx += 5
        
        self.learned_position_offset = vector[idx:idx+3]


# ============================================================================
# Neural Network Models for BC
# ============================================================================

class AlignmentNetwork(nn.Module):
    """
    Neural network for learning alignment parameters from demonstrations
    """
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, output_dim: int = 6):
        """
        Args:
            input_dim: Object features + current state
            hidden_dim: Hidden layer size
            output_dim: Alignment corrections (3 angles + 3 offsets)
        """
        super(AlignmentNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Alignment parameters [batch_size, output_dim]
        """
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        # Output alignment corrections
        output = self.fc4(x)
        
        # Split into angles and offsets
        angles = torch.tanh(output[:, :3]) * np.pi  # [-π, π]
        offsets = torch.tanh(output[:, 3:]) * 0.1   # [-0.1, 0.1] meters
        
        return torch.cat([angles, offsets], dim=1)


class GraspNetwork(nn.Module):
    """
    Neural network for learning grasp parameters from demonstrations
    """
    
    def __init__(self, input_dim: int = 15, hidden_dim: int = 256, output_dim: int = 10):
        """
        Args:
            input_dim: Object features + current state + tactile feedback
            hidden_dim: Hidden layer size
            output_dim: Grasp parameters (position offset, force profile, timing)
        """
        super(GraspNetwork, self).__init__()
        
        # Encoder for object features
        self.object_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Encoder for robot state
        self.state_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Encoder for tactile features (if available)
        self.tactile_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Combined processing
        combined_dim = 32 + 32 + 16  # object + state + tactile
        
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, object_features: torch.Tensor, 
                robot_state: torch.Tensor,
                tactile_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            object_features: Object characteristics [batch_size, 6]
            robot_state: Current robot state [batch_size, 6]
            tactile_features: Tactile sensor data [batch_size, 3]
            
        Returns:
            Grasp parameters [batch_size, output_dim]
        """
        # Encode different feature types
        obj_encoded = self.object_encoder(object_features)
        state_encoded = self.state_encoder(robot_state)
        tactile_encoded = self.tactile_encoder(tactile_features)
        
        # Combine features
        x = torch.cat([obj_encoded, state_encoded, tactile_encoded], dim=1)
        
        # Process through main network
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        # Output grasp parameters
        output = self.fc4(x)
        
        # Split and scale outputs appropriately
        position_offset = torch.tanh(output[:, :3]) * 0.05  # [-5cm, 5cm]
        force_profile = torch.sigmoid(output[:, 3:8]) * 15.0 + 1.0  # [1N, 16N]
        timing = torch.sigmoid(output[:, 8:]) * 2.0  # [0s, 2s]
        
        return torch.cat([position_offset, force_profile, timing], dim=1)


# ============================================================================
# Learnable Behavior Nodes
# ============================================================================

class LearnableAlign(BaseAlign):
    """
    Learnable version of the Align node with BC/optimization support
    """
    
    def __init__(self, blackboard: Blackboard, 
                 params: Optional[AlignParameters] = None,
                 network: Optional[AlignmentNetwork] = None):
        """
        Initialize learnable align node
        
        Args:
            blackboard: Shared blackboard
            params: Alignment parameters (creates default if None)
            network: Neural network for BC (optional)
        """
        super().__init__(blackboard)
        self.params = params or AlignParameters()
        self.network = network
        self.name = "LearnableAlign"
        
    def run(self) -> NodeStatus:
        """Execute alignment with learnable parameters"""
        # Get grasp target
        grasp_target = self.blackboard.get('grasp_target')
        if grasp_target is None:
            logger.error("No grasp target for alignment")
            return NodeStatus.FAILURE
        
        # Get object type for parameter selection
        object_type = self._detect_object_type()
        
        # Apply neural network corrections if available
        if self.network is not None:
            corrections = self._get_network_corrections()
            target_orientation = self._apply_corrections(
                self.params.approach_angles.get(object_type, self.params.approach_angles['default']),
                corrections
            )
        else:
            # Use learned parameters
            target_orientation = self.params.approach_angles.get(
                object_type, 
                self.params.approach_angles['default']
            ) + self.params.learned_corrections
        
        # Get current orientation
        current_orientation = self.blackboard.get('robot_orientation')
        
        # Calculate alignment error
        orientation_error = target_orientation - current_orientation
        orientation_error = np.array([self._normalize_angle(a) for a in orientation_error])
        
        # Check if already aligned
        if np.max(np.abs(orientation_error)) < self.params.alignment_tolerance:
            logger.info("Already aligned with target")
            return NodeStatus.SUCCESS
        
        # Perform alignment with learned parameters
        steps = self.params.interpolation_steps
        for i in range(steps + 1):
            t = i / steps
            # Use learned speed profile
            speed_factor = self._compute_speed_profile(t)
            
            # Interpolate orientation
            new_orientation = current_orientation + orientation_error * t * speed_factor
            
            # Update blackboard
            self.blackboard.set('robot_orientation', new_orientation)
            
            # Delay based on rotation speed
            time.sleep(0.1 / self.params.rotation_speed)
        
        # Verify alignment
        final_orientation = self.blackboard.get('robot_orientation')
        final_error = target_orientation - final_orientation
        final_error = np.array([self._normalize_angle(a) for a in final_error])
        
        if np.max(np.abs(final_error)) < self.params.alignment_tolerance:
            logger.info(f"Successfully aligned (max error: {np.max(np.abs(final_error)):.4f} rad)")
            
            # Record successful alignment for learning
            self._record_success(target_orientation, object_type)
            return NodeStatus.SUCCESS
        else:
            logger.error(f"Failed to align (max error: {np.max(np.abs(final_error)):.4f} rad)")
            
            # Record failure for learning
            self._record_failure(target_orientation, object_type, final_error)
            return NodeStatus.FAILURE
    
    def _detect_object_type(self) -> str:
        """Detect object type from scene information"""
        detected_objects = self.blackboard.get('detected_objects', {})
        
        # Simple heuristic - can be replaced with vision
        if 'cube' in str(detected_objects).lower():
            return 'cube'
        elif 'cylinder' in str(detected_objects).lower():
            return 'cylinder'
        elif 'sphere' in str(detected_objects).lower():
            return 'sphere'
        
        return 'default'
    
    def _get_network_corrections(self) -> np.ndarray:
        """Get corrections from neural network"""
        if self.network is None:
            return np.zeros(6)
        
        # Prepare input features
        features = self._extract_features()
        
        # Run network inference
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            corrections = self.network(input_tensor).numpy().squeeze()
        
        return corrections
    
    def _apply_corrections(self, base_angles: np.ndarray, corrections: np.ndarray) -> np.ndarray:
        """Apply network corrections to base angles"""
        # First 3 values are angle corrections
        angle_corrections = corrections[:3] if len(corrections) >= 3 else np.zeros(3)
        return base_angles + angle_corrections
    
    def _compute_speed_profile(self, t: float) -> float:
        """Compute speed profile for smooth alignment"""
        # S-curve profile for smooth acceleration/deceleration
        if t < 0.2:
            return 0.5 * (t / 0.2) ** 2
        elif t > 0.8:
            return 0.5 * ((1 - t) / 0.2) ** 2 + 0.5
        else:
            return 1.0
    
    def _extract_features(self) -> np.ndarray:
        """Extract features for neural network input"""
        # Get relevant state information
        robot_pos = self.blackboard.get('robot_position')
        robot_orient = self.blackboard.get('robot_orientation')
        grasp_target = self.blackboard.get('grasp_target')
        
        if isinstance(grasp_target, dict):
            target_pos = grasp_target.get('position', [0, 0, 0])
        else:
            target_pos = grasp_target[:3] if len(grasp_target) >= 3 else [0, 0, 0]
        
        # Combine features
        features = np.concatenate([
            robot_pos,
            robot_orient,
            target_pos,
            [0, 0, 0]  # Padding or additional features
        ])
        
        return features
    
    def _record_success(self, target_orientation: np.ndarray, object_type: str) -> None:
        """Record successful alignment for learning"""
        history = self.blackboard.get('execution_history')
        history.append({
            'node': 'LearnableAlign',
            'result': 'success',
            'target_orientation': target_orientation.tolist(),
            'object_type': object_type,
            'parameters': self.params.to_vector().tolist()
        })
    
    def _record_failure(self, target_orientation: np.ndarray, 
                       object_type: str, error: np.ndarray) -> None:
        """Record failed alignment for learning"""
        history = self.blackboard.get('execution_history')
        history.append({
            'node': 'LearnableAlign',
            'result': 'failure',
            'target_orientation': target_orientation.tolist(),
            'object_type': object_type,
            'error': error.tolist(),
            'parameters': self.params.to_vector().tolist()
        })


class LearnableGrasp(BaseGrasp):
    """
    Learnable version of the Grasp node with BC/optimization support
    """
    
    def __init__(self, blackboard: Blackboard,
                 params: Optional[GraspParameters] = None,
                 network: Optional[GraspNetwork] = None):
        """
        Initialize learnable grasp node
        
        Args:
            blackboard: Shared blackboard
            params: Grasp parameters (creates default if None)
            network: Neural network for BC (optional)
        """
        super().__init__(blackboard)
        self.params = params or GraspParameters()
        self.network = network
        self.name = "LearnableGrasp"
        self.attempt_count = 0
        
    def run(self) -> NodeStatus:
        """Execute grasping with learnable parameters"""
        # Reset attempt count
        self.attempt_count = 0
        
        # Check current gripper state
        gripper_state = self.blackboard.get('gripper_state')
        
        # Ensure gripper is open before grasping
        if gripper_state != 'open':
            logger.info("Opening gripper before grasp")
            self._control_gripper('open')
        
        # Get grasp target
        grasp_target = self.blackboard.get('grasp_target')
        if grasp_target is None:
            logger.error("No grasp target specified")
            return NodeStatus.FAILURE
        
        # Get object type and size
        object_type = self._detect_object_type()
        object_size = self._estimate_object_size()
        
        # Get neural network adjustments if available
        if self.network is not None:
            adjustments = self._get_network_adjustments()
        else:
            adjustments = self.params.learned_position_offset
        
        # Extract and adjust target position
        if isinstance(grasp_target, dict):
            base_target_pos = np.array(grasp_target.get('position', [0, 0, 0]))
        else:
            base_target_pos = np.array(grasp_target[:3])
        
        target_pos = base_target_pos + adjustments
        
        # Try grasping with retry adjustments
        for attempt in range(len(self.params.retry_offset_adjustments)):
            self.attempt_count = attempt
            
            # Apply retry offset
            retry_offset = self.params.retry_offset_adjustments[attempt]
            adjusted_target = target_pos + retry_offset
            
            # Move to pre-grasp position with learned offset
            pre_grasp_offset = self.params.pre_grasp_offsets.get(
                object_type, 
                self.params.pre_grasp_offsets['default']
            )
            pre_grasp_pos = adjusted_target + pre_grasp_offset
            
            logger.info(f"Grasp attempt {attempt + 1}: Moving to pre-grasp position")
            
            # Update blackboard for MoveTo node
            self.blackboard.set('current_target', pre_grasp_pos)
            move_to_pre = MoveTo(self.blackboard, 'current_target')
            
            if move_to_pre() != NodeStatus.SUCCESS:
                logger.warning(f"Failed to reach pre-grasp position (attempt {attempt + 1})")
                continue
            
            # Approach with learned speed
            logger.info("Approaching grasp target")
            approach_success = self._learned_approach(adjusted_target)
            
            if not approach_success:
                logger.warning(f"Failed to approach target (attempt {attempt + 1})")
                continue
            
            # Execute grasp with learned force profile
            logger.info("Executing grasp")
            grasp_success = self._learned_grasp_execution(object_size)
            
            if grasp_success:
                # Lift with learned parameters
                lift_success = self._learned_lift()
                
                if lift_success:
                    logger.info(f"Successfully grasped object (attempt {attempt + 1})")
                    
                    # Record success for learning
                    self._record_success(adjusted_target, object_type, attempt)
                    return NodeStatus.SUCCESS
            
            # Open gripper for retry
            if attempt < len(self.params.retry_offset_adjustments) - 1:
                self._control_gripper('open')
                time.sleep(0.3)
        
        # All attempts failed
        logger.error("Failed to grasp object after all attempts")
        self._record_failure(target_pos, object_type)
        return NodeStatus.FAILURE
    
    def _detect_object_type(self) -> str:
        """Detect object type from scene information"""
        detected_objects = self.blackboard.get('detected_objects', {})
        
        if 'cube' in str(detected_objects).lower():
            return 'cube'
        elif 'cylinder' in str(detected_objects).lower():
            return 'cylinder'
        elif 'sphere' in str(detected_objects).lower():
            return 'sphere'
        
        return 'default'
    
    def _estimate_object_size(self) -> float:
        """Estimate object size from vision or default"""
        detected_objects = self.blackboard.get('detected_objects', {})
        
        # Simple heuristic - can be replaced with actual vision
        for obj_id, obj_data in detected_objects.items():
            if isinstance(obj_data, dict) and 'size' in obj_data:
                return obj_data['size']
        
        return 0.05  # Default 5cm
    
    def _get_network_adjustments(self) -> np.ndarray:
        """Get adjustments from neural network"""
        if self.network is None:
            return np.zeros(3)
        
        # Prepare input features
        object_features, robot_state, tactile_features = self._extract_features()
        
        # Run network inference
        with torch.no_grad():
            obj_tensor = torch.FloatTensor(object_features).unsqueeze(0)
            state_tensor = torch.FloatTensor(robot_state).unsqueeze(0)
            tactile_tensor = torch.FloatTensor(tactile_features).unsqueeze(0)
            
            output = self.network(obj_tensor, state_tensor, tactile_tensor)
            adjustments = output.numpy().squeeze()
        
        # Return position adjustments (first 3 values)
        return adjustments[:3]
    
    def _learned_approach(self, target_pos: np.ndarray) -> bool:
        """Approach target with learned parameters"""
        current_pos = self.blackboard.get('robot_position')
        distance = np.linalg.norm(target_pos - current_pos)
        
        # Use learned approach steps
        steps = self.params.approach_steps
        
        for i in range(steps + 1):
            t = i / steps
            
            # Apply learned speed profile
            speed_factor = self._compute_approach_speed(t, distance)
            
            intermediate_pos = current_pos * (1 - t) + target_pos * t
            
            # Check for contact with learned threshold
            if self._check_learned_contact():
                logger.info("Contact detected during approach")
                return True
            
            self.blackboard.set('robot_position', intermediate_pos)
            time.sleep(0.05 / self.params.approach_speed)
        
        return True
    
    def _learned_grasp_execution(self, object_size: float) -> bool:
        """Execute grasp with learned force profile"""
        # Close gripper
        self._control_gripper('close')
        
        # Wait with learned timing
        time.sleep(self.params.gripper_close_time)
        
        # Apply learned force profile
        force_profile = self.params.learned_force_profile
        target_force = np.interp(
            object_size,
            [0.01, 0.05, 0.10, 0.15, 0.20],  # Size ranges
            force_profile  # Corresponding forces
        )
        
        # Check force with learned thresholds
        measured_force = self._measure_gripper_force()
        self.blackboard.set('gripper_force', measured_force)
        
        if self.params.min_grasp_force <= measured_force <= target_force * 1.2:
            return True
        
        return False
    
    def _learned_lift(self) -> bool:
        """Lift object with learned parameters"""
        current_pos = self.blackboard.get('robot_position')
        lift_pos = current_pos + np.array([0, 0, self.params.post_grasp_lift])
        
        # Move up slowly
        steps = int(self.params.post_grasp_lift / 0.005)  # 5mm steps
        
        for i in range(steps + 1):
            t = i / steps
            intermediate_pos = current_pos * (1 - t) + lift_pos * t
            
            self.blackboard.set('robot_position', intermediate_pos)
            time.sleep(0.05 / self.params.lift_speed)
            
            # Check if object is still grasped
            if self.blackboard.get('gripper_force', 0) < self.params.min_grasp_force * 0.8:
                return False
        
        return True
    
    def _check_learned_contact(self) -> bool:
        """Check for contact using learned threshold"""
        # Simulate force sensor reading
        force = self._measure_contact_force()
        return force > self.params.contact_force_threshold
    
    def _measure_contact_force(self) -> float:
        """Measure contact force (simulated)"""
        # In real implementation, read from F/T sensor
        return 0.0
    
    def _compute_approach_speed(self, t: float, distance: float) -> float:
        """Compute approach speed based on progress and distance"""
        # Slow down as approaching target
        base_speed = self.params.approach_speed
        
        if t > 0.7:  # Last 30% of approach
            return base_speed * 0.3
        elif t > 0.5:  # Middle section
            return base_speed * 0.7
        else:  # Initial approach
            return base_speed
    
    def _extract_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features for neural network input"""
        # Object features
        grasp_target = self.blackboard.get('grasp_target')
        if isinstance(grasp_target, dict):
            target_pos = grasp_target.get('position', [0, 0, 0])
            target_size = grasp_target.get('size', [0.05, 0.05, 0.05])
        else:
            target_pos = grasp_target[:3] if len(grasp_target) >= 3 else [0, 0, 0]
            target_size = [0.05, 0.05, 0.05]
        
        object_features = np.concatenate([target_pos, target_size])
        
        # Robot state
        robot_pos = self.blackboard.get('robot_position')
        robot_orient = self.blackboard.get('robot_orientation')
        robot_state = np.concatenate([robot_pos, robot_orient])
        
        # Tactile features (simulated)
        gripper_force = self.blackboard.get('gripper_force', 0)
        tactile_features = np.array([gripper_force, 0, 0])  # Can add more sensors
        
        return object_features, robot_state, tactile_features
    
    def _record_success(self, target_pos: np.ndarray, object_type: str, attempt: int) -> None:
        """Record successful grasp for learning"""
        history = self.blackboard.get('execution_history')
        history.append({
            'node': 'LearnableGrasp',
            'result': 'success',
            'target_position': target_pos.tolist(),
            'object_type': object_type,
            'attempt_number': attempt + 1,
            'parameters': self.params.to_vector().tolist()
        })
    
    def _record_failure(self, target_pos: np.ndarray, object_type: str) -> None:
        """Record failed grasp for learning"""
        history = self.blackboard.get('execution_history')
        history.append({
            'node': 'LearnableGrasp', 
            'result': 'failure',
            'target_position': target_pos.tolist(),
            'object_type': object_type,
            'total_attempts': self.attempt_count + 1,
            'parameters': self.params.to_vector().tolist()
        })


# ============================================================================
# Optimization Interface
# ============================================================================

class ParameterOptimizer:
    """
    Optimizer for learning Align and Grasp parameters using CMA-ES or other methods
    """
    
    def __init__(self, align_params: AlignParameters, grasp_params: GraspParameters):
        """
        Initialize optimizer
        
        Args:
            align_params: Alignment parameters to optimize
            grasp_params: Grasp parameters to optimize
        """
        self.align_params = align_params
        self.grasp_params = grasp_params
        
        # Combine parameters for optimization
        self.param_vector = self._combine_parameters()
        self.param_dim = len(self.param_vector)
        
        # Track optimization history
        self.history = []
        self.best_params = self.param_vector.copy()
        self.best_fitness = float('inf')
        
    def _combine_parameters(self) -> np.ndarray:
        """Combine align and grasp parameters into single vector"""
        align_vector = self.align_params.to_vector()
        grasp_vector = self.grasp_params.to_vector()
        
        return np.concatenate([align_vector, grasp_vector])
    
    def _split_parameters(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split combined vector back into align and grasp parameters"""
        align_dim = len(self.align_params.to_vector())
        
        align_vector = vector[:align_dim]
        grasp_vector = vector[align_dim:]
        
        return align_vector, grasp_vector
    
    def evaluate_fitness(self, param_vector: np.ndarray, 
                         test_scenarios: List[Dict]) -> float:
        """
        Evaluate fitness of parameter set on test scenarios
        
        Args:
            param_vector: Combined parameter vector
            test_scenarios: List of test scenarios to evaluate
            
        Returns:
            Fitness score (lower is better)
        """
        # Split parameters
        align_vector, grasp_vector = self._split_parameters(param_vector)
        
        # Update parameter objects
        self.align_params.from_vector(align_vector)
        self.grasp_params.from_vector(grasp_vector)
        
        # Run test scenarios
        total_fitness = 0.0
        success_count = 0
        
        for scenario in test_scenarios:
            result = self._run_scenario(scenario)
            
            # Calculate fitness components
            if result['success']:
                success_count += 1
                time_penalty = result['execution_time'] / 10.0  # Normalize time
                fitness = time_penalty
            else:
                fitness = 10.0  # High penalty for failure
            
            # Add precision penalty
            if 'position_error' in result:
                fitness += result['position_error'] * 2.0
            
            # Add force penalty
            if 'force_error' in result:
                fitness += abs(result['force_error']) * 0.5
            
            total_fitness += fitness
        
        # Add success rate bonus
        success_rate = success_count / len(test_scenarios)
        if success_rate < 0.8:  # Penalize low success rate
            total_fitness *= (2.0 - success_rate)
        
        # Update best if improved
        if total_fitness < self.best_fitness:
            self.best_fitness = total_fitness
            self.best_params = param_vector.copy()
        
        # Record in history
        self.history.append({
            'fitness': total_fitness,
            'success_rate': success_rate,
            'parameters': param_vector.tolist()
        })
        
        return total_fitness
    
    def _run_scenario(self, scenario: Dict) -> Dict:
        """
        Run a single test scenario with current parameters
        
        Args:
            scenario: Test scenario specification
            
        Returns:
            Result dictionary with success, time, errors
        """
        # This would integrate with the actual robot/simulation
        # For now, return simulated results
        
        # Simulate execution
        success = np.random.random() > 0.3  # 70% base success rate
        execution_time = np.random.uniform(3.0, 8.0)
        position_error = np.random.uniform(0.001, 0.01)
        force_error = np.random.uniform(-1.0, 1.0)
        
        return {
            'success': success,
            'execution_time': execution_time,
            'position_error': position_error,
            'force_error': force_error
        }
    
    def get_optimized_parameters(self) -> Tuple[AlignParameters, GraspParameters]:
        """Get the best parameters found during optimization"""
        align_vector, grasp_vector = self._split_parameters(self.best_params)
        
        self.align_params.from_vector(align_vector)
        self.grasp_params.from_vector(grasp_vector)
        
        return self.align_params, self.grasp_params
    
    def save_parameters(self, filepath: str) -> None:
        """Save optimized parameters to file"""
        params_dict = {
            'align': self.align_params.to_vector().tolist(),
            'grasp': self.grasp_params.to_vector().tolist(),
            'fitness': self.best_fitness,
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        logger.info(f"Saved parameters to {filepath}")
    
    def load_parameters(self, filepath: str) -> None:
        """Load parameters from file"""
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        
        self.align_params.from_vector(np.array(params_dict['align']))
        self.grasp_params.from_vector(np.array(params_dict['grasp']))
        self.best_fitness = params_dict.get('fitness', float('inf'))
        self.history = params_dict.get('history', [])
        
        logger.info(f"Loaded parameters from {filepath}")


# ============================================================================
# Training Interface for BC
# ============================================================================

class BCTrainer:
    """
    Behavioral Cloning trainer for learning from demonstrations
    """
    
    def __init__(self, align_network: AlignmentNetwork, grasp_network: GraspNetwork):
        """
        Initialize BC trainer
        
        Args:
            align_network: Network for learning alignment
            grasp_network: Network for learning grasping
        """
        self.align_network = align_network
        self.grasp_network = grasp_network
        
        # Optimizers
        self.align_optimizer = torch.optim.Adam(align_network.parameters(), lr=1e-3)
        self.grasp_optimizer = torch.optim.Adam(grasp_network.parameters(), lr=1e-3)
        
        # Loss functions
        self.align_criterion = nn.MSELoss()
        self.grasp_criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'align_loss': [],
            'grasp_loss': [],
            'validation_loss': []
        }
    
    def train_on_demonstrations(self, demonstrations: List[Dict], 
                               epochs: int = 100,
                               validation_split: float = 0.2) -> None:
        """
        Train networks on demonstration data
        
        Args:
            demonstrations: List of demonstration trajectories
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
        """
        # Split demonstrations
        n_val = int(len(demonstrations) * validation_split)
        val_demos = demonstrations[:n_val]
        train_demos = demonstrations[n_val:]
        
        logger.info(f"Training on {len(train_demos)} demonstrations, validating on {len(val_demos)}")
        
        for epoch in range(epochs):
            # Training
            align_loss = self._train_align_epoch(train_demos)
            grasp_loss = self._train_grasp_epoch(train_demos)
            
            # Validation
            val_loss = self._validate(val_demos)
            
            # Record history
            self.training_history['align_loss'].append(align_loss)
            self.training_history['grasp_loss'].append(grasp_loss)
            self.training_history['validation_loss'].append(val_loss)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Align Loss={align_loss:.4f}, "
                          f"Grasp Loss={grasp_loss:.4f}, Val Loss={val_loss:.4f}")
    
    def _train_align_epoch(self, demonstrations: List[Dict]) -> float:
        """Train alignment network for one epoch"""
        total_loss = 0.0
        n_batches = 0
        
        for demo in demonstrations:
            if 'align_data' not in demo:
                continue
            
            # Extract training data
            input_features = torch.FloatTensor(demo['align_data']['input'])
            target_output = torch.FloatTensor(demo['align_data']['output'])
            
            # Forward pass
            self.align_optimizer.zero_grad()
            predicted = self.align_network(input_features.unsqueeze(0))
            
            # Calculate loss
            loss = self.align_criterion(predicted, target_output.unsqueeze(0))
            
            # Backward pass
            loss.backward()
            self.align_optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _train_grasp_epoch(self, demonstrations: List[Dict]) -> float:
        """Train grasp network for one epoch"""
        total_loss = 0.0
        n_batches = 0
        
        for demo in demonstrations:
            if 'grasp_data' not in demo:
                continue
            
            # Extract training data
            obj_features = torch.FloatTensor(demo['grasp_data']['object_features'])
            robot_state = torch.FloatTensor(demo['grasp_data']['robot_state'])
            tactile = torch.FloatTensor(demo['grasp_data']['tactile'])
            target = torch.FloatTensor(demo['grasp_data']['output'])
            
            # Forward pass
            self.grasp_optimizer.zero_grad()
            predicted = self.grasp_network(
                obj_features.unsqueeze(0),
                robot_state.unsqueeze(0),
                tactile.unsqueeze(0)
            )
            
            # Calculate loss
            loss = self.grasp_criterion(predicted, target.unsqueeze(0))
            
            # Backward pass
            loss.backward()
            self.grasp_optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _validate(self, demonstrations: List[Dict]) -> float:
        """Validate on held-out demonstrations"""
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for demo in demonstrations:
                # Validate alignment
                if 'align_data' in demo:
                    input_features = torch.FloatTensor(demo['align_data']['input'])
                    target_output = torch.FloatTensor(demo['align_data']['output'])
                    predicted = self.align_network(input_features.unsqueeze(0))
                    loss = self.align_criterion(predicted, target_output.unsqueeze(0))
                    total_loss += loss.item()
                    n_samples += 1
                
                # Validate grasp
                if 'grasp_data' in demo:
                    obj_features = torch.FloatTensor(demo['grasp_data']['object_features'])
                    robot_state = torch.FloatTensor(demo['grasp_data']['robot_state'])
                    tactile = torch.FloatTensor(demo['grasp_data']['tactile'])
                    target = torch.FloatTensor(demo['grasp_data']['output'])
                    predicted = self.grasp_network(
                        obj_features.unsqueeze(0),
                        robot_state.unsqueeze(0),
                        tactile.unsqueeze(0)
                    )
                    loss = self.grasp_criterion(predicted, target.unsqueeze(0))
                    total_loss += loss.item()
                    n_samples += 1
        
        return total_loss / max(n_samples, 1)
    
    def save_models(self, align_path: str, grasp_path: str) -> None:
        """Save trained models"""
        torch.save(self.align_network.state_dict(), align_path)
        torch.save(self.grasp_network.state_dict(), grasp_path)
        logger.info(f"Saved models to {align_path} and {grasp_path}")
    
    def load_models(self, align_path: str, grasp_path: str) -> None:
        """Load trained models"""
        self.align_network.load_state_dict(torch.load(align_path))
        self.grasp_network.load_state_dict(torch.load(grasp_path))
        logger.info(f"Loaded models from {align_path} and {grasp_path}")


# ============================================================================
# Example Usage
# ============================================================================

def example_learnable_pick_and_place():
    """Example of pick-and-place with learnable nodes"""
    from behavior_tree import Blackboard, Sequence, NodeStatus, TaskBuilder
    
    # Create blackboard
    blackboard = Blackboard()
    
    # Create learnable parameters
    align_params = AlignParameters()
    grasp_params = GraspParameters()
    
    # Create neural networks (optional)
    align_network = AlignmentNetwork()
    grasp_network = GraspNetwork()
    
    # Create learnable nodes
    move_to_grasp = MoveTo(blackboard, 'grasp_target')  # Scripted
    align = LearnableAlign(blackboard, align_params, align_network)  # Learnable
    grasp = LearnableGrasp(blackboard, grasp_params, grasp_network)  # Learnable
    move_to_place = MoveTo(blackboard, 'place_target')  # Scripted
    place = Place(blackboard)  # Scripted
    
    # Build sequence
    task_sequence = Sequence(
        "LearnablePickAndPlace",
        blackboard,
        [move_to_grasp, align, grasp, move_to_place, place]
    )
    
    # Set up task
    blackboard.update({
        'grasp_target': np.array([0.4, 0.0, 0.1]),
        'place_target': np.array([0.6, 0.2, 0.1]),
        'robot_position': np.array([0.0, 0.0, 0.3]),
        'robot_orientation': np.array([0.0, 0.0, 0.0]),
        'gripper_state': 'open',
        'detected_objects': {'obj1': {'type': 'cube', 'size': 0.05}}
    })
    
    # Execute task
    logger.info("=" * 50)
    logger.info("Executing Learnable Pick-and-Place Task")
    logger.info("=" * 50)
    
    status = task_sequence()
    
    logger.info(f"Task completed with status: {status.value}")
    
    return status


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" LEARNABLE BEHAVIOR NODES DEMONSTRATION")
    print("="*60 + "\n")
    
    # Run example
    example_learnable_pick_and_place()
    
    print("\n" + "="*60)
    print(" DEMONSTRATION COMPLETE")
    print("="*60 + "\n")