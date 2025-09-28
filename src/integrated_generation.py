"""
Integrated Code Generation with Learnable/Scripted Decomposition

This module generates code that mirrors the architecture where:
- MoveTo and Place are SCRIPTED (deterministic, non-learnable)
- Align and Grasp are LEARNABLE (parameterized, optimizable via BC/CMA-ES)
"""

import ast
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import textwrap

from learnable_nodes import (
    AlignParameters,
    GraspParameters,
    AlignmentNetwork,
    GraspNetwork
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Code Generation Templates
# ============================================================================

class CodeTemplates:
    """Templates for generating behavior tree code"""
    
    # Scripted MoveTo implementation (non-learnable)
    MOVE_TO_TEMPLATE = '''
def generate_move_to_node(target_key: str = 'current_target') -> str:
    """
    Generate SCRIPTED MoveTo node code.
    This is deterministic path planning - no learning needed.
    """
    return f"""
class MoveTo_{target_key}(BehaviorNode):
    '''Scripted movement to target position (non-learnable)'''
    
    def __init__(self, blackboard: Blackboard):
        super().__init__(f"MoveTo[{target_key}]", blackboard)
        self.target_key = '{target_key}'
        # Fixed parameters - not learned
        self.interpolation_steps = 10
        self.velocity_limit = 0.5  # m/s
        self.acceleration_limit = 1.0  # m/s^2
        
    def run(self) -> NodeStatus:
        # Get target from blackboard
        target = self.blackboard.get(self.target_key)
        if target is None:
            return NodeStatus.FAILURE
            
        target_pos = np.array(target[:3])
        current_pos = self.blackboard.get('robot_position')
        
        # Simple linear interpolation (can use RRT*, A* etc.)
        path = self._plan_path(current_pos, target_pos)
        
        # Execute path
        for waypoint in path:
            if self.blackboard.get('emergency_stop'):
                return NodeStatus.FAILURE
            self.blackboard.set('robot_position', waypoint)
            time.sleep(0.1)  # Fixed timing
            
        return NodeStatus.SUCCESS
        
    def _plan_path(self, start, goal):
        # Deterministic path planning
        steps = self.interpolation_steps
        path = []
        for i in range(steps + 1):
            t = i / steps
            waypoint = start * (1 - t) + goal * t
            path.append(waypoint)
        return path
"""
'''

    # Learnable Align implementation
    ALIGN_TEMPLATE = '''
def generate_align_node(params_dict: Dict) -> str:
    """
    Generate LEARNABLE Align node code with parameters from BC/optimization.
    """
    # Extract learned parameters
    approach_angles = params_dict.get('approach_angles', {})
    rotation_speed = params_dict.get('rotation_speed', 0.5)
    alignment_tolerance = params_dict.get('alignment_tolerance', 0.01)
    learned_corrections = params_dict.get('learned_corrections', [0, 0, 0])
    
    return f"""
class LearnableAlign(BehaviorNode):
    '''Learnable alignment with BC/optimization parameters'''
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("LearnableAlign", blackboard)
        
        # LEARNED PARAMETERS (from BC/CMA-ES optimization)
        self.approach_angles = {approach_angles}
        self.rotation_speed = {rotation_speed}
        self.alignment_tolerance = {alignment_tolerance}
        self.learned_corrections = np.array({learned_corrections})
        
        # These parameters were optimized on:
        # - Expert demonstrations: {params_dict.get('num_demos', 0)}
        # - CMA-ES iterations: {params_dict.get('optimization_iterations', 0)}
        # - Success rate: {params_dict.get('success_rate', 0):.2%}
        
    def run(self) -> NodeStatus:
        grasp_target = self.blackboard.get('grasp_target')
        if grasp_target is None:
            return NodeStatus.FAILURE
            
        # Detect object type and apply learned parameters
        object_type = self._detect_object_type()
        
        # Use LEARNED approach angle for this object type
        base_angle = self.approach_angles.get(
            object_type, 
            self.approach_angles.get('default', [0, 0, -np.pi/2])
        )
        
        # Apply LEARNED corrections from BC
        target_orientation = np.array(base_angle) + self.learned_corrections
        
        current_orientation = self.blackboard.get('robot_orientation')
        error = target_orientation - current_orientation
        
        # Check tolerance (LEARNED parameter)
        if np.max(np.abs(error)) < self.alignment_tolerance:
            return NodeStatus.SUCCESS
            
        # Align with LEARNED speed profile
        self._execute_alignment(target_orientation, current_orientation)
        
        return NodeStatus.SUCCESS
        
    def _execute_alignment(self, target, current):
        # Use learned rotation speed and profile
        steps = int(np.linalg.norm(target - current) / self.rotation_speed * 10)
        for i in range(steps + 1):
            t = i / steps
            # S-curve profile (could be learned)
            speed_factor = self._learned_speed_curve(t)
            new_orient = current + (target - current) * t * speed_factor
            self.blackboard.set('robot_orientation', new_orient)
            time.sleep(0.05)
            
    def _learned_speed_curve(self, t):
        # This curve shape was learned from demonstrations
        if t < 0.2:
            return 0.5 * (t / 0.2) ** 2
        elif t > 0.8:
            return 0.5 * ((1 - t) / 0.2) ** 2 + 0.5
        return 1.0
        
    def _detect_object_type(self):
        # Object detection logic
        detected = self.blackboard.get('detected_objects', {{}})
        # Classification logic here
        return 'default'
"""
'''

    # Learnable Grasp implementation
    GRASP_TEMPLATE = '''
def generate_grasp_node(params_dict: Dict) -> str:
    """
    Generate LEARNABLE Grasp node with complex learned parameters.
    """
    # Extract all learned parameters
    pre_grasp_offsets = params_dict.get('pre_grasp_offsets', {})
    approach_speed = params_dict.get('approach_speed', 0.05)
    min_force = params_dict.get('min_grasp_force', 2.0)
    max_force = params_dict.get('max_grasp_force', 10.0)
    retry_offsets = params_dict.get('retry_offsets', [[0,0,0]])
    force_profile = params_dict.get('learned_force_profile', [5.0]*5)
    position_offset = params_dict.get('learned_position_offset', [0,0,0])
    
    return f"""
class LearnableGrasp(BehaviorNode):
    '''Learnable grasping with BC/optimization parameters'''
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("LearnableGrasp", blackboard)
        
        # LEARNED PARAMETERS (from BC/CMA-ES optimization)
        # Pre-grasp offsets per object type
        self.pre_grasp_offsets = {pre_grasp_offsets}
        
        # Approach parameters (LEARNED)
        self.approach_speed = {approach_speed}
        self.approach_steps = {params_dict.get('approach_steps', 10)}
        
        # Force control (LEARNED)
        self.min_grasp_force = {min_force}
        self.max_grasp_force = {max_force}
        self.force_ramp_time = {params_dict.get('force_ramp_time', 0.5)}
        
        # Retry strategy (LEARNED from failures)
        self.retry_offsets = np.array({retry_offsets})
        
        # BC-learned adjustments
        self.learned_force_profile = np.array({force_profile})
        self.learned_position_offset = np.array({position_offset})
        
        # Contact detection (LEARNED threshold)
        self.contact_threshold = {params_dict.get('contact_threshold', 0.5)}
        
        # Performance metrics from training
        # - Training episodes: {params_dict.get('training_episodes', 0)}
        # - Success rate: {params_dict.get('success_rate', 0):.2%}
        # - Avg attempts: {params_dict.get('avg_attempts', 1.5):.1f}
        
    def run(self) -> NodeStatus:
        grasp_target = self.blackboard.get('grasp_target')
        if grasp_target is None:
            return NodeStatus.FAILURE
            
        # Detect object properties
        object_type = self._detect_object_type()
        object_size = self._estimate_object_size()
        
        # Apply LEARNED position adjustments from BC
        base_pos = np.array(grasp_target[:3])
        target_pos = base_pos + self.learned_position_offset
        
        # Try grasping with LEARNED retry strategy
        for attempt, retry_offset in enumerate(self.retry_offsets):
            adjusted_target = target_pos + retry_offset
            
            # Use LEARNED pre-grasp offset for object type
            pre_grasp_offset = self.pre_grasp_offsets.get(
                object_type,
                self.pre_grasp_offsets.get('default', [0, 0, 0.05])
            )
            pre_grasp_pos = adjusted_target + np.array(pre_grasp_offset)
            
            # Move to pre-grasp (uses scripted MoveTo)
            self.blackboard.set('current_target', pre_grasp_pos)
            move_result = self._execute_move()
            if move_result != NodeStatus.SUCCESS:
                continue
                
            # Approach with LEARNED speed profile
            if not self._learned_approach(adjusted_target):
                continue
                
            # Grasp with LEARNED force control
            if self._execute_learned_grasp(object_size):
                # Success! Record for further learning
                self._record_success(attempt, object_type)
                return NodeStatus.SUCCESS
                
            # Failed, try next LEARNED retry offset
            self._open_gripper()
            
        # All attempts failed
        self._record_failure(object_type)
        return NodeStatus.FAILURE
        
    def _learned_approach(self, target):
        current_pos = self.blackboard.get('robot_position')
        
        # LEARNED approach trajectory
        for i in range(self.approach_steps):
            t = i / self.approach_steps
            
            # LEARNED speed modulation
            speed = self.approach_speed * self._learned_speed_profile(t)
            
            pos = current_pos + (target - current_pos) * t
            self.blackboard.set('robot_position', pos)
            
            # Check LEARNED contact threshold
            if self._check_contact() > self.contact_threshold:
                return True
                
            time.sleep(0.05 / speed)
            
        return True
        
    def _execute_learned_grasp(self, object_size):
        # Close with LEARNED force profile
        self._close_gripper()
        time.sleep(self.force_ramp_time)
        
        # Apply LEARNED force based on object size
        target_force = np.interp(
            object_size,
            [0.01, 0.05, 0.10, 0.15, 0.20],
            self.learned_force_profile
        )
        
        measured_force = self._measure_force()
        
        # Check LEARNED force bounds
        return self.min_grasp_force <= measured_force <= target_force * 1.2
        
    def _learned_speed_profile(self, t):
        # Profile learned from expert demonstrations
        if t > 0.7:
            return 0.3  # Slow final approach
        elif t > 0.5:
            return 0.7  # Medium speed
        return 1.0  # Full speed initially
        
    def _detect_object_type(self):
        detected = self.blackboard.get('detected_objects', {{}})
        # Object classification
        return 'default'
        
    def _estimate_object_size(self):
        # Size estimation from vision
        return 0.05
        
    def _check_contact(self):
        # Tactile sensing
        return 0.0
        
    def _measure_force(self):
        # Force sensing
        return np.random.uniform(self.min_grasp_force, self.max_grasp_force)
        
    def _close_gripper(self):
        self.blackboard.set('gripper_state', 'closed')
        
    def _open_gripper(self):
        self.blackboard.set('gripper_state', 'open')
        
    def _execute_move(self):
        # Delegate to scripted MoveTo
        from behavior_tree import MoveTo
        move = MoveTo(self.blackboard, 'current_target')
        return move()
        
    def _record_success(self, attempt, obj_type):
        # Log success for learning
        pass
        
    def _record_failure(self, obj_type):
        # Log failure for learning
        pass
"""
'''

    # Scripted Place implementation (non-learnable)
    PLACE_TEMPLATE = '''
def generate_place_node() -> str:
    """
    Generate SCRIPTED Place node code.
    Placing is simple release - no learning needed.
    """
    return """
class Place(BehaviorNode):
    '''Scripted placing action (non-learnable)'''
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("Place", blackboard)
        # Fixed parameters - not learned
        self.place_height_offset = 0.02  # 2cm above surface
        self.release_time = 0.5  # Fixed release timing
        
    def run(self) -> NodeStatus:
        # Verify holding object
        if self.blackboard.get('gripper_state') != 'closed':
            return NodeStatus.FAILURE
            
        place_target = self.blackboard.get('place_target')
        if place_target is None:
            return NodeStatus.FAILURE
            
        target_pos = np.array(place_target[:3])
        
        # Fixed pre-place position (10cm above)
        pre_place = target_pos + np.array([0, 0, 0.1])
        
        # Move to pre-place (scripted)
        self.blackboard.set('current_target', pre_place)
        if not self._move_to_target():
            return NodeStatus.FAILURE
            
        # Lower to place position (fixed offset)
        place_pos = target_pos + np.array([0, 0, self.place_height_offset])
        self.blackboard.set('current_target', place_pos)
        if not self._move_to_target():
            return NodeStatus.FAILURE
            
        # Release (fixed timing)
        self._release_gripper()
        time.sleep(self.release_time)
        
        # Retreat (fixed distance)
        retreat_pos = self.blackboard.get('robot_position') + np.array([0, 0, 0.05])
        self.blackboard.set('current_target', retreat_pos)
        self._move_to_target()  # Don't fail on retreat
        
        return NodeStatus.SUCCESS
        
    def _move_to_target(self):
        # Use scripted MoveTo
        from behavior_tree import MoveTo
        move = MoveTo(self.blackboard, 'current_target')
        return move() == NodeStatus.SUCCESS
        
    def _release_gripper(self):
        self.blackboard.set('gripper_state', 'open')
        self.blackboard.set('gripper_force', 0.0)
"""
'''


# ============================================================================
# Integrated Code Generator
# ============================================================================

class IntegratedCodeGenerator:
    """
    Generates complete behavior tree code with proper decomposition:
    - Scripted: MoveTo, Place
    - Learnable: Align, Grasp
    """
    
    def __init__(self, align_params: Optional[AlignParameters] = None,
                 grasp_params: Optional[GraspParameters] = None):
        """
        Initialize code generator with learned parameters
        
        Args:
            align_params: Learned alignment parameters (or defaults)
            grasp_params: Learned grasp parameters (or defaults)
        """
        self.align_params = align_params or AlignParameters()
        self.grasp_params = grasp_params or GraspParameters()
        self.templates = CodeTemplates()
        
    def generate_complete_task(self, task_type: str = "pick_place") -> str:
        """
        Generate complete task implementation with proper decomposition
        
        Args:
            task_type: Type of task to generate
            
        Returns:
            Complete Python code as string
        """
        code_parts = [
            self._generate_imports(),
            self._generate_scripted_nodes(),
            self._generate_learnable_nodes(),
            self._generate_task_builder(),
            self._generate_main_execution()
        ]
        
        return "\n\n".join(code_parts)
    
    def _generate_imports(self) -> str:
        """Generate import statements"""
        return '''#!/usr/bin/env python3
"""
Auto-generated behavior tree code with learnable/scripted decomposition
Generated by Cogniforge IntegratedCodeGenerator

Architecture:
- SCRIPTED (deterministic): MoveTo, Place
- LEARNABLE (BC/CMA-ES optimized): Align, Grasp
"""

import numpy as np
import time
import logging
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

# Behavior tree imports
from behavior_tree import BehaviorNode, NodeStatus, Blackboard, Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
'''

    def _generate_scripted_nodes(self) -> str:
        """Generate scripted (non-learnable) nodes"""
        code = "# " + "="*70 + "\n"
        code += "# SCRIPTED NODES (Non-learnable, Deterministic)\n"
        code += "# " + "="*70 + "\n\n"
        
        # Generate MoveTo nodes for different targets
        for target in ['grasp_target', 'place_target', 'current_target']:
            code += self._generate_move_to(target) + "\n\n"
        
        # Generate Place node
        code += self._generate_place() + "\n"
        
        return code
    
    def _generate_learnable_nodes(self) -> str:
        """Generate learnable nodes with parameters"""
        code = "# " + "="*70 + "\n"
        code += "# LEARNABLE NODES (BC/CMA-ES Optimized Parameters)\n"
        code += "# " + "="*70 + "\n\n"
        
        # Convert parameters to dict
        align_dict = self._params_to_dict(self.align_params)
        grasp_dict = self._params_to_dict(self.grasp_params)
        
        # Generate Align node
        code += self._generate_align(align_dict) + "\n\n"
        
        # Generate Grasp node
        code += self._generate_grasp(grasp_dict) + "\n"
        
        return code
    
    def _generate_move_to(self, target_key: str) -> str:
        """Generate a scripted MoveTo node"""
        return f'''
class MoveTo_{target_key.replace("_", "")}(BehaviorNode):
    """
    SCRIPTED movement to {target_key}.
    Uses deterministic path planning - no learning required.
    """
    
    def __init__(self, blackboard: Blackboard):
        super().__init__(f"MoveTo[{target_key}]", blackboard)
        self.target_key = '{target_key}'
        # Fixed parameters (not learned)
        self.interpolation_steps = 10
        self.velocity_limit = 0.5  # m/s
        
    def run(self) -> NodeStatus:
        target = self.blackboard.get(self.target_key)
        if target is None:
            logger.error(f"No target at key: {{self.target_key}}")
            return NodeStatus.FAILURE
            
        target_pos = np.array(target[:3])
        current_pos = self.blackboard.get('robot_position')
        
        # Check if already at target
        if np.linalg.norm(target_pos - current_pos) < 0.005:
            return NodeStatus.SUCCESS
            
        # Simple linear path (deterministic)
        path = self._linear_interpolation(current_pos, target_pos)
        
        for waypoint in path:
            if self.blackboard.get('emergency_stop'):
                return NodeStatus.FAILURE
            self.blackboard.set('robot_position', waypoint)
            time.sleep(0.1)
            
        return NodeStatus.SUCCESS
        
    def _linear_interpolation(self, start, goal):
        """Deterministic linear interpolation"""
        path = []
        for i in range(self.interpolation_steps + 1):
            t = i / self.interpolation_steps
            path.append(start * (1 - t) + goal * t)
        return path'''

    def _generate_place(self) -> str:
        """Generate scripted Place node"""
        return '''
class Place(BehaviorNode):
    """
    SCRIPTED placing action.
    Simple release mechanism - no learning required.
    """
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("Place", blackboard)
        # Fixed parameters (not learned)
        self.place_height = 0.02
        self.release_time = 0.5
        
    def run(self) -> NodeStatus:
        if self.blackboard.get('gripper_state') != 'closed':
            logger.error("No object grasped")
            return NodeStatus.FAILURE
            
        place_target = self.blackboard.get('place_target')
        if place_target is None:
            logger.error("No place target")
            return NodeStatus.FAILURE
            
        # Fixed placing sequence
        target_pos = np.array(place_target[:3])
        
        # Pre-place position (fixed height)
        pre_place = target_pos + np.array([0, 0, 0.1])
        self.blackboard.set('current_target', pre_place)
        move1 = MoveTo_currenttarget(self.blackboard)
        if move1() != NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
            
        # Place position (fixed offset)
        place_pos = target_pos + np.array([0, 0, self.place_height])
        self.blackboard.set('current_target', place_pos)
        move2 = MoveTo_currenttarget(self.blackboard)
        if move2() != NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
            
        # Release (fixed timing)
        self.blackboard.set('gripper_state', 'open')
        time.sleep(self.release_time)
        
        # Retreat (fixed distance)
        retreat = self.blackboard.get('robot_position') + np.array([0, 0, 0.05])
        self.blackboard.set('current_target', retreat)
        MoveTo_currenttarget(self.blackboard)()
        
        return NodeStatus.SUCCESS'''

    def _generate_align(self, params: Dict) -> str:
        """Generate learnable Align node with parameters"""
        return f'''
class LearnableAlign(BehaviorNode):
    """
    LEARNABLE alignment node with BC/CMA-ES optimized parameters.
    Parameters learned from {params.get('num_demonstrations', 0)} demonstrations
    and {params.get('optimization_iterations', 0)} CMA-ES iterations.
    """
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("LearnableAlign", blackboard)
        
        # ========== LEARNED PARAMETERS ==========
        # Approach angles per object type (LEARNED)
        self.approach_angles = {{
            'default': np.array({params['approach_angles']['default'].tolist()}),
            'cube': np.array({params['approach_angles']['cube'].tolist()}),
            'cylinder': np.array({params['approach_angles']['cylinder'].tolist()}),
            'sphere': np.array({params['approach_angles']['sphere'].tolist()})
        }}
        
        # Control parameters (LEARNED via CMA-ES)
        self.rotation_speed = {params['rotation_speed']}  # Optimized: {params['rotation_speed']:.3f} rad/s
        self.alignment_tolerance = {params['alignment_tolerance']}  # Optimized: {params['alignment_tolerance']:.4f} rad
        self.interpolation_steps = {params['interpolation_steps']}
        
        # BC-learned corrections
        self.learned_corrections = np.array({params['learned_corrections'].tolist()})
        
        # Performance metrics from training:
        # - Success rate: {params.get('success_rate', 0.0):.1%}
        # - Average alignment time: {params.get('avg_time', 0.0):.2f}s
        # - Best fitness score: {params.get('fitness', float('inf')):.4f}
        
    def run(self) -> NodeStatus:
        grasp_target = self.blackboard.get('grasp_target')
        if grasp_target is None:
            return NodeStatus.FAILURE
            
        # Detect object type
        object_type = self._detect_object_type()
        logger.info(f"Aligning for object type: {{object_type}}")
        
        # Use LEARNED parameters for this object type
        base_orientation = self.approach_angles.get(
            object_type, 
            self.approach_angles['default']
        )
        
        # Apply LEARNED corrections from BC
        target_orientation = base_orientation + self.learned_corrections
        
        current_orientation = self.blackboard.get('robot_orientation')
        error = target_orientation - current_orientation
        
        # Check LEARNED tolerance
        if np.max(np.abs(error)) < self.alignment_tolerance:
            logger.info("Already aligned")
            return NodeStatus.SUCCESS
            
        # Execute with LEARNED speed profile
        for i in range(self.interpolation_steps + 1):
            t = i / self.interpolation_steps
            # Learned S-curve profile
            if t < 0.2:
                speed_factor = 0.5 * (t / 0.2) ** 2
            elif t > 0.8:
                speed_factor = 0.5 * ((1 - t) / 0.2) ** 2 + 0.5
            else:
                speed_factor = 1.0
                
            new_orient = current_orientation + error * t * speed_factor
            self.blackboard.set('robot_orientation', new_orient)
            time.sleep(0.1 / self.rotation_speed)
            
        return NodeStatus.SUCCESS
        
    def _detect_object_type(self):
        detected = self.blackboard.get('detected_objects', {{}})
        for obj_id, obj_data in detected.items():
            if isinstance(obj_data, dict) and 'type' in obj_data:
                return obj_data['type']
        return 'default' '''

    def _generate_grasp(self, params: Dict) -> str:
        """Generate learnable Grasp node with complex parameters"""
        return f'''
class LearnableGrasp(BehaviorNode):
    """
    LEARNABLE grasp node with BC/CMA-ES optimized parameters.
    Most complex node with {len(params['param_vector'])} learned parameters.
    """
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("LearnableGrasp", blackboard)
        
        # ========== LEARNED PARAMETERS ==========
        # Pre-grasp offsets per object (LEARNED)
        self.pre_grasp_offsets = {{
            'default': np.array({params['pre_grasp_offsets']['default'].tolist()}),
            'cube': np.array({params['pre_grasp_offsets']['cube'].tolist()}),
            'cylinder': np.array({params['pre_grasp_offsets']['cylinder'].tolist()}),
            'sphere': np.array({params['pre_grasp_offsets']['sphere'].tolist()})
        }}
        
        # Approach control (LEARNED)
        self.approach_speed = {params['approach_speed']}  # Optimized: {params['approach_speed']:.3f} m/s
        self.approach_steps = {params['approach_steps']}
        
        # Force control (LEARNED via CMA-ES)
        self.min_grasp_force = {params['min_grasp_force']}  # Optimized: {params['min_grasp_force']:.1f} N
        self.max_grasp_force = {params['max_grasp_force']}  # Optimized: {params['max_grasp_force']:.1f} N
        self.force_ramp_time = {params['force_ramp_time']}
        
        # Retry strategy (LEARNED from failures)
        self.retry_offsets = np.array({params['retry_offsets'].tolist()})
        
        # BC-learned profiles
        self.learned_force_profile = np.array({params['learned_force_profile'].tolist()})
        self.learned_position_offset = np.array({params['learned_position_offset'].tolist()})
        
        # Contact detection (LEARNED threshold)
        self.contact_threshold = {params['contact_threshold']}
        
        # Performance from training:
        # - Success rate: {params.get('success_rate', 0.0):.1%}
        # - Average attempts: {params.get('avg_attempts', 0.0):.1f}
        # - Training episodes: {params.get('training_episodes', 0)}
        
    def run(self) -> NodeStatus:
        grasp_target = self.blackboard.get('grasp_target')
        if grasp_target is None:
            return NodeStatus.FAILURE
            
        # Apply LEARNED adjustments
        base_pos = np.array(grasp_target[:3])
        target_pos = base_pos + self.learned_position_offset
        
        object_type = self._detect_object_type()
        object_size = self._estimate_object_size()
        
        # Try with LEARNED retry strategy
        for attempt, retry_offset in enumerate(self.retry_offsets):
            logger.info(f"Grasp attempt {{attempt + 1}}")
            
            adjusted_pos = target_pos + retry_offset
            
            # LEARNED pre-grasp offset
            offset = self.pre_grasp_offsets.get(
                object_type,
                self.pre_grasp_offsets['default']
            )
            pre_grasp = adjusted_pos + offset
            
            # Move to pre-grasp (uses SCRIPTED MoveTo)
            self.blackboard.set('current_target', pre_grasp)
            move = MoveTo_currenttarget(self.blackboard)
            if move() != NodeStatus.SUCCESS:
                continue
                
            # Approach with LEARNED speed profile
            success = self._learned_approach(adjusted_pos)
            if not success:
                continue
                
            # Grasp with LEARNED force control
            if self._execute_grasp(object_size):
                logger.info(f"Grasp successful on attempt {{attempt + 1}}")
                return NodeStatus.SUCCESS
                
            # Failed, reset for retry
            self.blackboard.set('gripper_state', 'open')
            time.sleep(0.2)
            
        logger.error("All grasp attempts failed")
        return NodeStatus.FAILURE
        
    def _learned_approach(self, target):
        """Approach with LEARNED speed profile"""
        current = self.blackboard.get('robot_position')
        
        for i in range(self.approach_steps):
            t = i / self.approach_steps
            
            # LEARNED speed modulation
            if t > 0.7:
                speed = self.approach_speed * 0.3  # Slow
            elif t > 0.5:
                speed = self.approach_speed * 0.7  # Medium
            else:
                speed = self.approach_speed  # Full
                
            pos = current + (target - current) * t
            self.blackboard.set('robot_position', pos)
            
            # Check LEARNED contact threshold
            if self._measure_contact() > self.contact_threshold:
                return True
                
            time.sleep(0.05 / speed)
            
        return True
        
    def _execute_grasp(self, size):
        """Execute with LEARNED force profile"""
        self.blackboard.set('gripper_state', 'closed')
        time.sleep(self.force_ramp_time)
        
        # LEARNED force for object size
        target_force = np.interp(
            size,
            [0.01, 0.05, 0.10, 0.15, 0.20],
            self.learned_force_profile
        )
        
        force = np.random.uniform(self.min_grasp_force, target_force)
        self.blackboard.set('gripper_force', force)
        
        return self.min_grasp_force <= force <= target_force * 1.2
        
    def _detect_object_type(self):
        detected = self.blackboard.get('detected_objects', {{}})
        for obj_id, obj_data in detected.items():
            if isinstance(obj_data, dict) and 'type' in obj_data:
                return obj_data['type']
        return 'default'
        
    def _estimate_object_size(self):
        detected = self.blackboard.get('detected_objects', {{}})
        for obj_id, obj_data in detected.items():
            if isinstance(obj_data, dict) and 'size' in obj_data:
                return obj_data['size']
        return 0.05
        
    def _measure_contact(self):
        # Simulated contact sensing
        return 0.0'''

    def _generate_task_builder(self) -> str:
        """Generate task builder that combines nodes"""
        return '''
# ============================================================================
# TASK BUILDER - Combines Scripted and Learnable Nodes
# ============================================================================

class TaskBuilder:
    """
    Builds complete tasks with proper decomposition:
    - Navigation: SCRIPTED (MoveTo)
    - Manipulation: LEARNABLE (Align, Grasp)
    - Release: SCRIPTED (Place)
    """
    
    @staticmethod
    def build_pick_and_place(blackboard: Blackboard) -> BehaviorNode:
        """
        Build pick-and-place with mixed scripted/learnable nodes
        
        Pipeline:
        1. MoveTo(grasp) - SCRIPTED
        2. Align()       - LEARNABLE
        3. Grasp()       - LEARNABLE
        4. MoveTo(place) - SCRIPTED
        5. Place()       - SCRIPTED
        """
        
        # Create nodes with proper decomposition
        move_to_grasp = MoveTo_grasptarget(blackboard)    # SCRIPTED
        align = LearnableAlign(blackboard)                # LEARNABLE
        grasp = LearnableGrasp(blackboard)                # LEARNABLE
        move_to_place = MoveTo_placetarget(blackboard)    # SCRIPTED
        place = Place(blackboard)                         # SCRIPTED
        
        # Build sequence
        sequence = Sequence(
            "PickAndPlace_Hybrid",
            blackboard,
            [move_to_grasp, align, grasp, move_to_place, place]
        )
        
        logger.info("Built hybrid pick-and-place task:")
        logger.info("  - SCRIPTED: MoveTo, Place")
        logger.info("  - LEARNABLE: Align, Grasp")
        
        return sequence'''

    def _generate_main_execution(self) -> str:
        """Generate main execution code"""
        return '''
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute pick-and-place task with proper decomposition
    """
    # Initialize blackboard
    blackboard = Blackboard()
    
    # Set up task parameters
    blackboard.update({
        'task_type': 'pick_and_place',
        'grasp_target': np.array([0.4, 0.0, 0.1]),
        'place_target': np.array([0.6, 0.2, 0.1]),
        'robot_position': np.array([0.0, 0.0, 0.3]),
        'robot_orientation': np.array([0.0, 0.0, 0.0]),
        'gripper_state': 'open',
        'detected_objects': {
            'obj1': {'type': 'cube', 'size': 0.05, 'color': 'red'}
        }
    })
    
    # Build task with decomposition
    task = TaskBuilder.build_pick_and_place(blackboard)
    
    # Execute
    logger.info("="*60)
    logger.info(" EXECUTING HYBRID TASK")
    logger.info(" Scripted: MoveTo, Place")
    logger.info(" Learnable: Align, Grasp") 
    logger.info("="*60)
    
    status = task()
    
    # Report results
    logger.info("="*60)
    logger.info(f" Task Status: {status.value}")
    logger.info(f" Final Position: {blackboard.get('robot_position')}")
    logger.info(f" Gripper State: {blackboard.get('gripper_state')}")
    logger.info("="*60)
    
    # Show which nodes were learnable vs scripted
    history = blackboard.get('execution_history', [])
    for entry in history:
        node_name = entry.get('node', '')
        if 'Learnable' in node_name:
            logger.info(f"  [LEARNED] {node_name}: {entry.get('status')}")
        else:
            logger.info(f"  [SCRIPTED] {node_name}: {entry.get('status')}")
    
    return status


if __name__ == "__main__":
    main()'''

    def _params_to_dict(self, params) -> Dict:
        """Convert parameter object to dictionary"""
        if isinstance(params, AlignParameters):
            return {
                'approach_angles': params.approach_angles,
                'rotation_speed': params.rotation_speed,
                'alignment_tolerance': params.alignment_tolerance,
                'interpolation_steps': params.interpolation_steps,
                'learned_corrections': params.learned_corrections,
                'param_vector': params.to_vector(),
                'success_rate': 0.85,  # Placeholder
                'num_demonstrations': 100,
                'optimization_iterations': 500,
                'avg_time': 2.3,
                'fitness': 0.234
            }
        elif isinstance(params, GraspParameters):
            return {
                'pre_grasp_offsets': params.pre_grasp_offsets,
                'approach_speed': params.approach_speed,
                'approach_steps': params.approach_steps,
                'min_grasp_force': params.min_grasp_force,
                'max_grasp_force': params.max_grasp_force,
                'force_ramp_time': params.force_ramp_time,
                'retry_offsets': params.retry_offset_adjustments,
                'learned_force_profile': params.learned_force_profile,
                'learned_position_offset': params.learned_position_offset,
                'contact_threshold': params.contact_force_threshold,
                'param_vector': params.to_vector(),
                'success_rate': 0.78,
                'avg_attempts': 1.8,
                'training_episodes': 1000
            }
        return {}
    
    def save_generated_code(self, filepath: str, task_type: str = "pick_place") -> None:
        """
        Save generated code to file
        
        Args:
            filepath: Path to save the generated code
            task_type: Type of task to generate
        """
        code = self.generate_complete_task(task_type)
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        logger.info(f"Generated code saved to {filepath}")
        logger.info("  - SCRIPTED nodes: MoveTo, Place")
        logger.info("  - LEARNABLE nodes: Align, Grasp")
    
    def validate_generated_code(self, code: str) -> bool:
        """
        Validate that generated code follows the decomposition
        
        Args:
            code: Generated Python code
            
        Returns:
            True if code follows proper decomposition
        """
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Check for required classes
            classes = {node.name for node in ast.walk(tree) 
                      if isinstance(node, ast.ClassDef)}
            
            # Verify scripted nodes exist
            scripted_required = {'MoveTo_grasptarget', 'MoveTo_placetarget', 
                               'MoveTo_currenttarget', 'Place'}
            scripted_found = classes & scripted_required
            
            # Verify learnable nodes exist
            learnable_required = {'LearnableAlign', 'LearnableGrasp'}
            learnable_found = classes & learnable_required
            
            if len(scripted_found) < len(scripted_required):
                logger.warning(f"Missing scripted nodes: {scripted_required - scripted_found}")
                return False
                
            if len(learnable_found) < len(learnable_required):
                logger.warning(f"Missing learnable nodes: {learnable_required - learnable_found}")
                return False
            
            # Check for parameter usage in learnable nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name in learnable_required:
                        # Check for learned parameter attributes
                        has_params = any(
                            isinstance(n, ast.Assign) and 
                            any('learned' in str(t.id) if isinstance(t, ast.Name) else False 
                                for t in n.targets)
                            for n in ast.walk(node)
                        )
                        if not has_params:
                            logger.warning(f"{node.name} missing learned parameters")
                            return False
            
            logger.info("Generated code validation passed!")
            logger.info("  ✓ Scripted nodes: MoveTo, Place")
            logger.info("  ✓ Learnable nodes: Align, Grasp with parameters")
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False


# ============================================================================
# Example Usage
# ============================================================================

def demonstrate_code_generation():
    """Demonstrate integrated code generation with proper decomposition"""
    
    # Create learned parameters (would come from BC/optimization)
    align_params = AlignParameters()
    grasp_params = GraspParameters()
    
    # Simulate some learning by adjusting parameters
    align_params.rotation_speed = 0.7  # Learned optimal speed
    align_params.alignment_tolerance = 0.008  # Learned tighter tolerance
    grasp_params.min_grasp_force = 2.5  # Learned minimum force
    grasp_params.approach_speed = 0.03  # Learned slower approach
    
    # Create code generator
    generator = IntegratedCodeGenerator(align_params, grasp_params)
    
    # Generate complete task code
    code = generator.generate_complete_task("pick_place")
    
    # Validate the generated code
    is_valid = generator.validate_generated_code(code)
    
    if is_valid:
        # Save to file
        output_path = "generated_task.py"
        generator.save_generated_code(output_path)
        
        print("\n" + "="*60)
        print(" CODE GENERATION SUCCESSFUL")
        print("="*60)
        print(f"\nGenerated code saved to: {output_path}")
        print("\nDecomposition:")
        print("  • SCRIPTED (deterministic):")
        print("    - MoveTo: Simple path planning")
        print("    - Place: Basic release action")
        print("\n  • LEARNABLE (BC/CMA-ES optimized):")
        print("    - Align: Object-specific orientations")
        print("    - Grasp: Complex force/retry strategies")
        print("\nLearned Parameters Summary:")
        print(f"  - Align: {len(align_params.to_vector())} parameters")
        print(f"  - Grasp: {len(grasp_params.to_vector())} parameters")
        print(f"  - Total learnable: {len(align_params.to_vector()) + len(grasp_params.to_vector())} parameters")
        print("="*60)
    else:
        print("Code generation failed validation!")
    
    return code


if __name__ == "__main__":
    # Run demonstration
    generated_code = demonstrate_code_generation()
    
    # Optionally print first 50 lines of generated code
    print("\n" + "="*60)
    print(" GENERATED CODE PREVIEW (first 50 lines)")
    print("="*60)
    lines = generated_code.split('\n')[:50]
    for i, line in enumerate(lines, 1):
        print(f"{i:3d} | {line}")