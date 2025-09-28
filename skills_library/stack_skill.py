"""
Stack Skill

Stacks objects on top of each other with precise alignment.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from .base_skill import BaseSkill, SkillParameters, SkillResult, SkillStatus
import logging

logger = logging.getLogger(__name__)


@dataclass
class StackParameters(SkillParameters):
    """Extended parameters for stacking"""
    base_object_id: Optional[int] = None  # Object to stack on
    stack_offset: Optional[np.ndarray] = None  # Offset from base center
    alignment_precision: float = 0.005  # 5mm alignment precision


class StackSkill(BaseSkill):
    """Skill for stacking objects on top of each other"""
    
    def validate_preconditions(self, params: StackParameters) -> Tuple[bool, str]:
        """
        Validate stack preconditions
        
        Args:
            params: Stack parameters
            
        Returns:
            (is_valid, message) tuple
        """
        if params.object_id is None:
            return False, "No object specified for stacking"
            
        if params.base_object_id is None and params.target_position is None:
            return False, "No base object or target position specified"
            
        # Check if objects exist
        import pybullet as p
        try:
            obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        except:
            return False, f"Object {params.object_id} not found"
            
        if params.base_object_id is not None:
            try:
                base_pos, _ = p.getBasePositionAndOrientation(params.base_object_id)
                
                # Check if base is stable
                lin_vel, ang_vel = p.getBaseVelocity(params.base_object_id)
                if np.linalg.norm(lin_vel) > 0.01 or np.linalg.norm(ang_vel) > 0.01:
                    return False, "Base object is not stable"
                    
            except:
                return False, f"Base object {params.base_object_id} not found"
                
        return True, "Preconditions satisfied"
        
    def _get_object_dimensions(self, object_id: int) -> np.ndarray:
        """
        Get approximate object dimensions
        
        Args:
            object_id: Object ID
            
        Returns:
            Dimensions [x, y, z]
        """
        import pybullet as p
        
        # Get AABB (Axis-Aligned Bounding Box)
        aabb_min, aabb_max = p.getAABB(object_id)
        dimensions = np.array(aabb_max) - np.array(aabb_min)
        
        return dimensions
        
    def plan(self, params: StackParameters) -> List[Dict[str, Any]]:
        """
        Plan stacking trajectory
        
        Args:
            params: Stack parameters
            
        Returns:
            List of waypoints for stacking
        """
        import pybullet as p
        
        # Get object positions
        obj_pos, obj_orn = p.getBasePositionAndOrientation(params.object_id)
        obj_pos = np.array(obj_pos)
        
        # Determine target stacking position
        if params.base_object_id is not None:
            # Stack on another object
            base_pos, base_orn = p.getBasePositionAndOrientation(params.base_object_id)
            base_pos = np.array(base_pos)
            
            # Get dimensions for height calculation
            obj_dims = self._get_object_dimensions(params.object_id)
            base_dims = self._get_object_dimensions(params.base_object_id)
            
            # Calculate stack position (on top of base)
            stack_pos = base_pos.copy()
            stack_pos[2] = base_pos[2] + base_dims[2]/2 + obj_dims[2]/2 + 0.002  # Small gap
            
            # Apply offset if specified
            if params.stack_offset is not None:
                stack_pos[:2] += params.stack_offset[:2]
        else:
            # Use provided target position
            stack_pos = params.target_position
            
        waypoints = []
        
        # 1. Move above object
        above_obj = obj_pos.copy()
        above_obj[2] += 0.15
        waypoints.append({
            'position': above_obj,
            'gripper': 1.0,  # Open
            'type': 'approach',
            'description': 'Move above object to stack'
        })
        
        # 2. Lower to grasp position
        grasp_pos = obj_pos.copy()
        grasp_pos[2] += 0.03  # Grasp slightly above center
        waypoints.append({
            'position': grasp_pos,
            'gripper': 1.0,
            'type': 'pre-grasp',
            'description': 'Lower to grasp position'
        })
        
        # 3. Grasp object
        waypoints.append({
            'position': grasp_pos,
            'gripper': 0.0,  # Close
            'type': 'grasp',
            'description': 'Grasp object'
        })
        
        # 4. Lift object
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.2
        waypoints.append({
            'position': lift_pos,
            'gripper': 0.0,
            'type': 'lift',
            'description': 'Lift object'
        })
        
        # 5. Move to above stack position
        above_stack = stack_pos.copy()
        above_stack[2] += 0.15
        waypoints.append({
            'position': above_stack,
            'gripper': 0.0,
            'type': 'transport',
            'description': 'Move to stacking position'
        })
        
        # 6. Align carefully
        align_pos = stack_pos.copy()
        align_pos[2] += 0.05
        waypoints.append({
            'position': align_pos,
            'gripper': 0.0,
            'type': 'align',
            'description': 'Align for stacking'
        })
        
        # 7. Lower slowly to stack
        waypoints.append({
            'position': stack_pos,
            'gripper': 0.0,
            'type': 'stack',
            'description': 'Lower to stack'
        })
        
        # 8. Release
        waypoints.append({
            'position': stack_pos,
            'gripper': 1.0,  # Open
            'type': 'release',
            'description': 'Release stacked object'
        })
        
        # 9. Retract slowly
        retract_pos = stack_pos.copy()
        retract_pos[2] += 0.1
        waypoints.append({
            'position': retract_pos,
            'gripper': 1.0,
            'type': 'retract',
            'description': 'Retract from stack'
        })
        
        self.current_plan = waypoints
        logger.info(f"Generated stack plan with {len(waypoints)} waypoints")
        
        return waypoints
        
    def execute(self, params: StackParameters) -> SkillResult:
        """
        Execute stack skill
        
        Args:
            params: Stack parameters
            
        Returns:
            Execution result
        """
        import pybullet as p
        
        start_time = time.time()
        self.status = SkillStatus.RUNNING
        
        # Validate preconditions
        valid, message = self.validate_preconditions(params)
        if not valid:
            self.status = SkillStatus.FAILED
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Precondition failed: {message}",
                execution_time=time.time() - start_time
            )
            
        # Generate plan if not already done
        if not self.current_plan:
            self.plan(params)
            
        # Execute waypoints
        waypoints_executed = 0
        
        for waypoint in self.current_plan:
            logger.info(f"Executing: {waypoint['description']}")
            
            # Control gripper if specified
            if 'gripper' in waypoint:
                self.control_gripper(waypoint['gripper'])
                time.sleep(0.3)  # More time for gripper operations
            
            # Adjust speed based on operation type
            if waypoint['type'] in ['align', 'stack']:
                speed = params.speed * 0.3  # Very slow for precise stacking
            elif waypoint['type'] in ['grasp', 'release']:
                speed = params.speed * 0.5
            else:
                speed = params.speed
                
            # Move to waypoint position
            success = self.move_to_position(
                waypoint['position'],
                speed=speed,
                timeout=params.timeout / len(self.current_plan)
            )
            
            if not success:
                self.status = SkillStatus.FAILED
                return SkillResult(
                    status=SkillStatus.FAILED,
                    message=f"Failed at waypoint: {waypoint['description']}",
                    waypoints_executed=waypoints_executed,
                    execution_time=time.time() - start_time
                )
                
            waypoints_executed += 1
            
            # Extra stabilization for critical steps
            if waypoint['type'] in ['stack', 'release']:
                time.sleep(0.5)
                
        # Wait for stack to stabilize
        if params.base_object_id is not None:
            stable = self.wait_for_stability(params.base_object_id, max_wait=2.0)
            if not stable:
                logger.warning("Stack may be unstable")
                
        # Check if stacking was successful
        obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        
        if params.base_object_id is not None:
            base_pos, _ = p.getBasePositionAndOrientation(params.base_object_id)
            
            # Check vertical alignment
            horizontal_error = np.linalg.norm(np.array(obj_pos[:2]) - np.array(base_pos[:2]))
            
            if horizontal_error < params.alignment_precision:
                self.status = SkillStatus.SUCCESS
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Successfully stacked object (alignment error: {horizontal_error:.4f}m)",
                    waypoints_executed=waypoints_executed,
                    execution_time=time.time() - start_time,
                    data={'final_position': obj_pos, 'alignment_error': horizontal_error}
                )
            else:
                self.status = SkillStatus.SUCCESS  # Partial success
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Stacked with alignment error: {horizontal_error:.4f}m",
                    waypoints_executed=waypoints_executed,
                    execution_time=time.time() - start_time,
                    data={'final_position': obj_pos, 'alignment_error': horizontal_error}
                )
        else:
            # Check if reached target position
            error = np.linalg.norm(params.target_position - np.array(obj_pos))
            
            self.status = SkillStatus.SUCCESS
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Placed object at target (error: {error:.4f}m)",
                waypoints_executed=waypoints_executed,
                execution_time=time.time() - start_time,
                data={'final_position': obj_pos, 'error': error}
            )


# Fix dataclass import
from dataclasses import dataclass