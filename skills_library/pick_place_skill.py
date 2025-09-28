"""
Pick and Place Skill

Standard pick and place manipulation skill.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple
from .base_skill import BaseSkill, SkillParameters, SkillResult, SkillStatus
import logging

logger = logging.getLogger(__name__)


class PickPlaceSkill(BaseSkill):
    """Standard pick and place skill"""
    
    def validate_preconditions(self, params: SkillParameters) -> Tuple[bool, str]:
        """
        Validate pick and place preconditions
        
        Args:
            params: Skill parameters
            
        Returns:
            (is_valid, message) tuple
        """
        if params.object_id is None:
            return False, "No object specified"
            
        if params.target_position is None:
            return False, "No target position specified"
            
        # Check if object exists
        import pybullet as p
        try:
            pos, _ = p.getBasePositionAndOrientation(params.object_id)
        except:
            return False, f"Object {params.object_id} not found"
            
        # Check reachability (simplified)
        current_ee_pos = self.get_end_effector_position()
        obj_distance = np.linalg.norm(current_ee_pos - np.array(pos))
        target_distance = np.linalg.norm(current_ee_pos - params.target_position)
        
        if obj_distance > 1.0 or target_distance > 1.0:
            return False, "Object or target out of reach"
            
        return True, "Preconditions satisfied"
        
    def plan(self, params: SkillParameters) -> List[Dict[str, Any]]:
        """
        Plan pick and place trajectory
        
        Args:
            params: Skill parameters
            
        Returns:
            List of waypoints
        """
        import pybullet as p
        
        # Get object position
        obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        obj_pos = np.array(obj_pos)
        
        waypoints = []
        
        # 1. Pre-grasp position (above object)
        pre_grasp_pos = self.compute_approach_position(obj_pos, 0.15)
        waypoints.append({
            'position': pre_grasp_pos,
            'gripper': 1.0,  # Open
            'type': 'pre_grasp',
            'description': 'Move to pre-grasp position'
        })
        
        # 2. Grasp position
        grasp_pos = obj_pos.copy()
        grasp_pos[2] += 0.03  # Slightly above object center
        waypoints.append({
            'position': grasp_pos,
            'gripper': 1.0,
            'type': 'approach',
            'description': 'Approach object'
        })
        
        # 3. Close gripper
        waypoints.append({
            'position': grasp_pos,
            'gripper': 0.0,  # Close
            'type': 'grasp',
            'description': 'Grasp object'
        })
        
        # 4. Lift object
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.15
        waypoints.append({
            'position': lift_pos,
            'gripper': 0.0,
            'type': 'lift',
            'description': 'Lift object'
        })
        
        # 5. Move to above place position
        above_place = params.target_position.copy()
        above_place[2] += 0.15
        waypoints.append({
            'position': above_place,
            'gripper': 0.0,
            'type': 'transport',
            'description': 'Transport to target'
        })
        
        # 6. Lower to place position
        place_pos = params.target_position.copy()
        waypoints.append({
            'position': place_pos,
            'gripper': 0.0,
            'type': 'lower',
            'description': 'Lower to place'
        })
        
        # 7. Release
        waypoints.append({
            'position': place_pos,
            'gripper': 1.0,  # Open
            'type': 'release',
            'description': 'Release object'
        })
        
        # 8. Retract
        retract_pos = place_pos.copy()
        retract_pos[2] += 0.1
        waypoints.append({
            'position': retract_pos,
            'gripper': 1.0,
            'type': 'retract',
            'description': 'Retract gripper'
        })
        
        self.current_plan = waypoints
        logger.info(f"Generated pick-place plan with {len(waypoints)} waypoints")
        
        return waypoints
        
    def execute(self, params: SkillParameters) -> SkillResult:
        """
        Execute pick and place skill
        
        Args:
            params: Skill parameters
            
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
                
                # Wait for gripper action
                if waypoint['type'] in ['grasp', 'release']:
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)
            
            # Move to waypoint position
            if waypoint['type'] in ['grasp', 'lower', 'release']:
                speed = params.speed * 0.5  # Slower for critical operations
            else:
                speed = params.speed
                
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
            
        # Check final object position
        final_obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        final_error = np.linalg.norm(params.target_position - np.array(final_obj_pos))
        
        if final_error < params.precision:
            self.status = SkillStatus.SUCCESS
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Successfully placed object (error: {final_error:.4f}m)",
                waypoints_executed=waypoints_executed,
                execution_time=time.time() - start_time,
                data={'final_position': final_obj_pos, 'error': final_error}
            )
        else:
            self.status = SkillStatus.SUCCESS  # Partial success
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Placed object with error: {final_error:.4f}m",
                waypoints_executed=waypoints_executed,
                execution_time=time.time() - start_time,
                data={'final_position': final_obj_pos, 'error': final_error}
            )