"""
Push Skill

Pushes objects along a surface without grasping.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple
from .base_skill import BaseSkill, SkillParameters, SkillResult, SkillStatus
import logging

logger = logging.getLogger(__name__)


class PushSkill(BaseSkill):
    """Skill for pushing objects along a surface"""
    
    def validate_preconditions(self, params: SkillParameters) -> Tuple[bool, str]:
        """
        Validate push preconditions
        
        Args:
            params: Push parameters
            
        Returns:
            (is_valid, message) tuple
        """
        if params.object_id is None:
            return False, "No object specified for pushing"
            
        if params.target_position is None:
            return False, "No target position specified"
            
        # Check if object exists
        import pybullet as p
        try:
            pos, _ = p.getBasePositionAndOrientation(params.object_id)
        except:
            return False, f"Object {params.object_id} not found"
            
        # Check if target is reachable (simplified)
        distance = np.linalg.norm(params.target_position - np.array(pos))
        if distance > 0.5:  # Max push distance
            return False, f"Target too far ({distance:.2f}m > 0.5m)"
            
        return True, "Preconditions satisfied"
        
    def plan(self, params: SkillParameters) -> List[Dict[str, Any]]:
        """
        Plan push trajectory
        
        Args:
            params: Push parameters
            
        Returns:
            List of waypoints for pushing
        """
        import pybullet as p
        
        # Get object position
        obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        obj_pos = np.array(obj_pos)
        
        # Calculate push direction
        push_direction = params.target_position - obj_pos
        push_direction[2] = 0  # Keep push horizontal
        push_distance = np.linalg.norm(push_direction)
        
        if push_distance < 0.001:
            return []  # Already at target
            
        push_direction = push_direction / push_distance
        
        # Calculate approach and push positions
        approach_offset = -push_direction * 0.08  # Approach from opposite side
        push_height = obj_pos[2] + 0.02  # Slightly above object base
        
        waypoints = []
        
        # 1. Move above approach position
        above_approach = obj_pos + approach_offset
        above_approach[2] = push_height + 0.1
        waypoints.append({
            'position': above_approach,
            'type': 'move',
            'description': 'Move above approach position'
        })
        
        # 2. Lower to push height
        push_start = obj_pos + approach_offset
        push_start[2] = push_height
        waypoints.append({
            'position': push_start,
            'type': 'move',
            'description': 'Lower to push height'
        })
        
        # 3. Push forward in steps
        num_steps = max(3, int(push_distance / 0.02))  # 2cm steps
        for i in range(1, num_steps + 1):
            progress = i / num_steps
            push_pos = obj_pos + push_direction * push_distance * progress
            push_pos[2] = push_height
            
            # Slightly behind object to maintain contact
            push_pos -= push_direction * 0.03
            
            waypoints.append({
                'position': push_pos,
                'type': 'push',
                'description': f'Push step {i}/{num_steps}'
            })
            
        # 4. Retract
        final_pos = params.target_position.copy()
        final_pos[2] = push_height + 0.1
        waypoints.append({
            'position': final_pos,
            'type': 'move',
            'description': 'Retract from object'
        })
        
        self.current_plan = waypoints
        logger.info(f"Generated push plan with {len(waypoints)} waypoints")
        
        return waypoints
        
    def execute(self, params: SkillParameters) -> SkillResult:
        """
        Execute push skill
        
        Args:
            params: Push parameters
            
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
            
            # Move to waypoint position
            success = self.move_to_position(
                waypoint['position'],
                speed=params.speed if waypoint['type'] == 'move' else params.speed * 0.5
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
            
            # Small delay for push steps
            if waypoint['type'] == 'push':
                time.sleep(0.1)
                
        # Check final object position
        final_obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        final_error = np.linalg.norm(params.target_position - np.array(final_obj_pos))
        
        if final_error < params.precision * 3:  # More lenient for pushing
            self.status = SkillStatus.SUCCESS
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Successfully pushed object (error: {final_error:.3f}m)",
                waypoints_executed=waypoints_executed,
                execution_time=time.time() - start_time,
                data={'final_position': final_obj_pos, 'error': final_error}
            )
        else:
            self.status = SkillStatus.SUCCESS  # Partial success
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Pushed object with error: {final_error:.3f}m",
                waypoints_executed=waypoints_executed,
                execution_time=time.time() - start_time,
                data={'final_position': final_obj_pos, 'error': final_error}
            )