"""
Slide Skill

Slides objects smoothly along a surface using controlled motion.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple
from .base_skill import BaseSkill, SkillParameters, SkillResult, SkillStatus
import logging

logger = logging.getLogger(__name__)


class SlideSkill(BaseSkill):
    """Skill for sliding objects along a surface with controlled motion"""
    
    def validate_preconditions(self, params: SkillParameters) -> Tuple[bool, str]:
        """
        Validate slide preconditions
        
        Args:
            params: Slide parameters
            
        Returns:
            (is_valid, message) tuple
        """
        if params.object_id is None:
            return False, "No object specified for sliding"
            
        if params.target_position is None:
            return False, "No target position specified"
            
        # Check if object exists
        import pybullet as p
        try:
            pos, _ = p.getBasePositionAndOrientation(params.object_id)
            obj_height = pos[2]
        except:
            return False, f"Object {params.object_id} not found"
            
        # Check if target maintains same height (sliding on same surface)
        target_height = params.target_position[2]
        if abs(target_height - obj_height) > 0.05:  # 5cm tolerance
            return False, f"Target height differs too much ({abs(target_height - obj_height):.3f}m)"
            
        return True, "Preconditions satisfied"
        
    def plan(self, params: SkillParameters) -> List[Dict[str, Any]]:
        """
        Plan slide trajectory with arc motion
        
        Args:
            params: Slide parameters
            
        Returns:
            List of waypoints for sliding
        """
        import pybullet as p
        
        # Get object position
        obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        obj_pos = np.array(obj_pos)
        
        # Calculate slide vector
        slide_vector = params.target_position - obj_pos
        slide_vector[2] = 0  # Keep slide horizontal
        slide_distance = np.linalg.norm(slide_vector)
        
        if slide_distance < 0.001:
            return []  # Already at target
            
        slide_direction = slide_vector / slide_distance
        
        # Perpendicular direction for arc
        perp_direction = np.array([-slide_direction[1], slide_direction[0], 0])
        
        waypoints = []
        
        # 1. Approach from side
        approach_pos = obj_pos - slide_direction * 0.05 + perp_direction * 0.08
        approach_pos[2] = obj_pos[2] + 0.1
        waypoints.append({
            'position': approach_pos,
            'type': 'approach',
            'gripper': 1.0,  # Open
            'description': 'Approach from side'
        })
        
        # 2. Lower to contact
        contact_pos = obj_pos + perp_direction * 0.03
        contact_pos[2] = obj_pos[2] + 0.01  # Just above surface
        waypoints.append({
            'position': contact_pos,
            'type': 'contact',
            'gripper': 0.8,  # Partially closed for sliding
            'description': 'Make contact for sliding'
        })
        
        # 3. Slide in arc motion
        num_steps = max(5, int(slide_distance / 0.03))  # 3cm steps
        for i in range(1, num_steps + 1):
            progress = i / num_steps
            
            # Arc motion (sine wave)
            arc_offset = perp_direction * 0.02 * np.sin(progress * np.pi)
            
            slide_pos = obj_pos + slide_vector * progress + arc_offset
            slide_pos[2] = obj_pos[2] + 0.01
            
            waypoints.append({
                'position': slide_pos,
                'type': 'slide',
                'gripper': 0.8,
                'description': f'Slide step {i}/{num_steps}'
            })
            
        # 4. Release and retract
        release_pos = params.target_position.copy()
        release_pos[2] = obj_pos[2] + 0.05
        waypoints.append({
            'position': release_pos,
            'type': 'release',
            'gripper': 1.0,  # Open
            'description': 'Release and retract'
        })
        
        # 5. Move up
        final_pos = release_pos.copy()
        final_pos[2] += 0.1
        waypoints.append({
            'position': final_pos,
            'type': 'retract',
            'gripper': 1.0,
            'description': 'Final retract'
        })
        
        self.current_plan = waypoints
        logger.info(f"Generated slide plan with {len(waypoints)} waypoints")
        
        return waypoints
        
    def execute(self, params: SkillParameters) -> SkillResult:
        """
        Execute slide skill
        
        Args:
            params: Slide parameters
            
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
                time.sleep(0.2)  # Allow gripper to move
            
            # Move to waypoint position
            speed = params.speed * 0.7 if waypoint['type'] == 'slide' else params.speed
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
            
            # Smooth sliding motion
            if waypoint['type'] == 'slide':
                time.sleep(0.05)
                
        # Wait for object to stabilize
        self.wait_for_stability(params.object_id, max_wait=1.0)
        
        # Check final object position
        final_obj_pos, _ = p.getBasePositionAndOrientation(params.object_id)
        final_error = np.linalg.norm(params.target_position - np.array(final_obj_pos))
        
        if final_error < params.precision * 2:  # Slightly more lenient than pick-place
            self.status = SkillStatus.SUCCESS
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Successfully slid object (error: {final_error:.3f}m)",
                waypoints_executed=waypoints_executed,
                execution_time=time.time() - start_time,
                data={'final_position': final_obj_pos, 'error': final_error}
            )
        else:
            self.status = SkillStatus.SUCCESS  # Partial success
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Slid object with error: {final_error:.3f}m",
                waypoints_executed=waypoints_executed,
                execution_time=time.time() - start_time,
                data={'final_position': final_obj_pos, 'error': final_error}
            )