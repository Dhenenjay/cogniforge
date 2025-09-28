"""
Safe grasp execution with contact queries and vertical lift strategy.

This module implements:
1. Contact queries to avoid table penetration
2. Small vertical lift before lateral movements
3. Safe approach and retreat strategies
4. Collision detection and avoidance
"""

import numpy as np
import time
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    # Mock PyBullet for testing without it
    class MockPyBullet:
        def getLinkState(self, *args, **kwargs):
            return [(0, 0, 0.5), (0, 0, 0, 1)]  # Mock position and orientation
        def getContactPoints(self, *args, **kwargs):
            return []  # No contacts
    p = MockPyBullet()

logger = logging.getLogger(__name__)

# Safety constants
VERTICAL_LIFT_HEIGHT = 0.02  # 2cm lift before lateral moves
MIN_TABLE_CLEARANCE = 0.005  # 5mm minimum clearance from table
CONTACT_CHECK_DISTANCE = 0.01  # 10mm contact detection threshold
APPROACH_SPEED = 0.05  # 5cm/s approach speed
LIFT_SPEED = 0.1  # 10cm/s lifting speed


class ContactState(Enum):
    """Contact detection states."""
    NO_CONTACT = "no_contact"
    NEAR_CONTACT = "near_contact"
    IN_CONTACT = "in_contact"
    EXCESSIVE_FORCE = "excessive_force"


@dataclass
class ContactInfo:
    """Information about detected contacts."""
    has_contact: bool
    contact_points: List[Tuple[float, float, float]]
    contact_normals: List[Tuple[float, float, float]]
    contact_forces: List[float]
    bodies_in_contact: List[int]
    distance_to_nearest: float
    state: ContactState
    
    def is_safe(self) -> bool:
        """Check if contact state is safe for operation."""
        return self.state in [ContactState.NO_CONTACT, ContactState.NEAR_CONTACT]
    
    def get_max_force(self) -> float:
        """Get maximum contact force."""
        return max(self.contact_forces) if self.contact_forces else 0.0


@dataclass
class SafeGraspConfig:
    """Configuration for safe grasp execution."""
    # Vertical lift parameters
    vertical_lift_height: float = VERTICAL_LIFT_HEIGHT
    min_table_clearance: float = MIN_TABLE_CLEARANCE
    use_vertical_lift: bool = True
    
    # Contact detection parameters
    enable_contact_queries: bool = True
    contact_check_distance: float = CONTACT_CHECK_DISTANCE
    max_contact_force: float = 10.0  # Newtons
    contact_check_frequency: int = 10  # Check every N simulation steps
    
    # Movement parameters
    approach_speed: float = APPROACH_SPEED
    lift_speed: float = LIFT_SPEED
    lateral_speed: float = 0.08  # 8cm/s for lateral moves
    
    # Safety parameters
    table_height: float = 0.0  # Z-coordinate of table surface
    workspace_bounds: Tuple[float, float, float, float, float, float] = (
        -1.0, -1.0, 0.0,  # min x, y, z
        1.0, 1.0, 1.5      # max x, y, z
    )
    
    # Grasp parameters
    approach_height: float = 0.1  # Height above object for approach
    grasp_depth: float = 0.02  # How far below object center to grasp
    post_grasp_lift: float = 0.15  # Lift height after grasping


class SafeGraspExecutor:
    """
    Safe grasp executor with contact detection and vertical lift strategy.
    
    Key features:
    1. Contact queries before and during movements
    2. Automatic vertical lift before lateral movements
    3. Table collision avoidance
    4. Force monitoring during grasp
    """
    
    def __init__(
        self,
        robot_id: int,
        end_effector_link: int,
        table_id: Optional[int] = None,
        config: Optional[SafeGraspConfig] = None
    ):
        """
        Initialize safe grasp executor.
        
        Args:
            robot_id: PyBullet robot body ID
            end_effector_link: Link index for end-effector
            table_id: PyBullet table body ID (for contact checks)
            config: Safety configuration
        """
        self.robot_id = robot_id
        self.ee_link = end_effector_link
        self.table_id = table_id
        self.config = config or SafeGraspConfig()
        
        # Track current state
        self.current_contacts = ContactInfo(
            has_contact=False,
            contact_points=[],
            contact_normals=[],
            contact_forces=[],
            bodies_in_contact=[],
            distance_to_nearest=float('inf'),
            state=ContactState.NO_CONTACT
        )
        
        # Movement history for debugging
        self.movement_history = []
        self.contact_history = []
        
        logger.info(
            f"Safe grasp executor initialized with vertical_lift={config.vertical_lift_height}m, "
            f"contact_queries={'enabled' if config.enable_contact_queries else 'disabled'}"
        )
    
    def check_contacts(
        self,
        check_table: bool = True,
        check_all: bool = False
    ) -> ContactInfo:
        """
        Check for contacts using PyBullet contact queries.
        
        Args:
            check_table: Whether to specifically check table contacts
            check_all: Whether to check contacts with all bodies
            
        Returns:
            ContactInfo with detected contacts
        """
        if not self.config.enable_contact_queries:
            return ContactInfo(
                has_contact=False,
                contact_points=[],
                contact_normals=[],
                contact_forces=[],
                bodies_in_contact=[],
                distance_to_nearest=float('inf'),
                state=ContactState.NO_CONTACT
            )
        
        contact_points = []
        contact_normals = []
        contact_forces = []
        bodies_in_contact = []
        
        # Get end-effector position for distance calculations
        ee_state = p.getLinkState(self.robot_id, self.ee_link)
        ee_pos = np.array(ee_state[0])
        
        # Check contacts with table
        if check_table and self.table_id is not None:
            table_contacts = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.table_id,
                linkIndexA=self.ee_link
            )
            
            for contact in table_contacts:
                contact_points.append(contact[5])  # Position on B
                contact_normals.append(contact[7])  # Contact normal
                contact_forces.append(contact[9])  # Normal force
                bodies_in_contact.append(self.table_id)
        
        # Check contacts with all bodies if requested
        if check_all:
            all_contacts = p.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=self.ee_link
            )
            
            for contact in all_contacts:
                if contact[2] != self.table_id:  # Skip table if already checked
                    contact_points.append(contact[5])
                    contact_normals.append(contact[7])
                    contact_forces.append(contact[9])
                    bodies_in_contact.append(contact[2])
        
        # Calculate closest distance to table if no contact
        distance_to_table = float('inf')
        if self.table_id is not None and not contact_points:
            # Approximate distance to table surface
            distance_to_table = ee_pos[2] - self.config.table_height
        
        # Determine contact state
        has_contact = len(contact_points) > 0
        max_force = max(contact_forces) if contact_forces else 0
        
        if has_contact and max_force > self.config.max_contact_force:
            state = ContactState.EXCESSIVE_FORCE
        elif has_contact:
            state = ContactState.IN_CONTACT
        elif distance_to_table < self.config.contact_check_distance:
            state = ContactState.NEAR_CONTACT
        else:
            state = ContactState.NO_CONTACT
        
        contact_info = ContactInfo(
            has_contact=has_contact,
            contact_points=contact_points,
            contact_normals=contact_normals,
            contact_forces=contact_forces,
            bodies_in_contact=bodies_in_contact,
            distance_to_nearest=min(distance_to_table, self.config.contact_check_distance),
            state=state
        )
        
        self.current_contacts = contact_info
        self.contact_history.append({
            'time': time.time(),
            'state': state.value,
            'max_force': max_force,
            'distance': distance_to_table
        })
        
        return contact_info
    
    def move_with_vertical_lift(
        self,
        start_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        skip_lift: bool = False
    ) -> Dict[str, Any]:
        """
        Move from start to target with vertical lift to avoid collisions.
        
        Strategy:
        1. Lift vertically from start position
        2. Move laterally at safe height
        3. Descend vertically to target
        
        Args:
            start_pos: Starting position (x, y, z)
            target_pos: Target position (x, y, z)
            skip_lift: Skip vertical lift (e.g., if already at safe height)
            
        Returns:
            Movement result with success status and trajectory
        """
        trajectory = [start_pos]
        
        # Check if lateral movement is needed
        lateral_distance = np.sqrt(
            (target_pos[0] - start_pos[0])**2 + 
            (target_pos[1] - start_pos[1])**2
        )
        
        if lateral_distance < 0.001:  # Less than 1mm, just vertical movement
            skip_lift = True
        
        if not skip_lift and self.config.use_vertical_lift:
            # Step 1: Lift vertically
            lift_height = max(
                start_pos[2] + self.config.vertical_lift_height,
                self.config.table_height + self.config.min_table_clearance + self.config.vertical_lift_height
            )
            
            lift_pos = (start_pos[0], start_pos[1], lift_height)
            
            logger.debug(f"Lifting from {start_pos} to {lift_pos}")
            lift_result = self._move_to_position(lift_pos, check_contacts=True)
            
            if not lift_result['success']:
                return {
                    'success': False,
                    'reason': 'Vertical lift failed',
                    'trajectory': trajectory,
                    'contact_detected': lift_result.get('contact_detected', False)
                }
            
            trajectory.append(lift_pos)
            
            # Step 2: Lateral movement at safe height
            safe_lateral_pos = (target_pos[0], target_pos[1], lift_height)
            
            logger.debug(f"Lateral move to {safe_lateral_pos}")
            lateral_result = self._move_to_position(safe_lateral_pos, check_contacts=True)
            
            if not lateral_result['success']:
                return {
                    'success': False,
                    'reason': 'Lateral movement failed',
                    'trajectory': trajectory,
                    'contact_detected': lateral_result.get('contact_detected', False)
                }
            
            trajectory.append(safe_lateral_pos)
            
            # Step 3: Descend to target
            logger.debug(f"Descending to target {target_pos}")
            descent_result = self._move_to_position(target_pos, check_contacts=True)
            
            if not descent_result['success']:
                return {
                    'success': False,
                    'reason': 'Descent to target failed',
                    'trajectory': trajectory,
                    'contact_detected': descent_result.get('contact_detected', False)
                }
            
            trajectory.append(target_pos)
        else:
            # Direct movement without lift
            logger.debug(f"Direct move from {start_pos} to {target_pos}")
            move_result = self._move_to_position(target_pos, check_contacts=True)
            
            if not move_result['success']:
                return {
                    'success': False,
                    'reason': 'Direct movement failed',
                    'trajectory': trajectory,
                    'contact_detected': move_result.get('contact_detected', False)
                }
            
            trajectory.append(target_pos)
        
        return {
            'success': True,
            'trajectory': trajectory,
            'total_distance': self._calculate_trajectory_length(trajectory),
            'contact_events': len([h for h in self.contact_history if h['state'] != 'no_contact'])
        }
    
    def safe_approach_grasp(
        self,
        object_position: Tuple[float, float, float],
        approach_direction: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Safely approach and grasp object with contact monitoring.
        
        Args:
            object_position: Target object position
            approach_direction: Approach direction vector (default: from above)
            
        Returns:
            Grasp result with success status and diagnostics
        """
        if approach_direction is None:
            approach_direction = (0, 0, -1)  # Approach from above
        
        # Normalize approach direction
        approach_dir = np.array(approach_direction)
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        
        # Calculate approach position (above object)
        approach_pos = (
            object_position[0] - approach_dir[0] * self.config.approach_height,
            object_position[1] - approach_dir[1] * self.config.approach_height,
            object_position[2] - approach_dir[2] * self.config.approach_height
        )
        
        # Ensure approach position is above table
        approach_pos = (
            approach_pos[0],
            approach_pos[1],
            max(approach_pos[2], self.config.table_height + self.config.min_table_clearance)
        )
        
        logger.info(f"Starting safe approach to {object_position}")
        
        # Get current position
        ee_state = p.getLinkState(self.robot_id, self.ee_link)
        current_pos = ee_state[0]
        
        # Move to approach position with vertical lift
        approach_result = self.move_with_vertical_lift(
            current_pos,
            approach_pos,
            skip_lift=False
        )
        
        if not approach_result['success']:
            return {
                'success': False,
                'phase': 'approach',
                'reason': approach_result['reason'],
                'diagnostics': approach_result
            }
        
        # Slow descent to grasp position with contact monitoring
        grasp_pos = (
            object_position[0],
            object_position[1],
            object_position[2] - self.config.grasp_depth
        )
        
        # Ensure we don't go below table
        grasp_pos = (
            grasp_pos[0],
            grasp_pos[1],
            max(grasp_pos[2], self.config.table_height + 0.001)  # 1mm above table
        )
        
        logger.info("Descending for grasp with contact monitoring")
        
        # Descend with frequent contact checks
        descent_result = self._careful_descent(approach_pos, grasp_pos)
        
        if not descent_result['success']:
            return {
                'success': False,
                'phase': 'grasp_descent',
                'reason': descent_result['reason'],
                'diagnostics': descent_result
            }
        
        # Record final contact state
        final_contacts = self.check_contacts(check_table=True, check_all=True)
        
        return {
            'success': True,
            'phase': 'grasp_ready',
            'final_position': grasp_pos,
            'approach_trajectory': approach_result['trajectory'],
            'contact_state': final_contacts.state.value,
            'min_table_clearance': grasp_pos[2] - self.config.table_height
        }
    
    def safe_lift_and_retreat(
        self,
        current_pos: Optional[Tuple[float, float, float]] = None,
        retreat_pos: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Safely lift grasped object and retreat to safe position.
        
        Args:
            current_pos: Current position (auto-detect if None)
            retreat_pos: Target retreat position (default: lift straight up)
            
        Returns:
            Retreat result with success status
        """
        # Get current position if not provided
        if current_pos is None:
            ee_state = p.getLinkState(self.robot_id, self.ee_link)
            current_pos = ee_state[0]
        
        # Default retreat: lift straight up
        if retreat_pos is None:
            retreat_pos = (
                current_pos[0],
                current_pos[1],
                current_pos[2] + self.config.post_grasp_lift
            )
        
        logger.info(f"Lifting object from {current_pos}")
        
        # First, lift straight up to clear table
        initial_lift_pos = (
            current_pos[0],
            current_pos[1],
            max(
                current_pos[2] + self.config.vertical_lift_height,
                self.config.table_height + self.config.min_table_clearance + 0.05  # 5cm clearance
            )
        )
        
        # Lift with contact monitoring
        lift_result = self._move_to_position(initial_lift_pos, check_contacts=True)
        
        if not lift_result['success']:
            logger.warning("Initial lift failed, attempting recovery")
            # Try smaller lift
            recovery_pos = (
                current_pos[0],
                current_pos[1],
                current_pos[2] + self.config.vertical_lift_height / 2
            )
            lift_result = self._move_to_position(recovery_pos, check_contacts=True)
            
            if not lift_result['success']:
                return {
                    'success': False,
                    'phase': 'lift',
                    'reason': 'Failed to lift object',
                    'diagnostics': lift_result
                }
        
        # Now move to final retreat position
        retreat_result = self.move_with_vertical_lift(
            initial_lift_pos,
            retreat_pos,
            skip_lift=True  # Already at safe height
        )
        
        if not retreat_result['success']:
            return {
                'success': False,
                'phase': 'retreat',
                'reason': retreat_result['reason'],
                'diagnostics': retreat_result
            }
        
        return {
            'success': True,
            'phase': 'completed',
            'final_position': retreat_pos,
            'lift_trajectory': retreat_result['trajectory'],
            'total_height_gained': retreat_pos[2] - current_pos[2]
        }
    
    def _move_to_position(
        self,
        target_pos: Tuple[float, float, float],
        check_contacts: bool = True
    ) -> Dict[str, Any]:
        """
        Move to target position with optional contact checking.
        
        This is a placeholder for actual robot movement command.
        In practice, this would interface with your robot controller.
        """
        # Check for contacts before move
        if check_contacts:
            pre_contacts = self.check_contacts(check_table=True)
            if pre_contacts.state == ContactState.EXCESSIVE_FORCE:
                return {
                    'success': False,
                    'reason': 'Excessive contact force detected',
                    'contact_force': pre_contacts.get_max_force()
                }
        
        # TODO: Actual robot movement implementation
        # For now, just return success
        self.movement_history.append({
            'time': time.time(),
            'target': target_pos,
            'contact_state': self.current_contacts.state.value
        })
        
        return {'success': True}
    
    def _careful_descent(
        self,
        start_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        steps: int = 10
    ) -> Dict[str, Any]:
        """
        Carefully descend with frequent contact checks.
        
        Args:
            start_pos: Starting position
            target_pos: Target position
            steps: Number of intermediate steps
            
        Returns:
            Descent result
        """
        positions = []
        for i in range(steps + 1):
            alpha = i / steps
            pos = (
                start_pos[0] + alpha * (target_pos[0] - start_pos[0]),
                start_pos[1] + alpha * (target_pos[1] - start_pos[1]),
                start_pos[2] + alpha * (target_pos[2] - start_pos[2])
            )
            positions.append(pos)
        
        for i, pos in enumerate(positions[1:], 1):
            # Check contacts
            contacts = self.check_contacts(check_table=True)
            
            # Stop if we detect unexpected contact
            if contacts.state == ContactState.IN_CONTACT and i < steps:
                logger.warning(f"Early contact detected at step {i}/{steps}")
                return {
                    'success': True,  # Partial success
                    'reason': 'Early contact detected',
                    'final_position': positions[i-1],
                    'contact_step': i
                }
            
            # Move to next position
            result = self._move_to_position(pos, check_contacts=False)
            if not result['success']:
                return result
        
        return {
            'success': True,
            'final_position': target_pos
        }
    
    def _calculate_trajectory_length(self, trajectory: List[Tuple]) -> float:
        """Calculate total length of trajectory."""
        if len(trajectory) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(trajectory)):
            p1 = np.array(trajectory[i-1])
            p2 = np.array(trajectory[i])
            total += np.linalg.norm(p2 - p1)
        
        return total


def create_safe_grasp_executor(
    robot_id: int,
    end_effector_link: int,
    table_id: Optional[int] = None,
    table_height: float = 0.0,
    enable_contact_queries: bool = True,
    vertical_lift_height: float = VERTICAL_LIFT_HEIGHT
) -> SafeGraspExecutor:
    """
    Factory function to create a safe grasp executor.
    
    Args:
        robot_id: PyBullet robot body ID
        end_effector_link: End-effector link index
        table_id: Table body ID for collision checking
        table_height: Height of table surface
        enable_contact_queries: Whether to use contact detection
        vertical_lift_height: Height for vertical lift movements
        
    Returns:
        Configured SafeGraspExecutor instance
    """
    config = SafeGraspConfig(
        table_height=table_height,
        enable_contact_queries=enable_contact_queries,
        vertical_lift_height=vertical_lift_height,
        use_vertical_lift=True
    )
    
    executor = SafeGraspExecutor(
        robot_id=robot_id,
        end_effector_link=end_effector_link,
        table_id=table_id,
        config=config
    )
    
    logger.info(
        f"Created safe grasp executor with lift_height={vertical_lift_height}m, "
        f"table_height={table_height}m, contact_queries={enable_contact_queries}"
    )
    
    return executor