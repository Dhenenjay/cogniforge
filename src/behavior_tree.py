"""
Behavior Tree Implementation with Callable Nodes and Shared Blackboard

This module implements a behavior tree system where each action node
(MoveTo, Align, Grasp, Place) is a callable with access to a shared
blackboard for inter-node communication.
"""

import time
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Node Status Enum
# ============================================================================

class NodeStatus(Enum):
    """Status codes returned by behavior tree nodes"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"
    INVALID = "INVALID"


# ============================================================================
# Blackboard Implementation
# ============================================================================

class Blackboard:
    """
    Shared memory space for behavior tree nodes.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self):
        """Initialize the blackboard with default values"""
        self._data = {
            # Task information
            'task_type': None,
            'task_description': '',
            
            # Waypoints and trajectories
            'waypoints': [],
            'current_waypoint_index': 0,
            'trajectory_metadata': {},
            
            # Current targets
            'current_target': None,
            'grasp_target': None,
            'place_target': None,
            
            # Robot state
            'robot_position': np.array([0.0, 0.0, 0.0]),
            'robot_orientation': np.array([0.0, 0.0, 0.0]),
            'gripper_state': 'open',  # 'open' or 'closed'
            'gripper_force': 0.0,
            
            # Scene information
            'detected_objects': {},
            'obstacles': [],
            'workspace_bounds': {
                'x': [-1.0, 1.0],
                'y': [-1.0, 1.0],
                'z': [0.0, 2.0]
            },
            
            # Execution state
            'execution_history': deque(maxlen=100),
            'current_node': None,
            'node_status': {},
            
            # Control parameters
            'move_speed': 0.1,  # m/s
            'rotation_speed': 0.5,  # rad/s
            'grasp_force': 10.0,  # Newtons
            'alignment_tolerance': 0.01,  # meters
            'position_tolerance': 0.005,  # meters
            
            # Safety parameters
            'emergency_stop': False,
            'collision_detected': False,
            'force_limit_exceeded': False,
            
            # Debug information
            'debug_mode': False,
            'visualization_enabled': False
        }
        self._lock = threading.RLock()
        self._observers = []
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe get operation"""
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Thread-safe set operation with observer notification"""
        with self._lock:
            old_value = self._data.get(key)
            self._data[key] = value
            
            # Notify observers of change
            for observer in self._observers:
                observer(key, old_value, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values atomically"""
        with self._lock:
            for key, value in updates.items():
                self.set(key, value)
    
    def add_observer(self, callback: Callable) -> None:
        """Add an observer for blackboard changes"""
        self._observers.append(callback)
    
    def remove_observer(self, callback: Callable) -> None:
        """Remove an observer"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def clear_history(self) -> None:
        """Clear execution history"""
        with self._lock:
            self._data['execution_history'].clear()
    
    def add_history(self, entry: Dict[str, Any]) -> None:
        """Add entry to execution history"""
        with self._lock:
            entry['timestamp'] = time.time()
            self._data['execution_history'].append(entry)
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current blackboard state"""
        with self._lock:
            return self._data.copy()


# ============================================================================
# Base Node Class
# ============================================================================

class BehaviorNode(ABC):
    """Abstract base class for all behavior tree nodes"""
    
    def __init__(self, name: str, blackboard: Blackboard):
        """
        Initialize behavior node
        
        Args:
            name: Node identifier
            blackboard: Shared blackboard instance
        """
        self.name = name
        self.blackboard = blackboard
        self.status = NodeStatus.INVALID
        self.start_time = None
        self.timeout = 30.0  # Default timeout in seconds
        
    def __call__(self) -> NodeStatus:
        """Make the node callable"""
        return self.execute()
    
    def execute(self) -> NodeStatus:
        """Execute the node with timing and logging"""
        self.start_time = time.time()
        
        # Log execution start
        logger.info(f"Executing node: {self.name}")
        self.blackboard.set('current_node', self.name)
        
        # Check preconditions
        if not self.check_preconditions():
            logger.warning(f"Preconditions not met for {self.name}")
            self.status = NodeStatus.FAILURE
            return self.status
        
        # Execute main logic
        try:
            self.status = self.run()
            
            # Check timeout
            if time.time() - self.start_time > self.timeout:
                logger.error(f"Node {self.name} timed out")
                self.status = NodeStatus.FAILURE
            
        except Exception as e:
            logger.error(f"Error in node {self.name}: {e}")
            self.status = NodeStatus.FAILURE
        
        # Log result
        self.blackboard.add_history({
            'node': self.name,
            'status': self.status.value,
            'duration': time.time() - self.start_time
        })
        
        # Update blackboard status
        node_status = self.blackboard.get('node_status', {})
        node_status[self.name] = self.status.value
        self.blackboard.set('node_status', node_status)
        
        return self.status
    
    @abstractmethod
    def run(self) -> NodeStatus:
        """Main execution logic - must be implemented by subclasses"""
        pass
    
    def check_preconditions(self) -> bool:
        """Check if preconditions are met for execution"""
        # Check emergency stop
        if self.blackboard.get('emergency_stop'):
            logger.error("Emergency stop activated")
            return False
        
        # Check collision
        if self.blackboard.get('collision_detected'):
            logger.error("Collision detected")
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset node state"""
        self.status = NodeStatus.INVALID
        self.start_time = None


# ============================================================================
# Action Nodes
# ============================================================================

class MoveTo(BehaviorNode):
    """
    Move robot to target position
    """
    
    def __init__(self, blackboard: Blackboard, target_key: str = 'current_target'):
        """
        Initialize MoveTo node
        
        Args:
            blackboard: Shared blackboard
            target_key: Key in blackboard containing target position
        """
        super().__init__(f"MoveTo[{target_key}]", blackboard)
        self.target_key = target_key
        self.interpolation_steps = 10
        
    def run(self) -> NodeStatus:
        """Execute movement to target"""
        # Get target position
        target = self.blackboard.get(self.target_key)
        if target is None:
            logger.error(f"No target found at key: {self.target_key}")
            return NodeStatus.FAILURE
        
        # Convert to numpy array
        target_pos = np.array(target[:3]) if len(target) >= 3 else np.array(target)
        
        # Get current position
        current_pos = self.blackboard.get('robot_position')
        
        # Check if already at target
        distance = np.linalg.norm(target_pos - current_pos)
        tolerance = self.blackboard.get('position_tolerance', 0.005)
        
        if distance < tolerance:
            logger.info(f"Already at target position (distance: {distance:.4f}m)")
            return NodeStatus.SUCCESS
        
        # Check workspace bounds
        if not self._check_workspace_bounds(target_pos):
            logger.error(f"Target position {target_pos} outside workspace bounds")
            return NodeStatus.FAILURE
        
        # Plan interpolated path
        path = self._interpolate_path(current_pos, target_pos)
        
        # Execute movement along path
        for i, waypoint in enumerate(path):
            # Check for interrupts
            if self.blackboard.get('emergency_stop'):
                return NodeStatus.FAILURE
            
            # Simulate movement (replace with actual robot control)
            self._simulate_movement(waypoint)
            
            # Update blackboard
            self.blackboard.set('robot_position', waypoint)
            
            # Log progress
            progress = (i + 1) / len(path) * 100
            if i % 2 == 0:  # Log every other step
                logger.debug(f"Movement progress: {progress:.1f}%")
        
        # Verify final position
        final_pos = self.blackboard.get('robot_position')
        final_distance = np.linalg.norm(target_pos - final_pos)
        
        if final_distance < tolerance:
            logger.info(f"Successfully moved to target (error: {final_distance:.4f}m)")
            return NodeStatus.SUCCESS
        else:
            logger.error(f"Failed to reach target (error: {final_distance:.4f}m)")
            return NodeStatus.FAILURE
    
    def _check_workspace_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds"""
        bounds = self.blackboard.get('workspace_bounds')
        
        return (bounds['x'][0] <= position[0] <= bounds['x'][1] and
                bounds['y'][0] <= position[1] <= bounds['y'][1] and
                bounds['z'][0] <= position[2] <= bounds['z'][1])
    
    def _interpolate_path(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """Generate interpolated path from start to end"""
        steps = self.interpolation_steps
        path = []
        
        for i in range(steps + 1):
            t = i / steps
            waypoint = start * (1 - t) + end * t
            path.append(waypoint)
        
        return path
    
    def _simulate_movement(self, position: np.ndarray) -> None:
        """Simulate robot movement (replace with actual control)"""
        speed = self.blackboard.get('move_speed', 0.1)
        time.sleep(0.1)  # Simulate movement delay


class Align(BehaviorNode):
    """
    Align robot end-effector with target orientation
    """
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("Align", blackboard)
        self.alignment_axes = ['x', 'y', 'z']
        
    def run(self) -> NodeStatus:
        """Execute alignment with target"""
        # Get grasp target
        grasp_target = self.blackboard.get('grasp_target')
        if grasp_target is None:
            logger.error("No grasp target for alignment")
            return NodeStatus.FAILURE
        
        # Extract target orientation (if provided)
        target_orientation = None
        if isinstance(grasp_target, dict):
            target_orientation = grasp_target.get('orientation')
        elif len(grasp_target) > 3:
            target_orientation = np.array(grasp_target[3:6])
        
        if target_orientation is None:
            # Default alignment (pointing down for grasping)
            target_orientation = np.array([0, 0, -np.pi/2])
        
        # Get current orientation
        current_orientation = self.blackboard.get('robot_orientation')
        
        # Calculate alignment error
        orientation_error = target_orientation - current_orientation
        
        # Normalize angles to [-pi, pi]
        orientation_error = np.array([self._normalize_angle(a) for a in orientation_error])
        
        # Check if already aligned
        tolerance = self.blackboard.get('alignment_tolerance', 0.01)
        if np.max(np.abs(orientation_error)) < tolerance:
            logger.info("Already aligned with target")
            return NodeStatus.SUCCESS
        
        # Perform alignment
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            
            # Interpolate orientation
            new_orientation = current_orientation + orientation_error * t
            
            # Update blackboard
            self.blackboard.set('robot_orientation', new_orientation)
            
            # Simulate rotation delay
            time.sleep(0.05)
        
        # Verify alignment
        final_orientation = self.blackboard.get('robot_orientation')
        final_error = target_orientation - final_orientation
        final_error = np.array([self._normalize_angle(a) for a in final_error])
        
        if np.max(np.abs(final_error)) < tolerance:
            logger.info(f"Successfully aligned (max error: {np.max(np.abs(final_error)):.4f} rad)")
            return NodeStatus.SUCCESS
        else:
            logger.error(f"Failed to align (max error: {np.max(np.abs(final_error)):.4f} rad)")
            return NodeStatus.FAILURE
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class Grasp(BehaviorNode):
    """
    Execute grasping action
    """
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("Grasp", blackboard)
        self.grasp_speed = 0.05  # m/s approach speed
        self.max_attempts = 3
        
    def run(self) -> NodeStatus:
        """Execute grasping sequence"""
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
        
        # Extract position
        if isinstance(grasp_target, dict):
            target_pos = np.array(grasp_target.get('position', [0, 0, 0]))
        else:
            target_pos = np.array(grasp_target[:3])
        
        # Move to pre-grasp position (slightly above target)
        pre_grasp_offset = np.array([0, 0, 0.05])  # 5cm above
        pre_grasp_pos = target_pos + pre_grasp_offset
        
        logger.info(f"Moving to pre-grasp position: {pre_grasp_pos}")
        
        # Update blackboard for MoveTo node
        self.blackboard.set('current_target', pre_grasp_pos)
        move_to_pre = MoveTo(self.blackboard, 'current_target')
        
        if move_to_pre() != NodeStatus.SUCCESS:
            logger.error("Failed to reach pre-grasp position")
            return NodeStatus.FAILURE
        
        # Approach target slowly
        logger.info("Approaching grasp target")
        approach_result = self._approach_target(target_pos)
        
        if not approach_result:
            logger.error("Failed to approach target")
            return NodeStatus.FAILURE
        
        # Close gripper
        logger.info("Closing gripper")
        grasp_success = self._execute_grasp()
        
        if not grasp_success:
            logger.error("Failed to grasp object")
            return NodeStatus.FAILURE
        
        # Lift slightly to confirm grasp
        lift_pos = self.blackboard.get('robot_position') + np.array([0, 0, 0.02])
        self.blackboard.set('current_target', lift_pos)
        
        move_lift = MoveTo(self.blackboard, 'current_target')
        if move_lift() != NodeStatus.SUCCESS:
            logger.warning("Failed to lift after grasp")
        
        # Verify grasp
        if self._verify_grasp():
            logger.info("Successfully grasped object")
            return NodeStatus.SUCCESS
        else:
            logger.error("Grasp verification failed")
            return NodeStatus.FAILURE
    
    def _approach_target(self, target_pos: np.ndarray) -> bool:
        """Slowly approach the target position"""
        current_pos = self.blackboard.get('robot_position')
        distance = np.linalg.norm(target_pos - current_pos)
        
        # Approach in small steps
        steps = max(int(distance / 0.01), 5)  # 1cm steps
        
        for i in range(steps + 1):
            t = i / steps
            intermediate_pos = current_pos * (1 - t) + target_pos * t
            
            # Check for collision or force feedback
            if self._check_contact():
                logger.info("Contact detected during approach")
                return True
            
            self.blackboard.set('robot_position', intermediate_pos)
            time.sleep(0.05)
        
        return True
    
    def _execute_grasp(self) -> bool:
        """Execute the grasping action"""
        for attempt in range(self.max_attempts):
            # Close gripper
            self._control_gripper('close')
            
            # Wait for gripper to close
            time.sleep(0.5)
            
            # Check gripper force
            force = self._measure_gripper_force()
            self.blackboard.set('gripper_force', force)
            
            # Check if object is grasped
            min_force = 2.0  # Minimum force to confirm grasp
            max_force = self.blackboard.get('grasp_force', 10.0)
            
            if min_force <= force <= max_force:
                return True
            elif force > max_force:
                logger.warning(f"Excessive force detected: {force}N")
                self._control_gripper('open')
                time.sleep(0.2)
            else:
                logger.warning(f"Insufficient grasp force: {force}N (attempt {attempt + 1})")
        
        return False
    
    def _verify_grasp(self) -> bool:
        """Verify that object is successfully grasped"""
        # Check gripper state
        if self.blackboard.get('gripper_state') != 'closed':
            return False
        
        # Check force feedback
        force = self.blackboard.get('gripper_force', 0)
        if force < 1.0:  # Minimum force threshold
            return False
        
        # Additional verification could include vision check
        return True
    
    def _control_gripper(self, action: str) -> None:
        """Control gripper open/close"""
        self.blackboard.set('gripper_state', action)
        logger.debug(f"Gripper {action}")
    
    def _measure_gripper_force(self) -> float:
        """Measure current gripper force (simulated)"""
        # Simulate force measurement
        if self.blackboard.get('gripper_state') == 'closed':
            # Random force between 3-8 N when closed
            return np.random.uniform(3.0, 8.0)
        return 0.0
    
    def _check_contact(self) -> bool:
        """Check for contact/collision (simulated)"""
        # In real implementation, check force/torque sensors
        return False


class Place(BehaviorNode):
    """
    Execute placing action
    """
    
    def __init__(self, blackboard: Blackboard):
        super().__init__("Place", blackboard)
        self.place_height_offset = 0.02  # 2cm above surface
        
    def run(self) -> NodeStatus:
        """Execute placing sequence"""
        # Verify object is grasped
        if self.blackboard.get('gripper_state') != 'closed':
            logger.error("No object grasped for placing")
            return NodeStatus.FAILURE
        
        # Get place target
        place_target = self.blackboard.get('place_target')
        if place_target is None:
            logger.error("No place target specified")
            return NodeStatus.FAILURE
        
        # Extract position
        if isinstance(place_target, dict):
            target_pos = np.array(place_target.get('position', [0, 0, 0]))
        else:
            target_pos = np.array(place_target[:3])
        
        # Move to pre-place position (above target)
        pre_place_pos = target_pos + np.array([0, 0, 0.1])  # 10cm above
        
        logger.info(f"Moving to pre-place position: {pre_place_pos}")
        self.blackboard.set('current_target', pre_place_pos)
        
        move_to_pre = MoveTo(self.blackboard, 'current_target')
        if move_to_pre() != NodeStatus.SUCCESS:
            logger.error("Failed to reach pre-place position")
            return NodeStatus.FAILURE
        
        # Lower to place position
        place_pos = target_pos + np.array([0, 0, self.place_height_offset])
        
        logger.info(f"Lowering to place position: {place_pos}")
        self.blackboard.set('current_target', place_pos)
        
        move_to_place = MoveTo(self.blackboard, 'current_target')
        if move_to_place() != NodeStatus.SUCCESS:
            logger.error("Failed to reach place position")
            return NodeStatus.FAILURE
        
        # Release object
        logger.info("Releasing object")
        self._control_gripper('open')
        
        # Wait for gripper to open
        time.sleep(0.5)
        
        # Retreat to safe position
        retreat_pos = self.blackboard.get('robot_position') + np.array([0, 0, 0.05])
        self.blackboard.set('current_target', retreat_pos)
        
        logger.info("Retreating to safe position")
        move_retreat = MoveTo(self.blackboard, 'current_target')
        move_retreat()  # Don't fail if retreat fails
        
        # Verify placement
        if self._verify_placement():
            logger.info("Successfully placed object")
            return NodeStatus.SUCCESS
        else:
            logger.error("Placement verification failed")
            return NodeStatus.FAILURE
    
    def _control_gripper(self, action: str) -> None:
        """Control gripper open/close"""
        self.blackboard.set('gripper_state', action)
        self.blackboard.set('gripper_force', 0.0 if action == 'open' else None)
        logger.debug(f"Gripper {action}")
    
    def _verify_placement(self) -> bool:
        """Verify successful placement"""
        # Check gripper is open
        if self.blackboard.get('gripper_state') != 'open':
            return False
        
        # Check force is released
        if self.blackboard.get('gripper_force', 0) > 0.5:
            return False
        
        # Additional verification could include vision check
        return True


# ============================================================================
# Composite Nodes
# ============================================================================

class Sequence(BehaviorNode):
    """
    Execute child nodes in sequence until one fails or all succeed
    """
    
    def __init__(self, name: str, blackboard: Blackboard, children: List[BehaviorNode]):
        super().__init__(name, blackboard)
        self.children = children
        
    def run(self) -> NodeStatus:
        """Execute children in sequence"""
        for child in self.children:
            status = child.execute()
            
            if status == NodeStatus.FAILURE:
                logger.info(f"Sequence {self.name} failed at {child.name}")
                return NodeStatus.FAILURE
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        
        logger.info(f"Sequence {self.name} completed successfully")
        return NodeStatus.SUCCESS
    
    def reset(self) -> None:
        """Reset all children"""
        super().reset()
        for child in self.children:
            child.reset()


class Selector(BehaviorNode):
    """
    Execute child nodes until one succeeds (fallback behavior)
    """
    
    def __init__(self, name: str, blackboard: Blackboard, children: List[BehaviorNode]):
        super().__init__(name, blackboard)
        self.children = children
        
    def run(self) -> NodeStatus:
        """Execute children until one succeeds"""
        for child in self.children:
            status = child.execute()
            
            if status == NodeStatus.SUCCESS:
                logger.info(f"Selector {self.name} succeeded with {child.name}")
                return NodeStatus.SUCCESS
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        
        logger.info(f"Selector {self.name} failed - all children failed")
        return NodeStatus.FAILURE


class Parallel(BehaviorNode):
    """
    Execute child nodes in parallel
    """
    
    def __init__(self, name: str, blackboard: Blackboard, children: List[BehaviorNode],
                 success_threshold: int = None):
        super().__init__(name, blackboard)
        self.children = children
        self.success_threshold = success_threshold or len(children)
        
    def run(self) -> NodeStatus:
        """Execute children in parallel"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.children)) as executor:
            futures = [executor.submit(child.execute) for child in self.children]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for r in results if r == NodeStatus.SUCCESS)
        
        if success_count >= self.success_threshold:
            logger.info(f"Parallel {self.name} succeeded ({success_count}/{len(self.children)})")
            return NodeStatus.SUCCESS
        else:
            logger.info(f"Parallel {self.name} failed ({success_count}/{len(self.children)})")
            return NodeStatus.FAILURE


# ============================================================================
# Decorator Nodes
# ============================================================================

class Retry(BehaviorNode):
    """
    Retry child node on failure
    """
    
    def __init__(self, blackboard: Blackboard, child: BehaviorNode, max_retries: int = 3):
        super().__init__(f"Retry[{child.name}]", blackboard)
        self.child = child
        self.max_retries = max_retries
        
    def run(self) -> NodeStatus:
        """Execute child with retries"""
        for attempt in range(self.max_retries):
            status = self.child.execute()
            
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
            
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying {self.child.name} (attempt {attempt + 2}/{self.max_retries})")
                self.child.reset()
                time.sleep(0.5)  # Brief delay between retries
        
        logger.error(f"Max retries exceeded for {self.child.name}")
        return NodeStatus.FAILURE


class Timeout(BehaviorNode):
    """
    Add timeout to child node
    """
    
    def __init__(self, blackboard: Blackboard, child: BehaviorNode, timeout: float):
        super().__init__(f"Timeout[{child.name}]", blackboard)
        self.child = child
        self.timeout_duration = timeout
        
    def run(self) -> NodeStatus:
        """Execute child with timeout"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.child.execute)
            
            try:
                status = future.result(timeout=self.timeout_duration)
                return status
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout exceeded for {self.child.name}")
                return NodeStatus.FAILURE


# ============================================================================
# Task Builder
# ============================================================================

class TaskBuilder:
    """
    Build behavior trees for specific tasks
    """
    
    @staticmethod
    def build_pick_and_place(blackboard: Blackboard) -> BehaviorNode:
        """
        Build a pick-and-place behavior tree
        
        Structure:
        - Sequence
            - MoveTo(grasp_target)
            - Align()
            - Grasp()
            - MoveTo(place_target)
            - Place()
        """
        
        # Create action nodes
        move_to_grasp = MoveTo(blackboard, 'grasp_target')
        align = Align(blackboard)
        grasp = Retry(blackboard, Grasp(blackboard), max_retries=3)
        move_to_place = MoveTo(blackboard, 'place_target')
        place = Place(blackboard)
        
        # Create sequence
        pick_place_sequence = Sequence(
            "PickAndPlace",
            blackboard,
            [move_to_grasp, align, grasp, move_to_place, place]
        )
        
        # Add timeout wrapper
        return Timeout(blackboard, pick_place_sequence, timeout=60.0)
    
    @staticmethod
    def build_multi_waypoint_task(blackboard: Blackboard) -> BehaviorNode:
        """
        Build a task that visits multiple waypoints
        """
        waypoints = blackboard.get('waypoints', [])
        
        if not waypoints:
            logger.error("No waypoints provided")
            return None
        
        # Create MoveTo nodes for each waypoint
        move_nodes = []
        for i, waypoint in enumerate(waypoints):
            # Update blackboard with waypoint
            blackboard.set(f'waypoint_{i}', waypoint)
            move_node = MoveTo(blackboard, f'waypoint_{i}')
            move_nodes.append(move_node)
        
        # Create sequence
        return Sequence("MultiWaypoint", blackboard, move_nodes)
    
    @staticmethod
    def build_with_fallback(blackboard: Blackboard) -> BehaviorNode:
        """
        Build a task with fallback strategies
        """
        
        # Primary strategy: GPT-generated trajectory
        primary = Sequence(
            "PrimaryStrategy",
            blackboard,
            [
                MoveTo(blackboard, 'gpt_target'),
                Align(blackboard),
                Grasp(blackboard)
            ]
        )
        
        # Fallback strategy: Simple 3-waypoint trajectory
        fallback = Sequence(
            "FallbackStrategy",
            blackboard,
            [
                MoveTo(blackboard, 'fallback_waypoint_1'),
                MoveTo(blackboard, 'fallback_waypoint_2'),
                MoveTo(blackboard, 'fallback_waypoint_3')
            ]
        )
        
        # Selector tries primary first, then fallback
        return Selector("TaskWithFallback", blackboard, [primary, fallback])


# ============================================================================
# Example Usage
# ============================================================================

def example_pick_and_place():
    """Example of pick-and-place task execution"""
    
    # Create blackboard
    blackboard = Blackboard()
    
    # Set up task parameters
    blackboard.update({
        'task_type': 'pick_and_place',
        'task_description': 'Pick red cube and place on platform',
        'grasp_target': np.array([0.4, 0.0, 0.1]),
        'place_target': np.array([0.6, 0.2, 0.1]),
        'robot_position': np.array([0.0, 0.0, 0.3]),
        'robot_orientation': np.array([0.0, 0.0, 0.0]),
        'gripper_state': 'open',
        'debug_mode': True
    })
    
    # Build behavior tree
    task_tree = TaskBuilder.build_pick_and_place(blackboard)
    
    # Execute task
    logger.info("=" * 50)
    logger.info("Starting Pick and Place Task")
    logger.info("=" * 50)
    
    status = task_tree()
    
    # Report results
    logger.info("=" * 50)
    logger.info(f"Task completed with status: {status.value}")
    logger.info(f"Final robot position: {blackboard.get('robot_position')}")
    logger.info(f"Final gripper state: {blackboard.get('gripper_state')}")
    logger.info("=" * 50)
    
    # Print execution history
    history = blackboard.get('execution_history')
    logger.info("\nExecution History:")
    for entry in history:
        logger.info(f"  {entry['node']}: {entry['status']} ({entry['duration']:.2f}s)")
    
    return status


def example_waypoint_following():
    """Example of waypoint following task"""
    
    # Create blackboard
    blackboard = Blackboard()
    
    # Define waypoints
    waypoints = [
        np.array([0.2, 0.0, 0.2]),
        np.array([0.4, 0.1, 0.3]),
        np.array([0.6, 0.0, 0.2]),
        np.array([0.4, -0.1, 0.1])
    ]
    
    # Set up task parameters
    blackboard.update({
        'task_type': 'waypoint_following',
        'waypoints': waypoints,
        'robot_position': np.array([0.0, 0.0, 0.3]),
        'robot_orientation': np.array([0.0, 0.0, 0.0]),
        'gripper_state': 'closed'
    })
    
    # Build behavior tree
    task_tree = TaskBuilder.build_multi_waypoint_task(blackboard)
    
    if task_tree:
        # Execute task
        logger.info("=" * 50)
        logger.info("Starting Waypoint Following Task")
        logger.info("=" * 50)
        
        status = task_tree()
        
        # Report results
        logger.info("=" * 50)
        logger.info(f"Task completed with status: {status.value}")
        logger.info(f"Final robot position: {blackboard.get('robot_position')}")
        logger.info("=" * 50)
        
        return status
    
    return NodeStatus.FAILURE


if __name__ == "__main__":
    # Run examples
    print("\n" + "="*60)
    print(" BEHAVIOR TREE DEMONSTRATION")
    print("="*60 + "\n")
    
    # Example 1: Pick and Place
    print("\n--- Example 1: Pick and Place ---")
    example_pick_and_place()
    
    # Example 2: Waypoint Following
    print("\n--- Example 2: Waypoint Following ---")
    example_waypoint_following()
    
    print("\n" + "="*60)
    print(" DEMONSTRATION COMPLETE")
    print("="*60 + "\n")