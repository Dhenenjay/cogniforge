# Safe Grasp Execution Implementation

## Overview
This implementation provides a robust grasp execution system that uses **contact queries to avoid table penetration** and adds **vertical lift before lateral movements** for safe manipulation.

## Key Features Implemented

### 1. Contact Queries for Table Avoidance
- **PyBullet contact detection** to check for collisions before movements
- **4 contact states**: NO_CONTACT, NEAR_CONTACT, IN_CONTACT, EXCESSIVE_FORCE  
- **Real-time monitoring** of contact forces and distances
- **Automatic stopping** when unexpected contact is detected

### 2. Vertical Lift Strategy
- **2cm default vertical lift** (`VERTICAL_LIFT_HEIGHT = 0.02m`) before any lateral movement
- **Automatic trajectory planning** with 3 phases:
  1. **Lift vertically** from current position
  2. **Move laterally** at safe height
  3. **Descend vertically** to target
- **Configurable lift height** for different scenarios

### 3. Safety Parameters
- **Minimum table clearance**: 5mm (`MIN_TABLE_CLEARANCE = 0.005m`)
- **Contact detection threshold**: 10mm (`CONTACT_CHECK_DISTANCE = 0.01m`)
- **Maximum contact force**: 10N before triggering safety stop
- **Table height tracking** to prevent penetration

## Core Components

### `SafeGraspExecutor` Class
Main executor that orchestrates safe grasping with:
- Contact query methods
- Vertical lift movement planning
- Safe approach and retreat strategies
- Movement history tracking

### `ContactInfo` Dataclass
Stores contact detection information:
```python
@dataclass
class ContactInfo:
    has_contact: bool
    contact_points: List[Tuple[float, float, float]]
    contact_normals: List[Tuple[float, float, float]]
    contact_forces: List[float]
    bodies_in_contact: List[int]
    distance_to_nearest: float
    state: ContactState
```

### `SafeGraspConfig` Dataclass
Configuration parameters:
```python
@dataclass
class SafeGraspConfig:
    vertical_lift_height: float = 0.02  # 2cm
    min_table_clearance: float = 0.005  # 5mm
    enable_contact_queries: bool = True
    table_height: float = 0.0
    max_contact_force: float = 10.0  # Newtons
```

## Key Methods

### `check_contacts()`
- Queries PyBullet for contact points
- Calculates distance to table surface
- Determines contact state (safe/warning/contact/excessive)
- Returns `ContactInfo` with all details

### `move_with_vertical_lift()`
- Implements the 3-phase movement strategy
- Automatically adds vertical lift for lateral movements
- Tracks trajectory and contact events
- Returns success status with full trajectory

### `safe_approach_grasp()`
- Approaches object from specified direction
- Maintains minimum table clearance
- Uses contact monitoring during descent
- Stops early if unexpected contact detected

### `safe_lift_and_retreat()`
- Lifts object with initial vertical clearance
- Moves to retreat position at safe height
- Includes recovery strategy if initial lift fails
- Returns total height gained

## Usage Example

```python
from cogniforge.control.safe_grasp_execution import create_safe_grasp_executor

# Create executor with safety features
executor = create_safe_grasp_executor(
    robot_id=robot_id,
    end_effector_link=ee_link,
    table_id=table_id,
    table_height=0.4,  # 40cm table height
    enable_contact_queries=True,
    vertical_lift_height=0.02  # 2cm lift
)

# Safe approach to object
approach_result = executor.safe_approach_grasp(
    object_position=(0.5, 0.1, 0.42),
    approach_direction=(0, 0, -1)  # From above
)

if approach_result['success']:
    print(f"Min table clearance: {approach_result['min_table_clearance']}m")
    
    # After grasping, safely lift and retreat
    retreat_result = executor.safe_lift_and_retreat()
    print(f"Total lift: {retreat_result['total_height_gained']}m")
```

## Movement Trajectory Example

For a movement from `(0.3, 0.0, 0.1)` to `(0.5, 0.2, 0.15)`:

1. **Start**: (0.3, 0.0, 0.1)
2. **Vertical Lift**: (0.3, 0.0, 0.12) - lifted by 2cm
3. **Lateral Move**: (0.5, 0.2, 0.12) - moved at safe height  
4. **Descent**: (0.5, 0.2, 0.15) - descended to target

Total trajectory length: ~0.28m (vs direct path of ~0.22m)

## Safety Features

### Table Penetration Prevention
- Positions below table height are **automatically blocked**
- Positions within 5mm of table trigger **warning state**
- Grasp positions are **clamped** to maintain minimum clearance

### Contact Force Monitoring  
- Forces exceeding 10N trigger **immediate stop**
- Contact history is **logged** for debugging
- Early contact during descent **stops movement**

### Workspace Validation
- All positions checked against **workspace bounds**
- Trajectory waypoints **validated** before execution
- Recovery strategies for **failed movements**

## File Locations

- **Implementation**: `cogniforge/control/safe_grasp_execution.py` (666 lines)
- **Demo**: `examples/safe_grasp_demo.py` (397 lines)
- **Test**: `test_safe_grasp.py` (198 lines)

## Testing Results

All tests passed successfully:
- ✅ Contact state management (4 states)
- ✅ Vertical lift of 2cm before lateral moves
- ✅ Minimum table clearance of 5mm enforced
- ✅ Trajectory calculation accurate
- ✅ Table penetration prevention working
- ✅ Configuration management functional
- ✅ Factory function creation successful

## Benefits

1. **Safety**: Prevents robot from colliding with table or applying excessive force
2. **Reliability**: Consistent movement patterns reduce failure rates
3. **Debugging**: Contact and movement history for troubleshooting
4. **Flexibility**: Configurable parameters for different scenarios
5. **Recovery**: Automatic fallback strategies when movements fail

## Notes

- PyBullet is optional - system includes mock for testing without simulation
- Vertical lift adds ~27% to trajectory length but ensures safety
- Contact queries add minimal computational overhead
- System designed for real-time execution with robot controllers

This implementation ensures safe and reliable grasp execution by preventing table collisions and using proven movement strategies.