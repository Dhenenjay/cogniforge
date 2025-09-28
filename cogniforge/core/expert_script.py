"""
Expert script generation using Codex for parametric robot control.

This module generates expert manipulation scripts with waypoints and approach
vectors using OpenAI Codex or similar code-generation models.
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging

# Configure logging
logger = logging.getLogger(__name__)


def gen_expert_script(
    prompt: str,
    scene_summary: Dict[str, Any],
    use_parametric: bool = True,
    include_approach_vectors: bool = True,
    waypoint_density: str = "adaptive",
    codex_model: str = "code-davinci-002"
) -> str:
    """
    Generate expert script prompt for Codex to create parametric waypoint-based control.
    
    This function creates a specialized prompt for Codex that requests:
    - Parametric waypoint lists W = [(x,y,z), ...]
    - Approach vectors for each grasp/place operation
    - Smooth trajectory generation
    - Collision-aware path planning
    
    Args:
        prompt: Natural language task description
        scene_summary: Dictionary containing scene state and object positions
        use_parametric: Whether to request parametric (adjustable) waypoints
        include_approach_vectors: Whether to include approach vector calculation
        waypoint_density: Waypoint generation strategy ('sparse', 'dense', 'adaptive')
        codex_model: Target Codex model for generation
        
    Returns:
        Formatted prompt for Codex to generate expert script
        
    Example:
        scene = {
            'objects': [
                {'name': 'red_cube', 'position': [0.5, 0.0, 0.1], 'size': 0.05},
                {'name': 'blue_platform', 'position': [0.7, 0.2, 0.05]}
            ],
            'robot_state': {'ee_pos': [0.0, 0.0, 0.5]},
            'obstacles': [{'name': 'wall', 'bounds': [[0.8, -0.5, 0], [0.9, 0.5, 1.0]]}]
        }
        
        script_prompt = gen_expert_script(
            "Pick red cube and place on blue platform",
            scene
        )
        # Send to Codex API for code generation
    """
    
    # Format scene for clarity
    scene_description = _format_scene_for_codex(scene_summary)
    
    # Build the Codex prompt
    codex_prompt = f"""# Expert Robotic Manipulation Script Generator
# Task: {prompt}

{scene_description}

Generate a Python function that returns:
1. A list of waypoints W = [(x, y, z), ...] for the complete trajectory
2. Approach vectors for each pick/place operation
3. Gripper actions at each waypoint

Requirements:
- Use numpy arrays for vector calculations
- Include approach and retreat vectors for safe grasping
- Generate smooth, collision-free paths
- Make waypoints parametric (adjustable based on object positions)
- Include gripper state changes (open/close)

Output Format:
```python
import numpy as np
from typing import List, Tuple, Dict, Any

def generate_expert_trajectory(scene_objects: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    Generate expert trajectory with waypoints and approach vectors.
    
    Args:
        scene_objects: Dictionary with object names as keys and positions as values
        
    Returns:
        Dictionary containing:
        - 'waypoints': List of (x, y, z) tuples
        - 'approach_vectors': Dict mapping waypoint indices to approach vectors
        - 'gripper_actions': Dict mapping waypoint indices to 'open'/'close'
        - 'annotations': List of strings describing each waypoint purpose
    \"\"\"
    
    # Extract object positions (parametric)
    {_generate_object_extraction_code(scene_summary)}
    
    # Define approach vectors based on object geometry and scene
    {_generate_approach_vector_template(include_approach_vectors)}
    
    # Generate waypoint sequence
    W = []  # Waypoints list
    approach_vectors = {{}}  # Approach vectors for specific waypoints
    gripper_actions = {{}}  # Gripper commands at waypoints
    annotations = []  # Descriptions of each waypoint
    
    # TODO: Generate waypoint sequence here
    # Each waypoint should be a (x, y, z) tuple
    # Critical waypoints need approach vectors
    # Gripper actions at grasp/release points
    
    return {{
        'waypoints': W,
        'approach_vectors': approach_vectors,
        'gripper_actions': gripper_actions,
        'annotations': annotations,
        'metadata': {{
            'task': '{prompt}',
            'total_waypoints': len(W),
            'approach_height': approach_height,
            'grasp_force': grasp_force
        }}
    }}
```

IMPORTANT: Generate complete, executable code with actual waypoint calculations."""

    if use_parametric:
        codex_prompt += """

Make the trajectory parametric by:
1. Computing waypoints relative to object positions
2. Using configurable approach heights and offsets
3. Adapting to different object sizes
4. Scaling movements based on workspace bounds
"""

    if waypoint_density == "adaptive":
        codex_prompt += """

Use adaptive waypoint density:
- More waypoints near objects (fine control)
- Fewer waypoints in free space (efficiency)
- Extra waypoints for orientation changes
- Dense sampling around obstacles
"""
    elif waypoint_density == "dense":
        codex_prompt += "\nGenerate dense waypoints (small spacing ~0.05m) for smooth motion."
    else:  # sparse
        codex_prompt += "\nGenerate sparse waypoints (only critical points) for efficiency."

    return codex_prompt


def _format_scene_for_codex(scene_summary: Dict[str, Any]) -> str:
    """
    Format scene information for Codex understanding.
    
    Args:
        scene_summary: Scene dictionary
        
    Returns:
        Formatted scene description
    """
    lines = ["# Scene Configuration"]
    
    # Objects with positions
    if 'objects' in scene_summary:
        lines.append("# Objects in scene:")
        lines.append("scene_objects = {")
        for obj in scene_summary['objects']:
            name = obj.get('name', 'unknown')
            pos = obj.get('position', [0, 0, 0])
            size = obj.get('size', 0.05)
            graspable = obj.get('graspable', True)
            
            lines.append(f"    '{name}': {{")
            lines.append(f"        'position': {pos},")
            lines.append(f"        'size': {size},")
            lines.append(f"        'graspable': {graspable},")
            
            if 'orientation' in obj:
                lines.append(f"        'orientation': {obj['orientation']},")
            
            lines.append("    },")
        lines.append("}")
    
    # Robot state
    if 'robot_state' in scene_summary:
        lines.append("\n# Robot initial state:")
        robot = scene_summary['robot_state']
        if 'ee_pos' in robot:
            lines.append(f"initial_ee_pos = {robot['ee_pos']}")
        if 'gripper_state' in robot:
            lines.append(f"initial_gripper = '{robot['gripper_state']}'")
    
    # Obstacles
    if 'obstacles' in scene_summary:
        lines.append("\n# Obstacles to avoid:")
        lines.append("obstacles = [")
        for obs in scene_summary['obstacles']:
            lines.append(f"    {obs},")
        lines.append("]")
    
    # Workspace bounds
    if 'workspace' in scene_summary:
        ws = scene_summary['workspace']
        lines.append(f"\n# Workspace bounds:")
        lines.append(f"workspace = {{")
        lines.append(f"    'x': [{ws.get('x_min', -1)}, {ws.get('x_max', 1)}],")
        lines.append(f"    'y': [{ws.get('y_min', -1)}, {ws.get('y_max', 1)}],")
        lines.append(f"    'z': [{ws.get('z_min', 0)}, {ws.get('z_max', 2)}],")
        lines.append("}")
    
    return "\n".join(lines)


def _generate_object_extraction_code(scene_summary: Dict[str, Any]) -> str:
    """
    Generate code template for extracting object positions.
    
    Args:
        scene_summary: Scene information
        
    Returns:
        Code snippet for object extraction
    """
    code_lines = []
    
    if 'objects' in scene_summary:
        for obj in scene_summary['objects']:
            name = obj.get('name', 'unknown')
            var_name = name.replace(' ', '_').replace('-', '_')
            code_lines.append(f"{var_name}_pos = np.array(scene_objects['{name}']['position'])")
            code_lines.append(f"{var_name}_size = scene_objects['{name}'].get('size', 0.05)")
    
    return "\n    ".join(code_lines)


def _generate_approach_vector_template(include_approach: bool) -> str:
    """
    Generate approach vector calculation template.
    
    Args:
        include_approach: Whether to include approach vectors
        
    Returns:
        Code template for approach vectors
    """
    if not include_approach:
        return "# Approach vectors not requested"
    
    return """# Approach vector configuration
    approach_height = 0.15  # Height above object for approach
    retreat_height = 0.20   # Height for retreat after grasp
    grasp_force = 50.0      # Gripper force for grasping
    
    # Standard approach vectors (can be customized per object)
    vertical_approach = np.array([0, 0, -1])  # Top-down approach
    lateral_approach = np.array([1, 0, 0])     # Side approach
    
    def compute_approach_vector(obj_pos, obj_type='default'):
        \"\"\"Compute approach vector based on object position and type.\"\"\"
        if obj_type == 'flat':
            return vertical_approach
        elif obj_type == 'tall':
            return lateral_approach
        else:
            # Adaptive approach based on position
            to_object = obj_pos - initial_ee_pos
            to_object[2] = 0  # Project to horizontal plane
            if np.linalg.norm(to_object) > 0:
                return -to_object / np.linalg.norm(to_object)
            return vertical_approach"""


def parse_codex_response(
    response: str,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Parse Codex-generated expert script.
    
    Args:
        response: Raw Codex response containing Python code
        validate: Whether to validate the generated code
        
    Returns:
        Dictionary with parsed trajectory data
        
    Raises:
        ValueError: If response cannot be parsed or is invalid
    """
    import re
    
    # Extract Python code from response
    code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # Try to extract function definition directly
        func_match = re.search(r'def generate_expert_trajectory.*?return.*?\n}', 
                              response, re.DOTALL)
        if func_match:
            code = func_match.group(0)
        else:
            raise ValueError("No valid Python code found in response")
    
    # Validate if requested
    if validate:
        try:
            # Check for required function
            if 'def generate_expert_trajectory' not in code:
                raise ValueError("Missing generate_expert_trajectory function")
            
            # Check for required returns
            if "'waypoints'" not in code:
                raise ValueError("Missing waypoints in return")
            
            # Check for numpy import
            if 'import numpy' not in code:
                code = "import numpy as np\n" + code
        except Exception as e:
            logger.warning(f"Validation warning: {e}")
    
    return {'code': code, 'validated': validate}


def execute_expert_script(
    script_code: str,
    scene_objects: Dict[str, Any],
    safety_checks: bool = True
) -> Dict[str, Any]:
    """
    Execute generated expert script with scene data.
    
    Args:
        script_code: Python code from Codex
        scene_objects: Current scene object positions
        safety_checks: Whether to perform safety validation
        
    Returns:
        Trajectory data with waypoints and approach vectors
        
    Raises:
        RuntimeError: If execution fails
    """
    import numpy as np
    
    # Create execution namespace
    namespace = {
        'np': np,
        'numpy': np,
        'List': List,
        'Tuple': Tuple,
        'Dict': Dict,
        'Any': Any
    }
    
    try:
        # Execute the code
        exec(script_code, namespace)
        
        # Check for the function
        if 'generate_expert_trajectory' not in namespace:
            raise RuntimeError("Function generate_expert_trajectory not found")
        
        # Call the function
        trajectory_func = namespace['generate_expert_trajectory']
        result = trajectory_func(scene_objects)
        
        # Validate result
        if safety_checks:
            result = _validate_trajectory(result, scene_objects)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to execute expert script: {e}")
        raise RuntimeError(f"Script execution failed: {e}")


def _validate_trajectory(
    trajectory: Dict[str, Any],
    scene_objects: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate generated trajectory for safety.
    
    Args:
        trajectory: Generated trajectory data
        scene_objects: Scene object positions
        
    Returns:
        Validated trajectory (possibly with corrections)
    """
    import numpy as np
    
    # Check required fields
    if 'waypoints' not in trajectory:
        raise ValueError("Missing waypoints in trajectory")
    
    waypoints = trajectory['waypoints']
    
    # Validate each waypoint
    validated_waypoints = []
    for i, wp in enumerate(waypoints):
        if len(wp) != 3:
            logger.warning(f"Invalid waypoint {i}: {wp}, skipping")
            continue
        
        # Check bounds (example workspace)
        x, y, z = wp
        x = np.clip(x, -1.0, 1.0)
        y = np.clip(y, -1.0, 1.0) 
        z = np.clip(z, 0.0, 2.0)
        
        validated_waypoints.append((x, y, z))
    
    trajectory['waypoints'] = validated_waypoints
    
    # Validate approach vectors
    if 'approach_vectors' in trajectory:
        for idx, vec in trajectory['approach_vectors'].items():
            if len(vec) != 3:
                logger.warning(f"Invalid approach vector at {idx}")
                trajectory['approach_vectors'][idx] = [0, 0, -1]  # Default vertical
            else:
                # Normalize approach vectors
                vec_array = np.array(vec)
                norm = np.linalg.norm(vec_array)
                if norm > 0:
                    trajectory['approach_vectors'][idx] = (vec_array / norm).tolist()
    
    return trajectory


def create_parametric_waypoint_generator(
    base_waypoints: List[Tuple[float, float, float]],
    parameters: Dict[str, float]
) -> Callable:
    """
    Create a parametric waypoint generator function.
    
    Args:
        base_waypoints: Template waypoints
        parameters: Adjustable parameters
        
    Returns:
        Function that generates waypoints based on parameters
        
    Example:
        generator = create_parametric_waypoint_generator(
            base_waypoints=[(0, 0, 0.5), (0.5, 0, 0.5)],
            parameters={'scale': 1.0, 'offset_z': 0.1}
        )
        
        # Generate waypoints with different parameters
        waypoints = generator(scale=1.2, offset_z=0.15)
    """
    def generate_waypoints(**kwargs):
        """Generate waypoints with given parameters."""
        import numpy as np
        
        # Merge with default parameters
        params = parameters.copy()
        params.update(kwargs)
        
        # Apply transformations
        scale = params.get('scale', 1.0)
        offset_x = params.get('offset_x', 0.0)
        offset_y = params.get('offset_y', 0.0)
        offset_z = params.get('offset_z', 0.0)
        rotation = params.get('rotation', 0.0)  # Radians around Z
        
        transformed_waypoints = []
        
        for wp in base_waypoints:
            # Convert to numpy for easier manipulation
            point = np.array(wp)
            
            # Apply scale
            point *= scale
            
            # Apply rotation (around Z-axis)
            if rotation != 0:
                cos_r = np.cos(rotation)
                sin_r = np.sin(rotation)
                x, y = point[0], point[1]
                point[0] = x * cos_r - y * sin_r
                point[1] = x * sin_r + y * cos_r
            
            # Apply offset
            point[0] += offset_x
            point[1] += offset_y
            point[2] += offset_z
            
            transformed_waypoints.append(tuple(point))
        
        return transformed_waypoints
    
    return generate_waypoints


def interpolate_waypoints(
    waypoints: List[Tuple[float, float, float]],
    num_points: int = 100,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate between waypoints for smooth trajectory.
    
    Args:
        waypoints: List of waypoint coordinates
        num_points: Number of interpolated points
        method: Interpolation method ('linear', 'cubic', 'spline')
        
    Returns:
        Array of interpolated points (num_points, 3)
    """
    import numpy as np
    from scipy import interpolate
    
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints for interpolation")
    
    waypoints_array = np.array(waypoints)
    
    if method == 'linear':
        # Linear interpolation
        t = np.linspace(0, 1, len(waypoints))
        t_new = np.linspace(0, 1, num_points)
        
        interpolated = np.zeros((num_points, 3))
        for i in range(3):
            interpolated[:, i] = np.interp(t_new, t, waypoints_array[:, i])
        
        return interpolated
    
    elif method == 'cubic':
        # Cubic spline interpolation
        t = np.linspace(0, 1, len(waypoints))
        t_new = np.linspace(0, 1, num_points)
        
        interpolated = np.zeros((num_points, 3))
        for i in range(3):
            cs = interpolate.CubicSpline(t, waypoints_array[:, i])
            interpolated[:, i] = cs(t_new)
        
        return interpolated
    
    elif method == 'spline':
        # B-spline interpolation
        if len(waypoints) < 4:
            # Need at least 4 points for B-spline
            return interpolate_waypoints(waypoints, num_points, 'cubic')
        
        t = np.linspace(0, 1, len(waypoints))
        t_new = np.linspace(0, 1, num_points)
        
        interpolated = np.zeros((num_points, 3))
        for i in range(3):
            tck = interpolate.splrep(t, waypoints_array[:, i], s=0)
            interpolated[:, i] = interpolate.splev(t_new, tck, der=0)
        
        return interpolated
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def run_expert_and_record(
    expert_callable: Callable,
    env: Any,
    max_steps: int = 1000,
    record_frequency: int = 1,
    include_metadata: bool = True,
    save_to_file: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run expert trajectory and record (state, action) pairs at each simulation tick.
    
    This function executes an expert policy in the environment and logs detailed
    state-action pairs at specified intervals for training data collection.
    
    Args:
        expert_callable: Expert policy function that takes state and returns action
                        Can be either:
                        - Callable[[state], action] for reactive policies
                        - Callable[[state, t], action] for time-based policies
                        - Object with .get_action(state) method
        env: Simulation environment with standard gym-like interface
        max_steps: Maximum number of simulation steps
        record_frequency: Record every N steps (1 = every step)
        include_metadata: Whether to include additional metadata
        save_to_file: Optional path to save recorded data
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing:
        - 'states': List of recorded states
        - 'actions': List of recorded actions
        - 'rewards': List of step rewards (if available)
        - 'timestamps': List of simulation timestamps
        - 'metadata': Additional information about the trajectory
        - 'success': Whether the episode completed successfully
        - 'total_reward': Cumulative reward
        
    Example:
        # With waypoint-based expert
        def waypoint_expert(state, waypoints, t):
            target = waypoints[min(t // 10, len(waypoints)-1)]
            return compute_action_to_target(state, target)
        
        expert = lambda s, t: waypoint_expert(s, waypoints, t)
        data = run_expert_and_record(expert, env)
        
        # With learned policy
        data = run_expert_and_record(policy.get_action, env)
    """
    import time
    import inspect
    
    # Initialize recording structures
    recorded_data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'timestamps': [],
        'infos': [],
        'metadata': {
            'start_time': time.time(),
            'max_steps': max_steps,
            'record_frequency': record_frequency,
            'expert_type': str(type(expert_callable).__name__),
        }
    }
    
    # Check expert callable signature
    if hasattr(expert_callable, 'get_action'):
        # Object with get_action method
        get_action = expert_callable.get_action
        takes_time = False
    elif callable(expert_callable):
        # Function - check if it takes time parameter
        sig = inspect.signature(expert_callable)
        n_params = len(sig.parameters)
        takes_time = n_params >= 2
        get_action = expert_callable
    else:
        raise ValueError("expert_callable must be callable or have get_action method")
    
    # Reset environment
    try:
        if hasattr(env, 'reset'):
            state = env.reset()
        else:
            state = env.get_state()
        initial_state = state.copy() if hasattr(state, 'copy') else state
    except Exception as e:
        logger.error(f"Failed to reset environment: {e}")
        raise
    
    # Record initial state
    recorded_data['states'].append(_serialize_state(state))
    recorded_data['timestamps'].append(0.0)
    recorded_data['metadata']['initial_state'] = _serialize_state(initial_state)
    
    # Run episode
    done = False
    t = 0
    cumulative_reward = 0.0
    action = None
    
    if verbose:
        logger.info(f"Starting expert demonstration recording for {max_steps} steps")
    
    while t < max_steps and not done:
        try:
            # Get expert action
            if takes_time:
                action = get_action(state, t)
            else:
                action = get_action(state)
            
            # Step environment
            if hasattr(env, 'step'):
                next_state, reward, done, info = env.step(action)
            else:
                # Custom environment interface
                env.apply_action(action)
                next_state = env.get_state()
                reward = env.get_reward() if hasattr(env, 'get_reward') else 0.0
                done = env.is_done() if hasattr(env, 'is_done') else False
                info = env.get_info() if hasattr(env, 'get_info') else {}
            
            # Record data at specified frequency
            if t % record_frequency == 0:
                recorded_data['actions'].append(_serialize_action(action))
                recorded_data['rewards'].append(float(reward))
                recorded_data['timestamps'].append(float(t + 1))
                
                if t + 1 < max_steps:  # Don't record final state twice
                    recorded_data['states'].append(_serialize_state(next_state))
                
                if include_metadata:
                    recorded_data['infos'].append(info)
            
            # Update cumulative reward
            cumulative_reward += reward
            
            # Progress logging
            if verbose and (t + 1) % 100 == 0:
                logger.info(f"Step {t+1}/{max_steps}, Reward: {cumulative_reward:.3f}")
            
            # Update state
            state = next_state
            t += 1
            
        except Exception as e:
            logger.error(f"Error at step {t}: {e}")
            recorded_data['metadata']['error'] = str(e)
            recorded_data['metadata']['error_step'] = t
            break
    
    # Record final information
    recorded_data['success'] = done and not ('error' in recorded_data['metadata'])
    recorded_data['total_reward'] = cumulative_reward
    recorded_data['total_steps'] = t
    recorded_data['metadata']['end_time'] = time.time()
    recorded_data['metadata']['duration'] = (
        recorded_data['metadata']['end_time'] - recorded_data['metadata']['start_time']
    )
    
    # Ensure equal length arrays
    min_len = min(len(recorded_data['states']) - 1, len(recorded_data['actions']))
    recorded_data['states'] = recorded_data['states'][:min_len + 1]  # +1 for final state
    recorded_data['actions'] = recorded_data['actions'][:min_len]
    recorded_data['rewards'] = recorded_data['rewards'][:min_len]
    recorded_data['timestamps'] = recorded_data['timestamps'][:min_len + 1]
    
    if verbose:
        logger.info(f"Recording complete: {t} steps, Success: {recorded_data['success']}, "
                   f"Total reward: {cumulative_reward:.3f}")
        logger.info(f"Recorded {len(recorded_data['states'])} states and "
                   f"{len(recorded_data['actions'])} actions")
    
    # Save to file if requested
    if save_to_file:
        _save_trajectory_data(recorded_data, save_to_file)
        if verbose:
            logger.info(f"Saved trajectory data to {save_to_file}")
    
    return recorded_data


def _serialize_state(state: Any) -> Union[List, Dict, np.ndarray]:
    """
    Serialize state for recording.
    
    Args:
        state: State to serialize
        
    Returns:
        Serialized state
    """
    import numpy as np
    
    if isinstance(state, np.ndarray):
        return state.tolist()
    elif isinstance(state, dict):
        return {k: _serialize_state(v) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        return [_serialize_state(item) for item in state]
    elif hasattr(state, '__dict__'):
        # Custom state object
        return {k: _serialize_state(v) for k, v in state.__dict__.items()
                if not k.startswith('_')}
    else:
        return state


def _serialize_action(action: Any) -> Union[List, Dict, float]:
    """
    Serialize action for recording.
    
    Args:
        action: Action to serialize
        
    Returns:
        Serialized action
    """
    import numpy as np
    
    if isinstance(action, np.ndarray):
        return action.tolist()
    elif isinstance(action, dict):
        return {k: _serialize_action(v) for k, v in action.items()}
    elif isinstance(action, (list, tuple)):
        return [_serialize_action(item) for item in action]
    else:
        return action


def _save_trajectory_data(data: Dict[str, Any], filepath: str) -> None:
    """
    Save trajectory data to file.
    
    Args:
        data: Trajectory data
        filepath: Path to save file
    """
    import pickle
    import os
    
    # Determine format from extension
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    elif ext in ['.pkl', '.pickle']:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif ext == '.npz':
        # Save as numpy archive
        import numpy as np
        np.savez_compressed(
            filepath,
            states=np.array(data['states']),
            actions=np.array(data['actions']),
            rewards=np.array(data['rewards']),
            timestamps=np.array(data['timestamps'])
        )
    else:
        # Default to pickle
        with open(filepath + '.pkl', 'wb') as f:
            pickle.dump(data, f)


def create_waypoint_following_expert(
    waypoints: List[Tuple[float, float, float]],
    approach_vectors: Optional[Dict[int, List[float]]] = None,
    gripper_actions: Optional[Dict[int, str]] = None,
    control_type: str = 'position',
    interpolation: str = 'linear',
    waypoint_tolerance: float = 0.05,
    max_velocity: float = 0.5
) -> Callable:
    """
    Create an expert that follows waypoints.
    
    Args:
        waypoints: List of (x, y, z) waypoint positions
        approach_vectors: Optional approach vectors at specific waypoints
        gripper_actions: Optional gripper commands at specific waypoints
        control_type: 'position' or 'velocity' control
        interpolation: Interpolation method between waypoints
        waypoint_tolerance: Distance threshold to consider waypoint reached
        max_velocity: Maximum velocity for velocity control
        
    Returns:
        Expert function that takes state and returns action
        
    Example:
        expert = create_waypoint_following_expert(
            waypoints=[(0, 0, 0.5), (0.5, 0, 0.5), (0.5, 0, 0.1)],
            gripper_actions={2: 'close', 4: 'open'}
        )
        
        data = run_expert_and_record(expert, env)
    """
    import numpy as np
    
    # Interpolate waypoints if needed
    if interpolation != 'none':
        interpolated = interpolate_waypoints(
            waypoints, 
            num_points=len(waypoints) * 10,
            method=interpolation
        )
    else:
        interpolated = np.array(waypoints)
    
    # State tracking
    state_info = {
        'current_waypoint_idx': 0,
        'gripper_state': 'open',
        'last_position': None
    }
    
    def expert_policy(state: Any, t: Optional[int] = None) -> Any:
        """Waypoint-following expert policy."""
        
        # Extract end-effector position from state
        if isinstance(state, dict):
            if 'ee_pos' in state:
                ee_pos = np.array(state['ee_pos'])
            elif 'robot' in state and 'ee_pos' in state['robot']:
                ee_pos = np.array(state['robot']['ee_pos'])
            else:
                raise ValueError("Cannot extract end-effector position from state")
        elif isinstance(state, np.ndarray):
            # Assume first 3 elements are x, y, z
            ee_pos = state[:3]
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
        
        # Update last position
        if state_info['last_position'] is None:
            state_info['last_position'] = ee_pos.copy()
        
        # Get current target waypoint
        if state_info['current_waypoint_idx'] >= len(interpolated):
            # Reached end - stay at last waypoint
            target = interpolated[-1]
        else:
            target = interpolated[state_info['current_waypoint_idx']]
        
        # Check if reached current waypoint
        distance = np.linalg.norm(ee_pos - target)
        if distance < waypoint_tolerance:
            # Move to next waypoint
            state_info['current_waypoint_idx'] = min(
                state_info['current_waypoint_idx'] + 1,
                len(interpolated) - 1
            )
            
            # Check for gripper action
            if gripper_actions and state_info['current_waypoint_idx'] in gripper_actions:
                state_info['gripper_state'] = gripper_actions[state_info['current_waypoint_idx']]
        
        # Compute control action
        if control_type == 'position':
            # Position control - directly command target position
            action = {
                'position': target.tolist(),
                'gripper': state_info['gripper_state']
            }
            
            # Add approach vector if available
            if approach_vectors and state_info['current_waypoint_idx'] in approach_vectors:
                action['approach'] = approach_vectors[state_info['current_waypoint_idx']]
            
        elif control_type == 'velocity':
            # Velocity control - compute velocity towards target
            direction = target - ee_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Normalize and scale by max velocity
                velocity = (direction / distance) * min(max_velocity, distance)
            else:
                velocity = np.zeros(3)
            
            action = {
                'velocity': velocity.tolist(),
                'gripper': state_info['gripper_state']
            }
        
        else:
            raise ValueError(f"Unknown control type: {control_type}")
        
        # Update last position
        state_info['last_position'] = ee_pos.copy()
        
        return action
    
    # Add metadata to the function
    expert_policy.waypoints = waypoints
    expert_policy.interpolated = interpolated
    expert_policy.state_info = state_info
    
    return expert_policy


def replay_recorded_trajectory(
    recorded_data: Dict[str, Any],
    env: Any,
    speed_multiplier: float = 1.0,
    visualize: bool = True,
    compare_with_expert: bool = False
) -> Dict[str, Any]:
    """
    Replay a recorded expert trajectory in the environment.
    
    Args:
        recorded_data: Previously recorded trajectory data
        env: Environment to replay in
        speed_multiplier: Speed factor for replay (1.0 = normal speed)
        visualize: Whether to enable visualization
        compare_with_expert: Whether to compare with expert actions
        
    Returns:
        Replay statistics and comparison results
    """
    import numpy as np
    import time
    
    results = {
        'replay_rewards': [],
        'state_errors': [],
        'action_errors': [],
        'success': False
    }
    
    # Reset environment
    state = env.reset()
    
    # Replay actions
    for i, action in enumerate(recorded_data['actions']):
        # Apply action
        next_state, reward, done, info = env.step(action)
        results['replay_rewards'].append(reward)
        
        # Compare states if available
        if i + 1 < len(recorded_data['states']):
            expected_state = recorded_data['states'][i + 1]
            state_error = _compute_state_error(next_state, expected_state)
            results['state_errors'].append(state_error)
        
        # Visualization delay
        if visualize and speed_multiplier > 0:
            time.sleep(1.0 / (60 * speed_multiplier))  # 60 FPS base rate
        
        state = next_state
        
        if done:
            results['success'] = True
            break
    
    # Compute statistics
    results['total_replay_reward'] = sum(results['replay_rewards'])
    results['mean_state_error'] = np.mean(results['state_errors']) if results['state_errors'] else 0
    results['original_total_reward'] = recorded_data.get('total_reward', 0)
    results['reward_difference'] = (
        results['total_replay_reward'] - results['original_total_reward']
    )
    
    return results


def _compute_state_error(state1: Any, state2: Any) -> float:
    """
    Compute error between two states.
    
    Args:
        state1: First state
        state2: Second state
        
    Returns:
        Scalar error value
    """
    import numpy as np
    
    # Convert to numpy arrays if needed
    if isinstance(state1, dict) and isinstance(state2, dict):
        # Compare dictionaries key by key
        error = 0.0
        n_keys = 0
        for key in state1.keys():
            if key in state2:
                error += _compute_state_error(state1[key], state2[key])
                n_keys += 1
        return error / max(n_keys, 1)
    else:
        # Convert to arrays and compute L2 norm
        arr1 = np.array(state1).flatten()
        arr2 = np.array(state2).flatten()
        
        if arr1.shape != arr2.shape:
            return float('inf')
        
        return float(np.linalg.norm(arr1 - arr2))


def collect_dataset(
    sim: Any,
    expert: Callable,
    n_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    deterministic: bool = False,
    augment_states: bool = False,
    normalize: bool = True,
    add_noise: Optional[float] = None,
    success_only: bool = False,
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect dataset of (state, action) pairs from expert demonstrations.
    
    This function runs the expert policy in the simulation environment for multiple
    episodes and collects all state-action pairs into numpy arrays suitable for
    supervised learning.
    
    Args:
        sim: Simulation environment with reset() and step() methods
        expert: Expert policy callable that takes state and returns action
               Can be:
               - Callable[[state], action]
               - Callable[[state, t], action] for time-aware policies
               - Object with .get_action(state) method
        n_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether expert should use deterministic actions
        augment_states: Whether to augment states with additional features
        normalize: Whether to normalize states and actions
        add_noise: Optional noise level to add to actions (for robustness)
        success_only: Only keep episodes that succeeded
        verbose: Whether to print progress
        save_path: Optional path to save the dataset
        
    Returns:
        X: States array of shape (n_samples, state_dim)
        Y: Actions array of shape (n_samples, action_dim)
        
    Example:
        # Collect dataset from waypoint expert
        expert = create_waypoint_following_expert(waypoints, gripper_actions)
        X, Y = collect_dataset(sim, expert, n_episodes=50)
        
        # Train policy on collected data
        from cogniforge.core.policy import SimplePolicy
        policy = SimplePolicy(X.shape[1], Y.shape[1])
        # ... training code ...
    """
    import time
    import inspect
    
    # Check expert callable type
    if hasattr(expert, 'get_action'):
        get_action = expert.get_action
        takes_time = False
    elif callable(expert):
        sig = inspect.signature(expert)
        n_params = len(sig.parameters)
        takes_time = n_params >= 2
        get_action = expert
    else:
        raise ValueError("expert must be callable or have get_action method")
    
    # Storage for all episodes
    all_states = []
    all_actions = []
    episode_returns = []
    episode_lengths = []
    successful_episodes = 0
    
    if verbose:
        logger.info(f"Starting dataset collection: {n_episodes} episodes")
        start_time = time.time()
    
    # Collect episodes
    for episode_idx in range(n_episodes):
        # Reset environment
        try:
            if hasattr(sim, 'reset'):
                state = sim.reset()
            else:
                sim.reset_sim()
                state = sim.get_state()
        except Exception as e:
            logger.warning(f"Failed to reset sim for episode {episode_idx}: {e}")
            continue
        
        # Episode storage
        episode_states = []
        episode_actions = []
        episode_reward = 0.0
        done = False
        t = 0
        
        # Run episode
        while t < max_steps_per_episode and not done:
            # Store state
            episode_states.append(_process_state(state, augment_states))
            
            # Get expert action
            try:
                if takes_time:
                    if hasattr(expert, '__self__') and hasattr(expert.__self__, 'state_info'):
                        # For waypoint following experts with internal state
                        action = get_action(state, t)
                    else:
                        action = get_action(state, t)
                else:
                    if deterministic and 'deterministic' in inspect.signature(get_action).parameters:
                        action = get_action(state, deterministic=deterministic)
                    else:
                        action = get_action(state)
                
                # Add noise if requested
                if add_noise is not None and add_noise > 0:
                    action = _add_action_noise(action, add_noise)
                
            except Exception as e:
                logger.warning(f"Expert failed at step {t} in episode {episode_idx}: {e}")
                break
            
            # Store action
            episode_actions.append(_process_action(action))
            
            # Step environment
            try:
                if hasattr(sim, 'step'):
                    next_state, reward, done, info = sim.step(action)
                else:
                    # Custom sim interface
                    sim.apply_action(action)
                    next_state = sim.get_state()
                    reward = sim.get_reward() if hasattr(sim, 'get_reward') else 0.0
                    done = sim.is_done() if hasattr(sim, 'is_done') else False
                    info = {}
                
                episode_reward += reward
                state = next_state
                t += 1
                
            except Exception as e:
                logger.warning(f"Sim step failed at step {t} in episode {episode_idx}: {e}")
                break
        
        # Check if episode was successful
        success = done and (not success_only or episode_reward > 0 or 
                           (info.get('success', False) if 'info' in locals() else False))
        
        # Store episode data if successful or not filtering
        if not success_only or success:
            # Ensure equal length (remove last state if needed)
            min_len = min(len(episode_states), len(episode_actions))
            episode_states = episode_states[:min_len]
            episode_actions = episode_actions[:min_len]
            
            all_states.extend(episode_states)
            all_actions.extend(episode_actions)
            episode_returns.append(episode_reward)
            episode_lengths.append(t)
            
            if success:
                successful_episodes += 1
        
        # Progress logging
        if verbose and (episode_idx + 1) % 10 == 0:
            avg_return = np.mean(episode_returns[-10:]) if episode_returns else 0
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            logger.info(f"Episode {episode_idx + 1}/{n_episodes}: "
                       f"Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.1f}, "
                       f"Success Rate: {successful_episodes/(episode_idx+1):.2%}")
    
    # Convert to numpy arrays
    X = np.array(all_states, dtype=np.float32)
    Y = np.array(all_actions, dtype=np.float32)
    
    # Normalize if requested
    if normalize and len(X) > 0:
        X, Y, norm_params = _normalize_dataset(X, Y)
    else:
        norm_params = None
    
    # Final statistics
    if verbose:
        elapsed = time.time() - start_time
        logger.info(f"Dataset collection complete in {elapsed:.1f}s")
        logger.info(f"Collected {len(X)} state-action pairs from {len(episode_returns)} episodes")
        logger.info(f"Average episode return: {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
        logger.info(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        logger.info(f"Success rate: {successful_episodes/n_episodes:.2%}")
        logger.info(f"State shape: {X.shape}, Action shape: {Y.shape}")
    
    # Save if requested
    if save_path:
        dataset = {
            'states': X,
            'actions': Y,
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'n_episodes': len(episode_returns),
            'successful_episodes': successful_episodes,
            'norm_params': norm_params,
            'metadata': {
                'n_episodes_requested': n_episodes,
                'max_steps_per_episode': max_steps_per_episode,
                'deterministic': deterministic,
                'success_only': success_only,
                'normalized': normalize,
                'noise_level': add_noise
            }
        }
        
        # Determine format from extension
        import os
        ext = os.path.splitext(save_path)[1].lower()
        
        if ext == '.npz':
            np.savez_compressed(save_path, **dataset)
        elif ext == '.pkl':
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
        else:
            # Default to numpy
            np.savez_compressed(save_path + '.npz' if not save_path.endswith('.npz') else save_path, **dataset)
        
        if verbose:
            logger.info(f"Saved dataset to {save_path}")
    
    return X, Y


def _process_state(state: Any, augment: bool = False) -> np.ndarray:
    """
    Process state into numpy array.
    
    Args:
        state: Raw state from environment
        augment: Whether to add augmented features
        
    Returns:
        Processed state array
    """
    # Convert to numpy
    if isinstance(state, np.ndarray):
        state_array = state.flatten()
    elif isinstance(state, (list, tuple)):
        state_array = np.array(state).flatten()
    elif isinstance(state, dict):
        # Extract relevant fields from dict state
        state_parts = []
        for key in sorted(state.keys()):
            if key not in ['image', 'rendering', 'metadata']:  # Skip large/non-numeric fields
                value = state[key]
                if isinstance(value, (np.ndarray, list, tuple)):
                    state_parts.append(np.array(value).flatten())
                elif isinstance(value, (int, float)):
                    state_parts.append([value])
        state_array = np.concatenate(state_parts)
    elif hasattr(state, '__dict__'):
        # Object with attributes
        state_parts = []
        for key in sorted(state.__dict__.keys()):
            if not key.startswith('_'):
                value = getattr(state, key)
                if isinstance(value, (np.ndarray, list, tuple)):
                    state_parts.append(np.array(value).flatten())
                elif isinstance(value, (int, float)):
                    state_parts.append([value])
        state_array = np.concatenate(state_parts) if state_parts else np.array([])
    else:
        state_array = np.array([state])
    
    # Augment if requested
    if augment:
        augmented_features = []
        
        # Add squared terms for first few dimensions
        if len(state_array) > 0:
            augmented_features.append(state_array[:min(3, len(state_array))]**2)
        
        # Add velocity estimates if we have position-like features
        # (would need history for this, simplified here)
        
        if augmented_features:
            state_array = np.concatenate([state_array] + augmented_features)
    
    return state_array.astype(np.float32)


def _process_action(action: Any) -> np.ndarray:
    """
    Process action into numpy array.
    
    Args:
        action: Raw action from expert
        
    Returns:
        Processed action array
    """
    if isinstance(action, np.ndarray):
        return action.flatten().astype(np.float32)
    elif isinstance(action, (list, tuple)):
        return np.array(action, dtype=np.float32).flatten()
    elif isinstance(action, dict):
        # Extract action components from dict
        action_parts = []
        
        # Common action keys
        for key in ['position', 'velocity', 'force', 'torque', 'joint_angles']:
            if key in action:
                action_parts.append(np.array(action[key]).flatten())
        
        # Gripper as binary
        if 'gripper' in action:
            gripper_val = 1.0 if action['gripper'] in ['close', 'closed', 1, True] else 0.0
            action_parts.append([gripper_val])
        
        # Any remaining numeric values
        for key, value in action.items():
            if key not in ['position', 'velocity', 'force', 'torque', 'joint_angles', 'gripper', 'approach']:
                if isinstance(value, (int, float)):
                    action_parts.append([value])
                elif isinstance(value, (list, tuple, np.ndarray)):
                    action_parts.append(np.array(value).flatten())
        
        return np.concatenate(action_parts).astype(np.float32) if action_parts else np.array([], dtype=np.float32)
    else:
        # Single value action
        return np.array([action], dtype=np.float32)


def _add_action_noise(action: Any, noise_level: float) -> Any:
    """
    Add noise to action for robustness.
    
    Args:
        action: Original action
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        Noisy action
    """
    if isinstance(action, np.ndarray):
        noise = np.random.normal(0, noise_level, action.shape)
        return action + noise
    elif isinstance(action, (list, tuple)):
        action_array = np.array(action)
        noise = np.random.normal(0, noise_level, action_array.shape)
        noisy = action_array + noise
        return noisy.tolist() if isinstance(action, list) else tuple(noisy)
    elif isinstance(action, dict):
        noisy_action = action.copy()
        for key in ['position', 'velocity', 'force']:
            if key in action:
                value = np.array(action[key])
                noise = np.random.normal(0, noise_level, value.shape)
                noisy_action[key] = (value + noise).tolist()
        return noisy_action
    else:
        return action + np.random.normal(0, noise_level)


def _normalize_dataset(
    X: np.ndarray, 
    Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Normalize states and actions to zero mean and unit variance.
    
    Args:
        X: States array
        Y: Actions array
        
    Returns:
        X_norm: Normalized states
        Y_norm: Normalized actions
        norm_params: Normalization parameters for denormalization
    """
    # Compute statistics
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
    
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0) + 1e-8
    
    # Normalize
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std
    
    # Store parameters
    norm_params = {
        'state_mean': X_mean,
        'state_std': X_std,
        'action_mean': Y_mean,
        'action_std': Y_std
    }
    
    return X_norm, Y_norm, norm_params


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load dataset from file.
    
    Args:
        path: Path to dataset file
        
    Returns:
        X: States array
        Y: Actions array
        metadata: Additional dataset information
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npz':
        data = np.load(path)
        X = data['states']
        Y = data['actions']
        metadata = {k: data[k] for k in data.keys() if k not in ['states', 'actions']}
    elif ext in ['.pkl', '.pickle']:
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        X = data['states']
        Y = data['actions']
        metadata = {k: v for k, v in data.items() if k not in ['states', 'actions']}
    else:
        raise ValueError(f"Unknown file format: {ext}")
    
    return X, Y, metadata


def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and validation sets.
    
    Args:
        X: States array
        Y: Actions array
        train_ratio: Fraction of data for training
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train: Training states
        Y_train: Training actions
        X_val: Validation states
        Y_val: Validation actions
    """
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    if shuffle:
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_val = X[val_indices]
    Y_val = Y[val_indices]
    
    return X_train, Y_train, X_val, Y_val


# Example expert script templates
EXPERT_SCRIPT_TEMPLATES = {
    'pick_place_with_approach': """
import numpy as np
from typing import List, Tuple, Dict, Any

def generate_expert_trajectory(scene_objects: Dict[str, Any]) -> Dict[str, Any]:
    # Extract positions
    object_pos = np.array(scene_objects['{object_name}']['position'])
    target_pos = np.array(scene_objects['{target_name}']['position'])
    
    # Parameters
    approach_height = 0.15
    grasp_offset = 0.10
    place_offset = 0.05
    
    # Approach vectors
    pick_approach = np.array([0, 0, -1])  # Vertical approach
    place_approach = np.array([0, 0, -1])  # Vertical placement
    
    # Generate waypoints
    W = [
        # Initial position (assumed current position)
        tuple(object_pos + np.array([0, 0, approach_height + grasp_offset])),
        
        # Approach point above object
        tuple(object_pos + np.array([0, 0, approach_height])),
        
        # Grasp point
        tuple(object_pos),
        
        # Lift point
        tuple(object_pos + np.array([0, 0, approach_height])),
        
        # Transit point (mid-air between objects)
        tuple((object_pos + target_pos) / 2 + np.array([0, 0, 0.3])),
        
        # Approach point above target
        tuple(target_pos + np.array([0, 0, approach_height + place_offset])),
        
        # Place point
        tuple(target_pos + np.array([0, 0, place_offset])),
        
        # Retreat point
        tuple(target_pos + np.array([0, 0, approach_height + 0.1]))
    ]
    
    # Approach vectors for critical points
    approach_vectors = {
        1: pick_approach.tolist(),  # Approach to pick
        5: place_approach.tolist()  # Approach to place
    }
    
    # Gripper actions
    gripper_actions = {
        0: 'open',   # Ensure open at start
        2: 'close',  # Close at grasp point
        6: 'open',   # Open at place point
    }
    
    # Annotations
    annotations = [
        "Initial approach position",
        "Above object for pick",
        "Grasp object",
        "Lift object",
        "Transit to target",
        "Above target for place",
        "Place object",
        "Retreat from target"
    ]
    
    return {
        'waypoints': W,
        'approach_vectors': approach_vectors,
        'gripper_actions': gripper_actions,
        'annotations': annotations,
        'metadata': {
            'object': '{object_name}',
            'target': '{target_name}',
            'total_waypoints': len(W)
        }
    }
""",

    'circular_motion': """
import numpy as np
from typing import List, Tuple, Dict, Any

def generate_expert_trajectory(scene_objects: Dict[str, Any]) -> Dict[str, Any]:
    # Center and radius for circular motion
    center = np.array(scene_objects['{center_object}']['position'])
    radius = {radius}
    height = {height}
    num_points = {num_points}
    
    # Generate circular waypoints
    W = []
    angles = np.linspace(0, 2*np.pi, num_points)
    
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = height
        W.append((x, y, z))
    
    # Approach vectors (tangent to circle)
    approach_vectors = {}
    for i, angle in enumerate(angles):
        tangent = np.array([-np.sin(angle), np.cos(angle), 0])
        approach_vectors[i] = tangent.tolist()
    
    return {
        'waypoints': W,
        'approach_vectors': approach_vectors,
        'gripper_actions': {},
        'annotations': [f"Point {i} at angle {np.degrees(a):.1f}" 
                       for i, a in enumerate(angles)],
        'metadata': {
            'motion_type': 'circular',
            'center': center.tolist(),
            'radius': radius
        }
    }
"""
}


# Example usage
if __name__ == "__main__":
    # Example scene
    scene = {
        'objects': [
            {
                'name': 'red_cube',
                'position': [0.5, 0.0, 0.1],
                'size': 0.05,
                'graspable': True
            },
            {
                'name': 'blue_platform',
                'position': [0.7, 0.2, 0.05],
                'size': 0.15,
                'graspable': False
            }
        ],
        'robot_state': {
            'ee_pos': [0.0, 0.0, 0.5],
            'gripper_state': 'open'
        },
        'workspace': {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.0, 'z_max': 2.0
        }
    }
    
    # Generate Codex prompt
    print("=" * 60)
    print("CODEX PROMPT FOR EXPERT SCRIPT")
    print("=" * 60)
    
    codex_prompt = gen_expert_script(
        "Pick up the red cube and carefully place it on the blue platform",
        scene,
        use_parametric=True,
        include_approach_vectors=True
    )
    
    print(codex_prompt[:2000])
    print("\n...")
    
    # Example of using template
    print("\n" + "=" * 60)
    print("EXAMPLE GENERATED SCRIPT")
    print("=" * 60)
    
    example_script = EXPERT_SCRIPT_TEMPLATES['pick_place_with_approach'].format(
        object_name='red_cube',
        target_name='blue_platform'
    )
    
    print(example_script[:1500])
    
    # Test interpolation
    print("\n" + "=" * 60)
    print("WAYPOINT INTERPOLATION TEST")
    print("=" * 60)
    
    test_waypoints = [
        (0.0, 0.0, 0.5),
        (0.3, 0.0, 0.5),
        (0.5, 0.2, 0.3),
        (0.5, 0.2, 0.1)
    ]
    
    try:
        from scipy import interpolate
        interpolated = interpolate_waypoints(test_waypoints, num_points=20, method='cubic')
        print(f"Interpolated {len(test_waypoints)} waypoints to {len(interpolated)} points")
        print(f"First few points:\n{interpolated[:5]}")
    except ImportError:
        print("Note: scipy required for interpolation functionality")