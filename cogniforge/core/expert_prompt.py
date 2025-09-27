"""
Expert prompt generation for robotic manipulation tasks.

This module generates comprehensive prompts that describe the scene context
and available helper functions for GPT to generate executable robot code.
"""

import json
from typing import Dict, Any, List, Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)


def gen_expert_prompt(
    prompt: str,
    scene_summary: Dict[str, Any],
    include_examples: bool = True,
    output_format: str = "python"
) -> str:
    """
    Generate an expert prompt describing the task, scene, and available helper functions.
    
    This function creates a comprehensive prompt that includes:
    - The user's task description
    - Current scene state and object positions
    - Available helper functions with signatures and descriptions
    - Usage examples and best practices
    
    Args:
        prompt: User's natural language task description
        scene_summary: Dictionary containing scene information:
            - 'objects': List of objects with positions and properties
            - 'robot_state': Current robot configuration
            - 'gripper_state': Current gripper state (open/closed)
            - 'constraints': Any task constraints or limitations
        include_examples: Whether to include usage examples
        output_format: Desired output format ('python', 'pseudocode', 'json')
        
    Returns:
        Formatted expert prompt string for GPT
        
    Example:
        scene = {
            'objects': [
                {'name': 'red_cube', 'position': [0.5, 0.0, 0.1], 'graspable': True},
                {'name': 'blue_cube', 'position': [0.6, 0.1, 0.1], 'graspable': True},
                {'name': 'table', 'position': [0.5, 0.0, 0.0], 'graspable': False}
            ],
            'robot_state': {'position': [0.0, 0.0, 0.0], 'gripper': 'open'},
            'gripper_state': 'open'
        }
        
        prompt = gen_expert_prompt(
            "Pick up the red cube and stack it on the blue cube",
            scene
        )
    """
    
    # Format scene information
    scene_description = _format_scene_summary(scene_summary)
    
    # Get helper function descriptions
    helper_descriptions = _get_helper_function_descriptions(include_examples)
    
    # Get output format instructions
    format_instructions = _get_format_instructions(output_format)
    
    # Build the complete prompt
    expert_prompt = f"""You are an expert robotic manipulation system. Your task is to generate executable code to complete the following task using the available helper functions.

TASK DESCRIPTION:
{prompt}

CURRENT SCENE STATE:
{scene_description}

AVAILABLE HELPER FUNCTIONS:
{helper_descriptions}

IMPORTANT GUIDELINES:
1. ALWAYS check if objects are graspable before attempting to pick them
2. ALWAYS open gripper before picking and close after grasping
3. Use calculate_ik to verify reachability before moving
4. Use move_through_waypoints for smooth trajectories
5. Consider collision avoidance when planning paths
6. Verify gripper state matches the required action
7. Handle errors gracefully with try-except blocks

{format_instructions}

YOUR SOLUTION:
"""
    
    return expert_prompt


def _format_scene_summary(scene_summary: Dict[str, Any]) -> str:
    """
    Format scene summary into readable description.
    
    Args:
        scene_summary: Scene information dictionary
        
    Returns:
        Formatted scene description string
    """
    lines = []
    
    # Objects in scene
    if 'objects' in scene_summary:
        lines.append("Objects in scene:")
        for obj in scene_summary['objects']:
            name = obj.get('name', 'unknown')
            pos = obj.get('position', [0, 0, 0])
            graspable = obj.get('graspable', False)
            size = obj.get('size', 'medium')
            color = obj.get('color', '')
            
            obj_desc = f"  - {name}: position={pos}"
            if color:
                obj_desc += f", color={color}"
            if size != 'medium':
                obj_desc += f", size={size}"
            obj_desc += f", graspable={'Yes' if graspable else 'No'}"
            
            if 'state' in obj:
                obj_desc += f", state={obj['state']}"
            
            lines.append(obj_desc)
    
    # Robot state
    if 'robot_state' in scene_summary:
        lines.append("\nRobot state:")
        robot = scene_summary['robot_state']
        if 'position' in robot:
            lines.append(f"  - End-effector position: {robot['position']}")
        if 'orientation' in robot:
            lines.append(f"  - End-effector orientation: {robot['orientation']}")
        if 'joint_positions' in robot:
            lines.append(f"  - Joint positions: {robot['joint_positions']}")
    
    # Gripper state
    if 'gripper_state' in scene_summary:
        lines.append(f"\nGripper state: {scene_summary['gripper_state']}")
    
    # Constraints
    if 'constraints' in scene_summary:
        lines.append("\nConstraints:")
        for constraint in scene_summary['constraints']:
            lines.append(f"  - {constraint}")
    
    # Workspace bounds
    if 'workspace' in scene_summary:
        ws = scene_summary['workspace']
        lines.append(f"\nWorkspace bounds: x=[{ws['x_min']}, {ws['x_max']}], "
                    f"y=[{ws['y_min']}, {ws['y_max']}], z=[{ws['z_min']}, {ws['z_max']}]")
    
    return "\n".join(lines)


def _get_helper_function_descriptions(include_examples: bool = True) -> str:
    """
    Get descriptions of available helper functions.
    
    Args:
        include_examples: Whether to include usage examples
        
    Returns:
        Formatted helper function descriptions
    """
    descriptions = """
1. calculate_ik(robot_name, target_pos, target_orn=None, **kwargs)
   Description: Calculate inverse kinematics to find joint positions for target end-effector pose
   Parameters:
     - robot_name: Name of the robot (string)
     - target_pos: Target position [x, y, z] (list or tuple)
     - target_orn: Target orientation quaternion [x, y, z, w] (optional)
     - max_iterations: Maximum IK iterations (default: 100)
     - use_nullspace: Use null-space control (default: False)
   Returns: List of joint positions
   """
    
    if include_examples:
        descriptions += """   Example:
     joint_positions = calculate_ik("robot", [0.5, 0.0, 0.3])
     joint_positions = calculate_ik("robot", [0.5, 0.0, 0.3], [0, 0, 0, 1])
   """
    
    descriptions += """
2. move_through_waypoints(robot_name, waypoints, **kwargs)
   Description: Move robot through a sequence of waypoints using smooth trajectories
   Parameters:
     - robot_name: Name of the robot (string)
     - waypoints: List of target positions or dicts with 'pos' and optional 'orn'
     - steps_per_segment: Simulation steps between waypoints (default: 60)
     - use_orientation: Whether to use orientation from waypoints (default: False)
     - gripper_actions: Dict mapping waypoint indices to gripper actions (optional)
   Returns: Trajectory data if return_trajectories=True, else None
   """
    
    if include_examples:
        descriptions += """   Example:
     # Simple position waypoints
     waypoints = [[0.5, 0.0, 0.3], [0.5, 0.2, 0.3], [0.5, 0.2, 0.1]]
     move_through_waypoints("robot", waypoints)
     
     # With gripper control
     move_through_waypoints("robot", waypoints, gripper_actions={0: 'open', 2: 'close'})
   """
    
    descriptions += """
3. open_gripper(robot_name, force=100.0)
   Description: Open the robot's gripper fully
   Parameters:
     - robot_name: Name of the robot (string)
     - force: Maximum force to apply (default: 100.0 N)
   Returns: None
   """
    
    if include_examples:
        descriptions += """   Example:
     open_gripper("robot")
     open_gripper("robot", force=50.0)  # Gentle opening
   """
    
    descriptions += """
4. close_gripper(robot_name, force=50.0)
   Description: Close the robot's gripper to grasp an object
   Parameters:
     - robot_name: Name of the robot (string)
     - force: Maximum force to apply (default: 50.0 N for gentle grasping)
   Returns: None
   """
    
    if include_examples:
        descriptions += """   Example:
     close_gripper("robot")
     close_gripper("robot", force=20.0)  # Very gentle for fragile objects
   """
    
    descriptions += """
5. set_gripper(robot_name, opening, force=100.0)
   Description: Set gripper to specific opening amount
   Parameters:
     - robot_name: Name of the robot (string)
     - opening: Target opening (0.0 = closed, 1.0 = fully open)
     - force: Maximum force to apply (default: 100.0 N)
   Returns: None
   """
    
    if include_examples:
        descriptions += """   Example:
     set_gripper("robot", 0.5)  # Half open
     set_gripper("robot", 0.0, force=30.0)  # Close gently
   """
    
    descriptions += """
6. get_object_position(object_name)
   Description: Get the current position of an object in the scene
   Parameters:
     - object_name: Name of the object (string)
   Returns: Position [x, y, z] or None if object not found
   """
    
    if include_examples:
        descriptions += """   Example:
     pos = get_object_position("red_cube")
     if pos:
         print(f"Red cube is at {pos}")
   """
    
    descriptions += """
7. check_grasp_success(robot_name, object_name)
   Description: Check if object is successfully grasped
   Parameters:
     - robot_name: Name of the robot (string)
     - object_name: Name of the object to check (string)
   Returns: Boolean indicating grasp success
   """
    
    if include_examples:
        descriptions += """   Example:
     if check_grasp_success("robot", "red_cube"):
         print("Successfully grasped red cube")
   """
    
    descriptions += """
8. step_simulation(num_steps=1)
   Description: Step the physics simulation forward
   Parameters:
     - num_steps: Number of simulation steps (default: 1)
   Returns: None
   """
    
    if include_examples:
        descriptions += """   Example:
     step_simulation()  # Single step
     step_simulation(10)  # 10 steps for settling
   """
    
    return descriptions


def _get_format_instructions(output_format: str) -> str:
    """
    Get output format instructions based on desired format.
    
    Args:
        output_format: Desired output format
        
    Returns:
        Format-specific instructions
    """
    if output_format == "python":
        return """OUTPUT FORMAT:
Generate executable Python code that:
1. Imports necessary functions (assume they are available)
2. Implements the complete task step by step
3. Includes error handling with try-except blocks
4. Adds comments explaining each major step
5. Returns success status and any error messages

Example structure:
```python
def execute_task(sim):
    try:
        # Step 1: Open gripper
        open_gripper("robot")
        
        # Step 2: Move to object
        target_pos = get_object_position("red_cube")
        if target_pos:
            # Add approach offset
            approach_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
            move_through_waypoints("robot", [approach_pos, target_pos])
        
        # Step 3: Grasp object
        close_gripper("robot")
        step_simulation(10)  # Let gripper settle
        
        # Check grasp success
        if not check_grasp_success("robot", "red_cube"):
            return False, "Failed to grasp object"
        
        # Continue with task...
        return True, "Task completed successfully"
        
    except Exception as e:
        return False, f"Error: {str(e)}"
```"""
    
    elif output_format == "pseudocode":
        return """OUTPUT FORMAT:
Generate clear pseudocode that:
1. Lists each step sequentially
2. Includes decision points and loops
3. Shows error handling logic
4. Uses descriptive variable names

Example structure:
```
FUNCTION execute_task:
    OPEN gripper
    
    GET position of red_cube
    IF position exists:
        CALCULATE approach position (above object)
        MOVE through [approach_position, object_position]
        
        CLOSE gripper
        WAIT for gripper to settle
        
        CHECK if grasp successful
        IF NOT successful:
            RETURN failure
        
        // Continue task...
    ELSE:
        RETURN object not found
        
    RETURN success
```"""
    
    elif output_format == "json":
        return """OUTPUT FORMAT:
Generate a JSON action sequence:
```json
{
    "task": "task_description",
    "steps": [
        {
            "action": "open_gripper",
            "parameters": {"robot_name": "robot", "force": 100.0}
        },
        {
            "action": "move_through_waypoints",
            "parameters": {
                "robot_name": "robot",
                "waypoints": [[0.5, 0.0, 0.3], [0.5, 0.0, 0.1]]
            }
        },
        {
            "action": "close_gripper",
            "parameters": {"robot_name": "robot", "force": 50.0}
        }
    ],
    "success_criteria": "Object grasped and moved to target"
}
```"""
    
    else:
        return "OUTPUT FORMAT:\nProvide a clear, step-by-step solution using the available helper functions."


def gen_expert_prompt_with_context(
    prompt: str,
    scene_summary: Dict[str, Any],
    previous_attempts: Optional[List[Dict[str, Any]]] = None,
    task_constraints: Optional[Dict[str, Any]] = None,
    preferred_approach: Optional[str] = None
) -> str:
    """
    Generate an expert prompt with additional context and constraints.
    
    This enhanced version includes:
    - Previous attempt history for learning from failures
    - Task-specific constraints
    - Preferred approach or strategy hints
    
    Args:
        prompt: User's task description
        scene_summary: Scene information
        previous_attempts: List of previous attempts with outcomes
        task_constraints: Specific constraints for the task
        preferred_approach: Hint about preferred solution approach
        
    Returns:
        Enhanced expert prompt string
    """
    base_prompt = gen_expert_prompt(prompt, scene_summary)
    
    additional_context = []
    
    # Add previous attempts if available
    if previous_attempts:
        additional_context.append("\nPREVIOUS ATTEMPTS:")
        for i, attempt in enumerate(previous_attempts, 1):
            additional_context.append(f"Attempt {i}:")
            if 'approach' in attempt:
                additional_context.append(f"  Approach: {attempt['approach']}")
            if 'outcome' in attempt:
                additional_context.append(f"  Outcome: {attempt['outcome']}")
            if 'failure_reason' in attempt:
                additional_context.append(f"  Failure reason: {attempt['failure_reason']}")
        additional_context.append("Learn from these attempts and avoid the same mistakes.\n")
    
    # Add task constraints
    if task_constraints:
        additional_context.append("\nTASK CONSTRAINTS:")
        if 'time_limit' in task_constraints:
            additional_context.append(f"- Complete within {task_constraints['time_limit']} seconds")
        if 'force_limit' in task_constraints:
            additional_context.append(f"- Maximum gripper force: {task_constraints['force_limit']} N")
        if 'precision' in task_constraints:
            additional_context.append(f"- Position precision required: {task_constraints['precision']} meters")
        if 'forbidden_zones' in task_constraints:
            additional_context.append(f"- Avoid zones: {task_constraints['forbidden_zones']}")
        additional_context.append("")
    
    # Add preferred approach hint
    if preferred_approach:
        additional_context.append(f"\nPREFERRED APPROACH: {preferred_approach}\n")
    
    # Insert additional context before "YOUR SOLUTION:"
    if additional_context:
        context_str = "\n".join(additional_context)
        base_prompt = base_prompt.replace("YOUR SOLUTION:", 
                                          f"{context_str}\nYOUR SOLUTION:")
    
    return base_prompt


def create_scene_summary(
    objects: List[Dict[str, Any]],
    robot_position: Optional[List[float]] = None,
    gripper_state: str = "open",
    workspace_bounds: Optional[Dict[str, float]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a properly formatted scene summary dictionary.
    
    Args:
        objects: List of objects in the scene
        robot_position: Current robot end-effector position
        gripper_state: Current gripper state
        workspace_bounds: Robot workspace boundaries
        additional_info: Any additional scene information
        
    Returns:
        Formatted scene summary dictionary
        
    Example:
        scene = create_scene_summary(
            objects=[
                {'name': 'red_cube', 'position': [0.5, 0.0, 0.1], 'graspable': True},
                {'name': 'table', 'position': [0.5, 0.0, 0.0], 'graspable': False}
            ],
            robot_position=[0.0, 0.0, 0.5],
            gripper_state="open"
        )
    """
    scene_summary = {
        'objects': objects,
        'gripper_state': gripper_state
    }
    
    if robot_position is not None:
        scene_summary['robot_state'] = {
            'position': robot_position
        }
    
    if workspace_bounds is not None:
        scene_summary['workspace'] = workspace_bounds
    
    if additional_info is not None:
        scene_summary.update(additional_info)
    
    return scene_summary


def parse_gpt_code_response(response: str, output_format: str = "python") -> Union[str, Dict, List]:
    """
    Parse GPT's response to extract executable code or action sequence.
    
    Args:
        response: Raw GPT response
        output_format: Expected format of the response
        
    Returns:
        Parsed code, JSON structure, or action list
    """
    if output_format == "python":
        # Extract Python code from markdown code blocks
        import re
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        # Try to find any function definition
        func_match = re.search(r'def \w+\(.*?\):.*?(?=\n(?:def|\Z))', response, re.DOTALL)
        if func_match:
            return func_match.group(0)
        return response  # Return as-is if no code blocks found
    
    elif output_format == "json":
        # Extract and parse JSON
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return None
    
    else:
        return response


# Example templates for common tasks
TASK_TEMPLATES = {
    'pick_and_place': """
def execute_pick_and_place(sim, object_name, target_position):
    try:
        # Open gripper for picking
        open_gripper("robot")
        
        # Get object position
        obj_pos = get_object_position(object_name)
        if not obj_pos:
            return False, f"Object {object_name} not found"
        
        # Plan approach and grasp waypoints
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1]
        waypoints = [approach_pos, obj_pos]
        
        # Move to object and grasp
        move_through_waypoints("robot", waypoints)
        close_gripper("robot")
        step_simulation(10)
        
        # Verify grasp
        if not check_grasp_success("robot", object_name):
            return False, "Failed to grasp object"
        
        # Move to target position
        lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.2]
        target_approach = [target_position[0], target_position[1], target_position[2] + 0.1]
        waypoints = [lift_pos, target_approach, target_position]
        move_through_waypoints("robot", waypoints)
        
        # Release object
        open_gripper("robot")
        
        # Retract
        retract_pos = [target_position[0], target_position[1], target_position[2] + 0.2]
        move_through_waypoints("robot", [retract_pos])
        
        return True, "Pick and place completed successfully"
        
    except Exception as e:
        return False, f"Error during execution: {str(e)}"
""",
    
    'stacking': """
def execute_stacking(sim, object_to_stack, base_object):
    try:
        # Get base object position
        base_pos = get_object_position(base_object)
        if not base_pos:
            return False, f"Base object {base_object} not found"
        
        # Calculate stack position (on top of base)
        stack_height_offset = 0.06  # Adjust based on object size
        target_pos = [base_pos[0], base_pos[1], base_pos[2] + stack_height_offset]
        
        # Use pick and place to stack
        return execute_pick_and_place(sim, object_to_stack, target_pos)
        
    except Exception as e:
        return False, f"Error during stacking: {str(e)}"
"""
}


# Testing
if __name__ == "__main__":
    # Example scene
    scene = {
        'objects': [
            {'name': 'red_cube', 'position': [0.5, 0.0, 0.1], 'graspable': True, 'color': 'red'},
            {'name': 'blue_cube', 'position': [0.6, 0.1, 0.1], 'graspable': True, 'color': 'blue'},
            {'name': 'green_sphere', 'position': [0.4, -0.1, 0.1], 'graspable': True, 'color': 'green'},
            {'name': 'table', 'position': [0.5, 0.0, 0.0], 'graspable': False}
        ],
        'robot_state': {
            'position': [0.0, 0.0, 0.5],
            'orientation': [0, 0, 0, 1]
        },
        'gripper_state': 'open',
        'workspace': {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.0, 'z_max': 2.0
        },
        'constraints': [
            'Avoid collisions with the table',
            'Handle objects gently'
        ]
    }
    
    # Generate basic expert prompt
    print("=" * 60)
    print("BASIC EXPERT PROMPT")
    print("=" * 60)
    
    prompt = gen_expert_prompt(
        "Pick up the red cube and stack it on the blue cube",
        scene,
        include_examples=True,
        output_format="python"
    )
    
    print(prompt[:2000] + "\n...")  # Print first 2000 chars
    
    # Generate enhanced prompt with context
    print("\n" + "=" * 60)
    print("ENHANCED EXPERT PROMPT WITH CONTEXT")
    print("=" * 60)
    
    previous_attempts = [
        {
            'approach': 'Direct grasp without approach',
            'outcome': 'Failed',
            'failure_reason': 'Collision with table edge'
        }
    ]
    
    task_constraints = {
        'time_limit': 30,
        'force_limit': 20.0,
        'precision': 0.001
    }
    
    enhanced_prompt = gen_expert_prompt_with_context(
        "Carefully pick up the fragile red cube and place it on the platform",
        scene,
        previous_attempts=previous_attempts,
        task_constraints=task_constraints,
        preferred_approach="Use slow, controlled movements with minimal force"
    )
    
    print(enhanced_prompt[:2000] + "\n...")  # Print first 2000 chars