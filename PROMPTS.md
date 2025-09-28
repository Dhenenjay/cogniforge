# 🤖 Cogniforge LLM Prompts Documentation

This document contains all the Large Language Model (LLM) prompts used throughout the Cogniforge system. These prompts are designed to generate high-quality outputs while maintaining fallback compatibility.

---

## 📑 Table of Contents

1. [Task Planning Prompts](#task-planning-prompts)
2. [Reward Weight Generation](#reward-weight-generation)
3. [Expert Trajectory Generation](#expert-trajectory-generation)
4. [Code Generation Prompts](#code-generation-prompts)
5. [Vision Analysis Prompts](#vision-analysis-prompts)
6. [Error Recovery Prompts](#error-recovery-prompts)
7. [Optimization Prompts](#optimization-prompts)

---

## Task Planning Prompts

### Main Task Analysis Prompt

```python
TASK_ANALYSIS_PROMPT = """
You are an expert robotics task planner. Analyze the following natural language task description and convert it into a structured task plan.

Task Description: {task_description}

Current Scene State:
- Objects detected: {detected_objects}
- Robot position: {robot_position}
- Workspace bounds: {workspace_bounds}

Please provide a structured task plan in the following JSON format:
{
  "task_type": "pick_place" | "push" | "lift" | "stack" | "navigate",
  "primary_object": {
    "type": "object type",
    "color": "color if specified",
    "size": "size if specified",
    "position_constraint": "any position requirements"
  },
  "target_location": {
    "type": "absolute" | "relative" | "object",
    "coordinates": [x, y, z] or null,
    "reference_object": "object name if relative"
  },
  "subtasks": [
    {
      "id": 1,
      "action": "move_to" | "grasp" | "release" | "push" | "lift",
      "parameters": {},
      "preconditions": [],
      "effects": []
    }
  ],
  "constraints": [
    "List any specific constraints like 'gentle', 'fast', 'avoid_collision'"
  ],
  "success_criteria": {
    "primary": "main success condition",
    "secondary": ["additional success conditions"]
  }
}

Important considerations:
1. Break down complex tasks into simple, executable subtasks
2. Ensure all coordinates are within workspace bounds
3. Consider object physics and stability
4. Include safety constraints when handling fragile objects
5. Specify clear success criteria for validation

Return ONLY valid JSON, no additional text.
"""

# Fallback keyword extraction (when LLM fails)
KEYWORD_EXTRACTION_RULES = {
    "pick": ["grasp", "grab", "pick up", "lift", "take"],
    "place": ["put", "place", "drop", "set", "position"],
    "push": ["push", "slide", "move"],
    "stack": ["stack", "pile", "arrange"],
    "red": ["red", "crimson", "scarlet"],
    "blue": ["blue", "navy", "azure"],
    "green": ["green", "emerald", "lime"],
    "cube": ["cube", "box", "block"],
    "sphere": ["sphere", "ball", "orb"],
    "cylinder": ["cylinder", "tube", "rod"]
}
```

### Subtask Decomposition Prompt

```python
SUBTASK_DECOMPOSITION_PROMPT = """
Given a high-level robotic task, decompose it into atomic actions that can be directly executed.

High-level task: {high_level_task}
Current robot capabilities: {robot_capabilities}

Decompose this into atomic actions from the following set:
- move_to(position): Move end-effector to position
- grasp(): Close gripper
- release(): Open gripper  
- rotate(angle): Rotate end-effector
- wait(duration): Pause execution
- sense(): Update sensor readings

For each atomic action, specify:
1. Action name and parameters
2. Expected duration (seconds)
3. Failure conditions
4. Recovery strategy if failed

Format your response as:
```
Step 1: [action_name]([parameters])
  Duration: [X] seconds
  Failure: [what could go wrong]
  Recovery: [what to do if failed]
  
Step 2: ...
```

Ensure the sequence is executable and safe.
"""
```

---

## Reward Weight Generation

### Reward Weight Optimization Prompt

```python
REWARD_WEIGHT_PROMPT = """
You are an expert in robotic reward function design. Generate optimal reward weights for the following task.

Task Type: {task_type}
Task Description: {task_description}
Environment: {environment_description}
Robot Type: {robot_type}

Consider the following reward components:
1. task_completion: Achieving the primary goal
2. efficiency: Minimizing time/energy
3. smoothness: Minimizing jerk and acceleration
4. safety: Avoiding collisions and unsafe states
5. precision: Accuracy of final position
6. stability: Maintaining balance and control

Generate reward weights that:
- Sum to 1.0 (normalized)
- Prioritize task completion
- Balance efficiency with safety
- Are appropriate for the specific task

Provide your response in this JSON format:
{
  "weights": {
    "task_completion": 0.0,
    "efficiency": 0.0,
    "smoothness": 0.0,
    "safety": 0.0,
    "precision": 0.0,
    "stability": 0.0
  },
  "reasoning": {
    "task_completion": "Why this weight?",
    "efficiency": "Why this weight?",
    "smoothness": "Why this weight?",
    "safety": "Why this weight?",
    "precision": "Why this weight?",
    "stability": "Why this weight?"
  },
  "task_specific_adjustments": [
    "List any task-specific considerations"
  ]
}

Example weights for reference:
- Pick and place: task_completion=0.4, precision=0.25, safety=0.2, efficiency=0.1, smoothness=0.05
- Navigation: safety=0.35, efficiency=0.3, task_completion=0.25, smoothness=0.1
- Delicate manipulation: precision=0.35, safety=0.3, smoothness=0.2, task_completion=0.15

Return ONLY valid JSON.
"""

# Fallback reward weights
DEFAULT_REWARD_WEIGHTS = {
    "pick_place": {
        "task_completion": 0.40,
        "precision": 0.25,
        "safety": 0.20,
        "efficiency": 0.10,
        "smoothness": 0.05,
        "stability": 0.00
    },
    "navigation": {
        "safety": 0.35,
        "efficiency": 0.30,
        "task_completion": 0.25,
        "smoothness": 0.10,
        "precision": 0.00,
        "stability": 0.00
    },
    "manipulation": {
        "precision": 0.35,
        "safety": 0.30,
        "smoothness": 0.20,
        "task_completion": 0.15,
        "efficiency": 0.00,
        "stability": 0.00
    }
}
```

### Dynamic Reward Adjustment Prompt

```python
REWARD_ADJUSTMENT_PROMPT = """
Based on the execution history, adjust the reward weights to improve performance.

Current Weights: {current_weights}
Recent Performance Metrics:
- Success rate: {success_rate}%
- Average completion time: {avg_time} seconds
- Collision count: {collisions}
- Precision error: {precision_error} cm
- Energy consumption: {energy} units

Identified Issues:
{issues_list}

Adjust the weights to address the issues while maintaining overall task performance.
Provide adjustments as multipliers (0.5 = reduce by half, 2.0 = double).

Response format:
{
  "weight_adjustments": {
    "task_completion": 1.0,
    "efficiency": 1.0,
    "smoothness": 1.0,
    "safety": 1.0,
    "precision": 1.0,
    "stability": 1.0
  },
  "explanation": "Why these adjustments?"
}
"""
```

---

## Expert Trajectory Generation

### Expert Demonstration Prompt

```python
EXPERT_TRAJECTORY_PROMPT = """
You are an expert robotics engineer. Generate a Python script that produces an optimal trajectory for the given task.

Task: {task_description}
Robot: {robot_type}
Scene State:
- Object positions: {object_positions}
- Obstacles: {obstacles}
- Workspace: {workspace_bounds}

Requirements:
1. Generate waypoints as a list of (x, y, z) coordinates
2. Include gripper commands at each waypoint
3. Ensure collision-free path
4. Optimize for smoothness and efficiency
5. Stay within workspace bounds

Your Python script should define:
```python
def generate_expert_trajectory():
    '''
    Generates expert trajectory for the task.
    Returns:
        waypoints: List of (x, y, z) tuples
        gripper_actions: Dict mapping waypoint indices to 'open'/'close'
        trajectory_metadata: Dict with additional information
    '''
    # Your implementation here
    
    waypoints = [
        (x1, y1, z1),  # Approach point
        (x2, y2, z2),  # Grasp point
        # ... more waypoints
    ]
    
    gripper_actions = {
        0: 'open',
        1: 'close',
        # ... more actions
    }
    
    trajectory_metadata = {
        'expected_duration': 0.0,  # seconds
        'difficulty': 'easy|medium|hard',
        'key_points': ['description of important points']
    }
    
    return waypoints, gripper_actions, trajectory_metadata
```

IMPORTANT: 
- Use actual numeric values, not variables
- Ensure physical feasibility
- Include comments explaining each waypoint's purpose
- Return only the Python function, no other text
"""

# Fallback expert trajectories
FALLBACK_EXPERT_TRAJECTORIES = {
    "pick_place": """
def generate_expert_trajectory():
    '''Fallback 3-waypoint pick and place trajectory'''
    
    # Standard pick and place pattern
    waypoints = [
        (0.4, 0.0, 0.3),  # Approach above object
        (0.4, 0.0, 0.1),  # Lower to grasp
        (0.6, 0.2, 0.3),  # Lift and move to place
    ]
    
    gripper_actions = {
        0: 'open',   # Open before approach
        1: 'close',  # Close to grasp
        2: 'open'    # Open to release
    }
    
    trajectory_metadata = {
        'expected_duration': 5.0,
        'difficulty': 'easy',
        'key_points': ['approach', 'grasp', 'place']
    }
    
    return waypoints, gripper_actions, trajectory_metadata
""",
    
    "push": """
def generate_expert_trajectory():
    '''Fallback push trajectory'''
    
    waypoints = [
        (0.3, 0.0, 0.15),  # Approach behind object
        (0.5, 0.0, 0.15),  # Push forward
        (0.5, 0.0, 0.3),   # Lift away
    ]
    
    gripper_actions = {
        0: 'close',  # Keep gripper closed for pushing
        1: 'close',
        2: 'close'
    }
    
    trajectory_metadata = {
        'expected_duration': 4.0,
        'difficulty': 'easy',
        'key_points': ['approach', 'push', 'retreat']
    }
    
    return waypoints, gripper_actions, trajectory_metadata
"""
}
```

### Trajectory Refinement Prompt

```python
TRAJECTORY_REFINEMENT_PROMPT = """
Refine the given trajectory to improve its quality and safety.

Current Trajectory:
Waypoints: {waypoints}
Gripper Actions: {gripper_actions}

Issues Detected:
{issues}

Refinement Goals:
1. Smooth sharp turns (reduce jerk)
2. Add intermediate points if needed
3. Adjust timing for better dynamics
4. Ensure collision avoidance
5. Optimize path length

Provide refined trajectory in the same format:
{
  "waypoints": [[x1, y1, z1], [x2, y2, z2], ...],
  "gripper_actions": {0: "open", 1: "close", ...},
  "improvements": ["list of improvements made"],
  "quality_metrics": {
    "smoothness": 0.0-1.0,
    "safety": 0.0-1.0,
    "efficiency": 0.0-1.0
  }
}
"""
```

---

## Code Generation Prompts

### General Python Code Generation

```python
CODE_GENERATION_PROMPT = """
Generate Python code for the following robotic control task.

Task: {task_description}
Input Variables Available:
- robot: Robot control interface
- scene: Current scene state
- config: System configuration

Required Functionality:
{requirements}

Constraints:
- Use only safe operations
- Include error handling
- Add logging statements
- Follow Python best practices
- Maximum execution time: {max_time} seconds

Generate a complete Python function:
```python
def execute_task(robot, scene, config):
    '''
    {task_description}
    
    Args:
        robot: Robot control interface
        scene: Scene state object  
        config: Configuration dictionary
        
    Returns:
        success: Boolean indicating task completion
        data: Dictionary with execution data
    '''
    try:
        # Your implementation here
        
        return True, {'message': 'Task completed successfully'}
        
    except Exception as e:
        logging.error(f"Task execution failed: {e}")
        return False, {'error': str(e)}
```

Include:
1. Input validation
2. Safety checks
3. Progress logging
4. Clean error handling
5. Meaningful return data
"""
```

### Control Loop Generation

```python
CONTROL_LOOP_PROMPT = """
Generate a control loop for the robotic system.

Control Type: {control_type}  # 'position', 'velocity', 'force', 'impedance'
Control Frequency: {frequency} Hz
State Variables: {state_vars}
Control Parameters: {control_params}

Generate a control loop that:
1. Reads current state
2. Computes control action
3. Applies safety limits
4. Sends commands
5. Logs performance

Template:
```python
class {ControllerName}Controller:
    def __init__(self, params):
        self.params = params
        self.history = []
        
    def compute_control(self, state, target):
        '''
        Compute control action based on current state and target.
        
        Args:
            state: Current system state
            target: Desired target state
            
        Returns:
            action: Control action to apply
        '''
        # Your control law here
        
        # Apply safety limits
        action = self.apply_safety_limits(action)
        
        # Log data
        self.history.append({
            'timestamp': time.time(),
            'state': state,
            'action': action,
            'target': target
        })
        
        return action
        
    def apply_safety_limits(self, action):
        # Safety limiting logic
        return np.clip(action, self.params['min'], self.params['max'])
```

Ensure the controller is stable and robust.
"""
```

### Behavioral Cloning Network Generation

```python
BC_NETWORK_PROMPT = """
Generate a PyTorch neural network architecture for behavioral cloning.

Task Type: {task_type}
Input Dimensions: {input_dims}
Output Dimensions: {output_dims}
Dataset Size: {dataset_size}

Requirements:
1. Appropriate architecture for the task
2. Include dropout for regularization
3. Add batch normalization if beneficial
4. Use appropriate activation functions
5. Include training loop template

Generate the complete implementation:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCNetwork(nn.Module):
    def __init__(self, input_dim={input_dims}, output_dim={output_dims}):
        super(BCNetwork, self).__init__()
        
        # Define layers
        # Your architecture here
        
    def forward(self, x):
        # Forward pass
        return output
        
def train_bc_network(model, dataloader, epochs=100):
    '''
    Train the behavioral cloning network.
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            # Training loop
            pass
            
    return model
```

Consider:
- Overfitting prevention
- Gradient clipping if needed
- Learning rate scheduling
- Validation split
"""
```

---

## Vision Analysis Prompts

### Object Detection Prompt

```python
VISION_DETECTION_PROMPT = """
Analyze the provided image and detect objects relevant to robotic manipulation.

Image: [RGB image provided via API]
Task Context: {task_context}

Provide detailed object information:
{
  "objects": [
    {
      "id": "unique_identifier",
      "type": "cube|sphere|cylinder|custom",
      "color": "detected color",
      "position": {
        "pixel_coordinates": [x, y],
        "estimated_3d": [x, y, z],
        "confidence": 0.0-1.0
      },
      "dimensions": {
        "width": 0.0,
        "height": 0.0,
        "depth": 0.0,
        "unit": "meters"
      },
      "attributes": {
        "graspable": true|false,
        "fragile": true|false,
        "movable": true|false,
        "stacked": true|false
      },
      "bounding_box": {
        "top_left": [x, y],
        "bottom_right": [x, y]
      }
    }
  ],
  "scene_attributes": {
    "lighting": "good|poor|variable",
    "occlusions": true|false,
    "clutter_level": "low|medium|high"
  },
  "recommended_grasp_points": [
    {
      "object_id": "id",
      "position": [x, y, z],
      "approach_vector": [x, y, z],
      "confidence": 0.0-1.0
    }
  ]
}

Focus on:
1. Objects mentioned in the task
2. Potential obstacles
3. Workspace boundaries
4. Safety hazards
"""

# Fallback color detection parameters
COLOR_DETECTION_FALLBACK = {
    "red": {
        "hsv_lower": [0, 100, 100],
        "hsv_upper": [10, 255, 255]
    },
    "blue": {
        "hsv_lower": [100, 100, 100],
        "hsv_upper": [130, 255, 255]
    },
    "green": {
        "hsv_lower": [40, 100, 100],
        "hsv_upper": [80, 255, 255]
    },
    "yellow": {
        "hsv_lower": [20, 100, 100],
        "hsv_upper": [40, 255, 255]
    }
}
```

### Scene Understanding Prompt

```python
SCENE_UNDERSTANDING_PROMPT = """
Analyze the scene for high-level understanding and planning.

Image: [Provided via API]
Known Objects: {known_objects}
Task Goal: {task_goal}

Provide scene analysis:
{
  "scene_type": "tabletop|warehouse|kitchen|outdoor|unknown",
  "workspace_analysis": {
    "free_space_regions": [[x1,y1,x2,y2], ...],
    "occupied_regions": [[x1,y1,x2,y2], ...],
    "restricted_zones": [[x1,y1,x2,y2], ...]
  },
  "object_relationships": [
    {
      "type": "on_top_of|next_to|inside|blocked_by",
      "object1": "id1",
      "object2": "id2",
      "confidence": 0.0-1.0
    }
  ],
  "manipulation_order": [
    "If objects need to be moved in sequence, list the order"
  ],
  "potential_issues": [
    {
      "issue": "description",
      "severity": "low|medium|high",
      "mitigation": "suggested solution"
    }
  ],
  "confidence_scores": {
    "overall": 0.0-1.0,
    "object_detection": 0.0-1.0,
    "relationship_inference": 0.0-1.0
  }
}

Consider:
- Object stability
- Reachability
- Collision risks
- Task feasibility
"""
```

---

## Error Recovery Prompts

### Error Diagnosis Prompt

```python
ERROR_DIAGNOSIS_PROMPT = """
Diagnose the error that occurred during task execution.

Error Message: {error_message}
Error Type: {error_type}
System State at Error:
- Robot position: {robot_position}
- Gripper state: {gripper_state}
- Sensor readings: {sensor_data}
- Recent actions: {recent_actions}

Task Being Executed: {task_description}
Step That Failed: {failed_step}

Provide diagnosis:
{
  "error_category": "mechanical|perception|planning|control|external",
  "root_cause": "detailed explanation",
  "affected_components": ["list of affected parts"],
  "severity": "minor|moderate|critical",
  "recovery_options": [
    {
      "strategy": "retry|reset|alternative|abort",
      "description": "what to do",
      "success_probability": 0.0-1.0,
      "implementation": "code or steps"
    }
  ],
  "prevention": "how to prevent this in future"
}

Be specific and actionable in your recommendations.
"""
```

### Recovery Strategy Generation

```python
RECOVERY_STRATEGY_PROMPT = """
Generate a recovery strategy for the current error state.

Error: {error_description}
Current State: {current_state}
Original Goal: {original_goal}
Attempted Actions: {attempted_actions}

Generate recovery plan:
```python
def execute_recovery():
    '''
    Recovery strategy for {error_description}
    '''
    recovery_steps = [
        # Step 1: Safe state
        {
            'action': 'move_to_safe_position',
            'params': {'position': [x, y, z]},
            'verify': 'check_position_reached'
        },
        
        # Step 2: Reset components
        {
            'action': 'reset_gripper',
            'params': {},
            'verify': 'gripper_responsive'
        },
        
        # Step 3: Retry or alternative
        {
            'action': 'retry_with_adjustment|try_alternative|request_help',
            'params': {},
            'verify': 'task_progress'
        }
    ]
    
    for step in recovery_steps:
        result = execute_action(step['action'], step['params'])
        if not verify_condition(step['verify']):
            return False, "Recovery failed at: " + step['action']
    
    return True, "Recovery successful"
```

Ensure recovery is:
1. Safe
2. Incremental
3. Verifiable
4. Logged
"""
```

---

## Optimization Prompts

### Hyperparameter Tuning Prompt

```python
HYPERPARAMETER_PROMPT = """
Suggest hyperparameters for the CMA-ES optimization algorithm.

Problem Characteristics:
- Dimension: {dimension}
- Bounds: {bounds}
- Expected Difficulty: {difficulty}
- Time Budget: {time_budget} seconds
- Previous Best: {previous_best}

Provide hyperparameters:
{
  "population_size": "lambda (integer)",
  "mu": "number of parents (integer)",
  "sigma0": "initial step size (float)",
  "learning_rate": "float",
  "damping": "float",
  "weights_mode": "linear|equal|log",
  "adaptation_mode": "covariance|diagonal|none",
  "restart_strategy": {
    "enabled": true|false,
    "max_restarts": 0,
    "sigma_multiplier": 0.0
  },
  "termination_criteria": {
    "max_iterations": 0,
    "fitness_threshold": 0.0,
    "sigma_threshold": 0.0,
    "condition_number_threshold": 0.0
  },
  "rationale": "explanation of choices"
}

Consider:
- Higher population for multi-modal problems
- Smaller sigma for fine-tuning
- Adaptive strategies for unknown landscapes
- Time budget constraints
"""

# Fallback CMA-ES parameters
DEFAULT_CMAES_PARAMS = {
    "easy": {
        "population_size": 8,
        "sigma0": 0.5,
        "max_iterations": 100
    },
    "medium": {
        "population_size": 16,
        "sigma0": 0.3,
        "max_iterations": 200
    },
    "hard": {
        "population_size": 32,
        "sigma0": 0.2,
        "max_iterations": 500
    }
}
```

### Fitness Function Generation

```python
FITNESS_FUNCTION_PROMPT = """
Generate a fitness function for trajectory optimization.

Task: {task_description}
Optimization Goals: {goals}
Constraints: {constraints}
Reward Weights: {reward_weights}

Generate Python fitness function:
```python
def compute_fitness(trajectory, scene_state, config):
    '''
    Compute fitness score for trajectory.
    Lower is better (minimization).
    
    Args:
        trajectory: List of waypoints
        scene_state: Current scene information
        config: Configuration parameters
        
    Returns:
        fitness: Float score (lower is better)
        components: Dict of individual cost components
    '''
    
    # Initialize cost components
    costs = {
        'task_completion': 0.0,
        'path_length': 0.0,
        'smoothness': 0.0,
        'collision_penalty': 0.0,
        'time_cost': 0.0,
        'energy_cost': 0.0
    }
    
    # Task completion cost
    final_position = trajectory[-1]
    target_position = config['target']
    costs['task_completion'] = np.linalg.norm(
        final_position - target_position
    )
    
    # Path length cost
    for i in range(len(trajectory) - 1):
        costs['path_length'] += np.linalg.norm(
            trajectory[i+1] - trajectory[i]
        )
    
    # Smoothness cost (minimize jerk)
    if len(trajectory) > 2:
        for i in range(len(trajectory) - 2):
            acceleration = trajectory[i+2] - 2*trajectory[i+1] + trajectory[i]
            costs['smoothness'] += np.linalg.norm(acceleration)
    
    # Collision penalty
    for point in trajectory:
        if check_collision(point, scene_state['obstacles']):
            costs['collision_penalty'] += 100.0
    
    # Time cost (estimated)
    costs['time_cost'] = len(trajectory) * config['time_per_waypoint']
    
    # Combine with weights
    total_fitness = sum(
        costs[key] * config['weights'].get(key, 1.0)
        for key in costs
    )
    
    return total_fitness, costs

def check_collision(point, obstacles):
    '''Check if point collides with obstacles'''
    # Implementation here
    return False
```

Ensure fitness function:
1. Is differentiable where possible
2. Penalizes constraint violations heavily
3. Balances multiple objectives
4. Returns meaningful component breakdown
"""
```

---

## Meta-Prompts

### Prompt Improvement Meta-Prompt

```python
PROMPT_IMPROVEMENT_META = """
Analyze and improve the following prompt for better LLM performance.

Original Prompt:
{original_prompt}

Recent Outputs Quality:
- Success rate: {success_rate}%
- Common errors: {error_patterns}
- User feedback: {feedback}

Improve the prompt by:
1. Adding clarifying instructions
2. Providing better examples
3. Specifying output format more clearly
4. Removing ambiguities
5. Adding edge case handling

Provide:
{
  "improved_prompt": "the enhanced prompt text",
  "changes_made": [
    "list of specific improvements"
  ],
  "expected_improvements": [
    "what should get better"
  ],
  "validation_tests": [
    {
      "test_input": "example input",
      "expected_output": "what we want"
    }
  ]
}
"""
```

### Prompt Chaining Controller

```python
PROMPT_CHAIN_CONTROLLER = """
Orchestrate a chain of prompts to accomplish a complex task.

Final Goal: {final_goal}
Available Prompts: {available_prompts}
Current State: {current_state}

Design a prompt chain:
{
  "chain": [
    {
      "step": 1,
      "prompt_id": "prompt_name",
      "input_mapping": {
        "param1": "source.field",
        "param2": "previous_step.output"
      },
      "expected_output": "description",
      "fallback_prompt": "alternative_prompt_id",
      "validation": "how to verify success"
    }
  ],
  "error_handling": {
    "retry_limit": 3,
    "fallback_strategy": "description"
  },
  "expected_total_time": "seconds",
  "success_criteria": "final validation"
}
"""
```

---

## System Prompts

### Main System Prompt

```python
SYSTEM_PROMPT = """
You are Cogniforge, an advanced robotic control system that generates safe, efficient, and reliable robot behaviors.

Core Principles:
1. Safety First - Never generate actions that could cause harm
2. Robustness - Always provide fallback options
3. Efficiency - Optimize for task completion time and resource usage
4. Explainability - Provide clear reasoning for decisions
5. Adaptability - Learn from failures and improve

Capabilities:
- Natural language task understanding
- Trajectory planning and optimization
- Visual perception and scene understanding
- Error recovery and adaptive behavior
- Code generation for robot control

Constraints:
- Stay within workspace bounds
- Respect physical limits (joint limits, payload)
- Ensure stable grasps
- Avoid collisions
- Complete tasks within time budgets

Output Standards:
- Always return valid JSON when requested
- Include confidence scores
- Provide fallback options
- Log decision rationale
- Report potential issues proactively

Remember: You are controlling a physical robot. Every action must be safe, tested, and reversible.
"""
```

---

## Usage Examples

### Example 1: Complete Task Pipeline

```python
# Step 1: Parse task
task_prompt = TASK_ANALYSIS_PROMPT.format(
    task_description="Pick up the red cube and place it on the blue platform",
    detected_objects=["red_cube", "blue_platform", "green_sphere"],
    robot_position=[0.0, 0.0, 0.0],
    workspace_bounds=[[-1, 1], [-1, 1], [0, 2]]
)

# Step 2: Generate trajectory
trajectory_prompt = EXPERT_TRAJECTORY_PROMPT.format(
    task_description="Pick and place red cube on blue platform",
    robot_type="6-DOF manipulator",
    object_positions={"red_cube": [0.4, 0.0, 0.1], "blue_platform": [0.6, 0.2, 0.05]},
    obstacles=[],
    workspace_bounds=[[-1, 1], [-1, 1], [0, 2]]
)

# Step 3: If generation fails, use fallback
if trajectory_generation_failed:
    trajectory_code = FALLBACK_EXPERT_TRAJECTORIES["pick_place"]
```

### Example 2: Error Recovery

```python
# Detect error
error_prompt = ERROR_DIAGNOSIS_PROMPT.format(
    error_message="Gripper failed to close",
    error_type="GripperError",
    robot_position=[0.4, 0.0, 0.1],
    gripper_state="open",
    sensor_data={"force": 0.0, "current": 0.1},
    recent_actions=["move_to", "open_gripper", "close_gripper"],
    task_description="Pick up cube",
    failed_step="close_gripper"
)

# Generate recovery
recovery_prompt = RECOVERY_STRATEGY_PROMPT.format(
    error_description="Gripper not responding to close command",
    current_state={"gripper": "open", "position": [0.4, 0.0, 0.1]},
    original_goal="Pick up red cube",
    attempted_actions=["close_gripper", "close_gripper"]
)
```

---

## Prompt Optimization Guidelines

### 1. Clarity and Specificity
- Use concrete examples
- Define all technical terms
- Specify exact output format
- Include units for all measurements

### 2. Robustness
- Always include fallback instructions
- Handle edge cases explicitly
- Provide default values
- Include error handling

### 3. Safety
- Emphasize safety constraints
- Include workspace bounds
- Specify force/torque limits
- Add collision checking

### 4. Performance
- Set time budgets
- Limit output length when needed
- Request only necessary information
- Use temperature=0 for deterministic outputs

### 5. Validation
- Include success criteria
- Request confidence scores
- Add self-verification steps
- Enable output validation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-01 | Initial prompt set |
| 1.1.0 | 2024-02-01 | Added vision prompts |
| 1.2.0 | 2024-03-01 | Enhanced error recovery |
| 1.3.0 | 2024-04-01 | Optimization prompts added |
| 1.4.0 | 2024-05-01 | Meta-prompts for self-improvement |

---

## Notes

- All prompts are tested with GPT-4, Claude, and Llama models
- Fallback options are always available for system resilience
- Prompts are versioned and logged for reproducibility
- Regular evaluation ensures prompt quality
- User feedback is incorporated into prompt updates

For questions or improvements, contact: support@cogniforge.ai