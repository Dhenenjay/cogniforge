"""
GPT-5 prompt template and parser for generating reward function weights.

This module provides a standardized prompt format for GPT-5 to generate
reward weights and parsing utilities to integrate the response with the
reward system.
"""

import json
import re
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


# GPT-5 Prompt Template
GPT5_REWARD_PROMPT_TEMPLATE = """
You are a reward function designer for robotic manipulation tasks. Given a task description and available state variables, you must return ONLY a JSON dictionary with numerical weights for the reward function.

TASK DESCRIPTION:
{task_description}

AVAILABLE STATE VARIABLES:
- distance_to_target: Euclidean distance from end-effector to target object (meters)
- grasp_success: Boolean indicating if object is successfully grasped
- collision_detected: Boolean indicating if collision occurred
- task_completed: Boolean indicating if the overall task is complete
- time_elapsed: Number of simulation steps elapsed
- action_smoothness: Measure of action continuity (0-1, higher is smoother)
- orientation_alignment: Gripper alignment with object (0-1, 1 is perfect)

REWARD COMPONENTS TO WEIGHT:
You must provide a float value for each of these components:
- 'dist': Weight for distance_to_target (use NEGATIVE values for penalties)
- 'grasp': Weight for grasp_success (use POSITIVE values for rewards)
- 'collision': Weight for collision_detected (use NEGATIVE values for penalties)
- 'success': Weight for task_completed (use POSITIVE values for rewards)
- 'time': Weight for time_elapsed per step (use NEGATIVE values for penalties)
- 'smooth': Weight for action_smoothness (use POSITIVE values for rewards)
- 'orientation': Weight for orientation_alignment (use POSITIVE values for rewards)

INSTRUCTIONS:
1. Return ONLY a valid JSON object with the seven keys listed above
2. Use negative floats for penalties (things to minimize)
3. Use positive floats for rewards (things to maximize)
4. Scale weights based on task requirements
5. Consider task safety, efficiency, and precision

RESPONSE FORMAT:
{{"dist": -X.X, "grasp": X.X, "collision": -X.X, "success": X.X, "time": -X.X, "smooth": X.X, "orientation": X.X}}

YOUR RESPONSE (JSON ONLY):
"""


def create_gpt5_prompt(task_description: str) -> str:
    """
    Create a GPT-5 prompt for generating reward weights.
    
    Args:
        task_description: Natural language description of the robotic task
        
    Returns:
        Formatted prompt string for GPT-5
        
    Example:
        prompt = create_gpt5_prompt("Pick up the red cube carefully without dropping it")
        # Send prompt to GPT-5 API
        # response = gpt5_api.complete(prompt)
    """
    return GPT5_REWARD_PROMPT_TEMPLATE.format(task_description=task_description)


def parse_gpt_response(response: str) -> Dict[str, float]:
    """
    Parse GPT response to extract reward weights.
    
    This function robustly parses the GPT response, handling various formats
    and potential errors. It validates that all required keys are present
    and values are valid floats.
    
    Args:
        response: Raw text response from GPT
        
    Returns:
        Dictionary with validated reward weights
        
    Raises:
        ValueError: If response cannot be parsed or is invalid
        
    Example:
        response = '{"dist": -1.5, "grasp": 20.0, "collision": -10.0, ...}'
        weights = parse_gpt_response(response)
    """
    # Try to extract JSON from response
    json_match = re.search(r'\{[^}]+\}', response)
    if not json_match:
        raise ValueError(f"No valid JSON found in response: {response}")
    
    json_str = json_match.group(0)
    
    try:
        # Parse JSON
        weights = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON errors
        json_str = fix_common_json_errors(json_str)
        try:
            weights = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    # Validate required keys
    required_keys = {'dist', 'grasp', 'collision', 'success', 'time', 'smooth', 'orientation'}
    missing_keys = required_keys - set(weights.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    # Validate and convert values to floats
    validated_weights = {}
    for key in required_keys:
        try:
            value = float(weights[key])
            validated_weights[key] = value
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for '{key}': {weights[key]} - {e}")
    
    # Validate sign conventions (warnings only)
    if validated_weights['dist'] > 0:
        logger.warning("'dist' weight is positive - typically should be negative for distance penalty")
    if validated_weights['collision'] > 0:
        logger.warning("'collision' weight is positive - typically should be negative for collision penalty")
    if validated_weights['time'] > 0:
        logger.warning("'time' weight is positive - typically should be negative for time penalty")
    if validated_weights['grasp'] < 0:
        logger.warning("'grasp' weight is negative - typically should be positive for grasp reward")
    if validated_weights['success'] < 0:
        logger.warning("'success' weight is negative - typically should be positive for success reward")
    
    return validated_weights


def fix_common_json_errors(json_str: str) -> str:
    """
    Fix common JSON formatting errors in GPT responses.
    
    Args:
        json_str: Potentially malformed JSON string
        
    Returns:
        Fixed JSON string
    """
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Add missing quotes around keys
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    
    # Fix double-quoted keys
    json_str = re.sub(r'""(\w+)""', r'"\1"', json_str)
    
    return json_str


def _call_openai_for_weights(task_description: str) -> Optional[str]:
    """Call OpenAI Responses API to get weights as JSON text. Returns response text or None."""
    try:
        from cogniforge.core.config import settings
        client = settings.get_openai_client()

        system_prompt = (
            "You are GPT-5, a reward function designer. Return ONLY a JSON object with weights "
            "for keys: dist, grasp, collision, success, time, smooth, orientation."
        )
        prompt = create_gpt5_prompt(task_description)
        inputs = [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ]
        resp = client.responses.create(
            model="gpt-5",
            input=inputs,
            response_format={"type": "json_object"},
            max_output_tokens=300,
            temperature=0.1,
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        # Fallback parse
        try:
            parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if hasattr(c, "text") and c.text:
                        parts.append(c.text)
            return "\n".join(parts) if parts else None
        except Exception:
            pass
        if hasattr(resp, "choices"):
            try:
                return resp.choices[0].message.content
            except Exception:
                return None
        return None
    except Exception:
        return None


def generate_and_parse_weights(
    task_description: str,
    gpt_response_function: Optional[callable] = None,
    default_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Generate prompt, get GPT response, and parse weights.
    
    This is the main integration function that handles the complete workflow
    from task description to parsed weights ready for the reward function.
    
    Args:
        task_description: Natural language task description
        gpt_response_function: Function to get GPT response (for testing/mocking)
        default_weights: Fallback weights if GPT fails
        
    Returns:
        Dictionary of reward weights
        
    Example:
        # With actual GPT API
        def gpt_api_call(prompt):
            return openai.complete(prompt)
        
        weights = generate_and_parse_weights(
            "Stack blocks carefully",
            gpt_response_function=gpt_api_call
        )
        
        # Use weights in reward function
        from cogniforge.core.reward import compute_reward
        reward = compute_reward(state, action, next_state, info, weights)
    """
    # Create prompt
    prompt = create_gpt5_prompt(task_description)
    logger.info(f"Generated GPT-5 prompt for task: {task_description}")
    
    # Get response (use provided function or return mock)
    if gpt_response_function:
        try:
            response = gpt_response_function(prompt)
            logger.info("Received GPT response")
        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            if default_weights:
                logger.info("Using default weights")
                return default_weights
            raise
    else:
        # Try real API first; fall back to mock
        response = _call_openai_for_weights(task_description)
        if response is None:
            logger.warning("OpenAI not configured or call failed; using mock response")
            response = '{"dist": -1.0, "grasp": 10.0, "collision": -5.0, "success": 100.0, "time": -0.1, "smooth": 0.5, "orientation": 2.0}'
    
    # Parse response
    try:
        weights = parse_gpt_response(response)
        logger.info(f"Successfully parsed weights: {weights}")
        return weights
    except ValueError as e:
        logger.error(f"Failed to parse GPT response: {e}")
        if default_weights:
            logger.info("Using default weights")
            return default_weights
        raise


def integrate_with_reward_system(
    task_description: str,
    state: Dict[str, Any],
    action: Any,
    next_state: Dict[str, Any],
    info: Dict[str, Any],
    gpt_response_function: Optional[callable] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Complete integration from task description to reward computation.
    
    This function handles the entire pipeline:
    1. Generate GPT prompt from task description
    2. Get GPT response with weights
    3. Parse and validate weights
    4. Compute reward using the weights
    
    Args:
        task_description: Natural language task description
        state: Current state dictionary
        action: Action taken
        next_state: Resulting state
        info: Additional information
        gpt_response_function: Function to get GPT response
        
    Returns:
        Tuple of (reward_value, weights_used)
        
    Example:
        reward, weights = integrate_with_reward_system(
            "Pick up the fragile vase very carefully",
            current_state,
            robot_action,
            next_state,
            info_dict
        )
        print(f"Reward: {reward}, Weights: {weights}")
    """
    from cogniforge.core.reward import compute_reward
    
    # Get weights from GPT
    weights = generate_and_parse_weights(
        task_description,
        gpt_response_function=gpt_response_function,
        default_weights={
            'dist': -1.0,
            'grasp': 10.0,
            'collision': -5.0,
            'success': 100.0,
            'time': -0.1,
            'smooth': 0.5,
            'orientation': 2.0
        }
    )
    
    # Compute reward
    reward = compute_reward(state, action, next_state, info, weights)
    
    return reward, weights


# Task-specific prompt templates for better results
TASK_SPECIFIC_TEMPLATES = {
    'careful': {
        'hint': "Prioritize safety and precision over speed",
        'dist_range': (-3.0, -1.0),
        'collision_range': (-50.0, -20.0),
        'time_range': (-0.05, -0.01)
    },
    'fast': {
        'hint': "Prioritize speed and efficiency",
        'dist_range': (-5.0, -2.0),
        'collision_range': (-5.0, -2.0),
        'time_range': (-1.0, -0.5)
    },
    'precise': {
        'hint': "Prioritize accuracy and alignment",
        'dist_range': (-4.0, -2.0),
        'collision_range': (-20.0, -10.0),
        'orientation_range': (5.0, 15.0)
    },
    'fragile': {
        'hint': "Handle with extreme care, avoid any collisions",
        'collision_range': (-100.0, -50.0),
        'smooth_range': (2.0, 5.0),
        'time_range': (-0.02, -0.01)
    }
}


def create_enhanced_prompt(
    task_description: str,
    task_type: Optional[str] = None,
    custom_constraints: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create an enhanced GPT-5 prompt with task-specific hints.
    
    Args:
        task_description: Natural language task description
        task_type: Type of task ('careful', 'fast', 'precise', 'fragile')
        custom_constraints: Additional constraints or hints
        
    Returns:
        Enhanced prompt string
        
    Example:
        prompt = create_enhanced_prompt(
            "Stack the blocks",
            task_type='precise',
            custom_constraints={'max_time': 100}
        )
    """
    base_prompt = create_gpt5_prompt(task_description)
    
    # Add task-specific hints
    if task_type and task_type in TASK_SPECIFIC_TEMPLATES:
        template = TASK_SPECIFIC_TEMPLATES[task_type]
        hint_text = f"\nTASK TYPE HINT: {template['hint']}\n"
        
        # Add range suggestions
        range_hints = []
        for key, value in template.items():
            if key.endswith('_range'):
                component = key.replace('_range', '')
                range_hints.append(f"- {component}: typically {value[0]} to {value[1]}")
        
        if range_hints:
            hint_text += "SUGGESTED WEIGHT RANGES:\n" + "\n".join(range_hints) + "\n"
        
        # Insert hints before the response format section
        base_prompt = base_prompt.replace(
            "RESPONSE FORMAT:",
            hint_text + "\nRESPONSE FORMAT:"
        )
    
    # Add custom constraints
    if custom_constraints:
        constraint_text = "\nADDITIONAL CONSTRAINTS:\n"
        for key, value in custom_constraints.items():
            constraint_text += f"- {key}: {value}\n"
        
        base_prompt = base_prompt.replace(
            "RESPONSE FORMAT:",
            constraint_text + "\nRESPONSE FORMAT:"
        )
    
    return base_prompt


# Example presets for common scenarios
EXAMPLE_RESPONSES = {
    'pick_and_place': {
        'dist': -2.0,
        'grasp': 20.0,
        'collision': -5.0,
        'success': 100.0,
        'time': -0.1,
        'smooth': 1.0,
        'orientation': 3.0
    },
    'careful_stacking': {
        'dist': -3.0,
        'grasp': 15.0,
        'collision': -30.0,
        'success': 150.0,
        'time': -0.02,
        'smooth': 3.0,
        'orientation': 10.0
    },
    'fast_sorting': {
        'dist': -5.0,
        'grasp': 10.0,
        'collision': -3.0,
        'success': 100.0,
        'time': -0.5,
        'smooth': 0.2,
        'orientation': 1.0
    }
}


def validate_weight_ranges(weights: Dict[str, float]) -> List[str]:
    """
    Validate that weights are within reasonable ranges.
    
    Args:
        weights: Dictionary of reward weights
        
    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []
    
    # Check penalty ranges
    if weights['dist'] < -10 or weights['dist'] > 0:
        warnings.append(f"'dist' weight {weights['dist']} outside typical range [-10, 0]")
    
    if weights['collision'] < -100 or weights['collision'] > 0:
        warnings.append(f"'collision' weight {weights['collision']} outside typical range [-100, 0]")
    
    if weights['time'] < -2 or weights['time'] > 0:
        warnings.append(f"'time' weight {weights['time']} outside typical range [-2, 0]")
    
    # Check reward ranges
    if weights['grasp'] < 0 or weights['grasp'] > 100:
        warnings.append(f"'grasp' weight {weights['grasp']} outside typical range [0, 100]")
    
    if weights['success'] < 0 or weights['success'] > 500:
        warnings.append(f"'success' weight {weights['success']} outside typical range [0, 500]")
    
    if weights['smooth'] < 0 or weights['smooth'] > 10:
        warnings.append(f"'smooth' weight {weights['smooth']} outside typical range [0, 10]")
    
    if weights['orientation'] < 0 or weights['orientation'] > 20:
        warnings.append(f"'orientation' weight {weights['orientation']} outside typical range [0, 20]")
    
    return warnings


def safely_parse_json_weights(
    s: str,
    min_value: float = -10.0,
    max_value: float = 10.0,
    strict_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    clamp: bool = True,
    raise_on_invalid: bool = False
) -> Dict[str, float]:
    """
    Safely parse JSON weights with numeric range validation and clamping.
    
    This function parses JSON string containing reward weights and ensures
    all values are within safe numeric ranges. It can either clamp values
    to the allowed range or raise an error for out-of-range values.
    
    Args:
        s: JSON string containing weight dictionary
        min_value: Global minimum allowed value (default: -10.0)
        max_value: Global maximum allowed value (default: 10.0)
        strict_ranges: Optional dict with specific ranges for each key.
                      Format: {'key': (min, max)}
        clamp: If True, clamp values to allowed range. If False, use as-is.
        raise_on_invalid: If True, raise ValueError for out-of-range values.
                         If False, log warning and clamp/use value.
    
    Returns:
        Dictionary with validated and potentially clamped weight values
        
    Raises:
        ValueError: If JSON parsing fails or values are out of range
                   (when raise_on_invalid=True)
        
    Example:
        # Basic usage with global range [-10, 10]
        weights = safely_parse_json_weights(
            '{"dist": -15.0, "grasp": 20.0}'
        )
        # Returns: {'dist': -10.0, 'grasp': 10.0}  # Clamped to range
        
        # With specific ranges per component
        ranges = {
            'dist': (-5.0, 0.0),
            'grasp': (0.0, 50.0),
            'collision': (-20.0, 0.0),
            'success': (0.0, 200.0),
            'time': (-1.0, 0.0),
            'smooth': (0.0, 5.0),
            'orientation': (0.0, 10.0)
        }
        weights = safely_parse_json_weights(
            json_str,
            strict_ranges=ranges
        )
        
        # Strict validation (raises error instead of clamping)
        try:
            weights = safely_parse_json_weights(
                '{"dist": -100.0}',
                raise_on_invalid=True
            )
        except ValueError as e:
            print(f"Invalid weight: {e}")
    """
    # Try to extract and parse JSON
    try:
        # First try direct JSON parsing
        weights = json.loads(s)
    except json.JSONDecodeError:
        # Try to extract JSON from surrounding text
        json_match = re.search(r'\{[^}]+\}', s)
        if not json_match:
            raise ValueError(f"No valid JSON found in string: {s[:100]}...")
        
        json_str = json_match.group(0)
        
        # Try to fix common errors
        json_str = fix_common_json_errors(json_str)
        
        try:
            weights = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON after fixes: {e}")
    
    # Validate it's a dictionary
    if not isinstance(weights, dict):
        raise ValueError(f"Parsed JSON is not a dictionary: {type(weights)}")
    
    # Define default strict ranges if not provided
    if strict_ranges is None:
        strict_ranges = {
            'dist': (-10.0, 0.0),      # Distance penalty
            'grasp': (0.0, 100.0),      # Grasp reward
            'collision': (-100.0, 0.0), # Collision penalty
            'success': (0.0, 500.0),    # Success reward
            'time': (-2.0, 0.0),        # Time penalty
            'smooth': (0.0, 10.0),      # Smoothness reward
            'orientation': (0.0, 20.0)  # Orientation reward
        }
    
    # Validate and process each weight
    validated_weights = {}
    required_keys = {'dist', 'grasp', 'collision', 'success', 'time', 'smooth', 'orientation'}
    
    for key in required_keys:
        if key not in weights:
            if raise_on_invalid:
                raise ValueError(f"Missing required key '{key}' in weights")
            else:
                logger.warning(f"Missing key '{key}', using 0.0")
                validated_weights[key] = 0.0
                continue
        
        # Convert to float
        try:
            value = float(weights[key])
        except (TypeError, ValueError) as e:
            if raise_on_invalid:
                raise ValueError(f"Invalid value for '{key}': {weights[key]} - {e}")
            else:
                logger.warning(f"Invalid value for '{key}': {weights[key]}, using 0.0")
                validated_weights[key] = 0.0
                continue
        
        # Check NaN and Inf
        if not np.isfinite(value):
            if raise_on_invalid:
                raise ValueError(f"Non-finite value for '{key}': {value}")
            else:
                logger.warning(f"Non-finite value for '{key}': {value}, using 0.0")
                validated_weights[key] = 0.0
                continue
        
        # Determine the range to use
        if key in strict_ranges:
            min_val, max_val = strict_ranges[key]
        else:
            min_val, max_val = min_value, max_value
        
        # Check if value is within range
        if value < min_val or value > max_val:
            msg = f"Weight '{key}' = {value} outside range [{min_val}, {max_val}]"
            
            if raise_on_invalid:
                raise ValueError(msg)
            elif clamp:
                clamped_value = np.clip(value, min_val, max_val)
                logger.warning(f"{msg}, clamping to {clamped_value}")
                validated_weights[key] = float(clamped_value)
            else:
                logger.warning(f"{msg}, using as-is")
                validated_weights[key] = value
        else:
            validated_weights[key] = value
    
    # Additional semantic validation
    _validate_semantic_consistency(validated_weights)
    
    return validated_weights


def _validate_semantic_consistency(weights: Dict[str, float]):
    """
    Validate semantic consistency of weights.
    
    Logs warnings for semantically inconsistent weight combinations.
    
    Args:
        weights: Dictionary of validated weights
    """
    # Check for inconsistent penalty/reward signs
    if weights.get('dist', 0) > 0:
        logger.info("Note: 'dist' weight is positive - treating distance as reward")
    
    if weights.get('collision', 0) > 0:
        logger.info("Note: 'collision' weight is positive - treating collision as reward")
    
    if weights.get('grasp', 0) < 0:
        logger.info("Note: 'grasp' weight is negative - treating grasp as penalty")
    
    if weights.get('success', 0) < 0:
        logger.info("Note: 'success' weight is negative - treating success as penalty")
    
    # Check for all-zero weights
    if all(v == 0 for v in weights.values()):
        logger.warning("All weights are zero - reward will always be 0")
    
    # Check for extreme imbalance
    max_abs = max(abs(v) for v in weights.values())
    min_abs = min(abs(v) for v in weights.values() if v != 0) if any(v != 0 for v in weights.values()) else 0
    
    if max_abs > 0 and min_abs > 0 and max_abs / min_abs > 1000:
        logger.warning(f"Extreme weight imbalance detected: ratio {max_abs/min_abs:.1f}")


def create_safe_weight_parser(
    task_type: str = 'general',
    safety_level: str = 'normal'
) -> callable:
    """
    Create a configured safe parser function for specific task types.
    
    Args:
        task_type: Type of task ('general', 'careful', 'fast', 'training')
        safety_level: Safety level ('strict', 'normal', 'relaxed')
        
    Returns:
        Configured parser function
        
    Example:
        # Create parser for careful manipulation
        parser = create_safe_weight_parser('careful', 'strict')
        weights = parser(json_string)
    """
    # Define ranges based on task type
    task_ranges = {
        'general': {
            'dist': (-10.0, 0.0),
            'grasp': (0.0, 100.0),
            'collision': (-100.0, 0.0),
            'success': (0.0, 500.0),
            'time': (-2.0, 0.0),
            'smooth': (0.0, 10.0),
            'orientation': (0.0, 20.0)
        },
        'careful': {
            'dist': (-5.0, 0.0),
            'grasp': (0.0, 50.0),
            'collision': (-200.0, 0.0),  # Higher penalty allowed
            'success': (0.0, 200.0),
            'time': (-0.1, 0.0),         # Low time pressure
            'smooth': (0.0, 20.0),        # Higher smoothness reward
            'orientation': (0.0, 30.0)
        },
        'fast': {
            'dist': (-20.0, 0.0),        # Higher distance penalty
            'grasp': (0.0, 50.0),
            'collision': (-20.0, 0.0),   # Lower collision penalty
            'success': (0.0, 300.0),
            'time': (-5.0, 0.0),         # Higher time penalty
            'smooth': (0.0, 5.0),
            'orientation': (0.0, 10.0)
        },
        'training': {
            'dist': (-100.0, 100.0),     # Allow exploration
            'grasp': (-100.0, 100.0),
            'collision': (-100.0, 100.0),
            'success': (-100.0, 1000.0),
            'time': (-10.0, 10.0),
            'smooth': (-10.0, 10.0),
            'orientation': (-10.0, 50.0)
        }
    }
    
    # Get ranges for task type
    ranges = task_ranges.get(task_type, task_ranges['general'])
    
    # Configure based on safety level
    if safety_level == 'strict':
        clamp = True
        raise_on_invalid = True
    elif safety_level == 'normal':
        clamp = True
        raise_on_invalid = False
    else:  # relaxed
        clamp = False
        raise_on_invalid = False
    
    # Return configured parser
    def parser(s: str) -> Dict[str, float]:
        return safely_parse_json_weights(
            s,
            strict_ranges=ranges,
            clamp=clamp,
            raise_on_invalid=raise_on_invalid
        )
    
    return parser


def batch_parse_weights(
    json_strings: List[str],
    min_value: float = -10.0,
    max_value: float = 10.0,
    return_errors: bool = False
) -> Union[List[Dict[str, float]], Tuple[List[Dict[str, float]], List[str]]]:
    """
    Parse multiple JSON weight strings with validation.
    
    Args:
        json_strings: List of JSON strings to parse
        min_value: Global minimum allowed value
        max_value: Global maximum allowed value
        return_errors: If True, also return list of error messages
        
    Returns:
        List of weight dictionaries (or tuple with errors if requested)
        
    Example:
        json_list = [
            '{"dist": -1.0, "grasp": 10.0, ...}',
            '{"dist": -2.0, "grasp": 20.0, ...}'
        ]
        weights_list = batch_parse_weights(json_list)
    """
    results = []
    errors = []
    
    for i, json_str in enumerate(json_strings):
        try:
            weights = safely_parse_json_weights(
                json_str,
                min_value=min_value,
                max_value=max_value,
                clamp=True,
                raise_on_invalid=False
            )
            results.append(weights)
            errors.append(None)
        except Exception as e:
            logger.error(f"Failed to parse weight string {i}: {e}")
            # Use default weights on failure
            results.append({
                'dist': -1.0,
                'grasp': 10.0,
                'collision': -5.0,
                'success': 100.0,
                'time': -0.1,
                'smooth': 0.5,
                'orientation': 2.0
            })
            errors.append(str(e))
    
    if return_errors:
        return results, errors
    return results


# Testing and examples
if __name__ == "__main__":
    print("GPT-5 Reward Weight Generation System")
    print("=" * 60)
    
    # Example 1: Create basic prompt
    print("\nExample 1: Basic Prompt Generation")
    print("-" * 40)
    task = "Pick up the red cube and place it on the blue platform"
    prompt = create_gpt5_prompt(task)
    print(f"Task: {task}")
    print(f"\nGenerated Prompt (first 500 chars):")
    print(prompt[:500] + "...")
    
    # Example 2: Parse mock response
    print("\n\nExample 2: Response Parsing")
    print("-" * 40)
    mock_response = """
    Based on the task, here are the appropriate weights:
    {"dist": -2.0, "grasp": 25.0, "collision": -10.0, "success": 150.0, "time": -0.08, "smooth": 1.5, "orientation": 5.0}
    """
    
    try:
        weights = parse_gpt_response(mock_response)
        print("Successfully parsed weights:")
        for key, value in weights.items():
            print(f"  {key}: {value}")
    except ValueError as e:
        print(f"Parse error: {e}")
    
    # Example 3: Enhanced prompt with task type
    print("\n\nExample 3: Enhanced Prompt with Task Type")
    print("-" * 40)
    task = "Stack the fragile glass blocks"
    enhanced_prompt = create_enhanced_prompt(
        task,
        task_type='fragile',
        custom_constraints={'max_height': 5, 'time_limit': None}
    )
    print(f"Task: {task}")
    print(f"Task Type: fragile")
    print(f"\nEnhanced sections added to prompt")
    
    # Example 4: Validate weights
    print("\n\nExample 4: Weight Validation")
    print("-" * 40)
    test_weights = {
        'dist': -15.0,  # Too low
        'grasp': 150.0,  # Too high
        'collision': -5.0,
        'success': 100.0,
        'time': -0.1,
        'smooth': 0.5,
        'orientation': 2.0
    }
    
    warnings = validate_weight_ranges(test_weights)
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    else:
        print("All weights within typical ranges ✓")
    
    # Example 5: Safe parsing with range validation
    print("\n\nExample 5: Safe JSON Weight Parsing")
    print("-" * 40)
    
    # Test cases for safe parsing
    test_cases = [
        (
            '{"dist": -15.0, "grasp": 120.0, "collision": -5.0, "success": 100.0, "time": -0.1, "smooth": 0.5, "orientation": 2.0}',
            "Out of range values (will be clamped)"
        ),
        (
            '{"dist": -1.0, "grasp": "invalid", "collision": -5.0, "success": 100.0, "time": -0.1, "smooth": 0.5, "orientation": 2.0}',
            "Invalid value type"
        ),
        (
            'Some text before {"dist": -2.0, "grasp": 10.0, "collision": -5.0, "success": 100.0, "time": -0.1, "smooth": 0.5, "orientation": 2.0} and after',
            "JSON embedded in text"
        ),
    ]
    
    for json_str, description in test_cases:
        print(f"\n  Test: {description}")
        try:
            weights = safely_parse_json_weights(json_str)
            print(f"  ✓ Successfully parsed:")
            for key, value in weights.items():
                print(f"    {key}: {value}")
        except Exception as e:
            print(f"  ✗ Parse error: {e}")
    
    # Test with custom ranges
    print("\n  Test: Custom ranges for careful task")
    custom_ranges = {
        'dist': (-5.0, 0.0),
        'grasp': (0.0, 50.0),
        'collision': (-50.0, 0.0),
        'success': (0.0, 200.0),
        'time': (-0.5, 0.0),
        'smooth': (0.0, 5.0),
        'orientation': (0.0, 10.0)
    }
    
    json_str = '{"dist": -10.0, "grasp": 100.0, "collision": -100.0, "success": 300.0, "time": -1.0, "smooth": 10.0, "orientation": 20.0}'
    weights = safely_parse_json_weights(json_str, strict_ranges=custom_ranges)
    print(f"  Parsed with custom ranges (clamped):")
    for key, value in weights.items():
        print(f"    {key}: {value}")
    
    # Example 6: Complete integration example
    print("\n\nExample 6: Complete Integration")
    print("-" * 40)
    
    # Mock state and action
    state = {'robot_pose': ([0.5, 0, 0.3], [0, 0, 0, 1])}
    action = [0.1, 0, -0.05]
    next_state = {'robot_pose': ([0.55, 0, 0.25], [0, 0, 0, 1])}
    info = {
        'distance_to_target': 0.15,
        'grasp_success': False,
        'collision': False,
        'task_success': False,
        'orientation_alignment': 0.8
    }
    
    # Mock GPT function
    def mock_gpt(prompt):
        return '{"dist": -1.5, "grasp": 20.0, "collision": -8.0, "success": 120.0, "time": -0.1, "smooth": 0.8, "orientation": 4.0}'
    
    try:
        reward, used_weights = integrate_with_reward_system(
            "Carefully grasp the delicate object",
            state, action, next_state, info,
            gpt_response_function=mock_gpt
        )
        print(f"Computed reward: {reward:.3f}")
        print("Weights used:")
        for key, value in used_weights.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Integration error: {e}")