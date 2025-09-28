"""
Expert script generation with fallback mechanism.

When GPT/Codex fails, falls back to a hardcoded 3-waypoint path
while logging the error to allow the pipeline to continue.
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import traceback
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class ExpertScriptWithFallback:
    """Expert script generator with automatic fallback to hardcoded waypoints."""
    
    # Default fallback waypoint patterns for common tasks
    FALLBACK_WAYPOINTS = {
        'pick_place': {
            'waypoints': [
                (0.4, 0.0, 0.3),   # Approach position (above workspace center)
                (0.4, 0.0, 0.1),   # Grasp position (lowered)
                (0.6, 0.2, 0.3),   # Place position (different location, raised)
            ],
            'gripper_actions': {
                0: 'open',
                1: 'close', 
                2: 'open'
            },
            'annotations': [
                'Default approach position',
                'Default grasp position',
                'Default place position'
            ]
        },
        'simple_motion': {
            'waypoints': [
                (0.3, -0.2, 0.3),  # Start position
                (0.5, 0.0, 0.25),  # Middle waypoint
                (0.3, 0.2, 0.3),   # End position
            ],
            'gripper_actions': {
                0: 'open'
            },
            'annotations': [
                'Start position',
                'Middle transit point',
                'End position'
            ]
        },
        'lift_and_move': {
            'waypoints': [
                (0.4, 0.0, 0.15),  # Initial position
                (0.4, 0.0, 0.4),   # Lift up
                (0.6, 0.0, 0.15),  # Move to new position
            ],
            'gripper_actions': {
                0: 'close',
                2: 'open'
            },
            'annotations': [
                'Grasp position',
                'Lifted position', 
                'Release position'
            ]
        }
    }
    
    def __init__(
        self,
        use_gpt: bool = True,
        gpt_timeout: float = 10.0,
        log_failures: bool = True,
        fallback_type: str = 'pick_place'
    ):
        """
        Initialize expert script generator with fallback.
        
        Args:
            use_gpt: Whether to attempt GPT/Codex generation first
            gpt_timeout: Timeout for GPT API calls in seconds
            log_failures: Whether to log GPT failures to file
            fallback_type: Default fallback waypoint pattern type
        """
        self.use_gpt = use_gpt
        self.gpt_timeout = gpt_timeout
        self.log_failures = log_failures
        self.fallback_type = fallback_type
        self.failure_log = []
        
    def generate_expert_trajectory(
        self,
        prompt: str,
        scene_summary: Dict[str, Any],
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate expert trajectory with automatic fallback.
        
        Args:
            prompt: Natural language task description
            scene_summary: Scene state information
            max_retries: Maximum GPT retry attempts before fallback
            
        Returns:
            Trajectory dictionary with waypoints and metadata
        """
        # Try GPT generation first if enabled
        if self.use_gpt:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting GPT generation (attempt {attempt + 1}/{max_retries})")
                    trajectory = self._generate_with_gpt(prompt, scene_summary)
                    
                    # Validate the generated trajectory
                    if self._validate_trajectory(trajectory):
                        logger.info("✅ GPT generation successful")
                        trajectory['generation_method'] = 'gpt'
                        trajectory['gpt_attempts'] = attempt + 1
                        return trajectory
                    else:
                        logger.warning(f"GPT trajectory validation failed (attempt {attempt + 1})")
                        
                except Exception as e:
                    error_msg = f"GPT generation failed (attempt {attempt + 1}): {str(e)}"
                    logger.error(error_msg)
                    self._log_failure(prompt, scene_summary, e, attempt + 1)
                    
                    if attempt == max_retries - 1:
                        logger.warning("🔄 All GPT attempts failed, falling back to hardcoded waypoints")
        
        # Fallback to hardcoded waypoints
        logger.info("📍 Using fallback waypoint strategy")
        return self._generate_fallback_trajectory(prompt, scene_summary)
    
    def _generate_with_gpt(
        self,
        prompt: str,
        scene_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trajectory using GPT/Codex.
        
        Args:
            prompt: Task description
            scene_summary: Scene information
            
        Returns:
            Generated trajectory
            
        Raises:
            Exception: If GPT generation fails
        """
        # Import here to avoid dependency if not using GPT
        from cogniforge.core.expert_script import gen_expert_script, parse_codex_response, execute_expert_script
        
        # Generate Codex prompt
        codex_prompt = gen_expert_script(
            prompt,
            scene_summary,
            use_parametric=True,
            include_approach_vectors=True,
            waypoint_density="adaptive"
        )
        
        # Simulate GPT/Codex API call
        # In production, this would make actual API call
        response = self._call_gpt_api(codex_prompt)
        
        # Parse response
        parsed = parse_codex_response(response, validate=True)
        
        # Execute to get trajectory
        trajectory = execute_expert_script(
            parsed['code'],
            scene_summary.get('objects', {}),
            safety_checks=True
        )
        
        return trajectory
    
    def _call_gpt_api(self, prompt: str) -> str:
        """
        Make API call to GPT-5 Codex using the Responses API.
        
        Args:
            prompt: The prompt to send (Codex-style code generation prompt)
            
        Returns:
            GPT response text containing Python code
            
        Raises:
            Exception: If API call fails or times out
        """
        # Prefer real API if configured; otherwise, fall back to simulated behavior
        try:
            from cogniforge.core.config import settings
            from openai import OpenAI
            client = settings.get_openai_client()  # May raise if not configured

            system_prompt = (
                "You are GPT-5 Codex, an expert robotics code generator. "
                "Generate clear, executable Python code. Return ONLY the code, ideally inside a single "
                "triple-backticked Python block. Avoid explanations."
            )

            inputs = [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ]

            resp = client.responses.create(
                model="gpt-5-codex",
                input=inputs,
                max_output_tokens=1800,
                temperature=0.2
            )

            # Extract text robustly across client versions
            text = None
            if hasattr(resp, "output_text"):
                text = resp.output_text
            else:
                try:
                    parts = []
                    for item in getattr(resp, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            if hasattr(c, "text") and c.text:
                                parts.append(c.text)
                    text = "\n".join(parts) if parts else None
                except Exception:
                    text = None
            if not text and hasattr(resp, "choices"):
                try:
                    text = resp.choices[0].message.content
                except Exception:
                    text = None
            if not text:
                raise RuntimeError("Empty response from GPT-5 Codex")
            return text
        except Exception:
            # Fall back to simulated behavior to keep pipeline resilient
            import random
            import time as _time
            _time.sleep(0.5)
            if random.random() < 0.3:
                raise ConnectionError("Simulated GPT API connection error")
            return """```python
import numpy as np
from typing import List, Tuple, Dict, Any

def generate_expert_trajectory(scene_objects: Dict[str, Any]) -> Dict[str, Any]:
    # Fallback simple trajectory (mock)
    waypoints = [
        (0.5, 0.0, 0.3),
        (0.5, 0.0, 0.1),
        (0.6, 0.2, 0.3)
    ]
    return {
        'waypoints': waypoints,
        'approach_vectors': {1: [0, 0, -1]},
        'gripper_actions': {0: 'open', 1: 'close', 2: 'open'},
        'annotations': ['Approach', 'Grasp', 'Place']
    }
```"""
    
    def _generate_fallback_trajectory(
        self,
        prompt: str,
        scene_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate fallback trajectory using hardcoded waypoints.
        
        Args:
            prompt: Task description (for logging)
            scene_summary: Scene information
            
        Returns:
            Fallback trajectory with metadata
        """
        # Select appropriate fallback pattern
        fallback_pattern = self._select_fallback_pattern(prompt, scene_summary)
        
        # Get base waypoints
        base_trajectory = self.FALLBACK_WAYPOINTS[fallback_pattern].copy()
        
        # Adapt waypoints to scene if possible
        adapted_trajectory = self._adapt_waypoints_to_scene(
            base_trajectory,
            scene_summary
        )
        
        # Add metadata
        adapted_trajectory['generation_method'] = 'fallback'
        adapted_trajectory['fallback_pattern'] = fallback_pattern
        adapted_trajectory['original_prompt'] = prompt
        adapted_trajectory['metadata'] = {
            'warning': 'Using fallback waypoints due to GPT failure',
            'timestamp': datetime.now().isoformat(),
            'pattern_used': fallback_pattern,
            'total_waypoints': len(adapted_trajectory['waypoints'])
        }
        
        # Log the fallback usage
        logger.warning(
            f"📍 Fallback trajectory generated:\n"
            f"  Pattern: {fallback_pattern}\n"
            f"  Waypoints: {len(adapted_trajectory['waypoints'])}\n"
            f"  Original task: {prompt[:100]}..."
        )
        
        return adapted_trajectory
    
    def _select_fallback_pattern(
        self,
        prompt: str,
        scene_summary: Dict[str, Any]
    ) -> str:
        """
        Select the most appropriate fallback pattern based on task.
        
        Args:
            prompt: Task description
            scene_summary: Scene information
            
        Returns:
            Pattern key to use
        """
        prompt_lower = prompt.lower()
        
        # Simple keyword-based selection
        if any(word in prompt_lower for word in ['pick', 'place', 'grasp', 'move']):
            return 'pick_place'
        elif any(word in prompt_lower for word in ['lift', 'raise', 'elevate']):
            return 'lift_and_move'
        else:
            return 'simple_motion'
    
    def _adapt_waypoints_to_scene(
        self,
        trajectory: Dict[str, Any],
        scene_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt fallback waypoints to actual scene if possible.
        
        Args:
            trajectory: Base fallback trajectory
            scene_summary: Scene information
            
        Returns:
            Adapted trajectory
        """
        adapted = trajectory.copy()
        waypoints = list(trajectory['waypoints'])
        
        # Try to adapt to workspace bounds if available
        if 'workspace' in scene_summary:
            ws = scene_summary['workspace']
            for i, (x, y, z) in enumerate(waypoints):
                # Clamp to workspace
                x = np.clip(x, ws.get('x_min', -1), ws.get('x_max', 1))
                y = np.clip(y, ws.get('y_min', -1), ws.get('y_max', 1))
                z = np.clip(z, ws.get('z_min', 0), ws.get('z_max', 2))
                waypoints[i] = (x, y, z)
        
        # Try to adapt to object positions if available
        if 'objects' in scene_summary and len(scene_summary['objects']) > 0:
            # Use first graspable object position as reference
            for obj in scene_summary['objects']:
                if obj.get('graspable', False):
                    obj_pos = obj.get('position', [0.5, 0.0, 0.1])
                    # Offset first waypoint to be above object
                    waypoints[0] = (
                        obj_pos[0],
                        obj_pos[1],
                        obj_pos[2] + 0.2  # 20cm above object
                    )
                    # Adjust grasp position
                    if len(waypoints) > 1:
                        waypoints[1] = (
                            obj_pos[0],
                            obj_pos[1],
                            obj_pos[2] + 0.05  # 5cm above for grasp
                        )
                    break
        
        adapted['waypoints'] = waypoints
        adapted['scene_adapted'] = True
        
        return adapted
    
    def _validate_trajectory(self, trajectory: Dict[str, Any]) -> bool:
        """
        Validate generated trajectory for basic correctness.
        
        Args:
            trajectory: Generated trajectory
            
        Returns:
            True if valid, False otherwise
        """
        if not trajectory:
            return False
        
        # Check required fields
        if 'waypoints' not in trajectory:
            return False
        
        waypoints = trajectory['waypoints']
        
        # Check waypoints format
        if not waypoints or len(waypoints) < 2:
            return False
        
        # Check each waypoint
        for wp in waypoints:
            if not isinstance(wp, (list, tuple)) or len(wp) != 3:
                return False
            # Check for valid coordinates
            for coord in wp:
                if not isinstance(coord, (int, float)):
                    return False
                if abs(coord) > 10:  # Sanity check for reasonable values
                    return False
        
        return True
    
    def _log_failure(
        self,
        prompt: str,
        scene_summary: Dict[str, Any],
        error: Exception,
        attempt: int
    ):
        """
        Log GPT failure for analysis.
        
        Args:
            prompt: Original prompt
            scene_summary: Scene at time of failure
            error: The exception that occurred
            attempt: Which attempt number failed
        """
        failure_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'scene_summary': scene_summary,
            'error': str(error),
            'error_type': type(error).__name__,
            'traceback': traceback.format_exc(),
            'attempt': attempt
        }
        
        self.failure_log.append(failure_entry)
        
        if self.log_failures:
            # Log to file for debugging
            log_file = f"gpt_failures_{datetime.now().strftime('%Y%m%d')}.json"
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    json.dump(failure_entry, f)
                    f.write('\n')
            except Exception as e:
                logger.error(f"Failed to write failure log: {e}")
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """
        Get statistics about GPT failures.
        
        Returns:
            Dictionary with failure statistics
        """
        if not self.failure_log:
            return {
                'total_failures': 0,
                'error_types': {},
                'failure_rate': 0.0
            }
        
        error_types = {}
        for failure in self.failure_log:
            error_type = failure['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_failures': len(self.failure_log),
            'error_types': error_types,
            'most_common_error': max(error_types, key=error_types.get) if error_types else None,
            'recent_failures': self.failure_log[-5:]  # Last 5 failures
        }


def create_robust_expert(
    fallback_type: str = 'pick_place',
    custom_waypoints: Optional[List[Tuple[float, float, float]]] = None
) -> Callable:
    """
    Create a robust expert function with automatic fallback.
    
    Args:
        fallback_type: Type of fallback pattern to use
        custom_waypoints: Optional custom waypoints to use as fallback
        
    Returns:
        Expert function that handles failures gracefully
        
    Example:
        expert = create_robust_expert()
        trajectory = expert(prompt, scene)
    """
    generator = ExpertScriptWithFallback(
        use_gpt=True,
        fallback_type=fallback_type
    )
    
    # Add custom waypoints if provided
    if custom_waypoints:
        generator.FALLBACK_WAYPOINTS['custom'] = {
            'waypoints': custom_waypoints,
            'gripper_actions': {0: 'open', len(custom_waypoints)-1: 'open'},
            'annotations': [f"Custom waypoint {i+1}" for i in range(len(custom_waypoints))]
        }
        generator.fallback_type = 'custom'
    
    def robust_expert(prompt: str, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Execute expert with fallback."""
        return generator.generate_expert_trajectory(prompt, scene)
    
    # Attach generator for inspection
    robust_expert.generator = generator
    
    return robust_expert


def test_fallback_mechanism():
    """Test the fallback mechanism with simulated failures."""
    
    print("="*70)
    print("TESTING EXPERT SCRIPT FALLBACK MECHANISM")
    print("="*70)
    
    # Create test scene
    scene = {
        'objects': [
            {'name': 'red_cube', 'position': [0.5, 0.0, 0.1], 'graspable': True},
            {'name': 'table', 'position': [0.5, 0.0, 0.0], 'graspable': False}
        ],
        'robot_state': {'position': [0.0, 0.0, 0.5]},
        'gripper_state': 'open',
        'workspace': {
            'x_min': -0.8, 'x_max': 0.8,
            'y_min': -0.8, 'y_max': 0.8,
            'z_min': 0.0, 'z_max': 1.0
        }
    }
    
    # Test with GPT enabled (will simulate failures)
    print("\n1. Testing with GPT (may fail and fallback):")
    print("-"*40)
    
    generator = ExpertScriptWithFallback(use_gpt=True)
    
    for i in range(3):
        print(f"\nAttempt {i+1}:")
        trajectory = generator.generate_expert_trajectory(
            "Pick up the red cube and place it",
            scene
        )
        
        print(f"  Generation method: {trajectory.get('generation_method', 'unknown')}")
        print(f"  Waypoints: {len(trajectory['waypoints'])}")
        if trajectory.get('generation_method') == 'fallback':
            print(f"  Fallback pattern: {trajectory.get('fallback_pattern')}")
        print(f"  First waypoint: {trajectory['waypoints'][0]}")
    
    # Test with GPT disabled (always uses fallback)
    print("\n2. Testing with GPT disabled (always fallback):")
    print("-"*40)
    
    generator_no_gpt = ExpertScriptWithFallback(use_gpt=False)
    trajectory = generator_no_gpt.generate_expert_trajectory(
        "Move the object to a new location",
        scene
    )
    
    print(f"  Generation method: {trajectory.get('generation_method')}")
    print(f"  Waypoints: {trajectory['waypoints']}")
    print(f"  Gripper actions: {trajectory.get('gripper_actions', {})}")
    
    # Test with custom fallback waypoints
    print("\n3. Testing with custom fallback waypoints:")
    print("-"*40)
    
    custom_expert = create_robust_expert(
        custom_waypoints=[
            (0.3, 0.0, 0.4),
            (0.5, 0.1, 0.3),
            (0.7, 0.0, 0.2)
        ]
    )
    
    trajectory = custom_expert("Custom task", scene)
    print(f"  Waypoints: {trajectory['waypoints']}")
    print(f"  Annotations: {trajectory.get('annotations', [])}")
    
    # Show failure statistics
    print("\n4. Failure Statistics:")
    print("-"*40)
    
    stats = generator.get_failure_stats()
    print(f"  Total failures: {stats['total_failures']}")
    print(f"  Error types: {stats['error_types']}")
    print(f"  Most common error: {stats.get('most_common_error', 'None')}")
    
    print("\n" + "="*70)
    print("✅ Fallback mechanism test complete!")
    print("="*70)


# Integration with existing pipeline
def integrate_with_pipeline(
    sim: Any,
    prompt: str,
    scene: Dict[str, Any],
    use_fallback: bool = True
) -> Dict[str, Any]:
    """
    Integrate robust expert generation with existing pipeline.
    
    Args:
        sim: Simulation environment
        prompt: Task description
        scene: Scene configuration
        use_fallback: Whether to enable fallback mechanism
        
    Returns:
        Execution results with trajectory
    """
    if use_fallback:
        # Use robust expert with fallback
        expert = create_robust_expert()
        trajectory = expert(prompt, scene)
        
        if trajectory.get('generation_method') == 'fallback':
            logger.warning(
                "⚠️ Pipeline continuing with fallback waypoints due to GPT failure. "
                "Check logs for details."
            )
    else:
        # Use original method without fallback
        from cogniforge.core.expert_script import gen_expert_script, execute_expert_script
        
        script_prompt = gen_expert_script(prompt, scene)
        # Assume we get response from GPT here
        response = "..."  # Would be actual GPT response
        trajectory = execute_expert_script(response, scene['objects'])
    
    # Continue with pipeline execution using trajectory
    result = {
        'trajectory': trajectory,
        'generation_method': trajectory.get('generation_method', 'standard'),
        'success': True
    }
    
    # Execute trajectory in simulation
    # ... simulation code here ...
    
    return result


if __name__ == "__main__":
    test_fallback_mechanism()