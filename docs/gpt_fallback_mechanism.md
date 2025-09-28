# GPT Fallback Mechanism

## Overview

The GPT fallback mechanism ensures pipeline continuity when GPT/Codex API calls fail. Instead of halting execution, the system automatically falls back to predefined 3-waypoint trajectories while logging errors for debugging.

## Key Features

### 1. **Automatic Fallback**
- Detects GPT/Codex failures (timeouts, API errors, invalid responses)
- Seamlessly switches to hardcoded waypoint patterns
- Pipeline continues without interruption

### 2. **Hardcoded 3-Waypoint Patterns**

The system includes three default patterns:

#### Pick & Place Pattern
```python
waypoints = [
    (0.4, 0.0, 0.3),   # Approach position
    (0.4, 0.0, 0.1),   # Grasp position
    (0.6, 0.2, 0.3),   # Place position
]
```

#### Simple Motion Pattern
```python
waypoints = [
    (0.3, -0.2, 0.3),  # Start position
    (0.5, 0.0, 0.25),  # Middle waypoint
    (0.3, 0.2, 0.3),   # End position
]
```

#### Lift & Move Pattern
```python
waypoints = [
    (0.4, 0.0, 0.15),  # Initial position
    (0.4, 0.0, 0.4),   # Lift up
    (0.6, 0.0, 0.15),  # Move to new position
]
```

### 3. **Error Logging**
- All GPT failures are logged with timestamps
- Includes error type, traceback, and context
- Logs saved to `gpt_failures_YYYYMMDD.json`

### 4. **Scene Adaptation**
- Fallback waypoints adapt to workspace bounds
- Adjusts to object positions when available
- Ensures safety constraints are maintained

## Usage

### Basic Implementation

```python
from cogniforge.core.expert_script_with_fallback import create_robust_expert

# Create expert with automatic fallback
expert = create_robust_expert()

# Generate trajectory (falls back if GPT fails)
trajectory = expert(prompt="Pick up the red cube", scene=scene_data)

# Check generation method
if trajectory['generation_method'] == 'fallback':
    print("Used fallback waypoints due to GPT failure")
```

### Custom Fallback Waypoints

```python
# Define custom 3-waypoint path
custom_waypoints = [
    (0.3, 0.0, 0.4),
    (0.5, 0.1, 0.3),
    (0.7, 0.0, 0.2)
]

expert = create_robust_expert(custom_waypoints=custom_waypoints)
```

### Pipeline Integration

```python
from cogniforge.core.expert_script_with_fallback import ExpertScriptWithFallback

class RobustPipeline:
    def __init__(self):
        self.expert_generator = ExpertScriptWithFallback(
            use_gpt=True,
            log_failures=True,
            fallback_type='pick_place'
        )
    
    def execute_task(self, prompt, scene):
        # Automatically handles GPT failures
        trajectory = self.expert_generator.generate_expert_trajectory(
            prompt, scene, max_retries=2
        )
        
        if trajectory['generation_method'] == 'fallback':
            logger.warning("Pipeline continuing with fallback waypoints")
        
        return trajectory
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_gpt` | `True` | Enable/disable GPT attempts |
| `gpt_timeout` | `10.0` | Timeout for GPT API calls (seconds) |
| `log_failures` | `True` | Log failures to file |
| `fallback_type` | `'pick_place'` | Default fallback pattern |
| `max_retries` | `2` | GPT retry attempts before fallback |

## Failure Statistics

Access failure statistics for monitoring:

```python
stats = expert_generator.get_failure_stats()
print(f"Total failures: {stats['total_failures']}")
print(f"Error types: {stats['error_types']}")
print(f"Most common error: {stats['most_common_error']}")
```

## Console Output Examples

### Successful GPT Generation
```
📝 Step 1: Generating expert trajectory...
✅ GPT generated trajectory successfully
  Attempts: 1
  Waypoints generated: 8
```

### Fallback Activation
```
📝 Step 1: Generating expert trajectory...
🔄 Using FALLBACK waypoints (GPT unavailable)
  Pattern: pick_place
  ⚠️ Using fallback waypoints due to GPT failure
  Waypoints generated: 3
```

## Error Log Format

Failed GPT attempts are logged in JSON format:

```json
{
    "timestamp": "2024-09-27T19:10:32",
    "prompt": "Pick up the red cube and place it",
    "error": "Connection timeout",
    "error_type": "TimeoutError",
    "traceback": "...",
    "attempt": 2,
    "scene_summary": {...}
}
```

## Benefits

1. **Robustness**: Pipeline never fails due to GPT unavailability
2. **Debugging**: Complete error logging for troubleshooting
3. **Performance**: Fallback execution is instant (no API delays)
4. **Safety**: Validated waypoints ensure safe robot motion
5. **Flexibility**: Custom waypoints can be defined per task

## Testing

Run the test suite:

```bash
python -m cogniforge.core.expert_script_with_fallback
```

Run the pipeline demo:

```bash
python examples/demo_fallback_pipeline.py
```

## Future Enhancements

- [ ] Dynamic waypoint generation based on scene geometry
- [ ] Learning from successful GPT trajectories
- [ ] Multiple fallback strategies per task type
- [ ] Automatic retry with exponential backoff
- [ ] Integration with local LLMs as secondary fallback

## Troubleshooting

### Issue: Fallback waypoints out of workspace bounds
**Solution**: The system automatically clips waypoints to workspace bounds

### Issue: GPT failures not being logged
**Solution**: Ensure `log_failures=True` and check write permissions

### Issue: Always using fallback even when GPT available
**Solution**: Check `use_gpt=True` and verify API credentials

## Related Components

- `expert_script.py`: Original GPT-based trajectory generation
- `expert_prompt.py`: Prompt generation for GPT
- `waypoint_optimizer.py`: Waypoint optimization utilities