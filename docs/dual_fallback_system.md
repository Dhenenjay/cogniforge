# Dual Fallback System

## Overview

The Cogniforge dual fallback system ensures pipeline continuity even when both GPT Vision and GPT Expert Script generation fail. The system provides automatic, graceful degradation to maintain operational robustness.

## Components

### 1. GPT Vision Fallback
**Primary**: GPT Vision API for advanced object detection
**Fallback**: Color-threshold detection with HSV ranges

### 2. GPT Expert Script Fallback  
**Primary**: GPT/Codex for dynamic trajectory generation
**Fallback**: Hardcoded 3-waypoint paths

## Visual Indicators

### Vision Fallback Banner
When GPT Vision times out or fails:
```
╔══════════════════════════════════════════════════════════╗
║                  ⚠️  VISION FALLBACK IN USE              ║
║                                                          ║
║  GPT Vision unavailable - using color threshold         ║
║  Detection accuracy may be reduced                      ║
╚══════════════════════════════════════════════════════════╝
```

### Status Indicators
- 🤖 **Green** - GPT Vision active
- 🎨 **Yellow** - Color threshold fallback active
- 📍 **Location pin** - Using fallback waypoints
- ✅ **Check** - Task completed successfully

## Fallback Scenarios

| Scenario | Vision System | Expert System | Result |
|----------|--------------|---------------|--------|
| Normal Operation | GPT Vision | GPT/Codex | Full capabilities |
| Vision Timeout | Color Threshold | GPT/Codex | Reduced vision accuracy |
| Expert Failure | GPT Vision | 3-waypoint path | Simple trajectories |
| Both Failed | Color Threshold | 3-waypoint path | Basic functionality maintained |

## Implementation

### Vision System with Fallback

```python
from cogniforge.vision.vision_with_fallback import create_robust_vision_system

# Initialize with fallback
vision = create_robust_vision_system(
    fallback_color='red',      # Default color for threshold
    gpt_timeout=5.0,           # Timeout in seconds
    auto_fallback=True         # Enable automatic fallback
)

# Detect objects (automatically falls back if needed)
result = vision.detect_objects(image, prompt="Detect cube")

# Check which method was used
if result.method == 'color_threshold':
    print("Used fallback vision")
```

### Expert Script with Fallback

```python
from cogniforge.core.expert_script_with_fallback import create_robust_expert

# Initialize with fallback
expert = create_robust_expert(
    fallback_type='pick_place',  # Default pattern
    custom_waypoints=None         # Optional custom waypoints
)

# Generate trajectory (automatically falls back if needed)
trajectory = expert("Pick up the cube", scene)

# Check which method was used
if trajectory['generation_method'] == 'fallback':
    print("Used fallback waypoints")
```

## Color Threshold Ranges

Predefined HSV ranges for common colors:

| Color | Hue Range | Saturation | Value |
|-------|-----------|------------|-------|
| Red | 0-10, 170-180 | 120-255 | 70-255 |
| Blue | 100-130 | 150-255 | 50-255 |
| Green | 40-80 | 40-255 | 40-255 |
| Yellow | 20-30 | 100-255 | 100-255 |
| Orange | 10-20 | 100-255 | 100-255 |
| Purple | 130-160 | 50-255 | 50-255 |

## Fallback Waypoint Patterns

### Pick & Place Pattern
```python
waypoints = [
    (0.4, 0.0, 0.3),   # Approach position
    (0.4, 0.0, 0.1),   # Grasp position
    (0.6, 0.2, 0.3),   # Place position
]
```

### Simple Motion Pattern
```python
waypoints = [
    (0.3, -0.2, 0.3),  # Start position
    (0.5, 0.0, 0.25),  # Middle waypoint
    (0.3, 0.2, 0.3),   # End position
]
```

### Lift & Move Pattern
```python
waypoints = [
    (0.4, 0.0, 0.15),  # Initial position
    (0.4, 0.0, 0.4),   # Lift up
    (0.6, 0.0, 0.15),  # Move to new position
]
```

## Error Logging

Both systems log failures for debugging:

### Vision Failures
Saved to: `vision_failures_YYYYMMDD.json`
```json
{
    "timestamp": "2024-09-27T19:13:59",
    "type": "timeout",
    "error": "GPT vision API timeout after 5.0s",
    "prompt": "Detect red cube"
}
```

### Expert Script Failures
Saved to: `gpt_failures_YYYYMMDD.json`
```json
{
    "timestamp": "2024-09-27T19:10:32",
    "prompt": "Pick up the red cube",
    "error": "Connection timeout",
    "attempt": 2
}
```

## Performance Comparison

| Metric | GPT Vision | Color Threshold | Improvement |
|--------|------------|-----------------|-------------|
| Latency | ~500ms | ~2ms | 250x faster |
| Accuracy | 95% | 70% | -26% |
| Robustness | API-dependent | Always available | ∞ |

| Metric | GPT Expert | Fallback Waypoints | Improvement |
|--------|------------|-------------------|-------------|
| Generation Time | ~1000ms | ~1ms | 1000x faster |
| Adaptability | Dynamic | Static | Limited |
| Reliability | API-dependent | Always available | ∞ |

## Configuration

### Environment Variables
```env
# GPT Configuration
GPT_VISION_TIMEOUT=5.0
GPT_EXPERT_TIMEOUT=10.0
ENABLE_FALLBACK=true

# Fallback Configuration
FALLBACK_COLOR=red
FALLBACK_PATTERN=pick_place
MIN_DETECTION_AREA=500
```

### Runtime Configuration
```python
# Configure both systems
pipeline = DualFallbackPipeline(
    vision_config={
        'gpt_timeout': 5.0,
        'fallback_color': 'red',
        'show_banner': True
    },
    expert_config={
        'gpt_timeout': 10.0,
        'fallback_type': 'pick_place',
        'log_failures': True
    }
)
```

## Testing

### Unit Tests
```bash
# Test vision fallback
python -m cogniforge.vision.vision_with_fallback

# Test expert fallback
python -m cogniforge.core.expert_script_with_fallback
```

### Integration Tests
```bash
# Test dual fallback
python examples/test_dual_fallback.py

# Demo vision fallback with banner
python examples/demo_vision_fallback.py

# Demo expert fallback
python examples/demo_fallback_pipeline.py
```

## Benefits

1. **100% Uptime**: Pipeline never fails due to API issues
2. **Graceful Degradation**: Reduced capabilities better than failure
3. **Clear Communication**: Banners inform operators of fallback state
4. **Performance**: Fallbacks are orders of magnitude faster
5. **Debugging**: Complete error logs for troubleshooting
6. **Flexibility**: Custom fallback configurations possible

## Best Practices

1. **Monitor Fallback Rate**: High fallback rates may indicate API issues
2. **Tune Color Thresholds**: Adjust HSV ranges for your lighting conditions
3. **Customize Waypoints**: Define task-specific fallback patterns
4. **Review Logs**: Regularly check failure logs for patterns
5. **Test Fallbacks**: Ensure fallback paths work for your use cases

## Troubleshooting

### Issue: Fallback banner appears twice
**Solution**: This is normal - once for console and once for auto-scroller

### Issue: Color detection not working
**Solution**: Check lighting conditions and adjust HSV ranges

### Issue: Waypoints out of workspace
**Solution**: Fallback automatically clips to workspace bounds

### Issue: High fallback rate
**Solution**: Check API credentials and network connectivity

## Future Enhancements

- [ ] Machine learning-based color threshold adaptation
- [ ] Dynamic waypoint generation from scene geometry
- [ ] Fallback quality metrics and reporting
- [ ] Automatic fallback pattern learning from successful GPT runs
- [ ] Multi-tier fallback (GPT → Local LLM → Hardcoded)