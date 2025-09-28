#!/usr/bin/env python3
"""Test the log_event helper functionality."""

import sys
import time
from pathlib import Path

# Direct import without going through __init__.py
sys.path.insert(0, str(Path(__file__).parent))

# Import only what we need
from cogniforge.core.logging_utils import (
    log_event,
    EventPhase,
    LogLevel,
    set_request_context,
    get_execution_summary,
    log_info,
    log_warning,
    log_error,
    log_success,
    log_metric
)

def test_logging():
    """Test the logging functionality."""
    
    print("="*60)
    print("TESTING LOG_EVENT HELPER")
    print("="*60)
    
    # Set up request context
    set_request_context("test-request-001", "test_output.json")
    
    # Test different log levels and phases
    print("\n1. Testing basic logging...")
    log_event(EventPhase.PLANNING, "Starting task planning")
    log_event(EventPhase.PLANNING, "Task analyzed", complexity=0.7, steps=8)
    
    print("\n2. Testing metrics logging...")
    log_metric(EventPhase.BEHAVIOR_CLONING, "Epoch 1", epoch=1, loss=0.234)
    log_metric(EventPhase.BEHAVIOR_CLONING, "Epoch 2", epoch=2, loss=0.156)
    
    print("\n3. Testing warning and error...")
    log_warning(EventPhase.VISION_REFINEMENT, "Low confidence", confidence=0.3)
    log_error(EventPhase.EXECUTION, "Motion failed", waypoint=3, error="Collision")
    
    print("\n4. Testing success logging...")
    log_success(EventPhase.COMPLETED, "Pipeline complete", duration=12.5, success_rate=0.95)
    
    # Vision offset specific logging
    print("\n5. Testing vision offset logging...")
    log_event(
        EventPhase.VISION_REFINEMENT,
        "Vision offset detected",
        level=LogLevel.METRIC,
        dx=5,
        dy=-3,
        confidence=0.92,
        method="template_matching"
    )
    
    # Get summary
    summary = get_execution_summary()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Display phases
    print("\nPhases logged:")
    for phase, data in summary['phases'].items():
        print(f"  - {phase}: {data['message_count']} messages")
    
    # Display metrics
    if summary['metrics']:
        print("\nMetrics collected:")
        for metric, stats in summary['metrics'].items():
            if isinstance(stats, dict) and 'avg' in stats:
                print(f"  - {metric}: avg={stats['avg']:.3f}, "
                      f"min={stats['min']}, max={stats['max']}")
            else:
                print(f"  - {metric}: {stats}")
    
    print(f"\nTotal time: {summary['total_time']:.2f} seconds")
    print(f"Request ID: {summary['request_id']}")
    
    # Check if log file was created
    if Path("test_output.json").exists():
        print("\n✅ Log file created: test_output.json")
        with open("test_output.json", "r") as f:
            lines = f.readlines()
            print(f"   Contains {len(lines)} log entries")
    
    return summary

if __name__ == "__main__":
    test_logging()
    print("\n✅ Test complete!")