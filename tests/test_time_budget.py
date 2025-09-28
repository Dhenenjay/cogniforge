#!/usr/bin/env python3
"""
Unit tests for Time Budget Manager
"""

import unittest
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniforge.core.time_budget import (
    TimeBudgetManager, TimeoutException, Phase, PhaseStatistics
)


class TestTimeBudgetManager(unittest.TestCase):
    """Test cases for TimeBudgetManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = TimeBudgetManager(strict_mode=True, global_timeout=10.0)
    
    def test_phase_within_budget(self):
        """Test phase completing within budget."""
        with self.manager.phase("TEST_PHASE", budget=1.0) as stats:
            time.sleep(0.5)  # Well within budget
            self.assertLess(stats.elapsed, 1.0)
        
        # Check statistics
        self.assertEqual(len(self.manager.phase_stats), 1)
        phase_stat = self.manager.phase_stats[0]
        self.assertTrue(phase_stat.completed)
        self.assertFalse(phase_stat.aborted)
        self.assertLess(phase_stat.elapsed, 1.0)
    
    def test_phase_exceeds_budget_strict(self):
        """Test phase exceeding budget in strict mode."""
        with self.assertRaises(TimeoutException) as context:
            with self.manager.phase("TEST_PHASE", budget=0.5):
                time.sleep(1.0)  # Exceeds budget
        
        # Check exception details
        exception = context.exception
        self.assertEqual(exception.phase, "TEST_PHASE")
        self.assertEqual(exception.budget, 0.5)
        self.assertGreater(exception.elapsed, 0.5)
    
    def test_phase_exceeds_budget_non_strict(self):
        """Test phase exceeding budget in non-strict mode."""
        self.manager.strict_mode = False
        
        # Should not raise exception
        with self.manager.phase("TEST_PHASE", budget=0.5):
            time.sleep(0.7)  # Exceeds budget but won't raise
        
        # Check statistics
        phase_stat = self.manager.phase_stats[0]
        self.assertTrue(phase_stat.completed)  # Still marked as completed
        self.assertFalse(phase_stat.aborted)
        self.assertGreater(phase_stat.elapsed, 0.5)
    
    def test_abort_handler(self):
        """Test abort handler is called on timeout."""
        handler_called = False
        
        def abort_handler():
            nonlocal handler_called
            handler_called = True
        
        self.manager.register_abort_handler("TEST_PHASE", abort_handler)
        
        try:
            with self.manager.phase("TEST_PHASE", budget=0.1):
                time.sleep(0.5)  # Trigger timeout
        except TimeoutException:
            pass
        
        # Handler should have been called
        self.assertTrue(handler_called)
    
    def test_timeout_decorator(self):
        """Test timeout decorator functionality."""
        @self.manager.timeout(0.5)
        def fast_function():
            time.sleep(0.2)
            return "success"
        
        @self.manager.timeout(0.1)
        def slow_function():
            time.sleep(0.5)
            return "should_not_return"
        
        # Fast function should succeed
        result = fast_function()
        self.assertEqual(result, "success")
        
        # Slow function should timeout
        with self.assertRaises(TimeoutException):
            slow_function()
    
    def test_custom_phase_budgets(self):
        """Test setting custom phase budgets."""
        # Set custom budget
        self.manager.set_phase_budget("CUSTOM_PHASE", 2.5)
        self.assertEqual(self.manager.phase_budgets["CUSTOM_PHASE"], 2.5)
        
        # Use custom budget
        with self.manager.phase("CUSTOM_PHASE") as stats:
            self.assertEqual(stats.budget, 2.5)
            time.sleep(0.1)
        
        # Override with explicit budget
        with self.manager.phase("CUSTOM_PHASE", budget=1.0) as stats:
            self.assertEqual(stats.budget, 1.0)
            time.sleep(0.1)
    
    def test_warning_thresholds(self):
        """Test warning thresholds during phase execution."""
        self.manager.strict_mode = False  # Don't abort on timeout
        
        with self.manager.phase("TEST_PHASE", budget=1.0) as stats:
            # Wait for 50% threshold
            time.sleep(0.6)
            self.assertIn("50% threshold reached", stats.warnings)
            
            # Wait for 75% threshold
            time.sleep(0.3)
            self.assertIn("75% threshold reached", stats.warnings)
    
    def test_global_timeout(self):
        """Test global timeout enforcement."""
        self.manager.global_timeout = 1.0
        self.manager.start_global_timer()
        
        # First phase should succeed
        with self.manager.phase("PHASE1", budget=0.3):
            time.sleep(0.2)
        
        # Second phase should succeed
        with self.manager.phase("PHASE2", budget=0.3):
            time.sleep(0.2)
        
        # Third phase should trigger global timeout
        with self.assertRaises(TimeoutException) as context:
            with self.manager.phase("PHASE3", budget=5.0):
                time.sleep(1.0)
        
        exception = context.exception
        self.assertEqual(exception.phase, "GLOBAL")
    
    def test_phase_statistics(self):
        """Test phase statistics collection."""
        # Run several phases
        with self.manager.phase("PHASE1", budget=1.0):
            time.sleep(0.1)
        
        with self.manager.phase("PHASE2", budget=1.0):
            time.sleep(0.2)
        
        self.manager.strict_mode = False
        with self.manager.phase("PHASE3", budget=0.1):
            time.sleep(0.3)  # Exceeds but won't abort
        
        # Get summary
        summary = self.manager.get_summary()
        
        self.assertEqual(summary["total_phases"], 3)
        self.assertEqual(summary["completed_phases"], 3)
        self.assertEqual(summary["aborted_phases"], 0)
        self.assertGreater(summary["total_elapsed"], 0.6)
        
        # Check individual phase data
        phases = summary["phases"]
        self.assertEqual(len(phases), 3)
        
        for phase in phases:
            self.assertIn("phase", phase)
            self.assertIn("elapsed", phase)
            self.assertIn("budget", phase)
            self.assertIn("completed", phase)
    
    def test_nested_phases(self):
        """Test nested phase contexts."""
        with self.manager.phase("OUTER", budget=2.0):
            time.sleep(0.1)
            
            with self.manager.phase("INNER1", budget=0.5):
                time.sleep(0.1)
            
            with self.manager.phase("INNER2", budget=0.5):
                time.sleep(0.1)
        
        # Should have recorded all three phases
        self.assertEqual(len(self.manager.phase_stats), 3)
        
        # All should be completed
        for stat in self.manager.phase_stats:
            self.assertTrue(stat.completed)
    
    def test_predefined_phases(self):
        """Test predefined phase enum values."""
        # Check some predefined phases
        self.assertEqual(Phase.DATA_LOADING.value, 30.0)
        self.assertEqual(Phase.TRAINING_EPOCH.value, 300.0)
        self.assertEqual(Phase.CODE_GENERATION.value, 60.0)
        
        # Use predefined phase
        phase_name = Phase.DATA_LOADING.name
        self.assertIn(phase_name, self.manager.phase_budgets)
        
        with self.manager.phase(phase_name) as stats:
            self.assertEqual(stats.budget, 30.0)
            time.sleep(0.1)
    
    def test_efficiency_calculation(self):
        """Test efficiency calculation in summary."""
        # Run phase using only 50% of budget
        with self.manager.phase("EFFICIENT", budget=1.0):
            time.sleep(0.5)
        
        # Run phase using 150% of budget (non-strict)
        self.manager.strict_mode = False
        with self.manager.phase("INEFFICIENT", budget=0.5):
            time.sleep(0.75)
        
        summary = self.manager.get_summary()
        
        # Check efficiency calculations
        self.assertIn("efficiency", summary)
        efficiencies = summary["efficiency"]
        
        # Efficient phase should have > 100% efficiency
        # (budget/elapsed * 100, so 1.0/0.5 * 100 = 200%)
        self.assertGreater(efficiencies["EFFICIENT"]["efficiency"], 100)
        
        # Inefficient phase should have < 100% efficiency
        # (0.5/0.75 * 100 = ~67%)
        self.assertLess(efficiencies["INEFFICIENT"]["efficiency"], 100)


class TestTimeoutException(unittest.TestCase):
    """Test cases for TimeoutException."""
    
    def test_exception_formatting(self):
        """Test exception message formatting."""
        exception = TimeoutException(
            phase="TEST_PHASE",
            budget=10.0,
            elapsed=15.5,
            message="Custom help message"
        )
        
        message = exception.format_message()
        
        # Check message contains key information
        self.assertIn("TIME BUDGET EXCEEDED", message)
        self.assertIn("TEST_PHASE", message)
        self.assertIn("10.0 seconds", message)
        self.assertIn("15.5 seconds", message)
        self.assertIn("5.5 seconds", message)  # Exceeded by
        self.assertIn("Custom help message", message)


class TestPhaseStatistics(unittest.TestCase):
    """Test cases for PhaseStatistics."""
    
    def test_statistics_to_dict(self):
        """Test converting statistics to dictionary."""
        from datetime import datetime
        
        stats = PhaseStatistics(
            phase="TEST",
            start_time=datetime.now(),
            budget=10.0,
            elapsed=5.5,
            completed=True,
            aborted=False
        )
        
        data = stats.to_dict()
        
        self.assertEqual(data["phase"], "TEST")
        self.assertEqual(data["budget"], 10.0)
        self.assertEqual(data["elapsed"], 5.5)
        self.assertTrue(data["completed"])
        self.assertFalse(data["aborted"])
        self.assertIsNone(data["end_time"])
        
        # Set end time
        stats.end_time = datetime.now()
        data = stats.to_dict()
        self.assertIsNotNone(data["end_time"])


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False)


if __name__ == "__main__":
    print("="*60)
    print("TIME BUDGET MANAGER TESTS")
    print("="*60)
    
    # Run tests
    run_tests()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)