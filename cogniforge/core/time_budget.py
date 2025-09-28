"""
Time Budget Manager for CogniForge

Enforces time limits for different phases of execution with graceful abort
and helpful error messages.
"""

import time
import threading
import signal
import sys
import functools
import warnings
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when a time budget is exceeded."""
    
    def __init__(self, phase: str, budget: float, elapsed: float, message: str = None):
        self.phase = phase
        self.budget = budget
        self.elapsed = elapsed
        self.message = message or f"Phase '{phase}' exceeded time budget"
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format a helpful error message."""
        return (
            f"\n{'='*60}\n"
            f"⏱️ TIME BUDGET EXCEEDED\n"
            f"{'='*60}\n"
            f"Phase: {self.phase}\n"
            f"Budget: {self.budget:.1f} seconds\n"
            f"Elapsed: {self.elapsed:.1f} seconds\n"
            f"Exceeded by: {(self.elapsed - self.budget):.1f} seconds\n"
            f"{'='*60}\n"
            f"{self.message}\n"
        )


class Phase(Enum):
    """Predefined phases with default time budgets (in seconds)."""
    
    # Data processing phases
    DATA_LOADING = 30.0
    DATA_PREPROCESSING = 60.0
    DATA_AUGMENTATION = 45.0
    
    # Training phases
    TRAINING_EPOCH = 300.0  # 5 minutes per epoch
    VALIDATION = 60.0
    CHECKPOINT_SAVE = 30.0
    
    # Inference phases
    MODEL_LOADING = 20.0
    INFERENCE = 10.0
    POSTPROCESSING = 15.0
    
    # Code generation phases
    CODE_GENERATION = 60.0
    CODE_VALIDATION = 30.0
    CODE_EXECUTION = 120.0
    
    # Simulation phases
    SIMULATION_INIT = 30.0
    SIMULATION_STEP = 0.1  # Per step
    SIMULATION_RENDER = 0.05  # Per frame
    
    # API phases
    API_REQUEST = 30.0
    API_RESPONSE = 10.0
    
    # General phases
    INITIALIZATION = 60.0
    CLEANUP = 30.0
    TOTAL_EXECUTION = 3600.0  # 1 hour total


@dataclass
class PhaseStatistics:
    """Statistics for a single phase execution."""
    phase: str
    start_time: datetime
    end_time: Optional[datetime] = None
    budget: float = 0.0
    elapsed: float = 0.0
    completed: bool = False
    aborted: bool = False
    abort_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "phase": self.phase,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "budget": self.budget,
            "elapsed": self.elapsed,
            "completed": self.completed,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "warnings": self.warnings
        }


class TimeBudgetManager:
    """Manages time budgets for different execution phases."""
    
    def __init__(self, strict_mode: bool = True, global_timeout: float = 3600.0):
        """
        Initialize the time budget manager.
        
        Args:
            strict_mode: If True, raises exceptions on timeout. If False, logs warnings.
            global_timeout: Maximum total execution time in seconds.
        """
        self.strict_mode = strict_mode
        self.global_timeout = global_timeout
        self.global_start_time = None
        
        # Phase budgets (can be customized)
        self.phase_budgets: Dict[str, float] = {
            phase.name: phase.value for phase in Phase
        }
        
        # Execution statistics
        self.phase_stats: List[PhaseStatistics] = []
        self.current_phase: Optional[PhaseStatistics] = None
        
        # Warning thresholds (percentage of budget)
        self.warning_thresholds = [0.5, 0.75, 0.9]  # 50%, 75%, 90%
        
        # Abort handlers
        self.abort_handlers: Dict[str, Callable] = {}
        
        # Thread for monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
    
    def set_phase_budget(self, phase: str, budget: float):
        """
        Set or update the time budget for a specific phase.
        
        Args:
            phase: Phase name (can be custom or from Phase enum)
            budget: Time budget in seconds
        """
        self.phase_budgets[phase] = budget
        logger.info(f"Set budget for phase '{phase}': {budget:.1f} seconds")
    
    def register_abort_handler(self, phase: str, handler: Callable):
        """
        Register a cleanup handler for graceful abort.
        
        Args:
            phase: Phase name
            handler: Cleanup function to call on abort
        """
        self.abort_handlers[phase] = handler
        logger.debug(f"Registered abort handler for phase '{phase}'")
    
    @contextmanager
    def phase(self, name: str, budget: Optional[float] = None, 
              abort_handler: Optional[Callable] = None):
        """
        Context manager for timing a phase with automatic budget enforcement.
        
        Args:
            name: Phase name
            budget: Optional custom budget (uses default if not specified)
            abort_handler: Optional cleanup handler for this phase
        
        Example:
            with time_manager.phase("TRAINING_EPOCH", budget=300):
                train_model()
        """
        # Get budget
        if budget is None:
            budget = self.phase_budgets.get(name, 60.0)  # Default 60 seconds
        
        # Register abort handler if provided
        if abort_handler:
            self.register_abort_handler(name, abort_handler)
        
        # Create phase statistics
        stats = PhaseStatistics(
            phase=name,
            start_time=datetime.now(),
            budget=budget
        )
        self.current_phase = stats
        
        # Start monitoring
        monitor_thread = threading.Thread(
            target=self._monitor_phase,
            args=(stats,),
            daemon=True
        )
        monitor_thread.start()
        
        logger.info(f"Starting phase '{name}' with budget {budget:.1f} seconds")
        
        try:
            yield stats
            
            # Phase completed successfully
            stats.completed = True
            
        except TimeoutException:
            # Phase was aborted due to timeout
            stats.aborted = True
            stats.abort_reason = "Time budget exceeded"
            
            # Call abort handler if registered
            if name in self.abort_handlers:
                try:
                    logger.info(f"Running abort handler for phase '{name}'")
                    self.abort_handlers[name]()
                except Exception as e:
                    logger.error(f"Error in abort handler: {e}")
            
            if self.strict_mode:
                raise
            else:
                logger.warning(f"Phase '{name}' exceeded budget but continuing (strict_mode=False)")
                
        except Exception as e:
            # Other exception occurred
            stats.aborted = True
            stats.abort_reason = f"Exception: {str(e)}"
            raise
            
        finally:
            # Stop monitoring
            self._stop_monitor.set()
            monitor_thread.join(timeout=1)
            self._stop_monitor.clear()
            
            # Finalize statistics
            stats.end_time = datetime.now()
            stats.elapsed = (stats.end_time - stats.start_time).total_seconds()
            self.phase_stats.append(stats)
            self.current_phase = None
            
            # Log summary
            if stats.completed:
                logger.info(
                    f"Phase '{name}' completed in {stats.elapsed:.1f}s "
                    f"({(stats.elapsed/budget)*100:.0f}% of budget)"
                )
            else:
                logger.warning(
                    f"Phase '{name}' aborted after {stats.elapsed:.1f}s - "
                    f"Reason: {stats.abort_reason}"
                )
    
    def _monitor_phase(self, stats: PhaseStatistics):
        """Monitor a phase and enforce time budget."""
        warned_thresholds = set()
        
        while not self._stop_monitor.is_set():
            elapsed = (datetime.now() - stats.start_time).total_seconds()
            stats.elapsed = elapsed
            
            # Check if budget exceeded
            if elapsed > stats.budget:
                if self.strict_mode:
                    raise TimeoutException(
                        phase=stats.phase,
                        budget=stats.budget,
                        elapsed=elapsed,
                        message=self._get_timeout_help(stats.phase)
                    )
                else:
                    if "exceeded" not in warned_thresholds:
                        logger.warning(
                            f"Phase '{stats.phase}' exceeded budget "
                            f"({elapsed:.1f}s > {stats.budget:.1f}s)"
                        )
                        warned_thresholds.add("exceeded")
            
            # Issue warnings at thresholds
            for threshold in self.warning_thresholds:
                if elapsed > stats.budget * threshold and threshold not in warned_thresholds:
                    remaining = stats.budget - elapsed
                    logger.warning(
                        f"Phase '{stats.phase}' at {threshold*100:.0f}% of budget "
                        f"({remaining:.1f}s remaining)"
                    )
                    stats.warnings.append(f"{threshold*100:.0f}% threshold reached")
                    warned_thresholds.add(threshold)
            
            # Check global timeout
            if self.global_start_time:
                global_elapsed = (datetime.now() - self.global_start_time).total_seconds()
                if global_elapsed > self.global_timeout:
                    raise TimeoutException(
                        phase="GLOBAL",
                        budget=self.global_timeout,
                        elapsed=global_elapsed,
                        message="Total execution time limit exceeded"
                    )
            
            time.sleep(0.1)  # Check every 100ms
    
    def _get_timeout_help(self, phase: str) -> str:
        """Get helpful message for timeout errors."""
        messages = {
            "DATA_LOADING": (
                "💡 Suggestions:\n"
                "  • Check if data files exist and are accessible\n"
                "  • Consider loading data in smaller batches\n"
                "  • Use data caching to speed up subsequent loads\n"
                "  • Increase budget with: set_phase_budget('DATA_LOADING', 60)"
            ),
            "TRAINING_EPOCH": (
                "💡 Suggestions:\n"
                "  • Reduce batch size to speed up training\n"
                "  • Use mixed precision training (fp16)\n"
                "  • Consider using a smaller model\n"
                "  • Enable gradient checkpointing\n"
                "  • Increase budget with: set_phase_budget('TRAINING_EPOCH', 600)"
            ),
            "CODE_GENERATION": (
                "💡 Suggestions:\n"
                "  • Simplify the task description\n"
                "  • Use a faster model or API endpoint\n"
                "  • Check network connectivity\n"
                "  • Increase budget with: set_phase_budget('CODE_GENERATION', 120)"
            ),
            "SIMULATION_STEP": (
                "💡 Suggestions:\n"
                "  • Reduce simulation complexity\n"
                "  • Decrease physics simulation frequency\n"
                "  • Use simpler collision detection\n"
                "  • Consider headless mode (no rendering)"
            ),
            "API_REQUEST": (
                "💡 Suggestions:\n"
                "  • Check network connectivity\n"
                "  • Verify API endpoint is responding\n"
                "  • Consider implementing retry logic\n"
                "  • Use connection pooling for multiple requests"
            )
        }
        
        return messages.get(phase, (
            "💡 Suggestions:\n"
            "  • Check for infinite loops or blocking operations\n"
            "  • Profile the code to identify bottlenecks\n"
            "  • Consider parallelizing operations\n"
            f"  • Increase budget with: set_phase_budget('{phase}', <seconds>)"
        ))
    
    def timeout(self, seconds: float):
        """
        Decorator for functions with time limits.
        
        Example:
            @time_manager.timeout(30)
            def slow_function():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                phase_name = f"{func.__name__}_TIMEOUT"
                with self.phase(phase_name, budget=seconds):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_global_timer(self):
        """Start the global execution timer."""
        self.global_start_time = datetime.now()
        logger.info(f"Started global timer with {self.global_timeout:.0f}s budget")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary with all phase statistics."""
        total_elapsed = sum(s.elapsed for s in self.phase_stats)
        completed_phases = [s for s in self.phase_stats if s.completed]
        aborted_phases = [s for s in self.phase_stats if s.aborted]
        
        return {
            "total_phases": len(self.phase_stats),
            "completed_phases": len(completed_phases),
            "aborted_phases": len(aborted_phases),
            "total_elapsed": total_elapsed,
            "phases": [s.to_dict() for s in self.phase_stats],
            "efficiency": {
                phase: {
                    "budget": stats.budget,
                    "elapsed": stats.elapsed,
                    "efficiency": (stats.budget / stats.elapsed * 100) if stats.elapsed > 0 else 0
                }
                for phase, stats in [(s.phase, s) for s in self.phase_stats]
            }
        }
    
    def print_summary(self):
        """Print a formatted summary of execution times."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("⏱️  EXECUTION TIME SUMMARY")
        print("="*60)
        print(f"Total phases: {summary['total_phases']}")
        print(f"Completed: {summary['completed_phases']}")
        print(f"Aborted: {summary['aborted_phases']}")
        print(f"Total time: {summary['total_elapsed']:.1f} seconds")
        print("-"*60)
        
        for phase_stat in self.phase_stats:
            status = "✅" if phase_stat.completed else "❌"
            efficiency = (phase_stat.budget / phase_stat.elapsed * 100) if phase_stat.elapsed > 0 else 0
            
            print(f"{status} {phase_stat.phase:30s} "
                  f"{phase_stat.elapsed:7.1f}s / {phase_stat.budget:7.1f}s "
                  f"({efficiency:5.1f}% efficiency)")
            
            if phase_stat.warnings:
                for warning in phase_stat.warnings:
                    print(f"   ⚠️  {warning}")
            
            if phase_stat.aborted:
                print(f"   ❌ {phase_stat.abort_reason}")
        
        print("="*60)


# Global instance for convenience
default_manager = TimeBudgetManager(strict_mode=True)


# Convenience functions
def set_budget(phase: str, seconds: float):
    """Set time budget for a phase."""
    default_manager.set_phase_budget(phase, seconds)


def phase(name: str, budget: Optional[float] = None):
    """Context manager for phase timing."""
    return default_manager.phase(name, budget)


def timeout(seconds: float):
    """Decorator for function timeouts."""
    return default_manager.timeout(seconds)


def get_summary():
    """Get execution summary."""
    return default_manager.get_summary()


def print_summary():
    """Print execution summary."""
    default_manager.print_summary()


# Example usage
if __name__ == "__main__":
    import random
    
    # Create manager
    manager = TimeBudgetManager(strict_mode=False)
    
    # Example 1: Using context manager
    try:
        with manager.phase("DATA_LOADING", budget=2.0):
            print("Loading data...")
            time.sleep(1.0)  # Within budget
            print("Data loaded!")
    except TimeoutException as e:
        print(e)
    
    # Example 2: Exceeding budget
    try:
        with manager.phase("TRAINING", budget=1.0):
            print("Training model...")
            time.sleep(2.0)  # Exceeds budget!
            print("Training complete!")
    except TimeoutException as e:
        print(e)
    
    # Example 3: Using decorator
    @manager.timeout(1.0)
    def slow_function():
        print("Starting slow function...")
        time.sleep(0.5)
        return "Done!"
    
    result = slow_function()
    print(f"Result: {result}")
    
    # Print summary
    manager.print_summary()