"""
Logging utilities for CogniForge pipeline.
Provides unified logging to console and SSE streams.
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union
from enum import Enum
import asyncio
from pathlib import Path
import queue
import threading


class LogLevel(Enum):
    """Log levels for message categorization."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    METRIC = "metric"
    PHASE = "phase"


class EventPhase(Enum):
    """Pipeline execution phases."""
    CONNECTED = "connected"
    PLANNING = "planning"
    EXPERT_DEMONSTRATION = "expert_demonstration"
    BEHAVIOR_CLONING = "behavior_cloning"
    OPTIMIZATION = "optimization"
    VISION_REFINEMENT = "vision_refinement"
    CODE_GENERATION = "code_generation"
    EXECUTION = "execution"
    COMPLETED = "completed"
    FAILED = "failed"


class LogEventManager:
    """
    Manages logging events for both console and SSE output.
    Thread-safe implementation for concurrent access.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the log event manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.logger = self._setup_logger()
            self.sse_queue = queue.Queue()
            self.sse_clients = []
            self.request_id = None
            self.start_time = time.time()
            self.phase_timings = {}
            self.metrics_buffer = []
            self.enable_console = True
            self.enable_sse = True
            self.log_file = None
            
    def _setup_logger(self) -> logging.Logger:
        """Set up the console logger."""
        logger = logging.getLogger('cogniforge')
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Format with colors for console
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Clear existing handlers
        logger.handlers.clear()
        logger.addHandler(console_handler)
        
        return logger
    
    def set_request_id(self, request_id: str):
        """Set the current request ID for tracking."""
        self.request_id = request_id
        self.start_time = time.time()
        self.phase_timings = {}
        self.metrics_buffer = []
        
    def set_log_file(self, filepath: Union[str, Path]):
        """Set a log file for persistent logging."""
        self.log_file = Path(filepath)
        
        # Add file handler to logger
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def add_sse_client(self, client):
        """Add an SSE client for event streaming."""
        self.sse_clients.append(client)
    
    def remove_sse_client(self, client):
        """Remove an SSE client."""
        if client in self.sse_clients:
            self.sse_clients.remove(client)
    
    def log_event(
        self,
        phase: Union[str, EventPhase],
        message: str,
        level: Union[str, LogLevel] = LogLevel.INFO,
        **metrics
    ):
        """
        Log an event to both console and SSE streams.
        
        Args:
            phase: Current execution phase
            message: Log message
            level: Log level (info, warning, error, etc.)
            **metrics: Additional metrics as keyword arguments
        """
        # Convert enums to strings
        if isinstance(phase, EventPhase):
            phase = phase.value
        if isinstance(level, LogLevel):
            level = level.value
            
        # Create timestamp
        timestamp = datetime.now().isoformat()
        elapsed_time = time.time() - self.start_time
        
        # Track phase timing
        if phase not in self.phase_timings:
            self.phase_timings[phase] = {
                'start': elapsed_time,
                'messages': []
            }
        self.phase_timings[phase]['messages'].append(message)
        
        # Create event data
        event_data = {
            'timestamp': timestamp,
            'request_id': self.request_id,
            'phase': phase,
            'message': message,
            'level': level,
            'elapsed_time': round(elapsed_time, 2),
            'metrics': metrics
        }
        
        # Log to console if enabled
        if self.enable_console:
            self._log_to_console(phase, message, level, metrics)
        
        # Send to SSE clients if enabled
        if self.enable_sse:
            self._send_sse_event(event_data)
        
        # Buffer metrics for aggregation
        if metrics:
            self.metrics_buffer.append({
                'timestamp': timestamp,
                'phase': phase,
                'metrics': metrics
            })
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(event_data)
    
    def _log_to_console(self, phase: str, message: str, level: str, metrics: Dict):
        """Log to console with appropriate formatting."""
        # Create formatted message
        phase_emoji = {
            'connected': '🔌',
            'planning': '📋',
            'expert_demonstration': '👨‍🏫',
            'behavior_cloning': '🧠',
            'optimization': '⚙️',
            'vision_refinement': '👁️',
            'code_generation': '💻',
            'execution': '🤖',
            'completed': '✅',
            'failed': '❌'
        }
        
        emoji = phase_emoji.get(phase, '●')
        formatted_msg = f"{emoji} [{phase}] {message}"
        
        # Add metrics if present
        if metrics:
            metrics_str = ', '.join([f"{k}={v}" for k, v in metrics.items()])
            formatted_msg += f" | {metrics_str}"
        
        # Log based on level
        if level == 'debug':
            self.logger.debug(formatted_msg)
        elif level == 'warning':
            self.logger.warning(formatted_msg)
        elif level == 'error':
            self.logger.error(formatted_msg)
        elif level in ['success', 'metric']:
            self.logger.info(formatted_msg)
        else:
            self.logger.info(formatted_msg)
    
    def _send_sse_event(self, event_data: Dict[str, Any]):
        """Send event to SSE clients."""
        # Convert numpy types to native Python types for JSON serialization
        import numpy as np
        
        def convert_numpy(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Convert event data
        clean_data = convert_numpy(event_data)
        
        # Format as SSE
        sse_message = f"data: {json.dumps(clean_data)}\n\n"
        
        # Queue for async sending
        self.sse_queue.put(sse_message)
        
        # Send to all connected clients
        for client in self.sse_clients[:]:  # Create a copy to avoid modification during iteration
            try:
                if hasattr(client, 'send'):
                    client.send(sse_message)
                elif hasattr(client, 'write'):
                    client.write(sse_message.encode())
            except Exception as e:
                # Remove disconnected clients
                self.logger.debug(f"Failed to send to client: {e}")
                self.remove_sse_client(client)
    
    def _write_to_file(self, event_data: Dict):
        """Write event to log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {e}")
    
    def get_phase_summary(self) -> Dict:
        """Get summary of all phases with timings."""
        summary = {}
        for phase, data in self.phase_timings.items():
            summary[phase] = {
                'start_time': data['start'],
                'message_count': len(data['messages']),
                'last_message': data['messages'][-1] if data['messages'] else None
            }
        return summary
    
    def get_metrics_summary(self) -> Dict:
        """Get aggregated metrics summary."""
        if not self.metrics_buffer:
            return {}
            
        # Aggregate metrics by type
        aggregated = {}
        for entry in self.metrics_buffer:
            for key, value in entry['metrics'].items():
                if key not in aggregated:
                    aggregated[key] = []
                if isinstance(value, (int, float)):
                    aggregated[key].append(value)
        
        # Calculate statistics
        summary = {}
        for key, values in aggregated.items():
            if values:
                summary[key] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'last': values[-1]
                }
        
        return summary


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format the log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


# Global instance for easy access
_log_manager = None

def get_log_manager() -> LogEventManager:
    """Get the global log event manager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogEventManager()
    return _log_manager


def log_event(
    phase: Union[str, EventPhase],
    message: str,
    level: Union[str, LogLevel] = LogLevel.INFO,
    **metrics
):
    """
    Convenience function to log an event.
    
    Args:
        phase: Current execution phase
        message: Log message
        level: Log level (info, warning, error, etc.)
        **metrics: Additional metrics as keyword arguments
        
    Example:
        log_event("planning", "Starting task planning", level="info", 
                  task_id=123, complexity=0.7)
        
        log_event(EventPhase.BEHAVIOR_CLONING, "Training epoch complete",
                  epoch=5, loss=0.234, accuracy=0.89)
    """
    manager = get_log_manager()
    manager.log_event(phase, message, level, **metrics)


def set_request_context(request_id: str, log_file: Optional[str] = None):
    """
    Set the request context for logging.
    
    Args:
        request_id: Unique request identifier
        log_file: Optional log file path
    """
    manager = get_log_manager()
    manager.set_request_id(request_id)
    
    if log_file:
        manager.set_log_file(log_file)


def add_sse_client(client):
    """Add an SSE client for event streaming."""
    manager = get_log_manager()
    manager.add_sse_client(client)


def remove_sse_client(client):
    """Remove an SSE client."""
    manager = get_log_manager()
    manager.remove_sse_client(client)


def get_execution_summary() -> Dict:
    """
    Get execution summary with phase timings and metrics.
    
    Returns:
        Dictionary containing phase summary and metrics summary
    """
    manager = get_log_manager()
    return {
        'phases': manager.get_phase_summary(),
        'metrics': manager.get_metrics_summary(),
        'total_time': time.time() - manager.start_time,
        'request_id': manager.request_id
    }


# Convenience log functions for specific levels
def log_info(phase: Union[str, EventPhase], message: str, **metrics):
    """Log an info message."""
    log_event(phase, message, LogLevel.INFO, **metrics)


def log_warning(phase: Union[str, EventPhase], message: str, **metrics):
    """Log a warning message."""
    log_event(phase, message, LogLevel.WARNING, **metrics)


def log_error(phase: Union[str, EventPhase], message: str, **metrics):
    """Log an error message."""
    log_event(phase, message, LogLevel.ERROR, **metrics)


def log_success(phase: Union[str, EventPhase], message: str, **metrics):
    """Log a success message."""
    log_event(phase, message, LogLevel.SUCCESS, **metrics)


def log_metric(phase: Union[str, EventPhase], message: str, **metrics):
    """Log a metric message."""
    log_event(phase, message, LogLevel.METRIC, **metrics)


if __name__ == "__main__":
    """Example usage of the logging utilities."""
    
    # Set up request context
    set_request_context("test-request-123", "test_log.json")
    
    # Log various events
    log_event(EventPhase.PLANNING, "Starting task planning", task_type="pick_and_place")
    
    log_event(EventPhase.BEHAVIOR_CLONING, "Training epoch 1", 
              epoch=1, loss=0.456, learning_rate=0.001)
    
    log_event(EventPhase.BEHAVIOR_CLONING, "Training epoch 2",
              epoch=2, loss=0.234, learning_rate=0.001)
    
    log_warning(EventPhase.VISION_REFINEMENT, "Low confidence detection",
                confidence=0.3, dx=12, dy=-5)
    
    log_success(EventPhase.COMPLETED, "Task completed successfully",
                total_time=45.2, success_rate=1.0)
    
    # Get summary
    summary = get_execution_summary()
    print("\n=== Execution Summary ===")
    print(json.dumps(summary, indent=2))