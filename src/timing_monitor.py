"""
Timing Monitor with SSE Logging

This module provides wall-clock timing for each phase of the Cogniforge pipeline,
logs to Server-Sent Events (SSE) for real-time monitoring, and ensures total
execution stays under 150 seconds.
"""

import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from collections import deque
from contextlib import contextmanager
import warnings

# SSE server imports
from flask import Flask, Response, jsonify
from flask_cors import CORS
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Phase Definitions and Time Budgets
# ============================================================================

class ExecutionPhase(Enum):
    """Execution phases with time budgets"""
    # Phase name = (phase_id, max_time_seconds, is_critical)
    INITIALIZATION = ("init", 5.0, False)
    TASK_ANALYSIS = ("task_analysis", 5.0, False)
    VISION_PROCESSING = ("vision", 10.0, True)
    TRAJECTORY_GENERATION = ("trajectory", 15.0, True)
    OPTIMIZATION = ("optimization", 60.0, True)  # CMA-ES gets biggest budget
    ALIGN_EXECUTION = ("align", 10.0, True)
    GRASP_EXECUTION = ("grasp", 20.0, True)
    MOVE_EXECUTION = ("move", 15.0, False)
    PLACE_EXECUTION = ("place", 8.0, False)
    LEARNING_UPDATE = ("learning", 2.0, False)
    
    def __init__(self, phase_id: str, max_time: float, is_critical: bool):
        self.phase_id = phase_id
        self.max_time = max_time
        self.is_critical = is_critical


@dataclass
class PhaseMetrics:
    """Metrics for a single phase execution"""
    phase_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    status: str = "running"  # running, success, timeout, failed
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def complete(self, status: str = "success"):
        """Mark phase as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        self.progress = 1.0
        
    def to_sse_event(self) -> str:
        """Convert to SSE event format"""
        data = {
            "phase": self.phase_name,
            "start_time": self.start_time,
            "duration": self.duration,
            "status": self.status,
            "progress": self.progress,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "timestamp": time.time()
        }
        return f"data: {json.dumps(data)}\n\n"


# ============================================================================
# Timing Monitor with SSE Support
# ============================================================================

class TimingMonitor:
    """
    Central timing monitor that tracks all phases and ensures 150-second target
    """
    
    MAX_TOTAL_TIME = 150.0  # Target: 150 seconds total
    WARNING_THRESHOLD = 0.8  # Warn at 80% of budget
    
    def __init__(self, enable_sse: bool = True):
        """
        Initialize timing monitor
        
        Args:
            enable_sse: Whether to enable SSE server
        """
        self.phases: Dict[str, PhaseMetrics] = {}
        self.current_phase: Optional[PhaseMetrics] = None
        self.start_time = 0.0
        self.total_time_used = 0.0
        self.sse_queue = queue.Queue() if enable_sse else None
        self.enable_sse = enable_sse
        self.abort_flag = threading.Event()
        self.time_budget_exceeded = False
        
        # Performance history for learning
        self.execution_history = deque(maxlen=100)
        
        # Phase dependencies and critical path
        self.critical_path = [
            ExecutionPhase.TASK_ANALYSIS,
            ExecutionPhase.VISION_PROCESSING,
            ExecutionPhase.TRAJECTORY_GENERATION,
            ExecutionPhase.OPTIMIZATION,
            ExecutionPhase.GRASP_EXECUTION
        ]
        
        # Start SSE server if enabled
        if enable_sse:
            self._start_sse_server()
    
    def _start_sse_server(self):
        """Start Flask SSE server in background thread"""
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/timing/stream')
        def stream():
            """SSE endpoint for real-time timing updates"""
            def generate():
                while not self.abort_flag.is_set():
                    try:
                        # Get event from queue with timeout
                        event = self.sse_queue.get(timeout=1.0)
                        yield event
                    except queue.Empty:
                        # Send heartbeat
                        yield "event: heartbeat\ndata: {}\n\n"
            
            return Response(generate(), mimetype="text/event-stream")
        
        @app.route('/timing/status')
        def status():
            """Get current timing status"""
            return jsonify({
                "total_time_used": self.total_time_used,
                "total_budget": self.MAX_TOTAL_TIME,
                "time_remaining": self.MAX_TOTAL_TIME - self.total_time_used,
                "current_phase": self.current_phase.phase_name if self.current_phase else None,
                "phases_completed": len([p for p in self.phases.values() if p.status != "running"]),
                "abort_flag": self.abort_flag.is_set()
            })
        
        @app.route('/timing/abort', methods=['POST'])
        def abort():
            """Abort execution if time budget exceeded"""
            self.abort_execution("Manual abort requested")
            return jsonify({"status": "aborted"})
        
        # Run server in background thread
        def run_server():
            app.run(host='0.0.0.0', port=5555, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info("SSE server started on http://localhost:5555")
    
    @contextmanager
    def phase(self, phase: ExecutionPhase, metadata: Dict[str, Any] = None):
        """
        Context manager for timing a phase
        
        Args:
            phase: The execution phase
            metadata: Optional metadata for the phase
            
        Example:
            with monitor.phase(ExecutionPhase.OPTIMIZATION):
                # Run optimization
                pass
        """
        # Check if we should abort
        if self.abort_flag.is_set() or self.time_budget_exceeded:
            logger.error(f"Skipping {phase.phase_id} - time budget exceeded")
            raise TimeoutError("Total time budget exceeded")
        
        # Start phase
        metrics = self.start_phase(phase, metadata)
        
        try:
            # Set up phase timeout
            phase_start = time.time()
            
            # Create timer for phase timeout
            def check_timeout():
                while not self.abort_flag.is_set():
                    elapsed = time.time() - phase_start
                    metrics.progress = min(elapsed / phase.max_time, 1.0)
                    
                    # Update SSE
                    if self.sse_queue:
                        self.sse_queue.put(metrics.to_sse_event())
                    
                    # Check phase timeout
                    if elapsed > phase.max_time:
                        if phase.is_critical:
                            logger.error(f"CRITICAL phase {phase.phase_id} exceeded time budget!")
                            metrics.warnings.append(f"Exceeded budget by {elapsed - phase.max_time:.1f}s")
                        else:
                            logger.warning(f"Phase {phase.phase_id} exceeded time budget")
                        metrics.status = "timeout"
                        break
                    
                    # Check total timeout
                    if self.total_time_used + elapsed > self.MAX_TOTAL_TIME:
                        logger.error("TOTAL TIME BUDGET EXCEEDED!")
                        self.time_budget_exceeded = True
                        self.abort_flag.set()
                        break
                    
                    time.sleep(0.1)  # Check every 100ms
            
            timeout_thread = threading.Thread(target=check_timeout, daemon=True)
            timeout_thread.start()
            
            yield metrics
            
            # Phase completed successfully
            self.end_phase(phase, "success")
            
        except Exception as e:
            # Phase failed
            logger.error(f"Phase {phase.phase_id} failed: {e}")
            self.end_phase(phase, "failed")
            raise
    
    def start_phase(self, phase: ExecutionPhase, metadata: Dict[str, Any] = None) -> PhaseMetrics:
        """Start timing a phase"""
        if self.start_time == 0:
            self.start_time = time.time()
        
        metrics = PhaseMetrics(
            phase_name=phase.phase_id,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.current_phase = metrics
        self.phases[phase.phase_id] = metrics
        
        # Log to SSE
        if self.sse_queue:
            event_data = {
                "event": "phase_start",
                "phase": phase.phase_id,
                "budget": phase.max_time,
                "total_time_used": self.total_time_used,
                "time_remaining": self.MAX_TOTAL_TIME - self.total_time_used
            }
            self.sse_queue.put(f"data: {json.dumps(event_data)}\n\n")
        
        logger.info(f"▶ Starting {phase.phase_id} (budget: {phase.max_time}s, remaining: {self.MAX_TOTAL_TIME - self.total_time_used:.1f}s)")
        
        return metrics
    
    def end_phase(self, phase: ExecutionPhase, status: str = "success"):
        """End timing for a phase"""
        if phase.phase_id not in self.phases:
            logger.warning(f"Phase {phase.phase_id} was not started")
            return
        
        metrics = self.phases[phase.phase_id]
        metrics.complete(status)
        
        self.total_time_used += metrics.duration
        self.current_phase = None
        
        # Check if we're running low on time
        time_remaining = self.MAX_TOTAL_TIME - self.total_time_used
        if time_remaining < 20 and time_remaining > 0:  # Less than 20 seconds left
            logger.warning(f"⚠ Only {time_remaining:.1f}s remaining in total budget!")
        
        # Log to SSE
        if self.sse_queue:
            event_data = {
                "event": "phase_end",
                "phase": phase.phase_id,
                "duration": metrics.duration,
                "status": status,
                "total_time_used": self.total_time_used,
                "time_remaining": self.MAX_TOTAL_TIME - self.total_time_used
            }
            self.sse_queue.put(f"data: {json.dumps(event_data)}\n\n")
        
        # Log result
        icon = "✓" if status == "success" else "✗" if status == "failed" else "⏱"
        logger.info(f"{icon} Completed {phase.phase_id} in {metrics.duration:.2f}s (status: {status})")
        
        # Add to history
        self.execution_history.append({
            "phase": phase.phase_id,
            "duration": metrics.duration,
            "status": status,
            "timestamp": metrics.end_time
        })
    
    def abort_execution(self, reason: str = "Time budget exceeded"):
        """Abort all execution"""
        self.abort_flag.set()
        logger.error(f"EXECUTION ABORTED: {reason}")
        
        if self.sse_queue:
            event_data = {
                "event": "abort",
                "reason": reason,
                "total_time_used": self.total_time_used,
                "phases_completed": len([p for p in self.phases.values() if p.status != "running"])
            }
            self.sse_queue.put(f"data: {json.dumps(event_data)}\n\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution timing summary"""
        completed_phases = [p for p in self.phases.values() if p.status != "running"]
        
        summary = {
            "total_time": self.total_time_used,
            "budget_used": self.total_time_used / self.MAX_TOTAL_TIME,
            "phases_completed": len(completed_phases),
            "phases_timed_out": len([p for p in completed_phases if p.status == "timeout"]),
            "phases_failed": len([p for p in completed_phases if p.status == "failed"]),
            "within_budget": self.total_time_used <= self.MAX_TOTAL_TIME,
            "phase_breakdown": {}
        }
        
        for phase_id, metrics in self.phases.items():
            summary["phase_breakdown"][phase_id] = {
                "duration": metrics.duration,
                "status": metrics.status,
                "warnings": metrics.warnings
            }
        
        return summary
    
    def print_report(self):
        """Print detailed timing report"""
        print("\n" + "="*60)
        print(" TIMING REPORT")
        print("="*60)
        
        # Overall status
        within_budget = self.total_time_used <= self.MAX_TOTAL_TIME
        status_icon = "✓" if within_budget else "✗"
        print(f"\n{status_icon} Total Time: {self.total_time_used:.2f}s / {self.MAX_TOTAL_TIME}s")
        print(f"   Budget Used: {self.total_time_used / self.MAX_TOTAL_TIME:.1%}")
        
        # Phase breakdown
        print("\n Phase Breakdown:")
        print(" " + "-"*55)
        print(f" {'Phase':<20} {'Duration':<10} {'Budget':<10} {'Status':<10} {'Notes'}")
        print(" " + "-"*55)
        
        for phase in ExecutionPhase:
            if phase.phase_id in self.phases:
                metrics = self.phases[phase.phase_id]
                duration_str = f"{metrics.duration:.2f}s"
                budget_str = f"{phase.max_time:.1f}s"
                
                # Color coding for status
                if metrics.status == "success" and metrics.duration <= phase.max_time:
                    status = "✓ OK"
                elif metrics.status == "timeout":
                    status = "⏱ TIMEOUT"
                elif metrics.status == "failed":
                    status = "✗ FAILED"
                else:
                    status = metrics.status
                
                notes = ""
                if metrics.duration > phase.max_time:
                    notes = f"Over by {metrics.duration - phase.max_time:.1f}s"
                
                print(f" {phase.phase_id:<20} {duration_str:<10} {budget_str:<10} {status:<10} {notes}")
        
        print(" " + "-"*55)
        
        # Recommendations
        print("\n Recommendations:")
        if not within_budget:
            print(" ⚠ Total time exceeded budget!")
            print(" Consider:")
            
            # Find phases that went over budget
            over_budget = []
            for phase in ExecutionPhase:
                if phase.phase_id in self.phases:
                    metrics = self.phases[phase.phase_id]
                    if metrics.duration > phase.max_time:
                        over_budget.append((phase, metrics.duration - phase.max_time))
            
            if over_budget:
                over_budget.sort(key=lambda x: x[1], reverse=True)
                for phase, excess in over_budget[:3]:  # Top 3 offenders
                    print(f"   - Optimize {phase.phase_id}: {excess:.1f}s over budget")
        else:
            print(" ✓ Execution within time budget")
            time_margin = self.MAX_TOTAL_TIME - self.total_time_used
            print(f" ✓ Time margin: {time_margin:.1f}s ({time_margin/self.MAX_TOTAL_TIME:.1%})")
        
        print("="*60 + "\n")


# ============================================================================
# Performance Optimizer
# ============================================================================

class PerformanceOptimizer:
    """
    Analyzes timing data and suggests optimizations to meet 150-second target
    """
    
    def __init__(self, monitor: TimingMonitor):
        self.monitor = monitor
        self.optimization_suggestions = []
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations"""
        summary = self.monitor.get_summary()
        
        analysis = {
            "meets_target": summary["within_budget"],
            "total_time": summary["total_time"],
            "bottlenecks": [],
            "optimizations": []
        }
        
        # Identify bottlenecks
        phase_times = []
        for phase_id, data in summary["phase_breakdown"].items():
            phase = self._get_phase_by_id(phase_id)
            if phase and data["duration"] > 0:
                efficiency = data["duration"] / phase.max_time
                phase_times.append({
                    "phase": phase_id,
                    "duration": data["duration"],
                    "budget": phase.max_time,
                    "efficiency": efficiency,
                    "is_bottleneck": efficiency > 1.0
                })
        
        # Sort by efficiency (worst first)
        phase_times.sort(key=lambda x: x["efficiency"], reverse=True)
        analysis["bottlenecks"] = [p for p in phase_times if p["is_bottleneck"]]
        
        # Generate optimization suggestions
        for bottleneck in analysis["bottlenecks"]:
            phase_id = bottleneck["phase"]
            excess = bottleneck["duration"] - bottleneck["budget"]
            
            if phase_id == "optimization":
                analysis["optimizations"].append({
                    "phase": phase_id,
                    "suggestion": "Reduce CMA-ES population size or iterations",
                    "potential_savings": min(excess, 20.0),
                    "implementation": "Set smaller population_size or use early stopping"
                })
            elif phase_id == "grasp":
                analysis["optimizations"].append({
                    "phase": phase_id,
                    "suggestion": "Reduce grasp retry attempts",
                    "potential_savings": min(excess, 10.0),
                    "implementation": "Limit retries to 2 instead of 4"
                })
            elif phase_id == "vision":
                analysis["optimizations"].append({
                    "phase": phase_id,
                    "suggestion": "Use color threshold fallback immediately",
                    "potential_savings": min(excess, 5.0),
                    "implementation": "Skip GPT vision API, use color detection"
                })
            elif phase_id == "trajectory":
                analysis["optimizations"].append({
                    "phase": phase_id,
                    "suggestion": "Use pre-computed trajectories",
                    "potential_savings": min(excess, 10.0),
                    "implementation": "Use 3-waypoint fallback instead of GPT generation"
                })
        
        # Calculate if optimizations would meet target
        total_potential_savings = sum(opt["potential_savings"] for opt in analysis["optimizations"])
        analysis["optimized_time"] = summary["total_time"] - total_potential_savings
        analysis["would_meet_target"] = analysis["optimized_time"] <= 150.0
        
        return analysis
    
    def _get_phase_by_id(self, phase_id: str) -> Optional[ExecutionPhase]:
        """Get ExecutionPhase by its ID"""
        for phase in ExecutionPhase:
            if phase.phase_id == phase_id:
                return phase
        return None
    
    def print_analysis(self):
        """Print optimization analysis"""
        analysis = self.analyze()
        
        print("\n" + "="*60)
        print(" PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Current status
        status = "✓ MEETS TARGET" if analysis["meets_target"] else "✗ EXCEEDS TARGET"
        print(f"\n Status: {status}")
        print(f" Current Time: {analysis['total_time']:.1f}s")
        print(f" Target Time: 150.0s")
        
        if analysis["bottlenecks"]:
            print("\n Bottlenecks Identified:")
            for bottleneck in analysis["bottlenecks"]:
                print(f"   • {bottleneck['phase']}: {bottleneck['duration']:.1f}s (budget: {bottleneck['budget']:.1f}s)")
        
        if analysis["optimizations"]:
            print("\n Optimization Suggestions:")
            for i, opt in enumerate(analysis["optimizations"], 1):
                print(f"\n   {i}. {opt['phase'].upper()}")
                print(f"      Suggestion: {opt['suggestion']}")
                print(f"      Potential Savings: {opt['potential_savings']:.1f}s")
                print(f"      How: {opt['implementation']}")
            
            print(f"\n Projected Time After Optimizations: {analysis['optimized_time']:.1f}s")
            if analysis["would_meet_target"]:
                print(" ✓ Would meet 150s target with these optimizations")
            else:
                print(" ✗ Additional optimizations needed")
        else:
            print("\n ✓ No optimizations needed - already within target")
        
        print("="*60 + "\n")


# ============================================================================
# Integrated Task Executor with Timing
# ============================================================================

class TimedTaskExecutor:
    """
    Executes tasks with comprehensive timing to meet 150-second target
    """
    
    def __init__(self, enable_sse: bool = True):
        self.monitor = TimingMonitor(enable_sse=enable_sse)
        self.optimizer = PerformanceOptimizer(self.monitor)
        
    def execute_pick_and_place(self, blackboard) -> bool:
        """
        Execute pick-and-place task with timing constraints
        
        Args:
            blackboard: Shared blackboard for task
            
        Returns:
            True if completed within 150 seconds
        """
        from behavior_tree import NodeStatus
        from learnable_nodes import LearnableAlign, LearnableGrasp
        from behavior_tree import MoveTo, Place
        
        try:
            # Phase 1: Initialization
            with self.monitor.phase(ExecutionPhase.INITIALIZATION):
                logger.info("Initializing system...")
                # System checks, load models, etc.
                time.sleep(0.5)  # Simulate initialization
            
            # Phase 2: Task Analysis
            with self.monitor.phase(ExecutionPhase.TASK_ANALYSIS):
                logger.info("Analyzing task...")
                # Parse task, detect objects
                task_type = blackboard.get('task_type', 'pick_place')
                time.sleep(1.0)  # Simulate analysis
            
            # Phase 3: Vision Processing
            with self.monitor.phase(ExecutionPhase.VISION_PROCESSING):
                logger.info("Processing vision...")
                # Try GPT vision with quick timeout
                use_fallback = False
                
                try:
                    # Simulate GPT vision (with timeout)
                    time.sleep(2.0)
                    vision_result = "success"
                except:
                    # Use color threshold fallback
                    use_fallback = True
                    time.sleep(0.5)  # Fallback is faster
                    vision_result = "fallback"
                
                blackboard.set('vision_method', vision_result)
            
            # Phase 4: Trajectory Generation
            with self.monitor.phase(ExecutionPhase.TRAJECTORY_GENERATION):
                logger.info("Generating trajectory...")
                # Try GPT generation with fallback
                try:
                    # Simulate GPT trajectory generation
                    time.sleep(3.0)
                    trajectory_method = "gpt"
                except:
                    # Use 3-waypoint fallback
                    time.sleep(0.2)
                    trajectory_method = "fallback"
                
                blackboard.set('trajectory_method', trajectory_method)
            
            # Phase 5: Optimization (CMA-ES)
            with self.monitor.phase(ExecutionPhase.OPTIMIZATION) as metrics:
                logger.info("Running CMA-ES optimization...")
                
                # Adaptive optimization based on time remaining
                time_remaining = 150 - self.monitor.total_time_used
                
                if time_remaining > 80:
                    # Full optimization
                    iterations = 100
                    population_size = 20
                elif time_remaining > 50:
                    # Reduced optimization
                    iterations = 50
                    population_size = 10
                else:
                    # Minimal optimization
                    iterations = 20
                    population_size = 5
                
                metrics.metadata["iterations"] = iterations
                metrics.metadata["population_size"] = population_size
                
                # Simulate CMA-ES with early stopping
                for i in range(iterations):
                    if self.monitor.abort_flag.is_set():
                        break
                    
                    # Check time
                    if time.time() - metrics.start_time > 50:  # Hard limit at 50s
                        logger.warning("CMA-ES time limit reached, using best so far")
                        break
                    
                    # Simulate iteration
                    time.sleep(0.3)  # Each iteration takes ~300ms
                    
                    # Early stopping if converged
                    if i > 30 and np.random.random() < 0.1:  # Random convergence check
                        logger.info(f"CMA-ES converged at iteration {i}")
                        break
                
                metrics.metadata["final_iteration"] = i
            
            # Phase 6: Movement Execution
            with self.monitor.phase(ExecutionPhase.MOVE_EXECUTION):
                logger.info("Moving to grasp position...")
                move_to_grasp = MoveTo(blackboard, 'grasp_target')
                result = move_to_grasp()
                time.sleep(5.0)  # Simulate movement
                if result != NodeStatus.SUCCESS:
                    raise RuntimeError("Move to grasp failed")
            
            # Phase 7: Alignment (LEARNABLE)
            with self.monitor.phase(ExecutionPhase.ALIGN_EXECUTION):
                logger.info("Aligning with object...")
                align = LearnableAlign(blackboard)
                result = align()
                time.sleep(3.0)  # Simulate alignment
                if result != NodeStatus.SUCCESS:
                    logger.warning("Alignment failed, continuing anyway")
            
            # Phase 8: Grasping (LEARNABLE)
            with self.monitor.phase(ExecutionPhase.GRASP_EXECUTION) as metrics:
                logger.info("Executing grasp...")
                grasp = LearnableGrasp(blackboard)
                
                # Limit retries based on time remaining
                time_remaining = 150 - self.monitor.total_time_used
                max_retries = 4 if time_remaining > 30 else 2
                
                metrics.metadata["max_retries"] = max_retries
                
                # Simulate grasp with retries
                for attempt in range(max_retries):
                    time.sleep(3.0)  # Each attempt takes 3s
                    if np.random.random() < 0.7:  # 70% success rate
                        metrics.metadata["attempts"] = attempt + 1
                        break
                else:
                    raise RuntimeError("Grasp failed after all attempts")
            
            # Phase 9: Move to place
            with self.monitor.phase(ExecutionPhase.MOVE_EXECUTION):
                logger.info("Moving to place position...")
                move_to_place = MoveTo(blackboard, 'place_target')
                result = move_to_place()
                time.sleep(4.0)  # Simulate movement
            
            # Phase 10: Place
            with self.monitor.phase(ExecutionPhase.PLACE_EXECUTION):
                logger.info("Placing object...")
                place = Place(blackboard)
                result = place()
                time.sleep(2.0)  # Simulate placement
            
            # Phase 11: Learning update
            if self.monitor.total_time_used < 148:  # Only if time permits
                with self.monitor.phase(ExecutionPhase.LEARNING_UPDATE):
                    logger.info("Updating learning models...")
                    time.sleep(1.0)  # Quick update
            
            # Success!
            return True
            
        except TimeoutError as e:
            logger.error(f"Task aborted: {e}")
            return False
        except Exception as e:
            logger.error(f"Task failed: {e}")
            return False
        finally:
            # Print reports
            self.monitor.print_report()
            self.optimizer.print_analysis()
            
            # Send final SSE event
            if self.monitor.sse_queue:
                summary = self.monitor.get_summary()
                self.monitor.sse_queue.put(f"data: {json.dumps({'event': 'complete', 'summary': summary})}\n\n")


# ============================================================================
# SSE Dashboard HTML
# ============================================================================

def generate_dashboard_html() -> str:
    """Generate HTML dashboard for SSE monitoring"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Cogniforge Timing Monitor</title>
    <style>
        body {
            font-family: monospace;
            background: #1a1a1a;
            color: #00ff00;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            border-bottom: 2px solid #00ff00;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .timer {
            font-size: 48px;
            text-align: center;
            margin: 20px 0;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #333;
            border: 1px solid #00ff00;
            margin: 10px 0;
            position: relative;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff00, #ffff00, #ff0000);
            transition: width 0.3s;
        }
        .phase-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }
        .phase-card {
            border: 1px solid #00ff00;
            padding: 10px;
            background: #0a0a0a;
        }
        .phase-card.running {
            border-color: #ffff00;
            animation: pulse 1s infinite;
        }
        .phase-card.complete {
            border-color: #00ff00;
            opacity: 0.7;
        }
        .phase-card.timeout {
            border-color: #ff0000;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .log {
            height: 200px;
            overflow-y: auto;
            background: #0a0a0a;
            border: 1px solid #00ff00;
            padding: 10px;
            font-size: 12px;
        }
        .warning {
            color: #ffff00;
        }
        .error {
            color: #ff0000;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⏱ COGNIFORGE TIMING MONITOR</h1>
            <p>Target: 150 seconds | Real-time phase tracking via SSE</p>
        </div>
        
        <div class="timer" id="timer">0.0s / 150.0s</div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress" style="width: 0%"></div>
        </div>
        
        <div class="phase-grid" id="phases"></div>
        
        <h3>Event Log</h3>
        <div class="log" id="log"></div>
    </div>
    
    <script>
        const eventSource = new EventSource('http://localhost:5555/timing/stream');
        const phases = {};
        let totalTime = 0;
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.event === 'phase_start') {
                addPhase(data.phase, data.budget);
            } else if (data.event === 'phase_end') {
                updatePhase(data.phase, data.duration, data.status);
                totalTime = data.total_time_used;
                updateTimer(totalTime);
            } else if (data.event === 'abort') {
                addLog('ERROR: ' + data.reason, 'error');
            } else if (data.phase) {
                // Phase progress update
                updatePhaseProgress(data.phase, data.progress);
            }
        };
        
        function addPhase(name, budget) {
            const grid = document.getElementById('phases');
            const card = document.createElement('div');
            card.className = 'phase-card running';
            card.id = 'phase-' + name;
            card.innerHTML = `
                <h4>${name}</h4>
                <div>Budget: ${budget}s</div>
                <div>Status: Running...</div>
                <div class="progress-bar" style="height: 10px;">
                    <div class="progress-fill" id="progress-${name}" style="width: 0%"></div>
                </div>
            `;
            grid.appendChild(card);
            phases[name] = { budget: budget, start: Date.now() };
            addLog(`Started ${name} (budget: ${budget}s)`, 'info');
        }
        
        function updatePhase(name, duration, status) {
            const card = document.getElementById('phase-' + name);
            if (card) {
                card.className = 'phase-card ' + (status === 'success' ? 'complete' : 'timeout');
                card.querySelector('div:nth-child(3)').textContent = 
                    `Duration: ${duration.toFixed(2)}s (${status})`;
                
                const logClass = status === 'success' ? 'info' : 'warning';
                addLog(`Completed ${name}: ${duration.toFixed(2)}s (${status})`, logClass);
            }
        }
        
        function updatePhaseProgress(name, progress) {
            const progressBar = document.getElementById('progress-' + name);
            if (progressBar) {
                progressBar.style.width = (progress * 100) + '%';
            }
        }
        
        function updateTimer(time) {
            document.getElementById('timer').textContent = 
                `${time.toFixed(1)}s / 150.0s`;
            
            const percentage = (time / 150) * 100;
            document.getElementById('progress').style.width = percentage + '%';
            
            if (percentage > 100) {
                document.getElementById('timer').className = 'timer error';
            } else if (percentage > 80) {
                document.getElementById('timer').className = 'timer warning';
            }
        }
        
        function addLog(message, className) {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = className || '';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        
        // Heartbeat indicator
        eventSource.addEventListener('heartbeat', function() {
            // Could add visual heartbeat indicator
        });
    </script>
</body>
</html>
'''


# ============================================================================
# Example Usage
# ============================================================================

def example_timed_execution():
    """Example of executing a task with timing constraints"""
    from behavior_tree import Blackboard
    import numpy as np
    
    # Create blackboard
    blackboard = Blackboard()
    blackboard.update({
        'task_type': 'pick_and_place',
        'grasp_target': np.array([0.4, 0.0, 0.1]),
        'place_target': np.array([0.6, 0.2, 0.1]),
        'robot_position': np.array([0.0, 0.0, 0.3]),
        'robot_orientation': np.array([0.0, 0.0, 0.0]),
        'gripper_state': 'open'
    })
    
    # Create executor with timing
    executor = TimedTaskExecutor(enable_sse=True)
    
    # Save dashboard HTML
    with open('timing_dashboard.html', 'w') as f:
        f.write(generate_dashboard_html())
    print("Dashboard saved to timing_dashboard.html")
    print("Open in browser to see real-time timing")
    
    # Execute task with 150-second constraint
    print("\n" + "="*60)
    print(" EXECUTING TASK WITH 150-SECOND TARGET")
    print("="*60 + "\n")
    
    success = executor.execute_pick_and_place(blackboard)
    
    # Final result
    if success:
        print("\n✓ TASK COMPLETED SUCCESSFULLY WITHIN TIME BUDGET")
    else:
        print("\n✗ TASK FAILED OR EXCEEDED TIME BUDGET")
    
    return success


if __name__ == "__main__":
    # Run example
    example_timed_execution()
    
    # Keep SSE server running
    print("\nSSE server running at http://localhost:5555")
    print("Open timing_dashboard.html in a browser to see real-time updates")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")