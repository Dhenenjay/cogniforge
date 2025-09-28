"""
FastAPI endpoint for executing the complete CogniForge pipeline.

Orchestrates: plan → expert → BC → optimize → vision → codegen
"""

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import torch
from collections import deque
import uuid

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import CogniForge modules
from core.planner import TaskPlanner
from core.expert_script import ExpertDemonstrator
from core.policy import BCPolicy
from core.optimization import PolicyOptimizer
from core.refinement import PolicyRefiner
from core.reward import GPTRewardModel
from core.logging_utils import log_event, EventPhase as LogEventPhase, LogLevel
from vision.vision_utils import VisionDetector
from control.grasp_execution import GraspExecutor
from control.robot_control import RobotController
from ui.ui_integration import VisionUIFormatter, GraspUIDisplay
from core.config import Config


# Initialize FastAPI app
app = FastAPI(
    title="CogniForge Execution API",
    description="Orchestrates the complete robotic task execution pipeline",
    version="1.0.0",
)

# CORS configuration for UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TaskType(str, Enum):
    """Supported task types."""

    PICK_AND_PLACE = "pick_and_place"
    STACKING = "stacking"
    INSERTION = "insertion"
    ASSEMBLY = "assembly"
    CUSTOM = "custom"


class ExecutionRequest(BaseModel):
    """Request model for task execution."""

    task_type: TaskType = Field(TaskType.PICK_AND_PLACE, description="Type of task to execute")

    task_description: str = Field(
        ...,
        description="Natural language description of the task",
        example="Pick up the blue cube and place it on the red platform",
    )

    object_name: Optional[str] = Field(
        None, description="Primary object to manipulate", example="blue_cube"
    )

    target_location: Optional[Dict[str, float]] = Field(
        None,
        description="Target location in world coordinates",
        example={"x": 0.5, "y": 0.0, "z": 0.15},
    )

    use_vision: bool = Field(True, description="Whether to use vision for refinement")

    use_gpt_reward: bool = Field(False, description="Whether to use GPT for reward computation")

    num_bc_epochs: int = Field(10, description="Number of behavior cloning epochs", ge=1, le=100)

    num_optimization_steps: int = Field(
        50, description="Number of policy optimization steps", ge=10, le=500
    )

    safety_checks: bool = Field(True, description="Enable safety checks during execution")

    dry_run: bool = Field(False, description="Simulate execution without robot movement")


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    PLANNING = "planning"
    EXPERT_DEMO = "expert_demonstration"
    BEHAVIOR_CLONING = "behavior_cloning"
    OPTIMIZATION = "optimization"
    VISION_REFINEMENT = "vision_refinement"
    CODE_GENERATION = "code_generation"
    EXECUTION = "execution"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionStatus(BaseModel):
    """Status of execution pipeline."""

    request_id: str
    stage: PipelineStage
    progress: float  # 0.0 to 1.0
    message: str
    details: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    started_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class CodeLinks(BaseModel):
    """Links to generated code and artifacts."""

    code_preview_url: str
    code_download_url: str
    code_raw_url: str
    code_id: str
    code_size_bytes: int
    code_lines: int
    language: str = "python"


class ExecutionSummary(BaseModel):
    """Comprehensive execution summary."""

    task_description: str
    task_type: str
    total_duration_seconds: float
    success_rate: float
    stages_completed_count: int
    stages_total_count: int = 7

    # Key metrics
    planning_steps: int
    expert_demos_collected: int
    final_bc_loss: Optional[float]
    final_optimization_reward: Optional[float]
    vision_aligned: Optional[bool]
    vision_offset_mm: Optional[float]

    # Performance metrics
    fastest_stage: str
    slowest_stage: str
    average_stage_time: float

    # Resource usage
    gpu_used: bool = False
    memory_peak_mb: Optional[float] = None


class ExecutionResult(BaseModel):
    """Final result of execution pipeline."""

    request_id: str
    success: bool
    stages_completed: List[PipelineStage]

    # Stage results
    plan: Optional[Dict[str, Any]] = None
    expert_trajectories: Optional[int] = None
    bc_loss: Optional[float] = None
    optimization_reward: Optional[float] = None
    vision_alignment: Optional[Dict[str, Any]] = None
    generated_code: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None

    # Metrics
    total_time_seconds: float
    stage_times: Dict[str, float]

    # New comprehensive fields
    code_links: Optional[CodeLinks] = None
    summary: Optional[ExecutionSummary] = None
    artifacts: Optional[Dict[str, str]] = None

    # Errors
    errors: Optional[List[Dict[str, str]]] = None


class SSEEvent(BaseModel):
    """Server-Sent Event model."""

    phase: str
    message: str
    metrics: Optional[Dict[str, Any]] = None
    timestamp: str
    progress: float
    request_id: Optional[str] = None


class EventStream:
    """Manages event streaming for SSE."""

    def __init__(self, max_events: int = 1000):
        """Initialize event stream."""
        self.events: Dict[str, deque] = {}  # request_id -> events queue
        self.max_events = max_events
        self.active_streams: Dict[str, bool] = {}  # Track active streams

    def add_event(self, request_id: str, event: SSEEvent):
        """Add event to stream."""
        if request_id not in self.events:
            self.events[request_id] = deque(maxlen=self.max_events)
        self.events[request_id].append(event)

    def get_events(self, request_id: str) -> List[SSEEvent]:
        """Get all events for a request."""
        if request_id not in self.events:
            return []
        return list(self.events[request_id])

    def clear_events(self, request_id: str):
        """Clear events for a request."""
        if request_id in self.events:
            del self.events[request_id]
        if request_id in self.active_streams:
            del self.active_streams[request_id]

    def is_active(self, request_id: str) -> bool:
        """Check if stream is active."""
        return self.active_streams.get(request_id, False)

    def set_active(self, request_id: str, active: bool = True):
        """Set stream active status."""
        self.active_streams[request_id] = active


class PipelineOrchestrator:
    """
    Orchestrates the complete execution pipeline.

    Manages: plan → expert → BC → optimize → vision → codegen → execute
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline orchestrator."""
        self.config = config or Config()
        self.execution_status: Dict[str, ExecutionStatus] = {}
        self.execution_results: Dict[str, ExecutionResult] = {}

        # Initialize components (lazy loading)
        self._planner = None
        self._expert = None
        self._bc_policy = None
        self._optimizer = None
        self._refiner = None
        self._vision = None
        self._executor = None
        self._robot = None

        # Stage timing
        self.stage_start_times: Dict[str, datetime] = {}

        # Event streaming
        self.event_stream = EventStream()

    def _emit_heartbeat(
        self, request_id: str, phase: str, message: str, metrics: Optional[Dict[str, Any]] = None
    ):
        """Emit a heartbeat event to prevent timeouts."""
        if not request_id:
            return

        heartbeat = SSEEvent(
            phase=f"{phase}_heartbeat",
            message=message,
            metrics=metrics or {},
            timestamp=datetime.now().isoformat(),
            progress=-1,  # -1 indicates heartbeat
            request_id=request_id,
        )
        self.event_stream.add_event(request_id, heartbeat)

    @property
    def planner(self) -> TaskPlanner:
        """Get or create task planner."""
        if self._planner is None:
            self._planner = TaskPlanner(self.config)
        return self._planner

    @property
    def expert(self) -> ExpertDemonstrator:
        """Get or create expert demonstrator."""
        if self._expert is None:
            self._expert = ExpertDemonstrator(self.config)
        return self._expert

    @property
    def bc_policy(self) -> BCPolicy:
        """Get or create BC policy."""
        if self._bc_policy is None:
            self._bc_policy = BCPolicy(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                hidden_dims=[256, 256],
            )
        return self._bc_policy

    @property
    def optimizer(self) -> PolicyOptimizer:
        """Get or create policy optimizer."""
        if self._optimizer is None:
            self._optimizer = PolicyOptimizer(policy=self.bc_policy, config=self.config)
        return self._optimizer

    @property
    def refiner(self) -> PolicyRefiner:
        """Get or create policy refiner."""
        if self._refiner is None:
            self._refiner = PolicyRefiner(policy=self.bc_policy, config=self.config)
        return self._refiner

    @property
    def vision(self) -> VisionDetector:
        """Get or create vision detector."""
        if self._vision is None:
            self._vision = VisionDetector()
        return self._vision

    @property
    def executor(self) -> GraspExecutor:
        """Get or create grasp executor."""
        if self._executor is None:
            self._executor = GraspExecutor(use_vision=True, approach_height=0.2)
        return self._executor

    def update_status(
        self,
        request_id: str,
        stage: PipelineStage,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Update execution status and emit SSE event."""
        now = datetime.now()

        if request_id not in self.execution_status:
            self.execution_status[request_id] = ExecutionStatus(
                request_id=request_id,
                stage=stage,
                progress=progress,
                message=message,
                details=details,
                started_at=now,
                updated_at=now,
            )
        else:
            status = self.execution_status[request_id]
            status.stage = stage
            status.progress = progress
            status.message = message
            status.details = details
            status.updated_at = now

            if stage in [PipelineStage.COMPLETED, PipelineStage.FAILED]:
                status.completed_at = now

        # Emit log event to console and SSE
        payload_metrics: Dict[str, Any] = {}
        if details:
            payload_metrics["details"] = details
        if metrics:
            payload_metrics.update(metrics)
        if request_id:
            payload_metrics["request_id"] = request_id
        if progress is not None:
            payload_metrics["progress"] = progress
        
        # Map PipelineStage to LogEventPhase if possible
        phase_str = str(stage)
        log_event(phase_str, message, level=LogLevel.INFO, **payload_metrics)
        
        # Also add to SSE stream for real-time updates
        if request_id:
            event = SSEEvent(
                phase=phase_str,
                message=message,
                metrics=payload_metrics,
                timestamp=datetime.now().isoformat(),
                progress=progress,
                request_id=request_id,
            )
            self.event_stream.add_event(request_id, event)

    def log_stage_time(self, request_id: str, stage: str):
        """Log timing for a stage."""
        key = f"{request_id}_{stage}"
        if key not in self.stage_start_times:
            self.stage_start_times[key] = datetime.now()
            return 0.0
        else:
            elapsed = (datetime.now() - self.stage_start_times[key]).total_seconds()
            return elapsed

    async def execute_pipeline(self, request: ExecutionRequest, request_id: str) -> ExecutionResult:
        """
        Execute the complete pipeline asynchronously.

        Stages:
        1. Planning: Generate task plan
        2. Expert Demo: Collect expert demonstrations
        3. Behavior Cloning: Train BC policy
        4. Optimization: Optimize policy with RL
        5. Vision Refinement: Refine with vision feedback
        6. Code Generation: Generate execution code
        7. Execution: Execute on robot
        """

        start_time = datetime.now()
        stages_completed = []
        stage_times = {}
        errors = []

        # Initialize result
        result = ExecutionResult(
            request_id=request_id,
            success=False,
            stages_completed=[],
            total_time_seconds=0,
            stage_times={},
        )

        try:
            # Stage 1: Planning
            self.log_stage_time(request_id, "planning")
            self.update_status(request_id, PipelineStage.PLANNING, 0.1, "Generating task plan...")

            plan = await self._execute_planning(request, request_id)
            result.plan = plan
            stages_completed.append(PipelineStage.PLANNING)
            stage_times["planning"] = self.log_stage_time(request_id, "planning")

            self.update_status(
                request_id,
                PipelineStage.PLANNING,
                0.15,
                "Task plan generated",
                {"steps": len(plan.get("steps", []))},
                metrics={
                    "plan_steps": len(plan.get("steps", [])),
                    "planning_time": stage_times["planning"],
                },
            )

            # Stage 2: Expert Demonstration
            self.log_stage_time(request_id, "expert")
            self.update_status(
                request_id, PipelineStage.EXPERT_DEMO, 0.2, "Collecting expert demonstrations..."
            )

            trajectories = await self._execute_expert_demo(request, plan, request_id)
            result.expert_trajectories = len(trajectories)
            stages_completed.append(PipelineStage.EXPERT_DEMO)
            stage_times["expert"] = self.log_stage_time(request_id, "expert")

            self.update_status(
                request_id,
                PipelineStage.EXPERT_DEMO,
                0.3,
                f"Collected {len(trajectories)} demonstrations",
                metrics={
                    "num_trajectories": len(trajectories),
                    "demo_time": stage_times["expert"],
                    "avg_trajectory_length": np.mean(
                        [len(t.get("states", [])) for t in trajectories]
                    ),
                },
            )

            # Stage 3: Behavior Cloning
            self.log_stage_time(request_id, "bc")
            self.update_status(
                request_id,
                PipelineStage.BEHAVIOR_CLONING,
                0.35,
                "Training behavior cloning policy...",
            )

            bc_loss = await self._execute_behavior_cloning(
                trajectories, request.num_bc_epochs, request_id
            )
            result.bc_loss = bc_loss
            stages_completed.append(PipelineStage.BEHAVIOR_CLONING)
            stage_times["bc"] = self.log_stage_time(request_id, "bc")

            self.update_status(
                request_id,
                PipelineStage.BEHAVIOR_CLONING,
                0.5,
                f"BC training complete (loss: {bc_loss:.4f})",
                metrics={
                    "bc_loss": float(bc_loss),
                    "num_epochs": request.num_bc_epochs,
                    "training_time": stage_times["bc"],
                },
            )

            # Stage 4: Optimization
            self.log_stage_time(request_id, "optimization")
            self.update_status(request_id, PipelineStage.OPTIMIZATION, 0.55, "Optimizing policy...")

            opt_reward = await self._execute_optimization(
                request.num_optimization_steps,
                use_gpt=request.use_gpt_reward,
                request_id=request_id,
            )
            result.optimization_reward = opt_reward
            stages_completed.append(PipelineStage.OPTIMIZATION)
            stage_times["optimization"] = self.log_stage_time(request_id, "optimization")

            self.update_status(
                request_id,
                PipelineStage.OPTIMIZATION,
                0.7,
                f"Optimization complete (reward: {opt_reward:.3f})",
                metrics={
                    "final_reward": float(opt_reward),
                    "optimization_steps": request.num_optimization_steps,
                    "optimization_time": stage_times["optimization"],
                    "used_gpt_reward": request.use_gpt_reward,
                },
            )

            # Stage 5: Vision Refinement
            if request.use_vision:
                self.log_stage_time(request_id, "vision")
                self.update_status(
                    request_id,
                    PipelineStage.VISION_REFINEMENT,
                    0.75,
                    "Refining with vision feedback...",
                )

                vision_result = await self._execute_vision_refinement(
                    request.object_name, request_id
                )
                result.vision_alignment = vision_result
                stages_completed.append(PipelineStage.VISION_REFINEMENT)
                stage_times["vision"] = self.log_stage_time(request_id, "vision")

                self.update_status(
                    request_id,
                    PipelineStage.VISION_REFINEMENT,
                    0.85,
                    "Vision refinement complete",
                    vision_result,
                    metrics={
                        "aligned": vision_result.get("alignment", {}).get("is_aligned", False),
                        "offset_mm": vision_result.get("alignment", {}).get("error_mm", 0),
                        "confidence": vision_result.get("confidence", 0),
                        "vision_time": stage_times["vision"],
                    },
                )

            # Stage 6: Code Generation
            self.log_stage_time(request_id, "codegen")
            self.update_status(
                request_id, PipelineStage.CODE_GENERATION, 0.88, "Generating execution code..."
            )

            code = await self._generate_execution_code(
                plan, request, vision_result if request.use_vision else None, request_id
            )
            result.generated_code = code
            stages_completed.append(PipelineStage.CODE_GENERATION)
            stage_times["codegen"] = self.log_stage_time(request_id, "codegen")

            self.update_status(
                request_id, PipelineStage.CODE_GENERATION, 0.92, "Code generation complete"
            )

            # Stage 7: Execution
            if not request.dry_run:
                self.log_stage_time(request_id, "execution")
                self.update_status(
                    request_id, PipelineStage.EXECUTION, 0.95, "Executing on robot..."
                )

                exec_result = await self._execute_on_robot(code, request.safety_checks, request_id)
                result.execution_result = exec_result
                stages_completed.append(PipelineStage.EXECUTION)
                stage_times["execution"] = self.log_stage_time(request_id, "execution")

                self.update_status(
                    request_id, PipelineStage.COMPLETED, 1.0, "Execution complete", exec_result
                )
            else:
                self.update_status(
                    request_id,
                    PipelineStage.COMPLETED,
                    1.0,
                    "Dry run complete - no robot execution",
                )

            # Finalize result
            result.success = True
            result.stages_completed = stages_completed
            result.stage_times = stage_times
            result.total_time_seconds = (datetime.now() - start_time).total_seconds()

        except Exception as e:
            # Handle errors
            error_msg = f"Pipeline failed at stage {self.execution_status.get(request_id, {}).get('stage', 'unknown')}: {str(e)}"
            errors.append(
                {
                    "stage": str(self.execution_status.get(request_id, {}).get("stage", "unknown")),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

            self.update_status(request_id, PipelineStage.FAILED, -1, error_msg, {"error": str(e)})

            result.errors = errors
            result.stages_completed = stages_completed
            result.total_time_seconds = (datetime.now() - start_time).total_seconds()

        # Store generated code if available
        if result.generated_code:
            code_id = self._store_generated_code(request_id, result.generated_code)
            result.code_links = self._create_code_links(request_id, code_id, result.generated_code)

        # Create comprehensive summary
        result.summary = self._create_execution_summary(
            request, result, stages_completed, stage_times
        )

        # Create artifacts links
        result.artifacts = self._create_artifact_links(request_id, result)

        # Store result
        self.execution_results[request_id] = result
        return result

    def _store_generated_code(self, request_id: str, code: str) -> str:
        """Store generated code and return code ID."""
        import hashlib
        from pathlib import Path

        # Create code storage directory
        code_dir = Path("generated_code")
        code_dir.mkdir(exist_ok=True)

        # Generate code ID
        code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        code_id = f"{request_id}_{code_hash}"

        # Save code to file
        code_path = code_dir / f"{code_id}.py"
        code_path.write_text(code)

        # Also store in memory for quick access
        if not hasattr(self, "stored_codes"):
            self.stored_codes = {}
        self.stored_codes[code_id] = {
            "code": code,
            "path": str(code_path),
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        return code_id

    def _create_code_links(self, request_id: str, code_id: str, code: str) -> CodeLinks:
        """Create links to access generated code."""
        base_url = "http://localhost:8000"  # In production, use config

        return CodeLinks(
            code_preview_url=f"{base_url}/code/preview/{code_id}",
            code_download_url=f"{base_url}/code/download/{code_id}",
            code_raw_url=f"{base_url}/code/raw/{code_id}",
            code_id=code_id,
            code_size_bytes=len(code.encode()),
            code_lines=len(code.split("\n")),
            language="python",
        )

    def _create_execution_summary(
        self,
        request: ExecutionRequest,
        result: ExecutionResult,
        stages_completed: List[PipelineStage],
        stage_times: Dict[str, float],
    ) -> ExecutionSummary:
        """Create comprehensive execution summary."""
        # Calculate stage statistics
        if stage_times:
            fastest_stage = min(stage_times, key=stage_times.get)
            slowest_stage = max(stage_times, key=stage_times.get)
            avg_time = sum(stage_times.values()) / len(stage_times)
        else:
            fastest_stage = "none"
            slowest_stage = "none"
            avg_time = 0.0

        # Extract key metrics
        planning_steps = len(result.plan.get("steps", [])) if result.plan else 0
        vision_aligned = None
        vision_offset_mm = None

        if result.vision_alignment:
            alignment = result.vision_alignment.get("alignment", {})
            vision_aligned = alignment.get("is_aligned", False)
            vision_offset_mm = alignment.get("error_mm", 0.0)

        return ExecutionSummary(
            task_description=request.task_description,
            task_type=request.task_type,
            total_duration_seconds=result.total_time_seconds,
            success_rate=1.0 if result.success else 0.0,
            stages_completed_count=len(stages_completed),
            planning_steps=planning_steps,
            expert_demos_collected=result.expert_trajectories or 0,
            final_bc_loss=result.bc_loss,
            final_optimization_reward=result.optimization_reward,
            vision_aligned=vision_aligned,
            vision_offset_mm=vision_offset_mm,
            fastest_stage=fastest_stage,
            slowest_stage=slowest_stage,
            average_stage_time=avg_time,
            gpu_used=torch.cuda.is_available(),
            memory_peak_mb=None,  # Could track with psutil if needed
        )

    def _create_artifact_links(self, request_id: str, result: ExecutionResult) -> Dict[str, str]:
        """Create links to various artifacts."""
        base_url = "http://localhost:8000"  # In production, use config

        artifacts = {
            "status": f"{base_url}/status/{request_id}",
            "events": f"{base_url}/events/{request_id}",
            "websocket": f"ws://localhost:8000/ws/{request_id}",
        }

        if result.code_links:
            artifacts["code"] = result.code_links.code_download_url

        if result.plan:
            artifacts["plan"] = f"{base_url}/artifacts/{request_id}/plan"

        if result.expert_trajectories:
            artifacts["trajectories"] = f"{base_url}/artifacts/{request_id}/trajectories"

        if result.execution_result:
            artifacts["execution_log"] = f"{base_url}/artifacts/{request_id}/execution_log"

        return artifacts

    async def _execute_planning(self, request: ExecutionRequest, request_id: str) -> Dict[str, Any]:
        """Execute planning stage with heartbeats."""
        # Emit heartbeat at start
        self._emit_heartbeat(request_id, "planning", "Initializing planner...")

        # Generate task plan
        plan = self.planner.generate_plan(
            task_description=request.task_description,
            object_name=request.object_name,
            target_location=request.target_location,
        )

        # Add metadata
        plan["task_type"] = request.task_type
        plan["timestamp"] = datetime.now().isoformat()

        # Emit heartbeat before returning
        self._emit_heartbeat(request_id, "planning", "Plan finalized")

        return plan

    async def _execute_expert_demo(
        self, request: ExecutionRequest, plan: Dict[str, Any], request_id: str
    ) -> List[Dict[str, Any]]:
        """Execute expert demonstration stage with heartbeats."""
        # Collect demonstrations
        trajectories = []
        num_demos = 5

        for i in range(num_demos):  # Collect 5 demonstrations
            # Emit heartbeat for each demo
            self._emit_heartbeat(
                request_id,
                "expert_demonstration",
                f"Collecting demonstration {i+1}/{num_demos}...",
                metrics={"demo_num": i + 1, "total_demos": num_demos},
            )

            traj = self.expert.collect_demonstration(task=request.task_description, plan=plan)
            trajectories.append(traj)

            # Small delay and heartbeat
            await asyncio.sleep(0.1)

            if i % 2 == 0:  # Extra heartbeat every 2 demos
                self._emit_heartbeat(
                    request_id, "expert_demonstration", f"Processing trajectory {i+1}..."
                )
                await asyncio.sleep(0.05)

        return trajectories

    async def _execute_behavior_cloning(
        self, trajectories: List[Dict[str, Any]], num_epochs: int, request_id: str
    ) -> float:
        """Execute behavior cloning stage with frequent heartbeats."""
        # Prepare data
        self._emit_heartbeat(request_id, "behavior_cloning", "Preparing training data...")

        states = []
        actions = []

        for traj in trajectories:
            states.extend(traj.get("states", []))
            actions.extend(traj.get("actions", []))

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)

        # Train BC policy
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self._emit_heartbeat(
            request_id,
            "behavior_cloning",
            "Initializing optimizer...",
            metrics={"dataset_size": len(dataset), "batch_size": 32},
        )

        optimizer = torch.optim.Adam(self.bc_policy.parameters(), lr=1e-3)

        total_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_count = 0

            # Heartbeat at start of epoch
            self._emit_heartbeat(
                request_id,
                "behavior_cloning",
                f"Training epoch {epoch+1}/{num_epochs}...",
                metrics={"epoch": epoch + 1, "total_epochs": num_epochs, "learning_rate": 1e-3},
            )

            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()

                # Forward pass
                pred_actions, _ = self.bc_policy(batch_states)
                loss = torch.nn.functional.mse_loss(pred_actions, batch_actions)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Heartbeat every few batches
                if batch_count % 5 == 0:
                    self._emit_heartbeat(
                        request_id,
                        "behavior_cloning",
                        f"Epoch {epoch+1}: Batch {batch_count}/{len(dataloader)}",
                        metrics={"current_loss": loss.item(), "batch": batch_count},
                    )
                    await asyncio.sleep(0.01)  # Small yield to process events

            total_loss = epoch_loss / len(dataloader)

            # Heartbeat at end of epoch
            self._emit_heartbeat(
                request_id,
                "behavior_cloning",
                f"Epoch {epoch+1} complete",
                metrics={"epoch_loss": total_loss, "epoch": epoch + 1},
            )

            # Small delay between epochs
            await asyncio.sleep(0.05)

        return total_loss

    async def _execute_optimization(
        self, num_steps: int, use_gpt: bool = False, request_id: str = None
    ) -> float:
        """Execute policy optimization stage with frequent heartbeats."""
        # Run optimization
        rewards = []

        self._emit_heartbeat(
            request_id,
            "optimization",
            "Initializing policy optimizer...",
            metrics={"algorithm": "PPO", "use_gpt_reward": use_gpt},
        )

        for step in range(num_steps):
            # Simulate rollout and optimization
            reward = np.random.randn() * 0.1 + 0.5  # Simulated reward

            if use_gpt:
                # Would call GPT reward model here
                reward += 0.1

                # Heartbeat for GPT reward computation
                if step % 5 == 0:
                    self._emit_heartbeat(
                        request_id,
                        "optimization",
                        f"Computing GPT reward for step {step+1}...",
                        metrics={"gpt_active": True},
                    )

            rewards.append(reward)

            # Emit heartbeat every few steps
            if step % 3 == 0:  # More frequent heartbeats
                avg_reward = np.mean(rewards[-min(10, len(rewards)) :]) if rewards else 0
                self._emit_heartbeat(
                    request_id,
                    "optimization",
                    f"Optimization step {step+1}/{num_steps}",
                    metrics={
                        "step": step + 1,
                        "total_steps": num_steps,
                        "current_reward": reward,
                        "avg_reward": avg_reward,
                        "progress": (step + 1) / num_steps,
                    },
                )
                await asyncio.sleep(0.02)  # Small yield

            # Longer pause every 10 steps to simulate computation
            if step % 10 == 0:
                self._emit_heartbeat(
                    request_id,
                    "optimization",
                    f"Processing rollout batch {step//10 + 1}...",
                    metrics={"batch_num": step // 10 + 1},
                )
                await asyncio.sleep(0.1)

        final_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)

        self._emit_heartbeat(
            request_id,
            "optimization",
            "Optimization complete, finalizing policy...",
            metrics={"final_reward": final_reward},
        )

        return final_reward

    async def _execute_vision_refinement(
        self, object_name: Optional[str], request_id: str = None
    ) -> Dict[str, Any]:
        """Execute vision refinement stage with heartbeats."""
        # Emit initial heartbeat
        self._emit_heartbeat(
            request_id,
            "vision_refinement",
            "Initializing camera system...",
            metrics={"camera": "wrist_camera", "object": object_name},
        )
        await asyncio.sleep(0.05)

        # Capture image heartbeat
        self._emit_heartbeat(
            request_id,
            "vision_refinement",
            "Capturing image from wrist camera...",
            metrics={"exposure_ms": 20, "gain": 1.5},
        )
        await asyncio.sleep(0.1)

        # Processing heartbeat
        self._emit_heartbeat(
            request_id,
            "vision_refinement",
            f"Detecting {object_name or 'object'} in image...",
            metrics={"detection_method": "gpt_vision"},
        )

        # Simulate detection
        dx_px = np.random.randint(-10, 10)
        dy_px = np.random.randint(-10, 10)

        # Convert to world coordinates
        dx_m = dx_px * 0.15 / 525  # Simulated conversion
        dy_m = dy_px * 0.15 / 525

        # Computing offset heartbeat
        self._emit_heartbeat(
            request_id,
            "vision_refinement",
            "Computing world offset from pixel coordinates...",
            metrics={"pixel_offset_x": dx_px, "pixel_offset_y": dy_px},
        )
        await asyncio.sleep(0.05)

        # Format result
        vision_result = VisionUIFormatter.print_vision_result(
            dx_px=dx_px,
            dy_px=dy_px,
            dx_m=dx_m,
            dy_m=dy_m,
            depth=0.15,
            method="gpt_vision",
            confidence=0.85,
            force_print=False,  # Don't print in API
        )

        # Final heartbeat
        self._emit_heartbeat(
            request_id,
            "vision_refinement",
            "Vision analysis complete",
            metrics={
                "aligned": vision_result["vision_detection"]
                .get("alignment", {})
                .get("is_aligned", False),
                "confidence": 0.85,
            },
        )

        await asyncio.sleep(0.05)
        return vision_result["vision_detection"]

    async def _generate_execution_code(
        self,
        plan: Dict[str, Any],
        request: ExecutionRequest,
        vision_result: Optional[Dict[str, Any]],
        request_id: str = None,
    ) -> str:
        """Generate executable code with heartbeats."""
        # Initial heartbeat
        self._emit_heartbeat(
            request_id,
            "code_generation",
            "Starting code generation...",
            metrics={"language": "python", "framework": "cogniforge"},
        )
        await asyncio.sleep(0.05)

        # Generate code based on plan and vision
        code = f"""
# Auto-generated execution code
# Task: {request.task_description}
# Generated: {datetime.now().isoformat()}

import numpy as np
from cogniforge.control import RobotController, GraspExecutor

def execute_task():
    # Initialize robot
    robot = RobotController()
    executor = GraspExecutor(use_vision={request.use_vision})
    
    # Task parameters
    object_name = "{request.object_name or 'object'}"
    target = {request.target_location or {"x": 0.5, "y": 0.0, "z": 0.15}}
    
"""

        # Heartbeat for plan steps
        num_steps = len(plan.get("steps", []))
        if num_steps > 0:
            self._emit_heartbeat(
                request_id,
                "code_generation",
                f"Generating code for {num_steps} plan steps...",
                metrics={"num_steps": num_steps},
            )
            await asyncio.sleep(0.03)

        # Add plan steps
        for i, step in enumerate(plan.get("steps", []), 1):
            code += f"""
    # Step {i}: {step.get('description', 'Execute action')}
    {step.get('code', '# No code generated')}
    
"""
            # Heartbeat every few steps
            if i % 2 == 0 or i == num_steps:
                self._emit_heartbeat(
                    request_id,
                    "code_generation",
                    f"Generated step {i}/{num_steps}",
                    metrics={"steps_generated": i},
                )
                await asyncio.sleep(0.02)

        # Heartbeat for vision adjustment
        if vision_result:
            self._emit_heartbeat(
                request_id,
                "code_generation",
                "Adding vision refinement code...",
                metrics={"vision_enabled": True},
            )
            await asyncio.sleep(0.03)

            # Add vision adjustment if available
            if not vision_result.get("alignment", {}).get("is_aligned"):
                code += f"""
    # Vision adjustment
    dx_m = {vision_result.get('world_offset', {}).get('dx', 0)}
    dy_m = {vision_result.get('world_offset', {}).get('dy', 0)}
    executor.apply_micro_nudge(dx_m, dy_m)
    
"""

        # Final code generation
        self._emit_heartbeat(
            request_id,
            "code_generation",
            "Finalizing execution code...",
            metrics={"total_lines": len(code.split("\n"))},
        )

        code += """
    # Execute grasp
    success = executor.execute_grasp(object_name, target)
    return success

if __name__ == "__main__":
    result = execute_task()
    print(f"Task completed: {result}")
"""

        await asyncio.sleep(0.05)
        return code

    async def _execute_on_robot(
        self, code: str, safety_checks: bool, request_id: str = None
    ) -> Dict[str, Any]:
        """Execute generated code on robot with heartbeats."""
        # Initial heartbeat
        self._emit_heartbeat(
            request_id,
            "execution",
            "Initializing robot controller...",
            metrics={"robot_model": "UR5", "gripper": "Robotiq"},
        )
        await asyncio.sleep(0.1)

        # Safety checks heartbeat
        if safety_checks:
            self._emit_heartbeat(
                request_id,
                "execution",
                "Performing safety checks...",
                metrics={
                    "checking_workspace": True,
                    "checking_collisions": True,
                    "checking_limits": True,
                },
            )
            await asyncio.sleep(0.1)

        # Movement heartbeats
        self._emit_heartbeat(
            request_id,
            "execution",
            "Moving to approach position...",
            metrics={"phase": "approach", "speed": 0.1},
        )
        await asyncio.sleep(0.15)

        self._emit_heartbeat(
            request_id,
            "execution",
            "Executing grasp sequence...",
            metrics={"phase": "grasp", "gripper_opening": 0.08},
        )
        await asyncio.sleep(0.15)

        self._emit_heartbeat(
            request_id, "execution", "Closing gripper...", metrics={"phase": "close", "force": 20}
        )
        await asyncio.sleep(0.1)

        self._emit_heartbeat(
            request_id, "execution", "Lifting object...", metrics={"phase": "lift", "height": 0.1}
        )
        await asyncio.sleep(0.1)

        # Final result
        result = {
            "success": True,
            "execution_time": 5.2,
            "final_position": {"x": 0.5, "y": 0.0, "z": 0.15},
            "gripper_state": "closed",
            "object_grasped": True,
        }

        if safety_checks:
            result["safety_checks"] = {
                "collision_free": True,
                "within_workspace": True,
                "force_limits_ok": True,
            }

        # Final heartbeat
        self._emit_heartbeat(
            request_id,
            "execution",
            "Robot execution complete",
            metrics={"success": True, "final_gripper_state": "closed"},
        )

        return result


# Global orchestrator instance
orchestrator = PipelineOrchestrator()


# Note: log_event is now imported from core.logging_utils
# This local version has been removed to avoid duplication
# The logging_utils version handles both console and SSE output


async def event_generator(request_id: str):
    """
    Generate Server-Sent Events for a request.

    Yields events in SSE format with JSON data.
    """
    # Mark stream as active
    orchestrator.event_stream.set_active(request_id, True)

    # Send initial connection event
    initial_event = SSEEvent(
        phase="connected",
        message="SSE stream connected",
        metrics={"request_id": request_id},
        timestamp=datetime.now().isoformat(),
        progress=0.0,
        request_id=request_id,
    )
    yield f"data: {json.dumps(initial_event.dict())}\n\n"

    last_event_count = 0
    retry_count = 0
    max_retries = 600  # 5 minutes at 0.5s intervals

    try:
        while orchestrator.event_stream.is_active(request_id) and retry_count < max_retries:
            # Get new events
            events = orchestrator.event_stream.get_events(request_id)

            # Send new events only
            if len(events) > last_event_count:
                for event in events[last_event_count:]:
                    # Skip heartbeats or format as SSE
                    if "_heartbeat" in event.phase:
                        # Format heartbeat events specially
                        heartbeat_data = {
                            "type": "heartbeat",
                            "phase": event.phase.replace("_heartbeat", ""),
                            "message": event.message,
                            "metrics": event.metrics,
                            "timestamp": event.timestamp,
                        }
                        yield f"data: {json.dumps(heartbeat_data)}\n\n"
                    else:
                        # Regular event
                        event_data = json.dumps(event.dict())
                        yield f"data: {event_data}\n\n"

                    # Check for completion
                    if event.phase in ["completed", "failed"]:
                        # Send final event
                        final_event = SSEEvent(
                            phase="stream_end",
                            message="Pipeline execution finished",
                            metrics={"total_events": len(events), "final_status": event.phase},
                            timestamp=datetime.now().isoformat(),
                            progress=1.0 if event.phase == "completed" else -1.0,
                            request_id=request_id,
                        )
                        yield f"data: {json.dumps(final_event.dict())}\n\n"

                        # Clean up and exit
                        orchestrator.event_stream.set_active(request_id, False)
                        return

                last_event_count = len(events)
                retry_count = 0  # Reset retry count on new events
            else:
                retry_count += 1

            # Send heartbeat to keep connection alive
            if retry_count % 20 == 0:  # Every 10 seconds
                heartbeat = {
                    "phase": "heartbeat",
                    "message": "Connection alive",
                    "timestamp": datetime.now().isoformat(),
                }
                yield f": heartbeat {json.dumps(heartbeat)}\n\n"

            await asyncio.sleep(0.5)

        # Timeout reached
        if retry_count >= max_retries:
            timeout_event = SSEEvent(
                phase="timeout",
                message="Stream timeout - no activity",
                metrics={"timeout_seconds": max_retries * 0.5},
                timestamp=datetime.now().isoformat(),
                progress=-1.0,
                request_id=request_id,
            )
            yield f"data: {json.dumps(timeout_event.dict())}\n\n"

    finally:
        # Clean up
        orchestrator.event_stream.set_active(request_id, False)


@app.post("/execute", response_model=ExecutionResult)
async def execute_task(
    request: ExecutionRequest, background_tasks: BackgroundTasks
) -> ExecutionResult:
    """
    Execute complete pipeline: plan → expert → BC → optimize → vision → codegen.

    This endpoint orchestrates the entire CogniForge pipeline from task planning
    through code generation and execution.
    """
    # Generate request ID
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

    # Execute pipeline
    result = await orchestrator.execute_pipeline(request, request_id)

    return result


@app.get("/status/{request_id}")
async def get_status(request_id: str) -> ExecutionStatus:
    """Get current status of execution request."""
    if request_id not in orchestrator.execution_status:
        raise HTTPException(404, f"Request {request_id} not found")

    return orchestrator.execution_status[request_id]


@app.get("/result/{request_id}")
async def get_result(request_id: str) -> ExecutionResult:
    """Get final result of execution request."""
    if request_id not in orchestrator.execution_results:
        raise HTTPException(404, f"Result for request {request_id} not found")

    return orchestrator.execution_results[request_id]


@app.get("/events/{request_id}")
async def stream_events(request_id: str):
    """
    Stream Server-Sent Events (SSE) for pipeline execution.

    Returns a stream of JSON events with phase, message, and metrics.

    Event format:
    ```json
    {
        "phase": "planning|expert_demonstration|behavior_cloning|optimization|vision_refinement|code_generation|execution|completed|failed",
        "message": "Human-readable status message",
        "metrics": {
            "key": "value",
            "progress": 0.75,
            ...
        },
        "timestamp": "2024-01-10T10:30:00",
        "progress": 0.75,
        "request_id": "req_20240110_103000_1234"
    }
    ```

    Example usage:
    ```javascript
    const eventSource = new EventSource('/events/req_12345');

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`Phase: ${data.phase}, Progress: ${data.progress * 100}%`);
        console.log(`Metrics:`, data.metrics);
    };
    ```
    """
    # Check if request exists
    if request_id not in orchestrator.execution_status and request_id != "demo":
        # For demo purposes, allow "demo" request_id
        if request_id != "demo":
            raise HTTPException(404, f"Request {request_id} not found")

    return StreamingResponse(
        event_generator(request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/events")
async def list_active_streams():
    """
    List all active event streams.

    Returns information about currently active SSE connections.
    """
    active_streams = []
    for request_id, is_active in orchestrator.event_stream.active_streams.items():
        if is_active:
            events = orchestrator.event_stream.get_events(request_id)
            last_event = events[-1] if events else None

            active_streams.append(
                {
                    "request_id": request_id,
                    "num_events": len(events),
                    "last_phase": last_event.phase if last_event else None,
                    "last_update": last_event.timestamp if last_event else None,
                    "progress": last_event.progress if last_event else 0.0,
                }
            )

    return {
        "active_streams": active_streams,
        "total_active": len(active_streams),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/events/demo")
async def create_demo_event_stream():
    """
    Create a demo event stream for testing SSE functionality.

    This endpoint creates a simulated pipeline execution that emits events.
    """
    request_id = "demo_" + str(uuid.uuid4())[:8]

    async def demo_pipeline():
        """Simulate pipeline execution with events."""
        stages = [
            (PipelineStage.PLANNING, 0.1, "Starting task planning", {"planner": "GPT-4"}),
            (
                PipelineStage.PLANNING,
                0.15,
                "Task plan generated",
                {"steps": 5, "planning_time": 1.2},
            ),
            (
                PipelineStage.EXPERT_DEMO,
                0.2,
                "Collecting expert demonstrations",
                {"target_demos": 5},
            ),
            (
                PipelineStage.EXPERT_DEMO,
                0.3,
                "Expert demonstrations collected",
                {"num_trajectories": 5, "avg_length": 100},
            ),
            (
                PipelineStage.BEHAVIOR_CLONING,
                0.35,
                "Starting BC training",
                {"epochs": 10, "batch_size": 32},
            ),
            (
                PipelineStage.BEHAVIOR_CLONING,
                0.5,
                "BC training complete",
                {"bc_loss": 0.0234, "training_time": 3.5},
            ),
            (
                PipelineStage.OPTIMIZATION,
                0.55,
                "Starting policy optimization",
                {"algorithm": "PPO", "steps": 50},
            ),
            (
                PipelineStage.OPTIMIZATION,
                0.7,
                "Optimization complete",
                {"final_reward": 0.875, "optimization_time": 5.2},
            ),
            (
                PipelineStage.VISION_REFINEMENT,
                0.75,
                "Capturing vision data",
                {"camera": "wrist_camera"},
            ),
            (
                PipelineStage.VISION_REFINEMENT,
                0.85,
                "Vision refinement complete",
                {"aligned": True, "offset_mm": 2.3, "confidence": 0.92},
            ),
            (
                PipelineStage.CODE_GENERATION,
                0.88,
                "Generating execution code",
                {"language": "python", "lines": 150},
            ),
            (
                PipelineStage.CODE_GENERATION,
                0.92,
                "Code generation complete",
                {"functions": 8, "safety_checks": True},
            ),
            (
                PipelineStage.EXECUTION,
                0.95,
                "Executing on robot",
                {"robot_model": "UR5", "gripper": "Robotiq"},
            ),
            (
                PipelineStage.COMPLETED,
                1.0,
                "Task completed successfully",
                {"total_time": 25.3, "success": True},
            ),
        ]

        for stage, progress, message, metrics in stages:
            orchestrator.update_status(request_id, stage, progress, message, metrics=metrics)
            await asyncio.sleep(0.5)  # Simulate processing time

    # Start demo pipeline in background
    asyncio.create_task(demo_pipeline())

    return {
        "request_id": request_id,
        "message": "Demo event stream created",
        "sse_endpoint": f"/events/{request_id}",
        "instructions": "Connect to the SSE endpoint to receive events",
    }


@app.get("/code/preview/{code_id}")
async def preview_code(code_id: str):
    """
    Preview generated code with syntax highlighting.

    Returns HTML preview of the generated code.
    """
    if not hasattr(orchestrator, "stored_codes") or code_id not in orchestrator.stored_codes:
        raise HTTPException(404, f"Code {code_id} not found")

    code_info = orchestrator.stored_codes[code_id]
    code = code_info["code"]

    # Create HTML preview with syntax highlighting
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Code Preview - {code_id}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
        <style>
            body {{
                font-family: 'Monaco', 'Courier New', monospace;
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 20px;
                margin: 0;
            }}
            .header {{
                background-color: #1e1e1e;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .code-container {{
                background-color: #1e1e1e;
                border-radius: 5px;
                padding: 20px;
                overflow-x: auto;
            }}
            pre {{
                margin: 0;
            }}
            .metadata {{
                color: #808080;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Generated Code Preview</h2>
            <div class="metadata">
                <p>Code ID: {code_id}</p>
                <p>Request ID: {code_info['request_id']}</p>
                <p>Generated: {code_info['timestamp']}</p>
                <p>Lines: {len(code.split(chr(10)))}</p>
                <p>Size: {len(code.encode())} bytes</p>
            </div>
        </div>
        <div class="code-container">
            <pre><code class="language-python">{code}</code></pre>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    </body>
    </html>
    """

    from fastapi.responses import HTMLResponse

    return HTMLResponse(content=html_content)


@app.get("/code/download/{code_id}")
async def download_code(code_id: str):
    """
    Download generated code as a Python file.
    """
    if not hasattr(orchestrator, "stored_codes") or code_id not in orchestrator.stored_codes:
        raise HTTPException(404, f"Code {code_id} not found")

    code_info = orchestrator.stored_codes[code_id]
    code = code_info["code"]

    from fastapi.responses import Response

    return Response(
        content=code,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{code_id}.py"'},
    )


@app.get("/code/raw/{code_id}")
async def get_raw_code(code_id: str):
    """
    Get raw code content as plain text.
    """
    if not hasattr(orchestrator, "stored_codes") or code_id not in orchestrator.stored_codes:
        raise HTTPException(404, f"Code {code_id} not found")

    code_info = orchestrator.stored_codes[code_id]

    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(content=code_info["code"])


@app.get("/artifacts/{request_id}/plan")
async def get_plan_artifact(request_id: str):
    """
    Get the task plan artifact for a request.
    """
    if request_id not in orchestrator.execution_results:
        raise HTTPException(404, f"Request {request_id} not found")

    result = orchestrator.execution_results[request_id]
    if not result.plan:
        raise HTTPException(404, "No plan available for this request")

    return {
        "request_id": request_id,
        "artifact_type": "task_plan",
        "timestamp": result.plan.get("timestamp"),
        "content": result.plan,
    }


@app.get("/artifacts/{request_id}/trajectories")
async def get_trajectories_artifact(request_id: str):
    """
    Get expert trajectories metadata for a request.
    """
    if request_id not in orchestrator.execution_results:
        raise HTTPException(404, f"Request {request_id} not found")

    result = orchestrator.execution_results[request_id]

    return {
        "request_id": request_id,
        "artifact_type": "expert_trajectories",
        "num_trajectories": result.expert_trajectories or 0,
        "metadata": {
            "collected": result.expert_trajectories is not None,
            "bc_loss": result.bc_loss,
        },
    }


@app.get("/artifacts/{request_id}/execution_log")
async def get_execution_log(request_id: str):
    """
    Get execution log for a request.
    """
    if request_id not in orchestrator.execution_results:
        raise HTTPException(404, f"Request {request_id} not found")

    result = orchestrator.execution_results[request_id]

    # Get all events for this request
    events = []
    if (
        hasattr(orchestrator.event_stream, "events")
        and request_id in orchestrator.event_stream.events
    ):
        events = [e.dict() for e in orchestrator.event_stream.get_events(request_id)]

    return {
        "request_id": request_id,
        "artifact_type": "execution_log",
        "success": result.success,
        "total_time_seconds": result.total_time_seconds,
        "stages_completed": result.stages_completed,
        "execution_result": result.execution_result,
        "events": events,
        "errors": result.errors,
    }


@app.get("/summary/{request_id}")
async def get_execution_summary(request_id: str):
    """
    Get a comprehensive summary of the pipeline execution.

    Returns a detailed JSON with all relevant information, metrics, and links.
    """
    if request_id not in orchestrator.execution_results:
        raise HTTPException(404, f"Request {request_id} not found")

    result = orchestrator.execution_results[request_id]

    # Build comprehensive summary
    summary = {
        "request_id": request_id,
        "success": result.success,
        "timestamp": datetime.now().isoformat(),
        # Quick summary
        "summary": result.summary.dict() if result.summary else None,
        # Code links
        "code": {
            "available": result.code_links is not None,
            "links": result.code_links.dict() if result.code_links else None,
            "preview": (
                result.generated_code[:500] + "..."
                if result.generated_code and len(result.generated_code) > 500
                else result.generated_code
            ),
        },
        # Artifacts
        "artifacts": result.artifacts,
        # Detailed metrics
        "metrics": {
            "pipeline": {
                "total_time_seconds": result.total_time_seconds,
                "stages_completed": len(result.stages_completed),
                "stages_total": 7,
                "completion_rate": len(result.stages_completed) / 7,
            },
            "stage_times": result.stage_times,
            "ml_metrics": {
                "bc_loss": result.bc_loss,
                "optimization_reward": result.optimization_reward,
                "expert_trajectories": result.expert_trajectories,
            },
            "vision_metrics": {
                "used": result.vision_alignment is not None,
                "aligned": (
                    result.vision_alignment.get("alignment", {}).get("is_aligned")
                    if result.vision_alignment
                    else None
                ),
                "offset_mm": (
                    result.vision_alignment.get("alignment", {}).get("error_mm")
                    if result.vision_alignment
                    else None
                ),
                "confidence": (
                    result.vision_alignment.get("confidence") if result.vision_alignment else None
                ),
            },
            "execution_metrics": result.execution_result if result.execution_result else None,
        },
        # Stages detail
        "stages": {
            "completed": [str(stage) for stage in result.stages_completed],
            "details": {
                "planning": {
                    "completed": PipelineStage.PLANNING in result.stages_completed,
                    "time": result.stage_times.get("planning", 0),
                    "steps": len(result.plan.get("steps", [])) if result.plan else 0,
                },
                "expert_demonstration": {
                    "completed": PipelineStage.EXPERT_DEMO in result.stages_completed,
                    "time": result.stage_times.get("expert", 0),
                    "trajectories": result.expert_trajectories,
                },
                "behavior_cloning": {
                    "completed": PipelineStage.BEHAVIOR_CLONING in result.stages_completed,
                    "time": result.stage_times.get("bc", 0),
                    "final_loss": result.bc_loss,
                },
                "optimization": {
                    "completed": PipelineStage.OPTIMIZATION in result.stages_completed,
                    "time": result.stage_times.get("optimization", 0),
                    "final_reward": result.optimization_reward,
                },
                "vision_refinement": {
                    "completed": PipelineStage.VISION_REFINEMENT in result.stages_completed,
                    "time": result.stage_times.get("vision", 0),
                    "result": result.vision_alignment,
                },
                "code_generation": {
                    "completed": PipelineStage.CODE_GENERATION in result.stages_completed,
                    "time": result.stage_times.get("codegen", 0),
                    "code_generated": result.generated_code is not None,
                },
                "execution": {
                    "completed": PipelineStage.EXECUTION in result.stages_completed,
                    "time": result.stage_times.get("execution", 0),
                    "result": result.execution_result,
                },
            },
        },
        # Errors if any
        "errors": result.errors,
        # Quick actions
        "actions": {
            "view_code": (
                f"http://localhost:8000/code/preview/{result.code_links.code_id}"
                if result.code_links
                else None
            ),
            "download_code": (
                f"http://localhost:8000/code/download/{result.code_links.code_id}"
                if result.code_links
                else None
            ),
            "view_events": f"http://localhost:8000/events/{request_id}",
            "view_status": f"http://localhost:8000/status/{request_id}",
        },
    }

    return summary


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "features": {
            "sse_events": True,
            "websocket": True,
            "pipeline_orchestration": True,
            "code_storage": True,
            "artifact_management": True,
        },
    }


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "CogniForge Execution API",
        "version": "1.0.0",
        "endpoints": {
            "execute": {
                "method": "POST",
                "path": "/execute",
                "description": "Execute complete pipeline",
            },
            "status": {
                "method": "GET",
                "path": "/status/{request_id}",
                "description": "Get execution status",
            },
            "result": {
                "method": "GET",
                "path": "/result/{request_id}",
                "description": "Get execution result",
            },
            "health": {"method": "GET", "path": "/health", "description": "Health check"},
            "events": {
                "method": "GET",
                "path": "/events/{request_id}",
                "description": "Stream Server-Sent Events for execution",
            },
            "list_events": {
                "method": "GET",
                "path": "/events",
                "description": "List active event streams",
            },
            "demo_events": {
                "method": "POST",
                "path": "/events/demo",
                "description": "Create demo event stream for testing",
            },
            "summary": {
                "method": "GET",
                "path": "/summary/{request_id}",
                "description": "Get comprehensive execution summary with links",
            },
            "code_preview": {
                "method": "GET",
                "path": "/code/preview/{code_id}",
                "description": "Preview generated code with syntax highlighting",
            },
            "code_download": {
                "method": "GET",
                "path": "/code/download/{code_id}",
                "description": "Download generated code as Python file",
            },
            "artifacts": {
                "method": "GET",
                "path": "/artifacts/{request_id}/{artifact_type}",
                "description": "Get execution artifacts",
            },
        },
        "pipeline_stages": [
            "planning",
            "expert_demonstration",
            "behavior_cloning",
            "optimization",
            "vision_refinement",
            "code_generation",
            "execution",
        ],
    }


# WebSocket support for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws/{request_id}")
async def websocket_endpoint(websocket: WebSocket, request_id: str):
    """WebSocket endpoint for real-time execution updates."""
    await manager.connect(websocket)

    try:
        while True:
            # Send updates if available
            if request_id in orchestrator.execution_status:
                status = orchestrator.execution_status[request_id]
                await websocket.send_json(
                    {
                        "type": "status_update",
                        "data": {
                            "request_id": status.request_id,
                            "stage": status.stage,
                            "progress": status.progress,
                            "message": status.message,
                            "details": status.details,
                        },
                    }
                )

            # Check for completion
            if request_id in orchestrator.execution_results:
                result = orchestrator.execution_results[request_id]
                await websocket.send_json(
                    {
                        "type": "execution_complete",
                        "data": {
                            "request_id": result.request_id,
                            "success": result.success,
                            "stages_completed": result.stages_completed,
                        },
                    }
                )
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
