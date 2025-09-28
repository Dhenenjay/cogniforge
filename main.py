"""
CogniForge FastAPI Application

Main API application for serving CogniForge models and functionalities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import logging
from pathlib import Path
import sys

# Add cogniforge to path
sys.path.insert(0, str(Path(__file__).parent))

from cogniforge.core.seed_manager import SeedManager, SeedConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CogniForge API",
    description="AI-powered robotics and code generation API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class TrainingRequest(BaseModel):
    """Request model for training pipeline."""
    pipeline: str = Field(default="train", description="Pipeline type: train, eval, bc, etc.")
    task: Optional[str] = Field(None, description="Task name")
    epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    deterministic: bool = Field(default=False, description="Enable deterministic mode")
    device: str = Field(default="cuda", description="Device to use (cuda/cpu)")
    output_dir: Optional[str] = Field(None, description="Output directory for results")
    config_path: Optional[str] = Field(None, description="Path to configuration file")


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    description: str = Field(..., description="Natural language task description")
    framework: str = Field(default="pybullet", description="Target framework")
    model: str = Field(default="gpt-5", description="Language model to use")
    language: str = Field(default="python", description="Programming language")
    style: str = Field(default="modular", description="Code style: minimal, verbose, educational")
    include_tests: bool = Field(default=False, description="Include unit tests")
    include_docs: bool = Field(default=False, description="Include documentation")
    validate: bool = Field(default=True, description="Validate generated code")
    safety_check: bool = Field(default=True, description="Perform safety checks")
    max_length: int = Field(default=1000, description="Maximum code length in lines")


class DemoRequest(BaseModel):
    """Request model for running demos."""
    demo_type: str = Field(..., description="Demo type: grasp, navigation, manipulation, etc.")
    object_name: Optional[str] = Field(None, description="Object to interact with")
    robot: str = Field(default="franka", description="Robot model")
    environment: str = Field(default="tabletop", description="Environment setup")
    visualize: bool = Field(default=False, description="Enable visualization")
    interactive: bool = Field(default=False, description="Enable interactive mode")
    speed: float = Field(default=1.0, description="Simulation speed multiplier")


class TaskStatus(BaseModel):
    """Status of a running task."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============== In-memory task storage ==============
tasks: Dict[str, TaskStatus] = {}


# ============== Helper Functions ==============

async def run_training_pipeline(request: TrainingRequest, task_id: str):
    """Run training pipeline asynchronously."""
    try:
        tasks[task_id].status = "running"
        tasks[task_id].message = "Initializing training pipeline..."
        
        # Set up seed if deterministic mode is requested
        if request.deterministic or request.seed is not None:
            seed_config = SeedConfig(
                seed=request.seed or 42,
                enable_deterministic=request.deterministic,
                deterministic_cuda=request.device == "cuda" and request.deterministic
            )
            seed_manager = SeedManager()
            seed_manager.set_seed_from_config(seed_config)
            logger.info(f"Seeds set: {seed_manager.get_current_seeds()}")
        
        # Simulate training progress
        for epoch in range(request.epochs):
            await asyncio.sleep(0.1)  # Simulate training time
            progress = (epoch + 1) / request.epochs
            tasks[task_id].progress = progress
            tasks[task_id].message = f"Training epoch {epoch + 1}/{request.epochs}"
            
            if epoch % 10 == 0:
                logger.info(f"Task {task_id}: Epoch {epoch + 1}/{request.epochs}")
        
        # Complete task
        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].message = "Training completed successfully"
        tasks[task_id].result = {
            "final_loss": 0.0234,
            "final_accuracy": 0.956,
            "model_path": f"{request.output_dir or 'outputs'}/model_final.pth",
            "epochs_completed": request.epochs
        }
        
    except Exception as e:
        tasks[task_id].status = "failed"
        tasks[task_id].error = str(e)
        tasks[task_id].message = "Training failed"
        logger.error(f"Task {task_id} failed: {e}")


async def generate_code_async(request: CodeGenerationRequest, task_id: str):
    """Generate code asynchronously."""
    try:
        tasks[task_id].status = "running"
        tasks[task_id].message = "Generating code..."
        
        # Simulate code generation
        await asyncio.sleep(2)  # Simulate API call time
        
        # Generate mock code based on request
        generated_code = f"""#!/usr/bin/env python3
\"\"\"
Generated code for: {request.description}
Framework: {request.framework}
Style: {request.style}
\"\"\"

import {request.framework}
import numpy as np

class GeneratedTask:
    def __init__(self):
        self.description = "{request.description}"
        
    def execute(self):
        # Generated implementation
        print(f"Executing: {request.description}")
        # TODO: Implement task logic
        
    def validate(self):
        \"\"\"Validate the implementation.\"\"\"
        return True

if __name__ == "__main__":
    task = GeneratedTask()
    task.execute()
"""
        
        if request.include_tests:
            generated_code += """
# Unit tests
def test_task():
    task = GeneratedTask()
    assert task.validate()
    print("All tests passed!")
"""
        
        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].message = "Code generation completed"
        tasks[task_id].result = {
            "code": generated_code,
            "lines": len(generated_code.split("\n")),
            "validation_passed": request.validate,
            "safety_check_passed": request.safety_check
        }
        
    except Exception as e:
        tasks[task_id].status = "failed"
        tasks[task_id].error = str(e)
        tasks[task_id].message = "Code generation failed"
        logger.error(f"Task {task_id} failed: {e}")


# ============== API Routes ==============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to CogniForge API",
        "version": "0.1.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "training": "/api/v1/train",
            "code_generation": "/api/v1/generate",
            "demo": "/api/v1/demo"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CogniForge API",
        "version": "0.1.0"
    }


@app.post("/api/v1/train")
async def train(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a training pipeline."""
    import uuid
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Training task queued"
    )
    
    # Add training to background tasks
    background_tasks.add_task(run_training_pipeline, request, task_id)
    
    logger.info(f"Started training task: {task_id}")
    
    return {
        "task_id": task_id,
        "message": "Training task started",
        "status_url": f"/api/v1/task/{task_id}"
    }


@app.post("/api/v1/generate")
async def generate_code(request: CodeGenerationRequest, background_tasks: BackgroundTasks):
    """Generate code from natural language description."""
    import uuid
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Code generation task queued"
    )
    
    # Add code generation to background tasks
    background_tasks.add_task(generate_code_async, request, task_id)
    
    logger.info(f"Started code generation task: {task_id}")
    
    return {
        "task_id": task_id,
        "message": "Code generation started",
        "status_url": f"/api/v1/task/{task_id}"
    }


@app.post("/api/v1/demo")
async def run_demo(request: DemoRequest):
    """Run a demonstration."""
    logger.info(f"Running demo: {request.demo_type}")
    
    # Simulate demo execution
    result = {
        "demo_type": request.demo_type,
        "status": "completed",
        "message": f"Demo '{request.demo_type}' executed successfully",
        "visualization_enabled": request.visualize,
        "interactive_mode": request.interactive,
        "metrics": {
            "success_rate": 0.95,
            "execution_time": 12.34,
            "steps_completed": 42
        }
    }
    
    return result


@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a running task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]


@app.get("/api/v1/tasks")
async def list_tasks():
    """List all tasks."""
    return {
        "tasks": list(tasks.values()),
        "total": len(tasks)
    }


@app.delete("/api/v1/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task.status in ["completed", "failed"]:
        return {"message": f"Task {task_id} already {task.status}"}
    
    task.status = "cancelled"
    task.message = "Task cancelled by user"
    
    return {"message": f"Task {task_id} cancelled"}


@app.post("/api/v1/seed")
async def set_seed(seed: int, deterministic: bool = False):
    """Set global seed for reproducibility."""
    seed_config = SeedConfig(
        seed=seed,
        enable_deterministic=deterministic
    )
    
    seed_manager = SeedManager()
    seed_manager.set_seed_from_config(seed_config)
    current_seeds = seed_manager.get_current_seeds()
    
    logger.info(f"Seeds set: {current_seeds}")
    
    return {
        "message": "Seeds configured successfully",
        "seeds": current_seeds,
        "deterministic": deterministic
    }


# ============== WebSocket Support ==============

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/task/{task_id}")
async def websocket_task_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time task status updates."""
    await websocket.accept()
    
    try:
        while True:
            if task_id in tasks:
                task = tasks[task_id]
                await websocket.send_json(task.dict())
                
                if task.status in ["completed", "failed", "cancelled"]:
                    break
            else:
                await websocket.send_json({
                    "error": "Task not found",
                    "task_id": task_id
                })
                break
            
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")


# ============== Startup and Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("CogniForge API starting up...")
    logger.info(f"API documentation available at: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("CogniForge API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )