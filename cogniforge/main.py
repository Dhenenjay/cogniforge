"""
Main FastAPI application for CogniForge.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from cogniforge.core import settings, RobotSimulator, RobotType, SimulationMode
import os
import numpy as np

# Create FastAPI app using settings
app = FastAPI(
    title=settings.app_name,
    description="A Python project with FastAPI, PyBullet, and ML libraries",
    version=settings.app_version,
    debug=settings.debug,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        content={
            "message": "Welcome to CogniForge!",
            "version": "0.1.0",
            "status": "running"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "cogniforge"
        }
    )


@app.get("/info")
async def get_info():
    """Get information about installed libraries."""
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        torch_device = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    except ImportError:
        torch_version = "not installed"
        cuda_available = False
        torch_device = "cpu"
    except Exception as e:
        torch_version = f"error: {str(e)}"
        cuda_available = False
        torch_device = "cpu"
    
    import pybullet as pb
    import gymnasium as gym
    from PIL import Image
    import stable_baselines3 as sb3
    
    return JSONResponse(
        content={
            "libraries": {
                "numpy": np.__version__,
                "torch": torch_version,
                "pybullet": pb.__version__ if hasattr(pb, '__version__') else "3.2.5",
                "gymnasium": gym.__version__,
                "pillow": Image.__version__,
                "stable_baselines3": sb3.__version__,
            },
            "cuda_available": cuda_available,
            "torch_device": torch_device
        }
    )


@app.get("/config")
async def get_config():
    """Get current configuration (with sensitive data masked)."""
    return JSONResponse(
        content={
            "config": settings.to_dict(include_sensitive=False),
            "openai_configured": settings.validate_openai_config(),
            "debug_mode": settings.debug,
            "environment": "production" if not settings.debug else "development"
        }
    )


@app.get("/openai/test")
async def test_openai():
    """Test OpenAI API connection."""
    if not settings.validate_openai_config():
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file."
        )
    
    try:
        client = settings.get_openai_client()
        # Simple test - list available models
        response = client.models.list()
        model_ids = [model.id for model in response.data][:5]  # Get first 5 models
        
        return JSONResponse(
            content={
                "status": "connected",
                "message": "Successfully connected to OpenAI API",
                "sample_models": model_ids,
                "api_key_masked": f"{settings.openai_api_key[:8]}...{settings.openai_api_key[-4:]}"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to OpenAI API: {str(e)}"
        )


@app.get("/simulator/test")
async def test_simulator():
    """Test PyBullet simulator setup with blocks."""
    try:
        # Create simulator with DIRECT mode for API testing
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        
        # Connect to PyBullet
        sim.connect()
        
        # Load plane
        plane_id = sim.load_plane()
        
        # Load a KUKA robot
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="test_kuka"
        )
        
        # Spawn a static platform
        platform_id = sim.spawn_platform(
            color_rgb=(0.7, 0.7, 0.7),  # Light gray
            size=0.1,
            # Uses default position (0.6, 0.2, 0.05)
        )
        
        # Spawn a table-like platform
        table_id = sim.spawn_table(
            position=(0.3, 0.0, 0.0),
            table_height=0.25,
            table_size=0.15,
        )
        
        # Spawn colored blocks - some on platforms
        red_block_id = sim.spawn_block(
            color_rgb=(1.0, 0.0, 0.0),  # Red
            size=0.03,
            position=(0.6, 0.2, 0.15),  # Drop on platform
        )
        
        blue_block_id = sim.spawn_block(
            color_rgb=(0.0, 0.0, 1.0),  # Blue
            size=0.04,
            position=(0.3, 0.0, 0.35),  # Drop on table
            block_name="blue_cube",
        )
        
        green_block_id = sim.spawn_block(
            color_rgb=(0.0, 1.0, 0.0),  # Green
            size=0.025,
            position=(0.5, 0.0, 0.1),  # Drop on ground
            mass=0.05,
            block_name="green_cube",
        )
        
        # Get gripper info
        gripper_info = sim.get_gripper_info("test_kuka")
        
        # Get link names
        link_names = sim.get_link_names("test_kuka")
        
        # Get end-effector pose using the new ee_pose method
        ee_pos, ee_orn = sim.ee_pose("test_kuka")
        
        # Get simulation info
        sim_info = sim.get_simulation_info()
        
        # Get all IDs
        all_ids = sim.get_ids()
        
        # Disconnect
        sim.disconnect()
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "PyBullet simulator tested successfully with blocks",
                "simulation_info": sim_info,
                "plane_id": plane_id,
                "robot": {
                    "id": robot_info.robot_id,
                    "name": robot_info.name,
                    "type": robot_info.robot_type.value,
                    "num_joints": robot_info.num_joints,
                    "end_effector_index": robot_info.end_effector_index,
                    "tool_link_index": robot_info.tool_link_index,
                    "gripper_info": gripper_info,
                    "num_links": len(link_names),
                    "ee_pose": {
                        "position": list(ee_pos),
                        "orientation": list(ee_orn),
                    },
                },
                "platform": {
                    "id": platform_id,
                    "position": [0.6, 0.2, 0.05],
                    "size": 0.1,
                    "static": True,
                },
                "table": {
                    "id": table_id,
                    "position": [0.3, 0.0, 0.25],
                    "size": 0.15,
                    "height": 0.25,
                },
                "blocks": {
                    "red_block": {
                        "id": red_block_id,
                        "color": "red",
                        "position": [0.6, 0.2, 0.15],
                        "size": 0.03,
                        "on": "platform",
                    },
                    "blue_cube": {
                        "id": blue_block_id,
                        "color": "blue",
                        "position": [0.3, 0.0, 0.35],
                        "size": 0.04,
                        "on": "table",
                    },
                    "green_cube": {
                        "id": green_block_id,
                        "color": "green",
                        "position": [0.5, 0.0, 0.1],
                        "size": 0.025,
                        "on": "ground",
                    },
                },
                "all_ids": all_ids,
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulator test failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application using settings
    uvicorn.run(
        "cogniforge.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
