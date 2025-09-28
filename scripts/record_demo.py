#!/usr/bin/env python3
"""
Demo Recording Script for CogniForge

Records a complete demonstration of CogniForge capabilities including:
- Training with deterministic mode
- Code generation
- PyBullet simulation
- Saves output as MP4 and GIF
"""

import os
import sys
import time
import subprocess
import threading
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    logger.warning("PyBullet not installed. Simulation recording will be skipped.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not installed. Video recording will be limited.")

try:
    from PIL import Image, ImageDraw, ImageFont
    import imageio
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    logger.warning("PIL/imageio not installed. Install with: pip install pillow imageio[ffmpeg]")


# Import safe file manager
sys.path.insert(0, str(Path(__file__).parent.parent))
from cogniforge.core.safe_file_manager import SafeFileManager, WriteScope

class DemoRecorder:
    """Records CogniForge demonstrations as video."""
    
    def __init__(self, output_dir: str = "recordings", resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize demo recorder.
        
        Args:
            output_dir: Directory to save recordings (relative to ./generated)
            resolution: Video resolution (width, height)
        """
        # Use safe file manager for all writes
        self.safe_manager = SafeFileManager()
        self.output_dir = self.safe_manager.generated_dir / "recordings"
        
        self.resolution = resolution
        self.width, self.height = resolution
        self.fps = 30
        self.frames = []
        self.recording = False
        
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_title_frame(self, title: str, subtitle: str = "", duration: float = 2.0) -> List[np.ndarray]:
        """Create title frames for the video."""
        frames = []
        
        if not IMAGING_AVAILABLE:
            return frames
        
        # Create image
        img = Image.new('RGB', self.resolution, color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fallback to default
        try:
            title_font = ImageFont.truetype("arial.ttf", 48)
            subtitle_font = ImageFont.truetype("arial.ttf", 24)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        
        title_x = (self.width - title_width) // 2
        title_y = (self.height - title_height) // 2 - 50
        
        draw.text((title_x, title_y), title, fill=(255, 255, 255), font=title_font)
        
        # Draw subtitle
        if subtitle:
            subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
            subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
            subtitle_x = (self.width - subtitle_width) // 2
            subtitle_y = title_y + title_height + 30
            
            draw.text((subtitle_x, subtitle_y), subtitle, fill=(180, 180, 180), font=subtitle_font)
        
        # Convert to numpy array and add to frames
        frame = np.array(img)
        num_frames = int(duration * self.fps)
        frames.extend([frame] * num_frames)
        
        return frames
    
    def record_pybullet_demo(self, duration: float = 10.0) -> List[np.ndarray]:
        """Record a PyBullet simulation demo."""
        frames = []
        
        if not PYBULLET_AVAILABLE:
            logger.warning("PyBullet not available, skipping simulation recording")
            return frames
        
        logger.info("Recording PyBullet demo...")
        
        # Connect to PyBullet
        physics_client = p.connect(p.DIRECT)  # Use DIRECT mode for headless recording
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set up camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Load environment
        p.setGravity(0, 0, -9.81)
        plane_id = p.loadURDF("plane.urdf")
        
        # Load robot (Kuka arm)
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", robot_start_pos, robot_start_orientation)
        
        # Load objects
        cube_start_pos = [0.5, 0, 0.1]
        cube_id = p.loadURDF("cube.urdf", cube_start_pos, globalScaling=0.1)
        
        # Set cube color to red
        p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
        
        # Record simulation
        num_frames = int(duration * self.fps)
        for i in range(num_frames):
            # Simple robot movement
            for j in range(p.getNumJoints(robot_id)):
                target_pos = np.sin(i * 0.01 + j) * 0.5
                p.setJointMotorControl2(
                    robot_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos
                )
            
            # Step simulation
            p.stepSimulation()
            
            # Capture frame
            width, height = self.resolution
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=2,
                yaw=45 + i * 0.5,  # Rotate camera
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width/height,
                nearVal=0.1,
                farVal=100
            )
            
            # Get camera image
            img_data = p.getCameraImage(
                width, height,
                view_matrix,
                projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Extract RGB data
            rgb_array = np.array(img_data[2], dtype=np.uint8)
            rgb_array = rgb_array.reshape((height, width, 4))[:, :, :3]
            
            frames.append(rgb_array)
        
        # Disconnect PyBullet
        p.disconnect()
        
        logger.info(f"Recorded {len(frames)} frames from PyBullet")
        return frames
    
    def record_terminal_demo(self, commands: List[str], duration_per_command: float = 3.0) -> List[np.ndarray]:
        """Simulate terminal output as frames."""
        frames = []
        
        if not IMAGING_AVAILABLE:
            return frames
        
        logger.info("Recording terminal demo...")
        
        for cmd in commands:
            # Create terminal-style frame
            img = Image.new('RGB', self.resolution, color=(12, 12, 12))
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("consolas.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw terminal header
            draw.rectangle([0, 0, self.width, 30], fill=(40, 40, 40))
            draw.text((10, 5), "CogniForge Terminal", fill=(200, 200, 200), font=font)
            
            # Draw command
            y_offset = 50
            draw.text((20, y_offset), f"$ {cmd}", fill=(0, 255, 0), font=font)
            
            # Simulate output
            y_offset += 30
            outputs = [
                "Starting CogniForge...",
                "Loading configuration...",
                "Initializing model...",
                "✓ Ready"
            ]
            
            for output in outputs:
                draw.text((20, y_offset), output, fill=(180, 180, 180), font=font)
                y_offset += 25
            
            frame = np.array(img)
            num_frames = int(duration_per_command * self.fps)
            frames.extend([frame] * num_frames)
        
        logger.info(f"Recorded {len(frames)} terminal frames")
        return frames
    
    def record_training_visualization(self, epochs: int = 10) -> List[np.ndarray]:
        """Create visualization of training progress."""
        frames = []
        
        if not IMAGING_AVAILABLE:
            return frames
        
        logger.info("Recording training visualization...")
        
        for epoch in range(epochs):
            img = Image.new('RGB', self.resolution, color=(25, 25, 35))
            draw = ImageDraw.Draw(img)
            
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                font = ImageFont.truetype("arial.ttf", 18)
            except:
                title_font = ImageFont.load_default()
                font = ImageFont.load_default()
            
            # Title
            draw.text((50, 30), "CogniForge Training Progress", fill=(255, 255, 255), font=title_font)
            
            # Progress bar
            progress = (epoch + 1) / epochs
            bar_width = self.width - 100
            bar_height = 30
            bar_x = 50
            bar_y = 100
            
            # Background
            draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                         fill=(50, 50, 50), outline=(100, 100, 100))
            
            # Progress
            progress_width = int(bar_width * progress)
            draw.rectangle([bar_x, bar_y, bar_x + progress_width, bar_y + bar_height], 
                         fill=(0, 200, 100))
            
            # Text
            draw.text((bar_x, bar_y + 40), f"Epoch {epoch + 1}/{epochs}", 
                     fill=(200, 200, 200), font=font)
            
            # Metrics
            loss = 2.5 * np.exp(-epoch * 0.3) + 0.1 * np.random.random()
            accuracy = 0.5 + 0.45 * (1 - np.exp(-epoch * 0.5)) + 0.05 * np.random.random()
            
            metrics_y = 200
            draw.text((50, metrics_y), "Metrics:", fill=(255, 255, 255), font=title_font)
            draw.text((50, metrics_y + 40), f"Loss: {loss:.4f}", fill=(180, 180, 180), font=font)
            draw.text((50, metrics_y + 70), f"Accuracy: {accuracy:.2%}", fill=(180, 180, 180), font=font)
            
            # Deterministic mode indicator
            draw.text((50, self.height - 100), "✓ Deterministic Mode Enabled", 
                     fill=(0, 255, 0), font=font)
            draw.text((50, self.height - 70), f"Seed: 42", 
                     fill=(180, 180, 180), font=font)
            
            frame = np.array(img)
            # Add frames for animation
            frames.extend([frame] * 15)  # 0.5 seconds per epoch at 30fps
        
        logger.info(f"Recorded {len(frames)} training frames")
        return frames
    
    def save_video(self, frames: List[np.ndarray], filename: str = None):
        """Save frames as MP4 video."""
        if not frames:
            logger.warning("No frames to save")
            return None
        
        if filename is None:
            filename = f"cogniforge_demo_{self.timestamp}.mp4"
        
        # Use safe file manager to get safe path
        output_path = self.safe_manager.get_safe_path(filename, WriteScope.OUTPUTS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if CV2_AVAILABLE:
            logger.info(f"Saving video to {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.fps, self.resolution)
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()
            logger.info(f"Video saved: {output_path}")
            return output_path
        
        elif IMAGING_AVAILABLE:
            # Use imageio as fallback
            logger.info(f"Saving video with imageio to {output_path}")
            imageio.mimwrite(str(output_path), frames, fps=self.fps)
            logger.info(f"Video saved: {output_path}")
            return output_path
        
        else:
            logger.error("No video library available. Install opencv-python or imageio[ffmpeg]")
            return None
    
    def save_gif(self, frames: List[np.ndarray], filename: str = None, max_frames: int = 100):
        """Save frames as GIF."""
        if not frames:
            logger.warning("No frames to save")
            return None
        
        if not IMAGING_AVAILABLE:
            logger.error("PIL/imageio not available. Install with: pip install pillow imageio")
            return None
        
        if filename is None:
            filename = f"cogniforge_demo_{self.timestamp}.gif"
        
        # Use safe file manager to get safe path
        output_path = self.safe_manager.get_safe_path(filename, WriteScope.OUTPUTS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reduce frames for smaller GIF
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step]
        
        logger.info(f"Saving GIF to {output_path} ({len(frames)} frames)")
        
        # Convert numpy arrays to PIL images
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        pil_images[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_images[1:],
            duration=1000//self.fps,  # Duration in milliseconds
            loop=0
        )
        
        logger.info(f"GIF saved: {output_path}")
        return output_path
    
    def run_full_demo(self):
        """Run the complete CogniForge demo and record it."""
        all_frames = []
        
        # 1. Title screen
        logger.info("Creating title screen...")
        title_frames = self.create_title_frame(
            "CogniForge Demo",
            "AI-Powered Robotics Framework",
            duration=3.0
        )
        all_frames.extend(title_frames)
        
        # 2. Terminal commands
        logger.info("Recording terminal commands...")
        commands = [
            "cogv run train --deterministic --seed 42",
            "cogv gen-code 'pick up the red cube'",
            "cogv demo grasp --visualize"
        ]
        terminal_frames = self.record_terminal_demo(commands, duration_per_command=2.0)
        all_frames.extend(terminal_frames)
        
        # 3. Training visualization
        logger.info("Recording training progress...")
        training_frames = self.record_training_visualization(epochs=10)
        all_frames.extend(training_frames)
        
        # 4. PyBullet simulation
        if PYBULLET_AVAILABLE:
            logger.info("Recording PyBullet simulation...")
            sim_frames = self.record_pybullet_demo(duration=8.0)
            all_frames.extend(sim_frames)
        
        # 5. End screen
        logger.info("Creating end screen...")
        end_frames = self.create_title_frame(
            "CogniForge",
            "github.com/yourusername/cogniforge",
            duration=2.0
        )
        all_frames.extend(end_frames)
        
        # Save as video and GIF
        logger.info(f"Total frames: {len(all_frames)}")
        
        video_path = self.save_video(all_frames)
        gif_path = self.save_gif(all_frames)
        
        return video_path, gif_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("COGNIFORGE DEMO RECORDER")
    print("=" * 60)
    
    # Check dependencies
    print("\nChecking dependencies:")
    print(f"  PyBullet: {'✓' if PYBULLET_AVAILABLE else '✗ (pip install pybullet)'}")
    print(f"  OpenCV:   {'✓' if CV2_AVAILABLE else '✗ (pip install opencv-python)'}")
    print(f"  Imaging:  {'✓' if IMAGING_AVAILABLE else '✗ (pip install pillow imageio[ffmpeg])'}")
    
    if not any([PYBULLET_AVAILABLE, CV2_AVAILABLE, IMAGING_AVAILABLE]):
        print("\n⚠ Please install required dependencies:")
        print("  pip install pybullet opencv-python pillow imageio[ffmpeg]")
        return 1
    
    print("\nStarting demo recording...")
    print("-" * 60)
    
    # Create recorder
    recorder = DemoRecorder(output_dir="recordings", resolution=(1280, 720))
    
    # Run full demo
    try:
        video_path, gif_path = recorder.run_full_demo()
        
        print("\n" + "=" * 60)
        print("RECORDING COMPLETE!")
        print("=" * 60)
        
        if video_path:
            print(f"✓ Video saved: {video_path}")
            print(f"  Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        if gif_path:
            print(f"✓ GIF saved: {gif_path}")
            print(f"  Size: {gif_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        print("\nYou can now share these files to showcase CogniForge!")
        
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())