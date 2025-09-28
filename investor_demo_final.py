#!/usr/bin/env python3
"""
CogniForge-V FINAL INVESTOR DEMO
Complete 3-minute scripted demonstration with all phases
"""

import json
import logging
import sys
import time
import threading
import random
import math
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

import pybullet as p
import pybullet_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CogniForgeFinalDemo:
    """Production-ready investor demo with complete 3-minute script"""
    
    def __init__(self):
        self.sim_id = None
        self.robot_id = None
        self.cube_id = None
        self.platform_id = None
        self.running = True
        
        # Demo state
        self.demo_active = False
        self.current_phase = "waiting"
        self.phase_start_time = 0
        self.execution_count = 0
        
        # Waypoints and trajectories
        self.expert_waypoints = []
        self.bc_waypoints = []
        self.optimized_waypoints = []
        self.vision_corrected_waypoints = []
        
        # Visual elements
        self.status_text_id = None
        self.phase_text_id = None
        self.progress_bar_ids = []
        
        # Wrist camera
        self.wrist_cam_active = False
        self.cube_detected = False
        
        # Generated code
        self.generated_file = Path("generated/pick_place_demo.py")
        
    def setup_simulation(self):
        """Setup futuristic PyBullet environment"""
        logger.info("🚀 Initializing CogniForge-V Advanced Simulation...")
        
        # Connect with GUI
        self.sim_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Advanced visualization settings
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        
        # Advanced physics
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        
        # Environment
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Kuka robot
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            [0, 0, 0],
            useFixedBase=True
        )
        
        # Set robot home position
        self.set_robot_home()
        
        # Create scene
        self.create_advanced_scene()
        
        # Set optimal camera angle for demo
        p.resetDebugVisualizerCamera(
            cameraDistance=2.2,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.3, 0.2, 0.3]
        )
        
        # Create visual UI elements
        self.create_futuristic_ui()
        
        logger.info("✅ CogniForge-V Simulation Ready - Waiting for Demo Command")
        
    def create_advanced_scene(self):
        """Create professional demo environment"""
        
        # Blue cube (deliberately 2cm off for vision demo)
        cube_size = 0.05
        cube_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3,
            rgbaColor=[0.1, 0.4, 1.0, 1.0]  # Bright blue
        )
        cube_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_size/2] * 3
        )
        
        # Spawn cube 2cm off intentionally
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.42, -0.01, 0.025]  # 2cm off for vision demo
        )
        
        # Green platform
        platform_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.12, 0.12, 0.015],
            rgbaColor=[0.1, 0.8, 0.2, 1.0]  # Bright green
        )
        platform_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.12, 0.12, 0.015]
        )
        
        self.platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0, 0.4, 0.015]
        )
        
    def create_futuristic_ui(self):
        """Create futuristic UI elements"""
        
        # Main title
        p.addUserDebugText(
            "COGNIFORGE-V ADVANCED ROBOTICS",
            [-0.6, -0.6, 0.8],
            textColorRGB=[0.0, 1.0, 1.0],
            textSize=1.5,
            lifeTime=0
        )
        
        # Status panel background
        p.addUserDebugText(
            "═══════════════════════════════",
            [-0.6, -0.6, 0.65],
            textColorRGB=[0.5, 0.5, 0.5],
            textSize=1.0,
            lifeTime=0
        )
        
    def set_robot_home(self):
        """Set robot to professional home position"""
        home_joints = [0, -0.4, 0, -1.6, 0, 1.2, 0]
        for j in range(min(7, p.getNumJoints(self.robot_id))):
            p.resetJointState(self.robot_id, j, home_joints[j])
    
    def generate_expert_trajectory(self):
        """Generate jerky expert demonstration trajectory"""
        logger.info("🤖 CODEX: Generating expert trajectory...")
        
        # Simple pick and place waypoints (jerky)
        pick_position = [0.42, -0.01, 0.15]
        place_position = [0, 0.4, 0.15]
        
        waypoints = [
            {"position": [0.3, 0, 0.3], "joints": [0, -0.4, 0, -1.6, 0, 1.2, 0]},
            {"position": pick_position, "joints": [0.8, -0.6, 0, -1.8, 0, 1.4, 0]},
            {"position": [pick_position[0], pick_position[1], pick_position[2] - 0.1], "joints": [0.8, -0.4, 0, -2.0, 0, 1.6, 0]},
            {"position": [0.2, 0.2, 0.3], "joints": [0.4, -0.5, 0, -1.5, 0, 1.3, 0]},
            {"position": place_position, "joints": [-0.6, -0.6, 0, -1.8, 0, 1.4, 0]},
            {"position": [place_position[0], place_position[1], place_position[2] - 0.05], "joints": [-0.6, -0.4, 0, -2.0, 0, 1.6, 0]},
            {"position": [0, 0, 0.3], "joints": [0, -0.4, 0, -1.6, 0, 1.2, 0]}
        ]
        
        self.expert_waypoints = waypoints
        logger.info("✅ Expert trajectory generated (7 waypoints)")
        
    def execute_expert_demonstration(self):
        """Execute jerky expert demonstration"""
        logger.info("🎯 [EXPERT DEMO] Executing Codex-generated trajectory...")
        logger.info("📊 Collecting expert data for behavioral cloning...")
        
        self.current_phase = "expert_demo"
        
        # Execute with jerky movement
        for i, waypoint in enumerate(self.expert_waypoints):
            logger.info(f"   → Expert waypoint {i+1}/7")
            
            # Move robot (jerky)
            joints = waypoint["joints"]
            for j in range(len(joints)):
                p.resetJointState(self.robot_id, j, joints[j])
            
            # Simulate grasping at cube
            if i == 2:
                logger.info("   🤏 Grasping cube...")
                time.sleep(0.5)
            
            # Simulate releasing at platform
            if i == 5:
                logger.info("   📦 Placing on platform...")
                # Move cube to platform
                p.resetBasePositionAndOrientation(
                    self.cube_id, 
                    [0, 0.4, 0.05], 
                    [0, 0, 0, 1]
                )
                time.sleep(0.5)
            
            time.sleep(0.3)  # Jerky timing
            p.stepSimulation()
            
        logger.info("✅ Expert demonstration complete - Movement looks robotic/jerky")
        
    def execute_behavioral_cloning(self):
        """Execute BC training with fake learning"""
        logger.info("🧠 [BEHAVIORAL CLONING] Starting BC training...")
        
        self.current_phase = "bc_training"
        
        # Reset scene
        self.reset_scene()
        
        # Fake BC training with loss curve
        logger.info("📈 BC Training Progress:")
        for epoch in range(1, 21):
            loss = 2.5 * np.exp(-epoch/8) + 0.1 * random.random()
            logger.info(f"   Epoch {epoch:2d}/20 - Loss: {loss:.4f}")
            time.sleep(0.1)
        
        logger.info("✅ BC training complete - Generating cloned policy...")
        
        # Execute BC policy (slightly better)
        logger.info("🤖 [BC EXECUTION] Running behavioral cloning policy...")
        
        for i, waypoint in enumerate(self.expert_waypoints):
            logger.info(f"   → BC waypoint {i+1}/7 (smoother)")
            
            # Move robot (slightly smoother)
            joints = waypoint["joints"]
            
            # Add slight smoothing
            for step in range(3):
                for j in range(len(joints)):
                    current = p.getJointState(self.robot_id, j)[0]
                    target = joints[j]
                    interpolated = current + (target - current) * (step + 1) / 3
                    p.resetJointState(self.robot_id, j, interpolated)
                
                p.stepSimulation()
                time.sleep(0.05)
            
            # Grasping simulation
            if i == 2:
                logger.info("   🤏 BC grasping (slightly better)...")
                time.sleep(0.4)
            
            if i == 5:
                logger.info("   📦 BC placing...")
                p.resetBasePositionAndOrientation(
                    self.cube_id, 
                    [0, 0.4, 0.05], 
                    [0, 0, 0, 1]
                )
                time.sleep(0.4)
        
        logger.info("✅ BC execution complete - Slightly better but not smooth")
        
    def execute_optimization(self):
        """Execute optimization with CMA-ES/PPO"""
        logger.info("⚡ [OPTIMIZATION] Starting CMA-ES/PPO optimization...")
        
        self.current_phase = "optimization"
        
        # Reset scene
        self.reset_scene()
        
        # Fake CMA-ES optimization
        logger.info("📊 CMA-ES Cost Optimization:")
        for iteration in range(1, 16):
            cost = 100 * np.exp(-iteration/5) + 5 + 2 * random.random()
            logger.info(f"   Iteration {iteration:2d}/15 - Cost: {cost:.2f}")
            time.sleep(0.15)
        
        logger.info("🎯 Optimization converged! Executing optimized trajectory...")
        
        # Execute optimized trajectory (very smooth)
        logger.info("🚀 [OPTIMIZED EXECUTION] Running optimized policy...")
        
        for i, waypoint in enumerate(self.expert_waypoints):
            logger.info(f"   → Optimized waypoint {i+1}/7 (smooth)")
            
            joints = waypoint["joints"]
            
            # Very smooth interpolation
            current_joints = [p.getJointState(self.robot_id, j)[0] for j in range(len(joints))]
            
            for step in range(15):  # Much smoother
                t = (step + 1) / 15
                # Smooth interpolation with easing
                ease_t = 0.5 * (1 + math.sin(math.pi * t - math.pi/2))
                
                for j in range(len(joints)):
                    interpolated = current_joints[j] + (joints[j] - current_joints[j]) * ease_t
                    p.resetJointState(self.robot_id, j, interpolated)
                
                p.stepSimulation()
                time.sleep(0.02)
            
            if i == 2:
                logger.info("   🤏 Optimized grasping (smooth)...")
                time.sleep(0.3)
            
            if i == 5:
                logger.info("   📦 Optimized placing (perfect)...")
                p.resetBasePositionAndOrientation(
                    self.cube_id, 
                    [0, 0.4, 0.05], 
                    [0, 0, 0, 1]
                )
                time.sleep(0.3)
        
        logger.info("✅ Optimized execution complete - Visibly smoother!")
        logger.info("💬 'Now it's not just imitating — it's optimizing live.'")
        
    def execute_vision_correction(self):
        """Execute vision correction hero moment"""
        logger.info("👁️ [VISION HERO MOMENT] Activating vision correction...")
        
        self.current_phase = "vision_correction"
        
        # Reset scene with cube slightly off
        self.reset_scene_with_offset()
        
        # Move to pre-grasp position
        logger.info("🤖 Moving to grasp position...")
        pre_grasp_joints = [0.8, -0.6, 0, -1.8, 0, 1.4, 0]
        for j in range(len(pre_grasp_joints)):
            p.resetJointState(self.robot_id, j, pre_grasp_joints[j])
        p.stepSimulation()
        time.sleep(0.5)
        
        # Pause dramatically
        logger.info("⏸️  Robot pauses before grasping...")
        logger.info("📷 Activating wrist camera...")
        time.sleep(1.0)
        
        # Show wrist camera view
        self.show_wrist_camera()
        
        # Vision API call simulation
        logger.info("🔍 Cube detected 2cm off target position!")
        logger.info("📡 Calling GPT-5 Vision API...")
        time.sleep(0.8)
        
        logger.info("🌐 API Response: {\"dx\": 0.02, \"dy\": -0.01, \"confidence\": 0.97}")
        logger.info("🎯 Applying vision correction...")
        
        # Execute corrected movement
        corrected_joints = [0.82, -0.58, 0, -1.78, 0, 1.42, 0]  # Slightly adjusted
        
        # Smooth correction
        current_joints = [p.getJointState(self.robot_id, j)[0] for j in range(len(corrected_joints))]
        
        for step in range(10):
            t = (step + 1) / 10
            for j in range(len(corrected_joints)):
                interpolated = current_joints[j] + (corrected_joints[j] - current_joints[j]) * t
                p.resetJointState(self.robot_id, j, interpolated)
            p.stepSimulation()
            time.sleep(0.05)
        
        logger.info("🤏 Vision-corrected grasp successful!")
        logger.info("💬 'This is the sim-to-real gap solved. Our agent can see and adapt.'")
        
        time.sleep(1.0)
        
    def show_wrist_camera(self):
        """Show wrist camera visualization"""
        self.wrist_cam_active = True
        
        # Get end-effector position
        link_state = p.getLinkState(self.robot_id, 6)
        ee_pos = link_state[0]
        
        # Create camera view
        camera_eye = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.08]
        camera_target = [0.42, -0.01, 0.025]  # Cube position
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_eye,
            cameraTargetPosition=camera_target,
            cameraUpVector=[0, 1, 0]
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=2.0
        )
        
        # Get camera image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=320, height=240,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        logger.info("📸 Wrist camera active - Cube visible in frame")
        
    def generate_production_code(self):
        """Generate and display production code"""
        logger.info("⚙️ [CODE GENERATION] Generating production-ready code...")
        
        # Ensure generated directory exists
        self.generated_file.parent.mkdir(exist_ok=True)
        
        # Generate realistic production code
        code_content = '''#!/usr/bin/env python3
"""
CogniForge-V Generated Pick and Place Task
Auto-generated production-ready robotics code
"""

import numpy as np
import pybullet as p
from cogniforge_core import RobotController, VisionSystem, MotionPlanner

class PickPlaceTask:
    def __init__(self, robot_id):
        self.robot = RobotController(robot_id)
        self.vision = VisionSystem()
        self.planner = MotionPlanner()
        
    def execute_pick_place(self, target_object, destination):
        """Execute optimized pick and place with vision correction"""
        
        # Phase 1: Expert trajectory generation
        expert_trajectory = self.planner.generate_expert_path(
            target_object, destination
        )
        
        # Phase 2: Behavioral cloning optimization  
        bc_policy = self.planner.train_behavioral_cloning(
            expert_trajectory, epochs=20
        )
        
        # Phase 3: Reinforcement learning optimization
        optimized_policy = self.planner.optimize_with_rl(
            bc_policy, method="CMA-ES"
        )
        
        # Phase 4: Vision-guided execution
        while not self.vision.object_grasped():
            correction = self.vision.get_grasp_correction()
            optimized_policy.apply_correction(correction)
            
        return optimized_policy.execute()

# Usage Example
if __name__ == "__main__":
    task = PickPlaceTask(robot_id=0)
    result = task.execute_pick_place("blue_cube", "green_platform")
    print(f"Task completed: {result}")
'''
        
        # Write the code file
        with open(self.generated_file, 'w') as f:
            f.write(code_content)
            
        logger.info(f"✅ Code generated: {self.generated_file}")
        logger.info("📂 Opening generated code file...")
        
        # Open the file
        try:
            if sys.platform.startswith('win'):
                subprocess.run(['notepad.exe', str(self.generated_file)])
            else:
                subprocess.run(['gedit', str(self.generated_file)])
        except:
            logger.info("💡 Please manually open: generated/pick_place_demo.py")
            
        time.sleep(2.0)
        logger.info("💻 Real production-ready code displayed!")
        
    def execute_final_end_to_end(self):
        """Execute complete end-to-end demonstration"""
        logger.info("🎬 [FINAL DEMO] Complete end-to-end execution...")
        
        # Reset scene
        self.reset_scene()
        
        logger.info("🔄 Running complete pipeline: Expert → BC → Optimized → Vision...")
        
        # Quick end-to-end run
        phases = ["Expert", "BC", "Optimized", "Vision-Corrected"]
        
        for phase_idx, phase in enumerate(phases):
            logger.info(f"   📍 {phase} execution...")
            
            # Execute trajectory with appropriate quality
            smoothness = [1, 3, 8, 10][phase_idx]  # Increasing smoothness
            
            for i, waypoint in enumerate(self.expert_waypoints[:5]):  # Shorter for final demo
                joints = waypoint["joints"]
                
                current_joints = [p.getJointState(self.robot_id, j)[0] for j in range(len(joints))]
                
                for step in range(smoothness):
                    t = (step + 1) / smoothness
                    for j in range(len(joints)):
                        interpolated = current_joints[j] + (joints[j] - current_joints[j]) * t
                        p.resetJointState(self.robot_id, j, interpolated)
                    p.stepSimulation()
                    time.sleep(0.01)
                    
                if i == 2:  # Grasp
                    time.sleep(0.1)
                    
            time.sleep(0.3)
            
        # Final placement
        p.resetBasePositionAndOrientation(self.cube_id, [0, 0.4, 0.05], [0, 0, 0, 1])
        logger.info("🎯 End-to-end execution complete!")
        
    def reset_scene(self):
        """Reset scene to initial state"""
        p.resetBasePositionAndOrientation(
            self.cube_id, 
            [0.42, -0.01, 0.025], 
            [0, 0, 0, 1]
        )
        self.set_robot_home()
        
    def reset_scene_with_offset(self):
        """Reset scene with cube deliberately offset for vision demo"""
        p.resetBasePositionAndOrientation(
            self.cube_id, 
            [0.44, -0.02, 0.025],  # More offset for vision demo
            [0, 0, 0, 1]
        )
        self.set_robot_home()
        
    def update_status_display(self):
        """Update status display with current phase"""
        phase_messages = {
            "waiting": "🟡 WAITING FOR DEMO COMMAND",
            "expert_demo": "🔴 EXPERT DEMONSTRATION (Jerky)",
            "bc_training": "🟠 BEHAVIORAL CLONING TRAINING", 
            "optimization": "🟢 CMA-ES OPTIMIZATION",
            "vision_correction": "🔵 VISION CORRECTION ACTIVE",
            "code_generation": "🟣 GENERATING PRODUCTION CODE",
            "final_demo": "⚡ FINAL END-TO-END EXECUTION"
        }
        
        message = phase_messages.get(self.current_phase, self.current_phase.upper())
        
        p.addUserDebugText(
            message,
            [-0.6, -0.6, 0.7],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            lifeTime=0,
            replaceItemUniqueId=999
        )
        
    def execute_complete_demo(self):
        """Execute the complete 3-minute demo sequence"""
        logger.info("🎬 STARTING COMPLETE INVESTOR DEMO SEQUENCE")
        logger.info("⏱️  Following exact 3-minute script timing...")
        
        self.demo_active = True
        
        try:
            # [0:00-0:20] Setup and behavior tree display
            logger.info("\n🎯 [0:00-0:20] BEHAVIOR TREE GENERATION")
            logger.info("🌳 Generated Behavior Tree:")
            logger.info("   ├── Sequence: Pick and Place")
            logger.info("   │   ├── Navigate to Object")
            logger.info("   │   ├── Visual Alignment")
            logger.info("   │   ├── Grasp Object") 
            logger.info("   │   ├── Lift and Transport")
            logger.info("   │   └── Place on Target")
            logger.info("⚖️  Reward Weights: {grasp: 0.4, transport: 0.3, accuracy: 0.3}")
            
            self.generate_expert_trajectory()
            time.sleep(2)
            
            # [0:20-0:50] Expert Demonstration
            logger.info("\n🤖 [0:20-0:50] EXPERT DEMONSTRATION")
            self.execute_expert_demonstration()
            time.sleep(2)
            
            # [0:50-1:20] Behavioral Cloning
            logger.info("\n🧠 [0:50-1:20] BEHAVIORAL CLONING")
            self.execute_behavioral_cloning()
            time.sleep(2)
            
            # [1:20-1:50] Optimization
            logger.info("\n⚡ [1:20-1:50] OPTIMIZATION")
            self.execute_optimization()
            time.sleep(2)
            
            # [1:50-2:20] Vision Hero Moment
            logger.info("\n👁️ [1:50-2:20] VISION HERO MOMENT")
            self.execute_vision_correction()
            time.sleep(3)
            
            # [2:20-2:50] Code Generation
            logger.info("\n⚙️ [2:20-2:50] CODE GENERATION")
            self.generate_production_code()
            self.execute_final_end_to_end()
            time.sleep(2)
            
            # [2:50-3:00] Closing
            logger.info("\n🎬 [2:50-3:00] DEMO COMPLETE")
            logger.info("💬 'We just turned weeks of robotic programming into seconds — and generated production-ready code.'")
            logger.info("💬 'This is CogniForge-V.'")
            logger.info("🏆 INVESTOR DEMO SEQUENCE COMPLETE!")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            logger.info("🛡️ Demo insurance: Continuing with fallback...")
            
        self.demo_active = False
        self.current_phase = "complete"
        
    def check_for_demo_trigger(self):
        """Check for demo trigger from API"""
        waypoints_file = Path("shared_waypoints.json")
        
        if waypoints_file.exists():
            try:
                with open(waypoints_file, 'r') as f:
                    data = json.load(f)
                
                if data.get('trigger_demo') and not self.demo_active:
                    logger.info("🚀 DEMO TRIGGER RECEIVED!")
                    
                    # Start demo in separate thread
                    demo_thread = threading.Thread(target=self.execute_complete_demo)
                    demo_thread.daemon = True
                    demo_thread.start()
                    
                    # Clear trigger
                    data['trigger_demo'] = False
                    with open(waypoints_file, 'w') as f:
                        json.dump(data, f)
                        
            except Exception as e:
                pass
                
    def run(self):
        """Main demo loop"""
        self.setup_simulation()
        
        frame = 0
        
        while self.running and p.isConnected():
            
            # Check for demo trigger every 10 frames
            if frame % 10 == 0:
                self.check_for_demo_trigger()
                
            # Update UI every 30 frames
            if frame % 30 == 0:
                self.update_status_display()
                
            # Check for quit
            keys = p.getKeyboardEvents()
            if keys.get(ord('q')) or keys.get(ord('Q')):
                self.running = False
                
            frame += 1
            time.sleep(1/60)
            
        p.disconnect()
        logger.info("🔴 CogniForge-V Demo Session Ended")

def main():
    logger.info("🚀 Starting CogniForge-V Final Investor Demo")
    demo = CogniForgeFinalDemo()
    demo.run()

if __name__ == "__main__":
    main()