"""
Robot Simulator using PyBullet.

This module provides a RobotSimulator class that manages PyBullet simulations
with support for both GUI and headless (DIRECT) modes, robot loading, and
basic physics setup.
"""

import os
import time
import logging
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import pybullet as p
import pybullet_data

from cogniforge.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class RobotType(Enum):
    """Supported robot types."""

    KUKA_IIWA = "kuka_iiwa"
    FRANKA_PANDA = "franka_panda"
    UR5 = "ur5"
    CUSTOM = "custom"


class SimulationMode(Enum):
    """PyBullet simulation modes."""

    GUI = p.GUI
    DIRECT = p.DIRECT
    SHARED_MEMORY = p.SHARED_MEMORY
    UDP = p.UDP
    TCP = p.TCP


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""

    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    time_step: float = 1.0 / 240.0  # 240 Hz
    solver_iterations: int = 150
    use_real_time: bool = False
    camera_distance: float = 1.5
    camera_yaw: float = 45
    camera_pitch: float = -30
    camera_target: Tuple[float, float, float] = (0, 0, 0)
    plane_texture: Optional[str] = None
    enable_rendering: bool = True
    enable_shadows: bool = True
    enable_wireframe: bool = False
    seed: Optional[int] = None  # Random seed for deterministic simulations
    deterministic: bool = False  # Enable fully deterministic mode


@dataclass
class RobotInfo:
    """Information about a loaded robot."""

    robot_id: int
    name: str
    robot_type: RobotType
    base_position: Tuple[float, float, float]
    base_orientation: Tuple[float, float, float, float]
    num_joints: int
    joint_indices: List[int]
    end_effector_index: Optional[int] = None
    gripper_indices: Optional[List[int]] = None
    tool_link_index: Optional[int] = None


class RobotSimulator:
    """
    PyBullet-based robot simulator with automatic GUI/DIRECT mode selection.

    This class manages PyBullet simulations, automatically selecting GUI mode
    when DISPLAY is set (or on Windows) and DIRECT mode for headless operation.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        force_mode: Optional[SimulationMode] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the robot simulator.

        Args:
            config: Simulation configuration. Uses defaults if None.
            force_mode: Force a specific simulation mode. Auto-detects if None.
            seed: Random seed for reproducible simulations. Overrides config.seed if provided.
        """
        self.config = config or SimulationConfig()
        self.physics_client = None
        self.plane_id = None
        self.robots: Dict[str, RobotInfo] = {}
        self.objects: Dict[str, int] = {}
        self.constraints: Dict[str, int] = {}
        self._is_connected = False
        self._simulation_mode = force_mode
        self._step_counter = 0
        
        # Set seed from argument or config
        if seed is not None:
            self.config.seed = seed
        
        # Initialize random seeds if specified
        if self.config.seed is not None:
            self.set_seed(self.config.seed)

        # Determine simulation mode if not forced
        if self._simulation_mode is None:
            self._simulation_mode = self._detect_simulation_mode()

        logger.info(f"RobotSimulator initialized with mode: {self._simulation_mode.name}")
        if self.config.seed is not None:
            logger.info(f"Random seed set to: {self.config.seed}")

    def _detect_simulation_mode(self) -> SimulationMode:
        """
        Automatically detect the appropriate simulation mode.

        Returns:
            SimulationMode.GUI if DISPLAY is set or on Windows, DIRECT otherwise.
        """
        # Check if DISPLAY environment variable is set
        display = os.environ.get("DISPLAY")

        # Check if running in headless mode (from settings)
        pybullet_gui = settings.get("pybullet_gui", False)

        # Windows typically has GUI available
        is_windows = os.name == "nt"

        if force_gui := pybullet_gui:
            logger.info("GUI mode forced by settings")
            return SimulationMode.GUI
        elif display or is_windows:
            logger.info(
                f"GUI mode selected (DISPLAY={display}, Windows={is_windows})"
            )
            return SimulationMode.GUI
        else:
            logger.info("DIRECT mode selected (headless environment)")
            return SimulationMode.DIRECT
    
    def set_seed(self, seed: int):
        """
        Set random seeds for reproducible simulations.
        
        This sets seeds for:
        - NumPy random number generator
        - Python's random module
        - PyBullet's random number generator (if connected)
        
        Args:
            seed: Integer seed value for random number generators.
        
        Example:
            sim = RobotSimulator()
            sim.set_seed(42)  # Set seed before connecting
            sim.connect()
            
            # Or pass seed to constructor
            sim = RobotSimulator(seed=42)
            
            # Or use deterministic config
            config = SimulationConfig(seed=42, deterministic=True)
            sim = RobotSimulator(config=config)
        """
        self.config.seed = seed
        
        # Set NumPy seed
        np.random.seed(seed)
        logger.info(f"NumPy random seed set to {seed}")
        
        # Set Python random seed
        random.seed(seed)
        logger.info(f"Python random seed set to {seed}")
        
        # Set PyBullet seed if connected
        if self._is_connected:
            # PyBullet doesn't have a direct seed setting, but we can
            # ensure deterministic behavior through physics parameters
            logger.info("PyBullet deterministic mode will be applied on next connect()")
    
    def reset_seeds(self):
        """
        Reset random seeds to a new value based on current time.
        
        This is useful for switching back to non-deterministic behavior.
        """
        import time
        seed = int(time.time() * 1000) % 2**31  # Use current time as seed
        self.set_seed(seed)
        self.config.deterministic = False
        logger.info(f"Random seeds reset to time-based value: {seed}")

    def connect(self) -> int:
        """
        Connect to PyBullet physics server.

        Returns:
            Physics client ID.

        Raises:
            RuntimeError: If connection fails.
        """
        if self._is_connected:
            logger.warning("Already connected to PyBullet")
            return self.physics_client

        try:
            # Connect based on mode
            if self._simulation_mode == SimulationMode.GUI:
                self.physics_client = p.connect(p.GUI)
                self._setup_gui()
            else:
                self.physics_client = p.connect(p.DIRECT)

            self._is_connected = True
            logger.info(
                f"Connected to PyBullet in {self._simulation_mode.name} mode "
                f"(client_id={self.physics_client})"
            )

            # Set additional data path for PyBullet models
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Configure simulation
            self._configure_simulation()

            return self.physics_client

        except Exception as e:
            logger.error(f"Failed to connect to PyBullet: {e}")
            raise RuntimeError(f"PyBullet connection failed: {e}")

    def _setup_gui(self):
        """Configure GUI settings."""
        if self._simulation_mode != SimulationMode.GUI:
            return

        # Configure visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, self.config.enable_shadows)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, self.config.enable_wireframe)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.config.enable_rendering)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)

        # Set camera
        p.resetDebugVisualizerCamera(
            cameraDistance=self.config.camera_distance,
            cameraYaw=self.config.camera_yaw,
            cameraPitch=self.config.camera_pitch,
            cameraTargetPosition=self.config.camera_target,
        )

    def _configure_simulation(self):
        """Configure simulation parameters."""
        # Set gravity
        p.setGravity(*self.config.gravity)
        logger.info(f"Gravity set to: {self.config.gravity}")

        # Set time step
        p.setTimeStep(self.config.time_step)

        # Configure physics engine parameters for determinism
        if self.config.deterministic:
            # Use fixed time step and solver settings for deterministic simulation
            p.setPhysicsEngineParameter(
                fixedTimeStep=self.config.time_step,
                numSolverIterations=self.config.solver_iterations,
                numSubSteps=1,  # Single substep for determinism
                deterministicOverlappingPairs=1,  # Deterministic collision detection
                useSplitImpulse=1,  # More stable contact resolution
                splitImpulsePenetrationThreshold=-0.02,
                contactBreakingThreshold=0.02,
                enableConeFriction=1,
                solverResidualThreshold=1e-7,
            )
            logger.info("Deterministic physics engine parameters set")
        else:
            # Standard physics configuration
            p.setPhysicsEngineParameter(
                numSolverIterations=self.config.solver_iterations,
                numSubSteps=1,
            )

        # Set real-time simulation if requested
        p.setRealTimeSimulation(self.config.use_real_time)

    def load_plane(self, z_offset: float = 0.0) -> int:
        """
        Load a plane into the simulation.

        Args:
            z_offset: Height offset for the plane.

        Returns:
            Plane body ID.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")

        # Load plane URDF
        self.plane_id = p.loadURDF(
            "plane.urdf",
            basePosition=[0, 0, z_offset],
            useFixedBase=True,
        )

        # Apply texture if specified
        if self.config.plane_texture:
            texture_id = p.loadTexture(self.config.plane_texture)
            p.changeVisualShape(self.plane_id, -1, textureUniqueId=texture_id)

        logger.info(f"Loaded plane (id={self.plane_id}, z_offset={z_offset})")
        return self.plane_id

    def load_robot(
        self,
        robot_type: RobotType = RobotType.KUKA_IIWA,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Optional[Tuple[float, float, float, float]] = None,
        fixed_base: bool = True,
        robot_name: Optional[str] = None,
        urdf_path: Optional[str] = None,
        auto_fallback: bool = True,
    ) -> RobotInfo:
        """
        Load a robot into the simulation.

        Args:
            robot_type: Type of robot to load.
            position: Base position (x, y, z).
            orientation: Base orientation quaternion (x, y, z, w). Default is upright.
            fixed_base: Whether to fix the robot base.
            robot_name: Custom name for the robot. Auto-generated if None.
            urdf_path: Custom URDF path for CUSTOM robot type.
            auto_fallback: If True, automatically fallback to Panda if KUKA doesn't have 7 DOF.

        Returns:
            RobotInfo object containing robot information.

        Raises:
            ValueError: If robot type is not supported or URDF path is invalid.
            RuntimeError: If not connected to PyBullet.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")

        # Default orientation (upright)
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])

        # Track original robot type for fallback
        original_robot_type = robot_type
        
        # Try loading robot with fallback support
        try:
            robot_id = self._load_robot_urdf(
                robot_type, position, orientation, fixed_base, urdf_path
            )
            
            # Get robot information
            num_joints = p.getNumJoints(robot_id)
            
            # Verify KUKA has expected DOF (7), fallback to Panda if not
            if robot_type == RobotType.KUKA_IIWA and auto_fallback:
                # Count movable joints
                joint_info = [p.getJointInfo(robot_id, i) for i in range(num_joints)]
                movable_joints = [
                    i for i, info in enumerate(joint_info)
                    if info[2] != p.JOINT_FIXED
                ]
                
                # KUKA should have at least 7 movable joints
                if len(movable_joints) < 7:
                    logger.warning(
                        f"KUKA robot has only {len(movable_joints)} movable joints "
                        f"(expected 7). Falling back to Franka Panda."
                    )
                    # Remove the incorrectly loaded robot
                    p.removeBody(robot_id)
                    
                    # Load Panda instead
                    robot_type = RobotType.FRANKA_PANDA
                    robot_id = self._load_robot_urdf(
                        robot_type, position, orientation, fixed_base, None
                    )
                    num_joints = p.getNumJoints(robot_id)
                    
        except Exception as e:
            # If KUKA fails to load and auto_fallback is enabled, try Panda
            if original_robot_type == RobotType.KUKA_IIWA and auto_fallback:
                logger.warning(f"Failed to load KUKA: {e}. Trying Franka Panda as fallback.")
                robot_type = RobotType.FRANKA_PANDA
                robot_id = self._load_robot_urdf(
                    robot_type, position, orientation, fixed_base, None
                )
                num_joints = p.getNumJoints(robot_id)
            else:
                raise
        
        # Get joint indices
        joint_indices = list(range(num_joints))

        # Get joint info
        joint_info = [p.getJointInfo(robot_id, i) for i in joint_indices]
        movable_joints = [
            i for i, info in enumerate(joint_info)
            if info[2] != p.JOINT_FIXED
        ]

        # Determine end effector and gripper indices based on robot type
        end_effector_index, gripper_indices = self._get_robot_specific_indices(
            robot_type, num_joints
        )

        # Create robot info
        robot_name = robot_name or f"{robot_type.value}_{len(self.robots)}"
        robot_info = RobotInfo(
            robot_id=robot_id,
            name=robot_name,
            robot_type=robot_type,
            base_position=position,
            base_orientation=orientation,
            num_joints=num_joints,
            joint_indices=movable_joints,
            end_effector_index=end_effector_index,
            gripper_indices=gripper_indices,
            tool_link_index=None,  # Will be set after storing robot
        )

        # Store robot info
        self.robots[robot_name] = robot_info
        
        # Detect tool link after robot is stored
        tool_link = self.get_tool_link_index(robot_name)
        robot_info.tool_link_index = tool_link

        logger.info(
            f"Loaded {robot_type.value} robot: {robot_name} "
            f"(id={robot_id}, joints={num_joints}, fixed_base={fixed_base}, "
            f"tool_link={tool_link})"
        )

        # Reset robot to home position
        self.reset_robot(robot_name)

        return robot_info

    def _load_robot_urdf(
        self,
        robot_type: RobotType,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        fixed_base: bool,
        urdf_path: Optional[str] = None,
    ) -> int:
        """
        Load robot URDF based on type.
        
        Note: This method tries multiple URDF paths for each robot type
        to handle different PyBullet data installations.
        """
        # Define possible URDF paths for each robot type
        robot_urdf_paths = {
            RobotType.KUKA_IIWA: [
                "kuka_iiwa/model.urdf",
                "kuka_iiwa/kuka_iiwa.urdf", 
                "kuka_iiwa/iiwa14.urdf",
                "kuka_iiwa/iiwa7.urdf",
                "kuka/kuka_iiwa.urdf",
            ],
            RobotType.FRANKA_PANDA: [
                "franka_panda/panda.urdf",
                "panda/panda.urdf",
                "franka/panda.urdf",
                "franka_emika_panda/panda.urdf",
            ],
            RobotType.UR5: [
                "ur5/ur5.urdf",
                "universal_robot/ur5.urdf",
                "ur/ur5.urdf",
                "ur5/ur5_robot.urdf",
            ],
        }
        
        if robot_type == RobotType.CUSTOM:
            # Custom robot from provided URDF
            if not urdf_path:
                raise ValueError("URDF path must be provided for custom robot")
            
            # Check if it's an absolute path or relative path
            urdf_file = Path(urdf_path)
            if not urdf_file.is_absolute():
                # Try to find it in PyBullet data path
                import pybullet_data
                urdf_file = Path(pybullet_data.getDataPath()) / urdf_path
            
            if not urdf_file.exists():
                raise ValueError(f"URDF file not found: {urdf_path}")
            
            try:
                robot_id = p.loadURDF(
                    str(urdf_file),
                    basePosition=position,
                    baseOrientation=orientation,
                    useFixedBase=fixed_base,
                )
                logger.info(f"Loaded custom robot from: {urdf_file}")
                return robot_id
            except Exception as e:
                raise ValueError(f"Failed to load custom robot URDF: {e}")
        
        # Try multiple paths for standard robots
        if robot_type in robot_urdf_paths:
            paths_to_try = robot_urdf_paths[robot_type]
            last_error = None
            
            for urdf_path in paths_to_try:
                try:
                    robot_id = p.loadURDF(
                        urdf_path,
                        basePosition=position,
                        baseOrientation=orientation,
                        useFixedBase=fixed_base,
                    )
                    logger.info(f"Successfully loaded {robot_type.value} from: {urdf_path}")
                    return robot_id
                except Exception as e:
                    last_error = e
                    logger.debug(f"Failed to load {urdf_path}: {e}")
                    continue
            
            # If all paths failed, raise the last error
            raise RuntimeError(
                f"Failed to load {robot_type.value} robot. "
                f"Tried paths: {paths_to_try}. Last error: {last_error}"
            )
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        return robot_id

    def _get_robot_specific_indices(
        self, robot_type: RobotType, num_joints: int
    ) -> Tuple[Optional[int], Optional[List[int]]]:
        """Get robot-specific end effector and gripper indices."""
        if robot_type == RobotType.KUKA_IIWA:
            # KUKA iiwa typically has 7 DOF + gripper
            end_effector_index = 6  # Last link before gripper
            gripper_indices = [7, 8] if num_joints > 7 else None
        elif robot_type == RobotType.FRANKA_PANDA:
            # Panda has 7 DOF arm + 2 finger joints
            end_effector_index = 8  # Panda hand link
            gripper_indices = [9, 10] if num_joints > 9 else None
        elif robot_type == RobotType.UR5:
            # UR5 has 6 DOF
            end_effector_index = 5
            gripper_indices = None  # UR5 doesn't include gripper in standard URDF
        else:
            # For custom robots, assume last joint is end effector
            end_effector_index = num_joints - 1 if num_joints > 0 else None
            gripper_indices = None

        return end_effector_index, gripper_indices

    def get_tool_link_index(self, robot_name: str) -> Optional[int]:
        """
        Get the tool/gripper link index for a robot.
        
        This identifies the link used for tool attachment or grasping.
        For robots with grippers, this is typically the link between the fingers.
        For robots without grippers, this is the end-effector link.
        
        Args:
            robot_name: Name of the robot.
            
        Returns:
            Tool link index, or None if not found.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        robot_id = robot.robot_id
        
        # Try to detect tool link based on robot type
        if robot.robot_type == RobotType.KUKA_IIWA:
            # For KUKA with gripper, tool link is typically link 11 or last link
            # Check for common gripper link names
            tool_link = self._find_link_by_name(robot_id, 
                ["tool_link", "iiwa_link_ee", "iiwa_link_7", "gripper_link"])
            if tool_link is not None:
                return tool_link
            # Default to link after end effector if gripper exists
            if robot.gripper_indices:
                return robot.end_effector_index + 1 if robot.end_effector_index else None
            return robot.end_effector_index
            
        elif robot.robot_type == RobotType.FRANKA_PANDA:
            # For Panda, tool link is between the fingers (link 11)
            tool_link = self._find_link_by_name(robot_id,
                ["panda_grasptarget", "panda_hand", "tool_link"])
            if tool_link is not None:
                return tool_link
            # Panda tool link is typically at index 11
            if robot.num_joints >= 11:
                return 11
            return robot.end_effector_index
            
        elif robot.robot_type == RobotType.UR5:
            # For UR5, tool link is at the wrist
            tool_link = self._find_link_by_name(robot_id,
                ["tool0", "ee_link", "wrist_3_link", "tool_link"])
            if tool_link is not None:
                return tool_link
            return robot.end_effector_index
            
        else:
            # For custom robots, try common naming conventions
            tool_link = self._find_link_by_name(robot_id,
                ["tool_link", "tool0", "gripper_link", "ee_link", 
                 "end_effector", "grasptarget"])
            if tool_link is not None:
                return tool_link
            return robot.end_effector_index
    
    def _find_link_by_name(self, robot_id: int, names: List[str]) -> Optional[int]:
        """
        Find a link index by trying multiple possible names.
        
        Args:
            robot_id: Robot body ID.
            names: List of possible link names to search for.
            
        Returns:
            Link index if found, None otherwise.
        """
        num_joints = p.getNumJoints(robot_id)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            link_name = joint_info[12].decode('utf-8') if joint_info[12] else ""
            
            # Check if link name matches any of the target names
            for name in names:
                if name.lower() in link_name.lower():
                    logger.debug(f"Found link '{link_name}' at index {i}")
                    return i
        
        return None
    
    def get_gripper_info(self, robot_name: str) -> Dict[str, Any]:
        """
        Get detailed gripper information for a robot.
        
        Args:
            robot_name: Name of the robot.
            
        Returns:
            Dictionary with gripper information including:
            - has_gripper: Whether robot has a gripper
            - gripper_indices: Joint indices for gripper fingers
            - tool_link: Tool/grasp point link index
            - finger_names: Names of gripper finger joints
            - current_opening: Current gripper opening width
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        robot_id = robot.robot_id
        
        info = {
            "has_gripper": robot.gripper_indices is not None,
            "gripper_indices": robot.gripper_indices,
            "tool_link": self.get_tool_link_index(robot_name),
            "finger_names": [],
            "current_opening": None,
        }
        
        # Get gripper joint names and current positions
        if robot.gripper_indices:
            finger_positions = []
            for idx in robot.gripper_indices:
                joint_info = p.getJointInfo(robot_id, idx)
                joint_name = joint_info[12].decode('utf-8') if joint_info[12] else f"finger_{idx}"
                info["finger_names"].append(joint_name)
                
                # Get current position
                joint_state = p.getJointState(robot_id, idx)
                finger_positions.append(joint_state[0])
            
            # Calculate approximate gripper opening (sum of finger positions)
            info["current_opening"] = sum(abs(pos) for pos in finger_positions)
        
        return info
    
    def set_gripper(self, robot_name: str, opening: float, force: float = 100.0):
        """
        Set gripper opening for a robot.
        
        Args:
            robot_name: Name of the robot.
            opening: Target opening (0.0 = closed, 1.0 = fully open).
            force: Maximum force to apply.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        if not robot.gripper_indices:
            logger.warning(f"Robot '{robot_name}' has no gripper")
            return
        
        robot_id = robot.robot_id
        
        # Set gripper joint positions based on robot type
        if robot.robot_type == RobotType.FRANKA_PANDA:
            # Panda fingers move symmetrically
            target_pos = opening * 0.04  # Max opening ~4cm
            for idx in robot.gripper_indices:
                p.setJointMotorControl2(
                    robot_id, idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=force
                )
        elif robot.robot_type == RobotType.KUKA_IIWA:
            # KUKA gripper (if attached)
            target_pos = opening * 0.05  # Max opening ~5cm
            for idx in robot.gripper_indices:
                p.setJointMotorControl2(
                    robot_id, idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=force
                )
        else:
            # Generic gripper control
            for idx in robot.gripper_indices:
                p.setJointMotorControl2(
                    robot_id, idx,
                    p.POSITION_CONTROL,
                    targetPosition=opening,
                    force=force
                )
        
        logger.info(f"Set gripper for '{robot_name}' to {opening:.2f} (force={force}N)")
    
    def open_gripper(self, robot_name: str, force: float = 100.0):
        """
        Fully open the gripper.
        
        This is a convenience method that sets the gripper to its maximum opening.
        
        Args:
            robot_name: Name of the robot.
            force: Maximum force to apply (default: 100.0N).
        
        Raises:
            ValueError: If robot not found.
        """
        self.set_gripper(robot_name, opening=1.0, force=force)
        logger.info(f"Opened gripper for '{robot_name}'")
    
    def close_gripper(self, robot_name: str, force: float = 50.0):
        """
        Fully close the gripper.
        
        This is a convenience method that sets the gripper to its minimum opening.
        The default force is lower than for opening to allow for gentle grasping.
        
        Args:
            robot_name: Name of the robot.
            force: Maximum force to apply (default: 50.0N for gentler closing).
        
        Raises:
            ValueError: If robot not found.
        """
        self.set_gripper(robot_name, opening=0.0, force=force)
        logger.info(f"Closed gripper for '{robot_name}'")
    
    def get_link_names(self, robot_name: str) -> Dict[int, str]:
        """
        Get all link names for a robot.
        
        Args:
            robot_name: Name of the robot.
            
        Returns:
            Dictionary mapping link indices to names.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        robot_id = robot.robot_id
        link_names = {}
        
        # Base link
        base_info = p.getBodyInfo(robot_id)
        base_name = base_info[0].decode('utf-8') if base_info[0] else "base"
        link_names[-1] = base_name  # Base link has index -1
        
        # Other links
        for i in range(robot.num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            link_name = joint_info[12].decode('utf-8') if joint_info[12] else f"link_{i}"
            link_names[i] = link_name
        
        return link_names
    
    def get_tool_pose(self, robot_name: str) -> Optional[Tuple[Tuple[float, float, float], 
                                                                Tuple[float, float, float, float]]]:
        """
        Get the current pose of the tool/gripper link.
        
        Args:
            robot_name: Name of the robot.
            
        Returns:
            Tuple of (position, orientation) or None if no tool link.
        """
        tool_link = self.get_tool_link_index(robot_name)
        
        if tool_link is None:
            logger.warning(f"No tool link found for robot '{robot_name}'")
            return None
        
        robot = self.robots[robot_name]
        link_state = p.getLinkState(robot.robot_id, tool_link)
        
        return link_state[0], link_state[1]  # Position and orientation
    
    def ee_pose(self, robot_name: str) -> Tuple[Tuple[float, float, float], 
                                                 Tuple[float, float, float, float]]:
        """
        Get end-effector pose (position and orientation) via tool_link_index.
        
        This is a convenience method that returns the pose of the tool link,
        which represents the end-effector or gripper attachment point.
        
        Args:
            robot_name: Name of the robot.
            
        Returns:
            Tuple of (pos, orn) where:
            - pos: (x, y, z) position in world coordinates
            - orn: (x, y, z, w) quaternion orientation
            
        Raises:
            ValueError: If robot not found.
            RuntimeError: If tool link cannot be determined.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        # Use stored tool_link_index if available
        if robot.tool_link_index is not None:
            tool_link = robot.tool_link_index
        else:
            # Try to detect it
            tool_link = self.get_tool_link_index(robot_name)
            if tool_link is None:
                # Fall back to end_effector_index
                tool_link = robot.end_effector_index
                if tool_link is None:
                    raise RuntimeError(
                        f"Cannot determine tool link for robot '{robot_name}'. "
                        "No tool_link_index or end_effector_index found."
                    )
        
        # Get link state using PyBullet
        link_state = p.getLinkState(robot.robot_id, tool_link)
        
        # Extract position and orientation
        pos = link_state[0]  # World position (x, y, z)
        orn = link_state[1]  # World orientation quaternion (x, y, z, w)
        
        return pos, orn
    
    def get_ee_velocity(self, robot_name: str) -> Tuple[Tuple[float, float, float],
                                                        Tuple[float, float, float]]:
        """
        Get end-effector linear and angular velocity.
        
        Args:
            robot_name: Name of the robot.
            
        Returns:
            Tuple of (linear_velocity, angular_velocity) where each is (x, y, z).
            
        Raises:
            ValueError: If robot not found.
            RuntimeError: If tool link cannot be determined.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        # Determine tool link
        tool_link = robot.tool_link_index
        if tool_link is None:
            tool_link = robot.end_effector_index
            if tool_link is None:
                raise RuntimeError(f"Cannot determine tool link for robot '{robot_name}'")
        
        # Get link state with velocities (computeLinkVelocity=1)
        link_state = p.getLinkState(robot.robot_id, tool_link, computeLinkVelocity=1)
        
        # Extract velocities
        linear_velocity = link_state[6]   # World linear velocity (x, y, z)
        angular_velocity = link_state[7]  # World angular velocity (x, y, z)
        
        return linear_velocity, angular_velocity
    
    def get_ee_jacobian(self, robot_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Jacobian matrix for the end-effector.
        
        The Jacobian relates joint velocities to end-effector velocities.
        
        Args:
            robot_name: Name of the robot.
            
        Returns:
            Tuple of (linear_jacobian, angular_jacobian) as numpy arrays.
            Each is shape (3, num_joints).
            
        Raises:
            ValueError: If robot not found.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        # Get current joint positions
        joint_states = [p.getJointState(robot.robot_id, i)[0] for i in robot.joint_indices]
        
        # Determine tool link
        tool_link = robot.tool_link_index
        if tool_link is None:
            tool_link = robot.end_effector_index
            if tool_link is None:
                tool_link = robot.joint_indices[-1] if robot.joint_indices else 0
        
        # Calculate Jacobian
        zero_vec = [0.0] * len(robot.joint_indices)
        jac_t, jac_r = p.calculateJacobian(
            robot.robot_id,
            tool_link,
            [0, 0, 0],  # Local position on link
            joint_states,
            zero_vec,  # Joint velocities (not used)
            zero_vec   # Joint accelerations (not used)
        )
        
        # Convert to numpy arrays
        linear_jacobian = np.array(jac_t)
        angular_jacobian = np.array(jac_r)
        
        return linear_jacobian, angular_jacobian
    
    def set_ee_pose(self, robot_name: str, target_pos: Tuple[float, float, float],
                    target_orn: Optional[Tuple[float, float, float, float]] = None,
                    max_iterations: int = 100,
                    position_tolerance: float = 0.001,
                    orientation_tolerance: float = 0.01) -> bool:
        """
        Set end-effector pose using inverse kinematics.
        
        Args:
            robot_name: Name of the robot.
            target_pos: Target position (x, y, z).
            target_orn: Target orientation quaternion (x, y, z, w). 
                       If None, orientation is not constrained.
            max_iterations: Maximum IK iterations.
            position_tolerance: Position error tolerance in meters.
            orientation_tolerance: Orientation error tolerance in radians.
            
        Returns:
            True if target pose was reached within tolerance.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        # Determine tool link
        tool_link = robot.tool_link_index
        if tool_link is None:
            tool_link = robot.end_effector_index
            if tool_link is None:
                raise RuntimeError(f"Cannot determine tool link for robot '{robot_name}'")
        
        # Compute IK
        if target_orn is not None:
            joint_positions = p.calculateInverseKinematics(
                robot.robot_id,
                tool_link,
                target_pos,
                target_orn,
                maxNumIterations=max_iterations,
                residualThreshold=position_tolerance
            )
        else:
            joint_positions = p.calculateInverseKinematics(
                robot.robot_id,
                tool_link,
                target_pos,
                maxNumIterations=max_iterations,
                residualThreshold=position_tolerance
            )
        
        # Set joint positions
        for i, joint_idx in enumerate(robot.joint_indices):
            if i < len(joint_positions):
                p.resetJointState(robot.robot_id, joint_idx, joint_positions[i])
        
        # Check if target was reached
        current_pos, current_orn = self.ee_pose(robot_name)
        
        # Calculate position error
        pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        # Calculate orientation error if specified
        if target_orn is not None:
            # Quaternion difference
            orn_error = 1.0 - abs(np.dot(current_orn, target_orn))
        else:
            orn_error = 0.0
        
        success = (pos_error < position_tolerance and 
                  orn_error < orientation_tolerance)
        
        if success:
            logger.info(f"EE pose set successfully for '{robot_name}' "
                       f"(pos_error={pos_error:.4f}m, orn_error={orn_error:.4f})")
        else:
            logger.warning(f"EE pose not reached for '{robot_name}' "
                          f"(pos_error={pos_error:.4f}m, orn_error={orn_error:.4f})")
        
        return success
    
    def calculate_ik(self, robot_name: str, target_pos: Tuple[float, float, float],
                     target_orn: Optional[Tuple[float, float, float, float]] = None,
                     current_q: Optional[List[float]] = None,
                     max_iterations: int = 100,
                     residual_threshold: float = 0.001,
                     joint_damping: Optional[List[float]] = None,
                     solver: int = 0,
                     use_nullspace: bool = False,
                     joint_lower_limits: Optional[List[float]] = None,
                     joint_upper_limits: Optional[List[float]] = None,
                     joint_ranges: Optional[List[float]] = None,
                     rest_poses: Optional[List[float]] = None) -> List[float]:
        """
        Calculate inverse kinematics to find joint positions for target end-effector pose.
        
        This method uses PyBullet's calculateInverseKinematics to compute joint angles
        that would place the end-effector at the desired position and orientation.
        
        Args:
            robot_name: Name of the robot.
            target_pos: Target position (x, y, z) for the end-effector.
            target_orn: Target orientation quaternion (x, y, z, w). 
                       If None, only position is considered.
            current_q: Current joint positions to use as starting point.
                      If None, uses current robot state.
            max_iterations: Maximum iterations for IK solver (default: 100).
            residual_threshold: Convergence threshold for solver (default: 0.001).
            joint_damping: Damping coefficients for each joint (helps stability).
                          If None, uses default damping.
            solver: IK solver type (0: default, 1: DLS - Damped Least Squares).
            use_nullspace: If True, uses null-space control with joint limits.
            joint_lower_limits: Lower limits for joints (required if use_nullspace=True).
            joint_upper_limits: Upper limits for joints (required if use_nullspace=True).
            joint_ranges: Joint ranges for null-space (required if use_nullspace=True).
            rest_poses: Rest poses for null-space (required if use_nullspace=True).
            
        Returns:
            List of joint positions that achieve the target pose.
            
        Raises:
            ValueError: If robot not found or invalid parameters.
            RuntimeError: If tool link cannot be determined.
            
        Example:
            # Simple IK for position only
            q = sim.calculate_ik("robot", target_pos=(0.5, 0.3, 0.4))
            
            # IK with orientation
            q = sim.calculate_ik("robot", 
                                target_pos=(0.5, 0.3, 0.4),
                                target_orn=(0, 0, 0.707, 0.707))
            
            # IK with joint damping for stability
            q = sim.calculate_ik("robot",
                                target_pos=(0.5, 0.3, 0.4),
                                joint_damping=[0.1]*7)
            
            # IK with null-space control
            q = sim.calculate_ik("robot",
                                target_pos=(0.5, 0.3, 0.4),
                                use_nullspace=True,
                                joint_lower_limits=[-2.96]*7,
                                joint_upper_limits=[2.96]*7,
                                joint_ranges=[5.92]*7,
                                rest_poses=[0]*7)
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        # Determine tool link for IK target
        tool_link = robot.tool_link_index
        if tool_link is None:
            tool_link = robot.end_effector_index
            if tool_link is None:
                raise RuntimeError(
                    f"Cannot determine tool link for robot '{robot_name}'. "
                    "No tool_link_index or end_effector_index found."
                )
        
        # Get current joint positions if not provided
        if current_q is not None:
            if len(current_q) != len(robot.joint_indices):
                raise ValueError(
                    f"current_q has {len(current_q)} values but robot has "
                    f"{len(robot.joint_indices)} controllable joints"
                )
            # Set the robot to the specified configuration for IK computation
            for i, joint_idx in enumerate(robot.joint_indices):
                p.resetJointState(robot.robot_id, joint_idx, current_q[i])
        
        # Setup joint damping if not provided
        if joint_damping is None:
            # Default damping values for stability
            joint_damping = [0.01] * len(robot.joint_indices)
        elif len(joint_damping) != len(robot.joint_indices):
            raise ValueError(
                f"joint_damping has {len(joint_damping)} values but robot has "
                f"{len(robot.joint_indices)} controllable joints"
            )
        
        # Validate null-space parameters if requested
        if use_nullspace:
            if joint_lower_limits is None or joint_upper_limits is None:
                # Try to get limits from robot joints
                joint_lower_limits = []
                joint_upper_limits = []
                for joint_idx in robot.joint_indices:
                    joint_info = p.getJointInfo(robot.robot_id, joint_idx)
                    joint_lower_limits.append(joint_info[8])  # Lower limit
                    joint_upper_limits.append(joint_info[9])  # Upper limit
                    
            if joint_ranges is None:
                joint_ranges = [
                    upper - lower 
                    for lower, upper in zip(joint_lower_limits, joint_upper_limits)
                ]
                
            if rest_poses is None:
                # Use middle of joint range as rest pose
                rest_poses = [
                    (lower + upper) / 2.0 
                    for lower, upper in zip(joint_lower_limits, joint_upper_limits)
                ]
            
            # Validate lengths
            n_joints = len(robot.joint_indices)
            if len(joint_lower_limits) != n_joints:
                raise ValueError(f"joint_lower_limits must have {n_joints} values")
            if len(joint_upper_limits) != n_joints:
                raise ValueError(f"joint_upper_limits must have {n_joints} values")
            if len(joint_ranges) != n_joints:
                raise ValueError(f"joint_ranges must have {n_joints} values")
            if len(rest_poses) != n_joints:
                raise ValueError(f"rest_poses must have {n_joints} values")
        
        # Calculate inverse kinematics
        try:
            if use_nullspace:
                # IK with null-space control
                if target_orn is not None:
                    joint_positions = p.calculateInverseKinematics(
                        robot.robot_id,
                        tool_link,
                        target_pos,
                        target_orn,
                        lowerLimits=joint_lower_limits,
                        upperLimits=joint_upper_limits,
                        jointRanges=joint_ranges,
                        restPoses=rest_poses,
                        maxNumIterations=max_iterations,
                        residualThreshold=residual_threshold,
                        jointDamping=joint_damping,
                        solver=solver
                    )
                else:
                    joint_positions = p.calculateInverseKinematics(
                        robot.robot_id,
                        tool_link,
                        target_pos,
                        lowerLimits=joint_lower_limits,
                        upperLimits=joint_upper_limits,
                        jointRanges=joint_ranges,
                        restPoses=rest_poses,
                        maxNumIterations=max_iterations,
                        residualThreshold=residual_threshold,
                        jointDamping=joint_damping,
                        solver=solver
                    )
            else:
                # Standard IK
                if target_orn is not None:
                    joint_positions = p.calculateInverseKinematics(
                        robot.robot_id,
                        tool_link,
                        target_pos,
                        target_orn,
                        maxNumIterations=max_iterations,
                        residualThreshold=residual_threshold,
                        jointDamping=joint_damping,
                        solver=solver
                    )
                else:
                    joint_positions = p.calculateInverseKinematics(
                        robot.robot_id,
                        tool_link,
                        target_pos,
                        maxNumIterations=max_iterations,
                        residualThreshold=residual_threshold,
                        jointDamping=joint_damping,
                        solver=solver
                    )
            
            # Extract only the joint positions for the controllable joints
            result = list(joint_positions[:len(robot.joint_indices)])
            
            logger.debug(f"IK calculated for '{robot_name}': target_pos={target_pos}, "
                        f"target_orn={target_orn}, result={result}")
            
            return result
            
        except Exception as e:
            logger.error(f"IK calculation failed for '{robot_name}': {e}")
            raise RuntimeError(f"Failed to calculate IK: {e}")
    
    def move_through_waypoints(self, robot_name: str, 
                               waypoints: List[Union[Tuple[float, float, float],
                                                    Dict[str, Any]]],
                               steps_per_segment: int = 60,
                               action_repeat: int = 4,
                               use_orientation: bool = False,
                               max_velocity: float = 1.0,
                               force: float = 100.0,
                               gripper_actions: Optional[Dict[int, str]] = None,
                               return_trajectories: bool = False) -> Optional[Dict[str, List]]:
        """
        Move robot through a sequence of waypoints using IK and position control.
        
        This method computes IK for each waypoint, interpolates between joint configurations,
        and uses position control to execute smooth motion through the waypoints.
        
        Args:
            robot_name: Name of the robot to move.
            waypoints: List of target waypoints. Each can be:
                      - Tuple (x, y, z) for position-only waypoint
                      - Dict with 'pos' and optionally 'orn' keys for full pose
            steps_per_segment: Number of simulation steps between waypoints (default: 60).
                              More steps = slower, smoother motion.
            action_repeat: How many times to repeat each control action (default: 4).
                          Higher values = more stable but slower execution.
            use_orientation: If True, uses orientation from waypoint dicts.
                           If False, only position is considered.
            max_velocity: Maximum joint velocity in rad/s (default: 1.0).
            force: Maximum force/torque for position control (default: 100.0).
            gripper_actions: Optional dict mapping waypoint indices to gripper actions.
                           Actions can be 'open', 'close', or float values.
                           Example: {0: 'open', 3: 'close', 5: 0.5}
            return_trajectories: If True, returns dict with joint and EE trajectories.
            
        Returns:
            If return_trajectories=True, returns dict with:
            - 'joint_trajectory': List of joint configurations
            - 'ee_positions': List of achieved end-effector positions
            - 'ee_errors': List of position errors at each waypoint
            Otherwise returns None.
            
        Raises:
            ValueError: If robot not found or waypoints are invalid.
            RuntimeError: If IK fails for any waypoint.
            
        Example:
            # Simple position waypoints
            waypoints = [
                (0.5, 0.0, 0.4),
                (0.5, 0.2, 0.4),
                (0.3, 0.2, 0.5),
            ]
            sim.move_through_waypoints("robot", waypoints)
            
            # Waypoints with orientation
            waypoints = [
                {'pos': (0.5, 0.0, 0.4), 'orn': (0, 0, 0, 1)},
                {'pos': (0.5, 0.2, 0.4), 'orn': (0, 0, 0.707, 0.707)},
            ]
            sim.move_through_waypoints("robot", waypoints, use_orientation=True)
            
            # With gripper control
            sim.move_through_waypoints("robot", waypoints,
                                      gripper_actions={0: 'open', 2: 'close'})
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        if not waypoints:
            raise ValueError("Waypoints list cannot be empty")
        
        robot = self.robots[robot_name]
        robot_id = robot.robot_id
        
        # Process waypoints into consistent format
        processed_waypoints = []
        for i, wp in enumerate(waypoints):
            if isinstance(wp, (tuple, list)) and len(wp) == 3:
                # Position-only waypoint
                processed_waypoints.append({
                    'pos': tuple(wp),
                    'orn': None
                })
            elif isinstance(wp, dict):
                if 'pos' not in wp:
                    raise ValueError(f"Waypoint {i} dict must have 'pos' key")
                processed_waypoints.append({
                    'pos': tuple(wp['pos']),
                    'orn': tuple(wp.get('orn')) if wp.get('orn') else None
                })
            else:
                raise ValueError(f"Waypoint {i} must be tuple (x,y,z) or dict with 'pos' key")
        
        # Compute IK for all waypoints
        logger.info(f"Computing IK for {len(processed_waypoints)} waypoints...")
        joint_targets = []
        ik_errors = []
        
        for i, wp in enumerate(processed_waypoints):
            try:
                # Calculate IK
                if use_orientation and wp['orn'] is not None:
                    q_target = self.calculate_ik(
                        robot_name,
                        target_pos=wp['pos'],
                        target_orn=wp['orn'],
                        current_q=joint_targets[-1] if joint_targets else None
                    )
                else:
                    q_target = self.calculate_ik(
                        robot_name,
                        target_pos=wp['pos'],
                        current_q=joint_targets[-1] if joint_targets else None
                    )
                
                joint_targets.append(q_target)
                
                # Verify IK solution (optional)
                logger.debug(f"Waypoint {i}: IK solution = {[f'{q:.3f}' for q in q_target[:7]]}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to compute IK for waypoint {i}: {e}")
        
        # Initialize trajectory storage if requested
        if return_trajectories:
            trajectory_data = {
                'joint_trajectory': [],
                'ee_positions': [],
                'ee_errors': []
            }
        
        # Get current joint positions
        current_q = self.get_joint_positions(robot_name, include_gripper=False)
        
        # Add current position as first target for smooth start
        joint_targets.insert(0, current_q)
        
        logger.info(f"Executing trajectory through {len(processed_waypoints)} waypoints...")
        
        # Execute trajectory
        for segment_idx in range(len(joint_targets) - 1):
            q_start = joint_targets[segment_idx]
            q_end = joint_targets[segment_idx + 1]
            
            # Handle gripper actions if specified
            if gripper_actions and segment_idx in gripper_actions:
                action = gripper_actions[segment_idx]
                if action == 'open':
                    self.open_gripper(robot_name, force=force)
                elif action == 'close':
                    self.close_gripper(robot_name, force=force/2)  # Gentler closing
                elif isinstance(action, (int, float)):
                    self.set_gripper(robot_name, opening=float(action), force=force)
                logger.debug(f"Gripper action at waypoint {segment_idx}: {action}")
            
            # Interpolate between joint configurations
            num_substeps = steps_per_segment // action_repeat
            
            for substep in range(num_substeps):
                # Linear interpolation
                alpha = (substep + 1) / num_substeps
                q_interpolated = [
                    q_start[j] + alpha * (q_end[j] - q_start[j])
                    for j in range(len(q_start))
                ]
                
                # Apply position control
                for joint_idx, joint_pos in zip(robot.joint_indices[:len(q_interpolated)], 
                                               q_interpolated):
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_pos,
                        targetVelocity=0,
                        force=force,
                        maxVelocity=max_velocity
                    )
                
                # Step simulation with action repeat
                for _ in range(action_repeat):
                    self.step()
                
                # Record trajectory if requested
                if return_trajectories:
                    trajectory_data['joint_trajectory'].append(q_interpolated.copy())
                    ee_pos, _ = self.ee_pose(robot_name)
                    trajectory_data['ee_positions'].append(ee_pos)
            
            # Check accuracy at waypoint (excluding the initial position)
            if segment_idx > 0:
                achieved_pos, achieved_orn = self.ee_pose(robot_name)
                target_wp = processed_waypoints[segment_idx - 1]
                
                pos_error = np.linalg.norm(
                    np.array(achieved_pos) - np.array(target_wp['pos'])
                )
                
                if return_trajectories:
                    trajectory_data['ee_errors'].append(pos_error)
                
                if pos_error > 0.01:  # 1cm threshold
                    logger.warning(
                        f"Waypoint {segment_idx-1}: Position error = {pos_error:.4f}m"
                    )
                else:
                    logger.debug(
                        f"Waypoint {segment_idx-1}: Reached successfully (error = {pos_error:.4f}m)"
                    )
        
        # Final gripper action if specified
        last_idx = len(processed_waypoints)
        if gripper_actions and last_idx in gripper_actions:
            action = gripper_actions[last_idx]
            if action == 'open':
                self.open_gripper(robot_name, force=force)
            elif action == 'close':
                self.close_gripper(robot_name, force=force/2)
            elif isinstance(action, (int, float)):
                self.set_gripper(robot_name, opening=float(action), force=force)
        
        logger.info(f"Trajectory execution complete for '{robot_name}'")
        
        if return_trajectories:
            return trajectory_data
        return None

    def reset_robot(self, robot_name: str, q_default: Optional[List[float]] = None,
                   reset_gripper: bool = True) -> List[float]:
        """
        Reset robot joint states to default or specified positions.
        
        This method resets each controllable joint using p.resetJointState,
        which immediately sets joint positions without simulation.
        
        Args:
            robot_name: Name of the robot to reset.
            q_default: Target joint positions for controllable joints.
                      If None, uses robot-specific home position.
                      Must match the number of controllable joints.
            reset_gripper: Whether to reset gripper joints (if present).
        
        Returns:
            List of actual joint positions that were set.
            
        Raises:
            ValueError: If robot not found or q_default has wrong size.
        
        Example:
            # Reset to default home position
            sim.reset_robot("kuka")
            
            # Reset to specific joint configuration
            sim.reset_robot("kuka", q_default=[0, 0.5, 0, -1.5, 0, 1.0, 0])
            
            # Reset only arm joints, not gripper
            sim.reset_robot("panda", q_default=[0]*7, reset_gripper=False)
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")

        robot = self.robots[robot_name]
        robot_id = robot.robot_id
        
        # Determine which joints to reset
        joints_to_reset = robot.joint_indices.copy()
        
        # Optionally exclude gripper joints
        if not reset_gripper and robot.gripper_indices:
            joints_to_reset = [j for j in joints_to_reset 
                             if j not in robot.gripper_indices]
        
        # Get default positions if not provided
        if q_default is None:
            q_default = self._get_default_joint_positions(robot, len(joints_to_reset))
        else:
            # Validate input size
            if len(q_default) != len(joints_to_reset):
                raise ValueError(
                    f"q_default has {len(q_default)} values but robot '{robot_name}' "
                    f"has {len(joints_to_reset)} controllable joints "
                    f"(gripper={'excluded' if not reset_gripper else 'included'})"
                )
        
        # Reset each controllable joint
        actual_positions = []
        for i, joint_idx in enumerate(joints_to_reset):
            if i < len(q_default):
                target_pos = q_default[i]
            else:
                target_pos = 0.0  # Fallback to zero
            
            # Reset joint state using PyBullet
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                targetValue=target_pos,
                targetVelocity=0.0  # Also reset velocity to zero
            )
            actual_positions.append(target_pos)
        
        # Reset gripper if requested and available
        if reset_gripper and robot.gripper_indices:
            gripper_pos = self._get_default_gripper_position(robot.robot_type)
            for idx in robot.gripper_indices:
                p.resetJointState(robot_id, idx, gripper_pos, targetVelocity=0.0)
                actual_positions.append(gripper_pos)
        
        logger.info(
            f"Reset robot '{robot_name}': {len(joints_to_reset)} joints to "
            f"positions {[f'{p:.3f}' for p in actual_positions[:7]]}"  # Show first 7 for brevity
        )
        
        return actual_positions
    
    def _get_default_joint_positions(self, robot: RobotInfo, num_joints: int) -> List[float]:
        """
        Get default joint positions for a specific robot type.
        
        Args:
            robot: Robot information.
            num_joints: Number of joints to get positions for.
            
        Returns:
            List of default joint positions.
        """
        # Robot-specific home positions (comfortable/safe configurations)
        if robot.robot_type == RobotType.KUKA_IIWA:
            # KUKA iiwa 7-DOF home position
            defaults = [0, 0.5, 0, -1.5, 0, 1.0, 0]
        elif robot.robot_type == RobotType.FRANKA_PANDA:
            # Franka Panda 7-DOF home position
            defaults = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        elif robot.robot_type == RobotType.UR5:
            # UR5 6-DOF home position
            defaults = [0, -1.57, 1.57, -1.57, -1.57, 0]
        else:
            # Generic robot: all zeros
            defaults = [0] * num_joints
        
        # Trim or extend to match requested number of joints
        if len(defaults) > num_joints:
            return defaults[:num_joints]
        elif len(defaults) < num_joints:
            # Pad with zeros if needed
            return defaults + [0] * (num_joints - len(defaults))
        return defaults
    
    def _get_default_gripper_position(self, robot_type: RobotType) -> float:
        """
        Get default gripper position for a robot type.
        
        Args:
            robot_type: Type of robot.
            
        Returns:
            Default gripper joint position.
        """
        if robot_type == RobotType.FRANKA_PANDA:
            return 0.04  # Panda gripper open position
        elif robot_type == RobotType.KUKA_IIWA:
            return 0.0  # KUKA gripper closed by default
        else:
            return 0.0  # Default closed
    
    def reset_robot_velocity(self, robot_name: str, joint_velocities: Optional[List[float]] = None):
        """
        Reset robot joint velocities.
        
        Args:
            robot_name: Name of the robot.
            joint_velocities: Target velocities. If None, sets all to zero.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        if joint_velocities is None:
            joint_velocities = [0.0] * len(robot.joint_indices)
        
        # Reset velocities for each joint
        for i, joint_idx in enumerate(robot.joint_indices):
            if i < len(joint_velocities):
                # Get current position to maintain it
                current_state = p.getJointState(robot.robot_id, joint_idx)
                current_pos = current_state[0]
                
                # Reset with same position but new velocity
                p.resetJointState(
                    robot.robot_id,
                    joint_idx,
                    targetValue=current_pos,
                    targetVelocity=joint_velocities[i]
                )
    
    def get_joint_positions(self, robot_name: str, include_gripper: bool = True) -> List[float]:
        """
        Get current joint positions for a robot.
        
        Args:
            robot_name: Name of the robot.
            include_gripper: Whether to include gripper joint positions.
            
        Returns:
            List of current joint positions.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        positions = []
        
        # Get positions of controllable joints
        for idx in robot.joint_indices:
            # Skip gripper joints if requested
            if not include_gripper and robot.gripper_indices and idx in robot.gripper_indices:
                continue
            state = p.getJointState(robot.robot_id, idx)
            positions.append(state[0])
        
        return positions
    
    def set_joint_positions(self, robot_name: str, joint_positions: List[float],
                           use_physics: bool = False, max_force: float = 100.0):
        """
        Set robot joint positions.
        
        Args:
            robot_name: Name of the robot.
            joint_positions: Target joint positions.
            use_physics: If True, uses position control (smooth). 
                        If False, uses resetJointState (instant).
            max_force: Maximum force for position control (if use_physics=True).
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        if use_physics:
            # Use position control (smooth motion)
            for i, joint_idx in enumerate(robot.joint_indices):
                if i < len(joint_positions):
                    p.setJointMotorControl2(
                        robot.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=joint_positions[i],
                        force=max_force
                    )
        else:
            # Use reset (instant)
            for i, joint_idx in enumerate(robot.joint_indices):
                if i < len(joint_positions):
                    p.resetJointState(robot.robot_id, joint_idx, joint_positions[i])

    def get_robot_state(self, robot_name: str) -> Dict[str, Any]:
        """
        Get current state of a robot.

        Args:
            robot_name: Name of the robot.

        Returns:
            Dictionary containing joint positions, velocities, and end effector pose.
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")

        robot = self.robots[robot_name]

        # Get joint states
        joint_states = [
            p.getJointState(robot.robot_id, i) for i in robot.joint_indices
        ]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Get end effector state if available
        ee_state = None
        if robot.end_effector_index is not None:
            ee_state = p.getLinkState(robot.robot_id, robot.end_effector_index)
            ee_position = ee_state[0]
            ee_orientation = ee_state[1]
        else:
            ee_position = None
            ee_orientation = None

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_position": ee_position,
            "end_effector_orientation": ee_orientation,
        }

    def spawn_block(
        self,
        color_rgb: Tuple[float, float, float],
        size: float = 0.03,
        position: Optional[Tuple[float, float, float]] = None,
        mass: float = 0.1,
        block_name: Optional[str] = None,
    ) -> int:
        """
        Spawn a colored cube block in the simulation.

        Args:
            color_rgb: RGB color values (0.0 to 1.0) for the block.
            size: Half-extent of the cube (default 0.03 for 6cm cube).
            position: Initial position (x, y, z). Default is (0.5, 0.0, 0.05).
            mass: Mass of the block in kg (default 0.1).
            block_name: Optional name for the block. Auto-generated if None.

        Returns:
            Body ID of the spawned block.

        Raises:
            RuntimeError: If not connected to PyBullet.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")

        # Default position above ground
        if position is None:
            position = (0.5, 0.0, 0.05)

        # Create collision shape (box)
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size, size, size],
        )

        # Create visual shape (box) with color
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size, size, size],
            rgbaColor=[*color_rgb, 1.0],  # Add alpha channel
        )

        # Create the multi-body (the actual block)
        block_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )

        # Set additional dynamics properties
        p.changeDynamics(
            block_id,
            -1,  # Base link
            lateralFriction=0.5,
            spinningFriction=0.001,
            rollingFriction=0.001,
            restitution=0.4,  # Bounciness
        )

        # Generate block name if not provided
        if block_name is None:
            block_name = f"block_{len(self.objects)}"

        # Store block in objects dictionary
        self.objects[block_name] = block_id

        logger.info(
            f"Spawned block '{block_name}' (id={block_id}) at position {position} "
            f"with color RGB({color_rgb[0]:.2f}, {color_rgb[1]:.2f}, {color_rgb[2]:.2f})"
        )

        return block_id

    def spawn_platform(
        self,
        color_rgb: Tuple[float, float, float],
        size: float = 0.1,
        position: Optional[Tuple[float, float, float]] = None,
        platform_name: Optional[str] = None,
        height: Optional[float] = None,
    ) -> int:
        """
        Spawn a static platform (box) in the simulation.

        Args:
            color_rgb: RGB color values (0.0 to 1.0) for the platform.
            size: Half-extent for width and depth. Height is size/4 by default.
            position: Platform center position (x, y, z). Default is (0.6, 0.2, 0.05).
            platform_name: Optional name for the platform. Auto-generated if None.
            height: Optional custom height (half-extent). Default is size/4.

        Returns:
            Body ID of the spawned platform.

        Raises:
            RuntimeError: If not connected to PyBullet.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")

        # Default position
        if position is None:
            position = (0.6, 0.2, 0.05)

        # Platform height (thinner than width/depth)
        if height is None:
            height = size / 4  # Make platform relatively flat

        # Create collision shape (box with different dimensions)
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size, size, height],
        )

        # Create visual shape (box) with color
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size, size, height],
            rgbaColor=[*color_rgb, 1.0],  # Add alpha channel
        )

        # Create the multi-body as static (mass=0)
        platform_id = p.createMultiBody(
            baseMass=0,  # Mass = 0 makes it static/immovable
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )

        # Set friction properties for the platform
        p.changeDynamics(
            platform_id,
            -1,  # Base link
            lateralFriction=0.8,  # Higher friction for platform
            spinningFriction=0.01,
            rollingFriction=0.01,
            restitution=0.2,  # Less bouncy than blocks
        )

        # Generate platform name if not provided
        if platform_name is None:
            platform_name = f"platform_{len(self.objects)}"

        # Store platform in objects dictionary
        self.objects[platform_name] = platform_id

        logger.info(
            f"Spawned static platform '{platform_name}' (id={platform_id}) at position {position} "
            f"with color RGB({color_rgb[0]:.2f}, {color_rgb[1]:.2f}, {color_rgb[2]:.2f}) "
            f"and size {size:.3f}x{size:.3f}x{height:.3f}"
        )

        return platform_id

    def spawn_table(
        self,
        position: Tuple[float, float, float] = (0.6, 0.0, 0.0),
        table_height: float = 0.4,
        table_size: float = 0.3,
        color_rgb: Optional[Tuple[float, float, float]] = None,
    ) -> int:
        """
        Spawn a table-like platform at specified height.

        Args:
            position: Table center position (x, y, z).
            table_height: Height of the table surface.
            table_size: Size of the table surface.
            color_rgb: RGB color. Default is brown wood color.

        Returns:
            Body ID of the table.
        """
        if color_rgb is None:
            color_rgb = (0.6, 0.4, 0.2)  # Brown wood color
        
        # Adjust position to account for table height
        table_position = (position[0], position[1], table_height)
        
        return self.spawn_platform(
            color_rgb=color_rgb,
            size=table_size,
            position=table_position,
            platform_name=f"table_{len(self.objects)}",
            height=0.02,  # Thin table top
        )

    def spawn_random_blocks(
        self,
        num_blocks: int = 5,
        size_range: Tuple[float, float] = (0.02, 0.05),
        spawn_height: float = 0.3,
        spawn_radius: float = 0.3,
    ) -> List[int]:
        """
        Spawn multiple random colored blocks.

        Args:
            num_blocks: Number of blocks to spawn.
            size_range: Min and max size for blocks.
            spawn_height: Height to drop blocks from.
            spawn_radius: Radius around center to spawn blocks.

        Returns:
            List of body IDs for spawned blocks.
        """
        block_ids = []
        
        for i in range(num_blocks):
            # Random color
            color = np.random.rand(3)
            
            # Random size
            size = np.random.uniform(size_range[0], size_range[1])
            
            # Random position in circular area
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, spawn_radius)
            x = 0.5 + radius * np.cos(angle)
            y = 0.0 + radius * np.sin(angle)
            z = spawn_height + i * 0.1  # Stack them slightly
            
            # Spawn block
            block_id = self.spawn_block(
                color_rgb=tuple(color),
                size=size,
                position=(x, y, z),
                block_name=f"random_block_{i}",
            )
            block_ids.append(block_id)
        
        return block_ids
    
    def create_fixed_camera(
        self,
        camera_position: Optional[Tuple[float, float, float]] = None,
        target_position: Optional[Tuple[float, float, float]] = None,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 10.0,
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a fixed camera above the table and return camera matrices.
        
        This creates a virtual camera at the specified position looking at the
        target position, and computes the camera intrinsic and extrinsic matrices.
        
        Args:
            camera_position: Camera position (x, y, z). Default is above table.
            target_position: Target to look at (x, y, z). Default is table center.
            width: Image width in pixels (default: 640).
            height: Image height in pixels (default: 480).
            fov: Field of view in degrees (default: 60.0).
            near: Near clipping plane distance (default: 0.1).
            far: Far clipping plane distance (default: 10.0).
            camera_name: Optional name for the camera. Auto-generated if None.
            
        Returns:
            Dictionary containing:
            - 'intrinsic_matrix': 3x3 camera intrinsic matrix (K)
            - 'view_matrix': 4x4 view matrix (world to camera transform)
            - 'projection_matrix': 4x4 OpenGL projection matrix
            - 'camera_position': Camera position in world coordinates
            - 'target_position': Target position in world coordinates
            - 'width': Image width
            - 'height': Image height
            - 'fov': Field of view
            - 'near': Near plane
            - 'far': Far plane
            - 'camera_name': Camera identifier
            
        Example:
            # Create camera above table
            camera_info = sim.create_fixed_camera(
                camera_position=(0.5, 0.0, 1.0),  # 1m above table
                target_position=(0.5, 0.0, 0.4),   # Look at table surface
            )
            
            # Access camera matrices
            K = camera_info['intrinsic_matrix']  # For computer vision
            view_mat = camera_info['view_matrix']  # For rendering
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        # Default camera position (above table)
        if camera_position is None:
            camera_position = (0.5, 0.0, 1.0)  # 1m above typical table position
        
        # Default target position (table center)
        if target_position is None:
            target_position = (0.5, 0.0, 0.4)  # Typical table surface height
        
        # Calculate camera up vector (usually world Z-axis)
        up_vector = [0, 0, 1]
        
        # Compute view matrix using PyBullet
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector
        )
        
        # Compute projection matrix
        aspect_ratio = width / height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect_ratio,
            nearVal=near,
            farVal=far
        )
        
        # Convert view and projection matrices to numpy arrays
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T
        projection_matrix_np = np.array(projection_matrix).reshape(4, 4).T
        
        # Compute camera intrinsic matrix (K)
        # This is the standard pinhole camera model matrix
        fov_rad = np.deg2rad(fov)
        f_y = (height / 2.0) / np.tan(fov_rad / 2.0)
        f_x = f_y  # Assuming square pixels
        c_x = width / 2.0
        c_y = height / 2.0
        
        intrinsic_matrix = np.array([
            [f_x, 0,   c_x],
            [0,   f_y, c_y],
            [0,   0,   1  ]
        ])
        
        # Generate camera name if not provided
        if camera_name is None:
            camera_name = f"camera_{len(self.objects)}"  # Simple counter-based name
        
        # Store camera info
        camera_info = {
            'intrinsic_matrix': intrinsic_matrix,
            'view_matrix': view_matrix_np,
            'projection_matrix': projection_matrix_np,
            'camera_position': camera_position,
            'target_position': target_position,
            'up_vector': up_vector,
            'width': width,
            'height': height,
            'fov': fov,
            'near': near,
            'far': far,
            'aspect_ratio': aspect_ratio,
            'camera_name': camera_name,
            # Store raw PyBullet matrices for rendering
            '_view_matrix_pb': view_matrix,
            '_projection_matrix_pb': projection_matrix,
        }
        
        logger.info(
            f"Created fixed camera '{camera_name}' at position {camera_position} "
            f"looking at {target_position} (resolution: {width}x{height}, fov: {fov}°)"
        )
        
        return camera_info
    
    def create_wrist_camera(
        self,
        robot_name: str,
        camera_offset: Optional[Tuple[float, float, float]] = None,
        camera_orientation: Optional[Tuple[float, float, float, float]] = None,
        look_at_distance: float = 0.3,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near: float = 0.01,
        far: float = 2.0,
        camera_name: Optional[str] = None,
        attach_to_link: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a camera attached to the robot's tool/wrist link.
        
        This creates a virtual camera that moves with the robot's end-effector.
        The camera can be offset from the tool link and oriented to look at a
        specific direction relative to the tool frame.
        
        Args:
            robot_name: Name of the robot to attach camera to.
            camera_offset: Position offset from tool link in tool frame (x, y, z).
                         Default is (0, 0, -0.05) for slightly behind tool.
            camera_orientation: Orientation offset as quaternion (x, y, z, w).
                              Default is looking forward along tool Z-axis.
            look_at_distance: Distance along tool Z-axis to look at (for default orientation).
            width: Image width in pixels (default: 640).
            height: Image height in pixels (default: 480).
            fov: Field of view in degrees (default: 60.0).
            near: Near clipping plane distance (default: 0.01).
            far: Far clipping plane distance (default: 2.0).
            camera_name: Optional name for the camera. Auto-generated if None.
            attach_to_link: Optional specific link index. Uses tool_link if None.
            
        Returns:
            Dictionary containing:
            - 'intrinsic_matrix': 3x3 camera intrinsic matrix (K)
            - 'robot_name': Name of the attached robot
            - 'attach_link': Link index the camera is attached to
            - 'camera_offset': Camera position offset in link frame
            - 'camera_orientation': Camera orientation offset
            - 'transform_camera_to_link': 4x4 transform from camera to link frame
            - 'width', 'height', 'fov', 'near', 'far': Camera parameters
            - 'camera_name': Camera identifier
            
        Example:
            # Create wrist camera with default offset
            wrist_cam = sim.create_wrist_camera("robot")
            
            # Create camera with custom offset and orientation
            wrist_cam = sim.create_wrist_camera(
                "robot",
                camera_offset=(0, 0.05, -0.1),  # 5cm to side, 10cm back
                look_at_distance=0.2  # Look 20cm ahead
            )
            
            # Get current camera pose in world frame
            cam_pose = sim.get_wrist_camera_pose(wrist_cam)
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        # Determine which link to attach to
        if attach_to_link is not None:
            attach_link = attach_to_link
        else:
            # Use tool link by default
            attach_link = robot.tool_link_index
            if attach_link is None:
                attach_link = robot.end_effector_index
                if attach_link is None:
                    raise RuntimeError(
                        f"Cannot determine tool link for robot '{robot_name}'. "
                        "No tool_link_index or end_effector_index found."
                    )
        
        # Default camera offset (slightly behind and above the tool)
        if camera_offset is None:
            camera_offset = (0, 0, -0.05)  # 5cm behind tool along its Z-axis
        
        # Default camera orientation (looking forward along tool Z-axis)
        if camera_orientation is None:
            # Default: camera looks along positive Z of tool frame
            # This is identity quaternion - aligned with tool frame
            camera_orientation = (0, 0, 0, 1)
        
        # Compute camera intrinsic matrix
        fov_rad = np.deg2rad(fov)
        f_y = (height / 2.0) / np.tan(fov_rad / 2.0)
        f_x = f_y  # Assuming square pixels
        c_x = width / 2.0
        c_y = height / 2.0
        
        intrinsic_matrix = np.array([
            [f_x, 0,   c_x],
            [0,   f_y, c_y],
            [0,   0,   1  ]
        ])
        
        # Compute transform from camera to link frame
        # This is the relative pose of the camera in the link's coordinate frame
        transform_camera_to_link = self._compute_transform_matrix(
            camera_offset, camera_orientation
        )
        
        # Generate camera name if not provided
        if camera_name is None:
            camera_name = f"wrist_camera_{robot_name}"
        
        # Store wrist camera info
        camera_info = {
            'intrinsic_matrix': intrinsic_matrix,
            'robot_name': robot_name,
            'attach_link': attach_link,
            'camera_offset': camera_offset,
            'camera_orientation': camera_orientation,
            'transform_camera_to_link': transform_camera_to_link,
            'look_at_distance': look_at_distance,
            'width': width,
            'height': height,
            'fov': fov,
            'near': near,
            'far': far,
            'aspect_ratio': width / height,
            'camera_name': camera_name,
            'camera_type': 'wrist',  # Mark as wrist camera
        }
        
        logger.info(
            f"Created wrist camera '{camera_name}' attached to robot '{robot_name}' "
            f"link {attach_link} with offset {camera_offset}"
        )
        
        return camera_info
    
    def get_wrist_camera_pose(
        self,
        camera_info: Dict[str, Any],
        return_matrices: bool = False,
    ) -> Union[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
               Dict[str, Any]]:
        """
        Get the current world pose of a wrist-mounted camera.
        
        Args:
            camera_info: Camera info dictionary from create_wrist_camera().
            return_matrices: If True, also returns view and projection matrices.
            
        Returns:
            If return_matrices=False:
                Tuple of (position, orientation) in world frame.
            If return_matrices=True:
                Dictionary with pose and camera matrices.
                
        Example:
            wrist_cam = sim.create_wrist_camera("robot")
            
            # Get just the pose
            pos, orn = sim.get_wrist_camera_pose(wrist_cam)
            
            # Get pose with matrices for rendering
            cam_data = sim.get_wrist_camera_pose(wrist_cam, return_matrices=True)
            view_matrix = cam_data['view_matrix']
        """
        if camera_info.get('camera_type') != 'wrist':
            raise ValueError("Camera info is not from a wrist camera")
        
        robot_name = camera_info['robot_name']
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        attach_link = camera_info['attach_link']
        
        # Get link pose in world frame
        link_state = p.getLinkState(robot.robot_id, attach_link)
        link_pos = link_state[0]
        link_orn = link_state[1]
        
        # Convert link pose to transformation matrix
        link_to_world = self._pose_to_transform_matrix(link_pos, link_orn)
        
        # Get camera-to-link transform
        camera_to_link = camera_info['transform_camera_to_link']
        
        # Compute camera pose in world frame
        camera_to_world = link_to_world @ camera_to_link
        
        # Extract position and orientation from transform matrix
        camera_pos = tuple(camera_to_world[:3, 3])
        camera_rot_matrix = camera_to_world[:3, :3]
        camera_orn = self._rotation_matrix_to_quaternion(camera_rot_matrix)
        
        if not return_matrices:
            return camera_pos, camera_orn
        
        # Compute view and projection matrices if requested
        # Calculate where the camera is looking
        look_at_distance = camera_info['look_at_distance']
        
        # Look-at point is along camera's Z-axis
        look_direction = camera_to_world[:3, 2]  # Z-axis of camera in world
        target_pos = np.array(camera_pos) + look_at_distance * look_direction
        
        # Camera up vector is the Y-axis of camera in world frame
        up_vector = camera_to_world[:3, 1]
        
        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=up_vector.tolist()
        )
        
        # Compute projection matrix
        aspect_ratio = camera_info['width'] / camera_info['height']
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera_info['fov'],
            aspect=aspect_ratio,
            nearVal=camera_info['near'],
            farVal=camera_info['far']
        )
        
        # Convert to numpy arrays
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T
        projection_matrix_np = np.array(projection_matrix).reshape(4, 4).T
        
        return {
            'camera_position': camera_pos,
            'camera_orientation': camera_orn,
            'target_position': target_pos.tolist(),
            'up_vector': up_vector.tolist(),
            'view_matrix': view_matrix_np,
            'projection_matrix': projection_matrix_np,
            'transform_camera_to_world': camera_to_world,
            'transform_link_to_world': link_to_world,
            '_view_matrix_pb': view_matrix,  # Raw PyBullet matrix
            '_projection_matrix_pb': projection_matrix,
        }
    
    def get_wrist_camera_image(
        self,
        camera_info: Dict[str, Any],
        get_rgb: bool = True,
        get_depth: bool = True,
        get_segmentation: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Capture images from a wrist-mounted camera.
        
        This is a convenience method that handles the wrist camera's current pose
        and captures images from its perspective.
        
        Args:
            camera_info: Camera info dictionary from create_wrist_camera().
            get_rgb: Whether to capture RGB image.
            get_depth: Whether to capture depth image.
            get_segmentation: Whether to capture segmentation masks.
            
        Returns:
            Dictionary containing requested images:
            - 'rgb': RGB image (H, W, 3) uint8 if get_rgb=True
            - 'depth': Depth buffer (H, W) float32 if get_depth=True
            - 'segmentation': Segmentation masks (H, W) int32 if get_segmentation=True
            
        Example:
            # Create and use wrist camera
            wrist_cam = sim.create_wrist_camera("robot")
            
            # Move robot
            sim.set_ee_pose("robot", target_pos=(0.5, 0.3, 0.4))
            
            # Capture from wrist camera perspective
            images = sim.get_wrist_camera_image(wrist_cam)
            rgb = images['rgb']
            depth = images['depth']
        """
        if camera_info.get('camera_type') != 'wrist':
            raise ValueError("Camera info is not from a wrist camera")
        
        # Get current camera pose and matrices
        cam_data = self.get_wrist_camera_pose(camera_info, return_matrices=True)
        
        # Create temporary camera info for get_camera_image
        temp_camera_info = {
            'width': camera_info['width'],
            'height': camera_info['height'],
            'near': camera_info['near'],
            'far': camera_info['far'],
            '_view_matrix_pb': cam_data['_view_matrix_pb'],
            '_projection_matrix_pb': cam_data['_projection_matrix_pb'],
        }
        
        # Use the general camera image capture method
        return self.get_camera_image(
            temp_camera_info,
            get_rgb=get_rgb,
            get_depth=get_depth,
            get_segmentation=get_segmentation
        )
    
    def _compute_transform_matrix(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Compute 4x4 transformation matrix from position and quaternion.
        
        Args:
            position: Position (x, y, z).
            orientation: Quaternion (x, y, z, w).
            
        Returns:
            4x4 transformation matrix.
        """
        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = position
        
        return transform
    
    def _pose_to_transform_matrix(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Convert pose (position and quaternion) to 4x4 transformation matrix.
        
        Args:
            position: Position (x, y, z).
            orientation: Quaternion (x, y, z, w).
            
        Returns:
            4x4 transformation matrix.
        """
        return self._compute_transform_matrix(position, orientation)
    
    def _rotation_matrix_to_quaternion(self, rot_matrix: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Convert 3x3 rotation matrix to quaternion.
        
        Args:
            rot_matrix: 3x3 rotation matrix.
            
        Returns:
            Quaternion (x, y, z, w).
        """
        # Use PyBullet's built-in conversion
        # First flatten the rotation matrix
        rot_flat = rot_matrix.flatten().tolist()
        
        # PyBullet expects a 3x3 matrix as a list of 9 values
        quat = p.getQuaternionFromEuler(
            p.getEulerFromQuaternion(
                p.getQuaternionFromMatrix(rot_flat)
            )
        )
        
        return quat
    
    def compute_relative_transform(
        self,
        robot_name: str,
        from_link: Optional[int] = None,
        to_link: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute relative transform between two links of a robot.
        
        Args:
            robot_name: Name of the robot.
            from_link: Source link index. Uses base (-1) if None.
            to_link: Target link index. Uses tool link if None.
            
        Returns:
            Dictionary containing:
            - 'transform': 4x4 transformation matrix from source to target
            - 'position': Relative position (x, y, z)
            - 'orientation': Relative orientation quaternion (x, y, z, w)
            - 'distance': Euclidean distance between links
            
        Example:
            # Get transform from base to end-effector
            transform = sim.compute_relative_transform("robot")
            
            # Get transform between specific links
            transform = sim.compute_relative_transform("robot", from_link=2, to_link=5)
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot = self.robots[robot_name]
        
        # Default links
        if from_link is None:
            from_link = -1  # Base link
        if to_link is None:
            to_link = robot.tool_link_index
            if to_link is None:
                to_link = robot.end_effector_index
                if to_link is None:
                    raise RuntimeError(f"Cannot determine tool link for robot '{robot_name}'")
        
        # Get poses of both links
        if from_link == -1:
            # Base link
            from_pos, from_orn = p.getBasePositionAndOrientation(robot.robot_id)
        else:
            # Regular link
            from_state = p.getLinkState(robot.robot_id, from_link)
            from_pos = from_state[0]
            from_orn = from_state[1]
        
        to_state = p.getLinkState(robot.robot_id, to_link)
        to_pos = to_state[0]
        to_orn = to_state[1]
        
        # Convert to transformation matrices
        from_transform = self._pose_to_transform_matrix(from_pos, from_orn)
        to_transform = self._pose_to_transform_matrix(to_pos, to_orn)
        
        # Compute relative transform: T_from_to = inv(T_from_world) @ T_to_world
        relative_transform = np.linalg.inv(from_transform) @ to_transform
        
        # Extract position and orientation from relative transform
        relative_pos = tuple(relative_transform[:3, 3])
        relative_rot_matrix = relative_transform[:3, :3]
        relative_orn = self._rotation_matrix_to_quaternion(relative_rot_matrix)
        
        # Compute distance
        distance = np.linalg.norm(relative_pos)
        
        return {
            'transform': relative_transform,
            'position': relative_pos,
            'orientation': relative_orn,
            'distance': distance,
            'from_link': from_link,
            'to_link': to_link,
        }
    
    def get_camera_image(
        self,
        camera_info: Dict[str, Any],
        get_rgb: bool = True,
        get_depth: bool = True,
        get_segmentation: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Capture images from a fixed camera.
        
        Args:
            camera_info: Camera info dictionary from create_fixed_camera().
            get_rgb: Whether to capture RGB image.
            get_depth: Whether to capture depth image.
            get_segmentation: Whether to capture segmentation masks.
            
        Returns:
            Dictionary containing requested images:
            - 'rgb': RGB image (H, W, 3) uint8 if get_rgb=True
            - 'depth': Depth buffer (H, W) float32 if get_depth=True
            - 'segmentation': Segmentation masks (H, W) int32 if get_segmentation=True
            
        Example:
            # Create camera
            camera = sim.create_fixed_camera()
            
            # Capture images
            images = sim.get_camera_image(camera)
            rgb_image = images['rgb']
            depth_map = images['depth']
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        # Get camera parameters
        width = camera_info['width']
        height = camera_info['height']
        view_matrix = camera_info['_view_matrix_pb']
        projection_matrix = camera_info['_projection_matrix_pb']
        
        # Capture image using PyBullet
        img_data = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Parse the returned data
        result = {}
        
        if get_rgb:
            # Extract RGB image
            rgb_pixels = img_data[2]  # RGBA buffer
            rgb_array = np.array(rgb_pixels, dtype=np.uint8).reshape(height, width, 4)
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
            result['rgb'] = rgb_array
        
        if get_depth:
            # Extract depth buffer
            depth_buffer = img_data[3]
            depth_array = np.array(depth_buffer, dtype=np.float32).reshape(height, width)
            
            # Convert from depth buffer to actual depth values
            # PyBullet depth buffer is in [0,1], need to convert to actual distances
            near = camera_info['near']
            far = camera_info['far']
            depth_array = far * near / (far - (far - near) * depth_array)
            result['depth'] = depth_array
        
        if get_segmentation:
            # Extract segmentation mask
            seg_mask = img_data[4]
            seg_array = np.array(seg_mask, dtype=np.int32).reshape(height, width)
            result['segmentation'] = seg_array
        
        return result
    
    def compute_point_cloud(
        self,
        camera_info: Dict[str, Any],
        depth_image: Optional[np.ndarray] = None,
        rgb_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute 3D point cloud from depth image using camera parameters.
        
        Args:
            camera_info: Camera info dictionary from create_fixed_camera().
            depth_image: Depth image. If None, captures new depth image.
            rgb_image: Optional RGB image for colored point cloud.
            
        Returns:
            Point cloud as numpy array of shape (N, 3) or (N, 6) if RGB provided.
            Each point is (x, y, z) or (x, y, z, r, g, b).
            
        Example:
            camera = sim.create_fixed_camera()
            points = sim.compute_point_cloud(camera)
        """
        # Get depth image if not provided
        if depth_image is None:
            images = self.get_camera_image(camera_info, get_rgb=(rgb_image is None), get_depth=True)
            depth_image = images['depth']
            if rgb_image is None and 'rgb' in images:
                rgb_image = images['rgb']
        
        # Get camera parameters
        K = camera_info['intrinsic_matrix']
        width = camera_info['width']
        height = camera_info['height']
        view_matrix = camera_info['view_matrix']
        
        # Create pixel coordinate grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert pixel coordinates to camera coordinates
        z = depth_image
        x = (xx - K[0, 2]) * z / K[0, 0]
        y = (yy - K[1, 2]) * z / K[1, 1]
        
        # Stack into homogeneous coordinates
        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        
        # Transform to world coordinates
        # Invert view matrix to get camera-to-world transform
        cam_to_world = np.linalg.inv(view_matrix)
        
        # Reshape for matrix multiplication
        points_flat = points_cam.reshape(-1, 4)
        points_world = (cam_to_world @ points_flat.T).T
        points_world = points_world[:, :3]  # Remove homogeneous coordinate
        
        # Add RGB if available
        if rgb_image is not None:
            rgb_flat = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
            points_world = np.hstack([points_world, rgb_flat])
        
        # Filter out invalid points (e.g., far plane)
        valid_mask = z.flatten() < camera_info['far'] * 0.99
        points_world = points_world[valid_mask]
        
        return points_world
    
    def render_camera(
        self,
        view_matrix: Union[np.ndarray, List[float]],
        projection_matrix: Union[np.ndarray, List[float]],
        width: int = 256,
        height: int = 256,
        near: float = 0.01,
        far: float = 10.0,
        return_segmentation: bool = False,
        renderer: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Render camera view using provided view and projection matrices.
        
        This is a low-level rendering function that directly uses PyBullet's
        getCameraImage with custom matrices. Useful for custom camera setups
        or when matrices are already computed.
        
        Args:
            view_matrix: 4x4 view matrix (world to camera transform) as numpy array
                        or flat list of 16 values (PyBullet format).
            projection_matrix: 4x4 projection matrix as numpy array or flat list.
            width: Image width in pixels (default: 256).
            height: Image height in pixels (default: 256).
            near: Near clipping plane for depth conversion (default: 0.01).
            far: Far clipping plane for depth conversion (default: 10.0).
            return_segmentation: If True, also returns segmentation mask.
            renderer: PyBullet renderer type. If None, uses hardware OpenGL.
                     Options: p.ER_BULLET_HARDWARE_OPENGL, p.ER_TINY_RENDERER.
            
        Returns:
            Dictionary containing:
            - 'rgb': RGB image as uint8 numpy array of shape (height, width, 3)
            - 'depth': Depth map as float32 numpy array of shape (height, width)
            - 'segmentation': Segmentation mask if requested (height, width)
            
        Example:
            # Using with fixed camera
            camera = sim.create_fixed_camera()
            result = sim.render_camera(
                camera['_view_matrix_pb'],
                camera['_projection_matrix_pb']
            )
            rgb = result['rgb']
            depth = result['depth']
            
            # Using with custom matrices
            import numpy as np
            
            # Custom view matrix (looking down from above)
            eye_pos = [0.5, 0.0, 2.0]
            target_pos = [0.5, 0.0, 0.0]
            up_vector = [0, 1, 0]
            view_mat = p.computeViewMatrix(eye_pos, target_pos, up_vector)
            
            # Custom projection matrix
            proj_mat = p.computeProjectionMatrixFOV(
                fov=45, aspect=1.0, nearVal=0.1, farVal=10.0
            )
            
            # Render
            result = sim.render_camera(view_mat, proj_mat, width=512, height=512)
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        # Convert numpy arrays to PyBullet format if needed
        if isinstance(view_matrix, np.ndarray):
            if view_matrix.shape == (4, 4):
                # Convert from 4x4 matrix to flat list (column-major for PyBullet)
                view_matrix = view_matrix.T.flatten().tolist()
            elif view_matrix.shape == (16,):
                view_matrix = view_matrix.tolist()
            else:
                raise ValueError(
                    f"view_matrix must be 4x4 or flat array of 16 values, got shape {view_matrix.shape}"
                )
        
        if isinstance(projection_matrix, np.ndarray):
            if projection_matrix.shape == (4, 4):
                # Convert from 4x4 matrix to flat list (column-major for PyBullet)
                projection_matrix = projection_matrix.T.flatten().tolist()
            elif projection_matrix.shape == (16,):
                projection_matrix = projection_matrix.tolist()
            else:
                raise ValueError(
                    f"projection_matrix must be 4x4 or flat array of 16 values, got shape {projection_matrix.shape}"
                )
        
        # Validate list format
        if not isinstance(view_matrix, (list, tuple)) or len(view_matrix) != 16:
            raise ValueError("view_matrix must be a list/tuple of 16 values")
        
        if not isinstance(projection_matrix, (list, tuple)) or len(projection_matrix) != 16:
            raise ValueError("projection_matrix must be a list/tuple of 16 values")
        
        # Select renderer
        if renderer is None:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        
        # Capture image using PyBullet
        img_data = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
            flags=p.ER_NO_SEGMENTATION_MASK if not return_segmentation else 0
        )
        
        # Extract RGB image
        rgb_pixels = img_data[2]  # RGBA buffer
        rgb_array = np.array(rgb_pixels, dtype=np.uint8).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        # Extract and convert depth buffer
        depth_buffer = img_data[3]
        depth_array = np.array(depth_buffer, dtype=np.float32).reshape(height, width)
        
        # Convert from normalized depth buffer [0,1] to actual depth values
        # Using the standard perspective depth unprojection formula
        depth_array = far * near / (far - (far - near) * depth_array)
        
        # Build result dictionary
        result = {
            'rgb': rgb_array,
            'depth': depth_array
        }
        
        # Add segmentation if requested
        if return_segmentation:
            seg_mask = img_data[4]
            seg_array = np.array(seg_mask, dtype=np.int32).reshape(height, width)
            result['segmentation'] = seg_array
        
        return result
    
    def render_camera_from_pose(
        self,
        camera_position: Tuple[float, float, float],
        target_position: Optional[Tuple[float, float, float]] = None,
        up_vector: Optional[Tuple[float, float, float]] = None,
        fov: float = 60.0,
        width: int = 256,
        height: int = 256,
        near: float = 0.01,
        far: float = 10.0,
        return_matrices: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Render camera view from a specified pose (position and look-at target).
        
        This is a convenience method that computes view and projection matrices
        from camera pose parameters and then renders the view.
        
        Args:
            camera_position: Camera position in world coordinates (x, y, z).
            target_position: Position to look at. If None, looks at origin.
            up_vector: Camera up direction. If None, uses world Z-axis [0, 0, 1].
            fov: Field of view in degrees (default: 60.0).
            width: Image width in pixels (default: 256).
            height: Image height in pixels (default: 256).
            near: Near clipping plane (default: 0.01).
            far: Far clipping plane (default: 10.0).
            return_matrices: If True, also returns view and projection matrices.
            
        Returns:
            Dictionary containing:
            - 'rgb': RGB image (height, width, 3) uint8
            - 'depth': Depth map (height, width) float32
            - 'view_matrix': 4x4 view matrix (if return_matrices=True)
            - 'projection_matrix': 4x4 projection matrix (if return_matrices=True)
            - 'intrinsic_matrix': 3x3 camera intrinsic matrix K (if return_matrices=True)
            
        Example:
            # Render from above looking down
            result = sim.render_camera_from_pose(
                camera_position=(0.5, 0.0, 2.0),
                target_position=(0.5, 0.0, 0.0),
                fov=45,
                width=512,
                height=512
            )
            
            # Render with matrices returned
            result = sim.render_camera_from_pose(
                camera_position=(1.0, 1.0, 1.0),
                return_matrices=True
            )
            K = result['intrinsic_matrix']  # For computer vision
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        # Default parameters
        if target_position is None:
            target_position = (0.0, 0.0, 0.0)  # Look at origin
        
        if up_vector is None:
            up_vector = (0, 0, 1)  # World Z-axis
        
        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector
        )
        
        # Compute projection matrix
        aspect_ratio = width / height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect_ratio,
            nearVal=near,
            farVal=far
        )
        
        # Render using the matrices
        result = self.render_camera(
            view_matrix=view_matrix,
            projection_matrix=projection_matrix,
            width=width,
            height=height,
            near=near,
            far=far,
            return_segmentation=False
        )
        
        # Add matrices if requested
        if return_matrices:
            # Convert to numpy 4x4 matrices
            view_matrix_np = np.array(view_matrix).reshape(4, 4).T
            projection_matrix_np = np.array(projection_matrix).reshape(4, 4).T
            
            # Compute camera intrinsic matrix K
            fov_rad = np.deg2rad(fov)
            f_y = (height / 2.0) / np.tan(fov_rad / 2.0)
            f_x = f_y  # Assuming square pixels
            c_x = width / 2.0
            c_y = height / 2.0
            
            intrinsic_matrix = np.array([
                [f_x, 0,   c_x],
                [0,   f_y, c_y],
                [0,   0,   1  ]
            ])
            
            result['view_matrix'] = view_matrix_np
            result['projection_matrix'] = projection_matrix_np
            result['intrinsic_matrix'] = intrinsic_matrix
        
        return result
    
    def render_multiple_views(
        self,
        camera_positions: List[Tuple[float, float, float]],
        target_position: Optional[Tuple[float, float, float]] = None,
        fov: float = 60.0,
        width: int = 256,
        height: int = 256,
        near: float = 0.01,
        far: float = 10.0,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Render from multiple camera positions efficiently.
        
        This method renders views from multiple camera positions, all looking
        at the same target. Useful for multi-view reconstruction or generating
        training data from different viewpoints.
        
        Args:
            camera_positions: List of camera positions (x, y, z).
            target_position: Common target position. If None, uses origin.
            fov: Field of view for all cameras (default: 60.0).
            width: Image width for all views (default: 256).
            height: Image height for all views (default: 256).
            near: Near clipping plane (default: 0.01).
            far: Far clipping plane (default: 10.0).
            
        Returns:
            List of dictionaries, each containing 'rgb' and 'depth' arrays.
            
        Example:
            # Render from 4 viewpoints around an object
            positions = [
                (1.0, 0.0, 0.5),   # Front
                (0.0, 1.0, 0.5),   # Right
                (-1.0, 0.0, 0.5),  # Back
                (0.0, -1.0, 0.5),  # Left
            ]
            
            views = sim.render_multiple_views(
                camera_positions=positions,
                target_position=(0, 0, 0),
                width=512,
                height=512
            )
            
            # Access individual views
            for i, view in enumerate(views):
                rgb = view['rgb']
                depth = view['depth']
                # Process or save images...
        """
        results = []
        
        for cam_pos in camera_positions:
            result = self.render_camera_from_pose(
                camera_position=cam_pos,
                target_position=target_position,
                fov=fov,
                width=width,
                height=height,
                near=near,
                far=far,
                return_matrices=False
            )
            results.append(result)
        
        return results
    
    def render_wrist_cam(
        self,
        robot_name: str,
        wrist_camera_info: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        step_simulation: bool = True,
        return_matrices: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Render RGB and depth images from the robot's wrist camera, synchronized with simulation.
        
        This method captures images from the wrist-mounted camera's current perspective,
        optionally stepping the simulation to ensure synchronization. This is particularly
        useful for visual servoing, data collection during motion, or real-time monitoring.
        
        Args:
            robot_name: Name of the robot with the wrist camera.
            wrist_camera_info: Camera info from create_wrist_camera(). If None,
                              creates a default wrist camera temporarily.
            width: Override image width. If None, uses camera info or default (256).
            height: Override image height. If None, uses camera info or default (256).
            step_simulation: If True, steps simulation before rendering for synchronization.
            return_matrices: If True, also returns current camera matrices and pose.
            
        Returns:
            Dictionary containing:
            - 'rgb': RGB image as uint8 numpy array (height, width, 3)
            - 'depth': Depth map as float32 numpy array (height, width)
            - 'timestamp': Simulation step counter when image was captured
            - 'camera_position': Camera position in world frame (if return_matrices=True)
            - 'camera_orientation': Camera orientation quaternion (if return_matrices=True)
            - 'view_matrix': 4x4 view matrix (if return_matrices=True)
            - 'projection_matrix': 4x4 projection matrix (if return_matrices=True)
            - 'intrinsic_matrix': 3x3 camera intrinsic matrix K (if return_matrices=True)
            
        Raises:
            ValueError: If robot not found.
            RuntimeError: If not connected to PyBullet.
            
        Example:
            # Basic usage with existing wrist camera
            wrist_cam = sim.create_wrist_camera("robot")
            result = sim.render_wrist_cam("robot", wrist_cam)
            rgb = result['rgb']
            depth = result['depth']
            
            # Use default camera (creates temporary)
            result = sim.render_wrist_cam("robot")
            
            # Custom resolution
            result = sim.render_wrist_cam("robot", wrist_cam, width=512, height=512)
            
            # Get camera info along with images
            result = sim.render_wrist_cam("robot", wrist_cam, return_matrices=True)
            cam_pos = result['camera_position']
            K = result['intrinsic_matrix']
            
            # Capture during motion without stepping
            for i in range(100):
                sim.step()  # Step manually
                result = sim.render_wrist_cam("robot", wrist_cam, step_simulation=False)
                # Process images...
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        # Create temporary default wrist camera if not provided
        temp_camera = False
        if wrist_camera_info is None:
            temp_camera = True
            wrist_camera_info = self.create_wrist_camera(
                robot_name,
                camera_offset=(0, 0, -0.05),  # 5cm behind tool
                camera_orientation=(0, 0, 0, 1),  # Aligned with tool
                look_at_distance=0.3,
                width=width or 256,
                height=height or 256,
                fov=60.0,
                near=0.01,
                far=2.0,
                camera_name=f"temp_wrist_cam_{robot_name}"
            )
        
        # Override resolution if specified
        render_width = width if width is not None else wrist_camera_info['width']
        render_height = height if height is not None else wrist_camera_info['height']
        
        # Step simulation if requested (for synchronization)
        if step_simulation:
            self.step()
        
        # Record timestamp (simulation step)
        timestamp = self._step_counter
        
        # Get current camera pose and matrices
        cam_data = self.get_wrist_camera_pose(wrist_camera_info, return_matrices=True)
        
        # Render using the current camera matrices
        render_result = self.render_camera(
            view_matrix=cam_data['_view_matrix_pb'],
            projection_matrix=cam_data['_projection_matrix_pb'],
            width=render_width,
            height=render_height,
            near=wrist_camera_info['near'],
            far=wrist_camera_info['far'],
            return_segmentation=False
        )
        
        # Build result dictionary
        result = {
            'rgb': render_result['rgb'],
            'depth': render_result['depth'],
            'timestamp': timestamp,
        }
        
        # Add camera information if requested
        if return_matrices:
            # Compute intrinsic matrix for the render resolution
            fov_rad = np.deg2rad(wrist_camera_info['fov'])
            f_y = (render_height / 2.0) / np.tan(fov_rad / 2.0)
            f_x = f_y  # Assuming square pixels
            c_x = render_width / 2.0
            c_y = render_height / 2.0
            
            intrinsic_matrix = np.array([
                [f_x, 0,   c_x],
                [0,   f_y, c_y],
                [0,   0,   1  ]
            ])
            
            result.update({
                'camera_position': cam_data['camera_position'],
                'camera_orientation': cam_data['camera_orientation'],
                'target_position': cam_data['target_position'],
                'view_matrix': cam_data['view_matrix'],
                'projection_matrix': cam_data['projection_matrix'],
                'intrinsic_matrix': intrinsic_matrix,
                'transform_camera_to_world': cam_data['transform_camera_to_world'],
            })
        
        return result
    
    def render_wrist_cam_sequence(
        self,
        robot_name: str,
        num_frames: int,
        wrist_camera_info: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps_per_frame: int = 1,
        motion_callback: Optional[callable] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Render a sequence of images from the wrist camera during motion.
        
        This method captures a sequence of RGB and depth images from the wrist camera
        while the simulation runs. Optionally, a motion callback can be provided to
        control the robot during capture.
        
        Args:
            robot_name: Name of the robot with the wrist camera.
            num_frames: Number of frames to capture.
            wrist_camera_info: Camera info from create_wrist_camera(). If None,
                              creates a default wrist camera.
            width: Image width. If None, uses camera info or default (256).
            height: Image height. If None, uses camera info or default (256).
            steps_per_frame: Number of simulation steps between frames (default: 1).
            motion_callback: Optional function called before each frame.
                           Signature: callback(sim, robot_name, frame_index)
                           Can be used to control robot motion during capture.
            
        Returns:
            List of dictionaries, each containing:
            - 'rgb': RGB image (height, width, 3) uint8
            - 'depth': Depth map (height, width) float32
            - 'timestamp': Simulation step when captured
            - 'frame_index': Frame number in sequence
            
        Example:
            # Simple capture during simulation
            sequence = sim.render_wrist_cam_sequence("robot", num_frames=30)
            
            # Capture with robot motion
            def move_robot(sim, robot_name, frame_idx):
                # Move robot based on frame index
                target_pos = (0.5 + 0.01 * frame_idx, 0.3, 0.4)
                sim.set_ee_pose(robot_name, target_pos)
            
            sequence = sim.render_wrist_cam_sequence(
                "robot",
                num_frames=50,
                motion_callback=move_robot,
                steps_per_frame=4
            )
            
            # Process sequence
            for frame in sequence:
                rgb = frame['rgb']
                depth = frame['depth']
                print(f"Frame {frame['frame_index']} at step {frame['timestamp']}")
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        # Create temporary camera if needed
        temp_camera = False
        if wrist_camera_info is None:
            temp_camera = True
            wrist_camera_info = self.create_wrist_camera(
                robot_name,
                camera_offset=(0, 0, -0.05),
                width=width or 256,
                height=height or 256,
                camera_name=f"temp_sequence_cam_{robot_name}"
            )
        
        sequence = []
        
        for frame_idx in range(num_frames):
            # Call motion callback if provided
            if motion_callback is not None:
                try:
                    motion_callback(self, robot_name, frame_idx)
                except Exception as e:
                    logger.warning(f"Motion callback error at frame {frame_idx}: {e}")
            
            # Step simulation
            for _ in range(steps_per_frame):
                self.step()
            
            # Capture frame (don't step again since we just did)
            frame_data = self.render_wrist_cam(
                robot_name,
                wrist_camera_info,
                width=width,
                height=height,
                step_simulation=False,
                return_matrices=False
            )
            
            # Add frame index
            frame_data['frame_index'] = frame_idx
            sequence.append(frame_data)
        
        logger.info(f"Captured {num_frames} frames from wrist camera of '{robot_name}'")
        
        return sequence
    
    def render_stereo_wrist_cam(
        self,
        robot_name: str,
        baseline: float = 0.05,
        wrist_camera_info: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        step_simulation: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Render stereo RGB and depth images from dual wrist cameras.
        
        This creates a stereo camera pair offset horizontally from the wrist position,
        useful for stereo vision applications like depth estimation or 3D reconstruction.
        
        Args:
            robot_name: Name of the robot.
            baseline: Horizontal separation between cameras in meters (default: 0.05).
            wrist_camera_info: Base camera info. If None, creates default.
            width: Image width for both cameras.
            height: Image height for both cameras.
            step_simulation: If True, steps simulation before rendering.
            
        Returns:
            Dictionary containing:
            - 'rgb_left': Left camera RGB image (height, width, 3) uint8
            - 'rgb_right': Right camera RGB image (height, width, 3) uint8
            - 'depth_left': Left camera depth map (height, width) float32
            - 'depth_right': Right camera depth map (height, width) float32
            - 'disparity': Computed disparity map if possible
            - 'baseline': Stereo baseline used
            - 'timestamp': Simulation step when captured
            
        Example:
            # Capture stereo images
            stereo = sim.render_stereo_wrist_cam("robot", baseline=0.1)
            
            # Use for stereo matching
            left_rgb = stereo['rgb_left']
            right_rgb = stereo['rgb_right']
            disparity = compute_disparity(left_rgb, right_rgb)  # Your stereo algorithm
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet. Call connect() first.")
        
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        # Create base camera if not provided
        if wrist_camera_info is None:
            wrist_camera_info = self.create_wrist_camera(
                robot_name,
                camera_offset=(0, 0, -0.05),
                width=width or 256,
                height=height or 256,
                camera_name=f"temp_stereo_base_{robot_name}"
            )
        
        # Override resolution if specified
        render_width = width if width is not None else wrist_camera_info['width']
        render_height = height if height is not None else wrist_camera_info['height']
        
        # Step simulation if requested
        if step_simulation:
            self.step()
        
        timestamp = self._step_counter
        
        # Get current wrist pose
        robot = self.robots[robot_name]
        attach_link = wrist_camera_info['attach_link']
        link_state = p.getLinkState(robot.robot_id, attach_link)
        link_pos = link_state[0]
        link_orn = link_state[1]
        
        # Convert to transformation matrix
        link_to_world = self._pose_to_transform_matrix(link_pos, link_orn)
        
        # Create left camera offset (negative X in link frame)
        left_offset = np.array([-baseline/2, 0, wrist_camera_info['camera_offset'][2], 1])
        left_pos_world = (link_to_world @ left_offset)[:3]
        
        # Create right camera offset (positive X in link frame)
        right_offset = np.array([baseline/2, 0, wrist_camera_info['camera_offset'][2], 1])
        right_pos_world = (link_to_world @ right_offset)[:3]
        
        # Both cameras look forward along the link's Z-axis
        look_direction = link_to_world[:3, 2]  # Z-axis in world frame
        look_at_distance = wrist_camera_info.get('look_at_distance', 0.3)
        target_pos = np.array(link_pos) + look_at_distance * look_direction
        
        # Camera up vector is the Y-axis of link in world frame
        up_vector = link_to_world[:3, 1]
        
        # Compute view matrices for both cameras
        view_matrix_left = p.computeViewMatrix(
            cameraEyePosition=left_pos_world.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=up_vector.tolist()
        )
        
        view_matrix_right = p.computeViewMatrix(
            cameraEyePosition=right_pos_world.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=up_vector.tolist()
        )
        
        # Compute projection matrix (same for both cameras)
        aspect_ratio = render_width / render_height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=wrist_camera_info['fov'],
            aspect=aspect_ratio,
            nearVal=wrist_camera_info['near'],
            farVal=wrist_camera_info['far']
        )
        
        # Render left camera
        left_result = self.render_camera(
            view_matrix=view_matrix_left,
            projection_matrix=projection_matrix,
            width=render_width,
            height=render_height,
            near=wrist_camera_info['near'],
            far=wrist_camera_info['far']
        )
        
        # Render right camera
        right_result = self.render_camera(
            view_matrix=view_matrix_right,
            projection_matrix=projection_matrix,
            width=render_width,
            height=render_height,
            near=wrist_camera_info['near'],
            far=wrist_camera_info['far']
        )
        
        # Build result
        result = {
            'rgb_left': left_result['rgb'],
            'rgb_right': right_result['rgb'],
            'depth_left': left_result['depth'],
            'depth_right': right_result['depth'],
            'baseline': baseline,
            'timestamp': timestamp,
        }
        
        # Optionally compute simple disparity from depth (as reference)
        # Real disparity should be computed from stereo matching
        try:
            # Simplified disparity computation from depth
            # disparity = baseline * focal_length / depth
            fov_rad = np.deg2rad(wrist_camera_info['fov'])
            focal_length = (render_height / 2.0) / np.tan(fov_rad / 2.0)
            
            # Use left depth as reference
            with np.errstate(divide='ignore', invalid='ignore'):
                disparity = baseline * focal_length / left_result['depth']
                disparity[np.isinf(disparity)] = 0
                disparity[np.isnan(disparity)] = 0
                disparity = np.clip(disparity, 0, render_width)
            
            result['disparity'] = disparity.astype(np.float32)
        except Exception as e:
            logger.debug(f"Could not compute disparity: {e}")
            result['disparity'] = None
        
        return result
    
    def estimate_depth_at_center(self, depth_map: np.ndarray, 
                                  window_size: int = 5,
                                  use_median: bool = True,
                                  return_stats: bool = False) -> Union[float, Dict[str, float]]:
        """
        Estimate depth value at the center of a depth map from wrist camera.
        
        This function extracts the depth at the center pixel of the depth map,
        optionally averaging over a small window for more robust estimates.
        Useful for estimating distance to objects directly in front of the gripper.
        
        Args:
            depth_map: 2D depth map array from wrist camera (height, width).
            window_size: Size of window around center for averaging (default: 5).
                        Must be odd number. Use 1 for single pixel.
            use_median: If True, use median instead of mean for robustness (default: True).
            return_stats: If True, return additional statistics (default: False).
            
        Returns:
            If return_stats=False:
                Float depth value at center in meters.
            If return_stats=True:
                Dictionary containing:
                - 'depth': Center depth value
                - 'min': Minimum depth in window
                - 'max': Maximum depth in window  
                - 'mean': Mean depth in window
                - 'median': Median depth in window
                - 'std': Standard deviation in window
                - 'valid_pixels': Number of valid (non-zero) pixels
                
        Raises:
            ValueError: If depth_map is not 2D array or window_size is invalid.
            
        Example:
            # Simple depth at center
            wrist_cam = sim.create_wrist_camera("robot")
            img_data = sim.render_wrist_cam("robot", wrist_cam)
            center_depth = sim.estimate_depth_at_center(img_data['depth'])
            print(f"Object is {center_depth:.3f}m from wrist camera")
            
            # With statistics for quality check
            depth_stats = sim.estimate_depth_at_center(
                img_data['depth'],
                window_size=7,
                return_stats=True
            )
            if depth_stats['std'] < 0.01:  # Low variance = reliable reading
                print(f"Reliable depth: {depth_stats['depth']:.3f}m")
            
            # Single pixel reading
            exact_center = sim.estimate_depth_at_center(
                img_data['depth'],
                window_size=1
            )
        """
        # Validate input
        if not isinstance(depth_map, np.ndarray):
            depth_map = np.array(depth_map)
        
        if depth_map.ndim != 2:
            raise ValueError(f"depth_map must be 2D array, got shape {depth_map.shape}")
        
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")
        
        # Get image dimensions
        height, width = depth_map.shape
        
        # Find center pixel
        center_y = height // 2
        center_x = width // 2
        
        # Single pixel case
        if window_size == 1:
            center_depth = float(depth_map[center_y, center_x])
            if return_stats:
                return {
                    'depth': center_depth,
                    'min': center_depth,
                    'max': center_depth,
                    'mean': center_depth,
                    'median': center_depth,
                    'std': 0.0,
                    'valid_pixels': 1 if center_depth > 0 else 0,
                    'center_pixel': (center_x, center_y),
                }
            return center_depth
        
        # Calculate window bounds
        half_window = window_size // 2
        
        # Ensure window stays within image bounds
        y_min = max(0, center_y - half_window)
        y_max = min(height, center_y + half_window + 1)
        x_min = max(0, center_x - half_window)
        x_max = min(width, center_x + half_window + 1)
        
        # Extract window around center
        window = depth_map[y_min:y_max, x_min:x_max]
        
        # Filter out invalid (zero or near-zero) depth values
        valid_depths = window[window > 0.001]  # Ignore depths < 1mm
        
        if len(valid_depths) == 0:
            # No valid depth values in window
            logger.warning("No valid depth values found in center window")
            if return_stats:
                return {
                    'depth': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'valid_pixels': 0,
                    'center_pixel': (center_x, center_y),
                }
            return 0.0
        
        # Calculate depth estimate
        if use_median:
            center_depth = float(np.median(valid_depths))
        else:
            center_depth = float(np.mean(valid_depths))
        
        # Return simple depth or statistics
        if not return_stats:
            return center_depth
        
        # Calculate additional statistics
        return {
            'depth': center_depth,
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'mean': float(np.mean(valid_depths)),
            'median': float(np.median(valid_depths)),
            'std': float(np.std(valid_depths)),
            'valid_pixels': len(valid_depths),
            'total_pixels': window.size,
            'center_pixel': (center_x, center_y),
            'window_size': window_size,
            'actual_window_shape': window.shape,
        }
    
    def estimate_wrist_distance(
        self,
        robot_name: str,
        wrist_camera_info: Optional[Dict[str, Any]] = None,
        window_size: int = 5,
        use_median: bool = True,
    ) -> float:
        """
        Estimate distance to object directly in front of the wrist camera.
        
        This is a convenience method that captures an image from the wrist camera
        and estimates the depth at the center, which represents the distance to
        whatever object is directly in front of the gripper.
        
        Args:
            robot_name: Name of the robot.
            wrist_camera_info: Camera info from create_wrist_camera(). 
                              If None, creates temporary camera.
            window_size: Size of window for depth averaging (default: 5).
            use_median: If True, use median for robustness (default: True).
            
        Returns:
            Estimated distance in meters to object in front of gripper.
            Returns 0.0 if no valid depth reading.
            
        Example:
            # Quick distance check
            distance = sim.estimate_wrist_distance("robot")
            if distance > 0 and distance < 0.1:  # Within 10cm
                print(f"Object detected {distance*100:.1f}cm from gripper")
                sim.close_gripper("robot")  # Close to grasp
            
            # With custom camera
            wrist_cam = sim.create_wrist_camera("robot", fov=45)
            distance = sim.estimate_wrist_distance("robot", wrist_cam)
        """
        # Render from wrist camera
        img_data = self.render_wrist_cam(
            robot_name,
            wrist_camera_info=wrist_camera_info,
            step_simulation=False  # Don't step, just get current view
        )
        
        # Estimate depth at center
        depth = self.estimate_depth_at_center(
            img_data['depth'],
            window_size=window_size,
            use_median=use_median,
            return_stats=False
        )
        
        return depth
    
    def get_wrist_depth_profile(
        self,
        robot_name: str,
        wrist_camera_info: Optional[Dict[str, Any]] = None,
        num_regions: int = 5,
    ) -> Dict[str, float]:
        """
        Get depth profile across multiple regions of the wrist camera view.
        
        This divides the depth map into regions (center, top, bottom, left, right)
        and computes average depth for each, useful for obstacle detection and
        spatial awareness around the gripper.
        
        Args:
            robot_name: Name of the robot.
            wrist_camera_info: Camera info from create_wrist_camera().
            num_regions: Number of regions to analyze (5 or 9).
            
        Returns:
            Dictionary with depth values for each region:
            - 'center': Depth at center
            - 'top', 'bottom', 'left', 'right': Edge regions
            - 'top_left', 'top_right', etc.: Corners (if num_regions=9)
            - 'min': Minimum depth across all regions
            - 'max': Maximum depth across all regions
            - 'gradient_x': Horizontal depth gradient
            - 'gradient_y': Vertical depth gradient
            
        Example:
            profile = sim.get_wrist_depth_profile("robot")
            
            # Check for obstacles
            if profile['left'] < profile['center']:
                print("Obstacle on the left side")
            
            # Check depth gradient
            if abs(profile['gradient_x']) > 0.1:
                print("Significant horizontal depth change detected")
        """
        # Render from wrist camera
        img_data = self.render_wrist_cam(
            robot_name,
            wrist_camera_info=wrist_camera_info,
            step_simulation=False
        )
        
        depth_map = img_data['depth']
        height, width = depth_map.shape
        
        # Define region boundaries
        third_h = height // 3
        third_w = width // 3
        
        regions = {}
        
        # Center region
        center_region = depth_map[third_h:2*third_h, third_w:2*third_w]
        regions['center'] = float(np.median(center_region[center_region > 0.001]))
        
        # Edge regions
        top_region = depth_map[:third_h, third_w:2*third_w]
        regions['top'] = float(np.median(top_region[top_region > 0.001]) 
                              if np.any(top_region > 0.001) else 0)
        
        bottom_region = depth_map[2*third_h:, third_w:2*third_w]
        regions['bottom'] = float(np.median(bottom_region[bottom_region > 0.001])
                                if np.any(bottom_region > 0.001) else 0)
        
        left_region = depth_map[third_h:2*third_h, :third_w]
        regions['left'] = float(np.median(left_region[left_region > 0.001])
                               if np.any(left_region > 0.001) else 0)
        
        right_region = depth_map[third_h:2*third_h, 2*third_w:]
        regions['right'] = float(np.median(right_region[right_region > 0.001])
                                if np.any(right_region > 0.001) else 0)
        
        # Corners if requested
        if num_regions == 9:
            regions['top_left'] = float(np.median(
                depth_map[:third_h, :third_w][depth_map[:third_h, :third_w] > 0.001]
            ) if np.any(depth_map[:third_h, :third_w] > 0.001) else 0)
            
            regions['top_right'] = float(np.median(
                depth_map[:third_h, 2*third_w:][depth_map[:third_h, 2*third_w:] > 0.001]
            ) if np.any(depth_map[:third_h, 2*third_w:] > 0.001) else 0)
            
            regions['bottom_left'] = float(np.median(
                depth_map[2*third_h:, :third_w][depth_map[2*third_h:, :third_w] > 0.001]
            ) if np.any(depth_map[2*third_h:, :third_w] > 0.001) else 0)
            
            regions['bottom_right'] = float(np.median(
                depth_map[2*third_h:, 2*third_w:][depth_map[2*third_h:, 2*third_w:] > 0.001]
            ) if np.any(depth_map[2*third_h:, 2*third_w:] > 0.001) else 0)
        
        # Calculate statistics
        valid_depths = [d for d in regions.values() if d > 0]
        if valid_depths:
            regions['min'] = min(valid_depths)
            regions['max'] = max(valid_depths)
            
            # Calculate gradients
            if regions['left'] > 0 and regions['right'] > 0:
                regions['gradient_x'] = regions['right'] - regions['left']
            else:
                regions['gradient_x'] = 0.0
            
            if regions['top'] > 0 and regions['bottom'] > 0:
                regions['gradient_y'] = regions['bottom'] - regions['top']
            else:
                regions['gradient_y'] = 0.0
        else:
            regions['min'] = 0.0
            regions['max'] = 0.0
            regions['gradient_x'] = 0.0
            regions['gradient_y'] = 0.0
        
        return regions
    
    def get_camera_intrinsics(
        self,
        camera_info: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fov: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Get camera intrinsic parameters (fx, fy, cx, cy) for pixel-to-world conversion.
        
        These parameters define the camera's internal projection properties and are
        essential for converting between pixel coordinates and 3D world coordinates.
        
        Args:
            camera_info: Camera info dict from create_fixed_camera() or create_wrist_camera().
                        If None, must provide width, height, and fov.
            width: Image width in pixels (overrides camera_info if provided).
            height: Image height in pixels (overrides camera_info if provided).
            fov: Field of view in degrees (overrides camera_info if provided).
            
        Returns:
            Dictionary containing:
            - 'fx': Focal length in x-direction (pixels)
            - 'fy': Focal length in y-direction (pixels)
            - 'cx': Principal point x-coordinate (pixels)
            - 'cy': Principal point y-coordinate (pixels)
            - 'K': 3x3 intrinsic matrix
            - 'K_inv': Inverse of intrinsic matrix (for unprojection)
            - 'width': Image width
            - 'height': Image height
            - 'fov': Field of view in degrees
            - 'fov_rad': Field of view in radians
            
        Raises:
            ValueError: If neither camera_info nor required parameters provided.
            
        Example:
            # From existing camera
            camera = sim.create_fixed_camera()
            intrinsics = sim.get_camera_intrinsics(camera)
            print(f"Focal length: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
            print(f"Principal point: ({intrinsics['cx']:.1f}, {intrinsics['cy']:.1f})")
            
            # Direct parameters
            intrinsics = sim.get_camera_intrinsics(width=640, height=480, fov=60)
            K = intrinsics['K']  # Use for OpenCV functions
            
            # For wrist camera
            wrist_cam = sim.create_wrist_camera("robot")
            intrinsics = sim.get_camera_intrinsics(wrist_cam)
        """
        # Extract parameters from camera_info or use provided values
        if camera_info is not None:
            img_width = width if width is not None else camera_info['width']
            img_height = height if height is not None else camera_info['height']
            img_fov = fov if fov is not None else camera_info['fov']
        else:
            if width is None or height is None or fov is None:
                raise ValueError(
                    "Must provide either camera_info or all of (width, height, fov)"
                )
            img_width = width
            img_height = height
            img_fov = fov
        
        # Convert FOV to radians
        fov_rad = np.deg2rad(img_fov)
        
        # Calculate focal lengths
        # For a pinhole camera model with symmetric FOV:
        # tan(fov/2) = (height/2) / fy
        # Therefore: fy = (height/2) / tan(fov/2)
        fy = (img_height / 2.0) / np.tan(fov_rad / 2.0)
        fx = fy  # Assuming square pixels (aspect ratio = 1)
        
        # Principal point (image center)
        cx = img_width / 2.0
        cy = img_height / 2.0
        
        # Build intrinsic matrix K
        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ])
        
        # Compute inverse for unprojection
        K_inv = np.linalg.inv(K)
        
        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'K': K,
            'K_inv': K_inv,
            'width': img_width,
            'height': img_height,
            'fov': img_fov,
            'fov_rad': fov_rad,
        }
    
    def pixel_to_ray(
        self,
        pixel_x: Union[float, np.ndarray],
        pixel_y: Union[float, np.ndarray],
        camera_info: Optional[Dict[str, Any]] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convert pixel coordinates to 3D ray directions in camera frame.
        
        This computes the 3D ray direction for each pixel using the camera
        intrinsic parameters. The ray originates at the camera center and
        passes through the pixel.
        
        Args:
            pixel_x: X pixel coordinate(s) (can be scalar or array).
            pixel_y: Y pixel coordinate(s) (can be scalar or array).
            camera_info: Camera info dict (alternative to individual params).
            fx, fy, cx, cy: Camera intrinsic parameters (alternative to camera_info).
            
        Returns:
            Dictionary containing:
            - 'ray': Normalized ray direction(s) in camera frame
            - 'ray_unnormalized': Unnormalized ray direction(s)
            
        Example:
            # Single pixel to ray
            ray_data = sim.pixel_to_ray(320, 240, camera_info=camera)
            ray = ray_data['ray']  # Unit vector in camera frame
            
            # Multiple pixels
            pixels_x = np.array([100, 200, 300])
            pixels_y = np.array([150, 250, 350])
            rays = sim.pixel_to_ray(pixels_x, pixels_y, fx=525, fy=525, cx=320, cy=240)
        """
        # Get intrinsics
        if camera_info is not None:
            intrinsics = self.get_camera_intrinsics(camera_info)
            fx = intrinsics['fx']
            fy = intrinsics['fy']
            cx = intrinsics['cx']
            cy = intrinsics['cy']
        elif fx is None or fy is None or cx is None or cy is None:
            raise ValueError(
                "Must provide either camera_info or all intrinsic parameters (fx, fy, cx, cy)"
            )
        
        # Convert to numpy arrays for vectorized operations
        px = np.asarray(pixel_x)
        py = np.asarray(pixel_y)
        
        # Convert pixel coordinates to normalized image coordinates
        # In camera frame: Z points forward, X right, Y down
        x_norm = (px - cx) / fx
        y_norm = (py - cy) / fy
        z_norm = np.ones_like(x_norm)
        
        # Stack into ray vectors
        if x_norm.ndim == 0:  # Scalar input
            ray_unnormalized = np.array([x_norm, y_norm, z_norm])
            ray_normalized = ray_unnormalized / np.linalg.norm(ray_unnormalized)
        else:  # Array input
            ray_unnormalized = np.stack([x_norm, y_norm, z_norm], axis=-1)
            ray_norms = np.linalg.norm(ray_unnormalized, axis=-1, keepdims=True)
            ray_normalized = ray_unnormalized / ray_norms
        
        return {
            'ray': ray_normalized,
            'ray_unnormalized': ray_unnormalized,
        }
    
    def pixel_to_world(
        self,
        pixel_x: Union[float, np.ndarray],
        pixel_y: Union[float, np.ndarray],
        depth: Union[float, np.ndarray],
        camera_info: Dict[str, Any],
        return_camera_frame: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Convert pixel coordinates and depth to 3D world coordinates.
        
        This performs the full projection from 2D pixel + depth to 3D world
        coordinates using the camera intrinsics and extrinsics.
        
        Args:
            pixel_x: X pixel coordinate(s).
            pixel_y: Y pixel coordinate(s).
            depth: Depth value(s) in meters.
            camera_info: Camera info dict containing intrinsics and pose.
            return_camera_frame: If True, also return points in camera frame.
            
        Returns:
            Dictionary containing:
            - 'world_points': 3D points in world coordinates (N, 3) or (3,)
            - 'camera_points': 3D points in camera frame (if requested)
            
        Example:
            # Single pixel to world
            camera = sim.create_fixed_camera()
            img_data = sim.get_camera_image(camera)
            depth_at_pixel = img_data['depth'][240, 320]
            
            world_pos = sim.pixel_to_world(
                pixel_x=320,
                pixel_y=240, 
                depth=depth_at_pixel,
                camera_info=camera
            )
            print(f"3D position: {world_pos['world_points']}")
            
            # Multiple pixels
            pixels_x = np.array([100, 200, 300])
            pixels_y = np.array([150, 250, 350])
            depths = img_data['depth'][pixels_y, pixels_x]
            
            world_pos = sim.pixel_to_world(
                pixels_x, pixels_y, depths,
                camera_info=camera
            )
        """
        # Get camera intrinsics
        intrinsics = self.get_camera_intrinsics(camera_info)
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        
        # Convert to numpy arrays
        px = np.asarray(pixel_x)
        py = np.asarray(pixel_y)
        z = np.asarray(depth)
        
        # Convert pixels to camera coordinates using intrinsics
        x_cam = (px - cx) * z / fx
        y_cam = (py - cy) * z / fy
        z_cam = z
        
        # Stack into 3D points
        if x_cam.ndim == 0:  # Scalar input
            camera_points = np.array([x_cam, y_cam, z_cam])
        else:  # Array input
            camera_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        # Transform to world coordinates if camera pose is available
        if 'view_matrix' in camera_info:
            # Use inverse of view matrix (camera to world transform)
            view_matrix = camera_info['view_matrix']
            cam_to_world = np.linalg.inv(view_matrix)
            
            # Convert to homogeneous coordinates
            if camera_points.ndim == 1:
                points_homo = np.append(camera_points, 1.0)
                world_points = (cam_to_world @ points_homo)[:3]
            else:
                ones = np.ones((camera_points.shape[0], 1))
                points_homo = np.hstack([camera_points, ones])
                world_points = (cam_to_world @ points_homo.T).T[:, :3]
        else:
            # No camera pose available, return camera frame points
            world_points = camera_points
            logger.warning("No view_matrix in camera_info, returning camera frame coordinates")
        
        result = {'world_points': world_points}
        if return_camera_frame:
            result['camera_points'] = camera_points
        
        return result
    
    def world_to_pixel(
        self,
        world_points: np.ndarray,
        camera_info: Dict[str, Any],
        return_depth: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Project 3D world points to 2D pixel coordinates.
        
        This is the inverse of pixel_to_world, projecting 3D points onto
        the image plane using the camera matrices.
        
        Args:
            world_points: 3D points in world coordinates (N, 3) or (3,).
            camera_info: Camera info dict containing matrices.
            return_depth: If True, also return depth values.
            
        Returns:
            Dictionary containing:
            - 'pixels': 2D pixel coordinates (N, 2) or (2,)
            - 'depths': Depth values if requested
            - 'valid': Boolean mask for points in front of camera
            
        Example:
            # Project single 3D point
            camera = sim.create_fixed_camera()
            world_point = np.array([0.5, 0.0, 0.4])  # 3D position
            
            projection = sim.world_to_pixel(world_point, camera)
            pixel = projection['pixels']
            print(f"Projects to pixel: ({pixel[0]:.1f}, {pixel[1]:.1f})")
            
            # Project multiple points
            points = np.array([[0.5, 0.0, 0.4],
                              [0.6, 0.1, 0.3],
                              [0.4, -0.1, 0.5]])
            projection = sim.world_to_pixel(points, camera, return_depth=True)
        """
        # Ensure world_points is numpy array
        points = np.asarray(world_points)
        if points.ndim == 1:
            points = points.reshape(1, 3)
            single_point = True
        else:
            single_point = False
        
        # Get camera matrices
        if 'view_matrix' in camera_info and 'projection_matrix' in camera_info:
            view_mat = camera_info['view_matrix']
            proj_mat = camera_info['projection_matrix']
        else:
            raise ValueError("camera_info must contain view_matrix and projection_matrix")
        
        # Get intrinsics for pixel conversion
        intrinsics = self.get_camera_intrinsics(camera_info)
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        
        # Transform world points to camera frame
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        camera_points = (view_mat @ points_homo.T).T
        
        # Extract camera frame coordinates
        x_cam = camera_points[:, 0]
        y_cam = camera_points[:, 1]
        z_cam = camera_points[:, 2]
        
        # Check which points are in front of camera
        valid = z_cam > 0
        
        # Project to pixel coordinates
        pixels_x = fx * x_cam / z_cam + cx
        pixels_y = fy * y_cam / z_cam + cy
        pixels = np.stack([pixels_x, pixels_y], axis=-1)
        
        # Handle single point case
        if single_point:
            pixels = pixels[0]
            z_cam = z_cam[0]
            valid = valid[0]
        
        result = {
            'pixels': pixels,
            'valid': valid
        }
        
        if return_depth:
            result['depths'] = z_cam
        
        return result
    
    def get_pixel_3d_position(
        self,
        pixel_x: int,
        pixel_y: int,
        camera_info: Dict[str, Any],
        depth_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Get 3D world position of a specific pixel using its depth value.
        
        This is a convenience function that combines depth lookup with
        pixel-to-world conversion.
        
        Args:
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.
            camera_info: Camera info dict.
            depth_image: Depth image array. If None, captures new image.
            
        Returns:
            3D position in world coordinates, or None if invalid depth.
            
        Example:
            camera = sim.create_fixed_camera()
            
            # Click on pixel (320, 240) and get its 3D position
            pos = sim.get_pixel_3d_position(320, 240, camera)
            if pos is not None:
                print(f"3D position: {pos}")
                
                # Place a marker at that position
                sim.spawn_block(
                    color_rgb=(1, 0, 0),
                    position=tuple(pos),
                    size=0.01
                )
        """
        # Get depth image if not provided
        if depth_image is None:
            img_data = self.get_camera_image(camera_info, get_rgb=False, get_depth=True)
            depth_image = img_data['depth']
        
        # Check pixel bounds
        height, width = depth_image.shape
        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
            logger.warning(f"Pixel ({pixel_x}, {pixel_y}) out of bounds for image {width}x{height}")
            return None
        
        # Get depth at pixel
        depth = depth_image[pixel_y, pixel_x]
        
        # Check for valid depth
        if depth <= 0 or depth >= camera_info.get('far', 10.0):
            logger.debug(f"Invalid depth {depth} at pixel ({pixel_x}, {pixel_y})")
            return None
        
        # Convert to world coordinates
        world_pos = self.pixel_to_world(
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            depth=depth,
            camera_info=camera_info
        )
        
        return world_pos['world_points']
    
    def compute_depth_map_point_cloud(
        self,
        depth_image: np.ndarray,
        camera_info: Dict[str, Any],
        rgb_image: Optional[np.ndarray] = None,
        max_depth: Optional[float] = None,
        downsample: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Convert entire depth map to 3D point cloud with optimized computation.
        
        This efficiently converts all pixels in a depth map to 3D world coordinates,
        optionally including RGB colors.
        
        Args:
            depth_image: 2D depth map array.
            camera_info: Camera info dict with intrinsics and extrinsics.
            rgb_image: Optional RGB image for colored point cloud.
            max_depth: Maximum depth to include (filters far points).
            downsample: Downsampling factor (1 = no downsampling).
            
        Returns:
            Dictionary containing:
            - 'points': 3D point positions (N, 3)
            - 'colors': RGB colors if provided (N, 3)
            - 'pixels': Source pixel coordinates (N, 2)
            
        Example:
            camera = sim.create_fixed_camera()
            images = sim.get_camera_image(camera)
            
            # Generate point cloud
            cloud = sim.compute_depth_map_point_cloud(
                depth_image=images['depth'],
                camera_info=camera,
                rgb_image=images['rgb'],
                downsample=2  # Use every 2nd pixel
            )
            
            points = cloud['points']  # (N, 3) array
            colors = cloud['colors']  # (N, 3) array
            
            print(f"Generated {len(points)} 3D points")
        """
        height, width = depth_image.shape
        
        # Get camera intrinsics
        intrinsics = self.get_camera_intrinsics(camera_info)
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        
        # Create pixel coordinate grids with downsampling
        y_indices = np.arange(0, height, downsample)
        x_indices = np.arange(0, width, downsample)
        xx, yy = np.meshgrid(x_indices, y_indices)
        
        # Get depth values at sampled pixels
        z = depth_image[yy, xx]
        
        # Filter by depth
        if max_depth is None:
            max_depth = camera_info.get('far', 10.0) * 0.99
        
        valid_mask = (z > 0.001) & (z < max_depth)
        
        # Get valid coordinates and depths
        valid_x = xx[valid_mask]
        valid_y = yy[valid_mask]
        valid_z = z[valid_mask]
        
        # Convert to camera coordinates
        x_cam = (valid_x - cx) * valid_z / fx
        y_cam = (valid_y - cy) * valid_z / fy
        z_cam = valid_z
        
        # Stack into points
        camera_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        # Transform to world coordinates
        if 'view_matrix' in camera_info:
            view_matrix = camera_info['view_matrix']
            cam_to_world = np.linalg.inv(view_matrix)
            
            # Apply transformation
            ones = np.ones((camera_points.shape[0], 1))
            points_homo = np.hstack([camera_points, ones])
            world_points = (cam_to_world @ points_homo.T).T[:, :3]
        else:
            world_points = camera_points
        
        result = {
            'points': world_points,
            'pixels': np.stack([valid_x, valid_y], axis=-1)
        }
        
        # Add colors if RGB provided
        if rgb_image is not None:
            valid_colors = rgb_image[valid_y, valid_x]
            if valid_colors.dtype == np.uint8:
                valid_colors = valid_colors.astype(np.float32) / 255.0
            result['colors'] = valid_colors
        
        return result

    def disconnect(self):
        """
        Remove an object from the simulation.

        Args:
            object_name: Name of the object to remove.

        Raises:
            ValueError: If object not found.
        """
        if object_name not in self.objects:
            raise ValueError(f"Object '{object_name}' not found")
        
        object_id = self.objects[object_name]
        p.removeBody(object_id)
        del self.objects[object_name]
        
        logger.info(f"Removed object '{object_name}' (id={object_id})")

    def get_object_state(self, object_name: str) -> Dict[str, Any]:
        """
        Get the state of an object.

        Args:
            object_name: Name of the object.

        Returns:
            Dictionary containing position, orientation, and velocities.
        """
        if object_name not in self.objects:
            raise ValueError(f"Object '{object_name}' not found")
        
        object_id = self.objects[object_name]
        
        # Get position and orientation
        pos, orn = p.getBasePositionAndOrientation(object_id)
        
        # Get velocities
        lin_vel, ang_vel = p.getBaseVelocity(object_id)
        
        return {
            "position": pos,
            "orientation": orn,
            "linear_velocity": lin_vel,
            "angular_velocity": ang_vel,
        }

    def step(self):
        """Perform one simulation step."""
        if not self._is_connected:
            raise RuntimeError("Not connected to PyBullet")

        p.stepSimulation()
        self._step_counter += 1

        # Sleep if real-time simulation is enabled and in GUI mode
        if self.config.use_real_time and self._simulation_mode == SimulationMode.GUI:
            time.sleep(self.config.time_step)

    def disconnect(self):
        """Disconnect from PyBullet."""
        if self._is_connected:
            p.disconnect()
            self._is_connected = False
            self.physics_client = None
            self.robots.clear()
            self.objects.clear()
            self.constraints.clear()
            self._step_counter = 0
            logger.info("Disconnected from PyBullet")

    def get_ids(self) -> Dict[str, Any]:
        """
        Get all loaded IDs.

        Returns:
            Dictionary containing plane_id, robot IDs, and object IDs.
        """
        return {
            "plane_id": self.plane_id,
            "robots": {name: info.robot_id for name, info in self.robots.items()},
            "objects": self.objects.copy(),
            "constraints": self.constraints.copy(),
        }

    def is_connected(self) -> bool:
        """Check if connected to PyBullet."""
        return self._is_connected

    def get_simulation_info(self) -> Dict[str, Any]:
        """Get simulation information."""
        return {
            "connected": self._is_connected,
            "mode": self._simulation_mode.name if self._simulation_mode else None,
            "gravity": self.config.gravity,
            "time_step": self.config.time_step,
            "steps": self._step_counter,
            "num_robots": len(self.robots),
            "num_objects": len(self.objects),
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def __del__(self):
        """Cleanup on deletion."""
        if self._is_connected:
            self.disconnect()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create simulator
    sim = RobotSimulator()

    try:
        # Connect to PyBullet
        sim.connect()

        # Load plane
        plane_id = sim.load_plane()
        print(f"Plane ID: {plane_id}")

        # Load KUKA iiwa robot
        kuka_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="kuka_1",
        )
        print(f"KUKA Robot: {kuka_info}")

        # Load Franka Panda robot
        panda_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            position=(1, 0, 0),
            fixed_base=True,
            robot_name="panda_1",
        )
        print(f"Panda Robot: {panda_info}")

        # Spawn a static platform
        platform = sim.spawn_platform(
            color_rgb=(0.5, 0.5, 0.5),  # Gray
            size=0.1,
            position=(0.6, 0.2, 0.05),
        )
        print(f"Platform ID: {platform}")

        # Spawn a table
        table = sim.spawn_table(
            position=(0.3, 0.0, 0.0),
            table_height=0.3,
            table_size=0.2,
        )
        print(f"Table ID: {table}")

        # Spawn colored blocks
        red_block = sim.spawn_block(
            color_rgb=(1.0, 0.0, 0.0),  # Red
            size=0.03,
            position=(0.6, 0.2, 0.15),  # Drop on platform
        )
        print(f"Red block ID: {red_block}")

        blue_block = sim.spawn_block(
            color_rgb=(0.0, 0.0, 1.0),  # Blue
            size=0.04,
            position=(0.3, 0.0, 0.4),  # Drop on table
            block_name="blue_cube",
        )
        print(f"Blue block ID: {blue_block}")

        # Spawn random blocks
        random_blocks = sim.spawn_random_blocks(num_blocks=3)
        print(f"Random block IDs: {random_blocks}")

        # Get all IDs
        all_ids = sim.get_ids()
        print(f"All IDs: {all_ids}")

        # Get simulation info
        sim_info = sim.get_simulation_info()
        print(f"Simulation Info: {sim_info}")

        # Run simulation for a few steps
        if sim._simulation_mode == SimulationMode.GUI:
            print("Running simulation for 1000 steps...")
            for _ in range(1000):
                sim.step()
                time.sleep(0.01)

    finally:
        # Disconnect
        sim.disconnect()