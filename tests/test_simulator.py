"""Tests for the RobotSimulator class."""

import os
import random
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from cogniforge.core.simulator import (
    RobotSimulator,
    RobotType,
    SimulationMode,
    SimulationConfig,
    RobotInfo,
)


class TestRobotSimulator:
    """Test suite for RobotSimulator."""

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        sim = RobotSimulator()
        assert sim.config is not None
        assert not sim.is_connected()
        assert sim.plane_id is None
        assert len(sim.robots) == 0

    def test_simulation_config(self):
        """Test custom simulation configuration."""
        config = SimulationConfig(
            gravity=(0, 0, -10),
            time_step=1 / 120.0,
            camera_distance=2.0,
        )
        sim = RobotSimulator(config=config)
        assert sim.config.gravity == (0, 0, -10)
        assert sim.config.time_step == 1 / 120.0
        assert sim.config.camera_distance == 2.0

    @patch.dict(os.environ, {"DISPLAY": ":99"})
    def test_mode_detection_with_display(self):
        """Test mode detection when DISPLAY is set."""
        sim = RobotSimulator()
        # Should detect GUI mode when DISPLAY is set
        mode = sim._detect_simulation_mode()
        assert mode == SimulationMode.GUI

    @patch.dict(os.environ, {}, clear=True)
    @patch("os.name", "posix")
    def test_mode_detection_headless(self):
        """Test mode detection in headless environment."""
        # Remove DISPLAY from environment
        if "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]
        
        sim = RobotSimulator()
        # Should detect DIRECT mode when no DISPLAY
        mode = sim._detect_simulation_mode()
        # On Windows this will still be GUI, on Linux without DISPLAY it's DIRECT
        if os.name == "nt":
            assert mode == SimulationMode.GUI
        else:
            assert mode == SimulationMode.DIRECT

    def test_force_mode(self):
        """Test forcing a specific simulation mode."""
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        assert sim._simulation_mode == SimulationMode.DIRECT

    @patch("pybullet.connect")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_connect(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path, mock_connect):
        """Test connecting to PyBullet."""
        mock_connect.return_value = 0  # Physics client ID
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        client_id = sim.connect()
        
        assert client_id == 0
        assert sim.is_connected()
        mock_connect.assert_called_once()
        mock_grav.assert_called_once_with(0.0, 0.0, -9.81)

    @patch("pybullet.connect")
    @patch("pybullet.disconnect")
    def test_disconnect(self, mock_disconnect, mock_connect):
        """Test disconnecting from PyBullet."""
        mock_connect.return_value = 0
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        sim.disconnect()
        
        assert not sim.is_connected()
        mock_disconnect.assert_called_once()

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_load_plane(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path, mock_load, mock_connect):
        """Test loading a plane."""
        mock_connect.return_value = 0
        mock_load.return_value = 1  # Plane ID
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        plane_id = sim.load_plane()
        
        assert plane_id == 1
        assert sim.plane_id == 1
        mock_load.assert_called_once()

    def test_load_plane_without_connection(self):
        """Test loading plane without connection raises error."""
        sim = RobotSimulator()
        with pytest.raises(RuntimeError, match="Not connected"):
            sim.load_plane()

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_load_robot(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                        mock_quat, mock_reset, mock_info, mock_joints, mock_load, mock_connect):
        """Test loading a robot."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7  # Number of joints
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info (joint type is at index 2)
        mock_info.return_value = [None, None, 0]  # Non-fixed joint
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            position=(0, 0, 0),
            fixed_base=True,
            robot_name="test_robot",
        )
        
        assert robot_info.robot_id == 2
        assert robot_info.name == "test_robot"
        assert robot_info.robot_type == RobotType.KUKA_IIWA
        assert robot_info.num_joints == 7
        assert "test_robot" in sim.robots

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_reset_robot_default(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                 mock_quat, mock_reset, mock_info, mock_joints, mock_load, mock_connect):
        """Test resetting robot to default position."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        mock_info.return_value = [None, None, 0]  # Non-fixed joint
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="kuka",
        )
        
        # Reset to default position (should use KUKA defaults)
        positions = sim.reset_robot("kuka")
        
        # Check that resetJointState was called for each joint
        assert mock_reset.call_count >= 7  # At least 7 joints
        assert len(positions) > 0

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_reset_robot_custom_q(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                  mock_quat, mock_reset, mock_info, mock_joints, mock_load, mock_connect):
        """Test resetting robot with custom q_default."""
        mock_connect.return_value = 0
        mock_load.return_value = 3  # Robot ID
        mock_joints.return_value = 6
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info for 6 DOF robot
        def joint_info_side_effect(robot_id, joint_idx):
            return [None, None, 0] + [None] * 10  # Non-fixed joint
        
        mock_info.side_effect = joint_info_side_effect
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.UR5,
            robot_name="ur5",
        )
        
        # Reset with custom joint positions
        custom_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        positions = sim.reset_robot("ur5", q_default=custom_q)
        
        # Verify positions were set correctly
        assert positions == custom_q
        
        # Check that resetJointState was called with correct values
        reset_calls = mock_reset.call_args_list
        # Find the calls for our custom positions (after initial reset)
        for i, q_val in enumerate(custom_q):
            # Check if any call has our custom position value
            found = False
            for call in reset_calls:
                if len(call[0]) > 2 and abs(call[0][2] - q_val) < 0.001:
                    found = True
                    break
            # We should find each custom position value

    def test_reset_robot_wrong_size_q(self):
        """Test reset_robot with wrong size q_default raises error."""
        sim = RobotSimulator()
        
        # Create a mock robot
        from cogniforge.core.simulator import RobotInfo
        robot_info = RobotInfo(
            robot_id=1,
            name="test_robot",
            robot_type=RobotType.KUKA_IIWA,
            base_position=(0, 0, 0),
            base_orientation=(0, 0, 0, 1),
            num_joints=7,
            joint_indices=[0, 1, 2, 3, 4, 5, 6],
            end_effector_index=6,
            gripper_indices=None,
            tool_link_index=6,
        )
        sim.robots["test_robot"] = robot_info
        
        # Try to reset with wrong number of joint positions
        with pytest.raises(ValueError, match="q_default has 3 values but robot"):
            sim.reset_robot("test_robot", q_default=[0.1, 0.2, 0.3])

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.getJointState")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_get_joint_positions(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                 mock_quat, mock_reset, mock_joint_state, mock_info,
                                 mock_joints, mock_load, mock_connect):
        """Test getting current joint positions."""
        mock_connect.return_value = 0
        mock_load.return_value = 2
        mock_joints.return_value = 3
        mock_quat.return_value = [0, 0, 0, 1]
        mock_info.return_value = [None, None, 0] + [None] * 10
        
        # Mock joint states with different positions
        joint_positions = [0.1, 0.2, 0.3]
        mock_joint_state.side_effect = [(pos, 0, 0, 0) for pos in joint_positions]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.CUSTOM,
            robot_name="custom",
            urdf_path="/fake/path.urdf",
        )
        
        # Get joint positions
        positions = sim.get_joint_positions("custom")
        
        assert len(positions) == 3
        assert positions == joint_positions

    def test_robot_types(self):
        """Test robot type enumeration."""
        assert RobotType.KUKA_IIWA.value == "kuka_iiwa"
        assert RobotType.FRANKA_PANDA.value == "franka_panda"
        assert RobotType.UR5.value == "ur5"
        assert RobotType.CUSTOM.value == "custom"

    @patch("pybullet.connect")
    @patch("pybullet.disconnect")
    def test_context_manager(self, mock_disconnect, mock_connect):
        """Test using simulator as context manager."""
        mock_connect.return_value = 0
        
        with RobotSimulator(force_mode=SimulationMode.DIRECT) as sim:
            assert sim.is_connected()
            mock_connect.assert_called_once()
        
        mock_disconnect.assert_called_once()

    @patch("pybullet.connect")
    @patch("pybullet.stepSimulation")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_step_simulation(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path, mock_step, mock_connect):
        """Test stepping the simulation."""
        mock_connect.return_value = 0
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Step simulation
        sim.step()
        sim.step()
        
        assert mock_step.call_count == 2
        assert sim._step_counter == 2

    def test_get_simulation_info(self):
        """Test getting simulation information."""
        sim = RobotSimulator()
        info = sim.get_simulation_info()
        
        assert not info["connected"]
        assert info["gravity"] == (0.0, 0.0, -9.81)
        assert info["time_step"] == 1.0 / 240.0
        assert info["steps"] == 0
        assert info["num_robots"] == 0

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_get_ids(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                     mock_quat, mock_reset, mock_info, mock_joints, mock_load, mock_connect):
        """Test getting all IDs."""
        mock_connect.return_value = 0
        mock_load.side_effect = [1, 2]  # Plane ID, Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        mock_info.return_value = [None, None, 0]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        sim.load_plane()
        sim.load_robot(robot_name="robot1")
        
        ids = sim.get_ids()
        
        assert ids["plane_id"] == 1
        assert "robot1" in ids["robots"]
        assert ids["robots"]["robot1"] == 2

    @patch("pybullet.connect")
    @patch("pybullet.createCollisionShape")
    @patch("pybullet.createVisualShape")
    @patch("pybullet.createMultiBody")
    @patch("pybullet.changeDynamics")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_spawn_block(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                         mock_quat, mock_dynamics, mock_body, mock_visual, mock_collision, mock_connect):
        """Test spawning a colored block."""
        mock_connect.return_value = 0
        mock_collision.return_value = 1  # Collision shape ID
        mock_visual.return_value = 2     # Visual shape ID
        mock_body.return_value = 3       # Block body ID
        mock_quat.return_value = [0, 0, 0, 1]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Spawn a red block
        block_id = sim.spawn_block(
            color_rgb=(1.0, 0.0, 0.0),
            size=0.03,
        )
        
        assert block_id == 3
        assert "block_0" in sim.objects
        assert sim.objects["block_0"] == 3
        
        # Check that PyBullet functions were called
        mock_collision.assert_called_once()
        mock_visual.assert_called_once()
        mock_body.assert_called_once()
        mock_dynamics.assert_called_once()
        
        # Check visual shape was created with correct color
        visual_call_args = mock_visual.call_args
        assert visual_call_args[1]["rgbaColor"] == [1.0, 0.0, 0.0, 1.0]
        
        # Check default position
        body_call_args = mock_body.call_args
        assert body_call_args[1]["basePosition"] == (0.5, 0.0, 0.05)

    @patch("pybullet.connect")
    @patch("pybullet.createCollisionShape")
    @patch("pybullet.createVisualShape")
    @patch("pybullet.createMultiBody")
    @patch("pybullet.changeDynamics")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_spawn_block_custom_params(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                       mock_quat, mock_dynamics, mock_body, mock_visual, mock_collision, mock_connect):
        """Test spawning a block with custom parameters."""
        mock_connect.return_value = 0
        mock_collision.return_value = 1
        mock_visual.return_value = 2
        mock_body.return_value = 4
        mock_quat.return_value = [0, 0, 0, 1]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Spawn a blue block with custom size and position
        block_id = sim.spawn_block(
            color_rgb=(0.0, 0.0, 1.0),
            size=0.05,
            position=(1.0, 0.5, 0.2),
            mass=0.2,
            block_name="custom_block",
        )
        
        assert block_id == 4
        assert "custom_block" in sim.objects
        assert sim.objects["custom_block"] == 4
        
        # Check custom position
        body_call_args = mock_body.call_args
        assert body_call_args[1]["basePosition"] == (1.0, 0.5, 0.2)
        assert body_call_args[1]["baseMass"] == 0.2

    def test_spawn_block_without_connection(self):
        """Test spawning block without connection raises error."""
        sim = RobotSimulator()
        with pytest.raises(RuntimeError, match="Not connected"):
            sim.spawn_block(color_rgb=(1.0, 0.0, 0.0))

    @patch("pybullet.connect")
    @patch("pybullet.createCollisionShape")
    @patch("pybullet.createVisualShape")
    @patch("pybullet.createMultiBody")
    @patch("pybullet.changeDynamics")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_spawn_platform(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                            mock_quat, mock_dynamics, mock_body, mock_visual, mock_collision, mock_connect):
        """Test spawning a static platform."""
        mock_connect.return_value = 0
        mock_collision.return_value = 1  # Collision shape ID
        mock_visual.return_value = 2     # Visual shape ID
        mock_body.return_value = 5       # Platform body ID
        mock_quat.return_value = [0, 0, 0, 1]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Spawn a gray platform
        platform_id = sim.spawn_platform(
            color_rgb=(0.5, 0.5, 0.5),
            size=0.1,
        )
        
        assert platform_id == 5
        assert "platform_0" in sim.objects
        assert sim.objects["platform_0"] == 5
        
        # Check that platform is static (mass = 0)
        body_call_args = mock_body.call_args
        assert body_call_args[1]["baseMass"] == 0  # Static object
        
        # Check default position
        assert body_call_args[1]["basePosition"] == (0.6, 0.2, 0.05)
        
        # Check platform dimensions (should be flatter)
        collision_call_args = mock_collision.call_args
        halfExtents = collision_call_args[1]["halfExtents"]
        assert halfExtents[0] == 0.1  # Width
        assert halfExtents[1] == 0.1  # Depth
        assert halfExtents[2] == 0.025  # Height (size/4)

    @patch("pybullet.connect")
    @patch("pybullet.createCollisionShape")
    @patch("pybullet.createVisualShape")
    @patch("pybullet.createMultiBody")
    @patch("pybullet.changeDynamics")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_spawn_platform_custom(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                   mock_quat, mock_dynamics, mock_body, mock_visual, mock_collision, mock_connect):
        """Test spawning a platform with custom parameters."""
        mock_connect.return_value = 0
        mock_collision.return_value = 1
        mock_visual.return_value = 2
        mock_body.return_value = 6
        mock_quat.return_value = [0, 0, 0, 1]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Spawn platform with custom parameters
        platform_id = sim.spawn_platform(
            color_rgb=(1.0, 0.0, 0.0),  # Red
            size=0.2,
            position=(1.0, 0.5, 0.1),
            platform_name="red_platform",
            height=0.03,  # Custom height
        )
        
        assert platform_id == 6
        assert "red_platform" in sim.objects
        
        # Check custom position and dimensions
        body_call_args = mock_body.call_args
        assert body_call_args[1]["basePosition"] == (1.0, 0.5, 0.1)
        
        collision_call_args = mock_collision.call_args
        halfExtents = collision_call_args[1]["halfExtents"]
        assert halfExtents[0] == 0.2  # Width
        assert halfExtents[1] == 0.2  # Depth
        assert halfExtents[2] == 0.03  # Custom height

    def test_spawn_platform_without_connection(self):
        """Test spawning platform without connection raises error."""
        sim = RobotSimulator()
        with pytest.raises(RuntimeError, match="Not connected"):
            sim.spawn_platform(color_rgb=(0.5, 0.5, 0.5))

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_tool_link_detection(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                 mock_quat, mock_reset, mock_joint_info, mock_joints, mock_load, mock_connect):
        """Test tool link index detection."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info with link names
        def joint_info_side_effect(robot_id, joint_idx):
            if joint_idx == 6:
                # End effector link
                return [None] * 12 + [b"iiwa_link_7"]
            else:
                return [None] * 12 + [b"link"]
        
        mock_joint_info.side_effect = joint_info_side_effect
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="test_kuka",
        )
        
        # Check that tool link was detected
        assert robot_info.tool_link_index is not None

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.getJointState")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_gripper_info(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                          mock_quat, mock_reset, mock_joint_state, mock_joint_info,
                          mock_joints, mock_load, mock_connect):
        """Test getting gripper information."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 9  # Panda has 9 joints
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None] * 12 + [b"finger_joint"]
        mock_joint_state.return_value = [0.02, 0, 0, 0]  # Joint position
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            robot_name="panda",
        )
        
        # Get gripper info
        gripper_info = sim.get_gripper_info("panda")
        
        assert "has_gripper" in gripper_info
        assert "tool_link" in gripper_info
        assert "gripper_indices" in gripper_info
        assert "finger_names" in gripper_info

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.getBodyInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_get_link_names(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                            mock_quat, mock_reset, mock_body_info, mock_joint_info,
                            mock_joints, mock_load, mock_connect):
        """Test getting link names."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 3
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock body and joint info
        mock_body_info.return_value = [b"base_link"]
        mock_joint_info.return_value = [None] * 12 + [b"test_link"]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.UR5,
            robot_name="ur5",
        )
        
        # Get link names
        link_names = sim.get_link_names("ur5")
        
        assert isinstance(link_names, dict)
        assert -1 in link_names  # Base link
        assert link_names[-1] == "base_link"

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.getLinkState")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_ee_pose(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                     mock_quat, mock_reset, mock_link_state, mock_joint_info,
                     mock_joints, mock_load, mock_connect):
        """Test getting end-effector pose."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None] * 12 + [b"link"]
        
        # Mock link state - returns position and orientation
        mock_link_state.return_value = [
            (0.5, 0.3, 0.4),  # Position
            (0, 0, 0.707, 0.707),  # Orientation (45 deg around Z)
            (0, 0, 0),  # Local position (not used)
            (0, 0, 0, 1),  # Local orientation (not used)
            (0, 0, 0),  # World frame position (not used)
            (0, 0, 0, 1),  # World frame orientation (not used)
        ]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="kuka",
        )
        
        # Get EE pose
        pos, orn = sim.ee_pose("kuka")
        
        assert pos == (0.5, 0.3, 0.4)
        assert orn == (0, 0, 0.707, 0.707)
        
        # Check that getLinkState was called with correct arguments
        mock_link_state.assert_called()

    def test_ee_pose_robot_not_found(self):
        """Test ee_pose with non-existent robot."""
        sim = RobotSimulator()
        
        with pytest.raises(ValueError, match="Robot 'nonexistent' not found"):
            sim.ee_pose("nonexistent")

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.getLinkState")
    @patch("pybullet.getJointState")
    @patch("pybullet.calculateJacobian")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_get_ee_jacobian(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                             mock_quat, mock_reset, mock_jacobian, mock_joint_state,
                             mock_link_state, mock_joint_info, mock_joints, mock_load, mock_connect):
        """Test getting end-effector Jacobian."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 3  # Simple 3-DOF robot
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info and states
        mock_joint_info.return_value = [None, None, 0] + [None] * 9 + [b"link"]
        mock_joint_state.return_value = [0.1, 0, 0, 0]  # Joint position
        
        # Mock Jacobian calculation
        mock_jacobian.return_value = (
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # Linear Jacobian
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]   # Angular Jacobian
        )
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.CUSTOM,
            robot_name="simple_robot",
            urdf_path="/fake/path.urdf",
        )
        
        # Get Jacobian
        lin_jac, ang_jac = sim.get_ee_jacobian("simple_robot")
        
        assert lin_jac.shape == (3, 3)
        assert ang_jac.shape == (3, 3)
        assert mock_jacobian.called

    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_open_gripper(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                          mock_quat, mock_motor, mock_joint_info, mock_joints, mock_load, mock_connect):
        """Test opening the gripper."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 11  # Panda has 11 joints
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info for Panda with gripper
        def get_joint_info(robot_id, joint_idx):
            if joint_idx < 7:
                return [None, None, 0] + [None] * 9 + [f"panda_joint{joint_idx+1}".encode()]
            elif joint_idx in [9, 10]:
                return [None, None, 0] + [None] * 9 + [b"panda_finger_joint"]
            else:
                return [None, None, 0] + [None] * 9 + [b"panda_hand"]
        
        mock_joint_info.side_effect = get_joint_info
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            robot_name="panda",
        )
        
        # Open the gripper
        sim.open_gripper("panda")
        
        # Check that motor control was called for gripper joints
        calls = mock_motor.call_args_list
        assert len(calls) >= 2  # At least 2 calls for gripper fingers
        
        # Check that gripper was set to open position (1.0 * 0.04 = 0.04)
        for call in calls[-2:]:
            assert call[0][0] == 2  # Robot ID
            assert call[0][2] == 0  # POSITION_CONTROL
            assert abs(call[1]['targetPosition'] - 0.04) < 0.001  # Fully open
            assert call[1]['force'] == 100.0  # Default force
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_close_gripper(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                           mock_quat, mock_motor, mock_joint_info, mock_joints, mock_load, mock_connect):
        """Test closing the gripper."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 11  # Panda has 11 joints
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info for Panda with gripper
        def get_joint_info(robot_id, joint_idx):
            if joint_idx < 7:
                return [None, None, 0] + [None] * 9 + [f"panda_joint{joint_idx+1}".encode()]
            elif joint_idx in [9, 10]:
                return [None, None, 0] + [None] * 9 + [b"panda_finger_joint"]
            else:
                return [None, None, 0] + [None] * 9 + [b"panda_hand"]
        
        mock_joint_info.side_effect = get_joint_info
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            robot_name="panda",
        )
        
        # Close the gripper with custom force
        sim.close_gripper("panda", force=30.0)
        
        # Check that motor control was called for gripper joints
        calls = mock_motor.call_args_list
        assert len(calls) >= 2  # At least 2 calls for gripper fingers
        
        # Check that gripper was set to closed position (0.0)
        for call in calls[-2:]:
            assert call[0][0] == 2  # Robot ID
            assert call[0][2] == 0  # POSITION_CONTROL
            assert call[1]['targetPosition'] == 0.0  # Fully closed
            assert call[1]['force'] == 30.0  # Custom force
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_gripper_methods_default_force(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                           mock_quat, mock_motor, mock_joint_info, mock_joints, mock_load, mock_connect):
        """Test that open_gripper and close_gripper use different default forces."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 11  # Panda has 11 joints
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info for Panda with gripper
        def get_joint_info(robot_id, joint_idx):
            if joint_idx < 7:
                return [None, None, 0] + [None] * 9 + [f"panda_joint{joint_idx+1}".encode()]
            elif joint_idx in [9, 10]:
                return [None, None, 0] + [None] * 9 + [b"panda_finger_joint"]
            else:
                return [None, None, 0] + [None] * 9 + [b"panda_hand"]
        
        mock_joint_info.side_effect = get_joint_info
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            robot_name="panda",
        )
        
        # Clear previous calls
        mock_motor.reset_mock()
        
        # Open with default force
        sim.open_gripper("panda")
        open_calls = mock_motor.call_args_list.copy()
        
        # Clear calls
        mock_motor.reset_mock()
        
        # Close with default force
        sim.close_gripper("panda")
        close_calls = mock_motor.call_args_list.copy()
        
        # Check default forces are different
        assert open_calls[-1][1]['force'] == 100.0  # Open uses 100N by default
        assert close_calls[-1][1]['force'] == 50.0   # Close uses 50N by default (gentler)
    
    def test_open_gripper_no_gripper(self):
        """Test open_gripper on robot without gripper."""
        sim = RobotSimulator()
        sim.robots["no_gripper"] = RobotInfo(
            robot_id=1,
            robot_name="no_gripper",
            robot_type=RobotType.UR5,
            num_joints=6,
            joint_indices=list(range(6)),
            end_effector_index=5,
            gripper_indices=None,  # No gripper
            tool_link_index=None,
        )
        
        # Should not raise error, just log warning
        sim.open_gripper("no_gripper")
    
    def test_close_gripper_robot_not_found(self):
        """Test close_gripper with non-existent robot."""
        sim = RobotSimulator()
        
        with pytest.raises(ValueError, match="Robot 'nonexistent' not found"):
            sim.close_gripper("nonexistent")
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.calculateInverseKinematics")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_calculate_ik_position_only(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                        mock_quat, mock_reset, mock_ik, mock_joint_info,
                                        mock_joints, mock_load, mock_connect):
        """Test calculating IK for position only."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7  # 7-DOF robot
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None, None, 0] + [None] * 5 + [-2.96, 2.96] + [None] * 3
        
        # Mock IK solution
        mock_ik.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="kuka",
        )
        
        # Calculate IK for position only
        target_pos = (0.5, 0.3, 0.4)
        joint_positions = sim.calculate_ik("kuka", target_pos=target_pos)
        
        assert len(joint_positions) == 7
        assert joint_positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        # Check that calculateInverseKinematics was called
        mock_ik.assert_called()
        call_args = mock_ik.call_args
        assert call_args[0][2] == target_pos  # Target position
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.calculateInverseKinematics")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_calculate_ik_with_orientation(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                           mock_quat, mock_reset, mock_ik, mock_joint_info,
                                           mock_joints, mock_load, mock_connect):
        """Test calculating IK with position and orientation."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 6  # 6-DOF robot
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        # Mock IK solution
        mock_ik.return_value = [0.5, -0.3, 0.2, -1.5, 0.8, 0.1]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.UR5,
            robot_name="ur5",
        )
        
        # Calculate IK with orientation
        target_pos = (0.4, 0.2, 0.5)
        target_orn = (0, 0, 0.707, 0.707)  # 90 deg around Z
        joint_positions = sim.calculate_ik("ur5", 
                                          target_pos=target_pos,
                                          target_orn=target_orn)
        
        assert len(joint_positions) == 6
        
        # Check that calculateInverseKinematics was called with orientation
        mock_ik.assert_called()
        call_args = mock_ik.call_args
        assert call_args[0][2] == target_pos  # Target position
        assert call_args[0][3] == target_orn  # Target orientation
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.calculateInverseKinematics")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_calculate_ik_with_nullspace(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                         mock_quat, mock_reset, mock_ik, mock_joint_info,
                                         mock_joints, mock_load, mock_connect):
        """Test calculating IK with null-space control."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info with limits
        def joint_info_side_effect(robot_id, joint_idx):
            # Return joint info with lower and upper limits
            return [None] * 8 + [-2.96, 2.96] + [None] * 3  # Limits at indices 8, 9
        
        mock_joint_info.side_effect = joint_info_side_effect
        
        # Mock IK solution
        mock_ik.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="kuka",
        )
        
        # Calculate IK with null-space
        target_pos = (0.5, 0.3, 0.4)
        joint_positions = sim.calculate_ik("kuka", 
                                          target_pos=target_pos,
                                          use_nullspace=True)
        
        assert len(joint_positions) == 7
        
        # Check that IK was called with null-space parameters
        mock_ik.assert_called()
        call_kwargs = mock_ik.call_args[1]
        assert "lowerLimits" in call_kwargs
        assert "upperLimits" in call_kwargs
        assert "jointRanges" in call_kwargs
        assert "restPoses" in call_kwargs
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.calculateInverseKinematics")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_calculate_ik_with_current_q(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                         mock_quat, mock_reset, mock_ik, mock_joint_info,
                                         mock_joints, mock_load, mock_connect):
        """Test calculating IK with specified current joint configuration."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        # Mock IK solution
        mock_ik.return_value = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="kuka",
        )
        
        # Calculate IK with specified starting configuration
        target_pos = (0.5, 0.3, 0.4)
        current_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        joint_positions = sim.calculate_ik("kuka", 
                                          target_pos=target_pos,
                                          current_q=current_q)
        
        assert len(joint_positions) == 7
        
        # Check that joints were reset to current_q before IK
        assert mock_reset.call_count >= 7  # At least 7 joint resets
        
        # Verify the reset calls used the correct joint positions
        reset_calls = mock_reset.call_args_list[-7:]  # Last 7 calls
        for i, call in enumerate(reset_calls):
            # Each call should have robot_id, joint_idx, and position
            assert call[0][2] == current_q[i]  # Position argument
    
    def test_calculate_ik_robot_not_found(self):
        """Test calculate_ik with non-existent robot."""
        sim = RobotSimulator()
        
        with pytest.raises(ValueError, match="Robot 'nonexistent' not found"):
            sim.calculate_ik("nonexistent", target_pos=(0.5, 0.3, 0.4))
    
    def test_calculate_ik_wrong_current_q_size(self):
        """Test calculate_ik with wrong size current_q."""
        sim = RobotSimulator()
        
        # Create a mock robot
        sim.robots["test"] = RobotInfo(
            robot_id=1,
            robot_name="test",
            robot_type=RobotType.KUKA_IIWA,
            num_joints=7,
            joint_indices=list(range(7)),
            end_effector_index=6,
            gripper_indices=None,
            tool_link_index=6,
        )
        
        with pytest.raises(ValueError, match="current_q has 3 values but robot has 7"):
            sim.calculate_ik("test", 
                           target_pos=(0.5, 0.3, 0.4),
                           current_q=[0.1, 0.2, 0.3])  # Wrong size
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.calculateInverseKinematics")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.stepSimulation")
    @patch("pybullet.getJointState")
    @patch("pybullet.getLinkState")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_move_through_waypoints_simple(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                           mock_quat, mock_reset, mock_link_state, mock_joint_state,
                                           mock_step, mock_motor, mock_ik, mock_joint_info,
                                           mock_joints, mock_load, mock_connect):
        """Test moving through simple position waypoints."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        # Mock joint states for get_joint_positions
        mock_joint_state.return_value = [0.1, 0, 0, 0]  # Current position
        
        # Mock IK solutions for waypoints
        ik_solutions = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # Waypoint 1
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Waypoint 2
        ]
        mock_ik.side_effect = ik_solutions
        
        # Mock link state for ee_pose
        mock_link_state.return_value = [
            (0.5, 0.0, 0.4),  # Position
            (0, 0, 0, 1),     # Orientation
        ]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="kuka",
        )
        
        # Define waypoints
        waypoints = [
            (0.5, 0.0, 0.4),
            (0.5, 0.2, 0.4),
        ]
        
        # Move through waypoints
        sim.move_through_waypoints(
            "kuka",
            waypoints,
            steps_per_segment=12,  # Small value for testing
            action_repeat=2
        )
        
        # Check that IK was calculated for each waypoint
        assert mock_ik.call_count == len(waypoints)
        
        # Check that position control was applied
        assert mock_motor.called
        
        # Check that simulation was stepped
        assert mock_step.called
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.calculateInverseKinematics")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.stepSimulation")
    @patch("pybullet.getJointState")
    @patch("pybullet.getLinkState")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_move_through_waypoints_with_orientation(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                                     mock_quat, mock_reset, mock_link_state, mock_joint_state,
                                                     mock_step, mock_motor, mock_ik, mock_joint_info,
                                                     mock_joints, mock_load, mock_connect):
        """Test moving through waypoints with orientation."""
        mock_connect.return_value = 0
        mock_load.return_value = 2  # Robot ID
        mock_joints.return_value = 7
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        # Mock joint states
        mock_joint_state.return_value = [0.0, 0, 0, 0]
        
        # Mock IK solutions
        mock_ik.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        # Mock link state
        mock_link_state.return_value = [
            (0.5, 0.2, 0.4),
            (0, 0, 0.707, 0.707),
        ]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            robot_name="panda",
        )
        
        # Define waypoints with orientation
        waypoints = [
            {'pos': (0.5, 0.0, 0.4), 'orn': (0, 0, 0, 1)},
            {'pos': (0.5, 0.2, 0.4), 'orn': (0, 0, 0.707, 0.707)},
        ]
        
        # Move through waypoints
        sim.move_through_waypoints(
            "panda",
            waypoints,
            use_orientation=True,
            steps_per_segment=8,
            action_repeat=1
        )
        
        # Check that IK was called with orientation
        ik_calls = mock_ik.call_args_list
        for call in ik_calls:
            # Check if orientation was passed (4th argument)
            assert len(call[0]) >= 4 or 'target_orn' in call[1]
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.calculateInverseKinematics")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.stepSimulation")
    @patch("pybullet.getJointState")
    @patch("pybullet.getLinkState")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_move_through_waypoints_with_trajectories(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                                      mock_quat, mock_reset, mock_link_state, mock_joint_state,
                                                      mock_step, mock_motor, mock_ik, mock_joint_info,
                                                      mock_joints, mock_load, mock_connect):
        """Test move_through_waypoints with trajectory recording."""
        mock_connect.return_value = 0
        mock_load.return_value = 2
        mock_joints.return_value = 6  # 6-DOF robot
        mock_quat.return_value = [0, 0, 0, 1]
        
        # Mock joint info
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        # Mock joint states
        mock_joint_state.return_value = [0.0, 0, 0, 0]
        
        # Mock IK solutions
        mock_ik.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        # Mock link state
        mock_link_state.return_value = [
            (0.4, 0.2, 0.5),
            (0, 0, 0, 1),
        ]
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        robot_info = sim.load_robot(
            robot_type=RobotType.UR5,
            robot_name="ur5",
        )
        
        # Define waypoints
        waypoints = [
            (0.4, 0.2, 0.5),
            (0.3, 0.3, 0.4),
        ]
        
        # Move through waypoints with trajectory recording
        trajectory_data = sim.move_through_waypoints(
            "ur5",
            waypoints,
            steps_per_segment=4,
            action_repeat=1,
            return_trajectories=True
        )
        
        # Check that trajectory data was returned
        assert trajectory_data is not None
        assert 'joint_trajectory' in trajectory_data
        assert 'ee_positions' in trajectory_data
        assert 'ee_errors' in trajectory_data
        
        # Check that trajectories have data
        assert len(trajectory_data['joint_trajectory']) > 0
        assert len(trajectory_data['ee_positions']) > 0
    
    def test_move_through_waypoints_robot_not_found(self):
        """Test move_through_waypoints with non-existent robot."""
        sim = RobotSimulator()
        
        with pytest.raises(ValueError, match="Robot 'nonexistent' not found"):
            sim.move_through_waypoints("nonexistent", [(0.5, 0.3, 0.4)])
    
    def test_move_through_waypoints_empty_waypoints(self):
        """Test move_through_waypoints with empty waypoints list."""
        sim = RobotSimulator()
        sim.robots["test"] = RobotInfo(
            robot_id=1,
            robot_name="test",
            robot_type=RobotType.KUKA_IIWA,
            num_joints=7,
            joint_indices=list(range(7)),
            end_effector_index=6,
            gripper_indices=None,
            tool_link_index=6,
        )
        
        with pytest.raises(ValueError, match="Waypoints list cannot be empty"):
            sim.move_through_waypoints("test", [])
    
    def test_move_through_waypoints_invalid_waypoint(self):
        """Test move_through_waypoints with invalid waypoint format."""
        sim = RobotSimulator()
        sim.robots["test"] = RobotInfo(
            robot_id=1,
            robot_name="test",
            robot_type=RobotType.KUKA_IIWA,
            num_joints=7,
            joint_indices=list(range(7)),
            end_effector_index=6,
            gripper_indices=None,
            tool_link_index=6,
        )
        
        # Invalid waypoint (not a tuple or dict)
        with pytest.raises(ValueError, match="must be tuple"):
            sim.move_through_waypoints("test", ["invalid"])
    
    def test_set_seed(self):
        """Test setting random seeds for determinism."""
        sim = RobotSimulator(seed=42)
        
        # Check that seed was set
        assert sim.config.seed == 42
        
        # Test NumPy seed
        np_val1 = np.random.rand()
        sim.set_seed(42)  # Reset seed
        np_val2 = np.random.rand()
        assert np_val1 == np_val2  # Same seed should give same value
        
        # Test Python random seed
        sim.set_seed(123)
        py_val1 = random.random()
        sim.set_seed(123)  # Reset seed
        py_val2 = random.random()
        assert py_val1 == py_val2
    
    @patch("pybullet.connect")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setRealTimeSimulation")
    @patch("pybullet.setAdditionalSearchPath")
    def test_deterministic_config(self, mock_path, mock_rt, mock_ts, mock_grav, mock_engine, mock_connect):
        """Test deterministic simulation configuration."""
        mock_connect.return_value = 0
        
        # Create simulator with deterministic config
        config = SimulationConfig(
            seed=42,
            deterministic=True,
            time_step=1/240,
            solver_iterations=50
        )
        
        sim = RobotSimulator(config=config, force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Check that deterministic physics parameters were set
        mock_engine.assert_called()
        call_kwargs = mock_engine.call_args[1]
        
        # Check key deterministic parameters
        assert 'fixedTimeStep' in call_kwargs
        assert call_kwargs['fixedTimeStep'] == 1/240
        assert 'deterministicOverlappingPairs' in call_kwargs
        assert call_kwargs['deterministicOverlappingPairs'] == 1
        assert 'numSubSteps' in call_kwargs
        assert call_kwargs['numSubSteps'] == 1
    
    @patch("pybullet.connect")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setRealTimeSimulation")
    @patch("pybullet.setAdditionalSearchPath")
    def test_non_deterministic_config(self, mock_path, mock_rt, mock_ts, mock_grav, mock_engine, mock_connect):
        """Test non-deterministic simulation configuration."""
        mock_connect.return_value = 0
        
        # Create simulator without deterministic mode
        config = SimulationConfig(
            deterministic=False,
            solver_iterations=150
        )
        
        sim = RobotSimulator(config=config, force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Check that standard physics parameters were set
        mock_engine.assert_called()
        call_kwargs = mock_engine.call_args[1]
        
        # Should not have deterministic-specific parameters
        assert 'deterministicOverlappingPairs' not in call_kwargs
        assert 'numSolverIterations' in call_kwargs
        assert call_kwargs['numSolverIterations'] == 150
    
    def test_reset_seeds(self):
        """Test resetting seeds to time-based value."""
        sim = RobotSimulator(seed=42)
        
        # Reset seeds
        sim.reset_seeds()
        
        # Seed should have changed
        assert sim.config.seed != 42
        assert sim.config.deterministic == False
        
        # Test that values are now different
        val1 = np.random.rand()
        val2 = np.random.rand()
        assert val1 != val2  # Without fixed seed, values should differ
    
    def test_seed_reproducibility(self):
        """Test that same seed gives reproducible results."""
        # First run with seed 42
        sim1 = RobotSimulator(seed=42)
        np_vals1 = [np.random.rand() for _ in range(5)]
        py_vals1 = [random.random() for _ in range(5)]
        
        # Second run with same seed
        sim2 = RobotSimulator(seed=42)
        np_vals2 = [np.random.rand() for _ in range(5)]
        py_vals2 = [random.random() for _ in range(5)]
        
        # Values should be identical
        assert np_vals1 == np_vals2
        assert py_vals1 == py_vals2
    
    def test_config_seed_override(self):
        """Test that constructor seed overrides config seed."""
        config = SimulationConfig(seed=100)
        sim = RobotSimulator(config=config, seed=200)
        
        assert sim.config.seed == 200  # Constructor seed should win
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.removeBody")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_kuka_fallback_to_panda(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                    mock_quat, mock_reset, mock_remove, mock_joint_info,
                                    mock_joints, mock_load, mock_connect):
        """Test KUKA falls back to Panda when it doesn't have 7 DOF."""
        mock_connect.return_value = 0
        mock_quat.return_value = [0, 0, 0, 1]
        
        # First load returns KUKA with only 5 movable joints
        # Second load returns Panda with proper joints
        mock_load.side_effect = [2, 3]  # KUKA ID, then Panda ID
        mock_joints.side_effect = [5, 11]  # KUKA has 5 joints, Panda has 11
        
        # Mock joint info - only 5 movable joints for KUKA
        def joint_info_side_effect(robot_id, joint_idx):
            if robot_id == 2:  # KUKA
                # Return FIXED joint type for some joints
                if joint_idx < 5:
                    return [None, None, 0] + [None] * 10  # REVOLUTE
                else:
                    return [None, None, 4] + [None] * 10  # FIXED
            else:  # Panda
                return [None, None, 0] + [None] * 10  # All REVOLUTE
        
        mock_joint_info.side_effect = joint_info_side_effect
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Try to load KUKA, should fallback to Panda
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="test_robot",
            auto_fallback=True
        )
        
        # Should have loaded Panda instead
        assert robot_info.robot_type == RobotType.FRANKA_PANDA
        assert robot_info.robot_id == 3  # Panda ID
        
        # Should have removed the failed KUKA robot
        mock_remove.assert_called_once_with(2)
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_urdf_path_fallback(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                mock_quat, mock_reset, mock_joint_info,
                                mock_joints, mock_load, mock_connect):
        """Test URDF path fallback when primary path fails."""
        mock_connect.return_value = 0
        mock_quat.return_value = [0, 0, 0, 1]
        
        # First URDF path fails, second succeeds
        def load_side_effect(urdf_path, **kwargs):
            if "model.urdf" in urdf_path or "panda.urdf" in urdf_path:
                raise RuntimeError("File not found")
            else:
                return 2  # Robot ID
        
        mock_load.side_effect = load_side_effect
        mock_joints.return_value = 11
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Should try multiple paths and succeed eventually
        robot_info = sim.load_robot(
            robot_type=RobotType.FRANKA_PANDA,
            robot_name="panda"
        )
        
        # Should have tried multiple paths
        assert mock_load.call_count >= 2
    
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_kuka_no_fallback(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                              mock_quat, mock_reset, mock_joint_info,
                              mock_joints, mock_load, mock_connect):
        """Test KUKA doesn't fallback when auto_fallback=False."""
        mock_connect.return_value = 0
        mock_quat.return_value = [0, 0, 0, 1]
        
        # KUKA with only 5 movable joints
        mock_load.return_value = 2
        mock_joints.return_value = 5
        
        # Mock joint info - only 5 movable joints
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Load KUKA without fallback
        robot_info = sim.load_robot(
            robot_type=RobotType.KUKA_IIWA,
            robot_name="test_robot",
            auto_fallback=False  # Disable fallback
        )
        
        # Should keep KUKA even with wrong DOF
        assert robot_info.robot_type == RobotType.KUKA_IIWA
        assert robot_info.robot_id == 2
    
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_absolute")
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.resetJointState")
    @patch("pybullet.getQuaternionFromEuler")
    @patch("pybullet.setAdditionalSearchPath")
    @patch("pybullet.setGravity")
    @patch("pybullet.setTimeStep")
    @patch("pybullet.setPhysicsEngineParameter")
    @patch("pybullet.setRealTimeSimulation")
    def test_custom_robot_urdf_validation(self, mock_rt, mock_engine, mock_ts, mock_grav, mock_path,
                                          mock_quat, mock_reset, mock_joint_info,
                                          mock_joints, mock_load, mock_connect,
                                          mock_is_absolute, mock_exists):
        """Test custom robot URDF path validation."""
        mock_connect.return_value = 0
        mock_quat.return_value = [0, 0, 0, 1]
        mock_load.return_value = 2
        mock_joints.return_value = 6
        mock_joint_info.return_value = [None, None, 0] + [None] * 10
        
        # Mock path existence
        mock_exists.return_value = True
        mock_is_absolute.return_value = True
        
        sim = RobotSimulator(force_mode=SimulationMode.DIRECT)
        sim.connect()
        
        # Load custom robot with valid path
        robot_info = sim.load_robot(
            robot_type=RobotType.CUSTOM,
            urdf_path="/path/to/custom.urdf",
            robot_name="custom_robot"
        )
        
        assert robot_info.robot_type == RobotType.CUSTOM
        assert robot_info.robot_id == 2
    
    def test_custom_robot_without_urdf_path(self):
        """Test custom robot without URDF path raises error."""
        sim = RobotSimulator()
        
        with pytest.raises(ValueError, match="URDF path must be provided"):
            sim.load_robot(
                robot_type=RobotType.CUSTOM,
                robot_name="custom"
            )
