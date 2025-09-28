"""
Complete End-to-End Test Suite for CogniForge

This script tests all components of the robotic manipulation system.
"""

import sys
import time
import traceback
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_component(name, test_func):
    """Test a single component with error handling"""
    print(f"\n{Fore.YELLOW}Testing: {name}{Style.RESET_ALL}")
    print("="*60)
    
    try:
        test_func()
        print(f"{Fore.GREEN}✅ {name} - PASSED{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ {name} - FAILED{Style.RESET_ALL}")
        print(f"Error: {e}")
        print(traceback.format_exc())
        return False

def test_motion_controller():
    """Test motion controller"""
    from motion_controller import MotionController, MotionConfig
    
    config = MotionConfig(
        use_bc_model=False,  # Don't require trained model for test
        num_waypoints=5
    )
    controller = MotionController(config)
    
    # Test trajectory generation
    trajectory = controller.generate_trajectory(
        start=[0, 0, 0],
        goal=[1, 1, 0.5],
        num_waypoints=5
    )
    
    assert len(trajectory) == 5, "Should generate 5 waypoints"
    print(f"Generated {len(trajectory)} waypoints")

def test_waypoint_optimizer():
    """Test waypoint optimizer"""
    from waypoint_optimizer import WaypointOptimizer
    import numpy as np
    
    optimizer = WaypointOptimizer(method='spline')
    
    # Create test waypoints
    waypoints = np.array([
        [0, 0, 0],
        [0.2, 0.1, 0.1],
        [0.5, 0.5, 0.2],
        [0.8, 0.9, 0.3],
        [1, 1, 0.5]
    ])
    
    # Optimize
    optimized = optimizer.optimize_trajectory(waypoints, max_iterations=10)
    
    assert optimized is not None, "Should return optimized trajectory"
    print(f"Optimized {len(waypoints)} waypoints to {len(optimized)}")

def test_bc_trainer():
    """Test BC trainer with time cap"""
    from bc_trainer_enhanced import EnhancedBCTrainer, BCConfig
    
    # Create dummy trajectories
    class DemoTrajectory:
        def __init__(self):
            import numpy as np
            self.states = [np.random.randn(10) for _ in range(20)]
            self.actions = [np.random.randn(4) for _ in range(20)]
    
    trajectories = [DemoTrajectory() for _ in range(5)]
    
    # Configure with short time cap for testing
    config = BCConfig(
        max_time_seconds=5.0,  # 5 seconds for testing
        max_epochs=10,
        batch_size=32
    )
    
    trainer = EnhancedBCTrainer(config)
    
    # Train (should complete within time cap)
    result = trainer.train(trajectories, show_plot=False)
    
    assert result['within_time_cap'], "Should complete within time cap"
    print(f"Training completed in {result['total_time']:.2f}s")

def test_expert_collector():
    """Test expert collector with spinner"""
    from expert_collector import ExpertCollector
    
    collector = ExpertCollector(env_name="TestEnv")
    
    # Collect small number of trajectories for test
    trajectories = collector.collect_trajectories(
        num_trajectories=2,
        max_steps=10,
        render=False
    )
    
    assert len(trajectories) == 2, "Should collect 2 trajectories"
    print(f"Collected {len(trajectories)} trajectories")

def test_behavior_tree():
    """Test behavior tree with JSON output"""
    from behavior_tree_json import create_pick_and_place_tree
    
    tree = create_pick_and_place_tree()
    
    # Plan (generates JSON)
    goal = {
        "object_detected": True,
        "object_position": [0.3, 0.2, 0.05],
        "target_position": [0.5, 0.4, 0.05]
    }
    
    planning_result = tree.plan(goal, show_json=False)
    
    assert 'behavior_tree' in planning_result, "Should return tree structure"
    print(f"Tree has {planning_result['statistics']['total_nodes']} nodes")

def test_vision_system():
    """Test vision system with displacement calculation"""
    from vision_system import VisionSystem, CameraCalibration
    
    calibration = CameraCalibration(
        pixels_per_meter_x=2000,
        pixels_per_meter_y=2000
    )
    
    vision = VisionSystem(calibration)
    
    # Test displacement calculation
    current_pos = (320, 240)
    target_pos = (400, 300)
    
    displacement = vision.calculate_displacement(
        current_pos, target_pos, 
        show_visualization=False
    )
    
    assert 'dx_px' in displacement, "Should return displacement data"
    assert 'dx_m' in displacement, "Should include meter conversion"
    print(f"Displacement: {displacement['distance_px']:.1f}px / {displacement['distance_m']*1000:.1f}mm")

def test_skills_library():
    """Test skills library"""
    from skills_library import SkillRegistry
    
    registry = SkillRegistry()
    
    # Check available skills
    skills = registry.list_skills()
    assert len(skills) > 0, "Should have registered skills"
    print(f"Found {len(skills)} skills: {skills}")
    
    # Test one skill
    push_skill = registry.get_skill('push')
    assert push_skill is not None, "Should have push skill"
    print("Push skill validated")

def test_prompt_header():
    """Test prompt header display"""
    from prompt_header import PromptHeader
    
    # Just test that it doesn't crash
    PromptHeader.print_header(
        "Test Header",
        style='task',
        show_timestamp=False
    )
    print("Prompt header displayed")

def test_code_generator():
    """Test code generator"""
    from code_generator import CodeGenerator, CodeGenConfig
    
    config = CodeGenConfig(
        output_dir="test_generated",
        open_in_editor=False  # Don't open editor during test
    )
    
    generator = CodeGenerator(config)
    
    # Generate a simple file
    code = "def test():\n    return True"
    filepath = generator.generate_python_file(
        "test_module.py",
        code,
        "Test module",
        auto_open=False
    )
    
    assert filepath.exists(), "Should create file"
    print(f"Generated: {filepath}")
    
    # Clean up
    filepath.unlink()
    filepath.parent.rmdir()

def test_recovery_system():
    """Test recovery system"""
    from recovery_system import RecoverySystem
    
    recovery = RecoverySystem(checkpoint_dir="test_recovery")
    
    # Save a checkpoint
    recovery.save_checkpoint(
        script_path="test_script.py",
        parameters={"test": True},
        output="test output",
        execution_time=1.0
    )
    
    assert recovery.last_known_good is not None, "Should save checkpoint"
    print("Recovery checkpoint saved")
    
    # Clean up
    import shutil
    shutil.rmtree("test_recovery", ignore_errors=True)

def test_final_runner():
    """Test final runner demo mode toggle"""
    from final_runner import FinalRunner, DemoModeContext
    
    runner = FinalRunner()
    
    # Test toggle
    initial = runner.demo_mode
    runner.toggle_demo_mode()
    assert runner.demo_mode != initial, "Should toggle mode"
    print(f"Demo mode toggled: {initial} → {runner.demo_mode}")

def run_all_tests():
    """Run all component tests"""
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}COGNIFORGE END-TO-END TEST SUITE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    tests = [
        ("Motion Controller", test_motion_controller),
        ("Waypoint Optimizer", test_waypoint_optimizer),
        ("BC Trainer", test_bc_trainer),
        ("Expert Collector", test_expert_collector),
        ("Behavior Tree", test_behavior_tree),
        ("Vision System", test_vision_system),
        ("Skills Library", test_skills_library),
        ("Prompt Header", test_prompt_header),
        ("Code Generator", test_code_generator),
        ("Recovery System", test_recovery_system),
        ("Final Runner", test_final_runner)
    ]
    
    results = []
    for name, test_func in tests:
        passed = test_component(name, test_func)
        results.append((name, passed))
        time.sleep(0.5)  # Brief pause between tests
    
    # Print summary
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}TEST SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = f"{Fore.GREEN}✅ PASSED{Style.RESET_ALL}" if passed else f"{Fore.RED}❌ FAILED{Style.RESET_ALL}"
        print(f"{name:<25} {status}")
    
    print(f"\n{Fore.CYAN}Results: {passed_count}/{total_count} tests passed{Style.RESET_ALL}")
    
    if passed_count == total_count:
        print(f"\n{Fore.GREEN}🎉 ALL TESTS PASSED! System is ready.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}⚠️ Some tests failed. Please review the errors above.{Style.RESET_ALL}")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)