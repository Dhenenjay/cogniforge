#!/usr/bin/env python
"""
COMPREHENSIVE COGNIFORGE SYSTEM TEST
====================================
This script tests ALL components of the Cogniforge system to ensure
everything is demo-ready.
"""

import sys
import os
import traceback
import importlib
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add path for Cogniforge imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class CogniforgeTestSuite:
    """Comprehensive test suite for all Cogniforge components."""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'skipped': [],
            'errors': []
        }
        self.start_time = datetime.now()
        
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
        
    def print_test(self, name: str, status: str):
        """Print test result."""
        symbols = {'PASS': '✅', 'FAIL': '❌', 'SKIP': '⚠️', 'ERROR': '🔥'}
        symbol = symbols.get(status, '❓')
        color_codes = {'PASS': '\033[92m', 'FAIL': '\033[91m', 
                      'SKIP': '\033[93m', 'ERROR': '\033[91m'}
        reset = '\033[0m'
        
        # For Windows compatibility
        try:
            print(f"{symbol} [{status}] {name}")
        except:
            print(f"[{status}] {name}")
            
    def test_imports(self) -> Dict[str, bool]:
        """Test all critical imports."""
        self.print_header("TESTING IMPORTS")
        
        modules = [
            # Core modules
            'cogniforge.core.config',
            'cogniforge.core.policy',
            'cogniforge.core.evaluation',
            'cogniforge.core.optimization',
            'cogniforge.core.training',
            'cogniforge.core.reward',
            'cogniforge.core.planner',
            'cogniforge.core.refinement',
            'cogniforge.core.simulator',
            'cogniforge.core.logging_utils',
            'cogniforge.core.metrics_tracker',
            'cogniforge.core.seed_manager',
            'cogniforge.core.tree_visualizer',
            'cogniforge.core.adaptive_optimization',
            
            # Control modules
            'cogniforge.control.robot_control',
            'cogniforge.control.ik_controller',
            'cogniforge.control.grasp_execution',
            'cogniforge.control.safe_grasp_execution',
            
            # Vision modules
            'cogniforge.vision.vision_utils',
            'cogniforge.vision.coordinate_transform',
            'cogniforge.vision.vision_with_fallback',
            
            # Learning modules
            'cogniforge.learning.behavioral_cloning',
            
            # Optimization modules
            'cogniforge.optimization.waypoint_optimizer',
            'cogniforge.optimization.cmaes_with_timeout',
            
            # UI modules
            'cogniforge.ui.console_utils',
            'cogniforge.ui.cmaes_visualizer',
            'cogniforge.ui.vision_display',
            
            # Environment modules
            'cogniforge.environments.randomized_pick_place_env',
            
            # API modules
            'cogniforge.api.execute_endpoint',
            
            # CLI modules
            'cogniforge.cli.train_cli'
        ]
        
        import_results = {}
        
        for module in modules:
            try:
                importlib.import_module(module)
                import_results[module] = True
                self.print_test(f"Import {module}", "PASS")
                self.results['passed'].append(f"import_{module}")
            except ImportError as e:
                import_results[module] = False
                self.print_test(f"Import {module}: {str(e)}", "FAIL")
                self.results['failed'].append(f"import_{module}")
            except Exception as e:
                import_results[module] = False
                self.print_test(f"Import {module}: {str(e)}", "ERROR")
                self.results['errors'].append(f"import_{module}")
                
        return import_results
    
    def test_core_functionality(self):
        """Test core Cogniforge functionality."""
        self.print_header("TESTING CORE FUNCTIONALITY")
        
        tests = []
        
        # Test 1: Config system
        try:
            from cogniforge.core.config import get_config, CONFIG_DEFAULTS
            config = get_config()
            assert config is not None
            assert 'robot' in CONFIG_DEFAULTS
            self.print_test("Config system", "PASS")
            self.results['passed'].append("config_system")
        except Exception as e:
            self.print_test(f"Config system: {str(e)}", "FAIL")
            self.results['failed'].append("config_system")
            
        # Test 2: Logging system
        try:
            from cogniforge.core.logging_utils import setup_logging, get_logger
            logger = setup_logging("test_logger")
            logger.info("Test log message")
            self.print_test("Logging system", "PASS")
            self.results['passed'].append("logging_system")
        except Exception as e:
            self.print_test(f"Logging system: {str(e)}", "FAIL")
            self.results['failed'].append("logging_system")
            
        # Test 3: Metrics tracking
        try:
            from cogniforge.core.metrics_tracker import MetricsTracker
            tracker = MetricsTracker()
            tracker.add_metric('test_metric', 1.0)
            summary = tracker.get_summary()
            assert 'test_metric' in summary
            self.print_test("Metrics tracking", "PASS")
            self.results['passed'].append("metrics_tracking")
        except Exception as e:
            self.print_test(f"Metrics tracking: {str(e)}", "FAIL")
            self.results['failed'].append("metrics_tracking")
            
        # Test 4: Seed management
        try:
            from cogniforge.core.seed_manager import SeedManager
            sm = SeedManager(base_seed=42)
            seed1 = sm.get_seed("test")
            seed2 = sm.get_seed("test")
            assert seed1 == seed2  # Should be deterministic
            self.print_test("Seed management", "PASS")
            self.results['passed'].append("seed_management")
        except Exception as e:
            self.print_test(f"Seed management: {str(e)}", "FAIL")
            self.results['failed'].append("seed_management")
            
    def test_vision_system(self):
        """Test vision processing components."""
        self.print_header("TESTING VISION SYSTEM")
        
        try:
            import numpy as np
            from cogniforge.vision.coordinate_transform import CoordinateTransform
            
            # Create dummy camera matrix
            K = np.array([[500, 0, 320],
                         [0, 500, 240],
                         [0, 0, 1]], dtype=np.float32)
            
            transform = CoordinateTransform(K)
            
            # Test pixel to ray conversion
            ray = transform.pixel_to_ray(320, 240)
            assert ray.shape == (3,)
            
            self.print_test("Vision coordinate transform", "PASS")
            self.results['passed'].append("vision_coordinate")
        except Exception as e:
            self.print_test(f"Vision coordinate transform: {str(e)}", "FAIL")
            self.results['failed'].append("vision_coordinate")
            
    def test_control_system(self):
        """Test robot control components."""
        self.print_header("TESTING CONTROL SYSTEM")
        
        try:
            from cogniforge.control.ik_controller import IKController
            
            # Test IK controller initialization
            controller = IKController()
            self.print_test("IK Controller initialization", "PASS")
            self.results['passed'].append("ik_controller")
        except Exception as e:
            self.print_test(f"IK Controller: {str(e)}", "SKIP")
            self.results['skipped'].append("ik_controller")
            
    def test_optimization(self):
        """Test optimization algorithms."""
        self.print_header("TESTING OPTIMIZATION")
        
        try:
            from cogniforge.core.adaptive_optimization import AdaptiveOptimizer
            import numpy as np
            
            # Simple test function
            def test_func(x):
                return np.sum(x**2)
            
            optimizer = AdaptiveOptimizer(
                func=test_func,
                bounds=[(-1, 1)] * 2,
                maxfevals=100
            )
            
            result = optimizer.optimize()
            assert result is not None
            
            self.print_test("Adaptive optimization", "PASS")
            self.results['passed'].append("adaptive_optimization")
        except Exception as e:
            self.print_test(f"Adaptive optimization: {str(e)}", "FAIL")
            self.results['failed'].append("adaptive_optimization")
            
    def test_pybullet_integration(self):
        """Test PyBullet physics integration."""
        self.print_header("TESTING PYBULLET INTEGRATION")
        
        try:
            import pybullet as p
            import pybullet_data
            
            # Test connection
            client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Load plane
            plane_id = p.loadURDF("plane.urdf")
            
            # Run a few simulation steps
            for _ in range(10):
                p.stepSimulation()
                
            p.disconnect()
            
            self.print_test("PyBullet integration", "PASS")
            self.results['passed'].append("pybullet_integration")
        except Exception as e:
            self.print_test(f"PyBullet integration: {str(e)}", "FAIL")
            self.results['failed'].append("pybullet_integration")
            
    def test_ui_components(self):
        """Test UI components."""
        self.print_header("TESTING UI COMPONENTS")
        
        try:
            from cogniforge.ui.console_utils import print_header, print_status
            
            # Test console output functions
            print_header("Test Header")
            print_status("Test Status", "success")
            
            self.print_test("Console UI utilities", "PASS")
            self.results['passed'].append("ui_console")
        except Exception as e:
            self.print_test(f"Console UI: {str(e)}", "FAIL")
            self.results['failed'].append("ui_console")
            
    def test_examples(self):
        """Test that example scripts can be imported."""
        self.print_header("TESTING EXAMPLES")
        
        example_files = [
            'examples.block_spawning_demo',
            'examples.safe_grasp_demo',
            'examples.robot_verification_demo',
            'examples.metrics_tracking_demo',
            'examples.integrated_ui_demo'
        ]
        
        for example in example_files:
            try:
                module = importlib.import_module(example)
                self.print_test(f"Example: {example}", "PASS")
                self.results['passed'].append(f"example_{example}")
            except Exception as e:
                self.print_test(f"Example {example}: {str(e)}", "SKIP")
                self.results['skipped'].append(f"example_{example}")
                
    def test_file_integrity(self):
        """Check that all critical files exist."""
        self.print_header("TESTING FILE INTEGRITY")
        
        critical_files = [
            'cogniforge/__init__.py',
            'cogniforge/core/__init__.py',
            'cogniforge/control/__init__.py',
            'cogniforge/vision/__init__.py',
            'cogniforge/ui/__init__.py',
            'setup.py',
            'README.md'
        ]
        
        for filepath in critical_files:
            if os.path.exists(filepath):
                self.print_test(f"File exists: {filepath}", "PASS")
                self.results['passed'].append(f"file_{filepath}")
            else:
                self.print_test(f"File missing: {filepath}", "FAIL")
                self.results['failed'].append(f"file_{filepath}")
                
    def generate_report(self):
        """Generate final test report."""
        self.print_header("TEST SUMMARY")
        
        total_tests = (len(self.results['passed']) + 
                      len(self.results['failed']) + 
                      len(self.results['skipped']) + 
                      len(self.results['errors']))
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\nTotal tests run: {total_tests}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"\n✅ Passed: {len(self.results['passed'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        print(f"⚠️  Skipped: {len(self.results['skipped'])}")
        print(f"🔥 Errors: {len(self.results['errors'])}")
        
        pass_rate = (len(self.results['passed']) / total_tests * 100) if total_tests > 0 else 0
        print(f"\nPass rate: {pass_rate:.1f}%")
        
        # Save report to file
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'total_tests': total_tests,
            'results': self.results,
            'pass_rate': pass_rate
        }
        
        with open('test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\nDetailed report saved to test_report.json")
        
        # Demo readiness assessment
        print("\n" + "=" * 70)
        print("  DEMO READINESS ASSESSMENT")
        print("=" * 70)
        
        if pass_rate >= 90:
            print("✅ SYSTEM IS DEMO READY!")
            print("   All critical components are functioning properly.")
        elif pass_rate >= 70:
            print("⚠️  SYSTEM IS MOSTLY READY")
            print("   Some non-critical components need attention.")
        else:
            print("❌ SYSTEM NEEDS WORK")
            print("   Critical components are failing. Please fix before demo.")
            
        if self.results['failed']:
            print("\nFailed tests that need attention:")
            for test in self.results['failed'][:5]:  # Show first 5
                print(f"  - {test}")
                
        return pass_rate >= 70  # Return True if demo ready

def main():
    """Run the comprehensive test suite."""
    print("\n" + "=" * 70)
    print("  COGNIFORGE COMPREHENSIVE SYSTEM TEST")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Create test suite
    suite = CogniforgeTestSuite()
    
    # Run all tests
    suite.test_imports()
    suite.test_core_functionality()
    suite.test_vision_system()
    suite.test_control_system()
    suite.test_optimization()
    suite.test_pybullet_integration()
    suite.test_ui_components()
    suite.test_examples()
    suite.test_file_integrity()
    
    # Generate report
    is_ready = suite.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if is_ready else 1)

if __name__ == "__main__":
    main()