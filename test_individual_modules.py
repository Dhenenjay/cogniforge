"""
Individual Module Test Suite for CogniForge
Tests each module separately to identify issues
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_module(name, import_test):
    """Test if a module can be imported"""
    print(f"\nTesting {name}...", end=" ")
    try:
        import_test()
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def main():
    """Run all module tests"""
    results = {}
    
    # Test 1: Motion Controller
    results['motion_controller'] = test_module(
        'Motion Controller',
        lambda: __import__('motion_controller')
    )
    
    # Test 2: Waypoint Optimizer  
    results['waypoint_optimizer'] = test_module(
        'Waypoint Optimizer',
        lambda: __import__('waypoint_optimizer')
    )
    
    # Test 3: BC Trainer Enhanced
    results['bc_trainer'] = test_module(
        'BC Trainer Enhanced',
        lambda: __import__('bc_trainer_enhanced')
    )
    
    # Test 4: Expert Collector
    results['expert_collector'] = test_module(
        'Expert Collector',
        lambda: __import__('expert_collector')
    )
    
    # Test 5: Behavior Tree JSON
    results['behavior_tree'] = test_module(
        'Behavior Tree JSON',
        lambda: __import__('behavior_tree_json')
    )
    
    # Test 6: Vision System
    results['vision_system'] = test_module(
        'Vision System',
        lambda: __import__('vision_system')
    )
    
    # Test 7: Skills Library (from parent dir)
    sys.path.insert(0, str(Path(__file__).parent))
    results['skills_library'] = test_module(
        'Skills Library',
        lambda: __import__('skills_library')
    )
    
    # Test 8: Prompt Header
    results['prompt_header'] = test_module(
        'Prompt Header',
        lambda: __import__('prompt_header')
    )
    
    # Test 9: Code Generator
    results['code_generator'] = test_module(
        'Code Generator',
        lambda: __import__('code_generator')
    )
    
    # Test 10: Recovery System
    results['recovery_system'] = test_module(
        'Recovery System',
        lambda: __import__('recovery_system')
    )
    
    # Test 11: Final Runner
    results['final_runner'] = test_module(
        'Final Runner',
        lambda: __import__('final_runner')
    )
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for module, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{module:20} {status_str}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All modules imported successfully!")
        return 0
    else:
        print(f"\n⚠ {total - passed} modules failed to import")
        return 1

if __name__ == "__main__":
    exit(main())