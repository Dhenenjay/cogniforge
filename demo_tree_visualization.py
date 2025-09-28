"""
Demonstration of Behavior Tree Pretty Printing

This script demonstrates the enhanced behavior tree visualization with
pretty box formatting that automatically displays when planning completes.
"""

from cogniforge.core.planner import plan_to_behavior_tree
from cogniforge.core.tree_visualizer import print_behavior_tree, BehaviorTreePrinter

def demo_basic_tasks():
    """Demonstrate basic task planning with pretty visualization."""
    
    print("\n" + "🤖 COGNIFORGE BEHAVIOR TREE PLANNER DEMO 🤖".center(80))
    print("=" * 80)
    
    # Test various task prompts
    test_prompts = [
        "Pick up the red cube and place it on the blue platform",
        "Stack the green block on top of the yellow block",
        "Move all red objects to the left side of the table",
        "Sort objects by color",
        "If the box is empty, fill it with blue cubes"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'='*80}")
        print(f"EXAMPLE {i}: Planning task...")
        print(f"{'='*80}")
        print(f"📝 Input: \"{prompt}\"")
        
        # Generate and display behavior tree (automatically prints in box)
        nodes = plan_to_behavior_tree(prompt, print_tree=True)
        
        input("\nPress Enter to continue to next example...")


def demo_complex_tree():
    """Demonstrate a complex nested behavior tree."""
    
    print("\n\n" + "="*80)
    print("COMPLEX BEHAVIOR TREE EXAMPLE")
    print("="*80)
    
    # Create a complex behavior tree manually
    complex_tree = [
        {
            "type": "Sequence",
            "name": "CompleteManipulationTask",
            "children": [
                {
                    "type": "Parallel",
                    "name": "InitialSetup",
                    "children": [
                        {
                            "type": "Scan",
                            "parameters": {"radius": 2.0, "mode": "full"}
                        },
                        {
                            "type": "CheckCondition",
                            "condition": "workspace_clear"
                        }
                    ]
                },
                {
                    "type": "Selector",
                    "name": "ObjectHandling",
                    "children": [
                        {
                            "type": "Sequence",
                            "name": "HandleRedObjects",
                            "children": [
                                {
                                    "type": "CheckCondition",
                                    "condition": "red_objects_present"
                                },
                                {
                                    "type": "Repeat",
                                    "parameters": {"count": -1},
                                    "children": [
                                        {
                                            "type": "Pick",
                                            "object": "red_object"
                                        },
                                        {
                                            "type": "MoveTo",
                                            "target": "red_zone",
                                            "position": [0.5, 0.2, 0.1]
                                        },
                                        {
                                            "type": "Place",
                                            "target": "red_zone"
                                        }
                                    ]
                                }
                            ]
                        },
                        {
                            "type": "Sequence",
                            "name": "HandleBlueObjects",
                            "children": [
                                {
                                    "type": "CheckCondition",
                                    "condition": "blue_objects_present"
                                },
                                {
                                    "type": "Repeat",
                                    "parameters": {"count": 3},
                                    "children": [
                                        {
                                            "type": "Pick",
                                            "object": "blue_cube"
                                        },
                                        {
                                            "type": "Align",
                                            "target": "stack_position"
                                        },
                                        {
                                            "type": "Place",
                                            "target": "stack_position",
                                            "parameters": {"precision": "high"}
                                        }
                                    ]
                                }
                            ]
                        },
                        {
                            "type": "Wait",
                            "timeout": 5.0
                        }
                    ]
                },
                {
                    "type": "Sequence",
                    "name": "FinalValidation",
                    "children": [
                        {
                            "type": "Scan"
                        },
                        {
                            "type": "CheckCondition",
                            "condition": "task_complete"
                        }
                    ]
                }
            ]
        }
    ]
    
    # Print the complex tree
    print_behavior_tree(complex_tree, "Complex Manipulation Task", format="box", show_stats=True)


def demo_different_formats():
    """Show the same tree in different display formats."""
    
    print("\n\n" + "="*80)
    print("DIFFERENT VISUALIZATION FORMATS")
    print("="*80)
    
    # Simple pick and place task
    simple_tree = [
        {
            "type": "Sequence",
            "children": [
                {"type": "Pick", "object": "red_cube"},
                {"type": "MoveTo", "target": "blue_platform"},
                {"type": "Place", "target": "blue_platform"}
            ]
        }
    ]
    
    print("\n1. BOX FORMAT (with colors):")
    print("-" * 40)
    print_behavior_tree(simple_tree, "Pick and Place", format="box", use_colors=True)
    
    print("\n\n2. TREE FORMAT (with colors):")
    print("-" * 40)
    print_behavior_tree(simple_tree, "Pick and Place", format="tree", use_colors=True)
    
    print("\n\n3. BOX FORMAT (without colors):")
    print("-" * 40)
    print_behavior_tree(simple_tree, "Pick and Place", format="box", use_colors=False)
    
    print("\n\n4. COMPACT JSON FORMAT:")
    print("-" * 40)
    print_behavior_tree(simple_tree, None, format="compact")


def demo_tree_statistics():
    """Demonstrate tree statistics calculation."""
    
    print("\n\n" + "="*80)
    print("BEHAVIOR TREE STATISTICS")
    print("="*80)
    
    # Create a tree with various node types
    stats_tree = [
        {
            "type": "Sequence",
            "children": [
                {"type": "Scan"},
                {
                    "type": "Parallel",
                    "children": [
                        {"type": "Pick", "object": "object1"},
                        {"type": "Pick", "object": "object2"},
                        {
                            "type": "Sequence",
                            "children": [
                                {"type": "CheckCondition", "condition": "path_clear"},
                                {"type": "MoveTo", "target": "destination"}
                            ]
                        }
                    ]
                },
                {
                    "type": "Selector",
                    "children": [
                        {"type": "Place", "target": "location1"},
                        {"type": "Place", "target": "location2"},
                        {"type": "Wait", "timeout": 1.0}
                    ]
                }
            ]
        }
    ]
    
    # Print with statistics
    print_behavior_tree(stats_tree, "Statistical Analysis Example", format="box", show_stats=True)


def main():
    """Run all demonstrations."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    🌳 BEHAVIOR TREE VISUALIZATION DEMO 🌳                  ║
║                                                                            ║
║    This demo shows the automatic pretty printing of behavior trees        ║
║    that occurs when planning completes in the CogniForge framework.       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\nSelect a demo to run:")
        print("1. Basic Task Planning (automatic visualization)")
        print("2. Complex Nested Tree")
        print("3. Different Display Formats")
        print("4. Tree Statistics")
        print("5. Run All Demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "1":
            demo_basic_tasks()
        elif choice == "2":
            demo_complex_tree()
        elif choice == "3":
            demo_different_formats()
        elif choice == "4":
            demo_tree_statistics()
        elif choice == "5":
            demo_basic_tasks()
            demo_complex_tree()
            demo_different_formats()
            demo_tree_statistics()
        elif choice == "0":
            print("\n👋 Thank you for using CogniForge Behavior Tree Visualizer!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()