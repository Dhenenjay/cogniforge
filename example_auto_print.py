#!/usr/bin/env python3
"""
Simple Example: Automatic Behavior Tree Pretty Printing

This example shows how the behavior tree is automatically displayed
in a pretty box format when planning completes.
"""

from cogniforge.core.planner import plan_to_behavior_tree

# Example 1: Simple pick and place
print("\n" + "="*80)
print("Example 1: Simple Pick and Place Task")
print("="*80)

task1 = "Pick up the red cube and place it on the table"
result1 = plan_to_behavior_tree(task1)

# Example 2: Stacking task
print("\n" + "="*80)
print("Example 2: Stacking Task")
print("="*80)

task2 = "Stack the blue block on top of the green block"
result2 = plan_to_behavior_tree(task2)

# Example 3: Conditional task
print("\n" + "="*80)
print("Example 3: Conditional Task")
print("="*80)

task3 = "If the platform is empty, place all red cubes on it"
result3 = plan_to_behavior_tree(task3)

# Example 4: Complex sorting task
print("\n" + "="*80)
print("Example 4: Sorting Task")
print("="*80)

task4 = "Sort all objects by color into their respective zones"
result4 = plan_to_behavior_tree(task4)

print("\n" + "="*80)
print("✅ All behavior trees have been automatically displayed in pretty boxes!")
print("="*80)