"""
Task planner that converts natural language prompts to behavior tree representations.

This module provides functionality to parse task descriptions and generate
structured behavior trees that can be executed by robotic systems.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import tree visualizer for pretty printing
try:
    from .tree_visualizer import print_behavior_tree, BehaviorTreePrinter
except ImportError:
    # Fallback if visualizer not available
    print_behavior_tree = None
    BehaviorTreePrinter = None

# Configure logging
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Behavior tree node types."""
    # Action nodes (leaf nodes)
    MOVE_TO = "MoveTo"
    PICK = "Pick"
    PLACE = "Place"
    GRASP = "Grasp"
    RELEASE = "Release"
    PUSH = "Push"
    PULL = "Pull"
    ROTATE = "Rotate"
    WAIT = "Wait"
    CHECK_CONDITION = "CheckCondition"
    SCAN = "Scan"
    APPROACH = "Approach"
    RETRACT = "Retract"
    ALIGN = "Align"
    
    # Composite nodes
    SEQUENCE = "Sequence"
    SELECTOR = "Selector"
    PARALLEL = "Parallel"
    REPEAT = "Repeat"
    RETRY = "Retry"
    
    # Decorator nodes
    INVERTER = "Inverter"
    SUCCEEDER = "Succeeder"
    FORCE_FAILURE = "ForceFailure"
    TIMEOUT = "Timeout"


@dataclass
class BehaviorNode:
    """Represents a node in the behavior tree."""
    type: str
    name: Optional[str] = None
    object: Optional[str] = None
    target: Optional[str] = None
    position: Optional[List[float]] = None
    parameters: Optional[Dict[str, Any]] = None
    children: Optional[List['BehaviorNode']] = None
    condition: Optional[str] = None
    timeout: Optional[float] = None
    retries: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        result = {"type": self.type}
        
        if self.name:
            result["name"] = self.name
        if self.object:
            result["object"] = self.object
        if self.target:
            result["target"] = self.target
        if self.position:
            result["position"] = self.position
        if self.parameters:
            result["parameters"] = self.parameters
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        if self.condition:
            result["condition"] = self.condition
        if self.timeout:
            result["timeout"] = self.timeout
        if self.retries:
            result["retries"] = self.retries
            
        return result


class TaskPlanner:
    """Planner for converting task descriptions to behavior trees."""
    
    def __init__(self):
        """Initialize the task planner."""
        self.object_patterns = {
            'colors': r'\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|brown)\b',
            'shapes': r'\b(cube|block|sphere|ball|cylinder|pyramid|box|object)\b',
            'sizes': r'\b(small|large|big|tiny|medium)\b',
            'materials': r'\b(wooden|metal|plastic|glass)\b',
        }
        
        self.action_keywords = {
            'pick': ['pick', 'grab', 'grasp', 'take', 'get', 'fetch', 'collect'],
            'place': ['place', 'put', 'set', 'position', 'drop', 'deposit'],
            'move': ['move', 'transport', 'carry', 'transfer', 'bring'],
            'stack': ['stack', 'pile', 'arrange vertically'],
            'push': ['push', 'slide', 'shove'],
            'pull': ['pull', 'drag', 'draw'],
            'rotate': ['rotate', 'turn', 'spin', 'orient'],
            'align': ['align', 'line up', 'arrange', 'organize'],
            'sort': ['sort', 'categorize', 'separate', 'group'],
        }
        
        self.spatial_relations = {
            'on': ['on', 'on top of', 'above'],
            'under': ['under', 'below', 'beneath'],
            'next_to': ['next to', 'beside', 'adjacent to', 'near'],
            'between': ['between', 'among'],
            'in': ['in', 'inside', 'within'],
            'left': ['left of', 'to the left'],
            'right': ['right of', 'to the right'],
            'front': ['in front of', 'before'],
            'behind': ['behind', 'in back of'],
        }
    
    def extract_objects(self, text: str) -> List[str]:
        """Extract object references from text."""
        text_lower = text.lower()
        objects = []
        
        # Look for color + shape combinations
        color_pattern = self.object_patterns['colors']
        shape_pattern = self.object_patterns['shapes']
        
        # Find color-shape combinations
        combined_pattern = rf'({color_pattern})\s+({shape_pattern})'
        matches = re.finditer(combined_pattern, text_lower)
        
        for match in matches:
            color = match.group(1)
            shape = match.group(2)
            obj_name = f"{color}_{shape}"
            objects.append(obj_name)
        
        # Also look for standalone shapes if no color specified
        if not objects:
            shape_matches = re.finditer(shape_pattern, text_lower)
            for match in shape_matches:
                objects.append(match.group())
        
        # Look for specific named objects
        named_objects = re.findall(r'\b(table|platform|container|box|tray|target)\b', text_lower)
        objects.extend(named_objects)
        
        # Look for "all" or "every" patterns
        if re.search(r'\b(all|every)\s+\w+', text_lower):
            objects.append("all_objects")
        
        return objects
    
    def identify_action(self, text: str) -> str:
        """Identify the main action from text."""
        text_lower = text.lower()
        
        for action, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return action
        
        # Default to move if no specific action found
        return 'move'
    
    def extract_spatial_relation(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract spatial relationship and target from text."""
        text_lower = text.lower()
        
        for relation, keywords in self.spatial_relations.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Try to find what comes after the spatial keyword
                    pattern = rf'{keyword}\s+(?:the\s+)?(\w+(?:\s+\w+)?)'
                    match = re.search(pattern, text_lower)
                    if match:
                        target = match.group(1).strip()
                        return relation, target
        
        return None, None
    
    def parse_sequence_task(self, text: str) -> List[BehaviorNode]:
        """Parse tasks that involve sequences of actions."""
        nodes = []
        
        # Split by conjunctions and punctuation
        parts = re.split(r'\s+(?:and|then|after that|next)\s+|[,;.]', text)
        
        for part in parts:
            if not part.strip():
                continue
            
            action = self.identify_action(part)
            objects = self.extract_objects(part)
            relation, target = self.extract_spatial_relation(part)
            
            if action == 'pick' and objects:
                nodes.append(BehaviorNode(
                    type=NodeType.PICK.value,
                    object=objects[0]
                ))
            elif action == 'place':
                place_node = BehaviorNode(
                    type=NodeType.PLACE.value,
                    object=objects[0] if objects else None
                )
                if target:
                    place_node.target = target
                elif len(objects) > 1:
                    place_node.target = objects[1]
                nodes.append(place_node)
            elif action == 'move' and objects:
                if len(objects) >= 2:
                    # Move object to target
                    nodes.append(BehaviorNode(
                        type=NodeType.PICK.value,
                        object=objects[0]
                    ))
                    nodes.append(BehaviorNode(
                        type=NodeType.MOVE_TO.value,
                        target=objects[1]
                    ))
                    nodes.append(BehaviorNode(
                        type=NodeType.PLACE.value,
                        target=objects[1]
                    ))
                else:
                    nodes.append(BehaviorNode(
                        type=NodeType.MOVE_TO.value,
                        object=objects[0]
                    ))
        
        return nodes
    
    def parse_stack_task(self, text: str) -> List[BehaviorNode]:
        """Parse stacking tasks."""
        objects = self.extract_objects(text)
        nodes = []
        
        if len(objects) >= 2:
            # Stack first object on second
            nodes.extend([
                BehaviorNode(type=NodeType.PICK.value, object=objects[0]),
                BehaviorNode(type=NodeType.ALIGN.value, target=objects[1]),
                BehaviorNode(type=NodeType.PLACE.value, target=objects[1])
            ])
        elif "all" in text.lower():
            # Stack all objects
            nodes.append(BehaviorNode(
                type=NodeType.SEQUENCE.value,
                name="StackAllObjects",
                children=[
                    BehaviorNode(type=NodeType.SCAN.value),
                    BehaviorNode(
                        type=NodeType.REPEAT.value,
                        parameters={"count": -1},  # -1 means until no more objects
                        children=[
                            BehaviorNode(type=NodeType.PICK.value, object="next_object"),
                            BehaviorNode(type=NodeType.PLACE.value, target="stack_position")
                        ]
                    )
                ]
            ))
        
        return nodes
    
    def parse_sort_task(self, text: str) -> List[BehaviorNode]:
        """Parse sorting/organizing tasks."""
        nodes = []
        
        if "color" in text.lower():
            nodes.append(BehaviorNode(
                type=NodeType.SEQUENCE.value,
                name="SortByColor",
                children=[
                    BehaviorNode(type=NodeType.SCAN.value),
                    BehaviorNode(
                        type=NodeType.PARALLEL.value,
                        children=[
                            BehaviorNode(
                                type=NodeType.SEQUENCE.value,
                                name="SortRed",
                                children=[
                                    BehaviorNode(type=NodeType.PICK.value, object="red_object"),
                                    BehaviorNode(type=NodeType.PLACE.value, target="red_zone")
                                ]
                            ),
                            BehaviorNode(
                                type=NodeType.SEQUENCE.value,
                                name="SortBlue",
                                children=[
                                    BehaviorNode(type=NodeType.PICK.value, object="blue_object"),
                                    BehaviorNode(type=NodeType.PLACE.value, target="blue_zone")
                                ]
                            )
                        ]
                    )
                ]
            ))
        elif "size" in text.lower():
            nodes.append(BehaviorNode(
                type=NodeType.SEQUENCE.value,
                name="SortBySize",
                children=[
                    BehaviorNode(type=NodeType.SCAN.value),
                    BehaviorNode(type=NodeType.CHECK_CONDITION.value, condition="objects_sorted_by_size")
                ]
            ))
        
        return nodes
    
    def parse_conditional_task(self, text: str) -> List[BehaviorNode]:
        """Parse tasks with conditions."""
        nodes = []
        
        if "if" in text.lower():
            # Extract condition and actions
            if_match = re.search(r'if\s+(.+?)(?:then|,)\s+(.+?)(?:else|otherwise|$)', text.lower())
            if if_match:
                condition = if_match.group(1).strip()
                then_action = if_match.group(2).strip()
                
                nodes.append(BehaviorNode(
                    type=NodeType.SELECTOR.value,
                    children=[
                        BehaviorNode(
                            type=NodeType.SEQUENCE.value,
                            children=[
                                BehaviorNode(type=NodeType.CHECK_CONDITION.value, condition=condition),
                                *self.parse_sequence_task(then_action)
                            ]
                        ),
                        BehaviorNode(type=NodeType.SUCCEEDER.value)  # Default success if condition fails
                    ]
                ))
        
        return nodes
    
    def plan_to_behavior_tree(self, prompt: str, print_tree: bool = True) -> List[Dict[str, Any]]:
        """
        Convert a natural language prompt to a behavior tree representation.
        
        Args:
            prompt: Natural language description of the task.
            
        Returns:
            List of behavior tree nodes in JSON format.
            
        Example:
            >>> planner = TaskPlanner()
            >>> nodes = planner.plan_to_behavior_tree("Pick up the red cube and place it on the blue cube")
            >>> print(json.dumps(nodes, indent=2))
            [
                {"type": "Pick", "object": "red_cube"},
                {"type": "Place", "target": "blue_cube"}
            ]
        """
        prompt_lower = prompt.lower()
        nodes = []
        
        # Check for specific task types
        if "stack" in prompt_lower:
            nodes = self.parse_stack_task(prompt)
        elif "sort" in prompt_lower or "organize" in prompt_lower:
            nodes = self.parse_sort_task(prompt)
        elif "if" in prompt_lower:
            nodes = self.parse_conditional_task(prompt)
        elif any(word in prompt_lower for word in ["repeat", "times", "until"]):
            # Handle repetitive tasks
            repeat_match = re.search(r'repeat\s+(\d+)\s+times?', prompt_lower)
            count = int(repeat_match.group(1)) if repeat_match else -1
            
            action_nodes = self.parse_sequence_task(prompt)
            nodes.append(BehaviorNode(
                type=NodeType.REPEAT.value,
                parameters={"count": count},
                children=action_nodes
            ))
        else:
            # Default to sequence parsing
            nodes = self.parse_sequence_task(prompt)
        
        # Wrap in sequence if multiple top-level nodes
        if len(nodes) > 1:
            nodes = [BehaviorNode(
                type=NodeType.SEQUENCE.value,
                children=nodes
            )]
        
        # Convert to dictionary format
        result = [node.to_dict() for node in nodes]
        
        # Print the tree in a pretty box if enabled
        if print_tree and print_behavior_tree is not None:
            print("\n" + "="*80)
            print(" 🎯 BEHAVIOR TREE PLANNING COMPLETED ")
            print("="*80 + "\n")
            print_behavior_tree(result, f"Task: {prompt[:50]}..." if len(prompt) > 50 else f"Task: {prompt}", 
                              format="box", show_stats=True)
            print("\n" + "="*80 + "\n")
        elif print_tree:
            # Fallback to simple JSON printing if visualizer not available
            print("\n" + "="*80)
            print(" BEHAVIOR TREE GENERATED ")
            print("="*80)
            print(json.dumps(result, indent=2))
            print("="*80 + "\n")
        
        return result


# Global planner instance
_planner = TaskPlanner()


def plan_to_behavior_tree(prompt: str, print_tree: bool = True) -> List[Dict[str, Any]]:
    """
    Convert a natural language prompt to a behavior tree representation.
    
    This function parses task descriptions and generates structured behavior
    trees that can be executed by robotic systems. It handles various task
    types including pick-and-place, stacking, sorting, and conditional actions.
    
    Args:
        prompt: Natural language description of the task.
        
    Returns:
        List of behavior tree nodes in JSON format. Each node contains:
        - type: The node type (e.g., "MoveTo", "Pick", "Place", "Sequence")
        - object: The target object (if applicable)
        - target: The destination or reference object (if applicable)
        - children: Child nodes for composite nodes
        - parameters: Additional parameters for the node
        
    Example:
        >>> nodes = plan_to_behavior_tree("Pick up the red cube and place it on the table")
        >>> print(json.dumps(nodes, indent=2))
        [
            {
                "type": "Sequence",
                "children": [
                    {"type": "Pick", "object": "red_cube"},
                    {"type": "Place", "target": "table"}
                ]
            }
        ]
        
        >>> nodes = plan_to_behavior_tree("Move the blue cube next to the green cube")
        >>> print(json.dumps(nodes, indent=2))
        [
            {
                "type": "Sequence",
                "children": [
                    {"type": "Pick", "object": "blue_cube"},
                    {"type": "MoveTo", "target": "green_cube"},
                    {"type": "Place", "target": "green_cube"}
                ]
            }
        ]
        
        >>> nodes = plan_to_behavior_tree("Stack all blocks")
        >>> print(json.dumps(nodes, indent=2))
        [
            {
                "type": "Sequence",
                "name": "StackAllObjects",
                "children": [
                    {"type": "Scan"},
                    {
                        "type": "Repeat",
                        "parameters": {"count": -1},
                        "children": [
                            {"type": "Pick", "object": "next_object"},
                            {"type": "Place", "target": "stack_position"}
                        ]
                    }
                ]
            }
        ]
    """
    return _planner.plan_to_behavior_tree(prompt, print_tree=print_tree)


def create_pick_place_sequence(obj: str, target: str) -> List[Dict[str, Any]]:
    """
    Create a simple pick and place sequence.
    
    Args:
        obj: Object to pick up.
        target: Target location or object.
        
    Returns:
        List of behavior nodes for pick and place.
        
    Example:
        >>> nodes = create_pick_place_sequence("red_cube", "blue_platform")
        >>> print(nodes)
        [{"type": "Pick", "object": "red_cube"}, {"type": "Place", "target": "blue_platform"}]
    """
    return [
        {"type": NodeType.PICK.value, "object": obj},
        {"type": NodeType.PLACE.value, "target": target}
    ]


def create_move_sequence(obj: str, destination: str) -> List[Dict[str, Any]]:
    """
    Create a movement sequence for an object.
    
    Args:
        obj: Object to move.
        destination: Destination location.
        
    Returns:
        List of behavior nodes for moving an object.
        
    Example:
        >>> nodes = create_move_sequence("robot", "home_position")
    """
    return [
        {"type": NodeType.MOVE_TO.value, "object": obj, "target": destination}
    ]


def create_conditional_sequence(condition: str, then_nodes: List[Dict], 
                               else_nodes: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Create a conditional behavior sequence.
    
    Args:
        condition: Condition to check.
        then_nodes: Nodes to execute if condition is true.
        else_nodes: Nodes to execute if condition is false (optional).
        
    Returns:
        Selector node with conditional branches.
    """
    children = [
        {
            "type": NodeType.SEQUENCE.value,
            "children": [
                {"type": NodeType.CHECK_CONDITION.value, "condition": condition},
                *then_nodes
            ]
        }
    ]
    
    if else_nodes:
        children.append({
            "type": NodeType.SEQUENCE.value,
            "children": else_nodes
        })
    
    return {
        "type": NodeType.SELECTOR.value,
        "children": children
    }


def create_repeat_sequence(nodes: List[Dict], count: int = -1) -> Dict[str, Any]:
    """
    Create a repeating sequence.
    
    Args:
        nodes: Nodes to repeat.
        count: Number of repetitions (-1 for infinite).
        
    Returns:
        Repeat node with children.
    """
    return {
        "type": NodeType.REPEAT.value,
        "parameters": {"count": count},
        "children": nodes
    }


def validate_behavior_tree(nodes: List[Dict[str, Any]]) -> bool:
    """
    Validate a behavior tree structure.
    
    Args:
        nodes: List of behavior tree nodes.
        
    Returns:
        True if valid, False otherwise.
    """
    def validate_node(node: Dict) -> bool:
        # Check required fields
        if "type" not in node:
            return False
        
        node_type = node["type"]
        
        # Check if composite nodes have children
        if node_type in [NodeType.SEQUENCE.value, NodeType.SELECTOR.value, 
                         NodeType.PARALLEL.value, NodeType.REPEAT.value]:
            if "children" not in node or not node["children"]:
                return False
            
            # Recursively validate children
            for child in node["children"]:
                if not validate_node(child):
                    return False
        
        # Check if action nodes have required fields
        if node_type == NodeType.PICK.value and "object" not in node:
            return False
        
        if node_type == NodeType.PLACE.value and "target" not in node and "position" not in node:
            return False
        
        return True
    
    if not nodes:
        return False
    
    for node in nodes:
        if not validate_node(node):
            return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test various prompts
    test_prompts = [
        "Pick up the red cube",
        "Place the blue block on the table",
        "Move the green sphere to the platform",
        "Pick up the red cube and place it on the blue cube",
        "Stack all blocks",
        "Sort the objects by color",
        "If the red cube is on the table, move it to the blue platform",
        "Repeat 3 times: pick a block and place it in the container",
        "First grab the yellow ball, then put it next to the green box",
        "Take all red objects and place them on the left side",
    ]
    
    planner = TaskPlanner()
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        # Plan and print the tree (automatically prints in box format)
        nodes = plan_to_behavior_tree(prompt, print_tree=True)
        
        # Validate the generated tree
        if validate_behavior_tree(nodes):
            print("✅ Valid behavior tree generated successfully!")
        else:
            print("❌ Warning: Generated tree failed validation")
