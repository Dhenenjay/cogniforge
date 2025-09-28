"""
Behavior Tree System with JSON Visualization After Planning

This module implements a behavior tree for robotic task planning and execution,
with automatic JSON visualization after planning is complete.
"""

import json
import enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import time
from abc import ABC, abstractmethod
from colorama import init, Fore, Style
import numpy as np

# Initialize colorama
init(autoreset=True)

# Import our prompt header for nice display
try:
    from prompt_header import PromptHeader, show_prompt
except ImportError:
    # Fallback if prompt_header not available
    def show_prompt(msg, **kwargs):
        print(f"\n>>> {msg}")
    class PromptHeader:
        @staticmethod
        def print_tree(title, data):
            print(json.dumps(data, indent=2))
        @staticmethod
        def print_header(msg, **kwargs):
            print(f"\n=== {msg} ===")
        @staticmethod
        def print_section(title, items, **kwargs):
            print(f"\n{title}:")
            for item in items:
                print(f"  - {item}")


class NodeStatus(enum.Enum):
    """Status of behavior tree nodes"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class NodeType(enum.Enum):
    """Types of behavior tree nodes"""
    SEQUENCE = "sequence"
    SELECTOR = "selector"
    PARALLEL = "parallel"
    ACTION = "action"
    CONDITION = "condition"
    DECORATOR = "decorator"
    COMPOSITE = "composite"


@dataclass
class NodeResult:
    """Result from node execution"""
    status: NodeStatus
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0


@dataclass
class NodeConfig:
    """Configuration for a behavior tree node"""
    name: str
    type: NodeType
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    children: List['BehaviorNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BehaviorNode(ABC):
    """Abstract base class for behavior tree nodes"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.status = NodeStatus.IDLE
        self.last_result: Optional[NodeResult] = None
        self.execution_count = 0
        self.total_duration_ms = 0
        
    @abstractmethod
    def tick(self, context: Dict[str, Any]) -> NodeResult:
        """Execute the node logic"""
        pass
    
    def reset(self):
        """Reset node to initial state"""
        self.status = NodeStatus.IDLE
        self.last_result = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization"""
        data = {
            "name": self.config.name,
            "type": self.config.type.value,
            "description": self.config.description,
            "status": self.status.value,
            "execution_count": self.execution_count,
            "total_duration_ms": round(self.total_duration_ms, 2)
        }
        
        if self.config.params:
            data["params"] = self.config.params
            
        if self.config.metadata:
            data["metadata"] = self.config.metadata
            
        if self.config.children:
            data["children"] = [child.to_dict() for child in self.config.children]
            
        if self.last_result:
            data["last_result"] = {
                "status": self.last_result.status.value,
                "message": self.last_result.message,
                "data": self.last_result.data
            }
            
        return data


class SequenceNode(BehaviorNode):
    """Executes children in sequence until one fails"""
    
    def tick(self, context: Dict[str, Any]) -> NodeResult:
        start_time = time.time()
        
        for child in self.config.children:
            result = child.tick(context)
            if result.status != NodeStatus.SUCCESS:
                self.status = result.status
                self.last_result = result
                self.execution_count += 1
                self.total_duration_ms += (time.time() - start_time) * 1000
                return result
                
        self.status = NodeStatus.SUCCESS
        result = NodeResult(
            status=NodeStatus.SUCCESS,
            message=f"All {len(self.config.children)} children succeeded"
        )
        self.last_result = result
        self.execution_count += 1
        self.total_duration_ms += (time.time() - start_time) * 1000
        return result


class SelectorNode(BehaviorNode):
    """Executes children until one succeeds"""
    
    def tick(self, context: Dict[str, Any]) -> NodeResult:
        start_time = time.time()
        
        for child in self.config.children:
            result = child.tick(context)
            if result.status == NodeStatus.SUCCESS:
                self.status = NodeStatus.SUCCESS
                self.last_result = result
                self.execution_count += 1
                self.total_duration_ms += (time.time() - start_time) * 1000
                return result
                
        self.status = NodeStatus.FAILURE
        result = NodeResult(
            status=NodeStatus.FAILURE,
            message=f"All {len(self.config.children)} children failed"
        )
        self.last_result = result
        self.execution_count += 1
        self.total_duration_ms += (time.time() - start_time) * 1000
        return result


class ParallelNode(BehaviorNode):
    """Executes all children in parallel"""
    
    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self.success_threshold = config.params.get('success_threshold', len(config.children))
        
    def tick(self, context: Dict[str, Any]) -> NodeResult:
        start_time = time.time()
        results = []
        
        # In a real implementation, this would use threading/async
        # For now, we'll simulate parallel execution
        for child in self.config.children:
            results.append(child.tick(context))
            
        success_count = sum(1 for r in results if r.status == NodeStatus.SUCCESS)
        
        if success_count >= self.success_threshold:
            self.status = NodeStatus.SUCCESS
            result = NodeResult(
                status=NodeStatus.SUCCESS,
                message=f"{success_count}/{len(self.config.children)} children succeeded"
            )
        else:
            self.status = NodeStatus.FAILURE
            result = NodeResult(
                status=NodeStatus.FAILURE,
                message=f"Only {success_count}/{self.success_threshold} required successes"
            )
            
        self.last_result = result
        self.execution_count += 1
        self.total_duration_ms += (time.time() - start_time) * 1000
        return result


class ActionNode(BehaviorNode):
    """Executes a specific action"""
    
    def __init__(self, config: NodeConfig, action_fn: Optional[Callable] = None):
        super().__init__(config)
        self.action_fn = action_fn
        
    def tick(self, context: Dict[str, Any]) -> NodeResult:
        start_time = time.time()
        
        try:
            if self.action_fn:
                result = self.action_fn(context, self.config.params)
                if isinstance(result, NodeResult):
                    self.status = result.status
                    self.last_result = result
                else:
                    # Assume success if action returns truthy value
                    self.status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
                    self.last_result = NodeResult(
                        status=self.status,
                        message=str(result) if result else "Action failed"
                    )
            else:
                # Simulate action execution
                import random
                success = random.random() > 0.2  # 80% success rate
                self.status = NodeStatus.SUCCESS if success else NodeStatus.FAILURE
                self.last_result = NodeResult(
                    status=self.status,
                    message=f"Action '{self.config.name}' {'succeeded' if success else 'failed'}"
                )
                
        except Exception as e:
            self.status = NodeStatus.FAILURE
            self.last_result = NodeResult(
                status=NodeStatus.FAILURE,
                message=f"Action error: {str(e)}"
            )
            
        self.execution_count += 1
        self.total_duration_ms += (time.time() - start_time) * 1000
        return self.last_result


class ConditionNode(BehaviorNode):
    """Checks a condition"""
    
    def __init__(self, config: NodeConfig, condition_fn: Optional[Callable] = None):
        super().__init__(config)
        self.condition_fn = condition_fn
        
    def tick(self, context: Dict[str, Any]) -> NodeResult:
        start_time = time.time()
        
        try:
            if self.condition_fn:
                result = self.condition_fn(context, self.config.params)
                success = bool(result)
            else:
                # Check condition from context
                condition_key = self.config.params.get('key', self.config.name)
                expected_value = self.config.params.get('value', True)
                actual_value = context.get(condition_key)
                success = actual_value == expected_value
                
            self.status = NodeStatus.SUCCESS if success else NodeStatus.FAILURE
            self.last_result = NodeResult(
                status=self.status,
                message=f"Condition '{self.config.name}' {'met' if success else 'not met'}"
            )
            
        except Exception as e:
            self.status = NodeStatus.FAILURE
            self.last_result = NodeResult(
                status=NodeStatus.FAILURE,
                message=f"Condition error: {str(e)}"
            )
            
        self.execution_count += 1
        self.total_duration_ms += (time.time() - start_time) * 1000
        return self.last_result


class BehaviorTree:
    """Main behavior tree class with JSON visualization after planning"""
    
    def __init__(self, root: BehaviorNode, name: str = "BehaviorTree"):
        self.root = root
        self.name = name
        self.context: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.planning_time_ms = 0
        self.total_execution_time_ms = 0
        self.planned_json = None
        
    def plan(self, goal: Dict[str, Any], show_json: bool = True) -> Dict[str, Any]:
        """
        Plan the behavior tree execution and show JSON
        
        Args:
            goal: Goal specification
            show_json: Whether to display the JSON after planning
            
        Returns:
            Planning result with tree structure
        """
        start_time = time.time()
        
        # Show planning header
        PromptHeader.print_header(
            f"Planning: {self.name}",
            style='task',
            subtitle="Generating behavior tree structure",
            show_timestamp=True
        )
        
        # Update context with goal
        self.context.update(goal)
        
        # Simulate planning process (in real system, this would involve actual planning algorithms)
        print(f"\n{Fore.YELLOW}⚡ Analyzing goal requirements...{Style.RESET_ALL}")
        time.sleep(0.2)  # Simulate planning time
        
        print(f"{Fore.YELLOW}⚡ Building task decomposition...{Style.RESET_ALL}")
        time.sleep(0.2)
        
        print(f"{Fore.YELLOW}⚡ Optimizing execution order...{Style.RESET_ALL}")
        time.sleep(0.2)
        
        # Generate tree structure
        tree_dict = self.root.to_dict()
        
        # Calculate planning time
        self.planning_time_ms = (time.time() - start_time) * 1000
        
        # Create planning result
        planning_result = {
            "tree_name": self.name,
            "planning_time_ms": round(self.planning_time_ms, 2),
            "goal": goal,
            "context": self.context,
            "statistics": {
                "total_nodes": self._count_nodes(self.root),
                "max_depth": self._get_max_depth(self.root),
                "action_nodes": self._count_by_type(self.root, "action"),
                "condition_nodes": self._count_by_type(self.root, "condition")
            },
            "behavior_tree": tree_dict
        }
        
        self.planned_json = planning_result
        
        if show_json:
            # Display the JSON structure
            print(f"\n{Fore.GREEN}✅ Planning Complete!{Style.RESET_ALL}")
            print(f"Planning Time: {self.planning_time_ms:.2f}ms\n")
            
            # Show the JSON
            PromptHeader.print_header(
                "BEHAVIOR TREE JSON",
                style='info',
                subtitle="Generated tree structure",
                show_timestamp=False
            )
            
            json_str = json.dumps(planning_result, indent=2)
            
            # Color-code the JSON output
            colored_json = self._colorize_json(json_str)
            print(colored_json)
            
            # Also show visual tree
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN} VISUAL TREE REPRESENTATION{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
            self._print_tree_visual(self.root)
            
            # Show summary statistics
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN} PLANNING SUMMARY{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
            
            stats = planning_result["statistics"]
            print(f"📊 Total Nodes: {stats['total_nodes']}")
            print(f"📏 Max Depth: {stats['max_depth']}")
            print(f"⚡ Action Nodes: {stats['action_nodes']}")
            print(f"❓ Condition Nodes: {stats['condition_nodes']}")
            print(f"⏱️  Planning Time: {self.planning_time_ms:.2f}ms")
            
        return planning_result
        
    def execute(self, context: Optional[Dict[str, Any]] = None) -> NodeResult:
        """
        Execute the behavior tree
        
        Args:
            context: Execution context
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        if context:
            self.context.update(context)
            
        # Show execution header
        PromptHeader.print_header(
            f"Executing: {self.name}",
            style='task',
            subtitle="Running behavior tree",
            show_timestamp=True
        )
        
        # Execute tree
        result = self.root.tick(self.context)
        
        # Record execution
        execution_time_ms = (time.time() - start_time) * 1000
        self.total_execution_time_ms += execution_time_ms
        
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "result": {
                "status": result.status.value,
                "message": result.message,
                "data": result.data
            },
            "duration_ms": execution_time_ms,
            "tree_snapshot": self.root.to_dict()
        }
        self.execution_history.append(execution_record)
        
        # Show result
        status_color = self._get_status_color(result.status)
        print(f"\n{status_color}━━━ Execution Result ━━━{Style.RESET_ALL}")
        print(f"Status: {status_color}{result.status.value.upper()}{Style.RESET_ALL}")
        print(f"Message: {result.message}")
        print(f"Duration: {execution_time_ms:.2f}ms")
        
        # Show execution JSON if requested
        if self.context.get('show_execution_json', False):
            print(f"\n{Fore.YELLOW}Execution JSON:{Style.RESET_ALL}")
            print(json.dumps(execution_record, indent=2, default=str))
        
        return result
        
    def save_to_file(self, filepath: str):
        """Save tree to JSON file"""
        data = {
            "tree_name": self.name,
            "planning_result": self.planned_json,
            "execution_history": self.execution_history,
            "tree_structure": self.root.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"{Fore.GREEN}✓ Tree saved to {filepath}{Style.RESET_ALL}")
        
    def reset(self):
        """Reset tree to initial state"""
        self._reset_recursive(self.root)
        
    def _reset_recursive(self, node: BehaviorNode):
        """Recursively reset all nodes"""
        node.reset()
        for child in node.config.children:
            self._reset_recursive(child)
            
    def _count_nodes(self, node: BehaviorNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.config.children:
            count += self._count_nodes(child)
        return count
        
    def _get_max_depth(self, node: BehaviorNode, current_depth: int = 0) -> int:
        """Get maximum depth of tree"""
        if not node.config.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.config.children:
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
            
        return max_child_depth
        
    def _count_by_type(self, node: BehaviorNode, node_type: str) -> int:
        """Count nodes of specific type"""
        count = 1 if node.config.type.value == node_type else 0
        for child in node.config.children:
            count += self._count_by_type(child, node_type)
        return count
        
    def _print_tree_visual(self, node: BehaviorNode, prefix: str = "", is_last: bool = True):
        """Print visual representation of tree"""
        
        # Choose connector
        connector = "└── " if is_last else "├── "
        
        # Node symbol based on type
        symbols = {
            NodeType.SEQUENCE: "→",
            NodeType.SELECTOR: "?",
            NodeType.PARALLEL: "║",
            NodeType.ACTION: "▶",
            NodeType.CONDITION: "◊",
            NodeType.DECORATOR: "◈",
            NodeType.COMPOSITE: "◯"
        }
        symbol = symbols.get(node.config.type, "•")
        
        # Status color
        status_color = self._get_status_color(node.status)
        
        # Print node
        node_info = f"{symbol} {node.config.name}"
        if node.config.description:
            node_info += f" ({node.config.description})"
            
        print(f"{prefix}{connector}{status_color}{node_info}{Style.RESET_ALL}")
        
        # Print children
        if node.config.children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.config.children):
                is_last_child = i == len(node.config.children) - 1
                self._print_tree_visual(child, prefix + extension, is_last_child)
                
    def _get_status_color(self, status: NodeStatus) -> str:
        """Get color for status"""
        colors = {
            NodeStatus.IDLE: Fore.WHITE,
            NodeStatus.RUNNING: Fore.YELLOW,
            NodeStatus.SUCCESS: Fore.GREEN,
            NodeStatus.FAILURE: Fore.RED
        }
        return colors.get(status, Fore.WHITE)
        
    def _colorize_json(self, json_str: str) -> str:
        """Add colors to JSON string for better readability"""
        # This is a simple colorization - could be enhanced
        lines = json_str.split('\n')
        colored_lines = []
        
        for line in lines:
            if '"tree_name"' in line or '"name"' in line:
                colored_lines.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
            elif '"type"' in line:
                colored_lines.append(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
            elif '"status"' in line:
                if '"success"' in line:
                    colored_lines.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
                elif '"failure"' in line:
                    colored_lines.append(f"{Fore.RED}{line}{Style.RESET_ALL}")
                else:
                    colored_lines.append(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
            elif '"description"' in line:
                colored_lines.append(f"{Fore.MAGENTA}{line}{Style.RESET_ALL}")
            else:
                colored_lines.append(line)
                
        return '\n'.join(colored_lines)


# ============================================================================
# Robotic Manipulation Behavior Trees
# ============================================================================

def create_pick_and_place_tree() -> BehaviorTree:
    """Create a pick and place behavior tree for robotic manipulation"""
    
    root = SequenceNode(NodeConfig(
        name="PickAndPlaceSequence",
        type=NodeType.SEQUENCE,
        description="Complete pick and place task",
        children=[
            # Perception phase
            SequenceNode(NodeConfig(
                name="PerceptionPhase",
                type=NodeType.SEQUENCE,
                description="Detect and localize objects",
                children=[
                    ActionNode(NodeConfig(
                        name="ScanWorkspace",
                        type=NodeType.ACTION,
                        description="Scan for objects using vision system",
                        params={"sensor": "rgbd_camera", "resolution": "high"}
                    )),
                    ConditionNode(NodeConfig(
                        name="ObjectDetected",
                        type=NodeType.CONDITION,
                        description="Verify object detected",
                        params={"key": "object_detected", "value": True}
                    )),
                    ActionNode(NodeConfig(
                        name="ComputeGraspPose",
                        type=NodeType.ACTION,
                        description="Calculate optimal grasp pose",
                        params={"method": "geometric", "quality_threshold": 0.8}
                    ))
                ]
            )),
            
            # Approach phase
            SequenceNode(NodeConfig(
                name="ApproachPhase",
                type=NodeType.SEQUENCE,
                description="Move to grasp position",
                children=[
                    ActionNode(NodeConfig(
                        name="PlanApproachTrajectory",
                        type=NodeType.ACTION,
                        description="Plan collision-free path to object",
                        params={"planner": "RRT*", "planning_time": 5.0}
                    )),
                    ActionNode(NodeConfig(
                        name="MoveToPreGrasp",
                        type=NodeType.ACTION,
                        description="Move to pre-grasp position",
                        params={"offset_z": 0.1, "speed": 0.1}
                    )),
                    ActionNode(NodeConfig(
                        name="OpenGripper",
                        type=NodeType.ACTION,
                        description="Open gripper to grasp width",
                        params={"width": 0.08, "speed": 0.05}
                    ))
                ]
            )),
            
            # Grasp phase
            SequenceNode(NodeConfig(
                name="GraspPhase",
                type=NodeType.SEQUENCE,
                description="Execute grasp",
                children=[
                    ActionNode(NodeConfig(
                        name="MoveToGrasp",
                        type=NodeType.ACTION,
                        description="Move down to grasp position",
                        params={"speed": 0.05, "force_feedback": True}
                    )),
                    ActionNode(NodeConfig(
                        name="CloseGripper",
                        type=NodeType.ACTION,
                        description="Close gripper on object",
                        params={"force": 10.0, "detect_grasp": True}
                    )),
                    ConditionNode(NodeConfig(
                        name="GraspSuccessful",
                        type=NodeType.CONDITION,
                        description="Verify successful grasp",
                        params={"key": "grasp_success", "value": True}
                    ))
                ]
            )),
            
            # Transport phase
            SequenceNode(NodeConfig(
                name="TransportPhase",
                type=NodeType.SEQUENCE,
                description="Move object to target",
                children=[
                    ActionNode(NodeConfig(
                        name="LiftObject",
                        type=NodeType.ACTION,
                        description="Lift object to safe height",
                        params={"height": 0.15, "speed": 0.08}
                    )),
                    ActionNode(NodeConfig(
                        name="PlanPlaceTrajectory",
                        type=NodeType.ACTION,
                        description="Plan path to place location",
                        params={"planner": "RRT*", "avoid_obstacles": True}
                    )),
                    ActionNode(NodeConfig(
                        name="MoveToPlacePosition",
                        type=NodeType.ACTION,
                        description="Move to target position",
                        params={"speed": 0.1, "smooth_trajectory": True}
                    ))
                ]
            )),
            
            # Place phase
            SequenceNode(NodeConfig(
                name="PlacePhase",
                type=NodeType.SEQUENCE,
                description="Place object at target",
                children=[
                    ActionNode(NodeConfig(
                        name="LowerToPlace",
                        type=NodeType.ACTION,
                        description="Lower object to surface",
                        params={"speed": 0.05, "detect_contact": True}
                    )),
                    ActionNode(NodeConfig(
                        name="ReleaseObject",
                        type=NodeType.ACTION,
                        description="Open gripper to release",
                        params={"width": 0.08, "speed": 0.05}
                    )),
                    ActionNode(NodeConfig(
                        name="Retreat",
                        type=NodeType.ACTION,
                        description="Move to safe position",
                        params={"height": 0.1, "speed": 0.1}
                    ))
                ]
            ))
        ]
    ))
    
    return BehaviorTree(root, "PickAndPlaceTask")


def create_adaptive_manipulation_tree() -> BehaviorTree:
    """Create adaptive manipulation tree that selects strategy based on context"""
    
    root = SelectorNode(NodeConfig(
        name="AdaptiveManipulation",
        type=NodeType.SELECTOR,
        description="Try multiple manipulation strategies",
        children=[
            # Try learned policy first
            SequenceNode(NodeConfig(
                name="LearnedPolicyStrategy",
                type=NodeType.SEQUENCE,
                description="Use learned BC/PPO policy",
                children=[
                    ConditionNode(NodeConfig(
                        name="PolicyAvailable",
                        type=NodeType.CONDITION,
                        description="Check if trained policy exists",
                        params={"key": "policy_loaded", "value": True}
                    )),
                    ActionNode(NodeConfig(
                        name="ExecuteLearnedPolicy",
                        type=NodeType.ACTION,
                        description="Run neural network policy",
                        params={"model": "bc_ppo_hybrid", "confidence_threshold": 0.8}
                    ))
                ]
            )),
            
            # Try optimization-based approach
            SequenceNode(NodeConfig(
                name="OptimizationStrategy",
                type=NodeType.SEQUENCE,
                description="Use trajectory optimization",
                children=[
                    ActionNode(NodeConfig(
                        name="GenerateWaypoints",
                        type=NodeType.ACTION,
                        description="Generate initial waypoints",
                        params={"method": "geometric", "num_waypoints": 10}
                    )),
                    ActionNode(NodeConfig(
                        name="OptimizeTrajectory",
                        type=NodeType.ACTION,
                        description="Optimize using CMA-ES",
                        params={"optimizer": "cma-es", "iterations": 100}
                    )),
                    ActionNode(NodeConfig(
                        name="ExecuteOptimizedTrajectory",
                        type=NodeType.ACTION,
                        description="Execute optimized path",
                        params={"controller": "impedance", "tracking_error": 0.01}
                    ))
                ]
            )),
            
            # Fallback to primitive skills
            SequenceNode(NodeConfig(
                name="PrimitiveSkillsStrategy",
                type=NodeType.SEQUENCE,
                description="Use basic manipulation primitives",
                children=[
                    SelectorNode(NodeConfig(
                        name="SelectPrimitive",
                        type=NodeType.SELECTOR,
                        description="Choose appropriate primitive",
                        children=[
                            ActionNode(NodeConfig(
                                name="PushPrimitive",
                                type=NodeType.ACTION,
                                description="Push object to target",
                                params={"force": 5.0, "direction": "computed"}
                            )),
                            ActionNode(NodeConfig(
                                name="SlidePrimitive",
                                type=NodeType.ACTION,
                                description="Slide object along surface",
                                params={"speed": 0.05, "maintain_contact": True}
                            )),
                            ActionNode(NodeConfig(
                                name="PivotPrimitive",
                                type=NodeType.ACTION,
                                description="Pivot object around contact point",
                                params={"angle": 45, "pivot_point": "auto"}
                            ))
                        ]
                    ))
                ]
            ))
        ]
    ))
    
    return BehaviorTree(root, "AdaptiveManipulationTask")


# ============================================================================
# Demo Function
# ============================================================================

def demo_behavior_tree_with_json():
    """Demonstrate behavior tree planning with JSON output"""
    
    print(f"{Fore.CYAN}{'='*80}")
    print(f" BEHAVIOR TREE WITH JSON VISUALIZATION DEMO")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    # Create pick and place tree
    print(f"{Fore.YELLOW}Creating Pick and Place Behavior Tree...{Style.RESET_ALL}\n")
    pick_place_tree = create_pick_and_place_tree()
    
    # Plan with goal (this will automatically show JSON)
    goal = {
        "object_detected": True,
        "object_position": [0.3, 0.2, 0.05],
        "target_position": [0.5, 0.4, 0.05],
        "gripper_type": "parallel_jaw",
        "policy_loaded": False
    }
    
    planning_result = pick_place_tree.plan(goal, show_json=True)
    
    # Wait for user
    input(f"\n{Fore.YELLOW}Press Enter to execute the planned tree...{Style.RESET_ALL}")
    
    # Execute the tree
    execution_result = pick_place_tree.execute()
    
    # Save to file
    pick_place_tree.save_to_file("pick_place_tree_output.json")
    
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    # Create adaptive manipulation tree
    print(f"{Fore.YELLOW}Creating Adaptive Manipulation Tree...{Style.RESET_ALL}\n")
    adaptive_tree = create_adaptive_manipulation_tree()
    
    # Plan with different goal
    goal2 = {
        "policy_loaded": True,
        "task_type": "manipulation",
        "complexity": "high",
        "time_constraint": 10.0
    }
    
    planning_result2 = adaptive_tree.plan(goal2, show_json=True)
    
    # Execute
    input(f"\n{Fore.YELLOW}Press Enter to execute the adaptive tree...{Style.RESET_ALL}")
    execution_result2 = adaptive_tree.execute({"show_execution_json": True})
    
    print(f"\n{Fore.GREEN}✅ Demo Complete!{Style.RESET_ALL}")
    print(f"\nBehavior trees with JSON have been saved to:")
    print(f"  - pick_place_tree_output.json")


if __name__ == "__main__":
    demo_behavior_tree_with_json()