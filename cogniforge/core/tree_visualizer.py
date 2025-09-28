"""
Behavior Tree Visualization Module

Provides pretty printing and visualization capabilities for behavior trees,
displaying them in formatted boxes with proper hierarchy and styling.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import colorama
from colorama import Fore, Back, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)


class NodeSymbols:
    """Unicode symbols for tree visualization."""
    VERTICAL = "│"
    HORIZONTAL = "─"
    BRANCH = "├"
    LAST_BRANCH = "└"
    JUNCTION = "┼"
    CORNER_TOP_LEFT = "┌"
    CORNER_TOP_RIGHT = "┐"
    CORNER_BOTTOM_LEFT = "└"
    CORNER_BOTTOM_RIGHT = "┘"
    T_DOWN = "┬"
    T_UP = "┴"
    T_RIGHT = "├"
    T_LEFT = "┤"
    
    # Node type icons
    SEQUENCE = "→"
    SELECTOR = "?"
    PARALLEL = "⇉"
    REPEAT = "↻"
    ACTION = "◆"
    CONDITION = "◇"
    DECORATOR = "○"
    SUCCESS = "✓"
    FAILURE = "✗"
    RUNNING = "⟳"


class NodeColors:
    """Color scheme for different node types."""
    # Composite nodes
    SEQUENCE = Fore.CYAN
    SELECTOR = Fore.YELLOW
    PARALLEL = Fore.MAGENTA
    REPEAT = Fore.BLUE
    
    # Action nodes
    PICK = Fore.GREEN
    PLACE = Fore.GREEN
    MOVE = Fore.GREEN
    GRASP = Fore.GREEN
    
    # Condition nodes
    CONDITION = Fore.YELLOW
    
    # Decorator nodes
    DECORATOR = Fore.WHITE
    
    # Status colors
    SUCCESS = Fore.GREEN
    FAILURE = Fore.RED
    RUNNING = Fore.YELLOW
    DEFAULT = Fore.WHITE
    
    @classmethod
    def get_color(cls, node_type: str) -> str:
        """Get color for a node type."""
        node_type_upper = node_type.upper()
        
        # Check specific types
        if "SEQUENCE" in node_type_upper:
            return cls.SEQUENCE
        elif "SELECTOR" in node_type_upper or "SELECT" in node_type_upper:
            return cls.SELECTOR
        elif "PARALLEL" in node_type_upper:
            return cls.PARALLEL
        elif "REPEAT" in node_type_upper or "RETRY" in node_type_upper:
            return cls.REPEAT
        elif any(action in node_type_upper for action in ["PICK", "PLACE", "MOVE", "GRASP", "RELEASE", "PUSH", "PULL", "ROTATE", "SCAN", "APPROACH", "RETRACT", "ALIGN"]):
            return cls.PICK
        elif "CHECK" in node_type_upper or "CONDITION" in node_type_upper:
            return cls.CONDITION
        elif any(dec in node_type_upper for dec in ["INVERTER", "SUCCEEDER", "FORCE", "TIMEOUT"]):
            return cls.DECORATOR
        else:
            return cls.DEFAULT


class BehaviorTreePrinter:
    """Pretty printer for behavior trees."""
    
    def __init__(self, use_colors: bool = True, use_unicode: bool = True):
        """
        Initialize the tree printer.
        
        Args:
            use_colors: Whether to use colored output
            use_unicode: Whether to use Unicode symbols
        """
        self.use_colors = use_colors
        self.use_unicode = use_unicode
        self.symbols = NodeSymbols() if use_unicode else self._get_ascii_symbols()
    
    def _get_ascii_symbols(self) -> NodeSymbols:
        """Get ASCII alternatives for symbols."""
        symbols = NodeSymbols()
        symbols.VERTICAL = "|"
        symbols.HORIZONTAL = "-"
        symbols.BRANCH = "+"
        symbols.LAST_BRANCH = "+"
        symbols.JUNCTION = "+"
        symbols.CORNER_TOP_LEFT = "+"
        symbols.CORNER_TOP_RIGHT = "+"
        symbols.CORNER_BOTTOM_LEFT = "+"
        symbols.CORNER_BOTTOM_RIGHT = "+"
        symbols.T_DOWN = "+"
        symbols.T_UP = "+"
        symbols.T_RIGHT = "+"
        symbols.T_LEFT = "+"
        symbols.SEQUENCE = "->"
        symbols.SELECTOR = "?"
        symbols.PARALLEL = "||"
        symbols.REPEAT = "@"
        symbols.ACTION = "*"
        symbols.CONDITION = "?"
        symbols.DECORATOR = "o"
        return symbols
    
    def _get_node_icon(self, node_type: str) -> str:
        """Get icon for a node type."""
        node_type_upper = node_type.upper()
        
        if "SEQUENCE" in node_type_upper:
            return self.symbols.SEQUENCE
        elif "SELECTOR" in node_type_upper or "SELECT" in node_type_upper:
            return self.symbols.SELECTOR
        elif "PARALLEL" in node_type_upper:
            return self.symbols.PARALLEL
        elif "REPEAT" in node_type_upper or "RETRY" in node_type_upper:
            return self.symbols.REPEAT
        elif "CHECK" in node_type_upper or "CONDITION" in node_type_upper:
            return self.symbols.CONDITION
        elif any(dec in node_type_upper for dec in ["INVERTER", "SUCCEEDER", "FORCE", "TIMEOUT"]):
            return self.symbols.DECORATOR
        else:
            return self.symbols.ACTION
    
    def _format_node_label(self, node: Dict[str, Any]) -> str:
        """Format a node label with its properties."""
        node_type = node.get("type", "Unknown")
        icon = self._get_node_icon(node_type)
        
        # Build label parts
        parts = [f"{icon} {node_type}"]
        
        # Add name if present
        if "name" in node:
            parts[0] = f"{icon} {node['name']}"
        
        # Add object/target info
        if "object" in node:
            parts.append(f"[{node['object']}]")
        
        if "target" in node:
            parts.append(f"→ {node['target']}")
        
        # Add condition
        if "condition" in node:
            parts.append(f"if: {node['condition']}")
        
        # Add parameters
        if "parameters" in node:
            params_str = ", ".join(f"{k}={v}" for k, v in node["parameters"].items())
            parts.append(f"({params_str})")
        
        # Add position if present
        if "position" in node:
            pos = node["position"]
            parts.append(f"@[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        # Add timeout/retries
        if "timeout" in node:
            parts.append(f"timeout={node['timeout']}s")
        
        if "retries" in node:
            parts.append(f"retries={node['retries']}")
        
        label = " ".join(parts)
        
        # Apply color if enabled
        if self.use_colors:
            color = NodeColors.get_color(node_type)
            label = f"{color}{label}{Style.RESET_ALL}"
        
        return label
    
    def _print_tree_recursive(self, node: Dict[str, Any], prefix: str = "", 
                            is_last: bool = True, is_root: bool = True) -> List[str]:
        """Recursively print tree nodes."""
        lines = []
        
        # Determine the connector
        if is_root:
            connector = ""
            new_prefix = ""
        else:
            connector = self.symbols.LAST_BRANCH if is_last else self.symbols.BRANCH
            new_prefix = prefix + ("    " if is_last else f"{self.symbols.VERTICAL}   ")
        
        # Format and add the current node
        label = self._format_node_label(node)
        if connector:
            lines.append(f"{prefix}{connector}{self.symbols.HORIZONTAL}{self.symbols.HORIZONTAL} {label}")
        else:
            lines.append(label)
        
        # Process children if present
        if "children" in node and node["children"]:
            children = node["children"]
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                child_lines = self._print_tree_recursive(child, new_prefix, is_last_child, False)
                lines.extend(child_lines)
        
        return lines
    
    def print_tree(self, nodes: List[Dict[str, Any]], title: Optional[str] = None) -> str:
        """
        Print behavior tree in a pretty format.
        
        Args:
            nodes: List of behavior tree nodes
            title: Optional title for the tree
            
        Returns:
            Formatted string representation of the tree
        """
        if not nodes:
            return "Empty behavior tree"
        
        lines = []
        
        # Add title if provided
        if title:
            title_line = f" {title} "
            if self.use_colors:
                title_line = f"{Fore.CYAN}{Style.BRIGHT}{title_line}{Style.RESET_ALL}"
            lines.append(title_line.center(80, "═"))
            lines.append("")
        
        # Print each root node
        for i, node in enumerate(nodes):
            is_last = (i == len(nodes) - 1)
            node_lines = self._print_tree_recursive(node, "", is_last, True)
            lines.extend(node_lines)
            
            # Add spacing between multiple root nodes
            if not is_last:
                lines.append("")
        
        return "\n".join(lines)
    
    def print_tree_box(self, nodes: List[Dict[str, Any]], title: Optional[str] = None,
                      width: int = 80) -> str:
        """
        Print behavior tree in a box format.
        
        Args:
            nodes: List of behavior tree nodes
            title: Optional title for the tree
            width: Width of the box
            
        Returns:
            Formatted string representation of the tree in a box
        """
        tree_str = self.print_tree(nodes, None)
        tree_lines = tree_str.split("\n")
        
        # Calculate actual width needed
        max_line_length = max(len(line) for line in tree_lines) if tree_lines else 0
        box_width = max(width, max_line_length + 4)
        
        lines = []
        
        # Top border with title
        if title:
            title_str = f" {title} "
            padding = box_width - len(title_str) - 2
            left_pad = padding // 2
            right_pad = padding - left_pad
            
            if self.use_unicode:
                top_line = (f"{self.symbols.CORNER_TOP_LEFT}"
                          f"{self.symbols.HORIZONTAL * left_pad}"
                          f"{title_str}"
                          f"{self.symbols.HORIZONTAL * right_pad}"
                          f"{self.symbols.CORNER_TOP_RIGHT}")
            else:
                top_line = f"+{'-' * left_pad}{title_str}{'-' * right_pad}+"
            
            if self.use_colors:
                top_line = f"{Fore.CYAN}{Style.BRIGHT}{top_line}{Style.RESET_ALL}"
            
            lines.append(top_line)
        else:
            if self.use_unicode:
                lines.append(f"{self.symbols.CORNER_TOP_LEFT}"
                           f"{self.symbols.HORIZONTAL * (box_width - 2)}"
                           f"{self.symbols.CORNER_TOP_RIGHT}")
            else:
                lines.append(f"+{'-' * (box_width - 2)}+")
        
        # Content lines
        for tree_line in tree_lines:
            padding = box_width - len(tree_line) - 4
            if self.use_unicode:
                lines.append(f"{self.symbols.VERTICAL} {tree_line}{' ' * padding} {self.symbols.VERTICAL}")
            else:
                lines.append(f"| {tree_line}{' ' * padding} |")
        
        # Bottom border
        if self.use_unicode:
            bottom_line = (f"{self.symbols.CORNER_BOTTOM_LEFT}"
                         f"{self.symbols.HORIZONTAL * (box_width - 2)}"
                         f"{self.symbols.CORNER_BOTTOM_RIGHT}")
        else:
            bottom_line = f"+{'-' * (box_width - 2)}+"
        
        if self.use_colors:
            bottom_line = f"{Fore.CYAN}{bottom_line}{Style.RESET_ALL}"
        
        lines.append(bottom_line)
        
        return "\n".join(lines)
    
    def print_statistics(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Print statistics about the behavior tree.
        
        Args:
            nodes: List of behavior tree nodes
            
        Returns:
            Formatted statistics string
        """
        stats = self._calculate_statistics(nodes)
        
        lines = []
        lines.append("╔═══════════════════════════════════════╗")
        lines.append("║      Behavior Tree Statistics         ║")
        lines.append("╠═══════════════════════════════════════╣")
        lines.append(f"║ Total Nodes:        {stats['total_nodes']:18} ║")
        lines.append(f"║ Max Depth:          {stats['max_depth']:18} ║")
        lines.append(f"║ Leaf Nodes:         {stats['leaf_nodes']:18} ║")
        lines.append(f"║ Composite Nodes:    {stats['composite_nodes']:18} ║")
        lines.append(f"║ Action Nodes:       {stats['action_nodes']:18} ║")
        lines.append(f"║ Condition Nodes:    {stats['condition_nodes']:18} ║")
        lines.append("╚═══════════════════════════════════════╝")
        
        if self.use_colors:
            colored_lines = []
            for i, line in enumerate(lines):
                if i < 3 or i == len(lines) - 1:
                    colored_lines.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
                else:
                    colored_lines.append(line)
            return "\n".join(colored_lines)
        
        return "\n".join(lines)
    
    def _calculate_statistics(self, nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate tree statistics."""
        stats = {
            "total_nodes": 0,
            "max_depth": 0,
            "leaf_nodes": 0,
            "composite_nodes": 0,
            "action_nodes": 0,
            "condition_nodes": 0
        }
        
        def count_nodes(node: Dict[str, Any], depth: int = 1):
            stats["total_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)
            
            node_type = node.get("type", "").upper()
            
            # Categorize node
            if "children" in node and node["children"]:
                stats["composite_nodes"] += 1
                for child in node["children"]:
                    count_nodes(child, depth + 1)
            else:
                stats["leaf_nodes"] += 1
                
                if "CHECK" in node_type or "CONDITION" in node_type:
                    stats["condition_nodes"] += 1
                else:
                    stats["action_nodes"] += 1
        
        for node in nodes:
            count_nodes(node)
        
        return stats


def print_behavior_tree(nodes: List[Dict[str, Any]], title: Optional[str] = None,
                        format: str = "box", use_colors: bool = True,
                        show_stats: bool = False) -> None:
    """
    Print behavior tree in a pretty format.
    
    Args:
        nodes: List of behavior tree nodes
        title: Optional title for the tree
        format: Output format ("box", "tree", or "compact")
        use_colors: Whether to use colored output
        show_stats: Whether to show tree statistics
    """
    printer = BehaviorTreePrinter(use_colors=use_colors)
    
    if format == "box":
        print(printer.print_tree_box(nodes, title))
    elif format == "tree":
        print(printer.print_tree(nodes, title))
    else:  # compact
        print(json.dumps(nodes, indent=2))
    
    if show_stats:
        print()
        print(printer.print_statistics(nodes))


def tree_to_string(nodes: List[Dict[str, Any]], format: str = "tree") -> str:
    """
    Convert behavior tree to string representation.
    
    Args:
        nodes: List of behavior tree nodes
        format: Output format ("box", "tree", or "json")
        
    Returns:
        String representation of the tree
    """
    if format == "json":
        return json.dumps(nodes, indent=2)
    
    printer = BehaviorTreePrinter(use_colors=False)
    
    if format == "box":
        return printer.print_tree_box(nodes)
    else:
        return printer.print_tree(nodes)


# Example usage
if __name__ == "__main__":
    # Example behavior tree
    example_tree = [
        {
            "type": "Sequence",
            "name": "MainTask",
            "children": [
                {
                    "type": "Pick",
                    "object": "red_cube"
                },
                {
                    "type": "MoveTo",
                    "target": "blue_platform"
                },
                {
                    "type": "Place",
                    "target": "blue_platform"
                },
                {
                    "type": "Selector",
                    "name": "HandleObstacle",
                    "children": [
                        {
                            "type": "CheckCondition",
                            "condition": "path_clear"
                        },
                        {
                            "type": "Sequence",
                            "children": [
                                {
                                    "type": "Scan",
                                    "parameters": {"radius": 1.0}
                                },
                                {
                                    "type": "MoveTo",
                                    "target": "alternative_path"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    
    # Print in different formats
    print("\n" + "=" * 80)
    print("BOX FORMAT:")
    print("=" * 80)
    print_behavior_tree(example_tree, "Pick and Place Task", format="box", show_stats=True)
    
    print("\n" + "=" * 80)
    print("TREE FORMAT:")
    print("=" * 80)
    print_behavior_tree(example_tree, "Pick and Place Task", format="tree")
    
    print("\n" + "=" * 80)
    print("WITHOUT COLORS:")
    print("=" * 80)
    print_behavior_tree(example_tree, "Pick and Place Task", format="box", use_colors=False)