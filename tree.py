import argparse
import json
import re
import time
#from sglang import function, gen, RuntimeEndpoint
import fcntl
import os
import math
import threading
from typing import List, Optional, Callable, Any


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_prompts(args):
    test_cases = read_jsonl(args.input_path)
    prompts = []
    for test in test_cases:
        prompts.append(test["problem"])
    return prompts, test_cases


class TreeNode:
    """
    A generic tree node that stores a state, cumulative value (fitness or score),
    depth, and references to its parent and children.
    """

    def __init__(self, 
                 state: Any, 
                 score: float = 0.0, 
                 parent: Optional['TreeNode'] = None, 
                 depth: int = 0):
        self.state = state  # The state or data represented by this node
        self.score = score  # The value of this node (e.g., fitness or heuristic score) 
        self.parent = parent  # A reference to the parent node
        self.depth = depth  # The depth of this node in the tree
        self.children: List['TreeNode'] = []  # List of child nodes

    def add_child(self, child: 'TreeNode'):
        """Add a child node to this node and set its parent."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    def path(self) -> List[Any]:
        """
        Return the path from the root to this node as a list of states.
        Useful for tracing solutions in search algorithms.
        """
        node, path = self, []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]  # Reverse to get the path from root to this node

    def __lt__(self, other: 'TreeNode'):
        """Comparison based on cumulative value, useful for priority queues."""
        return self.score < other.score

    def __repr__(self):
        return (f"TreeNode(state={self.state} | " 
                f"Score={self.score}, depth={self.depth})")


class Tree:
    """
    A generic tree structure to support various search algorithms.
    Maintains a reference to the root node and provides utility functions
    to expand nodes, apply pruning, and retrieve solutions.
    """

    def __init__(self, root: TreeNode, threshold: Optional[float] = None):
        self.root = root  # The root of the tree
        self.threshold = threshold  # Threshold for pruning based on cumulative value

    def expand_node(self, node: TreeNode, 
                    generate_children: Callable[[Any], List[TreeNode]]):
        """
        Expand a given node using the provided function to generate children.
        Adds each generated child as a child of the given node only if its
        cumulative value exceeds the threshold.
        """
        children = generate_children(node.state)
        for child in children:
            if self.threshold is None or child.cumulative_value >= self.threshold:
                node.add_child(child)

    def find_leaf_nodes(self) -> List[TreeNode]:
        """Return all leaf nodes in the tree."""
        return [node for node in self.traverse(self.root) if node.is_leaf()]

    def traverse(self, node: Optional[TreeNode] = None) -> List[TreeNode]:
        """
        Traverse the tree using DFS and return a list of all nodes.
        """
        if node is None:
            node = self.root
        nodes = [node]
        for child in node.children:
            nodes.extend(self.traverse(child))
        return nodes

    def get_best_leaf(self) -> Optional[TreeNode]:
        """
        Get the leaf node with the highest cumulative value (fitness or heuristic).
        """
        leaf_nodes = self.find_leaf_nodes()
        if not leaf_nodes:
            return None
        return max(leaf_nodes, key=lambda n: n.score)

    def __repr__(self):
        return f"Tree(root={self.root})"
                
