import heapq
from typing import List, Callable, Any, Optional
from tree import Tree, TreeNode
import numpy as np
import random
import asyncio
import scipy.special as scsp

def is_terminal(node):
    if 'The answer is' in node.state['text']:
        return True
    if 'The final answer is' in node.state['text']:
        return True
    if '####' in node.state['text']:
        return True
    return False


class RebaseTree(Tree):
    """
    A tree structure to perform beam search, retaining the top-K solutions
    at each depth level based on their cumulative value.
    """

    def __init__(self, root: TreeNode, expansion_temp: float, top_k: int = None):
        super().__init__(root)  # Initializwidthse the base Tree with the root
        self.expansion_temp = expansion_temp  # Maximum number of nodes to retain per level
        self.top_k = top_k  # Number of top-K solutions to return
        self.budget = top_k # Set the budget initially to the top_k value

    def get_beam_widths(self, current_beam: List[TreeNode]) -> int:
        return np.round(self.budget * scsp.softmax([np.exp(n.score)/self.expansion_temp for n in current_beam])).astype(int) # exponentiate the logprob to get the prob

    async def search(self, generate_children: Callable[[Any], List[TreeNode]], max_depth: int) -> List[TreeNode]:
        """
        Perform Beam Search to find the top-K solutions with the highest cumulative values.

        Args:
            generate_children: Function to generate child nodes from a given state.
            max_depth: Maximum depth to explore.

        Returns:
            A list of the top-K nodes found after exploring up to the given depth.
        """
        # Start the beam with the root node
        terminal_nodes = []
        current_beam = [self.root]

        for depth in range(max_depth): 
            # print("Depth: ", depth)
            # List to store all children generated in this depth level
            next_beam = []
            widths = self.get_beam_widths(current_beam)

            # Expand each node in the current beam
            for idx, node in enumerate(current_beam):
                children = await generate_children(node, widths[idx])
                for child in children:
                    if is_terminal(child):
                        terminal_nodes.append(child)
                        self.budget -= 1
                    else:
                        next_beam.append(child)  # Add to the list of candidates for the next beam
                    node.add_child(child)  # Add the child to the parent node

            # If no more nodes are available to explore, stop
            if not next_beam:
                break

            # Update the current beam to the next beam
            current_beam = next_beam

        # Return the top-K nodes from the final beam
        if len(terminal_nodes) == 0:
            return heapq.nlargest(self.top_k, terminal_nodes + current_beam, key=lambda n: n.score)
 
        return heapq.nlargest(self.top_k, terminal_nodes, key=lambda n: n.score)