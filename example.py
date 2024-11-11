from tree import Tree, TreeNode
from search_algorithms.beam_search import BeamSearchTree
from search_algorithms.best_of_n import BestOfNTree
from typing import List, Callable, Any, Optional
import random

# from visualizer import TreeVisualizer

# Example function to generate child nodes with random values
def generate_children(node: TreeNode, num_children = 1) -> List[TreeNode]:
    mutations = ['A', 'C', 'G', 'T']
    children = []
    for _ in range(num_children):  # Generate 2 children for simplicity
        text = node.state['text']
        pos = random.randint(0, len(text) - 1)
        new_text = list(text)
        new_text[pos] = random.choice(mutations)
        new_text = ''.join(new_text)
        value = random.uniform(0, 1)
        children.append(TreeNode(state={'text' : new_text, 'logprob' : value}, score = node.score + value))
    return children

# Initialize the root node
root_state = {'text' : "AAAA", 'logprob' : 0.0}
root_node = TreeNode(state=root_state, score = 0.0)

# Create the Beam Search Tree with a beam width of 3 and retaining 2 solutions
beam_width = 10
k_solutions = 5
# tree = BeamSearchTree(root=root_node, beam_width=beam_width, top_k=k_solutions)
tree = BestOfNTree(root=root_node, n=beam_width, top_k=k_solutions)

# Perform Beam Search with a max depth of 3
# top_nodes = tree.beam_search(generate_children=generate_children, max_depth=5)
top_nodes = tree.best_of_n(generate_children=generate_children, max_depth=5)

print("Top-K Solutions:")
for i, node in enumerate(top_nodes, 1):
    print(f"Solution {i}: {node}")
    for parent in node.path():
        print(parent) 
    print()
print()

# breakpoint()  # To inspect the outputs