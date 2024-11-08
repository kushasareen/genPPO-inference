from search_algorithms.beam_search import BeamSearchTree
from search_algorithms.best_of_n import BestOfNTree
from generator import NodeGenerator

def get_search_tree_and_generator(root, llm, reward_model, sampling_params, args):
    if args.search_algorithm == "beamsearch":
        return BeamSearchTree(root=root, beam_width=args.beam_width), NodeGenerator(llm, reward_model, args.beam_width, sampling_params)
    elif args.search_algorithm == "bestofn":
        return BestOfNTree(root=root, n=args.n), NodeGenerator(llm, reward_model, num_children=1, sampling_params=sampling_params)
    else:
        raise ValueError(f"Search algorithm not implemented: {args.search_algorithm}")