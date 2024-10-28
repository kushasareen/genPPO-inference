import vllm

class Node:
    def __init__(self, context, probability, history=None):
        self.context = context
        self.probability = probability
        self.history = history if history is not None else []

    def add_step(self, step):
        self.history.append(step)

class TreeInference:
    def __init__(self, model, prompts, args):
        self.model = model
        self.num_beams = args.num_beams
        self.prompts = prompts
        self.verification_prompt = args.prompt
        self.num_steps_per_node = args.num_steps_per_node

    def expand_node(self, node):
        expanded_nodes = []
        for _ in range(self.num_steps_per_node):
            prompt = self.prompts.format(context=node.context)
            response = self.model.generate(prompt)
            step_context = response.text  # New reasoning step
            probability = self.get_yes_probability(step_context)
            new_node = Node(context=step_context, probability=probability, history=node.history + [step_context])
            expanded_nodes.append(new_node)
        return expanded_nodes

    def get_yes_probability(self, context):
        prompt = self.verification_prompt.format(context=context)
        response = self.model.generate(prompt)
        yes_probability = response.get_probability("Yes")
        return yes_probability

class BeamSearch(TreeInference):
    def __init__(self, model, prompts, args):
        super().__init__(model, prompts, args)

    def search(self, initial_context):
        beams = [Node(context=initial_context, probability=1.0, history=[initial_context])]

        for _ in range(self.num_steps_per_node):
            all_candidates = []
            
            for node in beams:
                expanded_nodes = self.expand_node(node)
                all_candidates.extend(expanded_nodes)
            
            all_candidates.sort(key=lambda x: x.probability, reverse=True)
            beams = all_candidates[:self.num_beams]
            
        return beams

class BestofN(TreeInference): # not exactly right...
    def __init__(self, model, prompts, args):
        super().__init__(model, prompts, args)
        self.n = args.n

    def get_best_n_responses(self, context):
        prompt = self.verification_prompt.format(context=context)
        responses = self.model.generate(prompt, num_return_sequences=self.n)
        return responses
    
    def pick_best_response(self, context):
        responses = self.get_best_n_responses(context)
        best_response = max(responses, key=lambda r: r.get_probability("Yes"))
        return best_response