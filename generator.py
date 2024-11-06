from tree import TreeNode


def run_inference(model, sampling_params, prompt):
    """Run inference on the given model with a prompt."""     
    responses = model.generate(prompt, sampling_params, use_tqdm=False)
    return responses

class NodeGenerator:
    def __init__(self, policy, reward_model, num_children, sampling_params):
        self.policy = policy
        self.reward_model = reward_model
        self.num_children = num_children
        self.sampling_params = sampling_params
    
    def __call__(self, node):
        prompt = node.state['text']
        batch_prompt = [prompt] * self.num_children
        responses = run_inference(self.policy, self.sampling_params, batch_prompt)
        all_children = []
        solutions = [candidate.outputs[0].text for candidate in responses]
        logprobs, tokens, full_feedbacks = self.reward_model(prompt, solutions) 
        for (solution, logprob, token, full_feedback) in zip(solutions, logprobs, tokens, full_feedbacks):
            text = prompt + solution + '\n'
            child = TreeNode(state = {'text' : text, 'logprob' : logprob, 'token' : token, 'step_solution' : solution, 
                                      'full_feedback': full_feedback}, 
                             score = node.score + logprob, 
                            parent = node, depth = 0)
            all_children.append(child)
        return all_children