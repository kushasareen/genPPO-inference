from tree import TreeNode
import asyncio
import uuid

def run_inference(model, sampling_params, prompt):
    """Run inference on the given model with a prompt."""     
    responses = model.generate(prompt, sampling_params, use_tqdm=False)
    return responses

async def run_async_inference(engine, sampling_params, prompt, id):
    results_generator = engine.generate(prompt, sampling_params, id)

    responses = None

    async for request_output in results_generator:
        responses = request_output
    
    return responses

class Generator:
    def __init__(self, policy, reward_model, num_children, sampling_params, args):
        self.policy = policy
        self.reward_model = reward_model
        self.num_children = num_children
        self.sampling_params = sampling_params
        self.aggregator = args.aggregator
        self.log_all_scores = args.log_all_scores
        self.token_count = 0

    def get_score(self, parent_score, logprob):
        if self.aggregator == 'sum':
            return parent_score + logprob
        elif self.aggregator == 'min':
            return min(parent_score, logprob)
        elif self.aggregator == 'last':
            return logprob
        else:
            raise ValueError(f"Aggregator not implemented: {self.aggregator}")
        
    def get_all_scores(self, parent_score, logprob):
        return {"sum": parent_score["sum"] + logprob, "min": min(parent_score["min"], logprob), "last": logprob}


class NodeGenerator(Generator):  
    async def __call__(self, node):
        prompt = node.state['text']
        batch_prompt = [prompt] * self.num_children
        responses = run_inference(self.policy, self.sampling_params, batch_prompt)
        for response in responses:
            self.token_count += len(response.outputs[0].token_ids)

        all_children = []
        solutions = [candidate.outputs[0].text for candidate in responses]
        logprobs, tokens, full_feedbacks = await self.reward_model(prompt, solutions) 
        for (solution, logprob, token, full_feedback) in zip(solutions, logprobs, tokens, full_feedbacks):
            text = prompt + solution + '\n'
            child = TreeNode(state = {'text' : text, 'logprob' : logprob, 'token' : token, 'step_solution' : solution, 
                                      'full_feedback': full_feedback}, 
                             score = self.get_score(node.score, logprob),
                            parent = node, depth = 0, all_scores = self.get_all_scores(node.all_scores, logprob) if self.log_all_scores else {self.aggregator: self.get_score(node.score, logprob)})
            all_children.append(child)
        return all_children

class AsyncNodeGenerator(Generator):
    async def __call__(self, node):
        prompt = node.state['text']
        batch_prompt = [prompt] * self.num_children

        tasks = []

        for prompt in batch_prompt:
            tasks.append(asyncio.create_task(run_async_inference(self.policy, self.sampling_params, prompt, uuid.uuid4())))

        responses = [await task for task in tasks]
        for response in responses:
            self.token_count += len(response.outputs[0].token_ids)

        all_children = []
        solutions = [candidate.outputs[0].text for candidate in responses]
        logprobs, tokens, full_feedbacks = await self.reward_model(prompt, solutions) 
        for (solution, logprob, token, full_feedback) in zip(solutions, logprobs, tokens, full_feedbacks):
            text = prompt + solution + '\n'
            child = TreeNode(state = {'text' : text, 'logprob' : logprob, 'token' : token, 'step_solution' : solution, 
                                      'full_feedback': full_feedback}, 
                             score = self.get_score(node.score, logprob), all_scores = self.get_all_scores(node.all_scores, logprob) if self.log_all_scores else None,
                            parent = node, depth = 0)
            all_children.append(child)
        return all_children
