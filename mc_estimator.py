from generator import run_async_inference
from verify_gsm8k import extract_gold_answer_from_text, extract_predicted_answer_from_text, grade_answer
import asyncio
import uuid
from vllm import SamplingParams

class MonteCarloEstimator:
    def __init__(self, policy, args, gamma=0.99):
        self.policy = policy
        self.gamma = gamma
        self.sampling_params = SamplingParams(temperature=args.generation_temp, max_tokens=1024) # \n not in stop words

    async def estimate(self,  node, ref, num_episodes=64): # vineppo uses 256 but that's too much for me
        """Estimate the value of a node using Monte Carlo estimation."""

        prompt = node.state['text']
        batch_prompt = [prompt] * num_episodes

        tasks = []

        for prompt in batch_prompt:
            tasks.append(asyncio.create_task(run_async_inference(self.policy, self.sampling_params, prompt, uuid.uuid4())))

        responses = [await task for task in tasks]        

        solution_candidates = [candidate.outputs[0].text for candidate in responses]
        gold_answer = extract_gold_answer_from_text(ref["answer"])
    
        assert len(solution_candidates) > 0
        answer_candidates = [
            extract_predicted_answer_from_text(sol)
            for sol in solution_candidates
        ]

        grading_results = [
            grade_answer(given_answer=ans, ground_truth=gold_answer, item=ref)
            for ans in answer_candidates
        ]

        return sum(grading_results) / len(grading_results) # note: for now, we are not using gamma, ask arian what he's doing during training

class NodeSampler:
    pass