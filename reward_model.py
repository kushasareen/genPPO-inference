from transformers import (AutoModelForCausalLM,
    AutoTokenizer,
    set_seed)
import torch
from vllm import SamplingParams
import numpy as np
from generator import run_inference, run_async_inference
import asyncio
import uuid

class MathSphereRewardModel(torch.nn.Module):
    def __init__(self, args, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.prm_tokenizer = AutoTokenizer.from_pretrained(f"{args.reward_model}", cache_dir=args.download_dir)
        self.prm_candidate_tokens = self.prm_tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.prm_tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        self.prm_model = AutoModelForCausalLM.from_pretrained(f"{args.reward_model}",
                                                        torch_dtype=torch.float16, cache_dir=args.download_dir).eval()
        self.prm_model.to(args.device)
        self.device = args.device
        self.token_count = 0

    async def get_score_from_model(self, question, solution):
        if len(solution) == 0:
            input_for_prm = f"{question} {self.step_tag}"
        elif solution[-1] != self.step_tag:
            input_for_prm = f"{question} {solution[:-1]}" + " " + self.step_tag
        else:
            input_for_prm = f"{question} {solution}"
        input_id = torch.tensor([self.prm_tokenizer.encode(input_for_prm)]).to(self.device)
        with torch.no_grad():
            logits = self.prm_model(input_id).logits[:,:,self.prm_candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            log_prob = scores.log()
            step_log_prob = log_prob[input_id == self.step_tag_id]
            step_log_prob = step_log_prob.cpu()[-1].item()
        return step_log_prob
    
    async def forward(self, prompt, solutions):
        tasks = []

        for solution in solutions:
            tasks.append(asyncio.create_task(self.get_score_from_model(prompt, solution)))

        logprobs = [await task for task in tasks]

        tokens = ['N/A'] * len(solutions)
        full_feedbacks = ['N/A'] * len(solutions)
        return logprobs, tokens, full_feedbacks

class GenVinePPOVerifier(torch.nn.Module):
    def __init__(self, args, vllm_model, tokenizer):
        super().__init__()

        self.llm = vllm_model
        self.tokenizer = tokenizer
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids('Yes')
        self.no_token_id = self.tokenizer.convert_tokens_to_ids('No')
        self.sampling_params = SamplingParams(temperature=args.verification_temp, max_tokens=1, logprobs=20)

        self.verification_question = args.verification_question
        self.token_count = 0
        self.args = args

    def get_verification_prompt(self, problem, solution):
        return problem + solution + self.verification_question
        
    def get_score(self, response):
        if len(response.outputs) == 0 or len(response.outputs[0].logprobs) == 0:
            score = -100.0
            token = 'N/A'
            feedback = 'N/A'
            return score, token, feedback

        first_output = response.outputs[0].logprobs[0]
        if self.yes_token_id in first_output: # if yes token is in the top 20 logprobs (it should always be), score is the logprob of yes token
            score = first_output[self.yes_token_id].logprob

        else: # otherwise, we set the score to -100
            score = -100.0

        token = response.outputs[0].text
        feedback = response.outputs[0].text

        return score, token, feedback


    async def forward(self, prompt, solutions): 
        verification_prompts = []
        for solution in solutions:
            verification_prompt = self.get_verification_prompt(prompt, solution)
            verification_prompts.append(verification_prompt)
        
        tasks = []

        for prompt in verification_prompts:
            tasks.append(asyncio.create_task(run_async_inference(self.llm, self.sampling_params, prompt, uuid.uuid4())))

        responses = [await task for task in tasks]

        for response in responses:
            self.token_count += len(response.outputs[0].token_ids)

        logprobs = []
        tokens = []
        full_feedbacks = []
        for response, solution in zip(responses, solutions):
            score, token, feedback = self.get_score(response)

            tokens.append(token)
            logprobs.append(score)
            full_feedbacks.append(feedback)
            
        return logprobs, tokens, full_feedbacks

class LLMAsAJudge(GenVinePPOVerifier):
    def __init__(self, args, vllm_model, tokenizer):
        super().__init__(args, vllm_model, tokenizer)
        self.sampling_params = SamplingParams(temperature=args.verification_temp, max_tokens=512, logprobs=20)

    def get_verification_prompt(self, problem, solution):
        return f"You are a math teacher. Grade the Solution, verifying correctness step by step. At the end of the Solution verification, when you give your final grade, write it in the form \"Verification: Is the answer correct (Yes/No)? X\", where X is either Yes or No. \n Question: {problem}\nSolution: {solution}\n"
    
    def get_score(self, response): # can also do some parsing, for now just take the last token
        if len(response.outputs) == 0 or len(response.outputs[0].logprobs) == 0:
            score = -100.0
            token = 'N/A'
            feedback = 'N/A'
            return score, token, feedback

        last_output = response.outputs[0].logprobs[-1] # double check
        if self.yes_token_id in last_output: # if yes token is in the top 20 logprobs (it should always be), score is the logprob of yes token
            score = last_output[self.yes_token_id].logprob

        else: # otherwise, we set the score to -100
            score = -100.0

        token = response.outputs[0].text
        feedback = response.outputs[0].text

        return score, token, feedback
