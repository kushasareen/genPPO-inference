from transformers import (AutoModelForCausalLM,
    AutoTokenizer,
    set_seed)
import torch
from vllm import SamplingParams
import numpy as np


cache_dir = "/network/scratch/k/khang.ngo/cache"

class MathSphereRewardModel(torch.nn.Module):
    def __init__(self, args, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.prm_tokenizer = AutoTokenizer.from_pretrained(f"{args.reward_model}", cache_dir=cache_dir)
        self.prm_candidate_tokens = self.prm_tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.prm_tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        self.prm_model = AutoModelForCausalLM.from_pretrained(f"{args.reward_model}",
                                                        torch_dtype=torch.float16, cache_dir=cache_dir).eval()
        self.prm_model.to(args.device)
        self.device = args.device

    def forward(self, question, solution):
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

class GenVinePPOVerifier(torch.nn.Module):
    def __init__(self, args, vllm_model):
        super().__init__()

        self.llm = vllm_model
        self.tokenizer = vllm_model.get_tokenizer()
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids('Yes')
        self.no_token_id = self.tokenizer.convert_tokens_to_ids('No')
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=128, logprobs=True)

        self.verification_question = "\nIs the solution likely to result in the correct answer (Yes/No)?"
        #self.verification_question = "\nIs this step correct (Yes/No)?"

    def forward(self, prompt, solutions): 
        verification_prompts = []
        for solution in solutions:
            verification_prompt = prompt + solution + self.verification_question
            verification_prompts.append(verification_prompt)
        
        responses = self.llm.generate(verification_prompts, self.sampling_params, use_tqdm=False)
        
        logprobs = []
        tokens = []
        full_feedbacks = []
        for response, solution in zip(responses, solutions):
            first_output = response.outputs[0].logprobs[0]
            if self.yes_token_id in first_output:
                score = first_output[self.yes_token_id].logprob
                token = first_output[self.yes_token_id].decoded_token
            if self.no_token_id in first_output:
                no_logprob = first_output[self.no_token_id].logprob
                score = np.log( 1- np.exp(no_logprob)) 
                token = first_output[self.no_token_id].decoded_token
            tokens.append(token)
            logprobs.append(score)
            full_feedbacks.append(response.outputs[0].text)
        return logprobs, tokens, full_feedbacks