Code for LLM Project - Scaling Test Time Compute for MC Value Self-Verification.

This repository contains all code for the inference time search. Here are some relevant files:
- main_gsm8k.py is the main entrypoint for the codebase. Configs for all experiments can be found in the config folder.
- The search algorithms are implemented in the search_algorithms folder
- Ranking methods can be found in verify_gsm8k.py
- generator.py and reward_model.py are relevant files containing code to generate new steps and score them with a reward model
- tree.py contains code for the inference trees