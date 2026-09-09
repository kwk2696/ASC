from typing import Union, List
import torch
import numpy as np

from embodied_cd.common.dataset_utils import PromptTemplate
from embodied_cd.common.agent import BaseAgent
from embodied_cd.trl.models.core import (
    _Type_Decoding,
    generation, 
    greedy_generation,
)
from embodied_cd.common.print_utils import *


class ASCAgent(BaseAgent):
    """ Agent for ASC' """
    name = "asc"

    gen_params = {
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.6,
        "temperature": 0.7
    }

    def __init__(
        self, 
        pre_model_name,
        base_model,
        base_tokenizer,
        model, 
        tokenizer,
        plan_model=None, 
        plan_tokenizer=None,
        env_name: str = 'virtualhome',
        cl_type: str = 'behavior',
        max_think_token: int = 80,
        total_think: int = 5,
        few_shot_example: str = None,
        decoding_strategy: _Type_Decoding = 'beam-action',
        test_time_thresh: dict = None,
        test_time_level: int = 3,
    ):
        super().__init__()

        self.pre_model_name = pre_model_name

        self.env_name = env_name
        self.cl_type = cl_type
        
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.max_think_token = max_think_token
        self.total_think = total_think
        self.think_template = PromptTemplate(env_name, "cd-think", few_shot_example)

        self.plan_model = plan_model
        self.plan_tokenizer = plan_tokenizer
        self.plan_template = PromptTemplate(env_name, "cd-action-think")
        self.action_format = PromptTemplate.load_env_action_format(env_name)

        self.decoding_strategy = decoding_strategy

        self.reset_stats()
        
        # test_time thresholds
        self.test_time_thresh = test_time_thresh
        self.test_time_level = test_time_level
        self.z_values = [-0.842, -0.524, -0.253, 0, 0.253, 0.524, 0.842]

    def reset(self, task, goal):
        return

    def reset_stats(self):
        self.forward_count = 0
        self.generated_tokens = [0, 0, 0, 0]  # full rationale, used rationale prefix, action, slots used
        self.reasoning_counts = [0 for _ in range(self.total_think)]

    def querize(self, _query):
        if 'instruct' in self.pre_model_name or 'Instruct' in self.pre_model_name:
            query = self._convert_to_chat([_query], list())
            query = self.tokenizer.apply_chat_template(
                query, tokenize=False, add_generation_prompt=True)
        else:
            query = f"### Human: {_query}\n### Assistant:"
        return query
    
    def generate(self, model, tokenizer, query, max_length, sample):
        # 2. Greedy Generation
        with torch.no_grad():
            if not sample:
                generation_output = greedy_generation(
                    model, tokenizer, query, max_length)
            else:
                generation_output = generation(
                    model, tokenizer, query, max_length, **self.gen_params) 
        return generation_output
    
    def get_token_length(self, query):
        return len(self.tokenizer.encode(query, return_tensors="np")[0])

    def get_think(
        self,
        model,
        tokenizer,
        instruction: str,
        state: str,
        history: str,
        sample: bool = False
    ):
        # Set Adapter
        model.set_adapter('reasoning_policy')
        
        think_list = []
        for j in range(self.total_think):
            # Generation
            query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\n{PromptTemplate.correct_think_prompts[j]}"
            query = self.querize(query)
            generation_output = self.generate(model, tokenizer, query, self.max_think_token, sample)

            # Post-processing
            think = generation_output.response
            think = think.strip().split("\n")[0]
            think = think.split(". ")[0] 
            think_list.append(think)
        return " ".join(think_list), think_list
    
    def get_action(
        self,
        instruction: str,
        state: str,
        history: str,
        think: str = None,
    ):
        # Set Adapter
        self.model.set_adapter('planning_policy')
        
        # Generation
        query = self.plan_template(instruction=instruction, state=state, think=think, history=history)["query"]
        query = self.querize(query)
        generation_output = self.generate(self.model, self.tokenizer, query, 20, False)

        # Post-processing
        action = generation_output.response
        log_prob = generation_output.log_probs
        action = action.strip().split("\n")[0]
        token_length = self.get_token_length(action)
        if token_length == 0:  # empty action: never confident
            return action, float("-inf")
        log_prob = np.sum(log_prob[:token_length]) / token_length
        return action, log_prob
    
    def forward(
        self,
        instruction: str,
        state: str,
        history: str,
        few_shot_examples: Union[str, List[str]] = None,
    ):
        state = PromptTemplate.preprocess(state)
        if self.test_time_thresh is not None:
            think, think_list = self.get_think(self.model, self.tokenizer, instruction, state, history, sample=False)
            self.forward_count += 1
            self.generated_tokens[0] += self.get_token_length(think)
            print("=" * 30)
            for j in range(len(think_list)):
                think_seg = " ".join(think_list[:j+1])
                action, log_prob = self.get_action(instruction, state, history, think_seg)
                self.generated_tokens[2] += self.get_token_length(action)
                thresh = self.test_time_thresh['average'][j] + \
                        self.z_values[self.test_time_level] * self.test_time_thresh['deviation'][j]
                print_check(f"At {j+1}: {log_prob} | {thresh}")
                if log_prob > thresh:
                    print_warn(f"Stop!!")
                    break
            self.generated_tokens[1] += self.get_token_length(think_seg)
            self.generated_tokens[3] += (j + 1)
            self.reasoning_counts[j] += 1
            print(f"Added ... {j+1}, {think_seg}")
        else:
            think, think_list = self.get_think(self.model, self.tokenizer, instruction, state, history, sample=False)
            action, prob = self.get_action(instruction, state, history, think)
            self.forward_count += 1
            self.generated_tokens[0] += self.get_token_length(think)
            self.generated_tokens[1] += self.get_token_length(think)
            self.generated_tokens[2] += self.get_token_length(action)
            self.generated_tokens[3] += len(think_list)
            self.reasoning_counts[len(think_list) - 1] += 1
            print("=" * 30)
            print(think)
            print(action)
        return action

    def get_generated_tokens(self):
        return np.array(self.generated_tokens) / self.forward_count

    def get_reasoning_counts(self):
        return self.reasoning_counts
