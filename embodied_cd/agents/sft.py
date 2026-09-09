from typing import Union, List
import torch
import numpy as np

from embodied_cd.common.dataset_utils import PromptTemplate
from embodied_cd.common.agent import BaseAgent
from embodied_cd.trl.models.core import (
    _Type_Decoding, 
    generation, 
    greedy_generation, 
    beam_action_generation,
)
from embodied_cd.common.print_utils import *


class SFTAgent(BaseAgent):
    """ Agent for SFT """
    name = "sft"

    gen_params = {
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.3,
        "temperature": 0.2
    }

    def __init__(
        self,
        base_model_name=None,
        think_model=None,
        think_tokenizer=None,
        plan_model=None,
        plan_tokenizer=None,
        env_name: str = 'virtualhome',
        cl_type: str = 'behavior',
        max_think_token: int = 80,
        max_think_times: int = 5,
        decoding_strategy: _Type_Decoding = 'beam-action',
    ):
        super().__init__()

        self.base_model_name = base_model_name

        self.env_name = env_name
        self.cl_type = cl_type

        self.think_model = think_model
        self.think_tokenizer = think_tokenizer
        self.max_think_token = max_think_token
        self.max_think_times = max_think_times
        self.think_template = PromptTemplate(env_name, "cd-think")
        
        self.plan_model = plan_model
        self.plan_tokenizer = plan_tokenizer
        if self.think_model is None:
            self.plan_template = PromptTemplate(env_name, "cd-action")
        else:
            self.plan_template = PromptTemplate(env_name, "cd-action-think")
        self.action_format = PromptTemplate.load_env_action_format(env_name)
        self.action_dict = PromptTemplate.load_env_action_dict(env_name)
        if plan_model is None:
            raise ValueError("SFTAgent needs a planning model (plan_model / plan_tokenizer)")

        self.decoding_strategy = decoding_strategy

    def reset(self, task, goal):
        return

    def get_think(
        self,
        instruction: str,
        state: str,
        history: str,
        sample: bool = False
    ):
        # 1. Setting Think Template
        query = self.think_template(instruction=instruction, state=state, history=history)
        if 'instruct' in self.base_model_name or 'Instruct' in self.base_model_name:
            query = self._convert_to_chat([query["query"]], list())
            query = self.think_tokenizer.apply_chat_template(
                query, tokenize=False, add_generation_prompt=True)
        else:
            query = self._convert_to_completion(query["query"])

        # 2. Greedy Generation
        with torch.no_grad():
            if not sample:
                generation_output = greedy_generation(
                    self.think_model, self.think_tokenizer, query, self.max_think_token)
            else:
                generation_output = generation(
                    self.think_model, self.think_tokenizer, query, self.max_think_token, **self.gen_params)

        # 3. Post-processing
        think = generation_output.response       
        think = think.strip().split("\n")[0]
        return think

    def get_action(
        self,
        instruction: str,
        state: str,
        history: str,
        think: str = None,
    ):
        # 1. Setting Action Template
        query = self.plan_template(instruction=instruction, state=state, think=think, history=history)
        if 'instruct' in self.base_model_name or 'Instruct' in self.base_model_name:
            query = self._convert_to_chat([query["query"]], list())
            query = self.plan_tokenizer.apply_chat_template(
                query, tokenize=False, add_generation_prompt=True)
        else:
            query = self._convert_to_completion(query["query"])
        
        # 2. Greedy Generation
        with torch.no_grad():
            if self.decoding_strategy == 'beam-action':
                object_list = PromptTemplate.get_object_list(state)
                generation_output = beam_action_generation(
                    self.plan_model, self.plan_tokenizer, query, self.action_format, object_list)
            elif self.decoding_strategy == 'greedy':
                generation_output = greedy_generation(
                    self.plan_model, self.plan_tokenizer, query, max_length=20)
            else:
                raise NotImplementedError
        
        # 3. Post-processing
        action = generation_output.response
        action = action.strip().split("\n")[0]
        prob = generation_output.prob
        if not isinstance(prob, float):
            prob = prob.item()
        return action, prob

    def forward(
        self,
        instruction: str,
        state: str,
        history: str,
        few_shot_examples: Union[str, List[str]] = None,
    ):
        think, action = '', ''
        state = PromptTemplate.preprocess(state)
        if self.think_model is not None: # thinking model + planning model
            if self.max_think_times == 1:
                think = self.get_think(instruction, state, history)
                action, prob = self.get_action(instruction, state, history, think)
            else:
                actions, thinks, probs = [], [], []
                for _ in range(self.max_think_times):
                    think = self.get_think(instruction, state, history, sample=True)
                    action, prob = self.get_action(instruction, state, history, think)
                    actions.append(action)
                    thinks.append(think)
                    probs.append(prob)
                idx = np.argmax(probs)
                action = actions[idx] 
        else: # only planning model
            action, prob = self.get_action(instruction, state, history, think)

        return action
