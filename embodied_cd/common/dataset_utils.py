import os
import re
import math
import random
import json
import tqdm
import numpy as np
import torch
from typing import Tuple, List, Dict, Generator, Literal
from functools import partial

from datasets import Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vh_dataset.dataset.jsonl import JsonlDataset
from vh_dataset.dataset.knowledge_graph import KG

from embodied_cd.common.llm_utils import OpenAILLM
from embodied_cd.common.print_utils import *

_Type_Environment = Literal["virtualhome", "alfred"]
_Type_Template = Literal[
    "default", "cd-think", "cd-think-NXT", "cd-action", "cd-action-think", "cd-reward"
]


# # # Prompt Templates # # #
class PromptTemplate:
    init_think_prompt = "You will analyze the instruction and current state where the character is tasked to complete the instruction. You should provide exactly 5 sentence responses that only incorporates:\n1. physical location and status of the character\n2. physical location and status of observations that are only related to the instruction\n3. summarization of previous action histories if previous actions are available\n4.break down the remaining plan to complete the instruction if remaining plans are required\n5. reasoning for what should do next to complete the instruction.\nEach sentence serves a specific purpose while maintaining clarity."

    correct_think_prompts = [
        "You should provide exactly 1 sentence response that only incorporate: physical location and status of the character.",
        "You should provide exactly 1 sentence response that only incorporate: physical location and status of observations that are only related to the instruction.",
        "You should provide exactly 1 sentence response that only incorporate: summarization of previous action histories if previous actions are available.",
        "You should provide exactly 1 sentence response that only incorporate: break down the remaining plan to complete the instruction if remaining plans are required.",
        "You should provide exactly 1 sentence response that only incorporate: reasoning for what should do next.",
    ]

    action_prompt = "In order to complete the given instruction, what should be the next immediate action?"

    def __init__(
        self,
        env_name: _Type_Environment,
        dataset_type: _Type_Template,
        example: str = None,
    ):
        self.dataset_type = dataset_type
        self.env_prompt = self.load_env_prompt(env_name)
        self.example = example

    @classmethod
    def load_env_prompt(cls, env_name):
        filepath = os.path.join(os.getcwd(), f"configs/{env_name}/env_prompt.json")
        with open(filepath, "r") as f:
            env_prompt = json.load(f)["env_prompt"]
        return env_prompt

    @classmethod
    def load_env_action_dict(cls, env_name):
        filepath = os.path.join(os.getcwd(), f"configs/{env_name}/env_prompt.json")
        with open(filepath, "r") as f:
            env_ngram = json.load(f)["env_action_dict"]
        return env_ngram

    @classmethod
    def load_env_action_format(cls, env_name):
        filepath = os.path.join(os.getcwd(), f"configs/{env_name}/env_prompt.json")
        with open(filepath, "r") as f:
            env_ngram = json.load(f)["env_action_format"]
        return env_ngram

    @classmethod
    def preprocess(cls, state, ret="str"):
        """process state"""
        pattern = r"\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*\)"
        matches = re.findall(pattern, state)
        matches = [list(mat) for mat in matches]
        if ret == "list":
            return matches

        state_char, state_room, state_else = [], [], []
        for mat in matches:
            if mat[0] == "character":
                state_char.append(str(tuple(mat)))
            elif mat[0] in ["kitchen", "bathroom", "bedroom", "livingroom"]:
                state_room.append(str(tuple(mat)))
            else:
                state_else.append(str(tuple(mat)))
        str_char = ", ".join(state_char)
        str_room = ", ".join(state_room)
        str_else = ", ".join(state_else)
        if len(state_char) == 0 and len(state_room) == 0:
            str_full = str_else
        else:
            str_full = str_char + ", " + str_room + ", " + str_else
        return str_full.replace("'", "")

    @classmethod
    def get_object_list(cls, state):
        ban_list = ["character", "closed", "open", "visible", "none", "off", "on"]

        matches = cls.preprocess(state, ret="list")
        state_list = []
        for mat in matches:
            if (mat[0] not in state_list) and (mat[0] not in ban_list):
                state_list.append(mat[0])
            if (mat[2] not in state_list) and (mat[2] not in ban_list):
                state_list.append(mat[2])
        return state_list

    # need to add possible action set
    def prompt_default(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        query = (
            f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}"
        )
        return {"query": query, "response": action}

    def prompt_default_base(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        query = (
            f"Instruction: {instruction}\nState: {state}"
        )
        return {"query": query, "response": action}

    def prompt_think(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        if example is None:
            query = f"{PromptTemplate.init_think_prompt}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}"
        else:
            query = f"{PromptTemplate.init_think_prompt}\n{example}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}"
        return {"query": query, "response": think}

    def prompt_think_nxt(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        if example is None:
            query = f"{PromptTemplate.init_think_prompt}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}"
        else:
            query = f"{PromptTemplate.init_think_prompt}\n{example}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}"
        # process think
        think_list = think.split(".")[:4]
        think_with_token = ". [NXT]".join(think_list) + "."
        return {"query": query, "response": think_with_token}

    def prompt_action(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        if example is None:
            query = f"{self.env_prompt}\n{PromptTemplate.action_prompt}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}"
        else:
            query = f"{self.env_prompt}\n{PromptTemplate.action_prompt}\n{example}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}"
        return {"query": query, "response": action}

    def prompt_action_think(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        if example is None:
            query = f"{self.env_prompt}\n{PromptTemplate.action_prompt}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}\nThink: {think}"
        else:
            query = f"{self.env_prompt}\n{PromptTemplate.action_prompt}\n{example}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}\nThink: {think}"
        return {"query": query, "response": action}

    def prompt_reward(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\nThink: {think}"
        return {"query": query, "response": action}

    def prompt_plan(
        self, instruction, state, think, action, history, example, **kwargs
    ):
        query = f"{self.env_prompt}\n{PromptTemplate.action_prompt}\nInstruction: {instruction}\nState: {state}\nPrevious Actions: {history}\nThink: {think}\nMake a full plan (action sequences) to complete the given instruction."
        return {"query": query}

    def __call__(
        self,
        instruction,
        state,
        think=None,
        action=None,
        history=None,
        example=None,
        **kwargs,
    ):
        state = self.preprocess(state)
        if example is None:
            example = self.example
        match self.dataset_type:
            case "default":
                return self.prompt_default(
                    instruction, state, think, action, history, example, **kwargs
                )
            case "default-base":
                return self.prompt_default_base(
                    instruction, state, think, action, history, example, **kwargs
                )
            case "cd-think":
                return self.prompt_think(
                    instruction, state, think, action, history, example, **kwargs
                )
            case "cd-think-NXT":
                return self.prompt_think_nxt(
                    instruction, state, think, action, history, example, **kwargs
                )
            case "cd-action":
                return self.prompt_action(
                    instruction, state, think, action, history, example, **kwargs
                )
            case "cd-action-think":
                return self.prompt_action_think(
                    instruction, state, think, action, history, example, **kwargs
                )
            case "cd-reward":
                return self.prompt_reward(
                    instruction, state, think, action, history, example, **kwargs
                )
            case "cd-plan":
                return self.prompt_plan(
                    instruction, state, think, action, history, example, **kwargs
                )


# # # Dataset # # #
class VirtualHomeDataset(JsonlDataset):
    additional_special_tokens = {"query": "### Human:", "response": "### Assistant:"} #{"query": "<|prompt|>", "response": "<|completion|>"}

    def set_template(self, dataset_type: _Type_Template, example: str = None):
        self.template = PromptTemplate("virtualhome", dataset_type, example)

    @classmethod
    def __generate_dataset(cls, dataset_path: str, percent: float, shuffle: bool):
        source = []
        with open(dataset_path) as f:
            for line in f:
                try:
                    json_line = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"malformed JSON line in {dataset_path}: {line[:120]!r}") from e
                source.append(json_line)

        if percent != 1.0:
            print_warn(f"Original Dataset: {len(source)}")
            total_traj, traj = [], []
            for s in source:
                traj.append(s)
                if s["reward"] == "1.0":
                    total_traj.append(traj)
                    traj = []

            print(f"Total Traj: {len(total_traj)}")

            new_source = []
            chunk_size = int(1 / percent)

            print(f"Chunk Size: {chunk_size}")
            for i in range(0, len(total_traj), chunk_size):
                chunk = total_traj[i:i + chunk_size]
                if chunk: # if not empty list
                    selected = random.choice(chunk)
                    new_source.extend(selected)
            print_check(f"New Dataset: {len(new_source)}")
            source = new_source
            
            """
            new_source = []
            count = 0
            if shuffle:
                random.shuffle(source)
            for s in source:
                if count % int(1 / percent) == 0:
                    new_source.append(s)
                count += 1
            source = new_source
            """

        keys = source[0].keys()
        data = {key: [elem[key] for elem in source] for key in keys}
        return cls.from_dict(data)

    @classmethod
    def load(cls, dataset_path: str, percent: float = 1.0, shuffle: bool = False):
        print_pass(f"[Dataset] Loading dataset at {dataset_path}")
        return cls.__generate_dataset(dataset_path, percent, shuffle)

    @classmethod
    def _convert_to_chat(self, example) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["response"]},
        ]

    def as_chat(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[Dataset, PreTrainedTokenizerBase]:
        """For Instruct Model using Chat template"""
        self.idx = 0

        def process(example):
            # covert to think_list
            think_list = example["think"].split(". ")
            think_list = [
                think + "." if think[-1] != "." else think for think in think_list
            ]
            think_copy_list = example["think"].split(". ")
            think_copy_list = [
                think + "." if think[-1] != "." else think for think in think_copy_list
            ]
            
            # the last slot is replaced by the action-grounding sentence
            think_list[-1] = f"Thus, the logical next action should be {example['action']}."
            think = " ".join(think_list)

            example = self.template(example['instruction'], example['state'], think, example['action'], example['history'], None)
            
            input_text = tokenizer.apply_chat_template(
                self._convert_to_chat(example),
                tokenize=False,
            )
            patterns = [
                # qwen2.5-instruct
                "<|im_start|>assistant\n",
                # llama3.2-instruct
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ]
            for pattern in patterns:
                input_split = input_text.split(pattern)
                if len(input_split) == 2:
                    input_query, input_response = input_split
                    input_query = input_query + pattern
                    break


            output = {
                **example,
                "think_list": think_list,
                "think_copy_list": think_copy_list,
                "text": input_text,
                # "query_text": input_query,
                # "response_text": input_response,
                "query_ids": tokenizer.encode(
                    input_query,
                    return_tensors="np",
                ),
                "response_ids": tokenizer.encode(
                    input_response,
                    return_tensors="np",  # np, pt
                ),
                "index": self.idx,
            }
            self.idx += 1
            return output

        return self.map(process)

    @classmethod
    def _convert_to_completion(self, example) -> str:
        return f"{VirtualHomeDataset.additional_special_tokens['query']} {example['query']}\n{VirtualHomeDataset.additional_special_tokens['response']} {example['response']}"

    def as_completion(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[Dataset, PreTrainedTokenizerBase]:
        """For Text Generation Model"""

        self.idx = 0

        def process(example):
            # covert to think_list
            think_list = example["think"].split(". ")
            think_list = [
                think + "." if think[-1] != "." else think for think in think_list
            ]
            think_copy_list = example["think"].split(". ")
            think_copy_list = [
                think + "." if think[-1] != "." else think for think in think_copy_list
            ]

            # the last slot is replaced by the action-grounding sentence
            think_list[-1] = f"Thus, the logical next action should be {example['action']}."
            think = " ".join(think_list)

            example = self.template(example['instruction'], example['state'], think, example['action'], example['history'], None)
            input_text = self._convert_to_completion(example) + f"\n" #{tokenizer.eos_token}"
            input_query, input_response = input_text.split(
                VirtualHomeDataset.additional_special_tokens["response"]
            )
            input_query = (
                input_query + VirtualHomeDataset.additional_special_tokens["response"]
            )
            
            output = {
                **example,
                "text": input_text,
                "think_list": think_list,
                #"think_copy_list": think_copy_list,
                # "query_text": input_query,
                # "response_text": input_response,
                "query_ids": tokenizer.encode(
                    input_query,
                    return_tensors="np",
                ),
                "response_ids": tokenizer.encode(
                    input_response,
                    return_tensors="np",
                ),
                "index": self.idx,
            }
            self.idx += 1
            return output

        return self.map(process)


class AlfredDataset(VirtualHomeDataset):
    
    def set_template(self, dataset_type: _Type_Template, example: str = None):
        self.template = PromptTemplate("alfred", dataset_type, example)

    @classmethod
    def __generate_dataset(cls, dataset_path: str, percent: float, shuffle: bool):
        source = []
        idx = 0
        with open(dataset_path) as f:
            for line in f:
                try:
                    json_line = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"malformed JSON line in {dataset_path}: {line[:120]!r}") from e
                json_line["index"] = idx
                idx += 1
                source.append(json_line)

        if percent != 1.0:
            print_warn(f"Original Dataset: {len(source)}")
            total_traj, traj = [], []
            for s in source:
                traj.append(s)
                if s["reward"] == "1.0":
                    total_traj.append(traj)
                    traj = []

            print(f"Total Traj: {len(total_traj)}")

            new_source = []
            chunk_size = int(1 / percent)

            print(f"Chunk Size: {chunk_size}")
            for i in range(0, len(total_traj), chunk_size):
                chunk = total_traj[i:i + chunk_size]
                if chunk: # if not empty list
                    selected = random.choice(chunk)
                    new_source.extend(selected)
            print_check(f"New Dataset: {len(new_source)}")
            source = new_source

            """
            new_source = []
            count = 0
            if shuffle:
                random.shuffle(source)
            for s in source:
                if count % int(1 / percent) == 0:
                    new_source.append(s)
                count += 1
            source = new_source
            """
        keys = source[0].keys()
        data = {key: [elem[key] for elem in source] for key in keys}
        return cls.from_dict(data)

    @classmethod
    def load(cls, dataset_path: str, percent: float = 1.0, shuffle: bool = False):
        print_pass(f"[Dataset] Loading dataset at {dataset_path}")
        return cls.__generate_dataset(dataset_path, percent, shuffle)

    def as_chat(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[Dataset, PreTrainedTokenizerBase]:
        """For Instruct Model using Chat template"""
        self.idx = 0

        def process(example):
            # covert to think_list
            think_list = example["think"].split(". ")
            think_list = [
                think + "." if think[-1] != "." else think for think in think_list
            ]
            think_copy_list = example["think"].split(". ")
            think_copy_list = [
                think + "." if think[-1] != "." else think for think in think_copy_list
            ]
            
            # the last slot is replaced by the action-grounding sentence
            think_list[-1] = f"Thus, the logical next action should be {example['action']}."
            think = " ".join(think_list)

            example = self.template(example['instruction'], example['state'], think, example['action'], example['history'], None)
            
            input_text = tokenizer.apply_chat_template(
                self._convert_to_chat(example),
                tokenize=False,
            )
            patterns = [
                # qwen2.5-instruct
                "<|im_start|>assistant\n",
                # llama3.2-instruct
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ]
            for pattern in patterns:
                input_split = input_text.split(pattern)
                if len(input_split) == 2:
                    input_query, input_response = input_split
                    input_query = input_query + pattern
                    break


            output = {
                **example,
                "think": think,
                "think_list": think_list,
                "think_copy_list": think_copy_list,
                "text": input_text,
                # "query_text": input_query,
                # "response_text": input_response,
                "query_ids": tokenizer.encode(
                    input_query,
                    return_tensors="np",
                ),
                "response_ids": tokenizer.encode(
                    input_response,
                    return_tensors="np",  # np, pt
                ),
                "index": self.idx,
            }
            self.idx += 1
            return output

        return self.map(process)

    @classmethod
    def _convert_to_completion(self, example) -> str:
        return f"{VirtualHomeDataset.additional_special_tokens['query']} {example['query']}\n{VirtualHomeDataset.additional_special_tokens['response']} {example['response']}"

    def as_completion(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[Dataset, PreTrainedTokenizerBase]:
        """For Text Generation Model"""

        self.idx = 0

        def process(example):
            # covert to think_list
            think_list = example["think"].split(". ")
            think_list = [
                think + "." if think[-1] != "." else think for think in think_list
            ]
            think_copy_list = example["think"].split(". ")
            think_copy_list = [
                think + "." if think[-1] != "." else think for think in think_copy_list
            ]

            # the last slot is replaced by the action-grounding sentence
            think_list[-1] = f"Thus, the logical next action should be {example['action']}."
            think = " ".join(think_list)

            example = self.template(example['instruction'], example['state'], think, example['action'], example['history'], None)
            input_text = self._convert_to_completion(example) + f"\n" #{tokenizer.eos_token}"
            input_query, input_response = input_text.split(
                VirtualHomeDataset.additional_special_tokens["response"]
            )
            input_query = (
                input_query + VirtualHomeDataset.additional_special_tokens["response"]
            )
            
            output = {
                **example,
                "think": think,
                "text": input_text,
                "think_list": think_list,
                #"think_copy_list": think_copy_list,
                # "query_text": input_query,
                # "response_text": input_response,
                "query_ids": tokenizer.encode(
                    input_query,
                    return_tensors="np",
                ),
                "response_ids": tokenizer.encode(
                    input_response,
                    return_tensors="np",
                ),
                "index": self.idx,
            }
            self.idx += 1
            return output

        return self.map(process)


def embedding_fns(tokenizer, model, queries):
    input_ids = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
    embeddings = model(
        input_ids=input_ids["input_ids"], attention_mask=input_ids["attention_mask"]
    ).pooler_output
    return embeddings


# # # Dataset Generator # # #
class VirtualHomeDatasetGenerator:
    def __init__(
        self,
        tokenizer,
        model,
        dataset_dir: str,
        num_topk_edge: int = 12,
        max_think_token: int = 150,
        gamma: float = 0.99,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.env_name = "virtualhome"
        self.env_prompt = PromptTemplate.load_env_prompt(self.env_name)

        self.dataset_dir = dataset_dir
        self.num_topk_edge = num_topk_edge
        self.gamma = gamma
        self.data, self.total_steps = self._load_data()

        template = "Instruction: {instruction}\nCurrent State: {state}\nNext Optimal Action: {action}\nPrevious Actions: {history}\nYou will analyze the instruction and current state, providing a concise five-sentence response that only incorporates:\n1. Physical location and status of the character\n2. Physical location and status of observations related to the instruction and without mentioning any other non-related observations\n3. Summarization of key previous action histories if previous actions are available\n4. Break down the remaining plan to complete the instruction if remaining plans are required\n5. Reasoning for what should do next considering the next optimal action. Note that you can only move to another room if the rooms are adjacent, otherwise you may need to explore.\nEach sentence serves a specific purpose while maintaining clarity and avoiding mention of specific actions and the next optimal action."

        self.llm = OpenAILLM(
            "gpt-4o-mini",
            temperature=0.2,
            top_p=0.1,
            max_tokens=max_think_token,
            template=template,
        )

        self.llm_correct = OpenAILLM(
            "gpt-4o-mini",
            temperature=0.7,
            top_p=0.8,
            max_tokens=max_think_token,
            template=template,
        )

        # Embedding tokenizer & model
        self.kg_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        self.kg_model = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

    def _load_data(self) -> List[List[Dict]]:
        data = []
        total_steps = 0
        # Traverse task directories
        for task_dir in os.listdir(self.dataset_dir):
            task_path = os.path.join(self.dataset_dir, task_dir)
            if not os.path.isdir(task_path):
                continue

            # Traverse each episode file in the task directory
            for episode_file in os.listdir(task_path):
                if episode_file.endswith(".jsonl"):
                    episode_path = os.path.join(task_path, episode_file)
                    with open(episode_path, "r") as f:
                        trajectory = []
                        for line in f:
                            entry = json.loads(line)
                            trajectory.append(entry)
                        data.append(trajectory)
                        total_steps += len(trajectory)
        return data, total_steps

    def generate_dataset(
        self,
    ) -> Generator[tuple[str, str], None, None]:
        for traj in self.data:
            histories = []
            for step in range(len(traj)):
                # kg retrieval
                kg = KG(traj[step]["position_graph"])
                kg.add(traj[step]["visible_graph"], step, use_refinement=True)
                kg.add(traj[step]["agent_graph"], step, use_refinement=True)
                instruction = traj[step]["instruction"]
                state = kg.retrieve(  # input of instruction should be list
                    [instruction],
                    embedding_fns=partial(
                        embedding_fns, self.kg_tokenizer, self.kg_model
                    ),
                    num_edges=self.num_topk_edge,
                    return_type="str",
                )
                state_full = kg.retrieve(  # X2 state
                    [instruction],
                    embedding_fns=partial(
                        embedding_fns, self.kg_tokenizer, self.kg_model
                    ),
                    num_edges=self.num_topk_edge * 2,
                    return_type="str",
                )
                action = traj[step]["action"]
                history = (
                    ", ".join(histories) if len(histories) > 0 else "No action history."
                )
                history = history.replace("'", "")
                histories.append(str((f"step {step+1}", action)))
                futures = [
                    str((f"step {step+1+i}", f"{traj[step+i]['action']}"))
                    for i in range(len(traj) - step)
                ]
                future = ", ".join(futures[1:])
                future = future.replace("'", "")
                plan = [traj[step + i]["action"] for i in range(len(traj) - step)]
                plan = ", ".join(plan)
                reward = math.pow(self.gamma, len(traj) - (step + 1)) * 1.0

                # ========================= #
                # think annotation with llm #
                # ========================= #
                llm_query = {
                    "env_prompt": self.env_prompt,
                    "instruction": instruction,
                    "state": state,
                    "action": (
                        f"(step {step+1}, {action})"
                        if "walk" not in action
                        else f"(step {step+1}, walk to {action.split(' ')[-1]})"
                    ),
                    "history": history,
                    "future": future,
                }

                if "put" in action:
                    topk = 10
                else:
                    topk = 5
                print("\n\n\n")
                print("=" * 20)

                # Generate N thinks
                count, think_list = 0, []
                while True:
                    done, think = self.generate_think(
                        count,
                        llm_query,
                        instruction,
                        state,
                        action,
                        history,
                        future,
                        topk=topk,
                    )
                    count += 1
                    if done:
                        think_list.append(think)
                    if len(think_list) == 5 or count == 30:
                        break

                if len(think_list) == 0:
                    print_error("Nothing available ...")
                    continue

                # Pick one of the think
                think = self.pick_think(
                    instruction, state, action, history, think_list, plan
                )
                # ========================= #
                yield instruction, state, state_full, think, think, action, history, future, reward

    def check_think(self, think):
        think_list = think.split(". ")
        if len(think_list) == 5:
            return True
        else:
            return False

    def generate_think(
        self, count, llm_query, instruction, state, action, history, future, topk=5,
    ):
        action_format = PromptTemplate.load_env_action_format(self.env_name)
        object_list = PromptTemplate.get_object_list(state)

        skill_list = []
        for act in action_format:
            if "noun2" in act:
                for obj1 in object_list:
                    for obj2 in object_list:
                        skill_list.append(act.format(noun1=obj1, noun2=obj2))
            else:
                for obj in object_list:
                    skill_list.append(act.format(noun1=obj))

        while True:
            if count == 0:
                think = self.llm.invoke(llm_query).strip()
            else:
                think = self.llm_correct.invoke(llm_query).strip()
            think = think.replace("\n", " ")
            think = think.replace('"', "'")
            if self.env_name == "virtualhome" and (
                "current optimal" in think or "Current Optimal" in think
            ):
                continue
            if self.check_think(think):
                break

        later_probs = []
        plan_template = PromptTemplate("virtualhome", "cd-action-think")
        prompt = plan_template(
            instruction=instruction, state=state, history=history, think=think
        )["query"]

        for skill in skill_list:
            query = [{"role": "user", "content": prompt}]
            query_id = self.tokenizer.apply_chat_template(
                query, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            input_length = len(query_id[0])

            query = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": skill},
            ]
            query_id = self.tokenizer.apply_chat_template(
                query, tokenize=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                model_output = self.model(query_id)
            logits = model_output.logits[0, input_length - 1 : -1, :]
            tokens = query_id[0, input_length:]

            later_prob = 1.0
            for i, logit in enumerate(logits):
                logit = torch.nn.functional.softmax(logit, dim=-1)
                later_prob *= logit[tokens[i].item()]
            later_probs.append(later_prob.item())  # /len(logits))

        sort_probs = np.sort(later_probs)[::-1][:topk]
        sort_probs_index = np.argsort(later_probs)[::-1][:topk]
        top_k_skill = [skill_list[idx] for idx in sort_probs_index]
        print_pass(top_k_skill)
        print(sort_probs)
        print(PromptTemplate.preprocess(state))
        print_warn(action)
        print(think)
        print_check(action in top_k_skill)
        return action in top_k_skill, think

    def pick_think(self, instruction, state, action, history, think_list, plan):
        plan_template = PromptTemplate(self.env_name, "cd-plan")
        probs = []
        for think in think_list:
            prompt = plan_template(
                instruction=instruction, state=state, history=history, think=think
            )["query"]
            query = [{"role": "user", "content": prompt}]
            query_id = self.tokenizer.apply_chat_template(
                query, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            input_length = len(query_id[0])

            query = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan},
            ]
            query_id = self.tokenizer.apply_chat_template(
                query, tokenize=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                model_output = self.model(query_id)

            logits = model_output.logits[0, input_length - 1 : -1, :]
            tokens = query_id[0, input_length:]
            prob = 1.0
            for i, logit in enumerate(logits):
                logit = torch.nn.functional.softmax(logit, dim=-1)
                prob *= logit[tokens[i].item()]
            probs.append(prob.item())

            print(think)
            print(prob)

        index = np.argmax(probs)
        print_pass(index)
        return think_list[index]

    def save_dataset(
        self,
        save_path: str = None,
        filename: str = "full_dataset.jsonl",
    ) -> None:
        if save_path is None:
            save_path = self.dataset_dir
        save_path = os.path.join(save_path, filename)

        # generate dataset
        dataset = self.generate_dataset()

        # save full dataset
        with open(save_path, "w") as f:
            for (
                instruction,
                state,
                state_full,
                think1,
                think2,
                action,
                history,
                future,
                reward,
            ) in tqdm.tqdm(dataset):
                f.write(
                    f'{{"instruction": "{instruction}", "state": "{state}", "state_full": "{state_full}","think": "{think1}", "think_copy": "{think2}", "action": "{action}", "history": "{history}", "future": "{future}", "reward": "{reward}"}}\n'
                )

    def __len__(self):
        return len(self.data)


class AlfredDatasetGenerator(VirtualHomeDatasetGenerator):
    def __init__(
        self,
        tokenizer,
        model,
        dataset_dir: str,
        num_topk_edge: int = 12,
        max_think_token: int = 100,
        gamma: float = 0.99,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.env_name = "alfred"
        self.env_prompt = PromptTemplate.load_env_prompt(self.env_name)

        self.dataset_dir = dataset_dir
        self.num_topk_edge = num_topk_edge
        self.gamma = gamma
        try:
            self.data, self.total_steps = self._load_data()
        except Exception as e:
            print_error(f"[[Could not load the raw ALFRED episodes from {dataset_dir}]]: {e!r}")
            raise

        template = "Instruction: {instruction}\nCurrent State: {state}\nNext Optimal Action: {action}\nPrevious Actions: {history}\nYou will analyze the instruction and current state, providing a concise five-sentence response that only incorporates:\n1. Physical location and status of the character\n2. Physical location and status of observations related to the instruction and without mentioning any other non-related observations\n3. Summarization of key previous action histories if previous actions are available\n4. Break down the remaining plan to complete the instruction if remaining plans are required\n5. Reasoning for what should do next considering the next optimal action. Note that before interacting with the object the agent should go to the object.\nEach sentence serves a specific purpose while maintaining clarity. If there are multiple same objets you must notify the number of the object."

        self.llm = OpenAILLM(
            "gpt-4o-mini",
            temperature=0.2,
            top_p=0.1,
            max_tokens=max_think_token,
            template=template,
        )

        self.llm_correct = OpenAILLM(
            "gpt-4o-mini",
            temperature=0.5,
            top_p=0.6,
            max_tokens=max_think_token,
            template=template,
        )

        # Embedding tokenizer & model
        self.kg_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        self.kg_model = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

    def generate_dataset(
        self,
    ) -> Generator[tuple[str, str], None, None]:
        for traj in self.data:
            histories = []
            for step in range(len(traj)):
                instruction = traj[step]["instruction"]
                state = traj[step]["state"]
                state_full = traj[step]["state_full"]
                action = traj[step]["action"]
                history = (
                    ", ".join(histories) if len(histories) > 0 else "No action history."
                )
                history = history.replace("'", "")
                histories.append(str((f"step {step+1}", action)))
                futures = [
                    str((f"step {step+1+i}", f"{traj[step+i]['action']}"))
                    for i in range(len(traj) - step)
                ]
                future = ", ".join(futures[1:])
                future = future.replace("'", "")
                plan = [traj[step + i]["action"] for i in range(len(traj) - step)]
                plan = ", ".join(plan)
                reward = math.pow(self.gamma, len(traj) - (step + 1)) * 1.0

                # ========================= #
                # think annotation with llm #
                # ========================= #
                if (
                    "heat" in action
                    or "cool" in action
                    or "take" in action
                    or "put" in action
                    or "clean" in action
                ):
                    topk = 8
                else:
                    topk = 4

                llm_query = {
                    "env_prompt": self.env_prompt,
                    "instruction": instruction,
                    "state": state,
                    "action": f"(step {step+1}, {action})",
                    "history": history,
                    "future": future,
                }

                print("\n\n\n")
                print("=" * 20)
                # Generate N thinks
                count, think_list = 0, []
                while True:
                    done, think = self.generate_think(
                        count,
                        llm_query,
                        instruction,
                        state,
                        action,
                        history,
                        future,
                        topk=topk,
                    )
                    count += 1
                    if done:
                        think_list.append(think)
                    if len(think_list) == 3 or count == 20:
                        break

                if len(think_list) == 0:
                    print_error("Nothing available ...")
                    continue

                # Pick one of the think
                think = self.pick_think(
                    instruction, state, action, history, think_list, plan
                )
                # ========================= #
                yield instruction, state, state_full, think, think, action, history, future, reward
