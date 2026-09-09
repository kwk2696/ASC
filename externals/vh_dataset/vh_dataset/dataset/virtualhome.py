from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from typing import List, Dict, Set, Literal, Generator
from datasets import Dataset, DatasetDict
from vh_dataset.dataset.jsonl import JsonlDataset
from vh_dataset.dataset.knowledge_graph import KG

import os
import json

_Phase = Literal["behavior-cloning", "belief-state-inference"]
ALLOWED_PHASES: Set[str] = {"behavior-cloning", "belief-state-inference"}


def prompt_template(knowledge_graph, task):
    prompts = f"Observation: {knowledge_graph}\n"
    prompts += f"Instruction: {task}"
    return prompts

"""
"""

def make_embedding_fn():
    kg_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    kg_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    def embedding_fn(sentences):
        input_ids = kg_tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        kg_triples = kg_model(
            input_ids=input_ids["input_ids"], attention_mask=input_ids["attention_mask"]
        ).pooler_output
        return kg_triples
    return embedding_fn


class VirtualHomeDataset(JsonlDataset):
    @classmethod
    def _convert_to_chat(cls, example) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["result"]},
        ]


class VirtualHomeDatasetGenerator:
    def __init__(
        self, trajectory_path: str, phase: _Phase = "behavior-cloning"
    ) -> None:
        """

        :param dataset_path:
        :param phase:
        """
        if phase not in ALLOWED_PHASES:
            raise ValueError(
                f"Invalid phase provided. Allowed phases are: {ALLOWED_PHASES}"
            )

        self.dataset_path = trajectory_path
        self.data = self._load_data()

    def _load_data(self) -> List[List[Dict]]:
        data = []
        # Traverse task directories
        for task_dir in os.listdir(self.dataset_path):
            task_path = os.path.join(self.dataset_path, task_dir)
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
        return data

    def generate_dataset(
        self,
        dataset_type: _Phase = "behavior-cloning",
    ) -> Generator[tuple[str, str], None, None]:
        """Convert loaded data to a Hugging Face Dataset object."""

        match dataset_type:
            case "behavior-cloning":
                for traj in self.data:
                    kg = KG(traj[0]["position_graph"])
                    for step in range(len(traj)):
                        kg.add(traj[step]["visible_graph"], step, use_refinement=True)
                        kg.add(traj[step]["agent_graph"], step, use_refinement=True)
                        action = traj[step]["action"]
                        prompt = prompt_template(
                            kg.retrieve(
                                traj[step]["instruction"],
                                embedding_fns=embedding_fns,
                                num_edges=12,
                            ),
                            traj[step]["instruction"],
                        )
                        yield prompt, action
