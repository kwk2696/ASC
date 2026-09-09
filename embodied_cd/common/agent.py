from abc import ABC, abstractmethod
from typing import List


from embodied_cd.common.dataset_utils import VirtualHomeDataset


class BaseAgent(ABC):
    default_gen_params = {
        "do_sample": False,
        "top_k": None,
        "top_p": None,
        "temperature": None,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }

    @abstractmethod
    def reset(self, task, goal):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

    def _convert_to_chat(self, user: List[str], assistant: List[str]):
        chat_query = []
        for i in range(len(assistant)):
            chat_query.append({"role": "user", "content": user[i]})
            chat_query.append({"role": "assistant", "content": assistant[i]})
        chat_query.append({"role": "user", "content": user[-1]})
        return chat_query

    def _convert_to_completion(self, user: str):
        return f"{VirtualHomeDataset.additional_special_tokens['query']} {user}\n{VirtualHomeDataset.additional_special_tokens['response']}"
