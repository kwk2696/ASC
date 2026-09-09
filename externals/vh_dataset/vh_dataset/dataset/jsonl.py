import json
from pathlib import Path
from typing import Any, Callable, Generator, cast

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from datasets import Dataset


class JsonlDataset(Dataset):
    @classmethod
    def __generate_dataset(cls, dataset_path: str | Path):
        source = []
        with open(dataset_path) as f:
            for line in f:
                try: 
                    json_line = json.loads(line)
                except:
                    print("Error in loading dataset")
                    print(line)
                    exit() 
                source.append(json_line)
            #source = [json.loads(line) for line in f]
        keys = source[0].keys()
        data = {key: [elem[key] for elem in source] for key in keys}
        return cls.from_dict(data)

    @classmethod
    def load(cls, dataset_path: str | Path):
        return cls.__generate_dataset(dataset_path)

    def as_chat(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        def process(example):
            return {
                **example,
                "text": tokenizer.apply_chat_template(
                    self._convert_to_chat(example),
                    tokenize=False,
                ),
            }

        return self.map(process)

    def _convert_to_chat(self, example) -> list[dict[str, str]]:
        raise NotImplementedError
