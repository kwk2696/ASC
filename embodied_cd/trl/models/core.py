from typing import Literal

import numpy as np
import torch
from torch import nn

from embodied_cd.trl.models.type_aliases import GenerationOutput
from embodied_cd.common.env_utils import build_skill_list

_Type_Decoding = Literal["greedy", "beam-action"]


def generation(model, tokenizer, query, max_length=80, **gen_params):
    device = model.device
    input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    input_length = input_ids.shape[-1]

    input_ids = model.generate(
        input_ids,
        **gen_params,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_ids = input_ids.squeeze()[input_length:]
    response = tokenizer.decode(input_ids, skip_special_tokens=True)

    return GenerationOutput(
        query=query,
        response=response,
        values=None,
        prob=None,
    )


def greedy_generation(model, tokenizer, query, max_length=80):
    device = model.device
    # tokenizer encoding
    input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    input_length = input_ids.shape[-1]
    prob = 1.0
    log_probs = []
    for i in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

        # smaple
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # argmax
        next_token = torch.argmax(logits, dim=-1)
        prob = prob * probs[0][next_token.item()]
        log_probs.append(torch.log(probs[0][next_token.item()]).item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # sequeeze
    input_ids = input_ids.squeeze()[input_length:]
    # tokenizer decoding
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    return GenerationOutput(
        query=query,
        response=response,
        values=None,
        prob=prob,
        log_probs=log_probs,
    )


def beam_action_generation(
    model,
    tokenizer,
    query,
    action_format,
    object_list,
    can_list=None,
    env_name="virtualhome",
    reduced_skills=True,
):
    skill_list = build_skill_list(action_format, object_list, env_name, reduced_skills)
    if can_list is not None:
        if env_name == "virtualhome":
            can_put_place_list = [
                c for c in can_list if c.startswith("put") or c.startswith("place")
            ]
            skill_put_place_list = [
                c for c in skill_list if c.startswith("put") or c.startswith("place")
            ]

            put_place_list = []
            for skill in skill_put_place_list:
                for can in can_put_place_list:
                    pre, suf = can.split()
                    if skill.startswith(pre) and skill.endswith(suf):
                        put_place_list.append(skill)
                        break

            skill_list = list((set(skill_list) & set(can_list)) | set(put_place_list))

        if env_name == "alfred":
            skill_list = list(set(skill_list) & set(can_list))

    query_ids = (
        tokenizer.encode(query, return_tensors="pt")
        .to(model.device)
        .repeat(len(skill_list), 1)
    )
    query_length = query_ids.shape[-1]

    output = tokenizer(skill_list, return_tensors="pt", padding=True)
    skill_ids = output["input_ids"].to(model.device)
    skill_masks = output["attention_mask"].to(model.device)
    skill_lengths = skill_masks.sum(dim=-1)

    input_ids = torch.concat([query_ids, skill_ids], dim=-1)
    attention_mask = torch.concat(
        [
            torch.ones_like(query_ids),
            skill_masks,
        ],
        dim=-1,
    )

    batch_size = 1
    action_probs = []
    for i in range(0, len(skill_list), batch_size):
        inputs = {
            "input_ids": input_ids[i : i + batch_size],
            "attention_mask": attention_mask[i : i + batch_size],
        }
        model_output = model(**inputs)

        for j in range(len(inputs["input_ids"])):
            logits = model_output.logits[j, query_length - 1 :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            tokens = skill_ids[i + j][: skill_lengths[i + j]]

            prob = 1.0
            for k, token in enumerate(tokens):
                prob *= probs[k][token.item()]
            action_probs.append(prob.item())


    max_index = np.argmax(action_probs)
    return GenerationOutput(
        query=query,
        response=skill_list[max_index],
        values=None,
        prob=action_probs[max_index],
    )

