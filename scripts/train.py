import os
import json
import torch
import yaml
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel, get_peft_model

from embodied_cd.common import dataset_utils
from embodied_cd.common.print_utils import *
from embodied_cd.trl.models import * 

_PI_MODEL_ALIASES = {
    'gpt2': GPT2LMHeadModel,
    'Qwen2.5': AutoModelForCausalLM,
    'Llama': AutoModelForCausalLM, 
}

_VF_MODEL_ALIASES = {
    'gpt2': GPT2HeadWithValueModel,
    'Qwen2.5': Qwen2HeadWithValueModel,
    'Llama': LlamaHeadWithValueModel,
}


# fraction of full_dataset.jsonl that the default subset (part_1.0) holds
BASE_FRACTION = {"virtualhome": 0.125, "alfred": 0.25}


def prepare_dataset(env_name, model_name, dataset_dir, tokenizer, dataset_type="default", few_shot_example=None, dataset_percent=1.0, dataset_scale=1.0, shuffle=False):
    # load dataset
    Loader = dataset_utils.VirtualHomeDataset if env_name == "virtualhome" else dataset_utils.AlfredDataset
    if env_name in ("virtualhome", "alfred"):
        # part_1.0 is the default training subset; part_2.0 / part_4.0 are 2x / 4x as large
        dataset = Loader.load(dataset_dir+f'/part_{float(dataset_scale)}_dataset.jsonl', dataset_percent, shuffle)
        print_warn(f"[dataset] Loading total {len(dataset)} dataset")
        dataset.set_template(dataset_type, few_shot_example)
    else:
        raise NotImplementedError

    if 'instruct' in model_name or 'Instruct' in model_name:
        # if instruct model
        print_warn("[Dataset] Using Chat Template")
        dataset = dataset.as_chat(tokenizer)
        def collate_fn(examples):
            texts = [example["text"] for example in examples]
            batch = tokenizer(texts, return_tensors="pt", padding=True)
            labels = batch['input_ids'].clone()
            for idx, label in enumerate(labels):
                # we mask the query part in the loss computation
                label[:len(examples[idx]['query_ids'][0])] = -100
            batch['labels'] = labels
            return batch
        collator = collate_fn
    else: 
        # else completion
        print_warn("[Dataset] Using Completion Template")
        dataset = dataset.as_completion(tokenizer)
        collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, 
            mlm=False,
            instruction_template=dataset_utils.VirtualHomeDataset.additional_special_tokens["query"],
            response_template=dataset_utils.VirtualHomeDataset.additional_special_tokens["response"])
    return dataset, collator


def prepare_model(model_name, vf=False, init=True, mlm=False, config=dict()):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if init:
        if tokenizer.pad_token is None:
            print_warn("[Tokenizer] has no pad token")
            tokenizer.pad_token = tokenizer.eos_token
        if mlm and (tokenizer.mask_token is None):
            print_error("[Tokenizer] Add mask token")
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            print(tokenizer.mask_token_id)
        tokenizer.padding_side = 'right'

    model_aliases = _PI_MODEL_ALIASES if not vf else _VF_MODEL_ALIASES

    # load model
    for key, cls in model_aliases.items():
        if key in model_name:
            print_warn(f"[Loading: Model] from {model_name}")
            model = cls.from_pretrained(model_name, **config, device_map='auto')
            if not init and vf:
                print_warn(f"[Loading: ValueHead] from vl_head.pth")
                model.vl_head.load_state_dict(torch.load(model_name + '/vl_head.pth'))
            if init and len(tokenizer) > model.vocab_size:
                print_warn(f"[Loading: Embedding] Resize token embedding size from {model.vocab_size} to {len(tokenizer)}")
                # resizing the embedding layer with tokenizer length, considering additional tokens
                model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def add_adapter(model, name="default"):    
    # config lora
    lora_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16, # lora rank
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["v_proj", "q_proj"],
    )

    # add adapter
    print_warn("[Model] Adding Lora Adapter")
    if not isinstance(model, PeftModel):
        model = get_peft_model(model, lora_config, name)
    else:
        model.add_adapter(name, lora_config)
    model.print_trainable_parameters()
    return model


def load_evaluation_configs(env_name, _type, _set, _eval): # behavior-il / environment-il
    config_path = f"./configs/{env_name}/{_type}_{_eval}.py"

    config = {}
    with open(config_path, 'r') as f:
        code = f.read()
    exec(code, config)
    
    # loading pending steps
    if "seen" in _eval:
        config_path = f"./datasets/{env_name}/{_type}_seen/{_set}/expert_traj.jsonl"
    if "unseen" in _eval:
        config_path = f"./datasets/{env_name}/{_type}_unseen/{_set}/expert_traj.jsonl"
    with open(config_path, 'r') as f:
        for line in f:
            ps_line = json.loads(line)
            break

    if env_name == "virtualhome": 
        return list(config['rooms'][_set].keys()), list(config['tasks'][_set].keys()), config['available_tasks'], ps_line
    if env_name == "alfred":
        return config['rooms'][_set], config['tasks'][_set], None, ps_line


def save_params(output_dir, params):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'params.yml'), 'w+') as f:
        yaml.dump(params, f, default_flow_style=False)

