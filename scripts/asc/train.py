import wandb
import torch
import json
import numpy as np
from tqdm import tqdm

from transformers import set_seed

from embodied_cd.common import argparse
from embodied_cd.common.print_utils import *
from embodied_cd.common.env_utils import *
from embodied_cd.common.dataset_utils import PromptTemplate
from embodied_cd.trl.asc import ASCTrainer
from embodied_cd.agents.asc import ASCAgent

from scripts.train import *


### ==================================###
#              Training                 #
### ==================================###
def prepare(args, path):
    tokenizer, model = prepare_model(args.pi_model_name, init=True)
    model = add_adapter(model, name="reasoning_policy")
    model = add_adapter(model, name="planning_policy")
    if path is not None:
        model.load_adapter(path + f'/policy/reasoning_policy', adapter_name='reasoning_policy')
        model.load_adapter(path + f'/policy/planning_policy', adapter_name='planning_policy')
    return tokenizer, model

def select_base_stage(args, prev_phases, dataset):
    """Pick which earlier stage's policy acts as the base (prior) policy: the stage
    whose rationales score highest on the current data."""
    if len(prev_phases) == 1:
        return prev_phases[0]
    scores = []
    for prev_phase in prev_phases:
        path = f"./results/ASC/{args.env_name}/{args.continual_type}_{args.continual_seq}/{prev_phase}_{args.pi_model_name.split('/')[-1]}_{args.prev_tag}"
        tokenizer, model = prepare(args, path=path)
        scores.append(ASCTrainer._setup_base_model(args.pi_model_name, tokenizer, model, dataset, args.max_think_token))
        del tokenizer, model
    print_check(f"[base stage] candidates {prev_phases} -> rationale scores {scores}")
    return prev_phases[int(np.argmax(scores))]


def train_policy(args):
    # wandb log on/off
    if args.wandb_off:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="ASC")
        wandb.run.name = f"policy_{args.env_name}_{args.continual_type}_{args.continual_set}_{args.tag}"

    tokenizer = AutoTokenizer.from_pretrained(args.pi_model_name)
    # prepare dataset
    dataset, collator = prepare_dataset(
        args.env_name, 
        args.pi_model_name, 
        args.dataset_dir, 
        tokenizer, 
        dataset_type="cd-think",
        dataset_scale=args.dataset_scale,
        few_shot_example=None,
    )
    del tokenizer

    # load configs and get continual phase
    config = {}
    with open(f"./configs/{args.env_name}/{args.continual_type}_train.py", "r") as f:
        code = f.read()
    exec(code, config)
    phase = config["sequences"][args.continual_seq].index(args.continual_set)
    
    if (phase == 0) or (args.seqft == 0):
        # first stage of the sequence (or no sequential fine-tuning): fresh LoRA, no base policy
        base_tokenizer, base_model = None, None
        tokenizer, model = prepare(args, path=None)
        prev_phase = None
    else:
        prev_phases = config["sequences"][args.continual_seq][:phase]
        prev_phase = select_base_stage(args, prev_phases, dataset)
        path = f"./results/ASC/{args.env_name}/{args.continual_type}_{args.continual_seq}/{prev_phase}_{args.pi_model_name.split('/')[-1]}_{args.prev_tag}"
        print_pass(f"[Loading Base Model] at {prev_phase}: {path}")
        base_tokenizer, base_model = prepare(args, path=path)
        print_error("Initialize with base reasoning policy")
        tokenizer, model = prepare(args, path=path)

    # set output_dir
    output_dir = args.output_dir + "/policy"

    # set configs
    params = {
        "total_epochs": 20,
        "inner_epochs": 4,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "max_think_token": args.max_think_token,
        "total_think": args.total_think,
        "base_continual_set": prev_phase, # last continual phase
        "base_continual_tag": args.prev_tag,
        "dataset_scale": args.dataset_scale,
        "seed": args.seed,
    }
    print(params)

    # create trainer
    trainer = ASCTrainer(
        args.env_name,
        args.pi_model_name,
        base_tokenizer,
        base_model,
        tokenizer,
        model,
        dataset,
        output_dir,
        **params,
    )

    # save params
    save_params(output_dir, trainer.params)

    print_warn("[ASC] Policy Training Start!")
    trainer.train()

    # save model
    trainer.save_pretrained(output_dir)

### ==================================###
#           Evaluation & Test           #
### ==================================###
def thresholds_name(args):
    """Calibration file written by --test and read by --evaluate; one per rationale length."""
    return "thresholds.json" if args.total_think == 5 else f"thresholds_think{args.total_think}.json"


def test(args):
    base_tokenizer, base_model = None, None
    # prepare model
    tokenizer, model = prepare(args, path=args.output_dir)
    
    # prepare dataset
    dataset, collator = prepare_dataset(
        args.env_name, 
        args.pi_model_name, 
        args.dataset_dir, 
        tokenizer, 
        dataset_type="cd-think",
        dataset_scale=args.dataset_scale,
        few_shot_example=None,
    )

    # prepare agent
    agent = ASCAgent(
        args.pi_model_name,
        base_model,
        base_tokenizer,
        model, 
        tokenizer, 
        plan_model=None,
        plan_tokenizer=None,
        env_name=args.env_name,
        cl_type=args.continual_type.split('_')[0],
        max_think_token=args.max_think_token, 
        total_think=args.total_think,
        decoding_strategy="greedy",
    )

    thresholds = [[] for _ in range(args.total_think)]
    for idx, data in enumerate(tqdm(dataset, desc="threshold")):
        instruction, state, action, history, think = data["instruction"], data["state"], data["action"], data["history"], data["think"]
        
        state = PromptTemplate.preprocess(state)
        with torch.no_grad():
            _, think_list = agent.get_think(agent.model, agent.tokenizer, instruction, state, history, sample=False)
            for j in range(len(think_list)):
                think = " ".join(think_list[:j+1])
                action_pred, prob = agent.get_action(instruction, state, history, think)
                if action_pred == action:
                    thresholds[j].append(prob)

    average, deviation = [], []
    for j, threshold in enumerate(thresholds):
        if not threshold:
            print_warn(f"[test] slot {j}: no correct prediction, threshold is undefined (nan)")
        average.append(np.mean(threshold) if threshold else float("nan"))
        deviation.append(np.std(threshold))

    data = {
        "average": average,
        "deviation": deviation,
        "raw": thresholds,
    }
    with open(args.output_dir + f'/{thresholds_name(args)}', 'w') as f:
        json.dump(data, f)


def evaluate(args):
    # prepare base model
    base_tokenizer, base_model = None, None

    # prepare model
    tokenizer, model = prepare(args, path=args.output_dir)

    # read threshold for test-time control
    test_time_thresh = None
    if args.test_time_control: 
        with open(args.output_dir + f'/{thresholds_name(args)}', 'r') as f:
            test_time_thresh = json.load(f)
        if any(np.isnan(v) for v in test_time_thresh["average"]):
            raise ValueError(f"{thresholds_name(args)} has an undefined (nan) slot; re-run --test 1 on more data before --test_time_control")

    # prepare agent
    agent = ASCAgent(
        args.pi_model_name,
        base_model,
        base_tokenizer,
        model, 
        tokenizer, 
        plan_model=None,
        plan_tokenizer=None,
        env_name=args.env_name,
        cl_type=args.continual_type.split('_')[0],
        max_think_token=args.max_think_token, 
        total_think=args.total_think,
        decoding_strategy="greedy",
        test_time_thresh=test_time_thresh,
        test_time_level=args.test_time_level,
    )
    
    if args.continual_eval_set is None:
        flag = False
        args.continual_eval_set = args.continual_set
    else:
        flag = True

    # make env
    split_path = f"./externals/cl-alfred/embodied_split/{args.continual_type.split('_')[0]}_il/embodied_data_disjoint_rand1_cls1_task{int(args.continual_eval_set.split('_')[-1])-1}.json" # for alfred
    env = make_env(args.env_name, args.num_topk_edge, args.env_ip, args.seed, args.env_port, split_path)
    
    if args.evaluate == "seen" or args.evaluate == "both":
        agent.reset_stats()
        # Evaluate on Train Environment
        eval_tag = "seen"
        print_error(f"Eval Tag {eval_tag}")
        env_list, task_list, available_tasks, ps_line = load_evaluation_configs(
            args.env_name, args.continual_type, args.continual_eval_set, eval_tag)
        
        print_warn(f"Evaluating in Seen Set")
        if not flag:
            save_path = args.output_dir + '/results/evaluation_seen'
        else:
            save_path = args.output_dir + f'/results/evaluation_seen_{args.continual_eval_set}'
        if args.test_time_control:
            save_path = save_path + f'_control_{args.test_time_level}'
        if args.total_think != 5:
            save_path = save_path + f'_think{args.total_think}'
        save_path += '.txt'
        evaluation_pipe(
            env, 
            agent, 
            env_list, 
            task_list, 
            available_tasks,
            ps_line,
            total_episode=args.eval_episode, 
            save_path=save_path
        )
    if args.evaluate == "unseen" or args.evaluate == "both": 
        agent.reset_stats()
        # Evaluate on Test Environment
        eval_tag = "unseen"
        print_error(f"Eval Tag {eval_tag}")
        env_list, task_list, available_tasks, ps_line = load_evaluation_configs(
            args.env_name, args.continual_type, args.continual_eval_set, eval_tag)
        
        if not flag:
            save_path = args.output_dir + '/results/evaluation_unseen'
        else:
            save_path = args.output_dir + f'/results/evaluation_unseen_{args.continual_eval_set}'
        if args.test_time_control:
            save_path = save_path + f'_control_{args.test_time_level}'
        if args.total_think != 5:
            save_path = save_path + f'_think{args.total_think}'
        save_path += '.txt'
        evaluation_pipe(
            env, 
            agent, 
            env_list, 
            task_list, 
            available_tasks,
            ps_line,
            total_episode=args.eval_episode, 
            save_path=save_path
        )


if __name__ == "__main__":
    args = argparse.parse_args()
    set_seed(args.seed)
    if args.max_think_token is None:
        args.max_think_token = 40  # rationale tokens per slot used for the reported results
    # directory settings
    if args.dataset_dir is None:
        args.dataset_dir = f"./datasets/{args.env_name}/{args.continual_type}_train/{args.continual_set}"
    if args.output_dir is None:
        args.output_dir = f"./results/ASC/{args.env_name}/{args.continual_type}_{args.continual_seq}/{args.continual_set}_{args.pi_model_name.split('/')[-1]}_{args.tag}"

    if args.prev_tag is None:
        args.prev_tag = args.tag

    if args.train:
        train_policy(args)
    if args.test:
        test(args)
    if args.evaluate:
        evaluate(args)
