import os
import json
import argparse
from functools import partial

from transformers import logging, set_seed
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder

from embodied_cd.environments.default import AlfredEnv
from embodied_cd.environments.default.alfred import AlfredKG
from embodied_cd.common.dataset_utils import embedding_fns
from embodied_cd.common.print_utils import *

logging.set_verbosity_error()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="behavior_train")
    parser.add_argument("--set", type=int, default=None)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--num_topk_edge", type=int, default=8)
    return parser.parse_args()


def run_episode(env):
    trajectory = []
    
    # 1. Reset Environment
    obs, info = env.reset()
    plan = info["expert_plan"]
    print_warn(f"Instruction: {info['task']}")
    success, done, history, timesteps = False, False, [], 0
    while not done:
        action = plan[timesteps]
        print(action)

        trajectory.append(
            {
                "instruction": info["task"],
                "state": info["obs"],
                "state_full": info["obs_full"],
                "action": action,
                "timesteps": timesteps,
                "knowledge_graph": {
                    "edges": info["kg"].edges,
                    "nouns": list(info["kg"].nouns),
                },
            }
        )
        obs, reward, done, info = env.step(action)
        if done:
            success = True
            break
        
        timesteps += 1
    
    # clean trajectory

    for traj in trajectory:
        kg = AlfredKG(**traj['knowledge_graph'])
        state = kg.retrieve(
            [traj['instruction']],
            embedding_fns=partial(
                embedding_fns, kg_tokenizer, kg_model),
            num_edges=8,
        )
        print_check(traj['state'])
        print(f"Action: {traj['action']}")

    print_check(f"Success {success}")
    return trajectory, success


def main(args):
    set_seed(args.seed)
    
    # load dataset configs
    config = {}
    with open(f'./configs/alfred/{args.type}.py', 'r') as f:
        code = f.read()
    exec(code, config)

    continual_sequence = config["sequences"]["sequence_1"] # default sequence
    if args.set is not None:
        continual_sequence = [f"set_{args.set}"]
    env_list, task_list = config["rooms"], config["tasks"]

    for continual_set in continual_sequence:
        dataset_path = f"./datasets/alfred/{args.type}/{continual_set}/raw"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # initialize environment
        split_path = f"./externals/cl-alfred/embodied_split/{args.type.split('_')[0]}_il/embodied_data_disjoint_rand1_cls1_task{int(continual_set.split('_')[-1])-1}.json"
        env = AlfredEnv(split_path, args.num_topk_edge)
        
        # first organzie game files
        game_files = []
        if "unseen" in args.type:
            env.game_files.reverse()
        for task_id, task in task_list[continual_set].items():
            count = 0
            for game in env.game_files:
                instruction = game.split('/')[-3]  # .../<task>-<scene>/<trial>/<game file>
                room = instruction.split('-')[-1]
                instruction = instruction[:-len(room)]
                if task in instruction:
                    game_files.append(game)
                    count += 1
                if count == args.episodes:
                    break

        assert len(game_files) == args.episodes * len(task_list[continual_set]),\
                f"game files {len(game_files)} != {args.episodes * len(task_list[continual_set])} are not enough!\n{game_files}"

        for i, task_id in enumerate(task_list[continual_set]):
            for epi_idx in range(args.episodes):
                print_pass(f"Episode: {epi_idx}")

                env.set_game_files([game_files[(i*args.episodes) + epi_idx]])
                trajectory, success = run_episode(env)

                if len(trajectory) > 2 and success:
                    if not os.path.exists(
                        os.path.join(dataset_path, f"task_{task_id}")
                    ):
                        os.makedirs(os.path.join(dataset_path, f"task_{task_id}"))
                    with open(
                        os.path.join(
                            dataset_path,
                            f"task_{task_id}",
                            f"episode_env_{epi_idx}.jsonl",
                        ),
                        "w",
                    ) as jsonl_file:
                        for entry in trajectory:
                            jsonl_file.write(json.dumps(entry) + "\n")


def sub(args):
    set_seed(args.seed)
    
    # load dataset configs
    config = {}
    with open(f'./configs/alfred/{args.type}.py', 'r') as f:
        code = f.read()
    exec(code, config)

    continual_sequence = config["sequences"]["sequence_1"] # default sequence
    if args.set is not None:
        continual_sequence = [f"set_{args.set}"]
    env_list, task_list = config["rooms"], config["tasks"]

    for continual_set in continual_sequence:
        dataset_path = f"./datasets/alfred/{args.type}/{continual_set}"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # initialize environment
        split_path = f"./externals/cl-alfred/embodied_split/{args.type.split('_')[0]}_il/embodied_data_disjoint_rand1_cls1_task{int(continual_set.split('_')[-1])-1}.json"
        env = AlfredEnv(split_path, args.num_topk_edge)
        
        # first organzie game files
        game_files = []
        if "unseen" in args.type:
            env.game_files.reverse()
        for task_id, task in task_list[continual_set].items():
            count = 0
            for game in env.game_files:
                instruction = game.split('/')[-3]  # .../<task>-<scene>/<trial>/<game file>
                room = instruction.split('-')[-1]
                instruction = instruction[:-len(room)]
                if task in instruction:
                    game_files.append(game)
                    count += 1
                if count == args.episodes:
                    break

        assert len(game_files) == args.episodes * len(task_list[continual_set]),\
                f"game files {len(game_files)} != {args.episodes * len(task_list[continual_set])} are not enough!\n{game_files}"

        entry = {}
        for i, task_id in enumerate(task_list[continual_set]):
            entry_task = {}
            for epi_idx in range(args.episodes):
                print_pass(f"Episode: {epi_idx}")

                env.set_game_files([game_files[(i*args.episodes) + epi_idx]])
                trajectory, success = run_episode(env)
                
                # add
                entry_task[epi_idx] = len(trajectory)
                print(entry_task[epi_idx])
            entry[task_id] = entry_task

        with open(
            os.path.join(
                dataset_path,
                f"expert_traj.jsonl"
            ),
            "w",
        ) as json_file:
            json_file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    args = parse_args()
    # knowledge-graph encoder used by run_episode (loaded here so --help needs no download)
    kg_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    kg_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    main(args)  # raw episodes -> <set>/raw/
    sub(args)   # evaluation episodes -> <set>/expert_traj.jsonl
