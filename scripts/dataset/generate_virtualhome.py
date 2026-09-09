import os
import json
import random
import argparse

from vh_dataset.dataset.vh_expert_policy import ExpertPolicy
from virtualhome.simulation.environment.resources import TASKS_SET, SUCCESS_CONDITIONS
from embodied_cd.environments.default.virtualhome import VirtualHomeEnv
from embodied_cd.common.print_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--type", type=str, default="behavior_train")
    parser.add_argument("--set", type=int, default=None)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--episodes", type=int, default=4)
    return parser.parse_args()


def run_episode(env, env_id, task_id, init_rooms, epi_id):
    success = False
    trajectory = []

    # 1. Reset Environment
    print_warn(f"Instruction: {TASKS_SET[task_id]}")
    print_warn(f"Success Condition: {SUCCESS_CONDITIONS[task_id]}")
    print_check(f"Init Room: {init_rooms[epi_id % 4]}")
    prev_action = ""

    obs = env.reset(task_id, env_id, init_rooms=[init_rooms[epi_id % 4]])
    if not obs:
        print("There are no task objects")
        return trajectory, success
    
    pos_kg = env.env.get_position_graph()
    policy = ExpertPolicy(TASKS_SET[task_id], pos_kg)
    success, done, history, timesteps = True, False, [], 0
    
    while not done:

        action = policy.predict(obs=obs, timesteps=timesteps, prev_action=prev_action)
        print(f"Action: {action}")

        if action is None:
            success = False
            break

        next_obs, reward, done, info = env.step(action)
        new_pos_kg = env.env.get_position_graph()

        if not info["success"]:
            success = False
            break

        trajectory.append(
            {
                "instruction": TASKS_SET[task_id],
                "action": action,
                "timesteps": timesteps,
                "env_id": env_id,
                "position_graph": pos_kg,
                "agent_graph": obs["agent_graph"],
                "visible_graph": obs["visible_graph"],
            }
        )

        timesteps += 1
        prev_action = action
        obs = next_obs
        pos_kg = new_pos_kg

    print_check(f"Success {success}")
    return trajectory, success


def main(args):
    # make env
    env = VirtualHomeEnv(ip=args.ip, seed=args.seed, use_unity=True)

    # load dataset configs
    config = {}
    with open(f'./configs/virtualhome/{args.type}.py', 'r') as f:
        code = f.read()
    exec(code, config)

    continual_sequence = config["sequences"]["sequence_1"]  # default sequence
    if args.set is not None:
        continual_sequence = [f"set_{args.set}"]
    env_list, task_list = config["rooms"], config["tasks"]
    available_tasks = config["available_tasks"]

    for continual_set in continual_sequence:
        # make dataset path
        dataset_path = f"./datasets/virtualhome/{args.type}/{continual_set}/raw"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        for env_id in env_list[continual_set]:
            for task_id in task_list[continual_set]:
                if task_id in available_tasks[env_id]:
                    print_check(f"[{continual_set}]: Env {env_id}, Task {task_id}")
                    print_check(f"Start saving expert trajectories")

                    init_rooms = ["livingroom", "bedroom", "kitchen", "bathroom"]
                    if args.episodes < 4:
                        random.shuffle(init_rooms) 
                        print_warn(f"Shuffling init rooms {init_rooms}")

                    for epi_id in range(args.episodes):  # n episode per task
                        trajectory, success = run_episode(env, env_id, task_id, init_rooms, epi_id)

                        if len(trajectory) > 2 and success:
                            if not os.path.exists(
                                os.path.join(dataset_path, f"task_{task_id}")
                            ):
                                os.makedirs(
                                    os.path.join(dataset_path, f"task_{task_id}")
                                )
                            with open(
                                os.path.join(
                                    dataset_path,
                                    f"task_{task_id}",
                                    f"episode_env_{env_id}_epi_{epi_id}.jsonl",
                                ),
                                "w",
                            ) as jsonl_file:
                                for entry in trajectory:
                                    jsonl_file.write(json.dumps(entry) + "\n")
                else:
                    print_warn(f"[{continual_set}]: Env {env_id}, Task {task_id}")
                    print_warn(f"Task is unavailable in this environment")

def sub(args):
    # make env
    env = VirtualHomeEnv(ip=args.ip, seed=args.seed, use_unity=True)

    # load dataset configs
    config = {}
    with open(f'./configs/virtualhome/{args.type}.py', 'r') as f:
        code = f.read()
    exec(code, config)

    continual_sequence = config["sequences"]["sequence_1"]  # default sequence
    if args.set is not None:
        continual_sequence = [f"set_{args.set}"]
    env_list, task_list = config["rooms"], config["tasks"]
    available_tasks = config["available_tasks"]

    for continual_set in continual_sequence:
        # make dataset path
        dataset_path = f"./datasets/virtualhome/{args.type}/{continual_set}"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        entry = {}
        for env_id in env_list[continual_set]:
            entry_env = {}
            for task_id in task_list[continual_set]:
                if task_id in available_tasks[env_id]:
                    print_check(f"[{continual_set}]: Env {env_id}, Task {task_id}")
                    print_check(f"Start saving expert trajectories")

                    init_rooms = ["livingroom", "bedroom", "kitchen", "bathroom"]
                    if args.episodes < 4:
                        random.shuffle(init_rooms) 
                        print_warn(f"Shuffling init rooms {init_rooms}")

                    entry_task = {}
                    for epi_id in range(args.episodes):  # n episode per task
                        trajectory, success = run_episode(env, env_id, task_id, init_rooms, epi_id)
                        # add
                        entry_task[init_rooms[epi_id]] = len(trajectory) if len(trajectory) != 1 else 2
                        print(entry_task[init_rooms[epi_id]])
                    # add
                    entry_env[task_id] = entry_task
                else:
                    print_warn(f"[{continual_set}]: Env {env_id}, Task {task_id}")
                    print_warn(f"Task is unavailable in this environment")
            entry[env_id] = entry_env

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
    main(args)  # raw episodes -> <set>/raw/
    sub(args)   # evaluation episodes -> <set>/expert_traj.jsonl
