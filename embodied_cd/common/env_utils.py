import os
import json
import random

from tqdm import tqdm
from typing import Union, List, Dict
import numpy as np

from embodied_cd.environments.default.virtualhome import TASKS_SET

from embodied_cd.environments.base import BaseEnvironment
from embodied_cd.environments import VirtualHomeEnv, AlfredEnv
from embodied_cd.common.agent import BaseAgent
from embodied_cd.common.print_utils import *


def make_env(env_name, num_topk_edge, ip, seed, port=8080, split_path=None):
    if env_name == "virtualhome":
        print_warn(f"Num TopK Edge: {num_topk_edge}")
        return VirtualHomeEnv(num_topk_edge=num_topk_edge, ip=ip, port=port, seed=seed)
    elif env_name == "alfred":
        print_warn(f"Num TopK Edge: 12")
        return AlfredEnv(split_path, num_topk_edge=12)


def evaluation_vh(
    env: Union[str, BaseEnvironment],
    agent: BaseAgent,
    env_id: int,
    task_id: int,
    total_episode: int = 4,
    max_timestep: int = 15,
    max_ps: List[int] = None,
    few_shot_examples: str = None,
    verbose: bool = True,
):
    (
        lengths,
        total_lengths,
        successes,
        goal_rewards,
        ps_successes,
        ps_rewards,
        subgoal_successes,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    rooms = ["kitchen", "bedroom", "livingroom", "bathroom"]
    for epi in range(total_episode):
        room = [rooms[epi % 4]] if total_episode >= 4 else random.sample(rooms, 1)
        print_pass(f"\n\nInstruction: {TASKS_SET[task_id]}")
        obs, info = env.reset(task_id=task_id, env_id=env_id, init_rooms=room)

        agent.reset(info["task_type"], info["task"])

        ps = max_ps[room[0]]
        if "Open" in info["task"] or "Turn" in info["task"]:
            max_timestep = 8
        else:
            max_timestep = 14

        print_warn(f"Maximun Timestep Limit: {max_timestep}")

        instruction = info["task"]
        done, success = False, False
        # agent should solve within max timestep
        for timestep in range(max_timestep):
            action = agent.forward(
                instruction, info["state"], info["history"], few_shot_examples
            )

            obs, reward, done, info = env.step(action)
            print_check(f"Action: {action}")

            if done:
                success = info["success"]
                break

        if success:
            print_pass(f"Task: {instruction} Success")

            # Sometimes the subgoal is not fully achieved even if the task is success
            if 0 in info["goal_info"]:
                info["goal_info"] = [1] * len(info["goal_info"])
                info["goal_reward"] = 1

            print_check(
                f"Goal Success Rate {info['goal_info']} / {info['goal_reward']}"
            )
            ps_reward = 1.0 - ((timestep + 1) - ps) / ps
            if ps_reward < 0.0:
                ps_reward = 0.0
            if (timestep + 1) <= ps:
                ps_successes.append(100)
            else:
                ps_successes.append(0)
            print_check(f"Max PS: {ps}, Cur PS: {timestep+1}")
            print_check(f"PSR: {ps_successes[-1]}, PRW: {ps_reward}")
        else:
            print_error(f"Task: {instruction} Failed")
            print_check(
                f"Goal Success Rate {info['goal_info']} / {info['goal_reward']}"
            )
            ps_successes.append(0)
            print_check(f"Max PS: {ps}, Cur PS: {timestep+1}")
            print_check(f"PSR: {ps_successes[-1]}")

        successes.append(int(success) * 100)
        if success:
            lengths.append(timestep + 1)
            ps_rewards.append(ps_reward)
        total_lengths.append(timestep + 1)
        goal_rewards.append(max(int(success), info["goal_reward"]) * 100)
        subgoal_successes.append(
            sum(info["subgoal_success"]) / len(info["subgoal_success"]) * 100
        )

    return (
        successes,
        lengths,
        total_lengths,
        goal_rewards,
        ps_successes,
        ps_rewards,
        subgoal_successes,
    )


def evaluation_af(
    env: Union[str, BaseEnvironment],
    agent: BaseAgent,
    env_id: int,
    task_id: int,
    total_episode: int = 1,  # not used
    max_timestep: int = 10,
    max_ps: List[int] = None,
    few_shot_examples: str = None,
    game_files: List[str] = None,
    verbose: bool = True,
):
    (
        lengths,
        total_lengths,
        successes,
        goal_rewards,
        ps_successes,
        ps_rewards,
        subgoal_successes,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for epi in range(total_episode):
        # set game
        env.set_game_files([game_files[epi]], expert_type="handcoded")

        # reset
        obs, info = env.reset()
        agent.reset(info["task_type"], info["task"])

        ps = max_ps[str(epi)]
        if info["task_type"] == "heat":  # set 2
            max_timestep = 8
        elif info["task_type"] == "puttwo":  # set 3
            max_timestep = 12
        elif info["task_type"] == "put":  # set 5
            max_timestep = 6
        elif info["task_type"] == "clean":  # set 6
            max_timestep = 8
        else:
            raise NotImplementedError
        print_warn(f"Maximun Timestep Limit: {max_timestep}")

        instruction = info["task"]
        done, success = False, False
        print_pass(f"\n\nInstruction: {instruction}")
        # agent should solve within max timestep
        for timestep in range(max_timestep):
            action = agent.forward(
                instruction, info["obs"], info["history"], few_shot_examples
            )

            try:
                obs, reward, done, info = env.step(action)
                print_check(f"Action: {action}")
            except Exception as e:
                print_error(f"env.step({action!r}) failed: {e!r}; ending the episode as a failure")
                info = dict(info, success=False)
                done = True

            if done:
                success = info["success"]
                break

        if success:
            print_pass(f"Task: {instruction} Success")
            print_check(f"Goal Success Rate {info['goal_info']}")
            ps_reward = 1.0 - ((timestep + 1) - ps) / ps
            if ps_reward < 0.0:
                ps_reward = 0.0
            if (timestep + 1) <= ps:
                ps_successes.append(100)
            else:
                ps_successes.append(0)
            print_check(f"Max PS: {ps}, Cur PS: {timestep+1}")
            print_check(f"PSR: {ps_successes[-1]}, PRW: {ps_reward}")
        else:
            print_error(f"Task: {instruction} Failed")
            print_check(f"Goal Success Rate {info['goal_info']}")
            ps_successes.append(0)
            print_check(f"Max PS: {ps}, Cur PS: {timestep+1}")
            print_check(f"PSR: {ps_successes[-1]}")

        successes.append(int(success) * 100)
        if success:
            lengths.append(timestep + 1)
            ps_rewards.append(ps_reward)
        total_lengths.append(timestep + 1)
        goal_rewards.append(info["goal_reward"] * 100)

    return successes, lengths, total_lengths, goal_rewards, ps_successes, ps_rewards, []


def evaluation_af_pipe(
    env: Union[str, BaseEnvironment],
    agent: BaseAgent,
    env_list: List[int],
    task_list: List[int],
    available_tasks: List[int] = None,
    pending_steps: Dict = None,
    total_episode: int = 1,
    max_timestep: int = 10,
    few_shot_examples: str = None,
    save_path: str = None,
    verbose: bool = False,
):
    # set game files
    game_files = []
    # reverse game file ...
    if "unseen" in save_path:
        print_error("Reversing !!!")
        env.game_files.reverse()

    for task_id, task in task_list.items():
        count = 0
        for game in env.game_files:
            instruction = game.split("/")[-3]  # .../<task>-<scene>/<trial>/<game file>
            room = instruction.split("-")[-1]
            instruction = instruction[: -len(room)]
            if task in instruction:
                game_files.append(game)
                count += 1
            if count == total_episode:
                break
    assert len(game_files) == total_episode * len(
        task_list
    ), f"game files {len(game_files)} != {total_episode * len(task_list)} are not enough!"

    successes, lengths, total_lengths, goal_rewards, ps_successes, ps_rewards = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, task_id in tqdm(enumerate(task_list.keys()), desc="task", position=0):
        cur_game_files = game_files[
            (i * total_episode) : (i * total_episode) + total_episode
        ]
        max_ps = pending_steps[str(task_id)]

        success, length, total_length, goal_reward, ps_success, ps_reward, _tmp= (
            evaluation_af(
                env,
                agent,
                None,
                task_id,
                total_episode,
                max_timestep,
                max_ps,
                few_shot_examples,
                cur_game_files,
                verbose,
            )
        )
        successes.extend(success)
        lengths.extend(length)
        total_lengths.extend(total_length)
        goal_rewards.extend(goal_reward)
        ps_successes.extend(ps_success)
        ps_rewards.extend(ps_reward)

        print("=" * 30)
        print_check(f"[[Env Single, Task {task_id}]]")
        print_pass(f"Evaluation Episodes: {len(success)}")
        print_pass(f"Average SR: {np.mean(success):.2f} +/- {np.std(success):.2f}")
        print_pass(
            f"Average GC: {np.mean(goal_reward):.2f} +/- {np.std(goal_reward):.2f}"
        )
        print_pass(
            f"Average Length: {np.mean(length) if length else 0:.2f} +/- {np.std(length) if length else 0:.2f}"
        )
        print_pass(
            f"Average PSR: {np.mean(ps_success):.2f} +/- {np.std(ps_success):.2f}"
        )
        print_pass(f"Average PRW: {np.mean(ps_reward):.2f} +/- {np.std(ps_reward):.2f}")
        print("=" * 30)

        if save_path:
            folder = os.path.dirname(save_path)
            os.makedirs(folder, exist_ok=True)

            with open(save_path, "a") as f:
                f.write(f"[[Env Single, Task {task_id}]]\n")
                f.write(f"Evaluation Episodes: {len(success)}\n")
                f.write(
                    f"Average SR: {np.mean(success):.2f} +/- {np.std(success):.2f}\n"
                )
                f.write(
                    f"Average GC: {np.mean(goal_reward):.2f} +/- {np.std(goal_reward):.2f}\n"
                )
                f.write(
                    f"Average Length: {np.mean(length) if length else 0:.2f} +/- {np.std(length) if length else 0:.2f}\n"
                )
                f.write(
                    f"Average Total Length: {np.mean(total_length) if total_length else 0:.2f} +/- {np.std(total_length) if total_length else 0:.2f}\n"
                )
                f.write(
                    f"Average PSR: {np.mean(ps_success):.2f} +/- {np.std(ps_success):.2f}\n"
                )
                f.write(
                    f"Average PRW: {np.mean(ps_reward):.2f} +/- {np.std(ps_reward):.2f}\n"
                )
                f.write(f"=" * 30 + "\n")

    evaluation_print(
        agent,
        successes,
        lengths,
        total_lengths,
        goal_rewards,
        ps_successes,
        ps_rewards,
        [],
        save_path,
    )


def evaluation_vh_pipe(
    env: Union[str, BaseEnvironment],
    agent: BaseAgent,
    env_list: List[int],
    task_list: List[int],
    available_tasks: List[int] = None,
    pending_steps: Dict = None,
    total_episode: int = 1,
    max_timestep: int = 15,
    few_shot_examples: str = None,
    save_path: str = None,
    verbose: bool = False,
):
    (
        successes,
        lengths,
        total_lengths,
        goal_rewards,
        ps_successes,
        ps_rewards,
        subgoal_successes,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for env_id in tqdm(env_list, desc="env", position=0):
        for task_id in tqdm(task_list, desc="task", position=1, leave=False):
            if available_tasks is not None and task_id in available_tasks[env_id]:
                try:
                    max_ps = pending_steps[str(env_id)][str(task_id)]
                except KeyError:
                    max_ps = {
                        "livingroom": 10,
                        "bedroom": 10,
                        "kitchen": 10,
                        "bathroom": 10,
                    }

                (
                    success,
                    length,
                    total_length,
                    goal_reward,
                    ps_success,
                    ps_reward,
                    subgoal_success,
                ) = evaluation_vh(
                    env,
                    agent,
                    env_id,
                    task_id,
                    total_episode,
                    max_timestep,
                    max_ps,
                    few_shot_examples,
                    verbose,
                )
                successes.extend(success)
                lengths.extend(length)
                total_lengths.extend(total_length)
                goal_rewards.extend(goal_reward)
                ps_successes.extend(ps_success)
                ps_rewards.extend(ps_reward)
                subgoal_successes.extend(subgoal_success)

                print("=" * 30)
                print_check(f"[[Env {env_id}, Task {task_id}]]")
                print_pass(f"Evaluation Episodes: {len(success)}")
                print_pass(
                    f"Average SR: {np.mean(success):.2f} +/- {np.std(success):.2f}"
                )
                print_pass(
                    f"Average Subgoal SR: {np.mean(subgoal_success):.2f} +/- {np.std(subgoal_success):.2f}"
                )
                print_pass(
                    f"Average GC: {np.mean(goal_reward):.2f} +/- {np.std(goal_reward):.2f}"
                )
                print_pass(
                    f"Average Length: {np.mean(length) if length else 0:.2f} +/- {np.std(length) if length else 0:.2f}"
                )
                print_pass(
                    f"Average PSR: {np.mean(ps_success):.2f} +/- {np.std(ps_success):.2f}"
                )
                print_pass(
                    f"Average PRW: {np.mean(ps_reward):.2f} +/- {np.std(ps_reward):.2f}"
                )
                print("=" * 30)

                if save_path:
                    folder = os.path.dirname(save_path)
                    os.makedirs(folder, exist_ok=True)

                    with open(save_path, "a") as f:
                        f.write(f"[[Env {env_id}, Task {task_id}]]\n")
                        f.write(f"Evaluation Episodes: {len(success)}\n")
                        f.write(
                            f"Average SR: {np.mean(success):.2f} +/- {np.std(success):.2f}\n"
                        )
                        f.write(
                            f"Average Subgoal SR: {np.mean(subgoal_success):.2f} +/- {np.std(subgoal_success):.2f}\n"
                        )
                        f.write(
                            f"Average GC: {np.mean(goal_reward):.2f} +/- {np.std(goal_reward):.2f}\n"
                        )
                        f.write(
                            f"Average Length: {np.mean(length) if length else 0:.2f} +/- {np.std(length) if length else 0:.2f}\n"
                        )
                        f.write(
                            f"Average Total Length: {np.mean(total_length) if total_length else 0:.2f} +/- {np.std(total_length) if total_length else 0:.2f}\n"
                        )
                        f.write(
                            f"Average PSR: {np.mean(ps_success):.2f} +/- {np.std(ps_success):.2f}\n"
                        )
                        f.write(
                            f"Average PRW: {np.mean(ps_reward):.2f} +/- {np.std(ps_reward):.2f}\n"
                        )
                        f.write(f"=" * 30 + "\n")

    evaluation_print(
        agent,
        successes,
        lengths,
        total_lengths,
        goal_rewards,
        ps_successes,
        ps_rewards,
        subgoal_successes,
        save_path,
    )


def evaluation_print(
    agent,
    successes,
    lengths,
    total_lengths,
    goal_rewards,
    ps_successes,
    ps_rewards,
    subgoal_successes,
    save_path,
):
    (
        successes,
        lengths,
        total_lengths,
        goal_rewards,
        ps_successes,
        ps_rewards,
        subgoal_successes,
    ) = (
        np.array(successes),
        np.array(lengths),
        np.array(total_lengths),
        np.array(goal_rewards),
        np.array(ps_successes),
        np.array(ps_rewards),
        np.array(subgoal_successes),
    )
    average_success, std_success = np.mean(successes), np.std(successes)
    average_length, std_length = np.mean(lengths), np.std(lengths)
    average_total_length, std_total_length = np.mean(total_lengths), np.std(
        total_lengths
    )
    average_goal_reward, std_goal_reward = np.mean(goal_rewards), np.std(goal_rewards)
    average_ps_success, std_ps_success = np.mean(ps_successes), np.std(ps_successes)
    average_ps_rewards, std_ps_rewards = np.mean(ps_rewards), np.std(ps_rewards)
    average_subgoal_success, std_subgoal_success = (
        np.mean(subgoal_successes),
        np.std(subgoal_successes),
    )

    print_check(f"Total Evaluation Episodes: {len(successes)}")
    print_pass(f"Average SR: {average_success:.2f} +/- {std_success:.2f}")
    print_pass(
        f"Average Subgoal SR: {average_subgoal_success:.2f} +/- {std_subgoal_success:.2f}"
    )
    print_pass(f"Average GC: {average_goal_reward:.2f} +/- {std_goal_reward:.2f}")
    print_pass(f"Average Success Length: {average_length:.2f} +/- {std_length:.2f}")
    print_pass(
        f"Average Total Length: {average_total_length:.2f} +/- {std_total_length:.2f}"
    )
    print_pass(f"Average PSR: {average_ps_success:.2f} +/- {std_ps_success:.2f}")
    print_pass(f"Average PRW: {average_ps_rewards:.2f} +/- {std_ps_rewards:.2f}")
    try:
        generated_tokens = agent.get_generated_tokens()
    except AttributeError:
        generated_tokens = None
    print_pass(f"Average Generated Tokens: {generated_tokens}")
    try:
        reasoning_counts = agent.get_reasoning_counts()
    except AttributeError:
        reasoning_counts = None
    print_pass(f"Reasoning Counts: {reasoning_counts}")

    if save_path:
        with open(save_path, "a") as f:
            f.write(f"[[Total Results]]\n")
            f.write(f"Evaluation Episodes: {len(successes)}\n")
            f.write(f"Average SR: {average_success:.2f} +/- {std_success:.2f}\n")
            f.write(
                f"Average Subgoal SR: {average_subgoal_success:.2f} +/- {std_subgoal_success:.2f}\n"
            )
            f.write(
                f"Average GC: {average_goal_reward:.2f} +/- {std_goal_reward:.2f}\n"
            )
            f.write(
                f"Average Success Length: {average_length:.2f} +/- {std_length:.2f}\n"
            )
            f.write(
                f"Average Total Length: {average_total_length:.2f} +/- {std_total_length:.2f}\n"
            )
            f.write(f"Average Generated Tokens: {generated_tokens}\n")
            f.write(f"Reasoning Counts: {reasoning_counts}\n")
            f.write(f"Average PSR: {average_ps_success:.2f} +/- {std_ps_success:.2f}\n")
            f.write(f"Average PRW: {average_ps_rewards:.2f} +/- {std_ps_rewards:.2f}\n")
            f.write(f"=" * 30 + "\n")


def evaluation_pipe(
    env: Union[str, BaseEnvironment],
    agent: BaseAgent,
    env_list: List[int],
    task_list: List[int],
    available_tasks: List[int] = None,
    pending_steps: Dict = None,
    total_episode: int = 1,
    max_timestep: int = 15,
    few_shot_examples: str = None,
    save_path: str = None,
    verbose: bool = False,
):
    if env.name == "virtualhome":
        evaluation_vh_pipe(
            env,
            agent,
            env_list,
            task_list,
            available_tasks,
            pending_steps,
            total_episode,
            max_timestep,
            few_shot_examples,
            save_path,
            verbose,
        )
    elif env.name == "alfred":
        evaluation_af_pipe(
            env,
            agent,
            env_list,
            task_list,
            available_tasks,
            pending_steps,
            total_episode,
            max_timestep,
            few_shot_examples,
            save_path,
            verbose,
        )
    else:
        raise ValueError(f"env should be one of virtualhome / alfred, currently {env}")


def build_skill_list(action_format, object_list, env_name, reduced_skills=False):
    if reduced_skills:
        if env_name == "virtualhome":
            property_path = (
                "externals/virtualhome/virtualhome/resources/properties_data_all.json"
            )
            with open(property_path, "r") as f:
                properties = json.load(f)
                for room in ["kitchen", "bedroom", "livingroom", "bathroom"]:
                    properties[room] = ["ROOM"]

            property_path = property_path.replace("_all", "_unity")
            with open(property_path, "r") as f:
                unity_properties = json.load(f)
                for obj in unity_properties:
                    if obj not in properties:
                        properties[obj] = unity_properties[obj]

            if "kitchencounterdrawers" in properties:
                properties["kitchencounterdrawer"] = properties["kitchencounterdrawers"]

            compatibility = {"source": {}, "target": {}}
            for obj in object_list:
                compatibility["source"][obj] = ["walk"]
                compatibility["target"][obj] = ["walk"]

                if "CAN_OPEN" in properties[obj]:
                    compatibility["source"][obj].append("open")
                    compatibility["source"][obj].append("close")
                if "HAS_SWITCH" in properties[obj]:
                    compatibility["source"][obj].append("switch")
                if "GRABBABLE" in properties[obj]:
                    compatibility["source"][obj].append("grab")
                    compatibility["source"][obj].append("place")
                    compatibility["source"][obj].append("put")
                if "ISROOM" not in properties[obj]:
                    compatibility["target"][obj].append("place")
                    compatibility["target"][obj].append("put")

        elif env_name == "alfred":
            property_path = "externals/alfworld/object_properties.json"
            with open(property_path, "r") as f:
                properties = json.load(f)

            compatibility = {"source": {}, "target": {}}
            for obj in object_list:
                base_obj = obj.split()[0]

                compatibility["source"][obj] = ["go"]
                compatibility["target"][obj] = ["go"]

                if base_obj in properties["grabbable"]:
                    compatibility["source"][obj].append("put")
                    compatibility["source"][obj].append("take")
                    compatibility["source"][obj].append("heat")
                    compatibility["source"][obj].append("clean")
                if base_obj in properties["receptacle"]:
                    compatibility["target"][obj].append("put")
                    compatibility["target"][obj].append("take")
                if base_obj == "sinkbasin":
                    compatibility["target"][obj].append("clean")
                if base_obj == "microwave":
                    compatibility["target"][obj].append("heat")

    skill_list = []
    for act in action_format:
        if reduced_skills:
            action = act.split(" ")[0].strip()
            source_pool = list(
                filter(lambda x: action in compatibility["source"][x], object_list)
            )
            target_pool = list(
                filter(lambda x: action in compatibility["target"][x], object_list)
            )
        else:
            source_pool = object_list
            target_pool = object_list

        if "noun2" in act:
            for obj1 in source_pool:
                for obj2 in target_pool:
                    skill = act.format(noun1=obj1, noun2=obj2)
                    if skill not in skill_list:
                        skill_list.append(skill)
        else:
            for obj in source_pool:
                skill = act.format(noun1=obj)
                if skill not in skill_list:
                    skill_list.append(skill)

    return skill_list
