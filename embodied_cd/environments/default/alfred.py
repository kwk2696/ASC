import os.path as osp
import copy
from itertools import product

import yaml
import numpy as np
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi
from vh_dataset.dataset.virtualhome import make_embedding_fn

from embodied_cd.environments.base import BaseEnvironment, CustomAlfredTWEnv


class AlfredEnv(BaseEnvironment):
    name = "alfred"

    def __init__(self, split="train", num_topk_edge=8):
        super().__init__()

        config_path = osp.join("externals/alfworld/configs/base_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        train_eval = {
            "train": "train",
            "valid_seen": "eval_in_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }.get(split, split)

        self.tw_env = CustomAlfredTWEnv(config, train_eval=train_eval)
        self.env = self.tw_env.init_env(batch_size=1)
        self.emb_fn = make_embedding_fn()

        self.repetitive_action_patience = 3

        # get total game files
        self.game_files = self.env.gamefiles
        self.num_game_files = len(self.env.gamefiles)

        # topk edge
        self.scene_dict = {
            0: "kitchen",
            2: "livingroom",
            3: "bedroom",
            4: "bathroom",
        }
        self.num_topk_edge = num_topk_edge

    def set_game_files(self, game_files, expert_type="planner"):
        self.tw_env.game_files = game_files
        self.tw_env.num_games = len(game_files)
        self.env = self.tw_env.init_env(batch_size=1, expert_type=expert_type)

    def step(self, action):
        origin_action = copy.deepcopy(action)

        if action.startswith("put") and "in/on" not in action.split(" "):
            action_tokens = action.split(" ")
            if "in" in action_tokens:
                action_tokens[action_tokens.index("in")] = "in/on"
            elif "on" in action_tokens:
                action_tokens[action_tokens.index("on")] = "in/on"
            action = " ".join(action_tokens)


        obs, reward, done, info = self.env.step([action])

        ################# Retrieve State ###################
        if obs[0] != "Nothing happens.":
            action_tokens = action.split(" ")
            if "take" in action:
                self.agent_obs.append(
                    ("character", "hold", f"{action_tokens[1]} {action_tokens[2]}")
                )
            elif "go" in action:
                for agent_obs in self.agent_obs:
                    if f"close" in agent_obs[1]:
                        self.agent_obs.remove(agent_obs)
                        break
                self.agent_obs.append(
                    ("character", "close", f"{action_tokens[-2]} {action_tokens[-1]}")
                )
            elif "put" in action:
                for agent_obs in self.agent_obs:
                    if f"{action_tokens[1]} {action_tokens[2]}" in agent_obs[2]:
                        self.agent_obs.remove(agent_obs)
                        break
            elif "heat" in action:
                for agent_obs in self.agent_obs:
                    if agent_obs[2] == f"{action_tokens[1]} {action_tokens[2]}":
                        self.agent_obs.remove(agent_obs)
                        break
                self.agent_obs.append(
                    ("character", "hold", f"{action_tokens[1]} {action_tokens[2]}")
                )
            elif "cool" in action:
                for agent_obs in self.agent_obs:
                    if agent_obs[2] == f"{action_tokens[1]} {action_tokens[2]}":
                        self.agent_obs.remove(agent_obs)
                        break
                self.agent_obs.append(
                    ("character", "hold", f"{action_tokens[1]} {action_tokens[2]}")
                )
            elif "clean" in action:
                for agent_obs in self.agent_obs:
                    if agent_obs[2] == f"{action_tokens[1]} {action_tokens[2]}":
                        self.agent_obs.remove(agent_obs)
                        break
                self.agent_obs.append(
                    ("character", "hold", f"{action_tokens[1]} {action_tokens[2]}")
                )

        state, state_full, edges = self._parse_obs(info["facts"][0], self.parse_goal)

        # append agent obs
        if len(self.agent_obs) > 0:
            agent_str = []
            for agent_obs in self.agent_obs:
                agent_str.append(str(agent_obs))
            state = " ".join(agent_str).replace("'", "") + ", " + state
            state_full = " ".join(agent_str).replace("'", "") + ", " + state_full
        ###################################################

        ############### Add Historty #####################
        self.history.append(str((f"step {self.timesteps+1}", origin_action)))

        try:
            action_history = [h.split("', '")[1][:-2] for h in self.history][
                -self.repetitive_action_patience :
            ]
        except Exception:  # sometimes input action is impossible to parse
            action_history = []

        if (
            len(action_history) == self.repetitive_action_patience
            and len(set(action_history)) == 1
        ):
            obs = f"Lost in repetitive action {action}."
            done = True
            info = {
                "success": False,
                "task": self.parse_goal,
                "goal_reward": sum(self.goal_rewards) / len(self.goal_rewards),
                "goal_info": self.goal_rewards,
            }
            return obs, 0, done, info

        self.timesteps += 1
        if obs[0] != "Nothing happens.":
            self.success_action_history.append(
                str((f"step {self.success_action_timesteps}", action))
            )
            self.success_action_timesteps += 1
            # goal success rate computation
            self.compute_goal_reward(self.parse_goal, action)

        ###################################################

        reward = int(info["won"][0])
        done = done[0]
        info = {
            "obs": state,
            "obs_full": state_full,
            "task": self.parse_goal,
            "kg": self.kg.clone(),
            "expert_plan": info["extra.expert_plan"][0],
            "success": bool(reward),
            "action_success": obs[0] != "Nothing happens.",
            "history": (
                ", ".join(self.success_action_history).replace("'", "")
                if len(self.success_action_history) > 0
                else "No action history."
            ),
            "goal_reward": sum(self.goal_rewards) / len(self.goal_rewards),
            "goal_info": self.goal_rewards,
            "can_list": info["admissible_commands"][0],
        }

        return obs, reward, done, info

    def reset(self):
        self.timesteps = 0
        self.history = []
        self.success_action_history = []
        self.success_action_timesteps = 0
        self.goal_rewards = 0

        obs, info = self.env.reset()
        self.goal = obs[0].split("Your task is to:")[-1].strip(" .")

        # add character inside room
        self.agent_obs = []
        scene = int(info["extra.gamefile"][0].split("/")[-3].split("-")[-1]) // 100
        scene = self.scene_dict[scene]
        self.agent_obs.append(("character", "inside", scene))

        self.parse_goal = self._parse_goal(self.goal)
        parse_obs, parse_obs_full, edges = self._parse_obs(
            info["facts"][0], self.parse_goal
        )

        # set goal conditions
        self.goal_conditions = self._get_goal_conditions(self.parse_goal)
        self.goal_rewards = [0 for _ in self.goal_conditions]

        # append agent obs
        if len(self.agent_obs) > 0:
            agent_str = []
            for agent_obs in self.agent_obs:
                agent_str.append(str(agent_obs))
            parse_obs = " ".join(agent_str).replace("'", "") + ", " + parse_obs
            parse_obs_full = (
                " ".join(agent_str).replace("'", "") + ", " + parse_obs_full
            )

        obs = "You are in the middle of a room. Looking quickly around you, \n"
        obs += parse_obs
        obs += f"\nYour task is to: {self.goal}"

        info = {
            "obs": parse_obs,
            "obs_full": parse_obs_full,
            "task_type": self._parse_task_type(info["extra.gamefile"][0]),
            "task": self.parse_goal,
            "kg": self.kg.clone(),
            "expert_plan": info["extra.expert_plan"][0],
            "history": "No action history.",
            "goal_reward": sum(self.goal_rewards) / len(self.goal_rewards),
            "goal_info": self.goal_rewards,
            "can_list": info["admissible_commands"][0],
        }
        return obs, info

    def _parse_goal(self, goal):
        goal = goal.split(" ")

        if goal[0] == "look":  # look at alarmclock under the desklamp
            return f"look at {goal[2]} under the {goal[-1]}"
        elif goal[0] == "examine":  # examine the newspaper with the desklamp
            return f"look at {goal[2]} under the {goal[-1]}"
        elif goal[0] == "heat":  # heat some mug and put it in coffemachine
            return f"heat some {goal[2]} in microwave and put it in/on {goal[-1]}"
        elif goal[2] == "hot":  # put a hot tomato in diningtable
            return f"heat some {goal[3]} in microwave and put it in/on {goal[-1]}"
        elif goal[0] == "find" and goal[1] == "two":  # put two newspaper in sofa
            return f"find two {goal[2]} and put them in/on {goal[-1]}"
        elif (
            goal[0] == "put" and goal[1] == "two"
        ):  # find two pan and put them in countertop
            return f"find two {goal[2]} and put them in/on {goal[-1]}"
        elif goal[0] == "cool":  # cool some potato and put it in garbagecan
            return f"cool some {goal[2]} in fridge and put it in/on {goal[-1]}"
        elif goal[2] == "cool":  # put a cool bread in countertop
            return f"cool some {goal[3]} in fridge and put it in/on {goal[-1]}"
        elif goal[0] == "put" and goal[1] == "some":  # put some handtowel on garbagecan
            return f"put a {goal[2]} in/on {goal[-1]}"
        elif (
            goal[0] == "put" and goal[1] == "a" and goal[2] != "clean"
        ):  # put a pen in shelf
            return f"put a {goal[2]} in/on {goal[-1]}"
        elif goal[0] == "clean":  # clean some kettle and put it in cabinet
            return f"clean some {goal[2]} in sinkbasin and put it in/on {goal[-1]}"
        elif goal[2] == "clean":  # put a clean bowl in fridge
            return f"clean some {goal[3]} in sinkbasin and put it in/on {goal[-1]}"
        else:
            raise NotImplementedError

    def _get_goal_conditions(self, goal):
        goal = goal.split(" ")

        if goal[0] == "heat":
            goal_conditions = ["take", "heat", "put"]
        elif goal[0] == "find" and goal[1] == "two":
            goal_conditions = ["take", "take", "put", "put"]
        elif goal[0] == "put" and goal[1] == "a":
            goal_conditions = ["take", "put"]
        elif goal[0] == "clean":
            goal_conditions = ["take", "clean", "put"]

        return goal_conditions

    def compute_goal_reward(self, goal, action):
        goal = goal.split(" ")
        action = action.split(" ")

        idx = None
        if goal[0] == "heat":
            target_obj = goal[2]
            aid_obj = goal[4]  # microwave
            recep_obj = goal[-1]

            if action[0] == "take" and action[1] == target_obj:
                if self.goal_rewards[0] == 0:
                    self.goal_rewards[0] = 1
                    idx = 0
            if action[0] == "heat" and action[1] == target_obj and action[4] == aid_obj:
                if self.goal_rewards[1] == 0:
                    self.goal_rewards[1] = 1
                    idx = 1
            if (
                action[0] == "put"
                and action[1] == target_obj
                and action[4] == recep_obj
            ):
                if self.goal_rewards[2] == 0:
                    self.goal_rewards[2] = 1
                    idx = 2
        elif goal[0] == "find" and goal[1] == "two":
            target_obj = goal[2]
            recep_obj = goal[-1]

            if action[0] == "take" and action[1] == target_obj:
                if self.goal_rewards[0] == 0:
                    self.goal_rewards[0] = 1
                    idx = 0
                else:
                    self.goal_rewards[1] = 1
                    idx = 1
            if (
                action[0] == "put"
                and action[1] == target_obj
                and action[4] == recep_obj
            ):
                if self.goal_rewards[2] == 0:
                    self.goal_rewards[2] = 1
                    idx = 2
                else:
                    self.goal_rewards[3] = 1
                    idx = 3
        elif goal[0] == "put" and goal[1] == "a":
            target_obj = goal[2]
            recep_obj = goal[-1]

            if action[0] == "take" and action[1] == target_obj:
                if self.goal_rewards[0] == 0:
                    self.goal_rewards[0] = 1
                    idx = 0
            if (
                action[0] == "put"
                and action[1] == target_obj
                and action[4] == recep_obj
            ):
                if self.goal_rewards[1] == 0:
                    self.goal_rewards[1] = 1
                    idx = 1
        elif goal[0] == "clean":
            target_obj = goal[2]
            aid_obj = goal[4]  # microwave
            recep_obj = goal[-1]

            if action[0] == "take" and action[1] == target_obj:
                if self.goal_rewards[0] == 0:
                    self.goal_rewards[0] = 1
                    idx = 0
            if (
                action[0] == "clean"
                and action[1] == target_obj
                and action[4] == aid_obj
            ):
                if self.goal_rewards[1] == 0:
                    self.goal_rewards[1] = 1
                    idx = 1
            if (
                action[0] == "put"
                and action[1] == target_obj
                and action[4] == recep_obj
            ):
                if self.goal_rewards[2] == 0:
                    self.goal_rewards[2] = 1
                    idx = 2
        else:
            raise NotImplementedError

        if idx is not None:
            print(f">> Goal Success at {idx}: {self.goal_rewards}")

    def _parse_task_type(self, raw_task):
        name = "/".join(raw_task.split("/")[-3:-1])
        prefixes = {
            "pick_and_place": "put",
            "pick_clean_then_place": "clean",
            "pick_heat_then_place": "heat",
            "pick_cool_then_place": "cool",
            "look_at_obj": "examine",
            "pick_two_obj": "puttwo",
        }
        for prefix in prefixes:
            if name.startswith(prefix):
                return prefixes[prefix]

    def _parse_obs(self, obs, goal):
        props = {}
        for prop in obs:
            if prop.name not in props:
                props[prop.name] = set()

            name, arguments = prop.name, prop.arguments
            arguments = tuple([arg.name for arg in arguments])

            props[name].add(arguments)

        edges = []
        nouns = self._collect_nouns(props)

        visible_objects = self._collect_visible_objects(props)
        edges += self._collect_edges(props, visible_objects)

        introduced_objects = set()
        for rel in edges:
            if rel[0] in visible_objects:
                introduced_objects.add(rel[0])
            if rel[2] in visible_objects:
                introduced_objects.add(rel[2])

        for obj in visible_objects - introduced_objects:
            edges.append((obj, "is", "visible"))

        # remove the closed & visible state ...
        new_edges = []
        for edge in edges:
            if edge[2] != "closed" and edge[2] != "open":
                new_edges.append(edge)
            else:
                edge = (edge[0], "is", "visible")
                if edge not in new_edges:
                    new_edges.append(edge)

        # remove holding object in visible
        tag = False
        for agent_obs in self.agent_obs:
            if "hold" == agent_obs[1]:
                hold_object = agent_obs[2].split(" ")
                hold_object = f"{hold_object[-2]} {hold_object[-1]}"
                tag = True
        if tag:
            for edge in new_edges:
                if edge[2] == "visible" and edge[0] == hold_object:
                    new_edges.remove(edge)
                    break

        self.kg = AlfredKG(new_edges, nouns)
        obs = self.kg.retrieve(
            [goal], self.emb_fn, num_edges=self.num_topk_edge, replace=False
        )
        obs_full = self.kg.retrieve(
            [goal], self.emb_fn, num_edges=self.num_topk_edge * 2, replace=False
        )
        return obs, obs_full, new_edges

    def _collect_visible_objects(self, props):
        receptacles = set([p[0] for p in props.get("receptacletype", set())])
        closed_receptacles = self._collect_closed_receptacles(props)

        visible_objects = set()
        objects = set([p[0] for p in props.get("objecttype", set())])
        inreceptacles = props.get("inreceptacle", set())

        for obj in objects:
            candidates = set(product([obj], closed_receptacles))
            if len(candidates & inreceptacles) == 0:
                visible_objects.add(obj)

        visible_objects = visible_objects | receptacles
        return visible_objects

    def _collect_edges(self, props, visible_objects):
        edges = []

        receptacles = set([p[0] for p in props.get("receptacletype", set())])
        openable = set([p[0] for p in props.get("openable", set())])
        closed_receptacles = self._collect_closed_receptacles(props)
        open_receptacles = (receptacles & openable) - closed_receptacles
        for recep in open_receptacles:
            edges.append((recep, "is", "open"))
        for recep in closed_receptacles:
            edges.append((recep, "is", "closed"))

        inreceptacles = props.get("inreceptacle", set())
        candidates = set(product(visible_objects, receptacles)) & inreceptacles
        for obj, recep in candidates:
            edges.append((obj, "inside", recep))

        return edges

    def _collect_nouns(self, props):
        objects = set([p[0].split()[0] for p in props.get("objecttype", set())])
        receptacles = set([p[0].split()[0] for p in props.get("receptacletype", set())])
        return objects | receptacles

    def _collect_closed_receptacles(self, props):
        receptacles = set([p[0] for p in props.get("receptacletype", set())])
        openable = set([p[0] for p in props.get("openable", set())])
        opened = set([p[0] for p in props.get("opened", set())])

        closed_receptacles = (receptacles & openable) - opened
        return closed_receptacles


############################################################################################################


class AlfredKG:
    def __init__(self, edges, nouns):
        self.edges = edges
        self.nouns = nouns

    def clone(self):
        edges = copy.deepcopy(self.edges)
        nouns = copy.deepcopy(self.nouns)
        return AlfredKG(edges, nouns)

    def retrieve(
        self,
        instructions,
        embedding_fns,
        num_edges=50,
        replace=False,
    ):
        if len(self.edges) == 0:
            return " "

        edge_without_number = []
        for edge in self.edges:
            new_edge = []
            for token in edge:
                new_edge.append(token.split()[0])
            edge_without_number.append(tuple(new_edge))

        edge_str = []
        for edge in edge_without_number:
            edge_str.append(f"({', '.join(edge).lower()})")

        bm25 = BM25Okapi(edge_without_number)
        doc_scores = []
        for instruction in instructions:
            doc_tokens = [t for t in instruction.split() if t in self.nouns]
            for i, doc_token in enumerate(
                doc_tokens
            ):  # 0. object, 1. aid object, 2. recep
                # give weight more on .. object > aid object > recep
                doc_scores.append(bm25.get_scores(doc_tokens) * (1 + 5.0 * (3 - i)))
        doc_scores = [np.sum(doc_scores, axis=0)]


        edge_embedding = embedding_fns(edge_str)
        instruction_embedding = embedding_fns(instructions)

        similarity = (
            np.array(doc_scores)
            + np.array(self._compute_similarity(instruction_embedding, edge_embedding))
            * 1e-6
        )
        edge_retrieval_prob = np.max(similarity, axis=0) / np.sum(
            (np.max(similarity, axis=0))
        )
        edge_str = []
        for edge in self.edges:
            edge_str.append(f"({', '.join(edge).lower()})")
        edge_str = list(
            np.random.choice(
                edge_str,
                p=edge_retrieval_prob,
                size=min(len(edge_str), num_edges),
                replace=replace,
            )
        )

        edge_str = self._remove_duplicates(edge_str)
        return ", ".join(edge_str).lower()

    def _compute_similarity(self, instruction, kg_triples):
        cos_sim_results = cos_sim(
            instruction.detach().numpy(), kg_triples.detach().numpy()
        )
        return cos_sim_results.cpu().numpy()

    def _remove_duplicates(self, my_list):
        seen = set()
        result = []
        for item in my_list:
            if item not in seen:
                result.append(item)
                seen.add(item)
        return result
