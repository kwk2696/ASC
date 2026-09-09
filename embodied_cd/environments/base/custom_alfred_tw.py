import os
import os.path as osp
import json

import textworld
from alfworld.agents.environment.alfred_tw_env import (
    AlfredTWEnv,
    AlfredDemangler,
    AlfredInfos,
    AlfredExpert,
)


TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


class CustomAlfredTWEnv(AlfredTWEnv):
    def __init__(self, config, train_eval="train"):
        self.config = config
        self.train_eval = train_eval
        self.collect_game_files()

    def collect_game_files(self, verbose=False):
        def log(info):
            if verbose:
                print(info)

        self.game_files = []

        if self.train_eval == "train":
            data_path = os.path.expandvars(self.config["dataset"]["data_path"])
            data_paths = list(os.walk(data_path, topdown=False))
        elif self.train_eval == "eval_in_distribution":
            data_path = os.path.expandvars(self.config["dataset"]["eval_id_data_path"])
            data_paths = list(os.walk(data_path, topdown=False))
        elif self.train_eval == "eval_out_of_distribution":
            data_path = os.path.expandvars(self.config["dataset"]["eval_ood_data_path"])
            data_paths = list(os.walk(data_path, topdown=False))
        else:
            data_paths = self._collect_custom_split(self.train_eval)

        log("Collecting solvable games...")

        # get task types
        assert len(self.config["env"]["task_types"]) > 0
        task_types = []
        for tt_id in self.config["env"]["task_types"]:
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        count = 0
        for root, dirs, files in data_paths:
            if "traj_data.json" in files:
                count += 1

                # Filenames
                json_path = os.path.join(root, "traj_data.json")
                game_file_path = os.path.join(root, "game.tw-pddl")

                if "movable" in root or "Sliced" in root:
                    log("Movable & slice trajs not supported %s" % (root))
                    continue

                # Get goal description
                with open(json_path, "r") as f:
                    traj_data = json.load(f)

                # Check for any task_type constraints
                if not traj_data["task_type"] in task_types:
                    log("Skipping task type")
                    continue

                # Check if a game file exists
                if not os.path.exists(game_file_path):
                    log(f"Skipping missing game! {game_file_path}")
                    continue

                with open(game_file_path, "r") as f:
                    gamedata = json.load(f)

                # Check if previously checked if solvable
                if "solvable" not in gamedata:
                    print(f"-> Skipping missing solvable key! {game_file_path}")
                    continue

                if not gamedata["solvable"]:
                    log("Skipping known %s, unsolvable game!" % game_file_path)
                    continue

                # Add to game file list
                self.game_files.append(game_file_path)

        self.num_games = len(self.game_files)

        if self.train_eval == "train":
            num_train_games = (
                self.config["dataset"]["num_train_games"]
                if self.config["dataset"]["num_train_games"] > 0
                else len(self.game_files)
            )
            self.game_files = self.game_files[:num_train_games]
            self.num_games = len(self.game_files)
        else:
            num_eval_games = (
                self.config["dataset"]["num_eval_games"]
                if self.config["dataset"]["num_eval_games"] > 0
                else len(self.game_files)
            )
            self.game_files = self.game_files[:num_eval_games]
            self.num_games = len(self.game_files)

    def _collect_custom_split(self, json_path):
        filename = osp.basename(json_path).split(".")[0]
        split = filename if "valid" in filename else "train"
        data_path = self.config["dataset"]["data_path"].replace("train", split)

        with open(json_path, "r") as f:
            data = json.load(f)
            if "valid" in filename:
                data = [d["task"] for d in data]
            else:
                data = [d["task"]["task"] for d in data]

        data_paths = []
        for d in data:
            root_path = osp.join(data_path, d)
            if osp.isdir(root_path):
                files = os.listdir(root_path)
                data_paths.append((root_path, None, files))

        return data_paths

    def init_env(self, batch_size, expert_type = "planner"):
        domain_randomization = self.config["env"]["domain_randomization"]
        if self.train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=False) #domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        # Register a new Gym environment.
        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"], facts=True
        )
        expert_type = expert_type # planner, handcoded
        max_nb_steps_per_episode = self.config["dagger"]["training"][
            "max_nb_steps_per_episode"
        ]

        wrappers.append(AlfredExpert(expert_type=expert_type))
        request_infos.extras.append("expert_plan")

        env_id = textworld.gym.register_games(
            self.game_files,
            request_infos,
            batch_size=batch_size,
            asynchronous=True,
            max_episode_steps=max_nb_steps_per_episode,
            wrappers=wrappers,
        )
        # Launch Gym environment.
        env = textworld.gym.make(env_id)
        return env
