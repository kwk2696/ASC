import json
from collections import deque

from vh_dataset.dataset.virtualhome import KG, make_embedding_fn
from virtualhome.simulation.environment.unity_environment import UnityEnvironment

from embodied_cd.environments.base import BaseEnvironment
from embodied_cd.common.print_utils import print_warn


TASKS_SET = [
    # turn on
    "Turn on tv",
    "Turn on radio",
    "Turn on microwave",
    "Turn on stove",
    "Turn on computer",
    # open
    "Open cabinet",
    "Open dishwasher",
    "Open microwave",
    "Open stove",
    "Open fridge",
    # place on
    "Place apple on sofa",  # 10
    "Place book on coffeetable",
    "Place mug on coffeetable",
    "Place towel on sofa",
    "Place paper on bed",
    # put in
    "Put mug in microwave",  # 15
    "Put plate in dishwasher",
    "Put towel in closet",
    "Put apple in fridge",
    "Put paper in bookshelf",
    # "Place paper in cabinet",
    # ========================#
    # turn on
    "Turn on coffeemaker",
    "Turn on dishwasher",
    "Turn on toaster",
    # open
    "Open coffeepot",
    "Open toilet",
    "",
    # place on
    "Place chips on sofa",
    "Place book on desk",
    "Place bananas on bed",
    # put in
    "Put bananas in fridge",
    "Put book in closet",
    "",
    # composite_two~
    "Open cabinet and open dishwasher",
    "Turn on tv and place chips on sofa",
    "Place paper on desk and turn on computer",
    "Put mug in microwave and place plate on microwave",
    "Open fridge and put apple in fridge",
    "Place chips on sofa and turn on tv",
    "Place book on coffeetable and turn on tv",
    "Put towel in closet and open cabinet",
    "Put apple in fridge and open cabinet",
    "Place towel on sofa and place apple on sofa",
    # composite_three
    "Turn on microwave and turn on radio and turn on tv",
    "Open cabinet and open dishwasher and open fridge",
    "Place book on desk and turn on computer and turn on tv",
    "Put apple in fridge and turn on stove and place paper on bed",
    "Turn on computer and turn on radio and turn on dishwasher",
    "Place apple on sofa and place book on coffeetable and place towel on sofa",
    "Open toilet and open cabinet and open stove",
    "Put paper in bookshelf and put book in closet and put towel in closet",
    "Turn on tv and place paper on bed and place book on desk",
    "Open dishwasher and turn on computer and put mug in microwave",
]

SUCCESS_CONDITIONS = [
    # turn on
    ("tv", "is", "on"),
    ("radio", "is", "on"),
    ("microwave", "is", "on"),
    ("stove", "is", "on"),
    ("computer", "is", "on"),
    # open
    ("cabinet", "is", "open"),
    ("dishwasher", "is", "open"),
    ("microwave", "is", "open"),
    ("stove", "is", "open"),
    ("fridge", "is", "open"),
    # place on
    ("apple", "on", "sofa"),
    ("book", "on", "coffeetable"),
    ("mug", "on", "coffeetable"),
    ("towel", "on", "sofa"),
    ("paper", "on", "bed"),
    # put in
    ("mug", "inside", "microwave"),
    ("plate", "inside", "dishwasher"),
    ("towel", "inside", "closet"),
    ("apple", "inside", "fridge"),
    ("paper", "inside", "bookshelf"),
    # ========================#
    # turn on
    ("coffeemaker", "is", "on"),
    ("dishwasher", "is", "on"),
    ("toaster", "is", "on"),
    # open
    ("coffeepot", "is", "open"),
    ("toilet", "is", "open"),
    (),
    # place on
    ("chips", "on", "sofa"),
    ("book", "on", "desk"),
    ("bananas", "on", "bed"),
    # put in
    ("bananas", "inside", "fridge"),
    ("book", "inside", "closet"),
    (),
    # composite_two
    [("cabinet", "is", "open"), ("dishwasher", "is", "open")],
    [("tv", "is", "on"), ("chips", "on", "sofa")],
    [("paper", "on", "desk"), ("computer", "is", "on")],
    [("mug", "inside", "microwave"), ("plate", "on", "microwave")],
    [("fridge", "is", "open"), ("apple", "inside", "fridge")],
    [("chips", "on", "sofa"), ("tv", "is", "on")],
    [("book", "on", "coffeetable"), ("tv", "is", "on")],
    [("towel", "inside", "closet"), ("cabinet", "is", "open")],
    [("apple", "inside", "fridge"), ("cabinet", "is", "open")],
    [("towel", "on", "sofa"), ("apple", "on", "sofa")],
    # composite_three
    [("microwave", "is", "on"), ("radio", "is", "on"), ("tv", "is", "on")],
    [
        ("cabinet", "is", "open"),
        ("dishwasher", "is", "open"),
        ("fridge", "is", "open"),
    ],
    [("book", "on", "desk"), ("computer", "is", "on"), ("tv", "is", "on")],
    [("apple", "inside", "fridge"), ("stove", "is", "on"), ("paper", "on", "bed")],
    [("computer", "is", "on"), ("radio", "is", "on"), ("dishwasher", "is", "on")],
    [
        ("apple", "on", "sofa"),
        ("book", "on", "coffeetable"),
        ("towel", "on", "sofa"),
    ],
    [("toilet", "is", "open"), ("cabinet", "is", "open"), ("stove", "is", "open")],
    [
        ("paper", "inside", "bookshelf"),
        ("book", "inside", "closet"),
        ("towel", "inside", "closet"),
    ],
    [("tv", "is", "on"), ("paper", "on", "bed"), ("book", "on", "desk")],
    [
        ("dishwasher", "is", "open"),
        ("computer", "is", "on"),
        ("mug", "inside", "microwave"),
    ],
]


class VirtualHomeEnv(BaseEnvironment):
    name = "virtualhome"

    def __init__(
        self,
        num_topk_edge: int = 12,
        easier_room_navigation=False,
        repetitive_action_patience=3,
        ip: str = "127.0.0.1",
        port: str = 8080,
        seed: int = 123,
        use_unity: bool = False,
    ):
        super().__init__()

        self.env = UnityEnvironment(url=ip, base_port=port, seed=seed)
        self.valid_env_id = self.env.valid_env_id

        self.kg = None
        self.task_id = None
        self.env_id = None
        self.timestep = 0
        self._holding = None

        self.emb_fn = make_embedding_fn()
        self.num_topk_edge = num_topk_edge

        self.repetitive_action_patience = repetitive_action_patience
        self.easier_room_navigation = easier_room_navigation

        self.use_unity = use_unity
        self.subgoal_success = None

    def step(self, action):
        action = action.lower().strip()
        ## replace switch to switchon
        action = action.replace("switch ", "switchon ")
        ## replace puton to put
        action = action.replace("puton ", "put ")
        if "place" in action and "on" in action:
            _action = action.split(" ")
            action = f"put {_action[-1]}"
        if "put" in action and "in" in action:
            _action = action.split(" ")
            action = f"putin {_action[-1]}"
        ########
        if not self._is_valid_action(action):
            obs = f"{action} is not a valid action."
            done = True
            info = {
                "success": False,
                "subgoal_success": self.subgoal_success,
                "task": self._parse_task_type(TASKS_SET[self.task_id]),
                "goal_reward": sum(self.goal_rewards) / len(self.goal_rewards),
                "goal_info": self.goal_rewards,
            }
            return obs, 0, done, info

        # To make agent easier to navigate to target room without knowing the room connections
        room_navigation = False
        if self.easier_room_navigation and action.startswith("walk"):
            target = action.split(" ")[-1]
            if target in ["livingroom", "kitchen", "bedroom", "bathroom"]:
                room_navigation = True
                self._navigate_adjacent_room(target)

        self.history.append(str((f"step {self.timestep+1}", action)))

        try:
            action_history = [h.split("', '")[1][:-2] for h in self.history][
                -self.repetitive_action_patience :
            ]
        except (IndexError, ValueError) as e:
            print_warn(f"[VirtualHomeEnv] could not parse the action history ({e!r}); repetition check skipped")
            action_history = []

        if (
            len(action_history) == self.repetitive_action_patience
            and len(set(action_history)) == 1
        ):
            obs = f"Lost in repetitive action {action}."
            done = True
            info = {
                "success": False,
                "subgoal_success": self.subgoal_success,
                "task": self._parse_task_type(TASKS_SET[self.task_id]),
                "goal_reward": sum(self.goal_rewards) / len(self.goal_rewards),
                "goal_info": self.goal_rewards,
            }
            return obs, 0, done, info

        obs, reward, done, info = self.env.step(action)
        if info["success"]:
            self.success_action_history.append(
                str((f"step {self.success_action_timesteps}", action))
            )
            self.success_action_timesteps += 1

        if self.use_unity:
            return obs, reward, done, info

        self.compute_goal_reward()
        raw_obs = obs
        self.timestep += 1

        # new kg !!
        self.kg = KG(self.env.get_position_graph())
        self.kg.add(obs["visible_graph"], self.timestep, use_refinement=True)
        self.kg.add(obs["agent_graph"], self.timestep, use_refinement=True)

        if not room_navigation and not info["success"]:
            obs = "Nothing seems to be happened."
            # # # # 0110 let nothing happened to be the end
            done = False
        else:
            obs = self._parse_obs(self.kg, TASKS_SET[self.task_id], prev_action=action)

        self.subgoal_success = info["condwise_success"]
        info["subgoal_success"] = self.subgoal_success

        info["raw_obs"] = raw_obs
        info["task_type"] = self._parse_task_type(TASKS_SET[self.task_id])
        info["task"] = TASKS_SET[self.task_id]
        info["kg"] = self.kg.clone()
        info["state"] = self.kg.retrieve(
            [TASKS_SET[self.task_id]],
            embedding_fns=self.emb_fn,
            num_edges=self.num_topk_edge,
        )
        info["history"] = (
            ", ".join(self.success_action_history).replace("'", "")
            if len(self.success_action_history) > 0
            else "No action history."
        )
        info["goal_reward"] = sum(self.goal_rewards) / len(self.goal_rewards)
        info["goal_info"] = self.goal_rewards
        return obs, reward, done, info

    def reset(self, task_id=0, env_id=None, init_rooms=None):
        self.task_id = task_id
        self.env_id = env_id
        self.timestep = 0
        self._holding = None
        self.history = []
        self.success_action_history = []
        self.success_action_timesteps = 0

        if env_id is None:
            env_id = self.env.valid_env_id[-1]
        if not isinstance(env_id, int):
            env_id = int(env_id)

        success_condition = SUCCESS_CONDITIONS[task_id]
        if not isinstance(success_condition, list):
            success_condition = [SUCCESS_CONDITIONS[task_id]]

        self.env.set_task(
            {
                "required_condition": success_condition,
                "prohibited_condition": [],
            }
        )
        obs = self.env.reset(environment_id=env_id, init_rooms=init_rooms)
        if self.use_unity:
            return obs
        raw_obs = obs
        if not isinstance(obs, dict):
            return None, None

        self.subgoal_success = [False] * len(success_condition)

        # new kg !!
        self.kg = KG(self.env.get_position_graph())
        self.kg.add(obs["visible_graph"], 0, use_refinement=True)
        self.kg.add(obs["agent_graph"], 0, use_refinement=True)

        # set goal conditions
        self.goal_conditions = self._get_goal_conditions(TASKS_SET[task_id], raw_obs)
        self.goal_rewards = [0 for _ in self.goal_conditions]
        self.compute_goal_reward()

        obs = self._parse_obs(self.kg, TASKS_SET[task_id], prev_action="reset env")
        info = {
            "raw_obs": raw_obs,
            "task_type": self._parse_task_type(TASKS_SET[task_id]),
            "task": TASKS_SET[task_id],
            "kg": self.kg.clone(),
            "state": self.kg.retrieve(
                [TASKS_SET[task_id]],
                embedding_fns=self.emb_fn,
                num_edges=self.num_topk_edge,
            ),
            "history": "No action history.",
            "goal_reward": sum(self.goal_rewards) / len(self.goal_rewards),
            "goal_info": self.goal_rewards,
            "success": False,
            "subgoal_success": self.subgoal_success,
        }
        return obs, info

    def compute_goal_reward(self):
        entire_graph = self.env.get_graph(mode="triples")
        agent_graph = self.env.get_agent_graph(mode="triples")

        for i, goal in enumerate(self.goal_conditions):
            if goal in entire_graph["edges"] or goal in agent_graph["edges"]:
                if self.goal_rewards[i] == 0:
                    self.goal_rewards[i] = 1
                    print(f">> Goal Success For {goal} at {i}: {self.goal_rewards}")

    def _get_goal_conditions(self, instruction, obs):
        if "and" not in instruction:
            instruction_type = instruction.split()[0]

            goal_conditions = []
            if instruction_type == "Turn":
                obj = instruction.split()[2]
                obj_room = self.kg.search_obj_room(obj)

                goal_conditions.append(("character", "INSIDE", obj_room))
                goal_conditions.append((obj, "is", "ON"))
            elif instruction_type == "Open":
                obj = instruction.split()[1]
                obj_room = self.kg.search_obj_room(obj)

                goal_conditions.append(("character", "INSIDE", obj_room))
                goal_conditions.append((obj, "is", "OPEN"))
            elif instruction_type == "Place":
                obj = instruction.split()[1]
                target_obj = instruction.split()[3]
                obj_room = self.kg.search_obj_room(obj)
                target_obj_room = self.kg.search_obj_room(target_obj)

                goal_conditions.append(("character", "INSIDE", obj_room))
                goal_conditions.append(("character", "HOLD", obj))
                goal_conditions.append(("character", "INSIDE", target_obj_room))
                goal_conditions.append((obj, "ON", target_obj))
            elif instruction_type == "Put":
                obj = instruction.split()[1]
                target_obj = instruction.split()[3]
                obj_room = self.kg.search_obj_room(obj)
                target_obj_room = self.kg.search_obj_room(target_obj)

                goal_conditions.append(("character", "INSIDE", obj_room))
                goal_conditions.append(("character", "HOLD", obj))
                goal_conditions.append(("character", "INSIDE", target_obj_room))
                goal_conditions.append((obj, "INSIDE", target_obj))
        else:
            sub_tasks = [inst.strip().capitalize() for inst in instruction.split("and")]
            goal_conditions = []
            for sub_task in sub_tasks:
                if sub_task.startswith("Turn"):
                    obj = sub_task.split()[2]
                    obj_room = self.kg.search_obj_room(obj)

                    goal_conditions.append(("character", "INSIDE", obj_room))
                    goal_conditions.append((obj, "is", "ON"))
                elif sub_task.startswith("Open"):
                    obj = sub_task.split()[1]
                    obj_room = self.kg.search_obj_room(obj)

                    goal_conditions.append(("character", "INSIDE", obj_room))
                    goal_conditions.append((obj, "is", "OPEN"))
                elif sub_task.startswith("Place"):
                    obj = sub_task.split()[1]
                    target_obj = sub_task.split()[3]
                    obj_room = self.kg.search_obj_room(obj)
                    target_obj_room = self.kg.search_obj_room(target_obj)

                    goal_conditions.append(("character", "INSIDE", obj_room))
                    goal_conditions.append(("character", "HOLD", obj))
                    goal_conditions.append(("character", "INSIDE", target_obj_room))
                    goal_conditions.append((obj, "ON", target_obj))
                elif sub_task.startswith("Put"):
                    obj = sub_task.split()[1]
                    target_obj = sub_task.split()[3]
                    obj_room = self.kg.search_obj_room(obj)
                    target_obj_room = self.kg.search_obj_room(target_obj)

                    goal_conditions.append(("character", "INSIDE", obj_room))
                    goal_conditions.append(("character", "HOLD", obj))
                    goal_conditions.append(("character", "INSIDE", target_obj_room))
                    goal_conditions.append((obj, "INSIDE", target_obj))

        return goal_conditions

    def _parse_task_type(self, raw_task):
        task = raw_task.lower().replace(" ", "_")
        prefixes = {
            # turn on
            "turn_on_tv": "turn_on",
            "turn_on_radio": "turn_on",
            "turn_on_microwave": "turn_on",
            "turn_on_stove": "turn_on",
            "turn_on_computer": "turn_on",
            "turn_on_coffeemaker": "turn_on",
            "turn_on_dishwasher": "turn_on",
            # open
            "open_cabinet": "open",
            "open_dishwasher": "open",
            "open_microwave": "open",
            "open_stove": "open",
            "open_fridge": "open",
            "open_coffeepot": "open",
            "open_toilet": "open",
            # put
            "place_apple_on_sofa": "place_on",
            "place_book_on_coffeetable": "place_on",
            "place_mug_on_coffeetable": "place_on",
            "place_towel_on_sofa": "place_on",
            "place_paper_on_bed": "place_on",
            "place_chips_on_sofa": "place_on",
            "place_book_on_desk": "place_on",
            "place_bananas_on_bed": "place_on",
            # open put
            "put_mug_in_microwave": "put_in",
            "put_plate_in_dishwasher": "put_in",
            "put_towel_in_closet": "put_in",
            "put_apple_in_fridge": "put_in",
            "put_paper_in_bookshelf": "put_in",
            "put_bananas_in_fridge": "put_in",
            "put_book_in_closet": "put_in",
        }
        task_type = prefixes.get(task, "composite")
        return task_type

    def _parse_obs(self, obs, goal, prev_action):
        edges = []
        for edge in obs.edges:
            source, relation, target = str(edge).strip("()").split(", ")
            edges.append((source, relation, target))

        action, target = prev_action.split(" ")
        player_room = self._get_player_room(edges)

        new_obs = f"You are in the middle of a {player_room}.\n"
        new_obs += self.kg.retrieve(
            [goal], embedding_fns=self.emb_fn, num_edges=self.num_topk_edge
        )
        if action == "reset":
            new_obs += f"\nYour task is to: {goal}."

        return new_obs

    def _get_player_room(self, edges):
        player_room = None
        for edge in edges:
            source, relation, target = edge
            if source == "character" and relation == "inside":
                player_room = target
                break
        return player_room

    def _is_valid_action(self, action):
        avail_actions = ["walk", "open", "close", "switchon", "grab", "put", "putin"]
        with open(
            "externals/virtualhome/virtualhome/resources/properties_data_unity.json",
            "r",
        ) as f:
            avail_targets = set(
                [
                    "livingroom",
                    "kitchen",
                    "bedroom",
                    "bathroom",
                ]
            )
            for k in json.load(f).keys():
                avail_targets.add(k)
        with open(
            "externals/virtualhome/virtualhome/resources/class_name_equivalence.json",
            "r",
        ) as f:
            for k, v in json.load(f).items():
                avail_targets.add("".join(k.split("_")))
                avail_targets.update(["".join(e.split("_")) for e in v])

        tokens = action.split(" ")
        if len(tokens) != 2:
            return False

        action, target = tokens
        if action not in avail_actions:
            return False
        if target not in avail_targets:
            return False

        return True

    def _navigate_adjacent_room(self, target_room):
        edges = []
        room_adjacency = {
            "livingroom": [],
            "kitchen": [],
            "bedroom": [],
            "bathroom": [],
        }
        for edge in self.kg.edges:
            source, relation, target = str(edge).strip("()").split(", ")
            edges.append((source, relation, target))
            if relation == "adjacent":
                room_adjacency[source].append(target)
                room_adjacency[target].append(source)

        player_room = self._get_player_room(edges)
        if player_room == target_room:
            return

        def find_path(room_adjacency, start_room, target_room):
            queue = deque([[start_room]])
            visited = set()
            while queue:
                path = queue.popleft()
                current_room = path[-1]
                if current_room == target_room:
                    return path
                if current_room not in visited:
                    visited.add(current_room)
                    for neighbor in room_adjacency[current_room]:
                        new_path = list(path)
                        new_path.append(neighbor)
                        queue.append(new_path)
            return []

        path = find_path(room_adjacency, player_room, target_room)
        # Already adjacent
        if len(path) == 2:
            return

        for room in path[1:-1]:
            action = f"walk {room}"
            self.history.append(str((f"step {self.timestep+1}", action)))

            obs, _, _, _ = self.env.step(action)
            self.timestep += 1

            self.kg.add(obs["visible_graph"], self.timestep, use_refinement=True)
            self.kg.add(obs["agent_graph"], self.timestep, use_refinement=True)
        return
