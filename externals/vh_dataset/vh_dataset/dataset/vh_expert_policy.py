import random

from vh_dataset.dataset.knowledge_graph import KG


def get_random_room(obj_room, target_room, knowledge_graph):
    rooms = ["kitchen", "livingroom", "bathroom", "bedroom"]
    if obj_room is not None:
        rooms.remove(obj_room)
    if target_room is not None:
        rooms.remove(target_room)

    target_adj_room = []
    agent_adj_room = []
    for room in rooms:
        if knowledge_graph.search_adjacent_obj((room, "adjacent", target_room)):
            target_adj_room.append(room)
        if knowledge_graph.search_adjacent_obj((obj_room, "adjacent", room)):
            agent_adj_room.append(room)

    if len(agent_adj_room) == 1:
        togo_room = agent_adj_room[0]
    else:
        togo_room = []
        for adj_room in agent_adj_room:
            if adj_room in target_adj_room:
                togo_room.append(adj_room)
        togo_room = random.choice(togo_room)
    return togo_room


class ExpertPolicy(object):
    def __init__(self, instruction, pos_kg):
        self.instruction = instruction
        self.instruction_type = instruction.split()[0]
        self.knowledge_graph = KG(pos_kg)

    def predict(self, obs=None, timesteps=0, prev_action=None):
        if obs is not None:
            self.knowledge_graph.add(
                obs["visible_graph"], timesteps, use_refinement=True
            )
            self.knowledge_graph.add(obs["agent_graph"], timesteps, use_refinement=True)
        if self.instruction_type == "Place":
            obj = self.instruction.split()[1]
            target_obj = self.instruction.split()[3]
            obj_room = self.knowledge_graph.search_obj_room(obj)
            target_obj_room = self.knowledge_graph.search_obj_room(target_obj)
            agent_room = self.knowledge_graph.search_obj_room("character")
            if self.knowledge_graph.search_edge(("character", "HOLD", obj)) is not None:
                if agent_room != target_obj_room:
                    if self.knowledge_graph.search_adjacent_obj(
                        (agent_room, "adjacent", target_obj_room)
                    ):
                        return f"walk {target_obj_room}"
                    else:
                        return f"walk {get_random_room(agent_room, target_obj_room, self.knowledge_graph)}"
                else:
                    if prev_action == f"walk {target_obj}":
                        return f"place {obj} on {target_obj}"
                    else:
                        return f"walk {target_obj}"
            if obj_room != agent_room:
                if self.knowledge_graph.search_adjacent_obj(
                    (agent_room, "adjacent", obj_room)
                ):
                    return f"walk {obj_room}"
                else:
                    return f"walk {get_random_room(agent_room, obj_room, self.knowledge_graph)}"
            else:
                if prev_action == f"walk {obj}":
                    return f"grab {obj}"
                else:
                    return f"walk {obj}"

        elif self.instruction_type == "Turn":
            obj = self.instruction.split()[2]
            obj_room = self.knowledge_graph.search_obj_room(obj)
            agent_room = self.knowledge_graph.search_obj_room("character")
            if obj_room != agent_room:
                if self.knowledge_graph.search_adjacent_obj(
                    (agent_room, "adjacent", obj_room)
                ):
                    return f"walk {obj_room}"
                else:
                    return f"walk {get_random_room(agent_room, obj_room, self.knowledge_graph)}"
            else:
                if not self.knowledge_graph.search_close_obj(obj):
                    return f"walk {obj}"
                else:
                    return f"switch {obj}"

        elif self.instruction_type == "Open":
            obj = self.instruction.split()[1]
            obj_room = self.knowledge_graph.search_obj_room(obj)
            agent_room = self.knowledge_graph.search_obj_room("character")
            if obj_room != agent_room:
                if self.knowledge_graph.search_adjacent_obj(
                    (agent_room, "adjacent", obj_room)
                ):
                    return f"walk {obj_room}"
                else:
                    return f"walk {get_random_room(agent_room, obj_room, self.knowledge_graph)}"
            else:
                if not self.knowledge_graph.search_close_obj(obj):
                    return f"walk {obj}"
                else:
                    return f"open {obj}"

        elif self.instruction_type == "Put":
            obj = self.instruction.split()[1]
            target_obj = self.instruction.split()[3]
            obj_room = self.knowledge_graph.search_obj_room(obj)
            target_obj_room = self.knowledge_graph.search_obj_room(target_obj)
            agent_room = self.knowledge_graph.search_obj_room("character")
            if self.knowledge_graph.search_edge(("character", "HOLD", obj)) is not None:
                if agent_room != target_obj_room:
                    if self.knowledge_graph.search_adjacent_obj(
                        (agent_room, "adjacent", target_obj_room)
                    ):
                        return f"walk {target_obj_room}"
                    else:
                        return f"walk {get_random_room(agent_room, target_obj_room, self.knowledge_graph)}"
                else:
                    if prev_action == f"walk {target_obj}":
                        if (
                            self.knowledge_graph.search_obj_closed(target_obj)
                            and target_obj != "closet"
                        ):
                            return f"open {target_obj}"
                        else:
                            return f"put {obj} in {target_obj}"
                    elif prev_action == f"open {target_obj}":
                        return f"put {obj} in {target_obj}"
                    else:
                        return f"walk {target_obj}"
            if obj_room != agent_room:
                if self.knowledge_graph.search_adjacent_obj(
                    (agent_room, "adjacent", obj_room)
                ):
                    return f"walk {obj_room}"
                else:
                    return f"walk {get_random_room(agent_room, obj_room, self.knowledge_graph)}"
            else:
                if prev_action == f"walk {obj}":
                    return f"grab {obj}"
                else:
                    return f"walk {obj}"
