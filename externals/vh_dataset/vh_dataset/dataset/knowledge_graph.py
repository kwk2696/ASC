import random
import numpy as np
import copy

from enum import Enum
from typing import Union
from rank_bm25 import BM25Okapi
from sentence_transformers.util import cos_sim


RelationList = [
    "on",
    "inside",
    "close",
    "hold",
    "is",
    "adjacent",
    "left",
    "right",
    "straight",
]
StateList = [
    "closed",
    "open",
    "on",
    "off",
    "plugin",
    "plugout",
    "dirty",
    "clean",
    "called",
    "quite",
    "waiting",
    "delivered",
    "packed",
]
RoomList = ["kitchen", "livingroom", "bathroom", "bedroom"]


def compute_similarity(instruction, kg_triples):
    cos_sim_results = cos_sim(instruction.detach().numpy(), kg_triples.detach().numpy())
    return cos_sim_results.cpu().numpy()


def OPPOSITE(node):
    if node == State.CLOSED:
        return State.OPEN
    if node == State.OPEN:
        return State.CLOSED
    if node == State.ON:
        return State.OFF
    if node == State.OFF:
        return State.ON
    if node == State.PLUGGED_IN:
        return State.PLUGGED_OUT
    if node == State.PLUGGED_OUT:
        return State.PLUGGED_IN
    if node == State.DIRTY:
        return State.CLEAN
    if node == State.CLEAN:
        return State.DIRTY


class Relation(Enum):
    ON = 0
    INSIDE = 1
    CLOSE = 2
    HOLD = 3
    IS = 4
    ADJACENT = 5

    # For CARLA
    LEFT = 6
    RIGHT = 7
    STRAIGHT = 8

    @classmethod
    def all(cls):
        return list(Relation)


class State(Enum):
    CLOSED = 0
    OPEN = 1
    ON = 2
    OFF = 3
    PLUGGED_IN = 4
    PLUGGED_OUT = 5
    DIRTY = 5
    CLEAN = 6

    CALLED = 7
    QUITE = 8
    WAITING = 9
    DELIVERED = 10
    PACKED = 11


class Room(Enum):
    KITCHEN = 0
    LIVINGROOM = 1
    BATHROOM = 2
    BEDROOM = 3


class GraphNode(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self.id == other.id and self.name == other.name:
            return True
        else:
            return False

    @staticmethod
    def from_dict(d):
        return GraphNode(d["id"], d["class_name"])


def remove_duplicates(my_list):
    seen = set()
    result = []
    for item in my_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def node_rel2str(node: Union[GraphNode, State, Room, Relation]):
    if isinstance(node, GraphNode):
        return str(node)
    elif isinstance(node, State):
        return StateList[node.value]
    elif isinstance(node, Room):
        return RoomList[node.value]
    elif isinstance(node, Relation):
        return RelationList[node.value]
    else:
        raise NotImplementedError()


class GraphEdge(object):
    def __init__(
        self,
        from_node: Union[GraphNode, Room],
        relation: Relation,
        to_node: Union[GraphNode, State, Room],
        timesteps: int,
    ):
        self.from_node = from_node
        self.relation = relation
        self.to_node = to_node
        self.timesteps = timesteps

    def __eq__(self, other):
        if (
            self.from_node == other.from_node
            and self.relation == other.relation
            and self.to_node == other.to_node
        ):
            return True
        else:
            return False

    def conflict(self, other):
        if (
            self.timesteps < other.timesteps
            and node_rel2str(self.from_node) == "character"
        ):
            return True
        if self.timesteps < other.timesteps and self.relation == Relation.ADJACENT:
            return True
        if (
            self.from_node == other.from_node
            and self.relation == other.relation
            and self.timesteps < other.timesteps
        ):
            if self.relation == Relation.INSIDE:
                if (
                    isinstance(self.to_node, Room)
                    and isinstance(other.to_node, Room)
                    and self.to_node != other.to_node
                ):
                    return True
                elif (
                    not isinstance(self.to_node, Room)
                    and not isinstance(other.to_node, Room)
                    and self.to_node != other.to_node
                ):
                    return True
            elif self.relation == Relation.ON:
                if self.to_node != other.to_node:
                    return True
            elif self.relation == Relation.IS:
                if OPPOSITE(self.to_node) == other.to_node:
                    return True
        else:
            return False

    def return_string_tuple(self):
        return (
            node_rel2str(self.from_node),
            node_rel2str(self.relation),
            node_rel2str(self.to_node),
        )

    def str_with_timestep(self):
        return (
            "(" + ", ".join(self.return_string_tuple() + (str(self.timesteps),)) + ")"
        )

    def __str__(self):
        return (
            "("
            + ", ".join(
                [
                    node_rel2str(self.from_node),
                    node_rel2str(self.relation),
                    node_rel2str(self.to_node),
                ]
            )
            + ")"
        )


class KG:
    def __init__(self, init_dict=None):
        """

        :param init_dict:
        """
        self.nodes = []
        self.edges = []

        self._node_map = {}
        self._name_map = {}
        self._edge_map = {}
        self._max_node_id = 0
        self.depreciated_edges = []
        if init_dict is not None:
            self.from_dictionary(init_dict, -1)

    def clone(self):
        kg = KG()
        kg.nodes = copy.deepcopy(self.nodes)
        kg.edges = copy.deepcopy(self.edges)

        kg._node_map = copy.deepcopy(self._node_map)
        kg._name_map = copy.deepcopy(self._name_map)
        kg._edge_map = copy.deepcopy(self._edge_map)
        kg._max_node_id = copy.deepcopy(self._max_node_id)
        kg.depreciated_edges = copy.deepcopy(self.depreciated_edges)
        return kg

    def from_dictionary(self, d, t):
        nodes = [GraphNode.from_dict(n) for n in d["nodes"]]
        for n in nodes:
            self._node_map[n.id] = n
            self._name_map.setdefault(n.name, []).append(n)
            if n.id > self._max_node_id:
                self._max_node_id = n.id

        hold_filter = lambda rel: (
            "HOLD" if rel == "HOLDS_RH" or rel == "HOLDS_LH" else rel
        )
        edges = [
            (
                ed["from_id"],
                Relation[hold_filter(ed["relation_type"].upper())],
                ed["to_id"],
            )
            for ed in d["edges"]
        ]
        for from_id, relation, to_id in edges:
            if isinstance(to_id, int):
                to_node = self._node_map[to_id]
            elif to_id.lower() in StateList:
                to_node = State[to_id.upper()]
            elif to_id.lower() in RoomList:
                to_node = Room[to_id.upper()]
            else:
                print(to_id)
                raise NotImplementedError()
            from_node = self._node_map[from_id]
            new_edge = GraphEdge(from_node, relation, to_node, t)

            self._edge_map.setdefault((from_id, relation), []).append(new_edge)
            self.edges.append(new_edge)

    def add_dictionary(self, d, t, use_refinement):
        nodes = [GraphNode.from_dict(n) for n in d["nodes"]]
        for n in nodes:
            if n.id in self._node_map.keys():
                continue
            else:
                self._node_map[n.id] = n
                self._name_map.setdefault(n.name, []).append(n)
                if n.id > self._max_node_id:
                    self._max_node_id = n.id

        hold_check = False
        hold_filter = lambda rel: (
            "HOLD" if rel == "HOLDS_RH" or rel == "HOLDS_LH" else rel
        )
        edges = [
            (
                ed["from_id"],
                Relation[hold_filter(ed["relation_type"].upper())],
                ed["to_id"],
            )
            for ed in d["edges"]
        ]
        for from_id, relation, to_id in edges:
            if isinstance(to_id, int):
                to_node = self._node_map[to_id]
            elif to_id.lower() in StateList:
                to_node = State[to_id.upper()]
            elif to_id.lower() in RoomList:
                to_node = Room[to_id.upper()]
            else:
                print(to_id)
                raise NotImplementedError()
            from_node = self._node_map[from_id]
            new_edge = GraphEdge(from_node, relation, to_node, t)

            if (
                use_refinement
                and (new_edge.from_node.id, new_edge.relation) in self._edge_map.keys()
            ):
                removed_edge_idx = []
                if not hold_check:
                    hold_edge = self.search_edge(("character", "hold", None))
                    if hold_edge is not None:
                        for edge in hold_edge:
                            if edge.conflict(new_edge) or new_edge == edge:
                                removed_edge_idx.append(edge)
                    hold_check = True

                for edge in self._edge_map[(new_edge.from_node.id, new_edge.relation)]:
                    if edge.conflict(new_edge) or new_edge == edge:
                        if new_edge == edge:
                            pass
                        else:
                            self.depreciated_edges.append(edge)
                        removed_edge_idx.append(edge)

                for removed_edge in removed_edge_idx:
                    self._edge_map[
                        (removed_edge.from_node.id, removed_edge.relation)
                    ].remove(removed_edge)
                    self.edges.remove(removed_edge)

            self._edge_map.setdefault((from_id, relation), []).append(new_edge)
            self.edges.append(new_edge)

    def add(self, new_graph, t, use_refinement=True):
        self.add_dictionary(new_graph, t, use_refinement)

    def add_edge(self, new_edge: GraphEdge):
        self._edge_map.setdefault(
            (new_edge.from_node.id, new_edge.relation), []
        ).append(new_edge)
        self.edges.append(new_edge)

    def search_edge(self, edge):
        name = edge[0]
        if name not in self._name_map.keys():
            return None
        else:
            from_id = self._name_map[name][0].id
            relation = Relation[edge[1].upper()]
            if (from_id, relation) not in self._edge_map.keys():
                return None
            return self._edge_map[(from_id, relation)]

    def search_obj_inside_obj(self, obj):
        edge = self.search_edge((obj, "INSIDE", None))
        if edge is None:
            return None
        else:
            for e in edge:
                if e.to_node.name not in RoomList:
                    return e.to_node.name
            return None

    def search_obj_closed(self, obj):
        edge = self.search_edge((obj, "is", None))
        if edge is None:
            return False
        else:
            for e in edge:
                if e.to_node.name.lower() == "closed":
                    return True
            return False

    def search_close_obj(self, obj):
        if self.search_edge(("character", "CLOSE", None)) is None:
            return False
        else:
            closed_objs = self.search_edge(("character", "CLOSE", None))
            for closed_obj in closed_objs:
                if closed_obj.to_node.name == obj:
                    return True
            return False

    def search_adjacent_obj(self, edge):
        closed_objs = self.search_edge((edge[0], "ADJACENT", None))
        for closed_obj in closed_objs:
            if closed_obj.to_node.name == edge[2]:
                return True
        return False

    def search_obj_room(self, obj):
        edge = self.search_edge((obj, "INSIDE", None))
        if edge is None:
            return None
        else:
            for e in edge:
                if e.to_node.name in RoomList:
                    return e.to_node.name
            return None

    def delete_edge(self, edge):
        edges = self.search_edge(edge)
        if edges is not None:
            for edge in edges:
                self.edges.remove(edge)
        return

    def search_node(self, name):
        if name not in self._name_map.keys():
            return None
        else:
            return self._name_map[name]

    def refinement(self, new_graph):
        for new_edge in new_graph.edges:
            removed_edge_idx = []
            for edge in self._edge_map[(new_edge.from_node.id, new_edge.relation)]:
                if edge.conflict(new_edge):
                    self.depreciated_edges.append(edge)
                    removed_edge_idx.append(edge)

            for removed_edge in removed_edge_idx:
                self._edge_map[
                    (removed_edge.from_node.id, removed_edge.relation)
                ].remove(removed_edge)
                self.edges.remove(removed_edge)

    def retrieve(
        self,
        instructions,
        embedding_fns=None,
        num_edges=50,
        return_type="str",
        character_info=True,
        replace=False,
        hold_info=True,
    ):
        if len(self.edges) == 0:
            return " "

        edge_str = []
        if return_type == "with_timestep" or return_type == "with_timestep_list":
            timestep_edge_str = []
            for edge in self.edges:
                timestep_edge_str.append(edge.str_with_timestep())

        for edge in self.edges:
            edge_str.append(str(edge))

        if len(edge_str) <= num_edges:
            pass
        elif embedding_fns is None:
            random.shuffle(edge_str)
            edge_str = edge_str[:num_edges]
        else:
            edge_str = [
                s
                for s in edge_str
                if not s.startswith("(character, inside") and "adjacent" not in s
            ]  #  not s.startswith("(character, hold") and
            if return_type == "with_timestep" or return_type == "with_timestep_list":
                timestep_edge_str = [
                    s
                    for s in timestep_edge_str
                    if not s.startswith("(character, inside") and "adjacent" not in s
                ]  #  not s.startswith("(character, hold") and
            if not hold_info:
                edge_str = [s for s in edge_str if not s.startswith("(character, hold")]
                if (
                    return_type == "with_timestep"
                    or return_type == "with_timestep_list"
                ):
                    timestep_edge_str = [
                        s
                        for s in timestep_edge_str
                        if not s.startswith("(character, hold")
                    ]
            if not character_info:
                edge_str = [
                    s for s in edge_str if not s.startswith("(character, close")
                ]
                if (
                    return_type == "with_timestep"
                    or return_type == "with_timestep_list"
                ):
                    timestep_edge_str = [
                        s
                        for s in timestep_edge_str
                        if not s.startswith("(character, close")
                    ]
            tokenized_edge_str = []
            for doc in edge_str:
                doc_tokens = doc[1:-1].split(", ")
                tokenized_edge_str.append(doc_tokens)

            bm25 = BM25Okapi(tokenized_edge_str)
            doc_scores = []
            for instruction in instructions:
                doc_tokens = instruction.split()
                for rel_and_sta in RelationList + StateList:
                    if rel_and_sta in doc_tokens:
                        doc_tokens.remove(rel_and_sta)
                doc_scores.append(bm25.get_scores(doc_tokens))
            edge_embedding = embedding_fns(edge_str)
            instruction_embedding = embedding_fns(instructions)

            similarity = (
                np.array(doc_scores)
                + np.array(compute_similarity(instruction_embedding, edge_embedding))
                * 1e-6
            )
            
            #if np.sum(np.max(similarity, axis=0)) != 0:
            edge_retrieval_prob = np.max(similarity, axis=0) / np.sum(
                (np.max(similarity, axis=0))
            )
            #else:
                #edge_retrieval_prob = [1 / len(np.max(similarity, axis=0)) for _ in range(len(np.max(similarity, axis=0)))]

            if return_type == "with_timestep" or return_type == "with_timestep_list":
                edge_str = timestep_edge_str

            try:
                edge_str = list(
                    np.random.choice(
                        edge_str, p=edge_retrieval_prob, size=num_edges, replace=replace
                    )
                )
            except:
                edge_str = list()

            if character_info:
                edge_str.append(str(self.search_edge(("character", "inside", None))[0]))
                if self.search_edge(
                    ("character", "hold", None)
                ) is None or not self.search_edge(("character", "hold", None)):
                    edge_str.append("(character, hold, none)")
                else:
                    edge_str.append(
                        str(self.search_edge(("character", "hold", None))[0])
                    )
                    holding = self.search_edge(("character", "hold", None))[0].return_string_tuple()

                    # remove "mug inside kitchen" if "character hold mug"
                    if str(self.search_edge((holding[2], "inside", None))[0]) in edge_str:
                        edge_str.remove(
                            str(self.search_edge((holding[2], "inside", None))[0])
                        )

                    # remove "character close mug" if "character hold mug"
                    search_result = self.search_edge(("character", "close", None))
                    for result in search_result:
                        if f"(character, close, {holding[2]})" == str(result):
                            if str(result) in edge_str:
                                edge_str.remove(str(result))

                adj_room = self.search_edge(
                    (self.search_obj_room("character"), "adjacent", None)
                )
                for room in adj_room:
                    edge_str.append(str(room))
            
            
            # remove desk is closed ...
            if "(desk, is, closed)" in edge_str:
                edge_str.remove("(desk, is, closed)")
            
            # add adjecent rooms
            rooms = ["livingroom", "kitchen", "bathroom", "bedroom"]
            random.shuffle(rooms)
            for i, room1 in enumerate(rooms):
                for room2 in rooms[i+1:]:
                    adj_room = self.search_edge((room1, "adjacent", None))
                    for adj in adj_room:
                        adj_tuple = adj.return_string_tuple()
                        if str((adj_tuple[0], 'adjacent', adj_tuple[2])).replace("'", '') not in edge_str \
                                and str((adj_tuple[2], 'adjacent', adj_tuple[0])).replace("'", '') not in edge_str:
                            edge_str.append(str(adj))

        edge_str = remove_duplicates(edge_str)
        if return_type == "str" or return_type == "with_timestep":
            return ", ".join(edge_str).lower()
        elif return_type == "str_list" or return_type == "with_timestep_list":
            return edge_str

    def history_retrieve(
        self,
        instructions,
        embedding_fns=None,
        num_edges=50,
        return_type="str",
        character_info=True,
    ):
        if len(self.depreciated_edges) == 0:
            return " "

        edge_str = []
        if return_type == "with_timestep" or return_type == "with_timestep_list":
            timestep_edge_str = []
            for edge in self.depreciated_edges:
                timestep_edge_str.append(edge.str_with_timestep())

        for edge in self.depreciated_edges:
            edge_str.append(str(edge))

        if len(edge_str) <= num_edges:
            if return_type == "with_timestep" or return_type == "with_timestep_list":
                edge_str = timestep_edge_str
        elif embedding_fns is None:
            random.shuffle(edge_str)
            edge_str = edge_str[:num_edges]
        else:
            edge_str = [
                s
                for s in edge_str
                if not s.startswith("(character, inside")
                and not s.startswith("(character, hold")
                and "adjacent" not in s
            ]
            if return_type == "with_timestep" or return_type == "with_timestep_list":
                timestep_edge_str = [
                    s
                    for s in timestep_edge_str
                    if not s.startswith("(character, inside")
                    and not s.startswith("(character, hold")
                    and "adjacent" not in s
                ]
            if not character_info:
                edge_str = [
                    s for s in edge_str if not s.startswith("(character, close")
                ]
                if (
                    return_type == "with_timestep"
                    or return_type == "with_timestep_list"
                ):
                    timestep_edge_str = [
                        s
                        for s in timestep_edge_str
                        if not s.startswith("(character, close")
                    ]

            if not edge_str:
                return ""

            tokenized_edge_str = []
            for doc in edge_str:
                doc_tokens = doc[1:-1].split(", ")
                tokenized_edge_str.append(doc_tokens)

            bm25 = BM25Okapi(tokenized_edge_str)
            doc_scores = []
            for instruction in instructions:
                doc_tokens = instruction.split()
                for rel_and_sta in RelationList + StateList:
                    if rel_and_sta in doc_tokens:
                        doc_tokens.remove(rel_and_sta)
                doc_scores.append(bm25.get_scores(doc_tokens))
            edge_embedding = embedding_fns(edge_str)
            instruction_embedding = embedding_fns(instructions)

            similarity = (
                np.array(doc_scores)
                + np.array(compute_similarity(instruction_embedding, edge_embedding))
                * 1e-6
            )
            edge_retrieval_prob = np.max(similarity, axis=0) / np.sum(
                (np.max(similarity, axis=0))
            )
            if return_type == "with_timestep":
                edge_str = timestep_edge_str
            edge_str = list(
                np.random.choice(
                    edge_str,
                    p=edge_retrieval_prob,
                    size=min(num_edges, len(edge_str)),
                    replace=False,
                )
            )
            if character_info:
                edge_str.append(str(self.search_edge(("character", "inside", None))[0]))
                if self.search_edge(
                    ("character", "hold", None)
                ) is None or self.search_edge(("character", "hold", None)):
                    edge_str.append("(character, hold, none)")
                else:
                    edge_str.append(
                        str(self.search_edge(("character", "hold", None))[0])
                    )
                adj_room = self.search_edge(
                    (self.search_obj_room("character"), "adjacent", None)
                )
                for room in adj_room:
                    edge_str.append(str(room))

        edge_str = remove_duplicates(edge_str)
        if return_type == "str" or return_type == "with_timestep":
            return ", ".join(edge_str).lower()
        else:
            return edge_str

    def return_string_tuple(self):
        edge_str = []
        for edge in self.edges:
            edge_str.append(edge.return_string_tuple())
        return edge_str

    def __str__(self):
        edge_str = []
        for edge in self.edges:
            edge_str.append(str(edge))
        return ", ".join(edge_str).lower()
