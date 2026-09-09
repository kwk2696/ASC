import argparse

from embodied_cd.environments.default import AlfredEnv
from embodied_cd.common.print_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="List the ALFRED game files of one split, grouped by instruction type.")
    parser.add_argument("--type", type=str, required=True, choices=["behavior_seen", "behavior_unseen", "environment"])
    parser.add_argument("--task_id", type=int, default=None, help="continual set index (1-based); required unless --type behavior_unseen")
    return parser.parse_args()


def main(args):
    split_dir = f"./externals/cl-alfred/embodied_split/{args.type.split('_')[0]}_il"
    if args.type == "behavior_unseen":
        split_path = f"{split_dir}/valid_unseen.json"
    else:
        if args.task_id is None:
            raise ValueError("--task_id is required for --type behavior_seen / environment")
        split_path = f"{split_dir}/embodied_data_disjoint_rand1_cls1_task{args.task_id-1}.json"
    env = AlfredEnv(split_path)

    disjoint_instruction = {}
    for game in env.game_files:
        task = game.split('/')[-3]  # .../<task>-<scene>/<trial>/<game file>
        room = task.split('-')[-1]
        instruction = task[:-len(room)]
        disjoint_instruction[instruction] = disjoint_instruction.get(instruction, 0) + 1

    for instruction, value in disjoint_instruction.items():
        print(instruction, value)


if __name__ == "__main__":
    main(parse_args())
