"""Sample a training subset from full_dataset.jsonl, one trajectory at a time.

--dataset_scale is a multiple of the default subset: 1.0 keeps 12.5% of the full set on
VirtualHome and 25% on ALFRED (BASE_FRACTION in scripts/train.py), 2.0 keeps twice that.
Trajectories are cut at reward == 1.0, grouped into chunks of 1/fraction, and one
trajectory is drawn per chunk, so every retained episode stays complete.

Reads  <dataset_dir>/full_dataset.jsonl  ->  writes <dataset_dir>/part_<scale>_dataset.jsonl
"""
import json
import random

from embodied_cd.common import argparse
from embodied_cd.common.print_utils import *
from scripts.train import BASE_FRACTION

FIELDS = [
    "instruction", "state", "state_full", "think", "think_copy",
    "action", "history", "future", "reward",
]

if __name__ == "__main__":
    args = argparse.parse_args()
    random.seed(args.seed)

    scale = float(args.dataset_scale)
    percent = BASE_FRACTION[args.env_name] * scale
    root = f"./datasets/{args.env_name}/{args.continual_type}_train/{args.continual_set}"
    dataset_path = f"{root}/full_dataset.jsonl"
    save_path = f"{root}/part_{scale}_dataset.jsonl"

    source = []
    with open(dataset_path) as f:
        for idx, line in enumerate(f):
            try:
                json_line = json.loads(line)
            except json.JSONDecodeError as e:
                print_error(f"Error in loading {dataset_path} at line {idx + 1}: {e}")
                raise SystemExit(1)
            json_line["index"] = idx
            source.append(json_line)

    if percent >= 1.0:
        print_warn(f"scale {scale} keeps the whole set ({percent:.2f} >= 1.0); writing full_dataset.jsonl as {save_path}")
    else:
        print_warn(f"Original Dataset: {len(source)}")
        total_traj, traj = [], []
        for s in source:
            traj.append(s)
            if s["reward"] == "1.0":
                total_traj.append(traj)
                traj = []
        print(f"Total Traj: {len(total_traj)}")

        chunk_size = max(1, round(1 / percent))  # keeps ~1/chunk_size of the trajectories
        print(f"Chunk Size: {chunk_size}")

        sampled = []
        for i in range(0, len(total_traj), chunk_size):
            chunk = total_traj[i:i + chunk_size]
            if chunk:
                sampled.extend(random.choice(chunk))
        print_check(f"New Dataset: {len(sampled)}")
        source = sampled

    with open(save_path, "w") as f:
        for s in source:
            f.write(json.dumps({k: s[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    print_pass(f"Saved {len(source)} steps to {save_path}")
