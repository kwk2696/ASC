import os
from embodied_cd.common import argparse
from embodied_cd.common.print_utils import *


def print_results(path, label, printer):
    """Print everything from the last "Total Results" marker to the end of an evaluation log."""
    if not os.path.isfile(path):
        return
    lines = open(path).read().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if "Total Results" in lines[i]:
            print("=" * 20)
            printer(f"[[{label} Results]]")
            print("\n".join(lines[i + 1:]))
            return
    print_warn(f"no 'Total Results' marker in {path}")


args = argparse.parse_args()

config = {}
with open(f"./configs/{args.env_name}/{args.continual_type}_train.py", "r") as f:
    code = f.read()
exec(code, config)
sequences = config["sequences"][args.continual_seq]

prefix = f"./results/ASC/{args.env_name}/{args.continual_type}_{args.continual_seq}"

print_error("=" * 50)
for seq in sequences:
    path = prefix + f"/{seq}_{args.pi_model_name.split('/')[-1]}_{args.tag}/results" 
    
    print_check(f"\n\n>> {args.env_name} {args.continual_type}: Sequence {args.continual_seq}, Set {seq}")
    
    path_dir = path + "/evaluation_seen"
    if args.test_time_control:
        path_dir = path_dir + f"_control_{args.test_time_level}"
    if args.total_think != 5:
        path_dir = path_dir + f"_think{args.total_think}"
    path_dir = path_dir + ".txt"
    print(path_dir)

    print_results(path_dir, "Seen", print_pass)

    path_dir = path + "/evaluation_unseen"
    if args.test_time_control:
        path_dir = path_dir + f"_control_{args.test_time_level}"
    if args.total_think != 5:
        path_dir = path_dir + f"_think{args.total_think}"
    path_dir = path_dir + ".txt"
    print(path_dir)

    print_results(path_dir, "Unseen", print_warn)
