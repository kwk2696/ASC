import argparse
from typing import Any, Dict

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--env_name", type=str, default="virtualhome")
    parser.add_argument("--env_ip", type=str, default="127.0.0.1")
    parser.add_argument("--env_port", type=int, default=8080)
    # continual settings
    parser.add_argument("--continual_type", type=str, default="behavior")
    parser.add_argument("--continual_seq", type=str, default="sequence_1")
    parser.add_argument("--continual_set", type=str, default="set_1")
    parser.add_argument("--continual_eval_set", type=str, default=None)
    # dataset
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_topk_edge", type=int, default=6)
    parser.add_argument("--max_think_token", type=int, default=None) # rationale tokens per slot; ASC uses 40, baselines and annotation 200
    parser.add_argument("--dataset_scale", type=float, default=1.0) # training subset as a multiple of the default (part_1.0 = 12.5% of full_dataset.jsonl on VirtualHome, 25% on ALFRED)
    # base model
    parser.add_argument("--pi_model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--prev_tag", type=str, default=None)
    # training
    parser.add_argument("--phase", type=int, default=1) # SFT-Think only: 1 trains the rationale model, 2 the action model
    parser.add_argument("--seqft", type=int, default=1)
    # inference
    parser.add_argument("--eval_episode", type=int, default=4)
    parser.add_argument("--test_time_control", action="store_true")
    parser.add_argument("--test_time_level", type=int, default=3, choices=range(7))
    parser.add_argument("--total_think", type=int, default=5, choices=range(1, 6)) # rationale slots; prompts and datasets define 5
    # msic
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--wandb_off", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", type=str, default=None) # scripts/asc: any value runs threshold calibration on the training split; baselines: "seen"
    parser.add_argument("--evaluate", type=str, default=None, choices=["seen", "unseen", "both"])
    
    args = parser.parse_args()
    return args
