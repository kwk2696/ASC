import os

from embodied_cd.common import argparse, dataset_utils
from embodied_cd.common.print_utils import *
from embodied_cd.common.dataset_utils import PromptTemplate
from scripts.train import *


def generate_vh_dataset(args):
    # prepare policy model
    tokenizer, model = prepare_model(args.pi_model_name, init=True)
    
    # prepare dataset generator
    vh_generator = dataset_utils.VirtualHomeDatasetGenerator(
        tokenizer=tokenizer,
        model=model,
        dataset_dir=os.path.join(args.dataset_dir, "raw"),
        num_topk_edge=args.num_topk_edge,
        max_think_token=args.max_think_token,
    )
    print(f"Episode Length at {args.dataset_dir}:", len(vh_generator))
    
    dataset_path = args.dataset_dir + "/full_dataset.jsonl"

    if not os.path.exists(dataset_path):
        vh_generator.save_dataset(args.dataset_dir, os.path.basename(dataset_path))
    else:
        print_warn(f"Dataset already exists at {dataset_path}")

    print_check("=" * 10, "Check KG Retrieval", "=" * 10)
    dataset = dataset_utils.VirtualHomeDataset.load(dataset_path)
    print_warn(f"Discarded Step Percentage: {len(dataset)/vh_generator.total_steps*100:.2f}")
    print_check("Length of dataset:", vh_generator.total_steps)
    for idx, value in enumerate(dataset):
        print_pass("\nInstruction:", value["instruction"])
        print("State:", PromptTemplate.preprocess(value["state"]))
        print("History:", value["history"])
        print("Think:", value["think"])
        print_check("Action:", value["action"])
        print("Reward:", float(value["reward"]))
    print_warn(f"Discarded Step Percentage: {len(dataset)/vh_generator.total_steps*100:.2f}")
    print_check("Length of dataset:", vh_generator.total_steps)
    print_check("=" * 10, "Check KG Retrieval End", "=" * 10)


def generate_alfred_dataset(args):
    # prepare policy model
    tokenizer, model = prepare_model(args.pi_model_name, init=True)
    
    # prepare dataset generator
    af_generator = dataset_utils.AlfredDatasetGenerator(
        tokenizer=tokenizer,
        model=model,
        dataset_dir=os.path.join(args.dataset_dir, "raw"),
        num_topk_edge=args.num_topk_edge,
        max_think_token=args.max_think_token,
    )
    print(f"Episode Length at {args.dataset_dir}:", len(af_generator))

    dataset_path = args.dataset_dir + "/full_dataset.jsonl"
    if not os.path.exists(dataset_path):
        af_generator.save_dataset(args.dataset_dir)
    else:
        print_warn(f"Dataset already exists at {dataset_path}")

    print_check("=" * 10, "Check KG Retrieval", "=" * 10)
    dataset = dataset_utils.VirtualHomeDataset.load(dataset_path)
    print_warn(f"Discarded Step Percentage: {len(dataset)/af_generator.total_steps*100:.2f}")
    print_check("Length of dataset:", af_generator.total_steps)
    for idx, value in enumerate(dataset):
        print_pass("\nInstruction:", value["instruction"])
        print("State:", value["state"])
        print("History:", value["history"])
        print("Think:", value["think"])
        print_check("Action:", value["action"])
        print("Reward:", float(value["reward"]))
    print_warn(f"Discarded Step Percentage: {len(dataset)/af_generator.total_steps*100:.2f}")
    print_check("Length of dataset:", af_generator.total_steps)
    print_check("=" * 10, "Check KG Retrieval End", "=" * 10)


if __name__ == "__main__":
    args = argparse.parse_args()
    if args.max_think_token is None:
        args.max_think_token = 200
    if args.env_name == "virtualhome":
        generate_vh_dataset(args)
    elif args.env_name == "alfred":
        generate_alfred_dataset(args)
