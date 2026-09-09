import os
import wandb
from trl import SFTConfig, SFTTrainer

from transformers import set_seed

from embodied_cd.common import argparse
from embodied_cd.common.print_utils import *
from embodied_cd.common.env_utils import *
from scripts.train import *

from embodied_cd.agents import SFTAgent


def train(args):
    # wandb log on/off
    if args.wandb_off:
        wandb.init(mode='disabled')
    else:
        wandb.init(project="ASC")
        wandb.run.name = "baseline_sft"

    output_dir = os.path.join(args.output_dir, "cd-action")
    os.makedirs(output_dir, exist_ok=True)

    # load configs and get continual phase
    config = {}
    with open(f"./configs/{args.env_name}/{args.continual_type}_train.py", "r") as f:
        code = f.read()
    exec(code, config)
    phase = config["sequences"][args.continual_seq].index(args.continual_set)
        
    # seqft 0: AppendOnly, seqft 1: SeqFT
    if (phase == 0) or (args.seqft == 0):
        # first stage of the sequence: initialize
        tokenizer, model = prepare_model(args.pi_model_name, init=True)
        # add adapter
        model = add_adapter(model)
    else:
        # if phase > 0, prepare model for sequential finetuning
        prev_continual_set = config["sequences"][args.continual_seq][phase - 1]
        prev_output_dir = f"./results/SFT/{args.env_name}/{args.continual_type}_{args.continual_seq}/{prev_continual_set}/{args.pi_model_name.split('/')[-1]}/{args.tag}"
        tokenizer, model = prepare_model(os.path.join(prev_output_dir, "cd-action"), init=False)
        model.set_adapter("default")

    # prepare dataset
    dataset, collator = prepare_dataset(
        args.env_name, 
        args.pi_model_name, 
        args.dataset_dir, 
        tokenizer, 
        dataset_scale=args.dataset_scale,
        dataset_type="cd-action",
        few_shot_example=None,
    )

    # set configs
    if 'instruct' in args.pi_model_name or 'Instruct' in args.pi_model_name:
        training_args = SFTConfig(
            output_dir=output_dir, 
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            num_train_epochs=40,
            learning_rate=1.41e-05,
            per_device_train_batch_size=4,
            # to avoid the default dataset process
            # we should disable 'remove_unused_columns' & 'skip_prepare_dataset'
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
    else:
        training_args = SFTConfig(
            output_dir=output_dir, 
            dataset_text_field="text",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            num_train_epochs=40,
            learning_rate=1.41e-05,
            per_device_train_batch_size=4,
        )

    params = training_args.__dict__
    params.update({"seqft": args.seqft})
     
    save_params(output_dir, params)

    # create trainer
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        args=training_args
    )

    # training
    print_warn("[SFT] Model Training")
    trainer.train()

    # saving
    trainer.save_model(output_dir)


def test(args):
    # load model
    tokenizer, model = prepare_model(os.path.join(args.output_dir, 'cd-action'), init=False)

    if args.test == 'seen':
        # prepare dataset
        dataset, collator = prepare_dataset(
            args.env_name, args.pi_model_name, args.dataset_dir, tokenizer, dataset_scale=args.dataset_scale)
    else:
        raise ValueError(f"--test only supports 'seen' (unseen sets hold expert trajectories for --evaluate), got {args.test!r}")

    # prepare agent
    agent = SFTAgent(
        args.pi_model_name,
        plan_model=model,
        plan_tokenizer=tokenizer,
        env_name=args.env_name,
        max_think_token=args.max_think_token,
        decoding_strategy='greedy',
    )

    check = 0
    for idx, data in enumerate(dataset):
        instruction, state, history = data["instruction"], data["state"], data["history"]

        action, _ = agent.get_action(instruction, state, history)

        check += (action == data['action'])
        print_pass(f"[[Ground Action]]\n{data['action']}")
        print_check(f"[[Pred Action]]\n{action}")
        print_warn(f"Reward: {action == data['action']}")
    print_warn(f"[[Total Accuracy]] {check/len(dataset)*100:.2f}%")


def evaluate(args):
    # load model
    tokenizer, model = prepare_model(os.path.join(args.output_dir, 'cd-action'), init=False)

    
    # prepare agent
    agent = SFTAgent(
        args.pi_model_name,
        None, 
        None, 
        model, 
        tokenizer, 
        args.env_name, 
        args.continual_type.split('_')[0],
        max_think_token=args.max_think_token, 
        max_think_times=None, 
        decoding_strategy='greedy'
    )
   
    if args.continual_eval_set is None:
        flag = False
        args.continual_eval_set = args.continual_set
    else:
        flag = True

    # make env
    split_path = f"./externals/cl-alfred/embodied_split/{args.continual_type.split('_')[0]}_il/embodied_data_disjoint_rand1_cls1_task{int(args.continual_eval_set.split('_')[-1])-1}.json" # for alfred
    env = make_env(args.env_name, args.num_topk_edge, args.env_ip, args.seed, args.env_port, split_path)

    if args.evaluate == "seen" or args.evaluate == "both":
        # Evaluate on Train Environment
        env_list, task_list, available_tasks, ps_line = load_evaluation_configs(
            args.env_name, args.continual_type, args.continual_eval_set, "seen")

        print_warn(f"Evaluating in Seen Set")
        if not flag:
            save_path = args.output_dir + '/results/evaluation_seen.txt'
        else:
            save_path = args.output_dir + f'/results/evaluation_seen_{args.continual_eval_set}.txt'
        evaluation_pipe(
            env, 
            agent, 
            env_list, 
            task_list, 
            available_tasks,
            ps_line,
            total_episode=args.eval_episode, 
            save_path=save_path
        )
    if args.evaluate == "unseen" or args.evaluate == "both": 
        # Evaluate on Test Environment
        env_list, task_list, available_tasks, ps_line = load_evaluation_configs(
            args.env_name, args.continual_type, args.continual_eval_set, "unseen")
        
        print_warn(f"Evaluating on Unseen Set")
        if not flag:
            save_path = args.output_dir + '/results/evaluation_unseen.txt'
        else:
            save_path = args.output_dir + f'/results/evaluation_unseen_{args.continual_eval_set}.txt'
        evaluation_pipe(
            env,
            agent, 
            env_list, 
            task_list, 
            available_tasks,
            ps_line,
            total_episode=args.eval_episode, 
            save_path=save_path
        )


if __name__ == "__main__":
    args = argparse.parse_args()
    set_seed(args.seed)
    if args.max_think_token is None:
        args.max_think_token = 200
    # directory settings
    if args.dataset_dir is None:
        args.dataset_dir = f"./datasets/{args.env_name}/{args.continual_type}_train/{args.continual_set}"
    if args.output_dir is None:
        output_dir = f"./results/SFT/{args.env_name}/{args.continual_type}_{args.continual_seq}/{args.continual_set}/{args.pi_model_name.split('/')[-1]}"
        args.output_dir = os.path.join(output_dir, args.tag)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.evaluate:
        evaluate(args)
