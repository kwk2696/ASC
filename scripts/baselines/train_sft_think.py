import os
import wandb
import json
from trl import SFTConfig, SFTTrainer

from transformers import set_seed

from embodied_cd.common import argparse
from embodied_cd.common.print_utils import *
from embodied_cd.common.env_utils import *
from embodied_cd.trl.algos.pipe import SentenceSimilarityPipeline, TfidfPipeline
from scripts.train import *

from embodied_cd.agents import SFTAgent


def train(args, _type):
    # wandb log on/off
    if args.wandb_off:
        wandb.init(mode='disabled')
    else:
        wandb.init(project="ASC")
        wandb.run.name = "baseline_sft_think"

    output_dir = os.path.join(args.output_dir, _type)
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
        prev_output_dir = f"./results/SFT-Think/{args.env_name}/{args.continual_type}_{args.continual_seq}/{prev_continual_set}/{args.pi_model_name.split('/')[-1]}/{args.tag}"
        tokenizer, model = prepare_model(os.path.join(prev_output_dir, _type), init=False)
        model.set_adapter("default")

    # prepare dataset
    dataset, collator = prepare_dataset(
        args.env_name, args.pi_model_name, args.dataset_dir, tokenizer, dataset_scale=args.dataset_scale, dataset_type=_type)
    
    # set configs
    if 'instruct' in args.pi_model_name or 'Instruct' in args.pi_model_name:
        training_args = SFTConfig(
            output_dir=output_dir, 
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            num_train_epochs=40 if _type == 'cd-action-think' else 80,
            learning_rate=1.41e-05 if _type == 'cd-action-think' else 5e-05,
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
            num_train_epochs=40 if _type == 'cd-action-think' else 80,
            learning_rate=1.41e-05 if _type == 'cd-action-think' else 5e-05,
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
    print_warn("[SFT-Think] Model Training")
    trainer.train()

    # saving
    trainer.save_model(output_dir)


def test(args):
    # load model
    think_tokenizer, think_model = prepare_model(os.path.join(args.output_dir, 'cd-think'), init=False)
    plan_tokenizer, plan_model = prepare_model(os.path.join(args.output_dir, 'cd-action-think'), init=False)

    if args.test == 'seen':
        # prepare dataset
        dataset, collator = prepare_dataset(
            args.env_name, args.pi_model_name, args.dataset_dir, think_tokenizer, dataset_scale=args.dataset_scale)
    else:
        raise ValueError(f"--test only supports 'seen' (unseen sets hold expert trajectories for --evaluate), got {args.test!r}")

    # prepare agent
    agent = SFTAgent(
        args.pi_model_name,
        think_model=think_model,
        think_tokenizer=think_tokenizer,
        plan_model=plan_model,
        plan_tokenizer=plan_tokenizer,
        env_name=args.env_name,
        max_think_token=args.max_think_token,
        decoding_strategy='greedy',
    )

    cossim_pipe = SentenceSimilarityPipeline()
    tfidf_pipe = TfidfPipeline()

    results = []
    cossim_score, tfidf_score, check = 0, 0, 0
    for idx, data in enumerate(dataset):
        instruction, state, history = \
            data["instruction"], data["state"], data["history"]

        think = agent.get_think(instruction, state, history)
        action, _ = agent.get_action(instruction, state, history, think)
        
        
        cossim = cossim_pipe(data["think"].split(". "), think.split(". "))
        cossim_score += cossim / len(dataset)
        tfidf = tfidf_pipe(data["think"].split(". "), think.split(". "))
        tfidf_score += tfidf / len(dataset)
        check += (action == data['action'])
        print_pass(f"[[Ground Think]]\n{data['think']}")
        print_check(f"[[Pred Think]]\n{think}")
        print_pass(f"[[Ground Action]]\n{data['action']}")
        print_check(f"[[Pred Action]]\n{action}")
        print_error(f"cossim score: {cossim}")
        print_error(f"tfidf score: {tfidf}")

        results.append(
            {
                "ground_think": data["think"],
                "init_think": think,
            }
        )

    print_check(f"[[Total Accuracy]] {check/len(dataset)*100:.2f}%")
    print_check(f"Final cossim score: {cossim_score}")
    print_check(f"Final tfidf score: {tfidf_score}")

    results.append(
        {
            "cossim": cossim_score,
            "tfidf": tfidf_score,
        }
    )

    # saving results
    with open(args.output_dir + f"/{args.test}_think_results.json", "w") as f:
        json.dump(results, f)


def evaluate(args):
    # load model
    think_tokenizer, think_model = prepare_model(os.path.join(args.output_dir, 'cd-think'), init=False)
    plan_tokenizer, plan_model = prepare_model(os.path.join(args.output_dir, 'cd-action-think'), init=False)

    
    # prepare agent
    agent = SFTAgent(
        args.pi_model_name,
        think_model, 
        think_tokenizer, 
        plan_model, 
        plan_tokenizer, 
        args.env_name, 
        args.continual_type.split('_')[0],
        max_think_token=args.max_think_token,
        max_think_times=1,
        decoding_strategy="greedy",
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
        output_dir = f"./results/SFT-Think/{args.env_name}/{args.continual_type}_{args.continual_seq}/{args.continual_set}/{args.pi_model_name.split('/')[-1]}"
        args.output_dir = os.path.join(output_dir, args.tag)
    print_warn(f"[Dataset Dir] {args.dataset_dir}")
    print_warn(f"[Output Dir] {args.output_dir}")

    if args.train:
        if args.phase == 1:
            train(args, 'cd-think')
        elif args.phase == 2:
            train(args, 'cd-action-think')
        else:
            raise NotImplementedError
    if args.test:
        test(args)
    if args.evaluate:
        evaluate(args)
