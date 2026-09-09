import os
import tqdm
import wandb
import torch
from torch.utils.data import DataLoader

from embodied_cd.common.print_utils import *
from embodied_cd.common.dataset_utils import PromptTemplate
from embodied_cd.trl.algos.pipe import SentenceSimilarityPipeline, TfidfPipeline
from embodied_cd.trl.algos.core import stack_dicts, stats_to_np, stats_print


def custom_collate(batch):
    return {
        "query_texts": [data["query"] for data in batch],
        "response_texts": [data["response"] for data in batch],
        "query_ids": [data["query_ids"] for data in batch],
        "response_ids": [data["response_ids"] for data in batch],
        "instructions": [data["instruction"] for data in batch],
        "states": [data["state"] for data in batch],
        "thinks": [data["think"] for data in batch],
        "think_lists": [data["think_list"] for data in batch],
        "actions": [data["action"] for data in batch],
        "histories": [data["history"] for data in batch],
        "rewards": [float(data["reward"]) for data in batch],
        "indexes": [data["index"] for data in batch],
    }


class ASCTrainer:
    """
        Amortizing Self-Correction (ASC) Trainer.
    """

    default_params = {
        "total_epochs": 20,
        "inner_epochs": 4,
        "learning_rate": 1.41e-5,
        "batch_size": 4,
        "max_think_token": 50,
        "total_think": 5,
    }

    gen1_params = {
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.3,
        "temperature": 0.2,
    }

    gen2_params = {
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.6,
        "temperature": 0.7,
    }

    def __init__(self, env_name, pre_model_name, base_tokenizer, base_model ,tokenizer, model, dataset, output_dir, **params):
        self.env_name = env_name
        self.pre_model_name = pre_model_name

        # set params
        self.params = {**self.default_params, **params}
        print_warn(f"Max Think Tokens: {params['max_think_token']}")
        print_warn(f"Total Think: {params['total_think']}")
        
        self.base_tokenizer = base_tokenizer
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device
        
        self.dataset = dataset
        self.output_dir = output_dir


        # setup
        self._setup_model()
        self._setup_init_response()

    def _setup_model(self) -> None:
        # create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            collate_fn=custom_collate,
            drop_last=True,
        )

        # create optimizer
        self.optimizer_r = torch.optim.AdamW( # rationale
            self.model.parameters(),
            lr=self.params["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )

        self.optimizer_p = torch.optim.AdamW( # action
            self.model.parameters(),
            lr=self.params["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )

        # reward pipe
        self.cossim_pipe = SentenceSimilarityPipeline()
        self.tfidf_pipe = TfidfPipeline()
        
        # plan template
        self.plan_template = PromptTemplate(self.env_name, "cd-action-think")

    @classmethod
    def _setup_base_model(cls, pre_model_name, tokenizer, model, dataset, max_think_token):
        # reward pipe
        cossim_pipe = SentenceSimilarityPipeline()
        tfidf_pipe = TfidfPipeline()
        
        cossim_score, tfidf_score, count = 0., 0., 0 
        for i, data in enumerate(tqdm.tqdm(dataset, desc="setup base")):
            instruction, state, history, think_list = data["instruction"], PromptTemplate.preprocess(data["state"]), data["history"], data["think_list"]

            # 1. Generate Initial Resposne
            model.set_adapter('reasoning_policy')
            init_response_list = []
            for j in range(len(think_list)):
                _query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\n{PromptTemplate.correct_think_prompts[j]}"
                if 'instruct' in pre_model_name or 'Instruct' in pre_model_name:
                    query = [{"role": "user", "content": _query}]
                    query_id = tokenizer.apply_chat_template(
                        query, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
                else:
                    query = f"### Human: {_query}\n### Assistant:"
                    query_id = tokenizer.encode(query, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    response_text = cls.generate(
                        cls, model, tokenizer, query_id, cls.gen1_params, max_think_token)

                # process response text 
                if 'instruct' in pre_model_name or 'Instruct' in pre_model_name:
                    response_text = response_text.split(". ")[0] 
                else:
                    response_text = response_text.strip().split("\n")[0]
                    response_text = response_text.split(". ")[0] 

                response_text = response_text + "." if response_text and response_text[-1] != '.' else response_text
                init_response_list.append(response_text)
            init_response = " ".join(init_response_list)

            cossim_score += cossim_pipe(think_list, init_response_list)
            tfidf_score += tfidf_pipe(think_list, init_response_list)
            count += 1
        
        return 0.5 * cossim_score / count  + 0.5 * tfidf_score / count

    def _setup_init_response(self):
        self.init_response = {}
        # 0. Setting Seed Rationale Model
        if self.base_model is None:
            self.model.set_adapter('reasoning_policy')
            tokenizer, model = self.tokenizer, self.model
        else:
            self.base_model.set_adapter('reasoning_policy')
            tokenizer, model = self.base_tokenizer, self.base_model

        for data in tqdm.tqdm(self.dataset, desc="init batch"): #iterate over dataset
            instruction, state, history = data["instruction"], PromptTemplate.preprocess(data["state"]), data["history"]
            response_list = []
            for j in range(self.params["total_think"]):
                _query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\n{PromptTemplate.correct_think_prompts[j]}"
                if 'instruct' in self.pre_model_name or 'Instruct' in self.pre_model_name:
                    query = [{"role": "user", "content": _query}]
                    query_id = tokenizer.apply_chat_template(
                        query, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
                else:
                    query = f"### Human: {_query}\n### Assistant:"
                    query_id = tokenizer.encode(query, return_tensors="pt").to(model.device)

                # disable adapters
                with torch.no_grad():
                    response_text = self.generate(
                        model, tokenizer, query_id, self.gen1_params, self.params["max_think_token"])

                # process response text 
                if 'instruct' in self.pre_model_name or 'Instruct' in self.pre_model_name:
                    response_text = response_text.split(". ")[0] 
                else:
                    response_text = response_text.strip().split("\n")[0]
                    response_text = response_text.split(". ")[0] 

                response_text = response_text + "." if response_text and response_text[-1] != '.' else response_text

                response_list.append(response_text)
            self.init_response[data["index"]] = response_list

        self.neg_response = {}
        for data in self.dataset:
            self.neg_response[data["index"]] = []

    def get_score_str(self, think1, think2):
        score = self.cossim_pipe(think1, think2)
        if score < 0.6:
            return score, "There are many errors in the Think. You need a major revision in the Think."
        elif score >= 0.6 and score < 0.82:
            return score, "There are some errors in the Think. You need a moderate revision in the Think."
        else:
            return score, "There is little error in the Think. You need a minor revision in the Think."

    def train(self):
        # enable adapter layers
        self.prev_neg_percent = 1.0

        for epoch in tqdm.tqdm(range(1, self.params["total_epochs"]+1), desc="epoch"): # iterate over epochs
            ####################### process optimize
            all_stats = []
            for batch in self.dataloader: # iterate over dataloader
                for inner_epoch in range(self.params["inner_epochs"]):
                    for i in range(self.params["batch_size"]):
                        instruction, state, action, history, idx, think, think_cat = \
                            batch["instructions"][i], \
                            PromptTemplate.preprocess(batch["states"][i]), \
                            batch["actions"][i], batch["histories"][i], \
                            batch["indexes"][i], batch["think_lists"][i], batch["thinks"][i]
                        
                        """ 1: Rationale Optimize """
                        self.model.set_adapter('reasoning_policy')
                        loss_r, loss_fr, loss_brb, loss_brs = \
                            torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device)
                        ########################################

                        """ 1-1: (Forward) Reasoning Optimize """
                        ########################################
                        reasoning_trace = ""
                        for j in range(self.params["total_think"]):
                            _query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\n{PromptTemplate.correct_think_prompts[j]}"
                            query, query_id, cquery, cquery_id, cquery_label = self.querize(_query, think[j])

                            # run model
                            model_output = self.model(cquery_id, labels=cquery_label)
                            loss_fr += model_output.loss / self.params["total_think"]
                        
                        self.optimizer_r.zero_grad()
                        loss_fr.backward()
                        self.optimizer_r.step()    
                        
                        """ 1-2: (Backward) Base-Correction Reasoning Optimize """
                        ########################################
                        init_response = self.init_response[idx]
                        reasoning_trace = ""
                        for j in range(self.params["total_think"]):
                            score, score_str = self.get_score_str(init_response[j], think[j])
                                
                            _query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\nReasoning Trace: {reasoning_trace}\nThink: {init_response[j]}\n{score_str} {PromptTemplate.correct_think_prompts[j]}"
                            query, query_id, cquery, cquery_id, cquery_label = self.querize(_query, think[j])

                            # run Model
                            model_output = self.model(cquery_id, labels=cquery_label)
                            loss_brb += model_output.loss / self.params["total_think"]
                            reasoning_trace = think[j] if j == 0 else reasoning_trace + " " + think[j]

                        self.optimizer_r.zero_grad()
                        loss_brb.backward()
                        self.optimizer_r.step()    

                        """ 1-3: (Backward) Self-Correction Reasoning Optimize """
                        ########################################
                        neg_responses = self.neg_response[idx]
                        for neg_response in neg_responses:
                            reasoning_trace = ""
                            for j in range(self.params["total_think"]):
                                score, score_str = self.get_score_str(neg_response[j][0], think[j])

                                _query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\nReasoning Trace: {reasoning_trace}\nThink: {neg_response[j][0]}\n{score_str} {PromptTemplate.correct_think_prompts[j]}"
                                query, query_id, cquery, cquery_id, cquery_label = self.querize(_query, think[j])
                                    
                                # run model
                                model_output = self.model(cquery_id, labels=cquery_label)
                                loss_brs += model_output.loss / self.params["total_think"]
                                reasoning_trace = think[j] if j == 0 else reasoning_trace + " " + think[j]
                                
                            self.optimizer_r.zero_grad()
                            loss_brs.backward()
                            self.optimizer_r.step()    

                        loss_r = loss_fr + loss_brb + loss_brs

                        """ Base-2: Action Optimize """
                        self.model.set_adapter('planning_policy')
                        loss_p = 0.
                        ########################################
                        
                        reasoning_trace = ""
                        for j in range(self.params["total_think"]+1):
                            _query = self.plan_template(instruction=instruction, state=state, think=reasoning_trace, history=history)["query"]
                            query, query_id, cquery, cquery_id, cquery_label = self.querize(_query, action)

                            # run model
                            model_output = self.model(cquery_id, labels=cquery_label)
                            loss_p += model_output.loss / (self.params["total_think"] + 1)
                            if j < self.params["total_think"]:
                                reasoning_trace = think[j] if j == 0 else reasoning_trace + " " + think[j]

                        self.optimizer_p.zero_grad()
                        loss_p.backward()
                        self.optimizer_p.step()

                        all_stats.append({
                            "loss/reasoning_policy": loss_r,
                            "loss/reasoning_policy/foward": loss_fr,
                            "loss/reasoning_policy/backward_base": loss_brb,
                            "loss/reasoning_policy/backward_self": loss_brs,
                            "loss/planning_policy": loss_p,
                        })
                                                    
            ######################  logging !!!!
            train_stats = stack_dicts(all_stats)
            stats = {}
            for k, v in train_stats.items():
                stats[f'{k}'] = torch.mean(v, axis=0)
            stats = stats_to_np(stats)
            stats_print(stats)

            ######################  exploration !!
            count = None
            if epoch != self.params["total_epochs"]: 
                self._setup_init_response()
                count = self.explore()

            if count is not None:
                self.prev_neg_percent = count / (len(self.dataset) * self.params["total_think"])
            else:
                self.prev_neg_percent = 0.0
            stats["explore/neg_percent"] = self.prev_neg_percent
            print_warn(f"explore/neg_precent: {self.prev_neg_percent}")
            wandb.log(stats)

            if epoch % 5 == 0:
                output_dir = os.path.join(self.output_dir, f"checkpoint_{epoch}")
                self.save_pretrained(output_dir)

    def querize(self, _query, response):
        if 'instruct' in self.pre_model_name or 'Instruct' in self.pre_model_name:
            query = [
                {"role": "user", "content": _query}]
            query_id = self.tokenizer.apply_chat_template(
                query, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
            cquery = [
                {"role": "user", "content": _query},
                {"role": "assistant", "content": response}]
            cquery_id = self.tokenizer.apply_chat_template(
                cquery, tokenize=True, return_tensors="pt").to(self.device)
        else:
            query = f"### Human: {_query}\n### Assistant:"
            query_id = self.tokenizer.encode(query, return_tensors="pt").to(self.device)

            cquery = f"### Human: {_query}\n### Assistant: {response}\n"
            cquery_id = self.tokenizer.encode(cquery, return_tensors="pt").to(self.device)

        # masking user part
        cquery_label = cquery_id.clone()
        cquery_label[:, :len(query_id[0])] = -100

        return query, query_id, cquery, cquery_id, cquery_label

    def explore(self):
        self.model.set_adapter('reasoning_policy')

        count = 0
        for batch in tqdm.tqdm(self.dataloader, desc="explore batch"): # iterate over dataloader
            for i in range(self.params["batch_size"]):
                instruction, state, history, idx = \
                    batch["instructions"][i], PromptTemplate.preprocess(batch["states"][i]), batch["histories"][i], batch["indexes"][i]
                init_response, ground_response = self.init_response[idx], batch["think_lists"][i]
                neg_response = [None for _ in range(self.params["total_think"])]
                reasoning_trace = ""
                for j in range(self.params["total_think"]):
                    score, score_str = self.get_score_str(init_response[j], ground_response[j])

                    _query = f"Instruction: {instruction}\nState: {state}\nPrevious Actions: {history}\nReasoning Trace: {reasoning_trace}\nThink: {init_response[j]}\n{score_str} {PromptTemplate.correct_think_prompts[j]}"
                    if 'instruct' in self.pre_model_name or 'Instruct' in self.pre_model_name:
                        query = [
                            {"role": "user", "content": _query}]
                        query_id = self.tokenizer.apply_chat_template(
                        query, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
                    else:
                        query = f"### Human: {_query}\n### Assistant:"
                        query_id = self.tokenizer.encode(query, return_tensors="pt").to(self.device)
                
                    with torch.no_grad():
                        response_text = self.generate(
                            self.model, self.tokenizer, query_id, self.gen2_params, self.params["max_think_token"])

                    # process response text
                    if 'instruct' in self.pre_model_name or 'Instruct' in self.pre_model_name:
                        response_text = response_text.split(". ")[0] 
                    else:
                        response_text = response_text.strip().split("\n")[0]
                        response_text = response_text.split(". ")[0] 

                    response_text = response_text + "." if response_text and response_text[-1] != '.' else response_text
                    
                    cossim_score_i = self.cossim_pipe(init_response[j], ground_response[j])
                    tfidf_score_i = self.tfidf_pipe(init_response[j], ground_response[j])
                    cossim_score_c = self.cossim_pipe(response_text, ground_response[j])
                    tfidf_score_c = self.tfidf_pipe(response_text, ground_response[j])
                   
                    ### collect all responses as neg
                    neg_response[j] = (response_text, cossim_score_c, tfidf_score_c)
                    ###

                    if cossim_score_i > cossim_score_c or tfidf_score_i > tfidf_score_c:
                        count += 1
                    
                    reasoning_trace = response_text if j == 0 else reasoning_trace + " " + response_text

                self.neg_response[idx].append(neg_response)
        return count

    def generate(self, model, tokenizer, query_id, gen_params, max_token):
        response_id = model.generate(
            query_id,
            **gen_params,
            max_new_tokens=max_token,
            pad_token_id=tokenizer.eos_token_id,
        )
        response_text = tokenizer.decode(
            response_id.squeeze()[len(query_id[0]):], skip_special_tokens=True)
        return response_text

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
