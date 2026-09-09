#
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers import LlamaModel, LlamaForCausalLM
from embodied_cd.trl.models.type_aliases import CausalLMOutputWithPastAndValue
from embodied_cd.trl.models.value import ValueHead


class LlamaHeadWithValueModel(LlamaForCausalLM):
    """The LlamaHeadWithValue Model class implementation."""
    def __init__(self, config, activation_fn):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vl_head = ValueHead(config.hidden_size, pdrop=0.1, activation_fn=activation_fn, detach=True)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        average_pool: bool = False,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastAndValue]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        if average_pool: # average pooling only for the response part
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
            hidden_states = torch.mean(hidden_states, dim=1).unsqueeze(0)
        logits = self.lm_head(hidden_states)
        values = self.vl_head(hidden_states).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits[:,-num_logits_to_keep:,:], labels[:,-num_logits_to_keep:], self.vocab_size, **loss_kwargs) 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndValue(
            loss=loss,
            logits=logits,
            values=values,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.past_key_values,
            attentions=outputs.attentions,
        )

