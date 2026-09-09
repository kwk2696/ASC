# 
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import GPT2Model, GPT2LMHeadModel
from embodied_cd.trl.models.type_aliases import CausalLMOutputWithCrossAttentionsAndValue
from embodied_cd.trl.models.value import ValueHead


class GPT2HeadWithValueModel(GPT2LMHeadModel):
    """The GPT2HeadWithValueModel class implementation."""
    def __init__(self, config, activation_fn):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.vl_head = ValueHead(config.hidden_size, pdrop=0.1, activation_fn=activation_fn, detach=True)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        average_pool: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentionsAndValue]:
        r"""
        lbels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if average_pool: # average pooling only for the response part
            hidden_states = hidden_states[:,-num_logits_to_keep:,:]
            hidden_states = torch.mean(hidden_states, dim=1).unsqueeze(0)
        logits = self.lm_head(hidden_states)
        values = self.vl_head(hidden_states).squeeze(-1)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[:,-num_logits_to_keep:,:]
            shift_logits = shift_logits[..., :-1, :].contiguous()
            shift_labels = labels[:,-num_logits_to_keep:]
            shift_labels = shift_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentionsAndValue(
            loss=loss,
            logits=logits,
            values=values,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
