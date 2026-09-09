#

import torch
from torch import nn

from embodied_cd.common.print_utils import *


class ValueHead(nn.Module):
    """The ValueHead class implemntation."""
    def __init__(self, hidden_size: int, pdrop: float = 0.0, activation_fn: str = 'identity', detach=True):
        super().__init__()
        print_warn(f"[Model: ValueHead] Use {activation_fn} activation function and detach is {detach}") 
        self.detach = detach # whether to backpropagate through LM
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Dropout(pdrop),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=True),
            nn.Sigmoid() if activation_fn == 'sigmoid' else nn.Identity()
        )

    def forward(self, hidden_states):
        if self.detach:
            hidden_states = hidden_states.detach()
        return self.proj_head(hidden_states)


class ValueHeadWithLogit(nn.Module):
    def __init__(self, hidden_size: int, pdrop: float = 0.0, activation_fn: str = 'identity', detach=True):
        super().__init__()
        print_warn(f"[Model: ValueHeadWithLogit] Use {activation_fn} activation function and detach is {detach}") 
        self.detach = detach # whether to backpropagate through LM
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_size+1, hidden_size, bias=True),
            nn.Dropout(pdrop),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=True),
            nn.Sigmoid() if activation_fn == 'sigmoid' else nn.Identity()
        )

    def forward(self, hidden_states, logits):
        if self.detach:
            hidden_states = hidden_states.detach()
        return self.proj_head(torch.cat((hidden_states, logits), dim=-1))
