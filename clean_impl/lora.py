"""
LoRA layer matching the reference repo's LoRALayer in t5_gainlora_inflora.py.

A standard low-rank adapter:  y = (dropout(x) @ A^T @ B^T) * scaling
  - A: [r, in_features]   — Kaiming-uniform init
  - B: [out_features, r]  — zero init (so initial output is zero)
  - scaling = lora_alpha / r

Output is added to a frozen base linear layer's output.
"""
import math
import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.scaling = self.lora_alpha / self.r

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_features]
        # returns: [batch, seq_len, out_features]
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
