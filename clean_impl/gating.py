"""Gating module — adapted from the reference repo Trans_input class."""
import math, torch
from torch import nn

class GatingModule(nn.Module):
    def __init__(self, d_model, hidden_dim=100, n_tasks=1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        self.input_linear  = nn.Parameter(torch.zeros((n_tasks, hidden_dim, d_model)))
        self.output_linear = nn.Parameter(torch.zeros((n_tasks, d_model, hidden_dim)))
        self.activation = nn.SiLU()
        self._init_params()
    def _init_params(self):
        nn.init.kaiming_uniform_(self.input_linear.view(-1, self.d_model), a=math.sqrt(3))
        nn.init.kaiming_uniform_(self.output_linear.view(-1, self.hidden_dim), a=math.sqrt(3))
    def forward(self, pooled):
        h = torch.einsum('bd,tkd->btk', pooled, self.input_linear)
        h = self.activation(h)
        y = torch.einsum('btk,tdk->btd', h, self.output_linear)
        y = self.activation(y)
        return y
    def add_task(self):
        old_n = self.n_tasks
        new_input = torch.zeros((old_n+1, self.hidden_dim, self.d_model),
                                device=self.input_linear.device, dtype=self.input_linear.dtype)
        new_output = torch.zeros((old_n+1, self.d_model, self.hidden_dim),
                                 device=self.output_linear.device, dtype=self.output_linear.dtype)
        with torch.no_grad():
            new_input[:old_n] = self.input_linear.data
            new_output[:old_n] = self.output_linear.data
            nn.init.kaiming_uniform_(new_input[old_n].view(-1, self.d_model), a=math.sqrt(3))
            nn.init.kaiming_uniform_(new_output[old_n].view(-1, self.hidden_dim), a=math.sqrt(3))
        self.input_linear  = nn.Parameter(new_input)
        self.output_linear = nn.Parameter(new_output)
        self.n_tasks = old_n + 1
        return old_n

def pool_encoder_hidden(encoder_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(encoder_hidden.dtype)
    summed = (encoder_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
