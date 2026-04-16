"""
Gating module — adapted from the reference repo's `Trans_input` class.

Architecture:
    Input:  pooled encoder hidden state  [batch, d_model]
    Output: per-task gate weights        [batch, n_tasks, d_model]

Two parameter tensors, packed across tasks:
    input_linear:  [n_tasks, hidden_dim, d_model]   (Kaiming init)
    output_linear: [n_tasks, d_model, hidden_dim]   (Kaiming init)

Forward (per task t in parallel via batched matmul):
    h_t = SiLU(x @ input_linear[t]^T)        → [batch, hidden_dim]
    y_t = SiLU(h_t @ output_linear[t]^T)     → [batch, d_model]
    output = stack along task dim            → [batch, n_tasks, d_model]

Note: this produces a per-dimension weighting per task. The reference repo
does NOT use the paper's `|2 * sigmoid(b) - 1|` activation. We follow the
code (which produces their reported numbers), not the paper.

We use einsum to make the per-task batched matmul explicit and correct
for any combination of batch size and n_tasks (the reference repo's
matmul-with-permute formulation has a broadcasting bug when batch > 1
and n_tasks > 1).
"""
import math
import torch
from torch import nn


class GatingModule(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 100, n_tasks: int = 1):
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

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        pooled: [batch, d_model]
        returns: [batch, n_tasks, d_model]

        einsum form:
          h[b, t, k] = sum_d  pooled[b, d] * input_linear[t, k, d]
          y[b, t, d] = sum_k  h[b, t, k]   * output_linear[t, d, k]
        """
        # First layer: project to hidden_dim, per task
        h = torch.einsum('bd,tkd->btk', pooled, self.input_linear)
        h = self.activation(h)
        # Second layer: project back to d_model, per task
        y = torch.einsum('btk,tdk->btd', h, self.output_linear)
        y = self.activation(y)
        return y

    def add_task(self):
        """Grow the parameter tensors by one task slot.
        Returns the new task index (0-based)."""
        old_n = self.n_tasks
        d_model = self.d_model
        hidden_dim = self.hidden_dim

        new_input = torch.zeros((old_n + 1, hidden_dim, d_model),
                                device=self.input_linear.device,
                                dtype=self.input_linear.dtype)
        new_output = torch.zeros((old_n + 1, d_model, hidden_dim),
                                 device=self.output_linear.device,
                                 dtype=self.output_linear.dtype)

        with torch.no_grad():
            new_input[:old_n]  = self.input_linear.data
            new_output[:old_n] = self.output_linear.data
            nn.init.kaiming_uniform_(new_input[old_n].view(-1, d_model),     a=math.sqrt(3))
            nn.init.kaiming_uniform_(new_output[old_n].view(-1, hidden_dim), a=math.sqrt(3))

        self.input_linear  = nn.Parameter(new_input)
        self.output_linear = nn.Parameter(new_output)
        self.n_tasks = old_n + 1

        return old_n


def pool_encoder_hidden(encoder_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool encoder hidden states with attention masking.
    encoder_hidden: [batch, seq_len, d_model]
    attention_mask: [batch, seq_len]
    returns:        [batch, d_model]
    """
    mask = attention_mask.unsqueeze(-1).to(encoder_hidden.dtype)
    summed = (encoder_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
