"""
Task routing module — matches the reference repo's mechanism exactly.

Each task has:
  - prompt_key: [1, d_model] learned identity vector
  - trans_input: nn.Sequential MLP (d_model → hidden → d_model, SiLU activations)

Routing:
  1. Pool input embeddings → [B, 1, D]
  2. Each task's MLP transforms the pooled embedding → [B, 1, D]
  3. Cosine similarity between each task's key and its MLP output → [B, 1, 1]
  4. V-shaped activation: |2*sigmoid(4*score) - 1| → [B, 1, 1]
  5. Stack across tasks → [B, n_tasks, 1]

Old tasks' keys and MLPs are frozen. New task gets fresh key + MLP.
"""
import math
import torch
from torch import nn


class TaskRouter(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 100):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Current task's components (task 0 at init)
        self.current_key = nn.Parameter(torch.randn(1, d_model))
        nn.init.uniform_(self.current_key, -1, 1)

        self.current_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model, bias=False),
            nn.SiLU(),
        )

        # Old tasks' components (None until add_task is called)
        self.old_keys = None       # nn.Parameter [n_old, d_model], frozen
        self.old_mlps = None       # nn.ModuleList of nn.Sequential, frozen

        self.n_tasks = 1

        # For GPM activation collection
        self.get_features = False
        self._cached_pooled = None
        self._cached_medium = None

    def _v_shaped(self, scores):
        """Paper equation 8: f(b) = |2*sigmoid(b) - 1|
        With the *4 scaling from their cal_attention."""
        return torch.abs(2.0 * torch.sigmoid(scores * 4.0) - 1.0)

    def _cosine_sim_weight(self, key, mlp_out):
        """Compute cosine similarity then V-shaped activation.
        key: [B, 1, D], mlp_out: [B, 1, D] → returns [B, 1, 1]"""
        key_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
        mlp_norm = mlp_out / (mlp_out.norm(dim=-1, keepdim=True) + 1e-8)
        score = (key_norm * mlp_norm).sum(dim=-1, keepdim=True)  # [B, 1, 1]
        return self._v_shaped(score)

    def forward(self, pooled_embeds: torch.Tensor) -> torch.Tensor:
        """
        pooled_embeds: [B, 1, D] — masked-mean of input embeddings.
        Returns: [B, n_tasks, 1] — scalar routing weight per task.
        """
        B = pooled_embeds.size(0)
        device = pooled_embeds.device

        # Current task's MLP + key
        # Run MLP in two stages so we can collect intermediate activations for GPM
        medium = self.current_mlp[1](self.current_mlp[0](pooled_embeds))  # after first linear + SiLU
        current_out = self.current_mlp[3](self.current_mlp[2](medium))     # after second linear + SiLU

        if self.get_features:
            self._cached_pooled = pooled_embeds.detach()
            self._cached_medium = medium.detach()

        current_key = self.current_key.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
        current_weight = self._cosine_sim_weight(current_key, current_out)  # [B, 1, 1]

        if self.old_keys is None:
            # Single task — just return current weight
            return current_weight

        # Old tasks' MLPs + keys
        old_weights = []
        for i in range(self.old_keys.size(0)):
            old_key = self.old_keys[i:i+1].unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
            old_out = self.old_mlps[i](pooled_embeds)                        # [B, 1, D]
            w = self._cosine_sim_weight(old_key, old_out)                    # [B, 1, 1]
            old_weights.append(w)

        # Combine: [current_weight, old_weight_0, old_weight_1, ...]
        # But we need ordering to match LoRA ordering: old tasks first, current last
        # In their code: cat([current, previous], dim=1) and LoRAs are [current, prev...]
        # So: task 0 = current (during task 0), then after add_task task 0 = first old, task 1 = current
        # Ordering: old_0, old_1, ..., current
        all_weights = old_weights + [current_weight]
        return torch.cat(all_weights, dim=1)  # [B, n_tasks, 1]

    def add_task(self):
        """Move current task's key+MLP to frozen storage, create fresh ones for new task."""
        # Freeze current key and MLP
        frozen_key = self.current_key.data.clone()
        frozen_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.d_model, bias=False),
            nn.SiLU(),
        )
        frozen_mlp.load_state_dict(self.current_mlp.state_dict())
        for p in frozen_mlp.parameters():
            p.requires_grad = False
        device = self.current_key.device
        frozen_mlp = frozen_mlp.to(device)

        if self.old_keys is None:
            self.old_keys = nn.Parameter(frozen_key.unsqueeze(0), requires_grad=False)  # [1, D]
            self.old_mlps = nn.ModuleList([frozen_mlp])
        else:
            new_keys = torch.cat([self.old_keys.data, frozen_key.unsqueeze(0)], dim=0)
            self.old_keys = nn.Parameter(new_keys, requires_grad=False)
            self.old_mlps.append(frozen_mlp)

        # Fresh current key + MLP for the new task
        self.current_key = nn.Parameter(torch.randn(1, self.d_model, device=device))
        nn.init.uniform_(self.current_key, -1, 1)

        self.current_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.d_model, bias=False),
            nn.SiLU(),
        ).to(device)

        self.n_tasks += 1
        return self.n_tasks - 1  # new task index

    def get_trainable_params(self):
        """Only current task's key + MLP are trainable."""
        params = [self.current_key]
        params.extend(self.current_mlp.parameters())
        return params

    def save_prompt_keys(self, path):
        """Save all keys (old + current) for loading as previous_prompts_keys."""
        if self.old_keys is not None:
            all_keys = torch.cat([self.old_keys.data, self.current_key.data.unsqueeze(0)], dim=0)
        else:
            all_keys = self.current_key.data.unsqueeze(0)
        torch.save(all_keys, path)


def pool_encoder_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling → [B, 1, D] (keepdim for compatibility with MLP input)."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (mask * hidden).sum(dim=1, keepdim=True)  # [B, 1, D]
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return pooled / counts
