"""
TaskRouter — matches the GainLoRA paper's routing mechanism exactly.

Each task t has:
  - current_mlp: two-layer MLP (G_{t,1}, G_{t,2})  [trainable for current task]
  - current_key: prompt_key vector G_{t,3}           [trainable for current task]

Routing weight for task t on input x:
  p0      = Pool(Embed(x))                    [B, 1, D]
  p_{t,1} = SiLU(G_{t,1}(p0))                [B, 1, hidden]
  p_{t,2} = SiLU(G_{t,2}(p_{t,1}))           [B, 1, D]
  score   = cosine(key_t, p_{t,2})            [B, 1, 1]
  weight  = |2*sigmoid(4*score) - 1|          [B, 1, 1]  (paper eq 8)

add_task():
  - Freezes current key+MLP (becomes old task's components)
  - Creates NEW MLP as a COPY of current MLP (paper eq 9, preserves shared structure)
  - Creates NEW key with random init (will be projected by GPM eq 10)
"""
import torch
from torch import nn


class TaskRouter(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 100):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.current_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model, bias=False),
            nn.SiLU(),
        )
        self.current_key = nn.Parameter(torch.empty(1, d_model))
        nn.init.uniform_(self.current_key, -1, 1)

        self.old_keys = None   # nn.Parameter [n_old, d_model], frozen
        self.old_mlps = None   # nn.ModuleList of frozen nn.Sequential
        self.n_tasks = 1

    def _v_shaped(self, scores):
        """f(b) = |2*sigmoid(4b) - 1|  —  range [0,1], paper eq 8."""
        return torch.abs(2.0 * torch.sigmoid(scores * 4.0) - 1.0)

    def _cosine_weight(self, key, mlp_out):
        """Cosine similarity + V-shaped activation -> [B, 1, 1]."""
        key_n = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
        out_n = mlp_out / (mlp_out.norm(dim=-1, keepdim=True) + 1e-8)
        score = (key_n * out_n).sum(dim=-1, keepdim=True)
        return self._v_shaped(score)

    def _run_mlp(self, mlp, pooled):
        """Run MLP, return (final_output, intermediate_after_first_layer).
        pooled: [B, 1, D]"""
        medium = mlp[1](mlp[0](pooled))   # [B, 1, hidden]
        output = mlp[3](mlp[2](medium))   # [B, 1, D]
        return output, medium

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        pooled: [B, 1, D]  (masked-mean of input embeddings)
        Returns: [B, n_tasks, 1]  scalar routing weight per task.
        Ordering: [old_0, old_1, ..., current]
        """
        B = pooled.size(0)
        current_out, _ = self._run_mlp(self.current_mlp, pooled)
        current_key = self.current_key.unsqueeze(0).expand(B, 1, -1)
        current_w = self._cosine_weight(current_key, current_out)  # [B, 1, 1]

        if self.old_keys is None:
            return current_w  # [B, 1, 1]

        old_weights = []
        for i in range(self.old_keys.size(0)):
            old_key = self.old_keys[i:i+1].unsqueeze(0).expand(B, 1, -1)
            old_out, _ = self._run_mlp(self.old_mlps[i], pooled)
            old_weights.append(self._cosine_weight(old_key, old_out))

        return torch.cat(old_weights + [current_w], dim=1)  # [B, n_tasks, 1]

    def add_task(self):
        """
        Freeze current key+MLP. Create new key+MLP for next task.
        New MLP is a COPY of current (paper eq 9).
        New key is random (GPM will project it in eq 10).
        Returns index of new task.
        """
        device = self.current_key.device

        # Freeze current MLP
        frozen_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim, bias=False), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.d_model, bias=False), nn.SiLU(),
        )
        frozen_mlp.load_state_dict(self.current_mlp.state_dict())
        for p in frozen_mlp.parameters():
            p.requires_grad = False
        frozen_mlp = frozen_mlp.to(device)

        # Freeze current key
        frozen_key = self.current_key.data.clone()  # [1, D]

        # Store
        if self.old_keys is None:
            self.old_keys = nn.Parameter(frozen_key, requires_grad=False)
            self.old_mlps = nn.ModuleList([frozen_mlp])
        else:
            self.old_keys = nn.Parameter(
                torch.cat([self.old_keys.data, frozen_key], dim=0), requires_grad=False)
            self.old_mlps.append(frozen_mlp)

        # New MLP: COPY of current (paper eq 9)
        new_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim, bias=False), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.d_model, bias=False), nn.SiLU(),
        )
        new_mlp.load_state_dict(self.current_mlp.state_dict())
        self.current_mlp = new_mlp.to(device)

        # New key: fresh random (projected by GPM eq 10 before training)
        self.current_key = nn.Parameter(torch.empty(1, self.d_model, device=device))
        nn.init.uniform_(self.current_key, -1, 1)

        self.n_tasks += 1
        return self.n_tasks - 1  # index of newly added task

    def get_trainable_params(self):
        """Only current task's key + MLP are trainable."""
        return [self.current_key] + list(self.current_mlp.parameters())


def pool_encoder_hidden(hidden, attention_mask):
    """Masked mean pool -> [B, 1, D] (keepdim so it's compatible with MLP input)."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (mask * hidden).sum(dim=1, keepdim=True)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return pooled / counts
