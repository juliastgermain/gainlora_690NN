"""
Gating module — exact implementation of paper Section 3.1 and Appendix B.3.

Per-task gating module g_i(x) (paper eq 3):
  p_{i,0} = p0 = Pool(Token(x))           [B, d]         — shared input
  p_{i,1} = σ(G_{i,1} · p_{i,0})          [B, h=100]     — G_{i,1} ∈ R^{100×1024}
  p_{i,2} = σ(G_{i,2} · p_{i,1})          [B, d=1024]    — G_{i,2} ∈ R^{1024×100}
  g_i(x)  = f(G_{i,3} · p_{i,2})          scalar [B, 1]  — G_{i,3} ∈ R^{1×1024}

where f(b) = |2·sigmoid(b) − 1|  (paper eq 8), which satisfies f(0)=0, range [0,1].

NOTE: G_{i,3} maps to a SCALAR — NOT cosine similarity. This is the critical
difference from the previous implementation. The paper's final layer is a linear
dot product, not a cosine distance with a learned key vector.

Initialization (paper eq 9):
  When adding task t, Init(G_{t,1}) ← G_{t-1,1}, Init(G_{t,2}) ← G_{t-1,2}
  Init(G_{t,3}): random, then projected ⊥ subspace of old p_{t,2} activations (eq 10).

MultiTaskRouter manages all per-task gating modules.
"""
import torch
from torch import nn


def _f(score: torch.Tensor) -> torch.Tensor:
    """Paper eq 8: f(b) = |2·sigmoid(b) − 1|.  Maps R→[0,1], f(0)=0."""
    return torch.abs(2.0 * torch.sigmoid(score) - 1.0)


class SingleTaskGate(nn.Module):
    """
    Gating module for ONE task.
    Appendix B.3: G_{i,1} ∈ R^{100×d}, G_{i,2} ∈ R^{d×100}, G_{i,3} ∈ R^{1×d}
    """
    def __init__(self, d_model: int, hidden_dim: int = 100):
        super().__init__()
        self.G1 = nn.Linear(d_model, hidden_dim, bias=False)  # [h, d]
        self.G2 = nn.Linear(hidden_dim, d_model, bias=False)  # [d, h]
        self.G3 = nn.Linear(d_model, 1, bias=False)           # [1, d] → scalar
        self.activation = nn.SiLU()

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        """p0: [B, d] → scalar weight [B, 1]."""
        p1 = self.activation(self.G1(p0))   # [B, h]
        p2 = self.activation(self.G2(p1))   # [B, d]
        return _f(self.G3(p2))              # [B, 1]

    def get_activations(self, p0: torch.Tensor):
        """Return (p1, p2) for GPM activation collection."""
        p1 = self.activation(self.G1(p0))
        p2 = self.activation(self.G2(p1))
        return p1, p2


class MultiTaskRouter(nn.Module):
    """
    Manages one SingleTaskGate per task.

    forward(p0) → [B, n_tasks, 1]  scalar weight per task.
    Ordering: [old_0, old_1, ..., current_task]
    """
    def __init__(self, d_model: int, hidden_dim: int = 100):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.current_gate = SingleTaskGate(d_model, hidden_dim)
        self.old_gates = nn.ModuleList()   # all frozen
        self.n_tasks = 1

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        """p0: [B, d] → [B, n_tasks, 1]."""
        current_w = self.current_gate(p0)  # [B, 1]

        if len(self.old_gates) == 0:
            return current_w.unsqueeze(1)  # [B, 1, 1]

        old_ws = [g(p0) for g in self.old_gates]   # list of [B, 1]
        all_ws = old_ws + [current_w]               # old first, current last
        return torch.stack(all_ws, dim=1)           # [B, n_tasks, 1]

    def add_task(self) -> int:
        """
        Freeze current gate → move to old_gates.
        Create new gate:
          - G1, G2 copied from current (paper eq 9)
          - G3 random-initialized (GPM will project it before training, paper eq 10)
        Returns index of the new task.
        """
        device = next(self.current_gate.parameters()).device

        # Freeze
        for p in self.current_gate.parameters():
            p.requires_grad = False
        self.old_gates.append(self.current_gate)

        # New gate
        new_gate = SingleTaskGate(self.d_model, self.hidden_dim).to(device)
        # Paper eq 9: copy G1 and G2 from previous gate
        new_gate.G1.weight.data.copy_(self.old_gates[-1].G1.weight.data)
        new_gate.G2.weight.data.copy_(self.old_gates[-1].G2.weight.data)
        # G3: leave as random init; GPM will project it in project_init_G3()
        self.current_gate = new_gate

        self.n_tasks += 1
        return self.n_tasks - 1   # index of newly added task

    def get_trainable_params(self):
        """Only current task's gate is trainable."""
        return list(self.current_gate.parameters())


def pool_encoder_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean pooling → [B, d_model].
    Note: returns 2D (no keepdim), matching the gating module's expected input.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (mask * hidden).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return pooled / counts   # [B, d]
