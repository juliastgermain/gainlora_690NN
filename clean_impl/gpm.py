"""
GPM for the TaskRouter — collects activations from the current task's MLP
and projects gradients to be orthogonal to old-task subspaces.

Two activation sets:
  - gate_input: pooled embeddings [N, d_model] — for projecting current_mlp[0] grads
  - gate_hidden: intermediate after first linear+SiLU [N, hidden_dim] — for projecting current_mlp[2] grads
"""
import torch
from torch.utils.data import DataLoader


class GPM:
    def __init__(self, threshold=0.995):
        self.threshold = threshold
        self.bases = {}

    @torch.no_grad()
    def collect_activations(self, model, dataloader):
        model.eval()
        gate_inputs = []
        gate_hiddens = []
        from gating import pool_encoder_hidden
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            input_embeds = model.shared(input_ids)
            pooled = pool_encoder_hidden(input_embeds, attention_mask)  # [B, 1, D]
            gate_inputs.append(pooled.squeeze(1).cpu().float())        # [B, D]
            # Get intermediate from current MLP
            medium = model.router.current_mlp[1](model.router.current_mlp[0](pooled))
            gate_hiddens.append(medium.squeeze(1).cpu().float())       # [B, H]
        return {
            "gate_input": torch.cat(gate_inputs, dim=0),
            "gate_hidden": torch.cat(gate_hiddens, dim=0),
        }

    @torch.no_grad()
    def update_bases(self, activations):
        for key in ["gate_input", "gate_hidden"]:
            A = activations[key].T  # [feature_dim, N]
            if key not in self.bases or self.bases[key] is None:
                residual = A
            else:
                M = self.bases[key]
                residual = A - M @ (M.T @ A)
            cov = residual @ residual.T
            U, S, _ = torch.linalg.svd(cov)
            total_var = S.sum()
            if total_var < 1e-10:
                continue
            cumvar = torch.cumsum(S, dim=0)
            if key in self.bases and self.bases[key] is not None:
                old_explained = ((self.bases[key].T @ A) ** 2).sum()
            else:
                old_explained = 0.0
            full_var = (A ** 2).sum()
            target = self.threshold * full_var
            n_new = 0
            for i in range(len(S)):
                if cumvar[i] + old_explained >= target:
                    n_new = i + 1
                    break
            if n_new == 0:
                n_new = min(5, len(S))
            new_bases = U[:, :n_new]
            if key not in self.bases or self.bases[key] is None:
                self.bases[key] = new_bases
            else:
                self.bases[key] = torch.cat([self.bases[key], new_bases], dim=1)
            print(f"  GPM [{key}]: {n_new} bases, total {self.bases[key].size(1)} (dim={self.bases[key].size(0)})")

    def make_projection_hook(self, model):
        def _project():
            with torch.no_grad():
                # Project current_mlp[0].weight gradients (shape [hidden, d_model])
                if "gate_input" in self.bases and self.bases["gate_input"] is not None:
                    w = model.router.current_mlp[0]  # nn.Linear
                    if w.weight.grad is not None:
                        M = self.bases["gate_input"].to(w.weight.grad.device)
                        g = w.weight.grad  # [hidden, d_model]
                        proj = g @ M @ M.T
                        w.weight.grad = g - proj

                # Project current_mlp[2].weight gradients (shape [d_model, hidden])
                if "gate_hidden" in self.bases and self.bases["gate_hidden"] is not None:
                    w = model.router.current_mlp[2]
                    if w.weight.grad is not None:
                        M = self.bases["gate_hidden"].to(w.weight.grad.device)
                        g = w.weight.grad  # [d_model, hidden]
                        proj = g @ M @ M.T
                        w.weight.grad = g - proj
        return _project
