"""
Gradient Projection Memory (GPM) for the TaskRouter.

Collects three activation sets after training each task:
  gate_input:  p0       pooled embeddings    [N, D]  -> projects G_{t,1} grads
  gate_hidden: p_{t,1}  after 1st MLP layer  [N, H]  -> projects G_{t,2} grads
  gate_output: p_{t,2}  after 2nd MLP layer  [N, D]  -> projects key grads + init projection

Three operations:
  1. collect_activations() — forward pass, cache activations
  2. update_bases()        — SVD on residual to extend orthonormal bases
  3. project_init_key()    — project new key ⊥ old output subspace (paper eq 10)
  4. make_projection_hook()— callable that projects all gate grads (paper eq 12)
"""
import torch
from torch.utils.data import DataLoader


class GPM:
    def __init__(self, threshold: float = 0.995):
        self.threshold = threshold
        self.bases = {}  # keys: gate_input, gate_hidden, gate_output

    @torch.no_grad()
    def collect_activations(self, model, dataloader):
        """Collect p0, p_{t,1}, p_{t,2} from current task's MLP."""
        model.eval()
        gate_inputs, gate_hiddens, gate_outputs = [], [], []
        from gating import pool_encoder_hidden

        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            # p0: pooled input embeddings
            embeds = model.shared(input_ids)
            pooled = pool_encoder_hidden(embeds, attention_mask)  # [B, 1, D]
            gate_inputs.append(pooled.squeeze(1).cpu().float())

            # p_{t,1}: after first linear + SiLU
            mlp = model.router.current_mlp
            medium = mlp[1](mlp[0](pooled))   # [B, 1, H]
            gate_hiddens.append(medium.squeeze(1).cpu().float())

            # p_{t,2}: after second linear + SiLU (MLP final output)
            output = mlp[3](mlp[2](medium))   # [B, 1, D]
            gate_outputs.append(output.squeeze(1).cpu().float())

        return {
            "gate_input":  torch.cat(gate_inputs,  dim=0),
            "gate_hidden": torch.cat(gate_hiddens, dim=0),
            "gate_output": torch.cat(gate_outputs, dim=0),
        }

    @torch.no_grad()
    def update_bases(self, activations: dict):
        """Extend orthonormal bases via SVD on activation residuals."""
        for key in ["gate_input", "gate_hidden", "gate_output"]:
            A = activations[key].T  # [feature_dim, N]

            if key not in self.bases or self.bases[key] is None:
                residual = A
                old_explained = 0.0
            else:
                M = self.bases[key]
                residual = A - M @ (M.T @ A)
                old_explained = ((M.T @ A) ** 2).sum()

            cov = residual @ residual.T
            U, S, _ = torch.linalg.svd(cov)
            total_var = S.sum()
            if total_var < 1e-10:
                print(f"  GPM [{key}]: near-zero variance, skipping")
                continue

            full_var = (A ** 2).sum()
            target = self.threshold * full_var
            cumvar = torch.cumsum(S, dim=0)
            n_new = min(5, len(S))  # fallback
            for i in range(len(S)):
                if cumvar[i] + old_explained >= target:
                    n_new = i + 1
                    break

            new_bases = U[:, :n_new]
            if key not in self.bases or self.bases[key] is None:
                self.bases[key] = new_bases
            else:
                self.bases[key] = torch.cat([self.bases[key], new_bases], dim=1)

            print(f"  GPM [{key}]: +{n_new} bases "
                  f"(total {self.bases[key].size(1)}, dim={self.bases[key].size(0)})")

    @torch.no_grad()
    def project_init_key(self, model):
        """
        Paper eq 10: project new key ⊥ gate_output subspace so that
        g_t(x) ≈ 0 for all old-task inputs x before task-t training starts.
        """
        if "gate_output" not in self.bases or self.bases["gate_output"] is None:
            print("  GPM: no gate_output bases yet, skipping key projection")
            return
        M = self.bases["gate_output"].to(model.router.current_key.device)  # [D, k]
        key = model.router.current_key.data  # [1, D]
        key_t = key.T                         # [D, 1]
        model.router.current_key.data = (key_t - M @ (M.T @ key_t)).T
        # Verify
        key_n = model.router.current_key.data / (model.router.current_key.data.norm() + 1e-8)
        max_cos = (key_n @ M).abs().max().item()
        print(f"  GPM: projected init key ⊥ gate_output "
              f"(max cosine with bases: {max_cos:.2e})")

    def make_projection_hook(self, model):
        """
        Paper eq 12: before each optimizer.step(), project gradients of
        G_{t,1}, G_{t,2}, G_{t,3} orthogonal to their respective bases.
        """
        def _project():
            with torch.no_grad():
                # G_{t,1}: weight [hidden, D], bases [D, k]
                if "gate_input" in self.bases and self.bases["gate_input"] is not None:
                    w = model.router.current_mlp[0]
                    if w.weight.grad is not None:
                        M = self.bases["gate_input"].to(w.weight.grad.device)
                        g = w.weight.grad   # [H, D]
                        w.weight.grad = g - g @ M @ M.T

                # G_{t,2}: weight [D, hidden], bases [hidden, k]
                if "gate_hidden" in self.bases and self.bases["gate_hidden"] is not None:
                    w = model.router.current_mlp[2]
                    if w.weight.grad is not None:
                        M = self.bases["gate_hidden"].to(w.weight.grad.device)
                        g = w.weight.grad   # [D, H]
                        w.weight.grad = g - g @ M @ M.T

                # G_{t,3} (key): [1, D], bases [D, k]
                if "gate_output" in self.bases and self.bases["gate_output"] is not None:
                    if model.router.current_key.grad is not None:
                        M = self.bases["gate_output"].to(model.router.current_key.grad.device)
                        g = model.router.current_key.grad  # [1, D]
                        model.router.current_key.grad = g - g @ M @ M.T
        return _project
