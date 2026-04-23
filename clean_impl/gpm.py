"""
Gradient Projection Memory — paper Appendix A.1.

Collects inputs to each layer of the current task's gating module:
  M_{t,1}: inputs to G_{t,1} = p0        [N, d=1024]
  M_{t,2}: inputs to G_{t,2} = p_{t,1}  [N, h=100]
  M_{t,3}: inputs to G_{t,3} = p_{t,2}  [N, d=1024]

SVD criterion (paper eq 16):
  ||( H^c_{t,l} )_u ||^2_F + || M_{t,l} M_{t,l}^T H_{t,l} ||^2_F  >=  ε_th ||H_{t,l}||^2_F

Three operations exposed:
  1. collect_activations()    — forward pass, store p0/p1/p2
  2. update_bases()           — extend orthonormal bases via SVD
  3. project_init_G3()        — project G_{t,3} ⊥ M_{t,3} (paper eq 10)
  4. make_projection_hook()   — callable: project ΔG1/ΔG2/ΔG3 before each step (paper eq 12)
"""
import torch
from torch.utils.data import DataLoader


class GPM:
    def __init__(self, threshold: float = 0.97):
        # Paper uses ε_th from the original GPM paper [47]; 0.97 is the GPM default.
        # Higher = more bases = stronger protection but less room for new task routing.
        self.threshold = threshold
        self.bases = {}   # keys: 'M1', 'M2', 'M3'

    @torch.no_grad()
    def collect_activations(self, model, dataloader):
        """
        Collect p0, p_{t,1}, p_{t,2} using the current task's gate.
        Must be called BEFORE add_task() (while current_gate is still task t's gate).
        """
        model.eval()
        p0_list, p1_list, p2_list = [], [], []
        from gating import pool_encoder_hidden

        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            # p0: masked-mean of token embeddings [B, d]
            embeds = model.shared(input_ids)
            p0 = pool_encoder_hidden(embeds, attention_mask)
            p0_list.append(p0.cpu().float())

            # p1, p2: intermediate activations of current gate
            p1, p2 = model.router.current_gate.get_activations(p0)
            p1_list.append(p1.cpu().float())
            p2_list.append(p2.cpu().float())

        return {
            "M1": torch.cat(p0_list, dim=0),   # [N, d]  — inputs to G1
            "M2": torch.cat(p1_list, dim=0),   # [N, h]  — inputs to G2
            "M3": torch.cat(p2_list, dim=0),   # [N, d]  — inputs to G3
        }

    @torch.no_grad()
    def update_bases(self, activations: dict):
        """
        Extend orthonormal bases using paper eq 15-16.

        H_{t,l} has shape [dim_l, N] (columns = samples).
        Criterion: ||(H^c_u)||^2_F + ||M M^T H||^2_F >= ε_th ||H||^2_F
        """
        for key in ["M1", "M2", "M3"]:
            H = activations[key].T   # [dim, N]

            if key not in self.bases or self.bases[key] is None:
                Hc = H
                old_explained = 0.0
            else:
                M = self.bases[key]                  # [dim, k]
                proj = M @ (M.T @ H)                 # [dim, N]  — component in old subspace
                Hc = H - proj                        # paper eq 15: residual
                old_explained = (proj ** 2).sum().item()

            full_var = (H ** 2).sum().item()
            target = self.threshold * full_var

            if full_var < 1e-10:
                print(f"  GPM [{key}]: near-zero variance, skipping")
                continue

            # SVD on residual covariance (paper: SVD on H^c (H^c)^T)
            cov = Hc @ Hc.T      # [dim, dim]
            U, S, _ = torch.linalg.svd(cov)
            cumvar = torch.cumsum(S, dim=0)

            # Find minimum u satisfying eq 16
            n_new = len(S)   # fallback: take all
            for i in range(len(S)):
                if cumvar[i].item() + old_explained >= target:
                    n_new = i + 1
                    break

            new_bases = U[:, :n_new]   # [dim, n_new]

            if key not in self.bases or self.bases[key] is None:
                self.bases[key] = new_bases
            else:
                self.bases[key] = torch.cat([self.bases[key], new_bases], dim=1)

            print(f"  GPM [{key}]: +{n_new} bases "
                  f"(total {self.bases[key].size(1)}, feature_dim={self.bases[key].size(0)})")

    @torch.no_grad()
    def project_init_G3(self, model):
        """
        Paper eq 10: project Init(G_{t,3}) to be ⊥ to M_{t,3}.
        Ensures g_t(x) = f(0) = 0 for all old-task inputs before task-t training.

        G3.weight shape: [1, d_model]. Bases M3 shape: [d_model, k].
        Operation: G3 ← G3 - G3 @ M3 @ M3^T
        """
        if "M3" not in self.bases or self.bases["M3"] is None:
            print("  GPM: no M3 bases yet, skipping G3 projection")
            return

        M = self.bases["M3"].to(model.router.current_gate.G3.weight.device)   # [d, k]
        g = model.router.current_gate.G3.weight.data   # [1, d]
        model.router.current_gate.G3.weight.data = g - g @ M @ M.T

        # Verify: G3 · any_column_of_M should be ~0
        g_n = model.router.current_gate.G3.weight.data
        max_cos = (g_n @ M).abs().max().item() / (g_n.norm().item() + 1e-8)
        print(f"  GPM: projected G3 ⊥ M3  (max |cosine| with bases: {max_cos:.2e}, target ~0)")

    def make_projection_hook(self, model):
        """
        Paper eq 12: before each optimizer.step(), project ΔG_{t,l} ⊥ M_{t,l}
        for l ∈ {1, 2, 3}.

        ΔG1.shape = [h, d],   bases M1 [d, k]:  ΔG1 ← ΔG1 - ΔG1 @ M1 @ M1^T
        ΔG2.shape = [d, h],   bases M2 [h, k]:  ΔG2 ← ΔG2 - ΔG2 @ M2 @ M2^T
        ΔG3.shape = [1, d],   bases M3 [d, k]:  ΔG3 ← ΔG3 - ΔG3 @ M3 @ M3^T
        """
        def _project():
            with torch.no_grad():
                gate = model.router.current_gate

                # G1: [h, d], bases [d, k]
                if "M1" in self.bases and self.bases["M1"] is not None:
                    if gate.G1.weight.grad is not None:
                        M = self.bases["M1"].to(gate.G1.weight.grad.device)
                        g = gate.G1.weight.grad   # [h, d]
                        gate.G1.weight.grad = g - g @ M @ M.T

                # G2: [d, h], bases [h, k]
                if "M2" in self.bases and self.bases["M2"] is not None:
                    if gate.G2.weight.grad is not None:
                        M = self.bases["M2"].to(gate.G2.weight.grad.device)
                        g = gate.G2.weight.grad   # [d, h]
                        gate.G2.weight.grad = g - g @ M @ M.T

                # G3: [1, d], bases [d, k]
                if "M3" in self.bases and self.bases["M3"] is not None:
                    if gate.G3.weight.grad is not None:
                        M = self.bases["M3"].to(gate.G3.weight.grad.device)
                        g = gate.G3.weight.grad   # [1, d]
                        gate.G3.weight.grad = g - g @ M @ M.T

        return _project

    def project_init_G3_dual(self, model):
        """Project BOTH learn and unlearn router G3s."""
        if "M3" not in self.bases or self.bases["M3"] is None:
            return
        for router_name in ["learn_router", "unlearn_router"]:
            router = getattr(model, router_name)
            M = self.bases["M3"].to(router.current_gate.G3.weight.device)
            g = router.current_gate.G3.weight.data
            router.current_gate.G3.weight.data = g - g @ M @ M.T
        print(f"  GPM: projected G3 ⊥ M3 for both learn and unlearn routers")

    def make_projection_hook_dual(self, model):
        """Project gradients for both learn and unlearn routers."""
        def _project():
            with torch.no_grad():
                for router_name in ["learn_router", "unlearn_router"]:
                    router = getattr(model, router_name)
                    gate = router.current_gate
                    if "M1" in self.bases and self.bases["M1"] is not None:
                        if gate.G1.weight.grad is not None:
                            M = self.bases["M1"].to(gate.G1.weight.grad.device)
                            g = gate.G1.weight.grad
                            gate.G1.weight.grad = g - g @ M @ M.T
                    if "M2" in self.bases and self.bases["M2"] is not None:
                        if gate.G2.weight.grad is not None:
                            M = self.bases["M2"].to(gate.G2.weight.grad.device)
                            g = gate.G2.weight.grad
                            gate.G2.weight.grad = g - g @ M @ M.T
                    if "M3" in self.bases and self.bases["M3"] is not None:
                        if gate.G3.weight.grad is not None:
                            M = self.bases["M3"].to(gate.G3.weight.grad.device)
                            g = gate.G3.weight.grad
                            gate.G3.weight.grad = g - g @ M @ M.T
        return _project
