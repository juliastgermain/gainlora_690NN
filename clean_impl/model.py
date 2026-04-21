"""
T5-large + multi-task LoRA + MultiTaskRouter.

Integration (paper eq 1-2):
  W_t = Σ_{i=1}^{t} a_i A_i B_i
  e = (W + W_t) h

where a_i = g_i(x) is a SCALAR in [0,1] produced by the i-th gating module.
The routing weight shape is [B, n_tasks, 1] — scalar per task per sample.
"""
import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from lora import LoRALayer
from gating import MultiTaskRouter, pool_encoder_hidden


class MultiTaskLinearWithLoRA(nn.Module):
    def __init__(self, base_linear, in_features, out_features, r, alpha, dropout):
        super().__init__()
        self.base = base_linear
        self.in_features = in_features
        self.out_features = out_features
        self.r = r; self.alpha = alpha; self.dropout = dropout
        for p in self.base.parameters():
            p.requires_grad = False
        self.loras = nn.ModuleList([
            LoRALayer(in_features, out_features, r=r, lora_alpha=alpha, lora_dropout=dropout)
        ])
        self._routing_weights = None   # [B, n_tasks, 1], scalar per task
        self._active_task_only = None  # int or None

    def add_lora(self):
        new_lora = LoRALayer(self.in_features, self.out_features,
                             r=self.r, lora_alpha=self.alpha, lora_dropout=self.dropout)
        device = next(self.loras[0].parameters()).device
        new_lora = new_lora.to(device)
        for old in self.loras:
            for p in old.parameters():
                p.requires_grad = False
        self.loras.append(new_lora)

    def forward(self, x):
        base_out = self.base(x)   # [B, S, out]

        if self._active_task_only is not None:
            return base_out + self.loras[self._active_task_only](x)

        if self._routing_weights is None:
            # Gating off: sum all LoRAs uniformly (used during evaluation with set_active_task=None)
            return base_out + sum(lora(x) for lora in self.loras)

        # Gating on: paper eq 1-2, scalar weight per task
        # _routing_weights: [B, n_tasks, 1]
        out = base_out
        for t, lora in enumerate(self.loras):
            # weight: [B, 1] → unsqueeze to [B, 1, 1] for broadcast against [B, S, out]
            w = self._routing_weights[:, t, :].unsqueeze(1)   # [B, 1, 1]
            out = out + lora(x) * w
        return out


def _is_t5_attention(module):
    return all(hasattr(module, attr) for attr in ("q", "k", "v", "o"))


class GainLoRAT5(T5ForConditionalGeneration):

    def _compute_and_set_routing(self, input_ids, attention_mask):
        """Pool input token embeddings, run all gates, set weights on every LoRA layer."""
        input_embeds = self.shared(input_ids)                            # [B, S, d]
        p0 = pool_encoder_hidden(input_embeds, attention_mask)           # [B, d]
        weights = self.router(p0)                                        # [B, n_tasks, 1]
        set_routing_weights(self, weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, decoder_attention_mask=None,
                encoder_outputs=None, **kwargs):
        if (self._routing_enabled
                and self._active_task_only is None
                and input_ids is not None):
            self._compute_and_set_routing(input_ids, attention_mask)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if (self._routing_enabled
                and self._active_task_only is None
                and input_ids is not None):
            self._compute_and_set_routing(input_ids, attention_mask)
        return super().generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def attach_loras_and_router(model, lora_r=4, lora_alpha=32, lora_dropout=0.0,
                             router_hidden_dim=100):
    for p in model.parameters():
        p.requires_grad = False
    d_model = model.config.d_model
    wrapped = []
    n_blocks = 0
    for name, module in model.named_modules():
        if _is_t5_attention(module):
            module.q = MultiTaskLinearWithLoRA(
                module.q, d_model, d_model, lora_r, lora_alpha, lora_dropout)
            module.v = MultiTaskLinearWithLoRA(
                module.v, d_model, d_model, lora_r, lora_alpha, lora_dropout)
            wrapped.append((name, module.q))
            wrapped.append((name, module.v))
            n_blocks += 1
    model.router = MultiTaskRouter(d_model=d_model, hidden_dim=router_hidden_dim)
    model._wrapped_loras = wrapped
    model._routing_enabled = False
    model._active_task_only = None
    return model, n_blocks


def set_routing_weights(model, weights):
    for _, w in model._wrapped_loras:
        w._routing_weights = weights

def set_active_task(model, task_idx):
    for _, w in model._wrapped_loras:
        w._active_task_only = task_idx
    model._active_task_only = task_idx

def enable_gating(model, enabled=True):
    model._routing_enabled = enabled
    if not enabled:
        set_routing_weights(model, None)

def add_task(model):
    for _, w in model._wrapped_loras:
        w.add_lora()
    return model.router.add_task()

def trainable_lora_params(model):
    params = []
    for _, w in model._wrapped_loras:
        for lora in w.loras:
            for p in lora.parameters():
                if p.requires_grad:
                    params.append(p)
    return params

def gating_params(model):
    return model.router.get_trainable_params()

def build_model(model_name="t5-large", lora_r=4, lora_alpha=32,
                lora_dropout=0.0, router_hidden_dim=100):
    model = GainLoRAT5.from_pretrained(model_name)
    model, n_blocks = attach_loras_and_router(
        model, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, router_hidden_dim=router_hidden_dim)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  attached LoRA to {n_blocks} attention blocks (q + v each)")
    print(f"  router hidden_dim={router_hidden_dim}, n_tasks=1")
    print(f"  trainable: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")
    # Paper Appendix B.4: GainLoRA(O-LoRA) T5-Large = 1,385,472 params
    print(f"  (paper target: 1,385,472)")
    return model
