"""
GainLoRA extended for safety learn + unlearn.

Each task t gets TWO LoRA adapters:
  - learn_lora_t:   trained with gradient DESCENT on safe responses
  - unlearn_lora_t: trained with gradient ASCENT on unsafe responses

Two independent gating modules per task:
  - learn_gate_t:   routes learn LoRA contributions
  - unlearn_gate_t: routes unlearn LoRA contributions

Forward (paper eq 1-2, extended):
  e = W·h
    + Σ_i [ a_i^learn  · LearnLoRA_i(h)  ]
    + Σ_i [ a_i^unlearn · UnlearnLoRA_i(h) ]

The unlearn gate is trained to have a_t^unlearn → 0 for old-task inputs
(same GPM constraints), but the unlearn LoRA itself is trained to INCREASE
loss on unsafe content — its contribution is directionally different from
a learn LoRA.

This is the benchmark failure mode we demonstrate:
  GainLoRA's gate suppresses BOTH adapters for non-matching tasks equally,
  treating learn and unlearn as symmetric. This means:
    - Old task's unlearn adapter gets gated to ~0 by new task training
    - The model partially re-learns to generate harmful old-task content
    - The unlearning effect degrades even though the LoRA weights are frozen

The forgetting metric for unlearning is distinct from learning forgetting:
  FT_learn   = drop in ROUGE-L on safe responses (standard)
  FT_unlearn = drop in "refusal rate" on unsafe prompts (novel metric)
"""
import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from lora import LoRALayer
from gating import MultiTaskRouter, pool_encoder_hidden


class DualAdapterLinear(nn.Module):
    """
    Linear layer with paired (learn, unlearn) LoRA adapters per task.

    Mode flags set by the training loop:
      _active_task:    int or None (None = gated mode)
      _active_adapter: 'learn', 'unlearn', or 'both'
    """
    def __init__(self, base_linear, in_features, out_features, r, alpha, dropout):
        super().__init__()
        self.base = base_linear
        self.in_features = in_features
        self.out_features = out_features
        self.r = r; self.alpha = alpha; self.dropout = dropout

        for p in self.base.parameters():
            p.requires_grad = False

        # Lists grow with add_task()
        self.learn_loras   = nn.ModuleList()
        self.unlearn_loras = nn.ModuleList()

        # Add task-0 adapters
        self._append_lora_pair(trainable=True)

        # Gate weights set externally before forward
        self._learn_weights   = None   # [B, n_tasks, 1]
        self._unlearn_weights = None   # [B, n_tasks, 1]

        # Eval/isolation mode
        self._active_task    = None    # int: use only this task's adapters
        self._active_adapter = 'both'  # 'learn', 'unlearn', or 'both'

    def _append_lora_pair(self, trainable=True):
        learn_lora = LoRALayer(self.in_features, self.out_features,
                               r=self.r, lora_alpha=self.alpha,
                               lora_dropout=self.dropout)
        unlearn_lora = LoRALayer(self.in_features, self.out_features,
                                 r=self.r, lora_alpha=self.alpha,
                                 lora_dropout=self.dropout)
        if not trainable:
            for p in learn_lora.parameters():
                p.requires_grad = False
            for p in unlearn_lora.parameters():
                p.requires_grad = False
        self.learn_loras.append(learn_lora)
        self.unlearn_loras.append(unlearn_lora)

    def add_task(self):
        """Freeze all existing adapters, add new trainable pair."""
        device = next(self.learn_loras[0].parameters()).device
        for lora in self.learn_loras:
            for p in lora.parameters():
                p.requires_grad = False
        for lora in self.unlearn_loras:
            for p in lora.parameters():
                p.requires_grad = False
        self._append_lora_pair(trainable=True)
        # Move new adapters to correct device
        self.learn_loras[-1] = self.learn_loras[-1].to(device)
        self.unlearn_loras[-1] = self.unlearn_loras[-1].to(device)

    def forward(self, x):
        base_out = self.base(x)

        # ── Isolation mode: single task, specified adapter ─────────────────
        if self._active_task is not None:
            t = self._active_task
            out = base_out
            if self._active_adapter in ('learn', 'both'):
                out = out + self.learn_loras[t](x)
            if self._active_adapter in ('unlearn', 'both'):
                out = out + self.unlearn_loras[t](x)
            return out

        # ── Gating off: uniform sum ────────────────────────────────────────
        if self._learn_weights is None:
            out = base_out
            for lora in self.learn_loras:
                out = out + lora(x)
            for lora in self.unlearn_loras:
                out = out + lora(x)
            return out

        # ── Gating on: weighted sum ────────────────────────────────────────
        out = base_out
        for t, lora in enumerate(self.learn_loras):
            w = self._learn_weights[:, t, :].unsqueeze(1)    # [B, 1, 1]
            out = out + lora(x) * w
        for t, lora in enumerate(self.unlearn_loras):
            w = self._unlearn_weights[:, t, :].unsqueeze(1)  # [B, 1, 1]
            out = out + lora(x) * w
        return out


def _is_t5_attention(module):
    return all(hasattr(module, attr) for attr in ("q", "k", "v", "o"))


class SafetyGainLoRAT5(T5ForConditionalGeneration):
    """
    T5 with dual (learn, unlearn) LoRA pairs per task.
    Two routers: one for learn adapters, one for unlearn adapters.
    """

    def _compute_routing(self, input_ids, attention_mask):
        embeds = self.shared(input_ids)
        p0 = pool_encoder_hidden(embeds, attention_mask)
        learn_w   = self.learn_router(p0)    # [B, n_tasks, 1]
        unlearn_w = self.unlearn_router(p0)  # [B, n_tasks, 1]
        for _, layer in self._dual_layers:
            layer._learn_weights   = learn_w
            layer._unlearn_weights = unlearn_w

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, decoder_attention_mask=None,
                encoder_outputs=None, **kwargs):
        if (self._routing_enabled
                and self._active_task is None
                and input_ids is not None):
            self._compute_routing(input_ids, attention_mask)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if (self._routing_enabled
                and self._active_task is None
                and input_ids is not None):
            self._compute_routing(input_ids, attention_mask)
        return super().generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs)


# ── Helper functions ──────────────────────────────────────────────────────────

def build_safety_model(model_name="t5-large", lora_r=4, lora_alpha=32,
                        lora_dropout=0.0, router_hidden_dim=100):
    model = SafetyGainLoRAT5.from_pretrained(model_name)

    for p in model.parameters():
        p.requires_grad = False

    d_model = model.config.d_model
    dual_layers = []
    n_blocks = 0

    for name, module in model.named_modules():
        if _is_t5_attention(module):
            module.q = DualAdapterLinear(
                module.q, d_model, d_model, lora_r, lora_alpha, lora_dropout)
            module.v = DualAdapterLinear(
                module.v, d_model, d_model, lora_r, lora_alpha, lora_dropout)
            dual_layers.append((name, module.q))
            dual_layers.append((name, module.v))
            n_blocks += 1

    model.learn_router   = MultiTaskRouter(d_model=d_model, hidden_dim=router_hidden_dim)
    model.unlearn_router = MultiTaskRouter(d_model=d_model, hidden_dim=router_hidden_dim)
    model._dual_layers   = dual_layers
    model._routing_enabled = False
    model._active_task   = None

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  {n_blocks} attention blocks with dual (learn, unlearn) LoRA pairs")
    print(f"  trainable: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")
    return model


def safety_add_task(model):
    """Add new (learn, unlearn) LoRA pair + new gate slot for both routers."""
    for _, layer in model._dual_layers:
        layer.add_task()
    new_learn_idx   = model.learn_router.add_task()
    new_unlearn_idx = model.unlearn_router.add_task()
    assert new_learn_idx == new_unlearn_idx
    return new_learn_idx


def safety_enable_gating(model, enabled=True):
    model._routing_enabled = enabled
    if not enabled:
        for _, layer in model._dual_layers:
            layer._learn_weights   = None
            layer._unlearn_weights = None


def set_active_task_and_adapter(model, task_idx, adapter='both'):
    """For isolated evaluation of a specific task and adapter type."""
    model._active_task = task_idx
    for _, layer in model._dual_layers:
        layer._active_task    = task_idx
        layer._active_adapter = adapter


def clear_active_task(model):
    model._active_task = None
    for _, layer in model._dual_layers:
        layer._active_task    = None
        layer._active_adapter = 'both'


def trainable_learn_params(model):
    params = []
    for _, layer in model._dual_layers:
        for p in layer.learn_loras[-1].parameters():  # only current task
            if p.requires_grad:
                params.append(p)
    return params


def trainable_unlearn_params(model):
    params = []
    for _, layer in model._dual_layers:
        for p in layer.unlearn_loras[-1].parameters():  # only current task
            if p.requires_grad:
                params.append(p)
    return params


def learn_gate_params(model):
    return model.learn_router.get_trainable_params()


def unlearn_gate_params(model):
    return model.unlearn_router.get_trainable_params()
