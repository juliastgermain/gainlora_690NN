"""T5-large + multi-task LoRA + gating module."""
import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from lora import LoRALayer
from gating import GatingModule, pool_encoder_hidden

class MultiTaskLinearWithLoRA(nn.Module):
    def __init__(self, base_linear, in_features, out_features, r, alpha, dropout):
        super().__init__()
        self.base = base_linear
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        for p in self.base.parameters():
            p.requires_grad = False
        self.loras = nn.ModuleList([
            LoRALayer(in_features, out_features, r=r, lora_alpha=alpha, lora_dropout=dropout)
        ])
        self._gate_weights = None
        self._active_task_only = None
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
        base_out = self.base(x)
        if self._active_task_only is not None:
            return base_out + self.loras[self._active_task_only](x)
        if self._gate_weights is None:
            return base_out + sum(lora(x) for lora in self.loras)
        n_tasks = self._gate_weights.size(1)
        if n_tasks != len(self.loras):
            raise RuntimeError(f"gate has {n_tasks} tasks but layer has {len(self.loras)} LoRAs")
        # Softmax across tasks (dim=1) so weights sum to 1 per dimension.
        # Without this, the new task's gate can grow unboundedly large and
        # dominate even on old-task inputs. The reference repo uses attention-
        # style normalization (attn_temperature parameter); softmax is the
        # clean equivalent.
        normed_gates = torch.softmax(self._gate_weights, dim=1)
        out = base_out
        for t, lora in enumerate(self.loras):
            weight = normed_gates[:, t, :].unsqueeze(1)
            out = out + lora(x) * weight
        return out

def _is_t5_attention(module):
    return all(hasattr(module, attr) for attr in ("q", "k", "v", "o"))

class GainLoRAT5(T5ForConditionalGeneration):
    def _compute_and_set_gate(self, input_ids, attention_mask):
        """Compute gate weights from input embeddings (not encoder output).
        This matches the reference repo which uses avg_inputs_embeds, and
        solves the chicken-and-egg problem: gate runs BEFORE encoder, so
        both encoder and decoder attention layers are properly gated."""
        input_embeds = self.shared(input_ids)  # [batch, seq, d_model]
        # Pool embeddings the same way we pooled encoder hidden states
        pooled = pool_encoder_hidden(input_embeds, attention_mask)
        gate_weights = self.gating(pooled)
        set_gate_weights(self, gate_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, decoder_attention_mask=None,
                encoder_outputs=None, **kwargs):
        if (self._gating_enabled and self._active_task_only is None
                and input_ids is not None):
            # Compute gate from embeddings BEFORE encoder runs.
            # This way both encoder and decoder are properly gated.
            self._compute_and_set_gate(input_ids, attention_mask)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        """Compute gate weights before generation."""
        if (self._gating_enabled and self._active_task_only is None
                and input_ids is not None):
            self._compute_and_set_gate(input_ids, attention_mask)
        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

def attach_loras_and_gate(model, lora_r=4, lora_alpha=32, lora_dropout=0.0, gate_hidden_dim=100):
    for p in model.parameters():
        p.requires_grad = False
    d_model = model.config.d_model
    wrapped = []
    n_blocks = 0
    for name, module in model.named_modules():
        if _is_t5_attention(module):
            module.q = MultiTaskLinearWithLoRA(module.q, d_model, d_model, lora_r, lora_alpha, lora_dropout)
            module.v = MultiTaskLinearWithLoRA(module.v, d_model, d_model, lora_r, lora_alpha, lora_dropout)
            wrapped.append((name, module.q))
            wrapped.append((name, module.v))
            n_blocks += 1
    model.gating = GatingModule(d_model=d_model, hidden_dim=gate_hidden_dim, n_tasks=1)
    model._wrapped_loras = wrapped
    model._gating_enabled = False
    model._active_task_only = None
    return model, n_blocks

def set_gate_weights(model, gate_weights):
    for _, wrapped in model._wrapped_loras:
        wrapped._gate_weights = gate_weights

def set_active_task(model, task_idx):
    for _, wrapped in model._wrapped_loras:
        wrapped._active_task_only = task_idx
    model._active_task_only = task_idx

def enable_gating(model, enabled=True):
    model._gating_enabled = enabled
    if not enabled:
        set_gate_weights(model, None)

def add_task(model):
    for _, wrapped in model._wrapped_loras:
        wrapped.add_lora()
    new_idx = model.gating.add_task()
    return new_idx

def trainable_lora_params(model):
    params = []
    for _, wrapped in model._wrapped_loras:
        for lora in wrapped.loras:
            for p in lora.parameters():
                if p.requires_grad:
                    params.append(p)
    return params

def gating_params(model):
    return list(model.gating.parameters())

def build_model(model_name="t5-large", lora_r=4, lora_alpha=32, lora_dropout=0.0, gate_hidden_dim=100):
    model = GainLoRAT5.from_pretrained(model_name)
    model, n_blocks = attach_loras_and_gate(model, lora_r=lora_r, lora_alpha=lora_alpha,
                                             lora_dropout=lora_dropout, gate_hidden_dim=gate_hidden_dim)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  attached LoRA to {n_blocks} attention blocks (q + v each)")
    print(f"  gate hidden_dim={gate_hidden_dim}, n_tasks=1")
    print(f"  trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")
    return model
