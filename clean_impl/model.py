"""
T5-large + LoRA at q and v projections.

Strategy: load stock T5ForConditionalGeneration, freeze all parameters,
then walk every T5Attention module and attach LoRA layers that add to the
output of the existing q and v linear projections. We use a forward hook
so we don't have to subclass T5Attention (which is what made the reference
code break across transformers versions).
"""
import torch
from torch import nn
from transformers import T5ForConditionalGeneration

from lora import LoRALayer


def _wrap_linear_with_lora(linear: nn.Linear, r: int, alpha: int, dropout: float) -> nn.Module:
    """
    Return a new module that computes:  linear(x) + lora(x)
    The original linear is frozen; only the LoRA params are trainable.
    """
    lora = LoRALayer(linear.in_features, linear.out_features, r=r, lora_alpha=alpha, lora_dropout=dropout)

    class LinearWithLoRA(nn.Module):
        def __init__(self, base_linear: nn.Linear, lora_layer: LoRALayer):
            super().__init__()
            self.base = base_linear
            self.lora = lora_layer
            for p in self.base.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.base(x) + self.lora(x)

    return LinearWithLoRA(linear, lora)


def attach_lora_to_t5(
    model: T5ForConditionalGeneration,
    r: int = 4,
    alpha: int = 32,
    dropout: float = 0.0,
) -> T5ForConditionalGeneration:
    """
    Walk the T5 model, find every T5Attention block, and replace its
    .q and .v sub-modules with a LinearWithLoRA wrapper.

    Returns the same model object, modified in place.
    """
    # First, freeze everything in the model.
    for p in model.parameters():
        p.requires_grad = False

    n_attached = 0
    for name, module in model.named_modules():
        # T5 attention blocks have .q, .k, .v, .o linear projections.
        # We only attach LoRA to q and v (matching the reference repo).
        if hasattr(module, 'q') and hasattr(module, 'k') and hasattr(module, 'v') and hasattr(module, 'o'):
            module.q = _wrap_linear_with_lora(module.q, r, alpha, dropout)
            module.v = _wrap_linear_with_lora(module.v, r, alpha, dropout)
            n_attached += 1

    return model, n_attached


def build_model(
    model_name: str = "t5-large",
    lora_r: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
):
    """Load T5-large and attach LoRA adapters."""
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model, n = attach_lora_to_t5(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  attached LoRA to {n} attention blocks")
    print(f"  trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.3f}%)")

    return model
