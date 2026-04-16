"""
Single-task training loop for T5-large + LoRA on a SuperNI task.

Plain PyTorch — no HuggingFace Trainer subclass. This is intentional:
the reference repo's Trainer subclass is the part that broke across
transformers versions. By writing the loop ourselves we sidestep that
entire class of problem.

Usage (from a notebook):
    from train import train_single_task
    metrics = train_single_task(
        task_name="task1572_samsum_summary",
        epochs=5,
        train_batch_size=4,
        grad_accum_steps=8,    # effective batch = 32
        learning_rate=3e-4,
        max_train_samples=200, # smoke test; None for full
        max_eval_samples=50,
    )
"""
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from data  import SuperNITaskDataset
from model import build_model


DATA_DIR = "/content/gainlora_690NN/CL_Benchmark"
MODEL_NAME = "t5-large"


def _compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Average ROUGE-L F1 across the eval set. Matches their `eval_rougeL`."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def evaluate(model, dataset, tokenizer, batch_size: int = 4, max_new_tokens: int = 50):
    """Generate predictions for the eval set and compute ROUGE-L."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds, refs = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

        # Recover reference text from labels (replace -100 with pad to decode)
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

    rouge_l = _compute_rouge_l(preds, refs)
    return {"rougeL": rouge_l, "n_eval": len(refs), "examples": list(zip(preds[:3], refs[:3]))}


def train_single_task(
    task_name: str,
    epochs: int = 5,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    grad_accum_steps: int = 8,
    learning_rate: float = 3e-4,
    lora_r: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    max_source_length: int = 512,
    max_target_length: int = 50,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    log_every: int = 20,
):
    """Train T5-large + LoRA on one SuperNI task, evaluate with ROUGE-L."""
    print(f"\n{'='*60}\nTraining task: {task_name}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = SuperNITaskDataset(
        data_dir=DATA_DIR, task_name=task_name, split="train",
        tokenizer=tokenizer,
        max_source_length=max_source_length, max_target_length=max_target_length,
        max_num_instances=max_train_samples,
    )
    eval_ds = SuperNITaskDataset(
        data_dir=DATA_DIR, task_name=task_name, split="dev",
        tokenizer=tokenizer,
        max_source_length=max_source_length, max_target_length=max_target_length,
        max_num_instances=max_eval_samples,
    )
    print(f"  train: {len(train_ds)} examples | eval: {len(eval_ds)} examples")

    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout).cuda()

    # Only LoRA params; everything else is frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    steps_per_epoch = len(train_loader)
    print(f"  steps/epoch: {steps_per_epoch} | grad accum: {grad_accum_steps} | "
          f"effective batch: {train_batch_size * grad_accum_steps}")

    # bf16 mixed precision
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    history = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            with autocast:
                out = model(
                    input_ids      = batch["input_ids"].cuda(),
                    attention_mask = batch["attention_mask"].cuda(),
                    labels         = batch["labels"].cuda(),
                )
                loss = out.loss / grad_accum_steps

            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            n_steps += 1

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % log_every == 0:
                avg = epoch_loss / n_steps
                print(f"  epoch {epoch+1} step {step+1:4d}/{steps_per_epoch}  loss={avg:.4f}")

        # Flush any remaining grads at end of epoch
        if (step + 1) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / max(n_steps, 1)
        print(f"  epoch {epoch+1} done  avg train loss={avg_train_loss:.4f}  "
              f"elapsed={time.time()-start:.0f}s")

    # Final eval
    print("\n  evaluating...")
    eval_metrics = evaluate(model, eval_ds, tokenizer, batch_size=eval_batch_size, max_new_tokens=max_target_length)
    print(f"  ROUGE-L: {eval_metrics['rougeL']:.4f}  (on {eval_metrics['n_eval']} examples)")
    print(f"\n  sample predictions:")
    for i, (pred, ref) in enumerate(eval_metrics["examples"]):
        print(f"    [{i+1}] PRED: {pred[:120]}")
        print(f"        REF:  {ref[:120]}")

    return {
        "task": task_name,
        "train_loss_final": avg_train_loss,
        "rougeL": eval_metrics["rougeL"],
        "model": model,
        "tokenizer": tokenizer,
    }
