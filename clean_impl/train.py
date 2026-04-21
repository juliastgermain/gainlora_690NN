"""
Sequential training — paper hyperparameters (Appendix B.2):
  - Batch size: 32 (via grad accumulation)
  - Learning rate: 3e-4 (AdamW, T5 backbone)
  - Epochs per task: 100
  - LoRA rank: 4 (Orders 1/2)

GainLoRA flow (paper Algorithm 1):
  Task 0: expand LoRA branch, init gate g0 (no constraints needed for first task),
          train LoRA + gate freely.
  → collect GPM activations from task-0 data (Appendix A.1)
  → update_bases()
  → add_task(): freeze task-0 LoRA+gate, create task-1 gate with
                G1,G2 copied from g0 (eq 9), G3 projected ⊥ M3 (eq 10)
  Task 1: train LoRA-1 + gate-1 with GPM gradient projection (eq 12).
  → eval both tasks.

Vanilla baseline: same LoRA trained on both tasks sequentially, no gate.
"""
import gc, time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from data import SuperNITaskDataset
from gpm import GPM
from model import (build_model, add_task, set_active_task, enable_gating,
                   trainable_lora_params, gating_params)

DATA_DIR = "/content/gainlora_690NN/CL_Benchmark"
MODEL_NAME = "t5-large"


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _compute_rouge_l(predictions, references):
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    if not predictions:
        return 0.0
    return sum(sc.score(r, p)['rougeL'].fmeasure
               for p, r in zip(predictions, references)) / len(predictions)


@torch.no_grad()
def evaluate_task(model, task_name, tokenizer, max_eval_samples=30,
                  max_source_length=512, max_target_length=50,
                  eval_batch_size=4, active_task_idx=None):
    eval_ds = SuperNITaskDataset(
        data_dir=DATA_DIR, task_name=task_name, split="dev",
        tokenizer=tokenizer, max_source_length=max_source_length,
        max_target_length=max_target_length, max_num_instances=max_eval_samples)
    loader = DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False)

    prev_active = model._active_task_only
    if active_task_idx is not None:
        set_active_task(model, active_task_idx)
    model.eval()

    preds, refs = [], []
    for batch in loader:
        gen_ids = model.generate(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            max_new_tokens=max_target_length, num_beams=1, do_sample=False)
        preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

    set_active_task(model, prev_active)
    return {"rougeL": _compute_rouge_l(preds, refs),
            "n_eval": len(refs),
            "examples": list(zip(preds[:3], refs[:3]))}


def _train_one_task(model, tokenizer, task_name, epochs, train_batch_size,
                    grad_accum_steps, learning_rate, max_source_length,
                    max_target_length, max_train_samples, log_every,
                    optimizer_param_groups, grad_mask_fn=None):
    train_ds = SuperNITaskDataset(
        data_dir=DATA_DIR, task_name=task_name, split="train",
        tokenizer=tokenizer, max_source_length=max_source_length,
        max_target_length=max_target_length, max_num_instances=max_train_samples)
    loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    print(f"  [{task_name}] {len(train_ds)} examples, "
          f"{len(loader)} steps/epoch, "
          f"eff. batch={train_batch_size * grad_accum_steps}")

    optimizer = AdamW(optimizer_param_groups(), lr=learning_rate)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_micro = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            with autocast:
                out = model(input_ids=batch["input_ids"].cuda(),
                            attention_mask=batch["attention_mask"].cuda(),
                            labels=batch["labels"].cuda())
                loss = out.loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            n_micro += 1

            if (step + 1) % grad_accum_steps == 0:
                if grad_mask_fn is not None:
                    grad_mask_fn()
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % log_every == 0:
                print(f"    epoch {epoch+1} step {step+1:4d}  "
                      f"loss={epoch_loss/n_micro:.4f}")

        if (step + 1) % grad_accum_steps != 0:
            if grad_mask_fn is not None:
                grad_mask_fn()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d} done  avg_loss={epoch_loss/max(n_micro,1):.4f}  "
                  f"elapsed={time.time()-start:.0f}s")

    return epoch_loss / max(n_micro, 1)


def run_phase3_gainlora(
    task_0, task_1,
    # Paper hyperparameters (Appendix B.2)
    epochs=100,            # paper: 100 epochs per task for T5
    train_batch_size=2,    # 2 samples × 16 accum = batch 32 (paper)
    grad_accum_steps=16,   # → effective batch = 32
    learning_rate=3e-4,    # paper: 3e-4 for T5
    lora_r=4,              # paper: rank 4 for Orders 1/2
    lora_alpha=32,
    lora_dropout=0.0,
    gate_hidden_dim=100,   # paper Appendix B.3
    gpm_threshold=0.97,    # GPM paper [47] default
    max_source_length=512,
    max_target_length=50,
    max_train_samples=None,   # None = use full dataset (paper uses full)
    max_eval_samples=None,
    log_every=50,
):
    print(f"\n{'='*60}\nGAINLORA: {task_0} -> {task_1}\n{'='*60}")
    print(f"  epochs={epochs}, lr={learning_rate}, batch={train_batch_size*grad_accum_steps}, "
          f"r={lora_r}, gpm_threshold={gpm_threshold}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        router_hidden_dim=gate_hidden_dim).cuda()

    # ── Task 0 ────────────────────────────────────────────────────────────
    # Paper Algorithm 1: for task 1 (t=1) there are no old tasks, so no
    # init or updating constraints needed. Train gate + LoRA freely.
    print(f"\n--- Train task 0: {task_0} ---")
    enable_gating(model, True)

    def task0_params():
        # All trainable params at learning rate 3e-4 (paper uses single lr)
        return trainable_lora_params(model) + gating_params(model)

    _train_one_task(model, tokenizer, task_0,
                    epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=task0_params)
    _free()

    print(f"\n--- Eval task 0 after task-0 training ---")
    eval_t0_after_t0 = evaluate_task(
        model, task_0, tokenizer, max_eval_samples=max_eval_samples,
        max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t0['rougeL']:.4f}  (n={eval_t0_after_t0['n_eval']})")
    _free()

    # ── GPM: collect activations BEFORE add_task ──────────────────────────
    # Must happen now while current_gate is still task-0's gate (Appendix A.1).
    print(f"\n--- Collecting GPM activations (task 0, ε_th={gpm_threshold}) ---")
    gpm = GPM(threshold=gpm_threshold)
    gpm_ds = SuperNITaskDataset(
        data_dir=DATA_DIR, task_name=task_0, split="train",
        tokenizer=tokenizer, max_source_length=max_source_length,
        max_target_length=max_target_length, max_num_instances=max_train_samples)
    gpm_loader = DataLoader(gpm_ds, batch_size=8, shuffle=False)
    activations = gpm.collect_activations(model, gpm_loader)
    gpm.update_bases(activations)
    del gpm_ds, gpm_loader, activations
    _free()

    # ── add_task + init constraints (paper eq 9, 10) ──────────────────────
    print(f"\n--- add_task() ---")
    new_idx = add_task(model)
    print(f"  added task {new_idx}, router now has {model.router.n_tasks} tasks")
    # G1, G2 were already copied in add_task() (paper eq 9)
    # Project G3 ⊥ M_{t,3} (paper eq 10)
    print(f"\n--- Init constraint: project G3 ⊥ M3 (paper eq 10) ---")
    gpm.project_init_G3(model)

    # ── Task 1 ────────────────────────────────────────────────────────────
    print(f"\n--- Train task 1: {task_1} ---")
    gpm_project = gpm.make_projection_hook(model)   # paper eq 12

    def task1_params():
        return trainable_lora_params(model) + gating_params(model)

    _train_one_task(model, tokenizer, task_1,
                    epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=task1_params, grad_mask_fn=gpm_project)
    _free()

    # ── Eval ──────────────────────────────────────────────────────────────
    print(f"\n--- Eval task 0 after task-1 training (forgetting test) ---")
    eval_t0_after_t1 = evaluate_task(
        model, task_0, tokenizer, max_eval_samples=max_eval_samples,
        max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t1['rougeL']:.4f}")
    _free()

    print(f"\n--- Eval task 1 ---")
    eval_t1 = evaluate_task(
        model, task_1, tokenizer, max_eval_samples=max_eval_samples,
        max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t1['rougeL']:.4f}")
    _free()

    print(f"\n--- Eval task 0 isolated (task-0 LoRA only, sanity check) ---")
    eval_t0_iso = evaluate_task(
        model, task_0, tokenizer, max_eval_samples=max_eval_samples,
        max_source_length=max_source_length, max_target_length=max_target_length,
        active_task_idx=0)
    print(f"  ROUGE-L = {eval_t0_iso['rougeL']:.4f}  "
          f"(should ≈ {eval_t0_after_t0['rougeL']:.4f})")
    _free()

    return {
        "flow": "gainlora",
        "task_0_after_task_0": eval_t0_after_t0["rougeL"],
        "task_0_after_task_1": eval_t0_after_t1["rougeL"],
        "task_0_isolated":     eval_t0_iso["rougeL"],
        "task_1":              eval_t1["rougeL"],
        "model": model, "tokenizer": tokenizer,
    }


def run_phase3_vanilla(
    task_0, task_1,
    epochs=100,
    train_batch_size=2, grad_accum_steps=16, learning_rate=3e-4,
    lora_r=4, lora_alpha=32, lora_dropout=0.0,
    max_source_length=512, max_target_length=50,
    max_train_samples=None, max_eval_samples=None, log_every=50,
):
    print(f"\n{'='*60}\nVANILLA LORA: {task_0} -> {task_1}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout).cuda()
    enable_gating(model, False)

    print(f"\n--- Train task 0: {task_0} ---")
    _train_one_task(model, tokenizer, task_0,
                    epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=lambda: trainable_lora_params(model))
    _free()

    print(f"\n--- Eval task 0 after task-0 training ---")
    eval_t0_after_t0 = evaluate_task(
        model, task_0, tokenizer, max_eval_samples=max_eval_samples,
        max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t0['rougeL']:.4f}")
    _free()

    print(f"\n--- Continue training SAME LoRA on task 1: {task_1} ---")
    _train_one_task(model, tokenizer, task_1,
                    epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=lambda: trainable_lora_params(model))
    _free()

    print(f"\n--- Eval task 0 after task-1 training ---")
    eval_t0_after_t1 = evaluate_task(
        model, task_0, tokenizer, max_eval_samples=max_eval_samples,
        max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t1['rougeL']:.4f}")
    _free()

    print(f"\n--- Eval task 1 ---")
    eval_t1 = evaluate_task(
        model, task_1, tokenizer, max_eval_samples=max_eval_samples,
        max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t1['rougeL']:.4f}")
    _free()

    return {
        "flow": "vanilla",
        "task_0_after_task_0": eval_t0_after_t0["rougeL"],
        "task_0_after_task_1": eval_t0_after_t1["rougeL"],
        "task_1":              eval_t1["rougeL"],
        "model": model, "tokenizer": tokenizer,
    }
