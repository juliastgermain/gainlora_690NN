"""
Full sequential training following paper Algorithm 1.
Runs all T tasks in order, computes AP and FT per paper eq 14.
"""
import gc, time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from data import SuperNITaskDataset, SUPERNI_ORDER_1
from gpm import GPM
from model import (build_model, add_task, set_active_task, enable_gating,
                   trainable_lora_params, gating_params)

DATA_DIR = "/content/gainlora_690NN/CL_Benchmark"
MODEL_NAME = "t5-large"


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _rouge_l(predictions, references):
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    if not predictions:
        return 0.0
    return sum(sc.score(r, p)['rougeL'].fmeasure
               for p, r in zip(predictions, references)) / len(predictions)


@torch.no_grad()
def evaluate_task(model, task_name, tokenizer,
                  max_eval_samples=None, max_source_length=512,
                  max_target_length=50, eval_batch_size=8,
                  active_task_idx=None):
    eval_ds = SuperNITaskDataset(
        data_dir=DATA_DIR, task_name=task_name, split="dev",
        tokenizer=tokenizer, max_source_length=max_source_length,
        max_target_length=max_target_length,
        max_num_instances=max_eval_samples)
    loader = DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False)

    prev = model._active_task_only
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

    set_active_task(model, prev)
    return _rouge_l(preds, refs)


def _train_one_task(model, tokenizer, task_name, epochs,
                    train_batch_size, grad_accum_steps, learning_rate,
                    max_source_length, max_target_length,
                    max_train_samples, log_every,
                    optimizer_param_groups, grad_mask_fn=None):
    train_ds = SuperNITaskDataset(
        data_dir=DATA_DIR, task_name=task_name, split="train",
        tokenizer=tokenizer, max_source_length=max_source_length,
        max_target_length=max_target_length,
        max_num_instances=max_train_samples)
    loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    print(f"  [{task_name}] {len(train_ds)} examples, "
          f"{len(loader)} steps/epoch")

    optimizer = AdamW(optimizer_param_groups(), lr=learning_rate)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            with autocast:
                out = model(input_ids=batch["input_ids"].cuda(),
                            attention_mask=batch["attention_mask"].cuda(),
                            labels=batch["labels"].cuda())
                loss = out.loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            n += 1
            if (step + 1) % grad_accum_steps == 0:
                if grad_mask_fn is not None:
                    grad_mask_fn()
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % log_every == 0:
                print(f"    e{epoch+1} s{step+1}  loss={epoch_loss/n:.4f}")
        if (step + 1) % grad_accum_steps != 0:
            if grad_mask_fn is not None:
                grad_mask_fn()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  loss={epoch_loss/max(n,1):.4f}  "
                  f"t={time.time()-start:.0f}s")


def compute_ap_ft(A):
    """
    Paper eq 14.
    A[j][i] = performance on task i after learning task j (0-indexed).
    AP = mean of A[T-1][i] for all i.
    FT = mean of (max_{l<T} A[l][i] - A[T-1][i]) for i < T.
    """
    T = len(A)
    AP = sum(A[T-1][i] for i in range(T)) / T
    FT = 0.0
    for i in range(T - 1):
        best = max(A[l][i] for l in range(T - 1) if l >= i)
        FT += best - A[T-1][i]
    FT /= (T - 1)
    return AP, FT


def run_gainlora_full(
    task_order=None,
    epochs=100,
    train_batch_size=2,
    grad_accum_steps=16,
    learning_rate=3e-4,
    lora_r=4,
    lora_alpha=32,
    lora_dropout=0.0,
    gate_hidden_dim=100,
    gpm_threshold=0.97,
    max_source_length=512,
    max_target_length=50,
    max_train_samples=None,
    max_eval_samples=None,
    log_every=200,
):
    """
    Full GainLoRA sequential training on all tasks.
    Records A[j][i] = eval on task i after learning task j.
    Returns AP and FT per paper eq 14.
    """
    if task_order is None:
        task_order = SUPERNI_ORDER_1

    T = len(task_order)
    print(f"\n{'='*60}")
    print(f"GAINLORA: {T} tasks, {epochs} epochs each")
    print(f"lr={learning_rate}, batch={train_batch_size*grad_accum_steps}, r={lora_r}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        router_hidden_dim=gate_hidden_dim).cuda()

    # A[j][i] = performance on task i after learning task j
    # We store as a list of lists; A[j] is filled after task j completes.
    A = [[None] * T for _ in range(T)]

    gpm = GPM(threshold=gpm_threshold)
    enable_gating(model, True)

    for t, task_name in enumerate(task_order):
        print(f"\n{'─'*60}")
        print(f"TASK {t+1}/{T}: {task_name}")
        print(f"{'─'*60}")

        # ── Train task t ──────────────────────────────────────────────────
        def param_groups():
            return trainable_lora_params(model) + gating_params(model)

        gpm_hook = gpm.make_projection_hook(model) if t > 0 else None

        _train_one_task(
            model, tokenizer, task_name,
            epochs=epochs, train_batch_size=train_batch_size,
            grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
            max_source_length=max_source_length, max_target_length=max_target_length,
            max_train_samples=max_train_samples, log_every=log_every,
            optimizer_param_groups=param_groups,
            grad_mask_fn=gpm_hook,
        )
        _free()

        # ── Eval all tasks seen so far ────────────────────────────────────
        print(f"\n  Evaluating after task {t+1}...")
        for i in range(t + 1):
            score = evaluate_task(
                model, task_order[i], tokenizer,
                max_eval_samples=max_eval_samples,
                max_source_length=max_source_length,
                max_target_length=max_target_length)
            A[t][i] = score
            print(f"    task {i+1:2d} ({task_order[i][:30]:30s}): {score:.4f}")
        _free()

        # ── GPM: collect current task's activations ───────────────────────
        # Must happen BEFORE add_task() while current_gate is still task t's
        if t < T - 1:
            print(f"\n  Collecting GPM activations for task {t+1}...")
            gpm_ds = SuperNITaskDataset(
                data_dir=DATA_DIR, task_name=task_name, split="train",
                tokenizer=tokenizer, max_source_length=max_source_length,
                max_target_length=max_target_length,
                max_num_instances=max_train_samples)
            gpm_loader = DataLoader(gpm_ds, batch_size=8, shuffle=False)
            activations = gpm.collect_activations(model, gpm_loader)
            gpm.update_bases(activations)
            del gpm_ds, gpm_loader, activations
            _free()

            # add_task: freeze current LoRA+gate, create new ones
            print(f"\n  add_task()...")
            add_task(model)
            print(f"  router now has {model.router.n_tasks} tasks")

            # Init constraint (paper eq 10): project new G3 ⊥ M3
            gpm.project_init_G3(model)

    # ── Final metrics ─────────────────────────────────────────────────────
    # Fill in None entries (tasks not yet seen when j < i)
    for j in range(T):
        for i in range(T):
            if A[j][i] is None:
                A[j][i] = 0.0

    AP, FT = compute_ap_ft(A)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"  AP  = {AP:.4f}   (paper GainLoRA(O-LoRA) Order 1: 47.84)")
    print(f"  FT  = {FT:.4f}   (paper GainLoRA(O-LoRA) Order 1:  2.26)")
    print(f"{'='*60}")

    # Print full matrix
    print(f"\nPerformance matrix A[j][i] (row=after task j, col=task i):")
    header = "       " + "".join(f"T{i+1:02d}   " for i in range(T))
    print(header)
    for j in range(T):
        row = f"T{j+1:02d}:   " + "".join(
            f"{A[j][i]:.3f} " if A[j][i] is not None else "  --- "
            for i in range(T))
        print(row)

    return {"A": A, "AP": AP, "FT": FT,
            "model": model, "tokenizer": tokenizer, "gpm": gpm}


def run_vanilla_full(
    task_order=None,
    epochs=100,
    train_batch_size=2,
    grad_accum_steps=16,
    learning_rate=3e-4,
    lora_r=4, lora_alpha=32, lora_dropout=0.0,
    max_source_length=512, max_target_length=50,
    max_train_samples=None, max_eval_samples=None,
    log_every=200,
):
    """Vanilla sequential LoRA baseline (no gating, same LoRA updated each task)."""
    if task_order is None:
        task_order = SUPERNI_ORDER_1

    T = len(task_order)
    print(f"\n{'='*60}")
    print(f"VANILLA LORA: {T} tasks, {epochs} epochs each")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout).cuda()
    enable_gating(model, False)

    A = [[None] * T for _ in range(T)]

    for t, task_name in enumerate(task_order):
        print(f"\n{'─'*60}")
        print(f"TASK {t+1}/{T}: {task_name}")

        _train_one_task(
            model, tokenizer, task_name,
            epochs=epochs, train_batch_size=train_batch_size,
            grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
            max_source_length=max_source_length, max_target_length=max_target_length,
            max_train_samples=max_train_samples, log_every=log_every,
            optimizer_param_groups=lambda: trainable_lora_params(model))
        _free()

        print(f"\n  Evaluating after task {t+1}...")
        for i in range(t + 1):
            score = evaluate_task(
                model, task_order[i], tokenizer,
                max_eval_samples=max_eval_samples,
                max_source_length=max_source_length,
                max_target_length=max_target_length)
            A[t][i] = score
            print(f"    task {i+1:2d}: {score:.4f}")
        _free()

    for j in range(T):
        for i in range(T):
            if A[j][i] is None:
                A[j][i] = 0.0

    AP, FT = compute_ap_ft(A)
    print(f"\n  AP={AP:.4f}  FT={FT:.4f}")
    return {"A": A, "AP": AP, "FT": FT,
            "model": model, "tokenizer": tokenizer}
