"""Sequential training for T5 + LoRA + gating (Phase 3)."""
import gc, time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from data import SuperNITaskDataset
from model import (build_model, add_task, set_active_task, enable_gating,
                   trainable_lora_params, gating_params)

DATA_DIR = "/content/gainlora_690NN/CL_Benchmark"
MODEL_NAME = "t5-large"

def _free():
    gc.collect(); torch.cuda.empty_cache()

def _compute_rouge_l(predictions, references):
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    if not predictions: return 0.0
    scores = [sc.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)

@torch.no_grad()
def evaluate_task(model, task_name, tokenizer, max_eval_samples=30,
                  max_source_length=512, max_target_length=50,
                  eval_batch_size=4, active_task_idx=None):
    eval_ds = SuperNITaskDataset(data_dir=DATA_DIR, task_name=task_name, split="dev",
                                 tokenizer=tokenizer, max_source_length=max_source_length,
                                 max_target_length=max_target_length, max_num_instances=max_eval_samples)
    loader = DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False)
    prev_active = model._active_task_only
    if active_task_idx is not None:
        set_active_task(model, active_task_idx)
    model.eval()
    preds, refs = [], []
    for batch in loader:
        gen_ids = model.generate(input_ids=batch["input_ids"].cuda(),
                                 attention_mask=batch["attention_mask"].cuda(),
                                 max_new_tokens=max_target_length, num_beams=1, do_sample=False)
        preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
    set_active_task(model, prev_active)
    rouge = _compute_rouge_l(preds, refs)
    return {"rougeL": rouge, "n_eval": len(refs), "examples": list(zip(preds[:3], refs[:3]))}

def _train_one_task(model, tokenizer, task_name, epochs, train_batch_size, grad_accum_steps,
                    learning_rate, max_source_length, max_target_length, max_train_samples,
                    log_every, optimizer_param_groups, grad_mask_fn=None):
    train_ds = SuperNITaskDataset(data_dir=DATA_DIR, task_name=task_name, split="train",
                                  tokenizer=tokenizer, max_source_length=max_source_length,
                                  max_target_length=max_target_length, max_num_instances=max_train_samples)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    print(f"  [{task_name}] train: {len(train_ds)}  steps/epoch: {len(train_loader)}  "
          f"effective batch: {train_batch_size * grad_accum_steps}")
    optimizer = AdamW(optimizer_param_groups(), lr=learning_rate)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_micro = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            with autocast:
                out = model(input_ids=batch["input_ids"].cuda(),
                            attention_mask=batch["attention_mask"].cuda(),
                            labels=batch["labels"].cuda())
                loss = out.loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            n_micro += 1
            if (step + 1) % grad_accum_steps == 0:
                if grad_mask_fn is not None: grad_mask_fn()
                optimizer.step(); optimizer.zero_grad()
            if (step + 1) % log_every == 0:
                print(f"    epoch {epoch+1} step {step+1:4d}  loss={epoch_loss/n_micro:.4f}")
        if (step + 1) % grad_accum_steps != 0:
            if grad_mask_fn is not None: grad_mask_fn()
            optimizer.step(); optimizer.zero_grad()
        avg = epoch_loss / max(n_micro, 1)
        print(f"  epoch {epoch+1} done  avg loss={avg:.4f}  elapsed={time.time()-start:.0f}s")
    return avg

def _make_freeze_old_gate_mask(model, n_old_tasks):
    def _mask():
        with torch.no_grad():
            if model.gating.input_linear.grad is not None:
                model.gating.input_linear.grad[:n_old_tasks] = 0.0
            if model.gating.output_linear.grad is not None:
                model.gating.output_linear.grad[:n_old_tasks] = 0.0
    return _mask

def run_phase3_gainlora(task_0, task_1, epochs=5, train_batch_size=2, grad_accum_steps=16,
                         learning_rate_lora=3e-4, learning_rate_gate=1e-3,
                         lora_r=4, lora_alpha=32, lora_dropout=0.0, gate_hidden_dim=100,
                         max_source_length=512, max_target_length=50,
                         max_train_samples=200, max_eval_samples=30, log_every=10):
    print(f"\n{'='*60}\nGAINLORA FLOW: {task_0} -> {task_1}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout, gate_hidden_dim=gate_hidden_dim).cuda()
    # Task 0
    print(f"\n--- Train task 0: {task_0} ---")
    enable_gating(model, False)
    _train_one_task(model, tokenizer, task_0, epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate_lora,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=lambda: trainable_lora_params(model))
    _free()
    print(f"\n--- Eval task 0 after task-0 training ---")
    eval_t0_after_t0 = evaluate_task(model, task_0, tokenizer, max_eval_samples=max_eval_samples,
                                      max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t0['rougeL']:.4f}  (n={eval_t0_after_t0['n_eval']})")
    _free()
    # Add task 1
    print(f"\n--- add_task() ---")
    new_idx = add_task(model)
    enable_gating(model, True)
    print(f"  added task {new_idx}, gating enabled")
    grad_mask = _make_freeze_old_gate_mask(model, n_old_tasks=1)
    # Task 1
    print(f"\n--- Train task 1: {task_1} ---")
    def task1_params():
        return [{"params": trainable_lora_params(model), "lr": learning_rate_lora},
                {"params": gating_params(model), "lr": learning_rate_gate}]
    _train_one_task(model, tokenizer, task_1, epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate_lora,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=task1_params, grad_mask_fn=grad_mask)
    _free()
    # Eval
    print(f"\n--- Eval task 0 after task-1 training (gating on) ---")
    eval_t0_after_t1 = evaluate_task(model, task_0, tokenizer, max_eval_samples=max_eval_samples,
                                      max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t1['rougeL']:.4f}")
    _free()
    print(f"\n--- Eval task 1 (gating on) ---")
    eval_t1 = evaluate_task(model, task_1, tokenizer, max_eval_samples=max_eval_samples,
                             max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t1['rougeL']:.4f}")
    _free()
    print(f"\n--- Eval task 0 isolated (task-0 LoRA only) ---")
    eval_t0_iso = evaluate_task(model, task_0, tokenizer, max_eval_samples=max_eval_samples,
                                 max_source_length=max_source_length, max_target_length=max_target_length,
                                 active_task_idx=0)
    print(f"  ROUGE-L = {eval_t0_iso['rougeL']:.4f}  (sanity: should match {eval_t0_after_t0['rougeL']:.4f})")
    _free()
    return {"flow": "gainlora",
            "task_0_after_task_0": eval_t0_after_t0["rougeL"],
            "task_0_after_task_1": eval_t0_after_t1["rougeL"],
            "task_0_isolated": eval_t0_iso["rougeL"],
            "task_1": eval_t1["rougeL"],
            "model": model, "tokenizer": tokenizer}

def run_phase3_vanilla(task_0, task_1, epochs=5, train_batch_size=2, grad_accum_steps=16,
                        learning_rate=3e-4, lora_r=4, lora_alpha=32, lora_dropout=0.0,
                        max_source_length=512, max_target_length=50,
                        max_train_samples=200, max_eval_samples=30, log_every=10):
    print(f"\n{'='*60}\nVANILLA LORA FLOW: {task_0} -> {task_1}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout).cuda()
    enable_gating(model, False)
    print(f"\n--- Train task 0: {task_0} ---")
    _train_one_task(model, tokenizer, task_0, epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=lambda: trainable_lora_params(model))
    _free()
    print(f"\n--- Eval task 0 after task-0 training ---")
    eval_t0_after_t0 = evaluate_task(model, task_0, tokenizer, max_eval_samples=max_eval_samples,
                                      max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t0['rougeL']:.4f}")
    _free()
    print(f"\n--- Continue training same LoRA on task 1: {task_1} ---")
    _train_one_task(model, tokenizer, task_1, epochs=epochs, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, learning_rate=learning_rate,
                    max_source_length=max_source_length, max_target_length=max_target_length,
                    max_train_samples=max_train_samples, log_every=log_every,
                    optimizer_param_groups=lambda: trainable_lora_params(model))
    _free()
    print(f"\n--- Eval task 0 after task-1 training (forgetting test) ---")
    eval_t0_after_t1 = evaluate_task(model, task_0, tokenizer, max_eval_samples=max_eval_samples,
                                      max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t0_after_t1['rougeL']:.4f}")
    _free()
    print(f"\n--- Eval task 1 ---")
    eval_t1 = evaluate_task(model, task_1, tokenizer, max_eval_samples=max_eval_samples,
                             max_source_length=max_source_length, max_target_length=max_target_length)
    print(f"  ROUGE-L = {eval_t1['rougeL']:.4f}")
    _free()
    return {"flow": "vanilla",
            "task_0_after_task_0": eval_t0_after_t0["rougeL"],
            "task_0_after_task_1": eval_t0_after_t1["rougeL"],
            "task_1": eval_t1["rougeL"],
            "model": model, "tokenizer": tokenizer}
