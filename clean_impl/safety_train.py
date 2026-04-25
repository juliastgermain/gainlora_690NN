"""
Sequential GainLoRA training — 6 tasks, 1 adapter each.
Tasks 0,2,4: LEARN  (gradient descent)
Tasks 1,3,5: UNLEARN (gradient ascent)
Checkpoints: model/GPM → Google Drive, metadata → GitHub.
"""
import gc, time, json, os, subprocess
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from safety_data import load_all_tasks, make_dataloaders
from gpm import GPM
from model import (build_model, add_task, enable_gating,
                   trainable_lora_params, gating_params, set_active_task)

MODEL_NAME   = "t5-large"
UNLEARN_CAP  = 5.0

# Checkpoints: large files → Drive, metadata → repo/checkpoints/
DRIVE_DIR    = "/content/drive/MyDrive/gainlora_checkpoints"
META_DIR     = "/content/gainlora_690NN/checkpoints"


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(model, gpm, A_learn, A_forget,
                    completed_tasks, task_order, task_names):
    os.makedirs(DRIVE_DIR, exist_ok=True)
    os.makedirs(META_DIR,  exist_ok=True)
    t = len(completed_tasks) - 1

    # Large files → Drive
    torch.save(model.state_dict(),
               f"{DRIVE_DIR}/model_after_task{t}.pt")
    gpm_save = {k: v.cpu() if v is not None else None
                for k, v in gpm.bases.items()}
    torch.save(gpm_save, f"{DRIVE_DIR}/gpm_after_task{t}.pt")
    print(f"  ✓ Model + GPM saved to Drive (task {t})")

    # Small metadata → repo so it can be pushed to GitHub
    meta = {
        "completed_tasks": completed_tasks,
        "task_order":      task_order,
        "task_names":      task_names,
        "A_learn":         A_learn,
        "A_forget":        A_forget,
        "n_router_tasks":  model.router.n_tasks,
    }
    meta_path = f"{META_DIR}/meta_after_task{t}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    GH_TOKEN = os.environ.get("GH_TOKEN", "")
    if GH_TOKEN:
        try:
            subprocess.run(["git", "-C", "/content/gainlora_690NN",
                            "add", "checkpoints/"], capture_output=True)
            subprocess.run(["git", "-C", "/content/gainlora_690NN",
                            "commit", "-m", f"checkpoint meta task {t}"],
                           capture_output=True)
            subprocess.run(["git", "-C", "/content/gainlora_690NN", "push",
                            f"https://juliastgermain:{GH_TOKEN}@github.com/"
                            f"juliastgermain/gainlora_690NN.git", "master"],
                           capture_output=True)
            print(f"  ✓ Metadata pushed to GitHub")
        except Exception as e:
            print(f"  ⚠ GitHub push failed: {e}")

    print(f"  ✓ Checkpoint complete: task {t} ({task_names[task_order[t]]})")


def find_latest_checkpoint(task_order):
    """Check Drive for latest saved model checkpoint."""
    if not os.path.exists(DRIVE_DIR):
        return -1
    for t in range(len(task_order) - 1, -1, -1):
        if os.path.exists(f"{DRIVE_DIR}/model_after_task{t}.pt"):
            print(f"  Found Drive checkpoint after task {t}")
            return t
    return -1


def load_checkpoint(model, gpm, task_idx):
    """Load model + GPM from Drive, metadata from repo."""
    state = torch.load(f"{DRIVE_DIR}/model_after_task{task_idx}.pt",
                       map_location="cuda")
    model.load_state_dict(state, strict=False)

    gpm_save = torch.load(f"{DRIVE_DIR}/gpm_after_task{task_idx}.pt",
                          map_location="cpu")
    gpm.bases = gpm_save

    meta_path = f"{META_DIR}/meta_after_task{task_idx}.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"A_learn": None, "A_forget": None,
                "n_router_tasks": task_idx + 1}

    print(f"  ✓ Loaded checkpoint: task {task_idx}")
    return meta


# ── Training utilities ────────────────────────────────────────────────────────

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
def evaluate_task(model, examples, tokenizer, task_idx,
                  max_source_length=256, max_target_length=128, batch_size=8):
    from safety_data import SafetyDataset
    ds     = SafetyDataset(examples, tokenizer, max_source_length, max_target_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    set_active_task(model, task_idx)
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
    set_active_task(model, None)
    return _rouge_l(preds, refs)


@torch.no_grad()
def eval_forget_score(model, examples, tokenizer, task_idx,
                      max_source_length=256, max_target_length=128, batch_size=8):
    from safety_data import SafetyDataset
    ds     = SafetyDataset(examples, tokenizer, max_source_length, max_target_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    set_active_task(model, task_idx)
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        out = model(input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    labels=batch["labels"].cuda())
        total += out.loss.item()
        n += 1
    set_active_task(model, None)
    return min(total / max(n, 1) / 10.0, 1.0)


def _train_one_task(model, loader, is_unlearn, epochs,
                    grad_accum_steps, learning_rate,
                    gpm_hook=None, log_every=500,
                    early_stop_patience=3, early_stop_delta=0.01):
    params    = trainable_lora_params(model) + gating_params(model)
    optimizer = AdamW(params, lr=learning_rate)
    autocast  = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    start     = time.time()
    best_loss, epochs_no_imp = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        ep_loss, n = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            with autocast:
                out = model(input_ids=batch["input_ids"].cuda(),
                            attention_mask=batch["attention_mask"].cuda(),
                            labels=batch["labels"].cuda())
                raw_loss = out.loss

            if is_unlearn:
                clamped = torch.clamp(raw_loss, max=UNLEARN_CAP)
                loss     = -clamped / grad_accum_steps
                ep_loss += clamped.item()
            else:
                loss     = raw_loss / grad_accum_steps
                ep_loss += raw_loss.item()

            loss.backward()
            n += 1

            if (step + 1) % grad_accum_steps == 0:
                if is_unlearn:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                if gpm_hook:
                    gpm_hook()
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % log_every == 0:
                print(f"    e{epoch+1} s{step+1}  "
                      f"loss={ep_loss/n:.4f} "
                      f"{'↑ascent' if is_unlearn else '↓descent'}")

        # Flush remaining grads
        if (step + 1) % grad_accum_steps != 0:
            if is_unlearn:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            if gpm_hook:
                gpm_hook()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = ep_loss / max(n, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  loss={avg_loss:.4f} "
                  f"({'ascent' if is_unlearn else 'descent'})  "
                  f"t={time.time()-start:.0f}s")

        # Early stopping
        if is_unlearn:
            if avg_loss >= UNLEARN_CAP * 0.99:
                print(f"  ✓ Early stop epoch {epoch+1}: saturated at {avg_loss:.4f}")
                break
        else:
            if avg_loss < best_loss - early_stop_delta:
                best_loss     = avg_loss
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= early_stop_patience:
                    print(f"  ✓ Early stop epoch {epoch+1}: "
                          f"no improvement for {early_stop_patience} epochs "
                          f"(best={best_loss:.4f})")
                    break


def _collect_gpm(gpm, model, loader):
    from gating import pool_encoder_hidden
    model.eval()
    p0s, p1s, p2s = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            p0   = pool_encoder_hidden(model.shared(ids), mask)
            p0s.append(p0.cpu().float())
            p1, p2 = model.router.current_gate.get_activations(p0)
            p1s.append(p1.cpu().float())
            p2s.append(p2.cpu().float())
    gpm.update_bases({
        "M1": torch.cat(p0s, dim=0),
        "M2": torch.cat(p1s, dim=0),
        "M3": torch.cat(p2s, dim=0),
    })


# ── Main entry point ──────────────────────────────────────────────────────────

def run_safety_gainlora(
    task_order=None,
    epochs=20,
    train_batch_size=8,
    grad_accum_steps=4,
    learning_rate=3e-4,
    lora_r=4, lora_alpha=32, lora_dropout=0.0,
    router_hidden_dim=100,
    gpm_threshold=0.97,
    n_train=None,      # None = use all preset data
    n_eval=None,
    max_source_length=256, max_target_length=128,
    log_every=500,
):
    tasks, task_names = load_all_tasks(n_train=n_train, n_eval=n_eval)

    if task_order is None:
        task_order = list(tasks.keys())

    T = len(task_order)
    print(f"\n{'='*60}")
    print(f"SAFETY GAINLORA: {T} tasks")
    print(f"Order: {' → '.join(task_names[k] for k in task_order)}")
    print(f"lr={learning_rate}, unlearn_cap={UNLEARN_CAP}, "
          f"batch={train_batch_size*grad_accum_steps}, r={lora_r}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    loaders   = make_dataloaders(tasks, tokenizer, batch_size=train_batch_size,
                                 max_source_length=max_source_length,
                                 max_target_length=max_target_length)

    model = build_model(MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        router_hidden_dim=router_hidden_dim).cuda()
    enable_gating(model, True)

    A_learn  = [[None]*T for _ in range(T)]
    A_forget = [[None]*T for _ in range(T)]
    gpm      = GPM(threshold=gpm_threshold)

    # Resume from checkpoint if available
    last_completed = find_latest_checkpoint(task_order)
    if last_completed >= 0:
        print(f"\n  Resuming from Drive checkpoint after task {last_completed}...")
        meta     = load_checkpoint(model, gpm, last_completed)
        A_learn  = meta["A_learn"]  or A_learn
        A_forget = meta["A_forget"] or A_forget
        n_needed = meta["n_router_tasks"]
        while model.router.n_tasks < n_needed:
            add_task(model)
        start_t = last_completed + 1
        print(f"  Resuming from task {start_t}/{T}")
    else:
        start_t = 0

    for t in range(start_t, T):
        task_key   = task_order[t]
        task_data  = tasks[task_key]
        task_name  = task_names[task_key]
        is_unlearn = task_data["train"][0]["is_unlearn"]
        loader     = loaders[task_key]

        print(f"\n{'─'*60}")
        print(f"TASK {t+1}/{T}: {task_name}  "
              f"[{'UNLEARN' if is_unlearn else 'LEARN'}]")
        print(f"{'─'*60}")

        gpm_hook = gpm.make_projection_hook(model) if t > 0 else None
        _train_one_task(model, loader, is_unlearn=is_unlearn,
                        epochs=epochs, grad_accum_steps=grad_accum_steps,
                        learning_rate=learning_rate, gpm_hook=gpm_hook,
                        log_every=log_every)
        _free()

        # Eval all tasks seen so far
        print(f"\n  Evaluating after {task_name}...")
        model.eval()
        for i in range(t + 1):
            tk       = task_order[i]
            td       = tasks[tk]
            is_unl_i = td["train"][0]["is_unlearn"]
            if is_unl_i:
                score = eval_forget_score(model, td["eval"], tokenizer, task_idx=i,
                                          max_source_length=max_source_length,
                                          max_target_length=max_target_length)
                A_forget[t][i] = score
                A_learn[t][i]  = 0.0
                print(f"    [{i}] {task_names[tk]:40s}: forget_score={score:.4f}")
            else:
                score = evaluate_task(model, td["eval"], tokenizer, task_idx=i,
                                      max_source_length=max_source_length,
                                      max_target_length=max_target_length)
                A_learn[t][i]  = score
                A_forget[t][i] = 0.0
                print(f"    [{i}] {task_names[tk]:40s}: ROUGE-L={score:.4f}")
        _free()

        # GPM + add_task
        if t < T - 1:
            print(f"\n  Collecting GPM activations for {task_name}...")
            _collect_gpm(gpm, model, loader)
            add_task(model)
            print(f"  router now has {model.router.n_tasks} tasks")
            gpm.project_init_G3(model)

        # Save after every task
        save_checkpoint(model, gpm, A_learn, A_forget,
                        completed_tasks=task_order[:t+1],
                        task_order=task_order,
                        task_names=task_names)

    # Fill None entries
    for j in range(T):
        for i in range(T):
            if A_learn[j][i]  is None: A_learn[j][i]  = 0.0
            if A_forget[j][i] is None: A_forget[j][i] = 0.0

    _print_results(A_learn, A_forget, task_order, task_names, tasks, T)

    return {
        "A_learn":   A_learn,  "A_forget":  A_forget,
        "task_order": task_order, "task_names": task_names,
        "_tasks":    tasks,    "model":     model,
        "tokenizer": tokenizer,
    }


def _print_results(A_learn, A_forget, task_order, task_names, tasks, T):
    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    learn_idx   = [i for i, k in enumerate(task_order)
                   if not tasks[k]["train"][0]["is_unlearn"]]
    unlearn_idx = [i for i, k in enumerate(task_order)
                   if tasks[k]["train"][0]["is_unlearn"]]

    print(f"\nLearn task ROUGE-L matrix:")
    headers = [f"T{i}({task_names[task_order[i]][:8]})" for i in learn_idx]
    print(f"  {'After task':<25} " + " ".join(f"{h:>12}" for h in headers))
    for j in range(T):
        row = f"  {task_names[task_order[j]][:25]:<25} "
        for i in learn_idx:
            v = A_learn[j][i]
            row += f"{v:>12.4f}" if v else "         ---"
        print(row)

    print(f"\nUnlearn task forget score matrix (higher=better):")
    headers = [f"T{i}({task_names[task_order[i]][:8]})" for i in unlearn_idx]
    print(f"  {'After task':<25} " + " ".join(f"{h:>12}" for h in headers))
    for j in range(T):
        row = f"  {task_names[task_order[j]][:25]:<25} "
        for i in unlearn_idx:
            v = A_forget[j][i]
            row += f"{v:>12.4f}" if v else "         ---"
        print(row)

    last_learn  = [A_learn[T-1][i]  for i in learn_idx]
    last_forget = [A_forget[T-1][i] for i in unlearn_idx]
    AP_learn  = sum(last_learn)  / len(last_learn)  if last_learn  else 0
    AP_forget = sum(last_forget) / len(last_forget) if last_forget else 0

    FT_learn = 0.0
    for i in learn_idx:
        vals = [A_learn[j][i] for j in range(i, T) if A_learn[j][i] > 0]
        if vals:
            FT_learn += max(vals) - A_learn[T-1][i]
    FT_learn /= max(len(learn_idx), 1)

    FT_forget = 0.0
    for i in unlearn_idx:
        vals = [A_forget[j][i] for j in range(i, T) if A_forget[j][i] > 0]
        if vals:
            FT_forget += max(vals) - A_forget[T-1][i]
    FT_forget /= max(len(unlearn_idx), 1)

    print(f"\nSummary:")
    print(f"  AP_learn  = {AP_learn:.4f}   avg ROUGE-L across learn tasks")
    print(f"  FT_learn  = {FT_learn:.4f}   learn forgetting (lower=better)")
    print(f"  AP_forget = {AP_forget:.4f}   avg forget score across unlearn tasks")
    print(f"  FT_forget = {FT_forget:.4f}   unlearn forgetting (lower=better)")
    if FT_forget > FT_learn + 0.05:
        print(f"\n  ✓ FT_forget ({FT_forget:.4f}) >> FT_learn ({FT_learn:.4f})")
        print(f"    GainLoRA fails to preserve unlearning across tasks.")
    else:
        print(f"\n  FT_forget={FT_forget:.4f}, FT_learn={FT_learn:.4f}")
        print(f"  Run gated evaluation for full results.")
