"""
Sequential safety training — 3 tasks, 6 adapters.
Tasks: PKU (safety), BBQ (bias), TOFU (privacy).
Each task: learn adapter (gradient descent) + unlearn adapter (gradient ascent).
Fixes: separate optimizers, clamped unlearn loss, gradient clipping.
"""
import gc, time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from safety_data import load_all_tasks, make_dataloaders, SafetyDataset
from gpm import GPM
from safety_model import (
    build_safety_model, safety_add_task, safety_enable_gating,
    trainable_learn_params, trainable_unlearn_params,
    learn_gate_params, unlearn_gate_params,
    set_active_task_and_adapter, clear_active_task,
)
from safety_eval import eval_learn_quality, eval_forget_score, compute_safety_ap_ft

MODEL_NAME    = "t5-large"
UNLEARN_CAP   = 5.0    # clamp unlearn loss — prevents explosion
UNLEARN_SCALE = 0.1    # unlearn lr = learn_lr * this = 3e-5


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def train_safety_task(
    model, tokenizer,
    learn_loader, unlearn_loader,
    epochs, grad_accum_steps, learning_rate,
    task_name="", gpm_hook=None, log_every=100,
):
    """
    One task's training loop.
    Learn and unlearn use completely separate optimizers.
    """
    learn_opt = AdamW(
        trainable_learn_params(model) + learn_gate_params(model),
        lr=learning_rate)

    unlearn_opt = None
    if unlearn_loader is not None:
        unlearn_opt = AdamW(
            trainable_unlearn_params(model) + unlearn_gate_params(model),
            lr=learning_rate * UNLEARN_SCALE)

    autocast     = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    unlearn_iter = _infinite(unlearn_loader) if unlearn_loader is not None else None
    start        = time.time()

    for epoch in range(epochs):
        model.train()
        ep_learn, ep_unlearn, n = 0.0, 0.0, 0
        learn_opt.zero_grad()
        if unlearn_opt:
            unlearn_opt.zero_grad()

        for step, lb in enumerate(learn_loader):

            # Learn: gradient descent
            with autocast:
                out = model(input_ids=lb["input_ids"].cuda(),
                            attention_mask=lb["attention_mask"].cuda(),
                            labels=lb["labels"].cuda())
                learn_loss = out.loss / grad_accum_steps
            learn_loss.backward()
            ep_learn += learn_loss.item() * grad_accum_steps

            # Unlearn: gradient ascent, clamped
            if unlearn_iter is not None:
                ub = next(unlearn_iter)
                with autocast:
                    uout = model(input_ids=ub["input_ids"].cuda(),
                                 attention_mask=ub["attention_mask"].cuda(),
                                 labels=ub["labels"].cuda())
                    raw = uout.loss
                clamped = torch.clamp(raw, max=UNLEARN_CAP)
                (-clamped / grad_accum_steps).backward()
                ep_unlearn += clamped.item()

            n += 1

            if (step + 1) % grad_accum_steps == 0:
                # Clip unlearn grads before GPM projection
                if unlearn_opt:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_unlearn_params(model) + unlearn_gate_params(model),
                        max_norm=1.0)
                if gpm_hook:
                    gpm_hook()
                learn_opt.step();   learn_opt.zero_grad()
                if unlearn_opt:
                    unlearn_opt.step(); unlearn_opt.zero_grad()

            if (step + 1) % log_every == 0:
                ul = ep_unlearn / max(n, 1)
                print(f"    e{epoch+1} s{step+1}  "
                      f"learn={ep_learn/n:.4f}  unlearn={ul:.4f}")

        # Flush
        if (step + 1) % grad_accum_steps != 0:
            if unlearn_opt:
                torch.nn.utils.clip_grad_norm_(
                    trainable_unlearn_params(model) + unlearn_gate_params(model),
                    max_norm=1.0)
            if gpm_hook:
                gpm_hook()
            learn_opt.step(); learn_opt.zero_grad()
            if unlearn_opt:
                unlearn_opt.step(); unlearn_opt.zero_grad()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  "
                  f"learn={ep_learn/max(n,1):.4f}  "
                  f"unlearn={ep_unlearn/max(n,1):.4f}  "
                  f"t={time.time()-start:.0f}s")


def _collect_gpm(gpm, model, learn_loader):
    from gating import pool_encoder_hidden
    model.eval()
    p0s, p1s, p2s = [], [], []
    with torch.no_grad():
        for batch in learn_loader:
            ids  = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            p0   = pool_encoder_hidden(model.shared(ids), mask)
            p0s.append(p0.cpu().float())
            p1, p2 = model.learn_router.current_gate.get_activations(p0)
            p1s.append(p1.cpu().float())
            p2s.append(p2.cpu().float())
    gpm.update_bases({
        "M1": torch.cat(p0s, dim=0),
        "M2": torch.cat(p1s, dim=0),
        "M3": torch.cat(p2s, dim=0),
    })


def run_safety_gainlora(
    epochs=50,
    train_batch_size=2,
    grad_accum_steps=16,
    learning_rate=3e-4,
    lora_r=4, lora_alpha=32, lora_dropout=0.0,
    router_hidden_dim=100,
    gpm_threshold=0.97,
    n_train=500,
    n_eval=100,
    max_source_length=256,
    max_target_length=128,
    log_every=200,
):
    print(f"\n{'='*60}")
    print(f"SAFETY GAINLORA: PKU + BBQ + TOFU, {epochs} epochs each")
    print(f"learn_lr={learning_rate}, unlearn_lr={learning_rate*UNLEARN_SCALE}")
    print(f"unlearn_cap={UNLEARN_CAP}, grad_clip=1.0, batch={train_batch_size*grad_accum_steps}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load datasets ─────────────────────────────────────────────────────
    tasks, task_names = load_all_tasks(n_train=n_train, n_eval=n_eval)
    loaders = make_dataloaders(
        tasks, tokenizer, batch_size=train_batch_size,
        max_source_length=max_source_length,
        max_target_length=max_target_length)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_safety_model(
        MODEL_NAME, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, router_hidden_dim=router_hidden_dim).cuda()
    safety_enable_gating(model, True)

    T = 3
    task_keys = ["task0", "task1", "task2"]
    A_learn  = [[None]*T for _ in range(T)]
    A_forget = [[None]*T for _ in range(T)]
    gpm      = GPM(threshold=gpm_threshold)

    for t, task_key in enumerate(task_keys):
        print(f"\n{'─'*60}")
        print(f"TASK {t+1}/3: {task_names[task_key]}")
        print(f"{'─'*60}")

        gpm_hook = gpm.make_projection_hook_dual(model) if t > 0 else None

        train_safety_task(
            model, tokenizer,
            learn_loader=loaders[task_key]["learn"],
            unlearn_loader=loaders[task_key]["unlearn"],
            epochs=epochs,
            grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate,
            task_name=task_names[task_key],
            gpm_hook=gpm_hook,
            log_every=log_every,
        )
        _free()

        # ── Eval all tasks seen so far ────────────────────────────────────
        print(f"\n  Evaluating after {task_names[task_key]}...")
        model.eval()
        for i in range(t + 1):
            tk = task_keys[i]
            learn_score = eval_learn_quality(
                model, tokenizer,
                tasks[tk]["eval"]["learn"],
                task_idx=i)
            forget_score = eval_forget_score(
                model, tokenizer,
                tasks[tk]["eval"].get("unlearn"),
                task_idx=i)
            A_learn[t][i]  = learn_score
            A_forget[t][i] = forget_score
            fs = f"{forget_score:.4f}" if forget_score is not None else "N/A"
            print(f"    {task_names[tk]:35s}: "
                  f"ROUGE={learn_score:.4f}  forget={fs}")
        _free()

        # ── GPM + add_task ────────────────────────────────────────────────
        if t < T - 1:
            print(f"\n  Collecting GPM activations for {task_names[task_key]}...")
            _collect_gpm(gpm, model, loaders[task_key]["learn"])
            safety_add_task(model)
            print(f"  router now has {model.learn_router.n_tasks} tasks")
            gpm.project_init_G3_dual(model)

    # ── Final metrics ─────────────────────────────────────────────────────
    for j in range(T):
        for i in range(T):
            if A_learn[j][i]  is None: A_learn[j][i]  = 0.0
            if A_forget[j][i] is None: A_forget[j][i] = 0.0

    metrics = compute_safety_ap_ft(A_learn, A_forget, T)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"  AP_learn  = {metrics['AP_learn']:.4f}  (learn quality, higher=better)")
    print(f"  FT_learn  = {metrics['FT_learn']:.4f}  (learn forgetting, lower=better)")
    print(f"  AP_forget = {metrics['AP_forget']:.4f}  (unlearn quality, higher=better)")
    print(f"  FT_forget = {metrics['FT_forget']:.4f}  (unlearn forgetting, lower=better)")
    print(f"{'='*60}")

    # Full matrix for paper
    print(f"\nLearn quality matrix (ROUGE-L):")
    print(f"{'':35s} {'PKU':>7} {'BBQ':>7} {'TOFU':>7}")
    labels = ["PKU", "BBQ", "TOFU"]
    for j in range(T):
        row = f"  After {labels[j]:30s}: "
        for i in range(T):
            v = A_learn[j][i]
            row += f"{v:7.4f}" if v else "    ---"
        print(row)

    print(f"\nForget score matrix (higher=better unlearning):")
    print(f"{'':35s} {'PKU':>7} {'BBQ':>7} {'TOFU':>7}")
    for j in range(T):
        row = f"  After {labels[j]:30s}: "
        for i in range(T):
            v = A_forget[j][i]
            row += f"{v:7.4f}" if v is not None and v else "    N/A"
        print(row)

    return {
        "A_learn":  A_learn,
        "A_forget": A_forget,
        "metrics":  metrics,
        "model":    model,
        "tokenizer": tokenizer,
        "task_names": task_names,
    }
