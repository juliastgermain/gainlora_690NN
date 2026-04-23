"""
Data loaders for three safety/alignment tasks.

Task 0: PKU-SafeRLHF  — safety (harmful content)
Task 1: BBQ           — social bias
Task 2: TOFU          — privacy (memorized personal info)

Each task has:
  learn_examples:   (prompt, safe/unbiased/private-free response) pairs
  unlearn_examples: (prompt, harmful/biased/private response) pairs

T5 input format:
  Learn:   "Learn safe behavior: {prompt}"
  Unlearn: "Respond to: {prompt}"   ← model trained to INCREASE loss here
"""
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


# ── Shared dataset class ──────────────────────────────────────────────────────

class SafetyDataset(Dataset):
    def __init__(self, examples, tokenizer,
                 max_source_length=256, max_target_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src = self.tokenizer(
            ex["input_text"],
            max_length=self.max_source_length,
            truncation=True, padding="max_length", return_tensors="pt")
        tgt = self.tokenizer(
            ex["target_text"],
            max_length=self.max_target_length,
            truncation=True, padding="max_length", return_tensors="pt")
        labels = tgt["input_ids"].squeeze(0)
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        return {
            "input_ids":      src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels":         labels,
            "type":           ex["type"],
        }


def _make_example(input_text, target_text, example_type):
    return {"input_text": input_text, "target_text": target_text, "type": example_type}


# ── Task 0: PKU-SafeRLHF ─────────────────────────────────────────────────────

def load_pku(n_train=500, n_eval=100, seed=42):
    """
    PKU-SafeRLHF: each row has two responses, one safe one unsafe.
    Learn: safe response to prompt.
    Unlearn: unsafe response to prompt.
    """
    print("  Loading PKU-SafeRLHF...")
    raw_train = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    raw_test  = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")

    def extract_pairs(ds, n, seed):
        rng = random.Random(seed)
        pairs = []
        for row in ds:
            r0_safe = row.get("is_response_0_safe", False)
            r1_safe = row.get("is_response_1_safe", False)
            if r0_safe and not r1_safe:
                pairs.append({
                    "prompt": row["prompt"],
                    "safe":   row["response_0"],
                    "unsafe": row["response_1"],
                })
            elif r1_safe and not r0_safe:
                pairs.append({
                    "prompt": row["prompt"],
                    "safe":   row["response_1"],
                    "unsafe": row["response_0"],
                })
        rng.shuffle(pairs)
        return pairs[:n]

    train_pairs = extract_pairs(raw_train, n_train, seed)
    eval_pairs  = extract_pairs(raw_test,  n_eval,  seed + 1)
    print(f"    {len(train_pairs)} train pairs, {len(eval_pairs)} eval pairs")

    def to_examples(pairs):
        learn   = [_make_example(
            f"Learn safe behavior: {p['prompt']}",
            p["safe"], "learn") for p in pairs]
        unlearn = [_make_example(
            f"Respond to: {p['prompt']}",
            p["unsafe"], "unlearn") for p in pairs]
        return learn, unlearn

    train_learn, train_unlearn = to_examples(train_pairs)
    eval_learn,  eval_unlearn  = to_examples(eval_pairs)
    return {
        "train": {"learn": train_learn, "unlearn": train_unlearn},
        "eval":  {"learn": eval_learn,  "unlearn": eval_unlearn},
    }


# ── Task 1: BBQ (Bias Benchmark for QA) ──────────────────────────────────────

def load_bbq(n_train=500, n_eval=100, seed=42):
    """
    BBQ: multiple-choice QA testing social biases across 9 categories.
    Learn:   answer the question correctly (unbiased answer).
    Unlearn: the biased/stereotyped answer the model might default to.

    We use ambiguous-context questions where the "unknown" answer is correct
    (the unbiased response) and one of the other options reflects a stereotype
    (the unlearn target).

    HuggingFace: heegyu/bbq (has all 9 categories merged)
    """
    print("  Loading BBQ...")
    try:
        raw = load_dataset("heegyu/bbq", split="test")
    except Exception:
        # Fallback: load individual category
        raw = load_dataset("lighteval/bbq_helm", "Age", split="test")

    rng = random.Random(seed)
    learn_ex, unlearn_ex = [], []

    for row in raw:
        question = row.get("question", "")
        context  = row.get("context", "")
        answers  = [row.get("ans0",""), row.get("ans1",""), row.get("ans2","")]
        label    = row.get("label", 0)           # correct (unbiased) answer index
        stereo   = row.get("stereotyped_groups", None)

        if not question or not answers[0]:
            continue

        prompt = f"Context: {context}\nQuestion: {question}\nOptions: {', '.join(answers)}"
        correct_answer = answers[label]

        # Unlearn target: pick answer that is NOT correct and NOT "Unknown"
        biased_answers = [a for i, a in enumerate(answers)
                         if i != label and "unknown" not in a.lower()
                         and "cannot" not in a.lower()]
        if not biased_answers:
            continue

        biased_answer = biased_answers[0]

        learn_ex.append(_make_example(
            f"Answer without bias: {prompt}",
            correct_answer, "learn"))
        unlearn_ex.append(_make_example(
            f"Answer: {prompt}",
            biased_answer, "unlearn"))

    rng.shuffle(learn_ex)
    rng.shuffle(unlearn_ex)

    # Pair them up so train/eval splits are matched
    n_total = min(len(learn_ex), len(unlearn_ex), n_train + n_eval)
    learn_ex   = learn_ex[:n_total]
    unlearn_ex = unlearn_ex[:n_total]

    print(f"    {min(n_train, n_total)} train, {min(n_eval, n_total-n_train)} eval pairs")

    return {
        "train": {
            "learn":   learn_ex[:n_train],
            "unlearn": unlearn_ex[:n_train],
        },
        "eval": {
            "learn":   learn_ex[n_train:n_train+n_eval],
            "unlearn": unlearn_ex[n_train:n_train+n_eval],
        },
    }


# ── Task 2: TOFU (Privacy / Memorization) ────────────────────────────────────

def load_tofu(n_train=500, n_eval=100, seed=42):
    """
    TOFU (Task of Fictitious Unlearning): QA about fictional authors.
    The "forget set" contains specific authors whose info should be unlearned.
    The "retain set" contains authors whose info should be preserved.

    Learn:   answer correctly about RETAIN authors (preserve this knowledge)
    Unlearn: the memorized answer about FORGET authors (erase this)

    HuggingFace: locuslab/TOFU
      split="forget10"  → 10% of authors to forget
      split="retain90"  → 90% of authors to retain
    """
    print("  Loading TOFU...")
    try:
        forget = load_dataset("locuslab/TOFU", "forget10", split="train")
        retain = load_dataset("locuslab/TOFU", "retain90", split="train")
    except Exception as e:
        print(f"    TOFU load error: {e}")
        print("    Falling back to full split...")
        full = load_dataset("locuslab/TOFU", "full", split="train")
        # Use first 20% as forget, rest as retain
        n = len(full)
        forget = full.select(range(n // 5))
        retain = full.select(range(n // 5, n))

    rng = random.Random(seed)

    def ds_to_examples(ds, prefix, ex_type, n):
        rows = list(ds)
        rng.shuffle(rows)
        return [
            _make_example(
                f"{prefix}: {row['question']}",
                row["answer"],
                ex_type)
            for row in rows[:n]
        ]

    train_learn   = ds_to_examples(retain, "Answer correctly", "learn",   n_train)
    train_unlearn = ds_to_examples(forget, "Answer",           "unlearn", n_train)
    eval_learn    = ds_to_examples(retain, "Answer correctly", "learn",   n_eval)
    eval_unlearn  = ds_to_examples(forget, "Answer",           "unlearn", n_eval)

    print(f"    {len(train_learn)} learn train, {len(train_unlearn)} unlearn train")
    print(f"    {len(eval_learn)} learn eval, {len(eval_unlearn)} unlearn eval")

    return {
        "train": {"learn": train_learn, "unlearn": train_unlearn},
        "eval":  {"learn": eval_learn,  "unlearn": eval_unlearn},
    }


# ── Master loader ─────────────────────────────────────────────────────────────

def load_all_tasks(n_train=500, n_eval=100, seed=42):
    """
    Load all three tasks.
    Returns:
        tasks: dict with keys "task0", "task1", "task2"
               each has "train" and "eval" with "learn" and "unlearn" lists
        task_names: human-readable names for printing
    """
    print("Loading all safety tasks...")
    tasks = {
        "task0": load_pku(  n_train=n_train, n_eval=n_eval, seed=seed),
        "task1": load_bbq(  n_train=n_train, n_eval=n_eval, seed=seed),
        "task2": load_tofu( n_train=n_train, n_eval=n_eval, seed=seed),
    }
    task_names = {
        "task0": "PKU-SafeRLHF (safety)",
        "task1": "BBQ (bias)",
        "task2": "TOFU (privacy)",
    }
    for k, v in tasks.items():
        print(f"  {task_names[k]}: "
              f"{len(v['train']['learn'])} learn train, "
              f"{len(v['train']['unlearn'] or [])} unlearn train")
    return tasks, task_names


def make_dataloaders(tasks, tokenizer, batch_size=2,
                     max_source_length=256, max_target_length=128):
    """Convert to DataLoaders. Returns same structure with DataLoader values."""
    loaders = {}
    for task_key, task_data in tasks.items():
        learn_ds = SafetyDataset(
            task_data["train"]["learn"], tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length)
        learn_loader = DataLoader(learn_ds, batch_size=batch_size, shuffle=True)

        unlearn_loader = None
        if task_data["train"]["unlearn"]:
            unlearn_ds = SafetyDataset(
                task_data["train"]["unlearn"], tokenizer,
                max_source_length=max_source_length,
                max_target_length=max_target_length)
            unlearn_loader = DataLoader(
                unlearn_ds, batch_size=batch_size, shuffle=True)

        loaders[task_key] = {
            "learn":   learn_loader,
            "unlearn": unlearn_loader,
        }
    return loaders
