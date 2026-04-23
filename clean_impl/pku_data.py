"""
PKU-SafeRLHF dataset loader for safety learn/unlearn experiments.

Dataset: PKU-Alignment/PKU-SafeRLHF
Each example has a prompt + two responses with safety labels.

We create three task splits:

  Task 0 (General Helpfulness):
    - Source: safe responses from PKU, filtered to benign topics
    - Learn adapter: train on safe (prompt, response) pairs
    - No unlearn adapter (first task, no harmful content to forget)

  Task 1 (Safety on PKU):
    - Learn adapter:   safe responses → gradient descent
    - Unlearn adapter: unsafe responses → gradient ascent

  Task 2 (Domain shift — e.g. only a specific harm category):
    - Learn adapter:   safe responses from a held-out category
    - Unlearn adapter: unsafe responses from that category → gradient ascent

Each split returns two DataLoaders: (learn_loader, unlearn_loader).
unlearn_loader is None for Task 0.

Label format for T5:
  Learn:   input = "Learn to respond safely: {prompt}"
           target = "{safe_response}"
  Unlearn: input = "Respond to: {prompt}"
           target = "{unsafe_response}"   ← model trained to INCREASE loss here
"""
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ── Prompt templates ──────────────────────────────────────────────────────────

LEARN_PREFIX   = "Learn to respond safely: "
UNLEARN_PREFIX = "Respond to: "


def _build_learn_example(prompt, safe_response):
    return {
        "input_text":  LEARN_PREFIX + prompt.strip(),
        "target_text": safe_response.strip(),
        "type": "learn",
    }

def _build_unlearn_example(prompt, unsafe_response):
    return {
        "input_text":  UNLEARN_PREFIX + prompt.strip(),
        "target_text": unsafe_response.strip(),
        "type": "unlearn",
    }


# ── Dataset class ─────────────────────────────────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_pku_raw(split="train", max_rows=None):
    """Load PKU-SafeRLHF from HuggingFace."""
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds


def _extract_pairs(ds, seed=42):
    """
    Extract (prompt, safe_response, unsafe_response) triples.
    Returns list of dicts with keys: prompt, safe, unsafe.
    Some examples may have both responses safe or both unsafe —
    we skip those and only keep clear safe/unsafe pairs.
    """
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
        # skip ambiguous rows

    rng.shuffle(pairs)
    return pairs


def make_task_splits(
    n_task0=500,     # general helpfulness (safe only, no unlearn)
    n_task1=500,     # safety task 1
    n_task2=500,     # safety task 2 (held-out slice)
    seed=42,
    max_pku_rows=20000,
):
    """
    Build learn/unlearn example lists for all 3 tasks.
    Returns dict: {
        "task0": {"learn": [...], "unlearn": None},
        "task1": {"learn": [...], "unlearn": [...]},
        "task2": {"learn": [...], "unlearn": [...]},
    }
    """
    print("Loading PKU-SafeRLHF...")
    raw = _load_pku_raw(split="train", max_rows=max_pku_rows)
    pairs = _extract_pairs(raw, seed=seed)
    print(f"  {len(pairs)} clean safe/unsafe pairs extracted")

    # Split into three non-overlapping slices
    n_total = n_task0 + n_task1 + n_task2
    if len(pairs) < n_total:
        print(f"  Warning: only {len(pairs)} pairs, reducing splits proportionally")
        factor = len(pairs) / n_total
        n_task0 = int(n_task0 * factor)
        n_task1 = int(n_task1 * factor)
        n_task2 = int(n_task2 * factor)

    slice0 = pairs[:n_task0]
    slice1 = pairs[n_task0 : n_task0 + n_task1]
    slice2 = pairs[n_task0 + n_task1 : n_task0 + n_task1 + n_task2]

    task0_learn   = [_build_learn_example(p["prompt"], p["safe"])  for p in slice0]
    task1_learn   = [_build_learn_example(p["prompt"], p["safe"])  for p in slice1]
    task1_unlearn = [_build_unlearn_example(p["prompt"], p["unsafe"]) for p in slice1]
    task2_learn   = [_build_learn_example(p["prompt"], p["safe"])  for p in slice2]
    task2_unlearn = [_build_unlearn_example(p["prompt"], p["unsafe"]) for p in slice2]

    print(f"  Task 0: {len(task0_learn)} learn examples, no unlearn")
    print(f"  Task 1: {len(task1_learn)} learn, {len(task1_unlearn)} unlearn")
    print(f"  Task 2: {len(task2_learn)} learn, {len(task2_unlearn)} unlearn")

    return {
        "task0": {"learn": task0_learn, "unlearn": None},
        "task1": {"learn": task1_learn, "unlearn": task1_unlearn},
        "task2": {"learn": task2_learn, "unlearn": task2_unlearn},
    }


def make_dataloaders(splits, tokenizer, batch_size=2,
                     max_source_length=256, max_target_length=128):
    """
    Convert example lists to DataLoaders.
    Returns dict of same structure with DataLoader values.
    """
    loaders = {}
    for task_key, task_data in splits.items():
        learn_ds = SafetyDataset(
            task_data["learn"], tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length)
        learn_loader = DataLoader(learn_ds, batch_size=batch_size, shuffle=True)

        unlearn_loader = None
        if task_data["unlearn"] is not None:
            unlearn_ds = SafetyDataset(
                task_data["unlearn"], tokenizer,
                max_source_length=max_source_length,
                max_target_length=max_target_length)
            unlearn_loader = DataLoader(unlearn_ds, batch_size=batch_size, shuffle=True)

        loaders[task_key] = {
            "learn":   learn_loader,
            "unlearn": unlearn_loader,
        }
    return loaders


# ── Evaluation data ───────────────────────────────────────────────────────────

def make_eval_split(n_eval=200, seed=99, max_pku_rows=5000):
    """
    Separate held-out eval set (never seen during training).
    Returns (learn_examples, unlearn_examples).
    """
    raw = _load_pku_raw(split="test", max_rows=max_pku_rows)
    pairs = _extract_pairs(raw, seed=seed)
    pairs = pairs[:n_eval]
    learn_ex   = [_build_learn_example(p["prompt"], p["safe"])   for p in pairs]
    unlearn_ex = [_build_unlearn_example(p["prompt"], p["unsafe"]) for p in pairs]
    return learn_ex, unlearn_ex
