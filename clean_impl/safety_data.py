"""
Data loaders for safety tasks — loads from preset JSON files in the repo.

All files share the same format:
  {"instruction": "...", "output": "...", ...}

PKU:
  safety_learn.json         → task0: PKU-Learn   (2500 examples)
  safety_unlearn.json       → task1: PKU-Unlearn  (2500 examples)
  test_safe_prompts.json    → PKU eval learn      (300 examples)
  test_harmful_prompts.json → PKU eval unlearn    (300 examples)

BBQ:
  bbq_learn_data.json       → task2: BBQ-Learn    (2500 examples)
  bbq_unlearn_data.json     → task3: BBQ-Unlearn  (2500 examples)
  bbq_test_factual.json     → BBQ eval learn      (272 examples)
  bbq_test_ambiguous.json   → BBQ eval unlearn    (272 examples)

TruthfulQA: not yet available, will be added as task4/task5.
"""
import json
import os
import random
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "/content/gainlora_690NN"


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
            "is_unlearn":     ex["is_unlearn"],
        }


def _ex(input_text, target_text, is_unlearn=False):
    return {"input_text": input_text, "target_text": target_text,
            "is_unlearn": is_unlearn}


def _load_json(filename):
    path = os.path.join(DATA_ROOT, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rows_to_examples(rows, input_prefix, is_unlearn, n=None, seed=42):
    """
    Convert rows with 'instruction'/'output' keys to example dicts.
    Shuffles and optionally caps at n examples.
    """
    rng = random.Random(seed)
    examples = []
    for row in rows:
        instruction = row.get("instruction", "").strip()
        output      = row.get("output", "").strip()
        if not instruction or not output:
            continue
        examples.append(_ex(
            f"{input_prefix}{instruction}",
            output,
            is_unlearn=is_unlearn))
    rng.shuffle(examples)
    return examples[:n] if n is not None else examples


# ── PKU tasks ─────────────────────────────────────────────────────────────────

def load_pku_tasks(n_train=None, n_eval=None, seed=42):
    print("  Loading PKU from preset JSON files...")
    task0_train = _rows_to_examples(
        _load_json("safety_learn.json"),
        "Learn safe behavior: ", is_unlearn=False, n=n_train, seed=seed)
    task1_train = _rows_to_examples(
        _load_json("safety_unlearn.json"),
        "Learn safe behavior: ", is_unlearn=True, n=n_train, seed=seed)
    task0_eval = _rows_to_examples(
        _load_json("test_safe_prompts.json"),
        "Learn safe behavior: ", is_unlearn=False, n=n_eval, seed=seed+1)
    task1_eval = _rows_to_examples(
        _load_json("test_harmful_prompts.json"),
        "Learn safe behavior: ", is_unlearn=True, n=n_eval, seed=seed+1)

    print(f"    PKU-Learn:   {len(task0_train)} train, {len(task0_eval)} eval")
    print(f"    PKU-Unlearn: {len(task1_train)} train, {len(task1_eval)} eval")
    return {
        "task0": {"train": task0_train, "eval": task0_eval,
                  "name": "PKU-Learn (safety)"},
        "task1": {"train": task1_train, "eval": task1_eval,
                  "name": "PKU-Unlearn (safety)"},
    }


# ── BBQ tasks ─────────────────────────────────────────────────────────────────

def load_bbq_tasks(n_train=None, n_eval=None, seed=42):
    print("  Loading BBQ from preset JSON files...")
    task2_train = _rows_to_examples(
        _load_json("bbq_learn_data.json"),
        "Answer without bias: ", is_unlearn=False, n=n_train, seed=seed)
    task3_train = _rows_to_examples(
        _load_json("bbq_unlearn_data.json"),
        "Answer without bias: ", is_unlearn=True, n=n_train, seed=seed)
    task2_eval = _rows_to_examples(
        _load_json("bbq_test_factual.json"),        # factual = learn eval
        "Answer without bias: ", is_unlearn=False, n=n_eval, seed=seed+1)
    task3_eval = _rows_to_examples(
        _load_json("bbq_test_ambiguous.json"),      # ambiguous = unlearn eval
        "Answer without bias: ", is_unlearn=True, n=n_eval, seed=seed+1)

    print(f"    BBQ-Learn:   {len(task2_train)} train, {len(task2_eval)} eval")
    print(f"    BBQ-Unlearn: {len(task3_train)} train, {len(task3_eval)} eval")
    return {
        "task2": {"train": task2_train, "eval": task2_eval,
                  "name": "BBQ-Learn (bias)"},
        "task3": {"train": task3_train, "eval": task3_eval,
                  "name": "BBQ-Unlearn (bias)"},
    }


# ── TruthfulQA tasks (placeholder) ───────────────────────────────────────────

def load_truthfulqa_tasks(n_train=None, n_eval=None, seed=42):
    learn_path = os.path.join(DATA_ROOT, "truthfulqa_learn.json")
    if not os.path.exists(learn_path):
        print("  TruthfulQA: files not yet available, skipping tasks 4/5")
        return {}

    print("  Loading TruthfulQA from preset JSON files...")
    task4_train = _rows_to_examples(
        _load_json("truthfulqa_learn.json"),
        "Answer truthfully: ", is_unlearn=False, n=n_train, seed=seed)
    task5_train = _rows_to_examples(
        _load_json("truthfulqa_unlearn.json"),
        "Answer truthfully: ", is_unlearn=True, n=n_train, seed=seed)

    eval_learn_path = os.path.join(DATA_ROOT, "truthfulqa_eval_learn.json")
    if os.path.exists(eval_learn_path):
        task4_eval = _rows_to_examples(
            _load_json("truthfulqa_eval_learn.json"),
            "Answer truthfully: ", is_unlearn=False, n=n_eval, seed=seed+1)
        task5_eval = _rows_to_examples(
            _load_json("truthfulqa_eval_unlearn.json"),
            "Answer truthfully: ", is_unlearn=True, n=n_eval, seed=seed+1)
    else:
        # Fall back to tail of training data
        task4_eval = task4_train[-(n_eval or 100):]
        task5_eval = task5_train[-(n_eval or 100):]

    print(f"    TruthfulQA-Learn:   {len(task4_train)} train, {len(task4_eval)} eval")
    print(f"    TruthfulQA-Unlearn: {len(task5_train)} train, {len(task5_eval)} eval")
    return {
        "task4": {"train": task4_train, "eval": task4_eval,
                  "name": "TruthfulQA-Learn (misinformation)"},
        "task5": {"train": task5_train, "eval": task5_eval,
                  "name": "TruthfulQA-Unlearn (misinformation)"},
    }


# ── Master loader ─────────────────────────────────────────────────────────────

def load_all_tasks(n_train=None, n_eval=None, seed=42):
    """
    Load all available tasks from preset JSON files.
    n_train=None uses all available examples (recommended —
    your preset files are already the right size).
    """
    print("Loading safety tasks from preset JSON files...")
    tasks = {}
    tasks.update(load_pku_tasks(n_train=n_train, n_eval=n_eval, seed=seed))
    tasks.update(load_bbq_tasks(n_train=n_train, n_eval=n_eval, seed=seed))
    tasks.update(load_truthfulqa_tasks(n_train=n_train, n_eval=n_eval, seed=seed))

    task_names = {k: v["name"] for k, v in tasks.items()}
    print(f"\nLoaded {len(tasks)} tasks:")
    for k, v in tasks.items():
        print(f"  {v['name']:40s}: {len(v['train'])} train, {len(v['eval'])} eval")
    return tasks, task_names


def make_dataloaders(tasks, tokenizer, batch_size=2,
                     max_source_length=256, max_target_length=128):
    loaders = {}
    for task_key, task_data in tasks.items():
        ds = SafetyDataset(
            task_data["train"], tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length)
        loaders[task_key] = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loaders
