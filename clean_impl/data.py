"""
Clean SuperNI data loader — bypasses HuggingFace `datasets` loading-script
machinery entirely. Matches the zero-shot preprocessing in the reference
repo's `cl_dataset.py` (load_SuperNI_dataset method).
"""
import json
import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


def _build_input_text(definition: str, instance_input: str) -> str:
    """Build the T5 input string for a single example."""
    prefix = f"Definition: {definition.strip()}\n\n"
    body = f"Now complete the following example -\nInput: {instance_input}\nOutput: "
    return prefix + body


def _pick_output(instance_output) -> str:
    if isinstance(instance_output, list):
        return instance_output[random.randint(0, len(instance_output) - 1)]
    return instance_output


def load_task_file(data_dir: str, task_name: str, split: str) -> list:
    path = Path(data_dir) / "SuperNI" / task_name / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    definition = data["Definition"]
    if isinstance(definition, list):
        definition = definition[0]
    definition = definition.strip()

    examples = []
    for idx, inst in enumerate(data["Instances"]):
        examples.append({
            "id": f"{task_name}##{split}##{idx}",
            "task": task_name,
            "input_text": _build_input_text(definition, inst["input"]),
            "target_text": _pick_output(inst["output"]),
        })
    return examples


class SuperNITaskDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        task_name: str,
        split: str,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 50,
        max_num_instances: Optional[int] = None,
        seed: int = 42,
    ):
        self.task_name = task_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        random.seed(seed)
        self.examples = load_task_file(data_dir, task_name, split)

        if max_num_instances is not None and len(self.examples) > max_num_instances:
            rng = random.Random(seed)
            self.examples = rng.sample(self.examples, max_num_instances)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        src = self.tokenizer(
            ex["input_text"],
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            ex["target_text"],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = tgt["input_ids"].squeeze(0)
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        return {
            "input_ids":       src["input_ids"].squeeze(0),
            "attention_mask":  src["attention_mask"].squeeze(0),
            "labels":          labels,
            "task":            ex["task"],
        }


SUPERNI_ORDER_1 = [
    "task1572_samsum_summary",
    "task363_sst2_polarity_classification",
    "task1290_xsum_summarization",
    "task181_outcome_extraction",
    "task002_quoref_answer_generation",
    "task1510_evalution_relation_extraction",
    "task639_multi_woz_user_utterance_generation",
    "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation",
    "task1590_diplomacy_text_generation",
    "task748_glucose_reverse_cause_event_detection",
    "task511_reddit_tifu_long_text_summarization",
    "task591_sciq_answer_generation",
    "task1687_sentiment140_classification",
    "task875_emotion_classification",
]
