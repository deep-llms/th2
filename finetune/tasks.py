"""Task definitions for generative fine-tuning.

Each task formats training data as (prompt, completion) pairs matching
the EXACT format that lm-evaluation-harness uses at eval time. Training
loss is computed only on completion tokens (prompt is masked with -100).

Right padding: text first, then padding after. Standard for causal LM
training — the model processes real tokens left-to-right, ignoring pads.
"""

import re

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


# -----------------------------------------------------------------------
# HellaSwag preprocessing (copied from lm-eval-harness hellaswag/utils.py)
# -----------------------------------------------------------------------

def _preprocess_hellaswag(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


# -----------------------------------------------------------------------
# Dataset: tokenized (prompt, completion) with labels=-100 for prompt
# -----------------------------------------------------------------------

class GenerativeDataset(Dataset):
    """Tokenized (prompt + completion) with loss masking on prompt tokens.

    Right-padded: real tokens first, padding at end. Labels use -100
    for both prompt and padding positions.
    """

    def __init__(self, prompts, completions, tokenizer, max_length):
        self.items = []
        pad_id = tokenizer.pad_token_id

        for prompt, completion in zip(prompts, completions):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
            comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

            full_ids = prompt_ids + comp_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            # Labels: -100 for prompt, real ids for completion
            n_prompt = min(len(prompt_ids), len(full_ids))
            if n_prompt >= len(full_ids):
                continue  # skip: prompt fills entire sequence, no completion
            labels = [-100] * n_prompt + full_ids[n_prompt:]

            # Right-pad
            pad_len = max_length - len(full_ids)
            input_ids = full_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * len(full_ids) + [0] * pad_len

            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# -----------------------------------------------------------------------
# Task configs
# -----------------------------------------------------------------------

TASK_CONFIGS = {
    "hellaswag": {
        "max_length": 256,
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 16,
        "eval_tasks": ["hellaswag", "hellaswag_ar", "hellaswag_de",
                        "hellaswag_ru", "hellaswag_vi"],
    },
    "arc_easy": {
        "max_length": 256,
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 32,
        "eval_tasks": ["arc_easy", "arc_ar", "arc_de", "arc_ru",
                        "arc_vi", "arc_zh"],
    },
    "xnli": {
        "max_length": 256,
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 32,
        "eval_tasks": ["xnli_en", "xnli_vi", "xnli_zh", "xnli_de",
                        "xnli_ru", "xnli_ar"],
    },
}

XNLI_KEYWORDS = {
    "en": {"q": "right", "yes": "Yes", "also": "Also", "no": "No"},
    "vi": {"q": "đúng", "yes": "Vâng", "also": "Vì vậy", "no": "Không"},
    "zh": {"q": "正确", "yes": "是的", "also": "所以", "no": "不是的"},
    "de": {"q": "richtig", "yes": "Ja", "also": "Auch", "no": "Nein"},
    "ru": {"q": "правильно", "yes": "Да", "also": "Так", "no": "Нет"},
    "ar": {"q": "صحيح", "yes": "نعم", "also": "لذا", "no": "رقم"},
}


# -----------------------------------------------------------------------
# Format functions: match lm-eval-harness exactly
# -----------------------------------------------------------------------

def format_hellaswag(doc):
    """Match hellaswag.yaml: doc_to_text={{query}}, doc_to_choice=choices."""
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = _preprocess_hellaswag(doc["activity_label"] + ": " + ctx)
    choices = [_preprocess_hellaswag(e) for e in doc["endings"]]
    correct = choices[int(doc["label"])]
    return query, " " + correct


def format_arc_easy(doc):
    """Match arc_easy.yaml: Question: {{question}}\nAnswer:"""
    prompt = f"Question: {doc['question']}\nAnswer:"
    idx = doc["choices"]["label"].index(doc["answerKey"])
    answer = doc["choices"]["text"][idx]
    return prompt, " " + answer


def format_xnli(doc, lang="en"):
    """Match xnli YAML: empty prompt, full sentence as completion."""
    kw = XNLI_KEYWORDS[lang]
    label_words = [kw["yes"], kw["also"], kw["no"]]
    correct = label_words[doc["label"]]
    sentence = f"{doc['premise']}, {kw['q']}? {correct}, {doc['hypothesis']}"
    return "", " " + sentence


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_train_data(task_name, tokenizer, max_length=256):
    """Load and format training data. Returns a GenerativeDataset."""
    prompts, completions = [], []

    if task_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="train")
        for doc in ds:
            p, c = format_hellaswag(doc)
            prompts.append(p)
            completions.append(c)

    elif task_name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        for doc in ds:
            p, c = format_arc_easy(doc)
            prompts.append(p)
            completions.append(c)

    elif task_name == "xnli":
        ds = load_dataset("facebook/xnli", "en", split="train")
        for doc in ds:
            p, c = format_xnli(doc, lang="en")
            prompts.append(p)
            completions.append(c)

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return GenerativeDataset(prompts, completions, tokenizer, max_length)


def get_train_loader(task_name, tokenizer, num_workers=2):
    """Build training DataLoader."""
    cfg = TASK_CONFIGS[task_name]
    ds = load_train_data(task_name, tokenizer, cfg["max_length"])
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                      num_workers=num_workers, pin_memory=True)
