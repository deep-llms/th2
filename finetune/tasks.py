"""Task definitions: data loading, preprocessing, eval splits.

Each task returns train/val/test DataLoaders via get_dataloaders().
Text-pair tasks (XNLI, PAWS-X) concatenate with the tokenizer's EOS.
HellaSwag returns (B, 4, L) tensors for multiple-choice scoring.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


# -----------------------------------------------------------------------
# Collation helpers
# -----------------------------------------------------------------------

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class MultipleChoiceDataset(Dataset):
    def __init__(self, contexts, endings_list, labels, tokenizer, max_length):
        self.items = []
        for ctx, endings, label in zip(contexts, endings_list, labels):
            texts = [ctx + " " + e for e in endings]
            enc = tokenizer(texts, max_length=max_length, padding="max_length",
                            truncation=True, return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"],       # (C, L)
                "attention_mask": enc["attention_mask"],
                "labels": torch.tensor(label, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# -----------------------------------------------------------------------
# Task configs
# -----------------------------------------------------------------------

TASK_CONFIGS = {
    "ag_news": {
        "num_classes": 4,
        "max_length": 128,
        "type": "classification",
        "full_epochs": 3,
        "probe_epochs": 10,
        "full_lr": 2e-5,
        "probe_lr": 1e-3,
        "full_batch": 32,
        "probe_batch": 64,
    },
    "sst2": {
        "num_classes": 2,
        "max_length": 128,
        "type": "classification",
        "full_epochs": 3,
        "probe_epochs": 10,
        "full_lr": 2e-5,
        "probe_lr": 1e-3,
        "full_batch": 32,
        "probe_batch": 64,
    },
    "xnli": {
        "num_classes": 3,
        "max_length": 256,
        "type": "classification",
        "full_epochs": 5,
        "probe_epochs": 20,
        "full_lr": 2e-5,
        "probe_lr": 1e-3,
        "full_batch": 32,
        "probe_batch": 64,
        "xling_langs": ["en", "vi", "zh", "de", "ru", "ar"],
    },
    "paws_x": {
        "num_classes": 2,
        "max_length": 256,
        "type": "classification",
        "full_epochs": 5,
        "probe_epochs": 20,
        "full_lr": 2e-5,
        "probe_lr": 1e-3,
        "full_batch": 32,
        "probe_batch": 64,
        "xling_langs": ["en", "de", "zh"],
    },
    "hellaswag": {
        "num_classes": 4,
        "max_length": 256,
        "type": "multiple_choice",
        "full_epochs": 3,
        "probe_epochs": 10,
        "full_lr": 2e-5,
        "probe_lr": 1e-3,
        "full_batch": 8,
        "probe_batch": 16,
    },
}

XNLI_LANGS = ["en", "vi", "zh", "de", "ru", "ar"]
PAWSX_LANGS = ["en", "de", "zh"]


def load_task_data(task_name, tokenizer, split="train", lang=None):
    """Load and preprocess a task split, returning a Dataset."""
    cfg = TASK_CONFIGS[task_name]
    sep = " " + tokenizer.eos_token + " "

    if task_name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news", split=split)
        return ClassificationDataset(
            ds["text"], ds["label"], tokenizer, cfg["max_length"])

    if task_name == "sst2":
        s = "validation" if split == "test" else split
        ds = load_dataset("stanfordnlp/sst2", split=s)
        return ClassificationDataset(
            ds["sentence"], ds["label"], tokenizer, cfg["max_length"])

    if task_name == "xnli":
        if split == "train":
            ds = load_dataset("nyu-mll/multi_nli", split="train")
            texts = [p + sep + h for p, h in
                     zip(ds["premise"], ds["hypothesis"])]
            return ClassificationDataset(
                texts, ds["label"], tokenizer, cfg["max_length"])
        lang = lang or "en"
        s = "validation" if split == "val" else "test"
        ds = load_dataset("facebook/xnli", lang, split=s)
        texts = [p + sep + h for p, h in
                 zip(ds["premise"], ds["hypothesis"])]
        return ClassificationDataset(
            texts, ds["label"], tokenizer, cfg["max_length"])

    if task_name == "paws_x":
        lang = lang or "en"
        ds = load_dataset("google-research-datasets/paws-x", lang,
                          split=split)
        texts = [s1 + sep + s2 for s1, s2 in
                 zip(ds["sentence1"], ds["sentence2"])]
        return ClassificationDataset(
            texts, ds["label"], tokenizer, cfg["max_length"])

    if task_name == "hellaswag":
        s = "validation" if split == "test" else split
        ds = load_dataset("Rowan/hellaswag", split=s)
        return MultipleChoiceDataset(
            ds["ctx"], ds["endings"],
            [int(l) for l in ds["label"]],
            tokenizer, cfg["max_length"])

    raise ValueError(f"Unknown task: {task_name}")


def get_dataloaders(task_name, tokenizer, mode="full", num_workers=4):
    """Build train/val/test DataLoaders for a task.

    Returns (train_loader, val_loader, test_splits) where test_splits is a
    dict of {split_name: DataLoader} — for cross-lingual tasks this includes
    per-language test sets.
    """
    cfg = TASK_CONFIGS[task_name]
    bs = cfg[f"{mode}_batch"]

    train_ds = load_task_data(task_name, tokenizer, "train")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_splits = {}
    if task_name == "xnli":
        val_ds = load_task_data(task_name, tokenizer, "val", lang="en")
        val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        for lang in XNLI_LANGS:
            ds = load_task_data(task_name, tokenizer, "test", lang=lang)
            test_splits[f"xnli_{lang}"] = DataLoader(
                ds, batch_size=bs * 2, shuffle=False,
                num_workers=num_workers, pin_memory=True)
    elif task_name == "paws_x":
        val_ds = load_task_data(task_name, tokenizer, "validation", lang="en")
        val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        for lang in PAWSX_LANGS:
            ds = load_task_data(task_name, tokenizer, "test", lang=lang)
            test_splits[f"paws_{lang}"] = DataLoader(
                ds, batch_size=bs * 2, shuffle=False,
                num_workers=num_workers, pin_memory=True)
    elif task_name == "sst2":
        # SST-2 test has no labels; use validation for both val and test
        val_ds = load_task_data(task_name, tokenizer, "test")
        val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        test_splits["sst2"] = val_loader
    elif task_name == "ag_news":
        # AG News has no val split; hold out 5% of train for validation
        full_train = load_task_data(task_name, tokenizer, "train")
        n_val = len(full_train) // 20
        val_indices = list(range(n_val))
        train_indices = list(range(n_val, len(full_train)))
        from torch.utils.data import Subset
        train_loader = DataLoader(
            Subset(full_train, train_indices), batch_size=bs, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            Subset(full_train, val_indices), batch_size=bs * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        test_ds = load_task_data(task_name, tokenizer, "test")
        test_splits["ag_news"] = DataLoader(
            test_ds, batch_size=bs * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        # HellaSwag: test has no labels; use validation for both
        test_ds = load_task_data(task_name, tokenizer, "test")
        val_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        test_splits[task_name] = val_loader

    return train_loader, val_loader, test_splits
