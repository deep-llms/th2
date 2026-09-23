"""Validate the complete train/validation stream before any scientific training."""
import hashlib

import numpy as np
import torch

from . import protocol
from .data import PreparedContexts, validate_matched_data
from .probe import verify_slice


def context_fingerprint(data):
    """Hash ordered model inputs with canonical dtypes and bounded scratch memory."""
    digest = hashlib.sha256()
    for name, array, dtype in (("input_ids", data.ids, "<i8"),
                               ("valid", data.valid, "|b1"),
                               ("position_ids", data.positions, "<i8"),
                               ("segments", data.segments, "<i8")):
        digest.update(name.encode() + b"\0")
        if array is None:
            digest.update(b"absent\0")
            continue
        digest.update(str(array.shape).encode() + b"\0")
        for start in range(0, len(array), 128):
            values = np.ascontiguousarray(array[start:start + 128], dtype=dtype)
            digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def describe_contexts(data):
    # Count targets with the same rule as Context.targets, including packing.
    targets = data.valid[:, :-1] & data.valid[:, 1:]
    if data.segments is not None:
        targets &= data.segments[:, :-1] == data.segments[:, 1:]
    return {"path": str(data.path), "metadata": data.metadata,
            "sequences": len(data), "input_tokens": int(data.valid.sum()),
            "target_tokens": int(targets.sum()), "context_sha256": context_fingerprint(data)}


def check_inputs(backbone, train_path, val_path, *, load_data=None):
    """No test-path parameter: checking inputs cannot unlock held-out data."""
    if backbone.model.dtype != torch.bfloat16:
        raise ValueError("PCC experiments require a bf16 frozen backbone")
    if backbone.model.training or any(p.requires_grad for p in backbone.model.parameters()):
        raise ValueError("PCC input checks require a frozen backbone in eval mode")
    load_data = load_data or PreparedContexts
    budgets = {"train": protocol.FULL_UPDATES * protocol.TOKENS_PER_UPDATE,
               "dev": protocol.PROBE_EVAL_TOKENS,
               "screen_train": protocol.SCREEN_UPDATES * protocol.TOKENS_PER_UPDATE,
               "screen_dev": protocol.SCREEN_DEV_TOKENS}
    print("INPUT CHECK full training stream", flush=True)
    train = load_data(train_path, "train", budgets["train"])
    print("INPUT CHECK full validation stream", flush=True)
    dev = load_data(val_path, "dev", budgets["dev"])
    validate_matched_data(train, dev)
    if max(train.ids.max(), dev.ids.max()) >= backbone.model.config.vocab_size:
        raise ValueError("Input IDs exceed the pinned model vocabulary")
    screen_train = load_data(train_path, "train", budgets["screen_train"])
    screen_dev = load_data(val_path, "dev", budgets["screen_dev"])
    verify_slice(screen_train, train)
    verify_slice(screen_dev, dev)
    streams = {"train": train, "dev": dev, "screen_train": screen_train, "screen_dev": screen_dev}
    report = {name: describe_contexts(data) for name, data in streams.items()}
    for name, budget in budgets.items():
        if report[name]["input_tokens"] != budget:
            raise ValueError(f"{name}: loader returned a different input-token budget")
        if report[name]["sequences"] < 2 or report[name]["target_tokens"] <= 0:
            raise ValueError(f"{name}: insufficient contexts for paired evaluation/training")
    return {"status": "ok", "streams": report, "screen_prefixes_verified": True,
            "test_data_inspected": False, "sampling_completion_verified": False,
            "scientific_training_performed": False}
