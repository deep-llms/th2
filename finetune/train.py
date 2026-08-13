"""Train one task/arm/mode/seed. Called by run_all.py or standalone.

Handles: model loading (baseline + compositional via eval/eval_checkpoint),
classifier wrapping, full-finetune vs probe (frozen backbone), training loop,
best-epoch selection on validation, final test evaluation, JSON output.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from finetune.models import CausalLMClassifier, CausalLMMultipleChoice
from finetune.tasks import TASK_CONFIGS, get_dataloaders


def load_base_model(checkpoint, device, dtype=torch.bfloat16):
    """Load baseline or compositional checkpoint."""
    from eval.eval_checkpoint import load_model
    return load_model(checkpoint, device, dtype=dtype)


@torch.no_grad()
def evaluate(wrapper, loader, device):
    wrapper.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = wrapper(ids, mask)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one(checkpoint, task_name, mode, seed, device, output_dir,
              tokenizer_name="Qwen/Qwen3-0.6B", num_workers=2):
    """Train one run. Returns result dict and saves JSON."""
    torch.manual_seed(seed)
    cfg = TASK_CONFIGS[task_name]
    is_mc = cfg["type"] == "multiple_choice"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {checkpoint}")
    base_model = load_base_model(checkpoint, device)
    hidden_size = base_model.config.hidden_size

    if is_mc:
        wrapper = CausalLMMultipleChoice(base_model, hidden_size)
    else:
        wrapper = CausalLMClassifier(base_model, cfg["num_classes"],
                                     hidden_size)
    wrapper.to(device)

    if mode == "probe":
        for param in base_model.parameters():
            param.requires_grad = False
        head_params = (list(wrapper.classifier.parameters()) if not is_mc
                       else list(wrapper.score_head.parameters()))
        optimizer = AdamW(head_params, lr=cfg["probe_lr"], weight_decay=0.0)
    else:
        optimizer = AdamW(wrapper.parameters(), lr=cfg["full_lr"],
                          weight_decay=0.01)

    epochs = cfg[f"{mode}_epochs"]
    print(f"  Loading data: {task_name} (mode={mode}, epochs={epochs})")
    train_loader, val_loader, test_splits = get_dataloaders(
        task_name, tokenizer, mode, num_workers=num_workers)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        wrapper.train()
        if mode == "probe":
            base_model.eval()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapper(ids, mask)
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            if mode == "full":
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        val_acc = evaluate(wrapper, val_loader, device)
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"    epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"val_acc={val_acc:.4f} ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            if mode == "probe":
                head = (wrapper.classifier if not is_mc
                        else wrapper.score_head)
                best_state = {k: v.cpu().clone()
                              for k, v in head.state_dict().items()}
            else:
                best_state = {k: v.cpu().clone()
                              for k, v in wrapper.state_dict().items()}

    # Restore best state
    if mode == "probe":
        head = wrapper.classifier if not is_mc else wrapper.score_head
        head.load_state_dict(best_state)
    else:
        wrapper.load_state_dict(best_state)

    test_results = {}
    for split_name, loader in test_splits.items():
        test_results[split_name] = evaluate(wrapper, loader, device)
        print(f"    test {split_name}: {test_results[split_name]:.4f}")

    result = {
        "checkpoint": checkpoint,
        "task": task_name,
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_results": test_results,
        "epochs": epochs,
        "lr": cfg[f"{mode}_lr"],
    }

    os.makedirs(output_dir, exist_ok=True)
    arm = os.path.basename(os.path.dirname(os.path.normpath(checkpoint)))
    out_path = os.path.join(output_dir,
                            f"{task_name}_{mode}_{arm}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    saved: {out_path}")

    del wrapper, base_model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Fine-tune one task/arm/seed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    parser.add_argument("--mode", required=True, choices=["full", "probe"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="temp/finetune")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    train_one(args.checkpoint, args.task, args.mode, args.seed,
              args.device, args.output_dir, args.tokenizer_name)


if __name__ == "__main__":
    main()
