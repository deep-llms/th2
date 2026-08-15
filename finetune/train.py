"""Fine-tune one task/arm/seed with the generative approach.

The model is fine-tuned with standard causal LM loss on (prompt +
completion) sequences, where loss is computed only on completion tokens
(prompt masked with labels=-100). No classification head — the model's
own lm_head is used.

After fine-tuning, evaluation uses lm-evaluation-harness (the same
zero-shot log-likelihood scoring as our existing eval/benchmarks.py),
run in-memory on the fine-tuned model. This guarantees the eval format
exactly matches the training format.

Right padding (text first, padding after) throughout.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from finetune.tasks import TASK_CONFIGS, get_train_loader


def load_base_model(checkpoint, device, dtype=torch.bfloat16):
    """Load baseline or compositional checkpoint."""
    from eval.eval_checkpoint import load_model
    return load_model(checkpoint, device, dtype=dtype)


def evaluate_with_lm_eval(model, tokenizer, eval_tasks, device,
                          batch_size=16):
    """Run lm-evaluation-harness on the in-memory model."""
    from eval.benchmarks import eval_benchmarks, patch_lm_eval_dataset_paths
    from lm_eval.models.huggingface import HFLM
    import lm_eval

    patch_lm_eval_dataset_paths()
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              batch_size=batch_size, device=str(device))
    results = lm_eval.simple_evaluate(model=lm, tasks=eval_tasks,
                                      num_fewshot=0)

    out = {}
    for task_name, task_results in results["results"].items():
        acc = task_results.get("acc,none", task_results.get("acc"))
        acc_norm = task_results.get("acc_norm,none",
                                    task_results.get("acc_norm"))
        out[task_name] = {"acc": acc}
        if acc_norm is not None:
            out[task_name]["acc_norm"] = acc_norm
    return out


def train_one(checkpoint, task_name, seed, device, output_dir,
              tokenizer_name="Qwen/Qwen3-0.6B", num_workers=2):
    """Fine-tune one run. Returns result dict and saves JSON."""
    torch.manual_seed(seed)
    cfg = TASK_CONFIGS[task_name]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {checkpoint}")
    model = load_base_model(checkpoint, device)

    print(f"  Loading data: {task_name} (epochs={cfg['epochs']})")
    train_loader = get_train_loader(task_name, tokenizer, num_workers)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    model.train()
    t_start = time.time()

    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"    epoch {epoch+1}/{cfg['epochs']}: loss={avg_loss:.4f} "
              f"({elapsed:.0f}s)")

    train_time = time.time() - t_start
    print(f"  Training done in {train_time:.0f}s")

    # Evaluate with lm-eval-harness (in-memory, same model)
    eval_results = {}
    try:
        print(f"  Evaluating with lm-eval-harness: {cfg['eval_tasks']}")
        model.eval()
        eval_results = evaluate_with_lm_eval(
            model, tokenizer, cfg["eval_tasks"], device)
        for task, metrics in eval_results.items():
            acc = metrics.get("acc", 0)
            acc_norm = metrics.get("acc_norm")
            extra = f" acc_norm={acc_norm:.4f}" if acc_norm is not None else ""
            print(f"    {task}: acc={acc:.4f}{extra}")
    except ImportError:
        print("  lm_eval not available — skipping eval (run separately)")
    except Exception as e:
        print(f"  eval failed: {e}")

    # Save result + fine-tuned model state
    os.makedirs(output_dir, exist_ok=True)
    arm = os.environ.get("FINETUNE_ARM_NAME",
                         os.path.basename(os.path.dirname(
                             os.path.normpath(checkpoint))))

    model_dir = os.path.join(output_dir, "models",
                             f"{task_name}_{arm}_seed{seed}")
    os.makedirs(model_dir, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               os.path.join(model_dir, "model_state.pt"))
    print(f"    model saved: {model_dir}")

    result = {
        "checkpoint": checkpoint,
        "task": task_name,
        "seed": seed,
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "train_time_s": train_time,
        "eval_results": eval_results,
        "model_path": model_dir,
    }

    out_path = os.path.join(output_dir, f"{task_name}_{arm}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    saved: {out_path}")

    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune one task/arm/seed (generative)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="temp/finetune")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    train_one(args.checkpoint, args.task, args.seed,
              args.device, args.output_dir, args.tokenizer_name)


if __name__ == "__main__":
    main()
