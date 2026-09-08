"""Single-process, exact scored-token PPL for a saved local capacity checkpoint."""
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import capacity_allocation  # Register the custom HF config/model before loading.
from capacity_allocation.data import load_text_data, preprocess_text, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--languages", default="en")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--preprocessing_batch_size", type=int, default=1000)
    parser.add_argument("--preprocessing_cache_dir")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() or args.batch_size <= 0:
        parser.error("Choose a fresh output and a positive batch size")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer.model_max_length = 10**30
    data = preprocess_text(load_text_data(args.data_dir, tuple(args.languages.split(","))), tokenizer,
                           block_size=args.block_size, num_proc=args.preprocessing_num_workers,
                           batch_size=args.preprocessing_batch_size, cache_dir=args.preprocessing_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True).to(args.device).eval()
    if (len(tokenizer) > model.config.vocab_size or
            model.config.eos_token_id != tokenizer.eos_token_id):
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
    nll_sum, target_count = 0., 0
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=args.batch_size, shuffle=False):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                # No Trainer num_items override: the HF model's own causal loss
                # normalizes by the correctly shifted/scored targets.
                loss = model(**batch, use_cache=False).loss.item()
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss")
            count = batch["labels"][:, 1:].ne(-100).sum().item()
            nll_sum += loss * count
            target_count += count
    if target_count != len(data) * (args.block_size - 1):
        raise AssertionError("Evaluation did not score every packed target exactly once")
    nll = nll_sum / target_count
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(success=True, split=args.split, nll=nll, ppl=math.exp(nll) if nll < 700 else None,
                  scored_targets=target_count, processed_tokens=len(data)*args.block_size,
                  precision=args.precision, device=args.device, checkpoint=str(Path(args.checkpoint).resolve()),
                  data_fingerprint=data._fingerprint, data_dir=str(Path(args.data_dir).resolve()),
                  preprocessing=dict(block_size=args.block_size, workers=args.preprocessing_num_workers,
                                     batch_size=args.preprocessing_batch_size),
                  checkpoint_files={p.name: sha256(p) for p in sorted(Path(args.checkpoint).glob("*"))
                                    if p.is_file() and (p.suffix == ".safetensors" or p.name == "config.json")})
    write_json(path, result)
    print(f"{args.split}: NLL={nll:.8f}, PPL={result['ppl']}, targets={target_count}", flush=True)


if __name__ == "__main__":
    main()
