"""Prepare the exact training/eval caches on CPU before reclaiming GPUs."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer
from capacity_allocation.data import load_text_data, preprocess_text, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--workers', type=int, default=160)
    parser.add_argument('--stop-at-step', type=int, default=10000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stop_at_step <= 0:
        parser.error('--stop-at-step must be positive')
    if Path(args.output).exists():
        parser.error('Use a fresh cache-verification report')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id != 151645 or len(tokenizer) > 151936:
        raise ValueError('Expected local fast Qwen3 tokenizer')
    tokenizer.model_max_length = 10**30
    result = dict(success=True, workers=args.workers, batch_size=1000, block_size=2048,
                  tokenizer=str(Path(args.tokenizer).resolve()), languages=['en'])
    for split in ('train', 'eval'):
        data = preprocess_text(load_text_data(Path(args.data_root)/split), tokenizer,
            block_size=2048, num_proc=args.workers, batch_size=1000, cache_dir=args.cache_dir)
        if split == 'train':
            data = data.shuffle(seed=42)
        result[split] = dict(blocks=len(data), fingerprint=data._fingerprint,
                             scored_targets=len(data)*2047, processed_tokens=len(data)*2048)
        print(split, result[split], flush=True)
    # Only the screening cutoff changes; prepare the entire English pool.
    if result['train']['blocks'] <= args.stop_at_step*512:
        raise ValueError(f'Not enough training blocks for {args.stop_at_step} steps at effective batch 512')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
