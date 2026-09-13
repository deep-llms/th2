"""Offline pilot command line. Nothing in this module submits remote jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import transformers
from .contracts import (require, read_json, write_json, digest_json, file_hash,
                        load_budget, PILOT, ARMS, VERSION)


def code_hash():
    return digest_json({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def asset_identity(path, metadata):
    """Metadata is supplied from a verified local controller/dev acquisition."""
    d = read_json(metadata)
    require(d.get("repo_id") == "Qwen/Qwen3-0.6B-Base", "Official Base tokenizer/config required")
    rev = d.get("revision", "")
    require(len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), "Pin Base commit SHA")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        require(name in d["files"] and file_hash(Path(path)/name) == d["files"][name], f"Base asset mismatch: {name}")
    return d


def prepare(args):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    from .data import verify_sources, source_id, build_manifest
    require(transformers.__version__ == "5.9.0", "Validate a new Transformers version before changing the pin")
    tokenizer_info = asset_identity(args.tokenizer_path, args.tokenizer_manifest)
    sources = verify_sources(args.raw_dir, args.source_manifest, args.dataset_revision)
    budget = load_budget(read_json(args.budget) if args.budget else PILOT.to_dict(), args.engineering)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    def docs():
        for name, path in sources:
            row = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
                for text in batch.column(0).to_pylist():
                    yield source_id(args.dataset_revision, name, row), text
                    row += 1
    provenance = dict(dataset="uonlp/CulturaX", language="en", revision=args.dataset_revision,
                      raw_manifest_hash=file_hash(args.source_manifest), tokenizer=tokenizer_info,
                      transformers=transformers.__version__, source_code_hash=code_hash())
    return build_manifest(docs(), tokenizer, args.output, budget, provenance, args.engineering)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare")
    for name in ("raw-dir", "source-manifest", "dataset-revision", "tokenizer-path", "tokenizer-manifest", "output"):
        s.add_argument("--"+name, required=True)
    s.add_argument("--budget")
    s.add_argument("--engineering", action="store_true")
    s = sub.add_parser("validate-data")
    s.add_argument("--data", required=True)
    for command in ("vocabulary", "coverage", "compile", "train", "evaluate"):
        s = sub.add_parser(command)
        s.add_argument("--data", required=True)
        s.add_argument("--output", required=True)
        if command != "vocabulary":
            s.add_argument("--vocabulary", required=command != "train")
        if command in ("train", "compile", "evaluate"):
            s.add_argument("--checkpoint", required=command != "train")
            s.add_argument("--device", default="cuda")
            s.add_argument("--microbatch-segments", type=int, default=4)
        if command in ("train", "evaluate"):
            s.add_argument("--loss-chunk", type=int, default=128)
        if command == "compile":
            s.add_argument("--isolated-batch", type=int, default=256)
        if command == "train":
            s.add_argument("--phase", choices=("common", "stage1", "stage2"), required=True)
            s.add_argument("--arm", choices=ARMS, required=True)
            s.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
            s.add_argument("--model-config")
            s.add_argument("--table")
            s.add_argument("--coverage")
            s.add_argument("--delta-decision")
            s.add_argument("--activation-checkpointing", action="store_true")
            s.add_argument("--engineering", action="store_true")
            s.add_argument("--online", action="store_true")
            s.add_argument("--save-every", type=int, default=1000)
            s.add_argument("--log-every", type=int, default=10)
        if command == "evaluate":
            s.add_argument("--role", choices=("dev", "val"), default="dev")
            s.add_argument("--diagnostic-table")
            s.add_argument("--final-evaluation", action="store_true")
            s.add_argument("--final-lock")
    s = sub.add_parser("compare")
    s.add_argument("--left", nargs="+", required=True)
    s.add_argument("--right", nargs="+", required=True)
    s.add_argument("--population", choices=("overall", "hit", "miss", "eligible_miss"), default="overall")
    s.add_argument("--cluster", choices=("doc_id", "content_hash"), required=True)
    s.add_argument("--cross-seed-coupling", choices=("shared", "independent"), required=True)
    s.add_argument("--replicates", type=int, default=10000)
    s.add_argument("--final-report", action="store_true", help="Report locked D_val results; never used by development decisions")
    s.add_argument("--output", required=True)
    s = sub.add_parser("delta-decision")
    for name in ("delta", "contextual", "shuffled", "output"):
        s.add_argument("--"+name, required=True)
    s.add_argument("--cluster", choices=("doc_id", "content_hash"), required=True)
    s.add_argument("--replication-policy", choices=("seed17_then_all",), required=True)
    s = sub.add_parser("lock-final")
    s.add_argument("--checkpoints", nargs="+", required=True)
    s.add_argument("--include-delta", action="store_true")
    s.add_argument("--confirm-choices-locked", action="store_true")
    s.add_argument("--engineering", action="store_true")
    s.add_argument("--cluster", choices=("doc_id", "content_hash"), required=True)
    s.add_argument("--cross-seed-coupling", choices=("shared", "independent"), required=True)
    s.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(transformers.__version__ == "5.9.0", "This implementation is tested against Transformers 5.9.0")
    args.source_code_hash = code_hash()
    for name in ("microbatch_segments", "loss_chunk", "isolated_batch", "save_every", "log_every"):
        if hasattr(args, name):
            require(getattr(args, name) > 0, f"Positive {name} required")
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "lock-final":
        from .decisions import lock_final
        result = lock_final(args)
    elif args.command in ("compare", "delta-decision"):
        from .statistics import compare, delta_decision
        result = compare(args) if args.command == "compare" else delta_decision(args)
    else:
        from .data import Corpus, validate_splits
        from .keys import Vocabulary, count_vocabulary
        corpus = Corpus(args.data)
        vocab = Vocabulary.load(args.vocabulary) if getattr(args, "vocabulary", None) else None
        if args.command == "validate-data":
            result = dict(documents=validate_splits(corpus), manifest_hash=corpus.meta["manifest_hash"])
            for role in corpus.meta["quotas"]:
                require(sum(len(r["tokens"]) for r in corpus.segments(role)) == corpus.meta["quotas"][role], "Wrong role budget")
        elif args.command == "vocabulary":
            vocab = count_vocabulary(corpus.segments("compile"), corpus.meta["special_ids"], corpus.budget.slots,
                                     str(args.output)+".counts.sqlite", dict(corpus_hash=corpus.meta["manifest_hash"],
                                     compile_segment_hash=corpus.meta["files"]["compile.segments.jsonl"],
                                     tokenizer=corpus.meta["provenance"]["tokenizer"]))
            vocab.save(args.output)
            result = dict(vocabulary_hash=vocab.hash, slots=len(vocab.keys))
        elif args.command == "coverage":
            from .compiler import coverage
            result = coverage(corpus, vocab, args.output)
        elif args.command == "compile":
            from .compiler import compile_tables
            result = compile_tables(args, corpus, vocab)
        elif args.command == "train":
            from .runtime import train
            require(args.save_every > 0 and args.log_every > 0, "Positive log/save intervals required")
            if args.phase == "common":
                require(args.model_config is not None, "Common training requires a local Base config")
                if not args.engineering:
                    expected = corpus.meta["provenance"]["tokenizer"]["files"]["config.json"]
                    require(file_hash(Path(args.model_config)/"config.json") == expected, "Training config differs from pinned Base asset")
            result = train(args, corpus, vocab)
        else:
            from .evaluation import evaluate
            result = evaluate(args, corpus, vocab)
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
