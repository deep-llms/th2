"""Offline pilot command line. Nothing in this module submits remote jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import transformers
from .contracts import (require, read_json, write_json, digest_json, file_hash,
                        load_budget, PILOT, ARMS, VERSION)
from .studies import STUDIES, PILOT_STUDY, SCALEUP, SCALEUP_BUDGET, open_corpus, require_vocabulary


def code_hash():
    return digest_json({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def local_git_commit():
    # Local repository metadata only; never contacts a remote. Source-byte
    # hash remains authoritative for non-Git or dirty development directories.
    result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                            text=True, capture_output=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else None


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
                      transformers=transformers.__version__, source_code_hash=code_hash(),
                      source_git_commit=getattr(args, "source_git_commit", None))
    if args.command == "prepare-scaleup":
        from .scaleup_data import prepare_scaleup
        from .data import Corpus
        from .keys import Vocabulary
        require(args.budget is None or args.engineering, "Scientific scale-up budget is fixed")
        return prepare_scaleup(docs, tokenizer, Corpus(args.historical_data), Vocabulary.load(args.vocabulary),
                               args.output, provenance, budget if args.budget else SCALEUP_BUDGET, args.engineering)
    return build_manifest(docs(), tokenizer, args.output, budget, provenance, args.engineering)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "prepare-scaleup"):
        s = sub.add_parser(command)
        for name in ("raw-dir", "source-manifest", "dataset-revision", "tokenizer-path", "tokenizer-manifest", "output"):
            s.add_argument("--"+name, required=True)
        s.add_argument("--budget")
        s.add_argument("--engineering", action="store_true")
        if command == "prepare-scaleup":
            s.add_argument("--historical-data", required=True)
            s.add_argument("--vocabulary", required=True)
    s = sub.add_parser("validate-data")
    s.add_argument("--data", required=True)
    s = sub.add_parser("validate-run")
    for name in ("data", "run", "output", "phase", "arm"):
        s.add_argument("--"+name, required=True)
    s.add_argument("--seed", type=int, choices=(17,29,43), default=17)
    s = sub.add_parser("scaleup-jobs")
    s.add_argument("--queue", choices=("common", "panel", "shallow12"), required=True)
    for name in ("data", "vocabulary", "output"):
        s.add_argument("--"+name, required=True)
    for name in ("checkpoint", "tables", "model-config", "contextual-eval", "lr-failure-report"):
        s.add_argument("--"+name)
    s.add_argument("--gpus", type=int, nargs="+", required=True)
    s.add_argument("--seed", type=int, choices=(17,29,43), default=17)
    s.add_argument("--microbatch-segments", type=int, default=8)
    # Decision 1(c) 2026-09-15: stability smoke past the 763-update warmup.
    s.add_argument("--stability-steps", type=int, default=1024)
    s.add_argument("--common-lr", type=float, choices=(3e-4,2e-4), default=3e-4)
    s.add_argument("--compiler", choices=("distributed", "reference"), default="distributed")
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
            s.add_argument("--distributed", action="store_true")
            s.add_argument("--compiler-validation")
            s.add_argument("--validate-distributed", action="store_true")
            s.add_argument("--validation-batches", type=int, default=16)
            s.add_argument("--validation-eval-batches", type=int, default=4)
            s.add_argument("--reserved-gpus", type=int, default=8)
        if command == "train":
            s.add_argument("--study", choices=STUDIES, default=PILOT_STUDY)
            s.add_argument("--common-lr", type=float, choices=(3e-4, 2e-4), default=3e-4)
            s.add_argument("--stability-steps", type=int)
            s.add_argument("--stability-report")
            s.add_argument("--lr-failure-report")
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
    s.add_argument("--replication-policy", choices=("seed17_then_all", "per_seed"), required=True)
    s = sub.add_parser("lock-final")
    s.add_argument("--study", choices=STUDIES, default=PILOT_STUDY)
    s.add_argument("--checkpoints", nargs="+", required=True)
    s.add_argument("--include-delta", action="store_true")
    s.add_argument("--confirm-choices-locked", action="store_true")
    s.add_argument("--single-seed-terminal", action="store_true",
                   help="Declare the 28L study stops at seed 17; required for a seeds=[17] scale-up lock (decision 2a, 2026-09-15)")
    s.add_argument("--engineering", action="store_true")
    s.add_argument("--cluster", choices=("doc_id", "content_hash"), required=True)
    s.add_argument("--cross-seed-coupling", choices=("shared", "independent"), required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("scaleup-decision")
    s.add_argument("--isolated-report", required=True)
    s.add_argument("--shuffled-report", required=True)
    s.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(transformers.__version__ == "5.9.0", "This implementation is tested against Transformers 5.9.0")
    args.source_code_hash = code_hash()
    args.source_git_commit = local_git_commit()
    for name in ("microbatch_segments", "loss_chunk", "isolated_batch", "save_every", "log_every",
                 "stability_steps", "validation_batches", "validation_eval_batches"):
        if hasattr(args, name):
            require(getattr(args, name) is None or getattr(args, name) > 0, f"Positive {name} required")
    if args.command in ("prepare", "prepare-scaleup"):
        result = prepare(args)
    elif args.command == "scaleup-jobs":
        from .scaleup_jobs import make_jobs
        result = make_jobs(args)
        write_json(args.output, result)
    elif args.command == "scaleup-decision":
        from .scaleup_protocol import replication_decision
        result = replication_decision(args.isolated_report, args.shuffled_report, args.output)
    elif args.command == "lock-final":
        from .decisions import lock_final
        result = lock_final(args)
    elif args.command in ("compare", "delta-decision"):
        from .statistics import compare, delta_decision
        result = compare(args) if args.command == "compare" else delta_decision(args)
    else:
        from .data import Corpus, validate_splits
        from .keys import Vocabulary, count_vocabulary
        corpus = open_corpus(args.data)
        vocab = Vocabulary.load(args.vocabulary) if getattr(args, "vocabulary", None) else None
        if args.command == "validate-data":
            if corpus.meta.get("study") == SCALEUP:
                from .scaleup_data import validate_scaleup
                result = validate_scaleup(corpus)
            else:
                result = dict(documents=validate_splits(corpus), manifest_hash=corpus.meta["manifest_hash"])
            for role in corpus.meta["quotas"]:
                require(sum(len(r["tokens"]) for r in corpus.segments(role)) == corpus.meta["quotas"][role], "Wrong role budget")
        elif args.command == "validate-run":
            from .scaleup_jobs import validate_run
            result = validate_run(args, corpus)
        elif args.command == "vocabulary":
            require(corpus.meta.get("study") != SCALEUP, "Reuse the checked historical vocabulary; do not recount/rebind it")
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
            if args.distributed or args.validate_distributed:
                from .distributed_compiler import distributed_compile
                result = distributed_compile(args, corpus, vocab)
            else:
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
