"""Offline PCC stages and sequential pipeline. Never submits remote jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import socket


def configure_offline(physical_gpu=None):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if physical_gpu is None else str(physical_gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight", help="Correctness checks on synthetic tokens")
    preflight.add_argument("--tiny", action="store_true", help="CPU-only random tiny Qwen; not scientific evidence")
    screening = sub.add_parser("screen", help="Train deep/shallow adapters for four layer pairs (128 updates each)")
    screening.add_argument("--train-data", required=True, type=Path)
    screening.add_argument("--val-data", required=True, type=Path)
    screening.add_argument("--microbatch", type=int, default=1)
    full = sub.add_parser("probe", help="Train teacher and PCC/control adapters (610 updates each) after an eligible screen")
    full.add_argument("--screen-dir", required=True, type=Path)
    full.add_argument("--train-data", required=True, type=Path)
    full.add_argument("--val-data", required=True, type=Path)
    full.add_argument("--test-data", type=Path)
    full.add_argument("--microbatch", type=int, default=1)
    sequential = sub.add_parser("pipeline", help="Train adapters on the frozen pretrained model: screen, then eligible full probe",
        description="Run real adapter optimization and evaluation using the fixed train/validation data. "
                    "The pretrained backbone stays frozen. Use --check-only to stop before experiment training.")
    sequential.add_argument("--config", required=True, type=Path)
    sequential.add_argument("--output", type=Path, help="Fresh pipeline run directory (required except for --dry-run)")
    sequential.add_argument("--dry-run", action="store_true", help="Print plan without model/data loading or GPU inspection")
    sequential.add_argument("--check-only", action="store_true", help="Check the local model and full train/validation inputs, then stop before experiment training")
    sequential.add_argument("--physical-gpu", type=int, help="Explicit local GPU allocation; omission selects CPU")
    for command in (preflight, screening, full):
        command.add_argument("--model-path", type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--physical-gpu", type=int, help="Explicit local GPU allocation; omission selects CPU")
    args = parser.parse_args()
    if args.physical_gpu is not None and args.physical_gpu < 0:
        parser.error("--physical-gpu must be nonnegative")
    config = None
    if args.mode == "pipeline":
        from .plan import load_pipeline_config, pipeline_plan
        try:
            config = load_pipeline_config(args.config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.dry_run:
            print(json.dumps({**pipeline_plan(config, check_only=args.check_only), "physical_gpu": args.physical_gpu}, indent=2))
            return
        if args.output is None:
            parser.error("pipeline requires --output unless --dry-run is used")
        args.model_path = Path(config["model_path"])
    if getattr(args, "tiny", False) and (args.model_path or args.physical_gpu is not None):
        parser.error("--tiny is CPU-only and cannot take a pretrained path")
    if not getattr(args, "tiny", False) and args.model_path is None:
        parser.error("Supply the exact local --model-path, or use preflight --tiny")
    if args.output.exists():
        parser.error("Output already exists; use a fresh report/run path")
    # Set before importing torch/transformers. There is no Hub fallback.
    configure_offline(args.physical_gpu)
    if args.physical_gpu is not None:
        from scripts.gpu_status import require_free
        require_free([args.physical_gpu])
    import torch
    import transformers
    from .diagnostics import tiny_backbone, run_preflight
    from .model import load_local
    from .protocol import MODEL_ID, REVISION
    from .screen import screen, write_json
    device = torch.device("cpu" if args.physical_gpu is None else "cuda:0")
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    tiny = getattr(args, "tiny", False)
    backbone, tokenizer = (tiny_backbone(), None) if tiny else load_local(args.model_path, device)
    from .data import SampledDataLoader
    load_data = SampledDataLoader(tokenizer, args.output / "data-cache")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    provenance = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "hostname": socket.gethostname(),
                  "model_id": "tiny-random-qwen3" if tiny else MODEL_ID,
                  "model_revision": None if tiny else REVISION,
                  "local_model_path": str(args.model_path.resolve()) if args.model_path else None,
                  "revision_evidence": "operator-labeled local snapshot directory" if not tiny else None,
                  "model_config": backbone.model.config.to_dict(),
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "python": sys.version, "device": str(device),
                  "code_commit": commit.stdout.strip() if commit.returncode == 0 else None}
    if args.mode == "preflight":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            report = run_preflight(backbone)
        except BaseException as error:
            write_json(args.output, {"status": "failed", "error": str(error), "provenance": provenance})
            raise
        report["provenance"] = provenance
        report["synthetic_model"] = tiny
        write_json(args.output, report)
    elif args.mode == "screen":
        report = screen(backbone, args.train_data, args.val_data, args.output,
                        microbatch=args.microbatch, provenance=provenance, load_data=load_data)
    elif args.mode == "probe":
        from .probe import probe
        report = probe(backbone, args.screen_dir, args.train_data, args.val_data,
                       args.test_data, args.output, microbatch=args.microbatch, provenance=provenance, load_data=load_data)
    else:
        from .pipeline import pipeline
        report = pipeline(backbone, config, args.output, provenance=provenance, load_data=load_data,
                          check_only=args.check_only)
    print(json.dumps({"output": str(args.output), "status": report.get("status", report.get("decision"))}))


if __name__ == "__main__":
    main()
