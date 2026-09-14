"""Snapshot result-only files without touching jobs, checkpoints, or GPU state."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

RUNS = (
    "ccm_pilot_seed17_20260913_a01",
    "ccm_stage1_seed17_20260913_a01",
    "ccm_stage2_seed17_20260913_a01",
    "ccm_pilot_seed29_20260913_a01",
)
PART_BYTES = 20 * 1024**2
SUFFIXES = {".json", ".jsonl", ".log", ".tsv"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(base, out):
    base = base.resolve(strict=True)
    required = []
    for name in RUNS:
        root = base/name
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Missing/linked run: {name}")
        complete = json.loads((root/"complete.json").read_text())
        if complete.get("success") is not True:
            raise ValueError(f"Incomplete run: {name}")
        required.append(root/"complete.json")
    for stage, arms in (
        (1, ("base", "contextual", "isolated", "shuffled", "shallow", "delta", "grad")),
        (2, ("base", "contextual", "isolated", "shuffled", "grad")),
    ):
        root = base/f"ccm_stage{stage}_seed17_20260913_a01"
        for arm in arms:
            required += [root/"eval"/arm/"metrics.json", root/"eval"/arm/"segments.jsonl",
                         root/f"validated_eval_{arm}.json"]
        for right in (("isolated", "shuffled") if stage == 1 else ("isolated", "shuffled", "base", "grad")):
            required.append(root/"reports"/f"contextual_vs_{right}.json")
    required += [base/RUNS[0]/"validated_common.json", base/RUNS[3]/"validated_common.json",
                 base/RUNS[2]/"validated_panel.json"]
    for p in required:
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"Missing required result: {p}")
    selected = []
    for name in RUNS:
        for p in (base/name).rglob("*"):
            if p.is_symlink():
                raise ValueError(f"Refuse linked source: {p}")
            if p.is_file() and p.suffix in SUFFIXES and not p.name.endswith(".next.json"):
                selected.append(p)
        p = base/(name+".handoff.log")
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"Missing/linked handoff: {p}")
        selected.append(p)
    # Fresh export only; never overwrite or clean a previous export/run.
    out.mkdir(parents=False, exist_ok=False)
    archive = out/"results.tar.gz"
    entries = []
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for p in sorted(selected):
            # Capture each file once, bounded to its initial size for live logs.
            size = p.stat().st_size
            if size > 256*1024**2:
                raise ValueError(f"Unexpectedly large result text: {p}")
            with p.open("rb") as f:
                data = f.read(size)
            if len(data) != size:
                raise ValueError(f"Source shrank during snapshot: {p}")
            rel = p.relative_to(base).as_posix()
            info = tarfile.TarInfo(rel)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
            entries.append(dict(path=rel, bytes=len(data), sha256=sha(data)))
    parts, full = [], hashlib.sha256()
    with archive.open("rb") as f:
        index = 0
        while data := f.read(PART_BYTES):
            part = out/f"results.tar.gz.part{index:03d}"
            with part.open("xb") as dest:
                dest.write(data)
            full.update(data)
            parts.append(dict(path=part.name, bytes=len(data), sha256=sha(data)))
            index += 1
    manifest = dict(success=True, created_utc=datetime.now(timezone.utc).isoformat(),
                    runs=list(RUNS), files=entries, parts=parts, archive_sha256=full.hexdigest(),
                    note="JSON/JSONL/log/TSV only; no model/optimizer/table binaries. Live logs are per-file snapshots.")
    with (out/"manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(dict(success=True, files=len(entries), parts=parts,
                          archive_sha256=full.hexdigest())), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack(args.base, args.output)
