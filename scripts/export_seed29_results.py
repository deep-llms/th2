"""Offline, result-only snapshot of the completed seed29 replication."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

RUN = "ccm_replication_seed29_20260914_a01"
ARMS1 = ("base", "contextual", "isolated", "shuffled", "shallow", "delta", "grad")
ARMS2 = ("base", "contextual", "isolated", "shuffled", "grad")
def read(p):
    return json.loads(p.read_text())
def digest(b):
    return hashlib.sha256(b).hexdigest()
def require(ok, why):
    if not ok:
        raise ValueError(why)

def pack(base, out):
    base = base.resolve(strict=True)
    root = base/RUN
    require(root.is_dir() and not root.is_symlink(), "Missing/linked run")
    done = read(root/"complete.json")
    require(done.get("success") is True and done.get("event") ==
            "seed29_replication_verified_and_burns_active", "Workflow not verified complete")
    decision = read(root/"stage1/delta_decision.json")
    require(decision["decision_seed"] == 29 and decision["replication_policy"] == "per_seed"
            and type(decision["include_delta"]) is bool, "Wrong Delta decision")
    arms2 = ARMS2 + (("delta",) if decision["include_delta"] else ())
    require(done["stage2_arms"] == list(arms2), "Wrong final arm panel")
    for phase, arms in (("stage1", ARMS1), ("stage2", arms2)):
        p = root/phase
        panel = read(p/"validated_panel.json")
        require(read(p/"complete.json")["success"] and panel["success"]
                and panel["seed"] == 29 and panel["evaluated_arms"] == list(arms), "Incomplete phase")
        for arm in arms:
            e = p/"eval"/arm
            gate = read(p/f"validated_eval_{arm}.json")
            m = read(e/"metrics.json")
            require(gate["success"] and m["seed"] == 29 and m["arm"] == arm
                    and m["role"] == "dev" and not m["final_evaluation"]
                    and digest((e/"metrics.json").read_bytes()) == gate["metrics_sha256"]
                    and digest((e/"segments.jsonl").read_bytes()) ==
                        gate["segments_sha256"] == m["segments_sha256"]
                    and m["checkpoint_hash"] == gate["checkpoint_hash"], "Invalid evaluation")
    selected = []
    for p in root.rglob("*"):
        require(not p.is_symlink(), "Linked source")
        if p.is_file() and p.suffix in {".json", ".jsonl", ".log", ".tsv"} and not p.name.endswith(".next.json"):
            selected.append(p)
    handoff = base/(RUN+".handoff.log")
    require(handoff.is_file() and not handoff.is_symlink(), "Missing/linked handoff")
    selected.append(handoff)
    require(not out.exists() and not out.is_symlink(), "Export must be fresh")
    out.mkdir()
    entries = []
    archive = out/"results.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for p in sorted(selected):
            size = p.stat().st_size
            require(size <= 256*1024**2, "Oversized result text")
            with p.open("rb") as f:
                data = f.read(size)
            require(len(data) == size, "Source shrank")
            rel = p.relative_to(base).as_posix()
            info = tarfile.TarInfo(rel)
            info.size, info.mode = size, 0o644
            tar.addfile(info, io.BytesIO(data))
            entries.append(dict(path=rel, bytes=size, sha256=digest(data)))
    parts, full = [], hashlib.sha256()
    with archive.open("rb") as f:
        while data := f.read(20*1024**2):
            name = f"results.tar.gz.part{len(parts):03d}"
            with (out/name).open("xb") as dest:
                dest.write(data)
            full.update(data)
            parts.append(dict(path=name, bytes=len(data), sha256=digest(data)))
    manifest = dict(success=True, runs=[RUN], files=entries, parts=parts,
                    archive_sha256=full.hexdigest(), created_utc=datetime.now(timezone.utc).isoformat(),
                    note="Result text only; live heartbeat/logs are per-file snapshots.")
    with (out/"manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(dict(success=True, files=len(entries), parts=parts, archive_sha256=full.hexdigest())), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    pack(a.base, a.output)
