"""Verify and extract the complete seed29 Stage1/Stage2 result bundle."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def verify(bundle, dest):
    manifest = json.loads((bundle/"manifest.json").read_text())
    if manifest.get("runs") != ["ccm_replication_seed29_20260914_a01"]:
        raise ValueError("Wrong run bundle")
    if manifest.get("success") is not True:
        raise ValueError("Incomplete bundle")
    data = []
    for part in manifest["parts"]:
        path = safe_relative(part["path"])
        if len(path.parts) != 1:
            raise ValueError("Parts must be direct children")
        content = (bundle/part["path"]).read_bytes()
        if len(content) != part["bytes"] or digest(content) != part["sha256"]:
            raise ValueError("Part checksum mismatch")
        data.append(content)
    archive = b"".join(data)
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("Full archive checksum mismatch")
    expected = {item["path"]: item for item in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate manifest paths")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Archive membership mismatch")
        for member in members:
            path = safe_relative(member.name)
            if not member.isfile() or path.parts[0] not in manifest["runs"] + [r+".handoff.log" for r in manifest["runs"]]:
                raise ValueError("Unsafe archive member")
            record = expected[member.name]
            content = tar.extractfile(member).read()
            if len(content) != record["bytes"] or digest(content) != record["sha256"]:
                raise ValueError("Member checksum mismatch")
        dest.mkdir(parents=False, exist_ok=False)
        for member in members:
            target = dest/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(tar.extractfile(member).read())
    root = dest/"ccm_replication_seed29_20260914_a01"
    decision = json.loads((root/"stage1/delta_decision.json").read_text())
    if decision["decision_seed"] != 29 or decision["replication_policy"] != "per_seed":
        raise ValueError("Wrong seed decision")
    arms1 = ["base", "contextual", "isolated", "shuffled", "shallow", "delta", "grad"]
    arms2 = ["base", "contextual", "isolated", "shuffled", "grad"] + (["delta"] if decision["include_delta"] else [])
    expected_evals = {f"{root.name}/{phase}/eval/{arm}/metrics.json"
                      for phase, arms in (("stage1", arms1), ("stage2", arms2)) for arm in arms}
    done = json.loads((root/"complete.json").read_text())
    if (done.get("success") is not True or done.get("event") != "seed29_replication_verified_and_burns_active"
            or done["stage2_arms"] != arms2):
        raise ValueError("Unverified workflow")
    for phase, arms in (("stage1", arms1), ("stage2", arms2)):
        gate = json.loads((root/phase/"validated_panel.json").read_text())
        complete = json.loads((root/phase/"complete.json").read_text())
        if not gate["success"] or not complete["success"] or gate["seed"] != 29 or gate["evaluated_arms"] != arms:
            raise ValueError("Wrong phase panel")
    evaluations = []
    for metrics in sorted(dest.glob("*/stage*/eval/*/metrics.json")):
        report = json.loads(metrics.read_text())
        if report["seed"] != 29 or report["role"] != "dev" or report["final_evaluation"]:
            raise ValueError("Wrong evaluation seed/role")
        segments = metrics.with_name("segments.jsonl")
        gate = json.loads((metrics.parents[2]/f"validated_eval_{metrics.parent.name}.json").read_text())
        if (gate.get("success") is not True or digest(metrics.read_bytes()) != gate["metrics_sha256"] or
            digest(segments.read_bytes()) != gate["segments_sha256"] or
            report["segments_sha256"] != gate["segments_sha256"] or
            report["checkpoint_hash"] != gate["checkpoint_hash"]):
            raise ValueError("Evaluation differs from its original validation gate")
        evaluations.append(str(metrics.relative_to(dest)))
    if set(evaluations) != expected_evals:
        raise ValueError("Evaluation set differs from the scheduled two-stage panel")
    result = dict(success=True, files=len(expected), evaluations=evaluations,
                  archive_sha256=manifest["archive_sha256"])
    with (dest/"VERIFIED.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    verify(args.bundle, args.output)
