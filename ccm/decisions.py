"""Explicit final-evaluation lock; no automatic scale-up or remote execution."""
from .contracts import require, write_json, STAGE2_ARMS, digest_json
from .runtime import checkpoint_meta


def lock_final(args):
    require(args.confirm_choices_locked, "Explicit confirmation of locked choices is required")
    require(args.cluster in ("doc_id", "content_hash"), "Bootstrap cluster policy required")
    require(args.cross_seed_coupling in ("shared", "independent"), "Cross-seed bootstrap policy required")
    metas = [checkpoint_meta(p) for p in args.checkpoints]
    first = metas[0]
    expected_arms = set(STAGE2_ARMS) | ({"delta"} if args.include_delta else set())
    seen = set()
    for m in metas:
        require(m["phase"] == "stage2" and m["step"] == m["total_steps"], "Final evaluation requires complete Stage-2 runs")
        require(m["corpus_hash"] == first["corpus_hash"], "Final checkpoints have different data")
        require(m["arm"] in expected_arms, "Undeclared final comparison arm")
        pair = (m["seed"], m["arm"])
        require(pair not in seen, "Duplicate seed/arm in final lock")
        seen.add(pair)
    seeds = sorted({m["seed"] for m in metas})
    require(seeds == [17, 29, 43] or args.engineering, "Scientific final lock requires all three backbone seeds")
    require(seen == {(s, a) for s in seeds for a in expected_arms}, "Missing required seed/arm in final lock")
    require(all(m["engineering"] == args.engineering for m in metas), "Engineering/pilot lock mismatch")
    for seed in seeds:
        panel = [m for m in metas if m["seed"] == seed]
        require(len({m["source_checkpoint_hash"] for m in panel}) == 1, "Final arms use different common writers")
        memory = [m for m in panel if m["arm"] != "base"]
        require(len({m["vocabulary_hash"] for m in memory}) == 1, "Final memory arms use different vocabularies")
        require(len({m["paired_initial_reader_hash"] for m in memory}) == 1, "Final readers were not paired")
    lock = dict(choices_locked=True, engineering=args.engineering, corpus_hash=first["corpus_hash"],
                required_arms=sorted(expected_arms), seeds=seeds,
                checkpoint_hashes=[m["model_sha256"] for m in metas],
                checkpoint_metadata_hashes=[m["metadata_hash"] for m in metas],
                statistical_policy=dict(cluster=args.cluster, cross_seed_coupling=args.cross_seed_coupling),
                note="Authorizes one locked held-out evaluation per named checkpoint; does not establish scientific success")
    lock["lock_hash"] = digest_json(lock)
    write_json(args.output, lock)
    return lock
