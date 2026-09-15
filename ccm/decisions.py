"""Explicit final-evaluation lock; no automatic scale-up or remote execution."""
from .contracts import require, write_json, STAGE2_ARMS, digest_json
from .runtime import checkpoint_meta
from .studies import SCALEUP, FOLLOWUP, PILOT_STUDY, SCALEUP_STAGE2


def lock_final(args):
    require(args.confirm_choices_locked, "Explicit confirmation of locked choices is required")
    require(args.cluster in ("doc_id", "content_hash"), "Bootstrap cluster policy required")
    require(args.cross_seed_coupling in ("shared", "independent"), "Cross-seed bootstrap policy required")
    metas = [checkpoint_meta(p) for p in args.checkpoints]
    require(metas, "Final lock requires a nonempty panel")
    first = metas[0]
    study = getattr(args, "study", PILOT_STUDY)
    require(study in (PILOT_STUDY, FOLLOWUP, SCALEUP), "Unknown final study")
    require(study == PILOT_STUDY or not args.include_delta, "Delta is not in either new final panel")
    expected_arms = (set(STAGE2_ARMS) | ({"delta"} if args.include_delta else set())
                     if study == PILOT_STUDY else set(SCALEUP_STAGE2))
    seen = set()
    for m in metas:
        allowed = (PILOT_STUDY, FOLLOWUP) if study == FOLLOWUP else (study,)
        require(m.get("study", PILOT_STUDY) in allowed, "Final lock mixes studies")
        if not args.engineering:
            require(m["backbone_contract"]["config"]["num_hidden_layers"] == (28 if study == SCALEUP else 12),
                    "Wrong final backbone depth")
            require(m["total_steps"] == (7629 if study == SCALEUP else 3815), "Wrong final optimization budget")
        require(m["phase"] == "stage2" and m["step"] == m["total_steps"], "Final evaluation requires complete Stage-2 runs")
        require(m["corpus_hash"] == first["corpus_hash"], "Final checkpoints have different data")
        require(m["arm"] in expected_arms, "Undeclared final comparison arm")
        pair = (m["seed"], m["arm"])
        require(pair not in seen, "Duplicate seed/arm in final lock")
        seen.add(pair)
    seeds = sorted({m["seed"] for m in metas})
    valid_seeds = (seeds == [17, 29] if study == FOLLOWUP else
                   seeds in ([17], [17, 29, 43]) if study == SCALEUP else seeds == [17, 29, 43])
    require(valid_seeds or args.engineering, "Scientific final lock has an incomplete declared backbone panel")
    if study == SCALEUP and seeds == [17] and not args.engineering:
        # Pre-launch decision 2(a), 2026-09-15: D_val_28 waits for the frozen
        # replication panel unless the study explicitly stops at seed 17.
        require(getattr(args, "single_seed_terminal", False),
                "A seed-17-only 28L lock requires --single-seed-terminal declaring the study stops at seed 17")
    require(seen == {(s, a) for s in seeds for a in expected_arms}, "Missing required seed/arm in final lock")
    require(all(m["engineering"] == args.engineering for m in metas), "Engineering/pilot lock mismatch")
    for seed in seeds:
        panel = [m for m in metas if m["seed"] == seed]
        require(len({m["source_checkpoint_hash"] for m in panel}) == 1, "Final arms use different common writers")
        memory = [m for m in panel if m["arm"] != "base"]
        require(len({m["vocabulary_hash"] for m in memory}) == 1, "Final memory arms use different vocabularies")
        require(len({m["paired_initial_reader_hash"] for m in memory}) == 1, "Final readers were not paired")
    lock = dict(study=study, choices_locked=True, engineering=args.engineering, corpus_hash=first["corpus_hash"],
                required_arms=sorted(expected_arms), seeds=seeds,
                checkpoint_hashes=[m["model_sha256"] for m in metas],
                checkpoint_metadata_hashes=[m["metadata_hash"] for m in metas],
                statistical_policy=dict(cluster=args.cluster, cross_seed_coupling=args.cross_seed_coupling),
                note="Authorizes one locked held-out evaluation per named checkpoint; does not establish scientific success")
    if study == FOLLOWUP:
        lock["protocol_amendment"] = "Post-hoc Shallow; two-seed 12L heldout, not the original three-seed confirmation"
    if study == SCALEUP:
        lock["holdout"] = "Fresh D_val_28 only, bound to the extended corpus hash"
        if seeds == [17]:
            lock["single_seed_terminal"] = bool(getattr(args, "single_seed_terminal", False))
    lock["lock_hash"] = digest_json(lock)
    write_json(args.output, lock)
    return lock
