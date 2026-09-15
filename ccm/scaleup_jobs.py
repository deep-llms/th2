"""Generate explicit sequential manifests; never launch jobs or manage GPUs.

Common and 12L Shallow queues can be submitted independently on disjoint GPU
subsets. Final evaluation is a separate, explicitly confirmed invocation.
"""
from pathlib import Path
import torch
from .contracts import require, write_json, read_json, file_hash, digest_json
from .runtime import checkpoint_meta
from .studies import SCALEUP, FOLLOWUP, SCALEUP_STAGE1, SCALEUP_STAGE2, study_of


def validate_run(args, corpus):
    run = Path(args.run)
    marker = read_json(run/"complete.json")
    total = dict(common=corpus.budget.common_steps, stage1=corpus.budget.adapt_steps,
                 stage2=corpus.budget.continue_steps)[args.phase]
    require(marker.get("success") is True and marker.get("step") == total
            and marker.get("phase") == args.phase and marker.get("arm") == args.arm
            and marker.get("input_tokens") == total*corpus.budget.batch_tokens
            and marker.get("checkpoint") == f"checkpoint-{total}", "Incomplete/wrong training endpoint")
    ck = run/f"checkpoint-{total}"
    meta = checkpoint_meta(ck)
    require(meta["phase"] == args.phase and meta["arm"] == args.arm and meta["seed"] == args.seed
            and meta["step"] == meta["total_steps"] == total and meta["corpus_hash"] == corpus.meta["manifest_hash"],
            "Checkpoint does not match requested study/phase/seed/data/steps")
    require(meta["engineering"] == corpus.meta["engineering"], "Checkpoint engineering mode mismatch")
    if not meta["engineering"]:
        require(meta["backbone_contract"]["config"]["num_hidden_layers"] == (28 if study_of(corpus) == SCALEUP else 12),
                "Unexpected checkpoint depth")
    require(file_hash(ck/"optimizer.pt") == meta["optimizer_sha256"], "Optimizer checksum mismatch")
    # Verify tensors, not just JSON summaries. mmap avoids eagerly copying the
    # full Grad optimizer snapshot into a second large CPU allocation.
    for name in ("model.pt", "optimizer.pt"):
        state = torch.load(ck/name, weights_only=True, map_location="cpu", mmap=True)
        def finite(value):
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), f"Nonfinite checkpoint tensor in {name}")
            elif isinstance(value, dict):
                for v in value.values(): finite(v)
            elif isinstance(value, (tuple, list)):
                for v in value: finite(v)
        finite(state)
        del state
    records = 0
    with (run/"train.jsonl").open() as f:
        import json, math
        for line in f:
            row = json.loads(line); records += 1
            require(row["step"] == records and row["input_tokens"] == records*corpus.budget.batch_tokens,
                    "Training log skips or repeats updates")
            require(all(math.isfinite(row[k]) for k in ("nll", "grad_norm", "lr")), "Nonfinite training log")
    require(records == total, "Wrong number of training updates")
    report = dict(success=True, checkpoint=str(ck.resolve()), metadata_hash=meta["metadata_hash"],
                  corpus_hash=corpus.meta["manifest_hash"], phase=args.phase, arm=args.arm, seed=args.seed)
    write_json(args.output, report)
    return report


def make_jobs(args):
    require(args.gpus and len(set(args.gpus)) == len(args.gpus) and min(args.gpus) >= 0, "Unique nonnegative GPUs required")
    require(args.queue == "shallow12" or args.seed == 17, "Only the first 28L seed is queued by this plan")
    require(args.queue != "shallow12" or args.seed in (17,29), "12L follow-up is seeds 17 and 29 only")
    # Pre-launch decision 1(c), 2026-09-15: a fixed 32-update pipeline smoke runs
    # first; the stability smoke must then run past the 763-update warmup at peak LR.
    require(763 < args.stability_steps < 38147 and args.microbatch_segments > 0, "Invalid training smoke/microbatch size")
    # Only argv arrays, never shell interpolation. These placeholders are the
    # documented run_experiments.py placeholders, not environment substitution.
    root = "{run_dir}"
    jobs = []
    def add(name, command, output, equals=None, gpu=False, ddp=False):
        argv = ["{python}"]
        if ddp:
            argv += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(args.gpus)}"]
        argv += ["-m", "ccm"]+list(map(str,command))
        job = dict(name=name, argv=argv, required_outputs=[dict(path=output)])
        if equals is not None: job["required_outputs"][0]["json_equals"] = equals
        if gpu: job["gpus"] = args.gpus
        jobs.append(job)
    data, vocab = args.data, args.vocabulary
    cov = f"{root}/coverage.json"
    def valid(phase, arm, run, name):
        add(f"verify-{name}", ["validate-run","--data",data,"--run",run,"--phase",phase,"--arm",arm,
            "--seed",args.seed,"--output",f"{root}/verified-{name}.json"],f"verified-{name}.json",dict(success=True))
    def training(phase, arm, study, checkpoint=None, table=None, stability=None, name=None):
        name = name or f"{phase}-{arm}"
        cmd=["train","--study",study,"--data",data,"--phase",phase,"--arm",arm,"--seed",args.seed,
             "--output",f"{root}/{name}","--device","cuda","--microbatch-segments",args.microbatch_segments,
             "--loss-chunk","1024","--activation-checkpointing"]
        if study == SCALEUP:
            # Pre-launch decision 3a, 2026-09-15: sparse scale-up checkpoints
            # (common 5000, Stage-1 final-only, Stage-2 2000, plus the endpoint
            # each phase always saves). Smokes save only their endpoint. The 12L
            # follow-up keeps the historical every-1000 convention of its arms.
            cmd += ["--save-every", stability or dict(common=5000, stage1=977, stage2=2000)[phase]]
        if phase == "common":
            require(args.model_config is not None,"Common queue requires local model config")
            cmd += ["--model-config",args.model_config,"--common-lr",args.common_lr]
            if args.lr_failure_report: cmd += ["--lr-failure-report",args.lr_failure_report]
            if stability is not None: cmd += ["--stability-steps",stability]
            else: cmd += ["--stability-report",f"{root}/common-stability/stability.json"]
        else:
            cmd += ["--checkpoint",checkpoint,"--vocabulary",vocab,"--coverage",cov]
            if table: cmd += ["--table",table]
        end = "stability.json" if stability is not None else "complete.json"
        add(name,cmd,f"{name}/{end}",dict(success=True),gpu=True,ddp=True)
        if stability is None: valid(phase,arm,f"{root}/{name}",name)
        steps = 38147 if phase=="common" else 977 if phase=="stage1" else 7629 if study==SCALEUP else 3815
        return f"{root}/{name}/checkpoint-{steps}"
    def eval_arm(phase,arm,checkpoint,tables):
        name=f"{phase}-eval-{arm}"
        add(name,["evaluate","--data",data,"--vocabulary",vocab,"--checkpoint",checkpoint,
            "--diagnostic-table",f"{tables}/contextual","--device","cuda","--microbatch-segments","4",
            "--loss-chunk","1024","--output",f"{root}/{name}"],f"{name}/metrics.json",
            dict(role="dev",phase="common" if phase=="stage1" and arm=="base" else phase,arm=arm,seed=args.seed),gpu=True)
        # evaluate is single-device; only GPU0 of the explicitly assigned set
        # is visible. Idle-workload policy belongs to the outer live handoff.
        jobs[-1]["gpus"]=[args.gpus[0]]
    def contrasts(phase,controls):
        for control in controls:
            name=f"{phase}-contextual-vs-{control}"
            add(name,["compare","--left",f"{root}/{phase}-eval-contextual","--right",f"{root}/{phase}-eval-{control}",
                "--cluster","doc_id","--cross-seed-coupling","independent","--replicates","10000",
                "--output",f"{root}/{name}.json"],f"{name}.json",dict(final_report=False,population="overall"))
    if args.queue == "common":
        training("common","base",SCALEUP,stability=32,name="common-pipeline")
        training("common","base",SCALEUP,stability=args.stability_steps,name="common-stability")
        training("common","base",SCALEUP)
    else:
        require(args.checkpoint is not None,"Panel/follow-up requires a completed common checkpoint")
        valid("common","base",str(Path(args.checkpoint).parent),"common-input")
        add("coverage",["coverage","--data",data,"--vocabulary",vocab,"--output",cov],"coverage.json",dict(passed=True))
        if args.queue == "shallow12":
            require(args.tables is not None,"Shallow follow-up requires historical tables")
            ck=training("stage2","shallow",FOLLOWUP,args.checkpoint,str(Path(args.tables)/"shallow"))
            eval_arm("stage2","shallow",ck,args.tables)
            require(args.contextual_eval is not None,"Follow-up must compare to the historical Contextual evaluation")
            add("deep-vs-shallow",["compare","--left",args.contextual_eval,"--right",f"{root}/stage2-eval-shallow",
                "--cluster","doc_id","--cross-seed-coupling","independent","--replicates","10000",
                "--output",f"{root}/deep-vs-shallow.json"],"deep-vs-shallow.json",dict(final_report=False))
        else:
            require(args.seed == 17,"First-backbone scale-up queue only; no automatic replication")
            tables=f"{root}/tables"
            common=["compile","--data",data,"--vocabulary",vocab,"--checkpoint",args.checkpoint,
                    "--device","cuda","--microbatch-segments","4","--reserved-gpus",len(args.gpus)]
            if args.compiler == "distributed":
                add("compiler-gate",common+["--validate-distributed","--output",f"{root}/compiler-gate"],
                    "compiler-gate/validation.json",dict(success=True),gpu=True,ddp=True)
                add("compile",common+["--distributed","--compiler-validation",f"{root}/compiler-gate/validation.json",
                    "--output",tables],"tables/complete.json",dict(success=True),gpu=True,ddp=True)
            else:
                add("compile-reference",common+["--output",tables],"tables/complete.json",dict(success=True),gpu=True)
                jobs[-1]["gpus"]=[args.gpus[0]]
            eval_arm("stage1","base",args.checkpoint,tables)
            for phase,arms in (("stage1",SCALEUP_STAGE1),("stage2",SCALEUP_STAGE2)):
                for arm in arms:
                    ck=training(phase,arm,SCALEUP,args.checkpoint,f"{tables}/{arm}" if arm not in ("base","grad") else None)
                    eval_arm(phase,arm,ck,tables)
                contrasts(phase,("isolated","shuffled","shallow","grad") if phase=="stage1" else ("isolated","shuffled","shallow","base","grad"))
            add("replication-decision",["scaleup-decision","--isolated-report",f"{root}/stage2-contextual-vs-isolated.json",
                "--shuffled-report",f"{root}/stage2-contextual-vs-shuffled.json","--output",f"{root}/decision.json"],"decision.json",dict(automatic_launch=False))
    return dict(jobs=jobs)
