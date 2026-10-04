"""Generate a sequential arm queue or compare its results. Training is train.py."""
import argparse
import json
from pathlib import Path
from . import ALL_ARMS, PROXY_ARMS, SCREEN_ARMS


def jobs(config_path, stop_after=None, arms=None, seeds=None):
    config = json.loads(Path(config_path).read_text())
    arms = tuple(arms if arms is not None else SCREEN_ARMS if config.get('proxy_screen') else 'ABCD')
    if not arms or len(set(arms)) != len(arms) or any(arm not in ALL_ARMS for arm in arms):
        raise ValueError("Select nonempty, unique arms from " + "/".join(ALL_ARMS))
    # A is shared by both experiment families. Select its matching model path
    # before constructing any jobs; otherwise comparison fails after training.
    if config.get('proxy_screen') or any(arm in PROXY_ARMS for arm in arms):
        if any(arm not in ('A',) + PROXY_ARMS for arm in arms):
            raise ValueError('Cannot mix legacy Deep-KV arms with proxy screening arms')
        config['proxy_screen'] = True
    if seeds is None and config.get('proxy_screen'):
        seeds = [42,43,44]
    if seeds is None:
        return arm_jobs(config,stop_after,arms)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    items = []
    for seed in seeds:
        prefix = f'seed-{seed}'
        for job in arm_jobs({**config,'seed':seed,'data_seed':seed},stop_after,arms)['jobs']:
            job['name'] = prefix+'-'+job['name']
            job['argv'] = [a.replace('{run_dir}', '{run_dir}/'+prefix) for a in job['argv']]
            for output in job['required_outputs']:
                output['path'] = prefix+'/'+output['path']
            items.append(job)
    items.append(dict(name='compare-seeds',argv=['{python}','-m','deep_kv','report-seeds','--run-dir','{run_dir}',
                     '--arms',*arms,'--seeds',*map(str,seeds)],
                     required_outputs=[dict(path='comparison-seeds.json',json_equals={'status':'complete'})]))
    return {'jobs':items}


def arm_jobs(config, stop_after, arms):
    end = config.get("stop_after") if stop_after is None else stop_after
    if end is None:
        end = config["max_steps"]
    if type(end) is not int or not 1 <= end <= config["max_steps"]:
        raise ValueError("Invalid fixed-schedule cutoff")
    root = Path(__file__).resolve().parent.parent
    items = []
    for arm in arms:
        args = {**config, "arm": arm, "output_dir": "{run_dir}/" + arm,
                "run_name": f"deep-kv-{arm}", "stop_after": end}
        argv = ["{python}", "-m", "accelerate.commands.launch", "--config_file",
                str(root / "resources/accelerate_config.yaml"), str(root / "train.py")]
        for key, value in args.items():
            # HF's pinned CLI accepts one report_to string, even though JSON
            # configs also allow a list. Other list arguments use nargs='+'.
            if key == "report_to" and isinstance(value, list):
                if len(value) > 1:
                    raise ValueError("The queue CLI supports one report_to integration")
                value = value[0] if value else "none"
            if value is not None:
                values = value if isinstance(value, list) else [value]
                if not values:
                    raise ValueError(f"Empty CLI list for {key}; omit it to use the HF default")
                argv.append(f"--{key}")
                argv.extend(json.dumps(v) if isinstance(v, (dict, bool)) else str(v) for v in values)
        items.append({"name": f"arm-{arm}", "gpus": list(range(8)), "argv": argv,
                      "required_outputs": [{"path": f"{arm}/result.json", "json_equals": {
                          "arm": arm, "global_step": end,
                          "status": "complete" if end == config["max_steps"] else "stopped"}}]})
    items.append({"name": "compare", "argv": ["{python}", "-m", "deep_kv", "report", "--run-dir", "{run_dir}", "--arms", *arms],
                  "required_outputs": [{"path": "comparison.json", "json_equals": {
                      "status": "complete", "compared_update": end}}]})
    return {"jobs": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-jobs")
    p.add_argument("--config", type=Path, default=Path("deep_kv.b200.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--arms", nargs="+", choices=ALL_ARMS)
    p.add_argument('--seeds', nargs='+', type=int)
    p = sub.add_parser("report")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list("ABCD"))
    p = sub.add_parser('report-seeds')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--arms', nargs='+', choices=ALL_ARMS, default=list(SCREEN_ARMS))
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    args = parser.parse_args()
    if args.command == "make-jobs":
        value = jobs(args.config, args.stop_after, args.arms, args.seeds)
        with args.output.open("x") as handle:
            json.dump(value, handle, indent=2)
    elif args.command == 'report':
        from .report import report
        print(json.dumps(report(args.run_dir, args.arms), indent=2))
    else:
        from .report import report_seeds
        print(json.dumps(report_seeds(args.run_dir,args.arms,args.seeds),indent=2))


if __name__ == "__main__":
    main()
