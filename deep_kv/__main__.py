"""Generate the sequential four-arm queue or compare its results. Training is train.py."""
import argparse
import json
from pathlib import Path


def jobs(config_path, stop_after=None):
    config = json.loads(Path(config_path).read_text())
    end = config.get("stop_after") if stop_after is None else stop_after
    if end is None:
        end = config["max_steps"]
    if type(end) is not int or not 1 <= end <= config["max_steps"]:
        raise ValueError("Invalid fixed-schedule cutoff")
    root = Path(__file__).resolve().parent.parent
    items = []
    for arm in "ABCD":
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
    items.append({"name": "compare", "argv": ["{python}", "-m", "deep_kv", "report", "--run-dir", "{run_dir}"],
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
    p = sub.add_parser("report")
    p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "make-jobs":
        value = jobs(args.config, args.stop_after)
        with args.output.open("x") as handle:
            json.dump(value, handle, indent=2)
    else:
        from .report import report
        print(json.dumps(report(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
