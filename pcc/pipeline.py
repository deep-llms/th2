"""One in-process screen → gated full probe, usable as a generic runner job."""
from pathlib import Path
import time

from .plan import pipeline_plan
from .probe import probe, read_json
from .screen import screen, write_json
from .diagnostics import run_preflight
from .input_check import check_inputs


def completed_stage(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise ValueError(f"Stage has a failure marker: {directory}")
    completed = read_json(directory / "complete.json")
    if not isinstance(completed, dict) or completed.get("status") != "ok":
        raise ValueError(f"Stage did not complete successfully: {directory}")
    return completed


def pipeline(backbone, config, output, *, provenance=None, load_data=None, check_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages = []
    try:
        write_json(output / "plan.json", pipeline_plan(config, check_only=check_only))
        write_json(output / "provenance.json", provenance or {"invoked_via_python_api": True})
        if check_only:
            write_json(output / "preflight.json", run_preflight(backbone))
        report = check_inputs(backbone, config["train_data"], config["val_data"], load_data=load_data)
        write_json(output / "input-check.json", report)
        if check_only:
            result = {"status": "ok", "decision": "inputs_validated", "test_unlocked": False,
                      "scientific_training_performed": False, "pretraining_authorized": False,
                      "elapsed_seconds": time.monotonic() - started}
            print("PIPELINE model/data checks complete; no experiment training started", flush=True)
            write_json(output / "complete.json", result)
            return result
        write_json(output / "screen-started.json", {"status": "running"})
        print("PIPELINE screen started", flush=True)
        decision = screen(backbone, config["train_data"], config["val_data"],
                          output / "screen", microbatch=config["microbatch"], provenance=provenance, load_data=load_data)
        screen_complete = completed_stage(output / "screen")
        if screen_complete.get("decision") != decision.get("decision"):
            raise ValueError("Screen completion disagrees with its returned decision")
        stages.append({"name": "screen", "status": "ok", "decision": decision["decision"]})
        write_json(output / "screen-finished.json", stages[-1])
        if decision["decision"] == "stop_negative_screen" and decision.get("selected_pair") is None:
            stages.append({"name": "probe", "status": "skipped", "reason": "negative_screen"})
            result = {"status": "ok", "decision": "stop_negative_screen", "test_unlocked": False,
                      "pilot_specification_recommended": False, "pretraining_authorized": False}
            print("PIPELINE negative screen; full probe skipped", flush=True)
        elif decision["decision"] == "proceed_to_full_probe" and decision.get("selected_pair") is not None:
            write_json(output / "probe-started.json", {"status": "running", "selected_pair": decision["selected_pair"]})
            print("PIPELINE full probe started", flush=True)
            result = probe(backbone, output / "screen", config["train_data"], config["val_data"],
                           config.get("test_data"), output / "probe",
                           microbatch=config["microbatch"], provenance=provenance, load_data=load_data)
            probe_complete = completed_stage(output / "probe")
            if probe_complete != result:
                raise ValueError("Probe completion disagrees with its returned result")
            stages.append({"name": "probe", "status": "ok", "decision": result["decision"]})
            write_json(output / "probe-finished.json", stages[-1])
        else:
            raise ValueError("Unrecognized or inconsistent screen decision")
        result = {**result, "stages": stages, "elapsed_seconds": time.monotonic() - started}
        print(f"PIPELINE complete: {result['decision']}", flush=True)
        write_json(output / "complete.json", result)
        return result
    except BaseException as error:
        try:
            write_json(output / "failure.json", {"status": "failed", "error": str(error),
                                                 "finished_stages": stages})
        except Exception as report_error:
            error.add_note(f"Could not save pipeline failure.json: {report_error}")
        raise
