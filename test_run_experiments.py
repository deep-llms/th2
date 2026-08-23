"""Safety tests for experiment-runner checkpoint and fresh-output guards."""

import json
import os
import tempfile

from run_experiments import run_experiment, validate_checkpoint_artifacts


REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "embedding.pt",
    "output_head.pt",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
    "rng_state_4.pth",
    "rng_state_5.pth",
    "rng_state_6.pth",
    "rng_state_7.pth",
]


def _write_checkpoint(output_dir, step, trainer_step=None):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir)
    for filename in REQUIRED_FILES:
        path = os.path.join(checkpoint_dir, filename)
        if filename == "trainer_state.json":
            with open(path, "w") as handle:
                json.dump(
                    {"global_step": step if trainer_step is None else trainer_step},
                    handle,
                )
        else:
            with open(path, "wb") as handle:
                handle.write(b"test")


def test_validate_checkpoint_artifacts_accepts_complete_checkpoint():
    with tempfile.TemporaryDirectory() as output_dir:
        _write_checkpoint(output_dir, 10000)
        assert validate_checkpoint_artifacts(
            output_dir, 10000, REQUIRED_FILES
        ) is None


def test_validate_checkpoint_artifacts_rejects_partial_or_wrong_step():
    with tempfile.TemporaryDirectory() as output_dir:
        _write_checkpoint(output_dir, 10000, trainer_step=9999)
        error = validate_checkpoint_artifacts(
            output_dir, 10000, REQUIRED_FILES
        )
        assert "global_step=9999" in error

        os.remove(os.path.join(
            output_dir, "checkpoint-10000", "output_head.pt"
        ))
        error = validate_checkpoint_artifacts(
            output_dir, 10000, REQUIRED_FILES
        )
        assert "output_head.pt" in error


def test_fresh_output_guard_refuses_before_launch():
    with tempfile.TemporaryDirectory() as output_dir:
        result = run_experiment(
            {
                "name": "fresh-only-test",
                "cmd": "this-command-must-not-run",
                "output_dir": output_dir,
                "require_fresh_output": True,
            },
            stop_at_step=10000,
            log_dir=output_dir,
        )
    assert result["status"] == "FAILED (output path already exists)"
