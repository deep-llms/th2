"""Sequential orchestration checks; all workloads and subprocesses are local CPU fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_experiments as runner
from pcc.plan import PATH_FIELDS, load_pipeline_config, pipeline_plan
from test_pcc import HAS_ML

ROOT = Path(__file__).resolve().parents[1]
if HAS_ML:
    from pcc.pipeline import pipeline
    from pcc.screen import write_json
    import test_probe as probe_fixtures


class PlanTests(unittest.TestCase):
    def config(self):
        return {key: f"missing/{key}" for key in PATH_FIELDS}

    def test_config_defaults_and_relative_paths_need_no_input_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "config.json"
            path.write_text(json.dumps(self.config()))
            with patch.object(Path, "resolve", side_effect=AssertionError("Must not resolve test symlinks")), \
                    patch.object(Path, "stat", side_effect=AssertionError("Must not inspect inputs")):
                config = load_pipeline_config(path)
            self.assertEqual(config["microbatch"], 1)
            self.assertEqual(config["test_data"], str(root / "missing/test_data"))
            self.assertEqual([x["name"] for x in pipeline_plan(config)["ordered_stages"]], ["screen", "probe"])

    def test_invalid_configs_fail_before_model_loading(self):
        bad = [[], {}, {**self.config(), "updates": 2}, {**self.config(), "microbatch": True},
               {**self.config(), "microbatch": 3}, {**self.config(), "test_data": ""}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for value in bad:
                path.write_text(json.dumps(value))
                with self.subTest(config=value), self.assertRaises(ValueError):
                    load_pipeline_config(path)

    def test_test_set_is_optional_without_manufacturing_a_split(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            config = self.config()
            del config["test_data"]
            path.write_text(json.dumps(config))
            loaded = load_pipeline_config(path)
            self.assertIsNone(loaded["test_data"])
            self.assertIn("finish at validation", pipeline_plan(loaded)["without_test_data"])

    def test_cli_dry_run_does_not_import_torch_create_outputs_or_query_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.json"
            config.write_text(json.dumps(self.config()))
            output = root / "never-created"
            code = """import sys
from unittest.mock import patch
from pcc.__main__ import main
from scripts import gpu_status
with patch.object(gpu_status, 'require_free', side_effect=AssertionError('GPU query')):
    main()
assert 'torch' not in sys.modules
"""
            result = subprocess.run([sys.executable, "-c", code, "pipeline", "--config", str(config),
                "--output", str(output), "--dry-run", "--physical-gpu", "7"], cwd=ROOT,
                text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["physical_gpu"], 7)
            self.assertFalse(output.exists())

    def test_example_runner_manifest_validates_without_launch(self):
        jobs = runner.load_jobs(ROOT / "jobs.pcc.example.json")
        self.assertEqual(len(jobs), 1)
        self.assertNotIn("gpus", jobs[0])
        self.assertIn("pipeline", jobs[0]["argv"])

    def test_check_only_dry_run_plans_no_training_and_does_not_load_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.json"
            config.write_text(json.dumps(self.config()))
            code = "from pcc.__main__ import main; import sys; main(); assert 'torch' not in sys.modules"
            result = subprocess.run([sys.executable, "-c", code, "pipeline", "--config", str(config),
                                     "--dry-run", "--check-only"], cwd=ROOT, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertTrue(plan["check_only"])
            self.assertEqual([stage["name"] for stage in plan["ordered_stages"]], ["input_checks"])


@unittest.skipUnless(HAS_ML, "Requires the project ML environment")
class PipelineTests(unittest.TestCase):
    data = probe_fixtures.ProbeTests.data if HAS_ML else None

    def setUp(self):
        # Stage sequencing uses stubs; real input checking has separate tests
        # and is exercised unpatched in the CLI subprocess below.
        check = patch("pcc.pipeline.check_inputs", return_value={"status": "ok"})
        self.input_check = check.start()
        self.addCleanup(check.stop)

    def config(self):
        return {**{key: f"unused-{key}" for key in PATH_FIELDS}, "microbatch": 1}

    def screen_stub(self, selected=True, missing_marker=False):
        def stage(backbone, train, dev, output, **kwargs):
            output.mkdir()
            decision = {"decision": "proceed_to_full_probe" if selected else "stop_negative_screen",
                        "selected_pair": [4, 16] if selected else None}
            if not missing_marker:
                write_json(output / "complete.json", {"status": "ok", "decision": decision["decision"]})
            return decision
        return stage

    def test_stages_are_sequential_and_completion_waits_for_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "pipeline"
            def full(backbone, screen, train, dev, test, output, **kwargs):
                self.assertTrue((screen / "complete.json").exists())
                self.assertTrue((out / "screen-finished.json").exists())
                self.assertFalse((out / "complete.json").exists())
                output.mkdir()
                result = {"status": "ok", "decision": "stop_dev_gates", "test_unlocked": False,
                          "pretraining_authorized": False}
                write_json(output / "complete.json", result)
                return result
            with patch("pcc.pipeline.screen", side_effect=self.screen_stub()) as screen, \
                    patch("pcc.pipeline.probe", side_effect=full) as probe:
                result = pipeline(object(), self.config(), out)
            self.assertEqual(screen.call_count, 1)
            self.input_check.assert_called_once()
            self.assertEqual(probe.call_count, 1)
            self.assertEqual([stage["status"] for stage in result["stages"]], ["ok", "ok"])
            self.assertEqual(json.loads((out / "complete.json").read_text()), result)

    def test_negative_screen_is_success_and_never_calls_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "pipeline"
            with patch("pcc.pipeline.screen", side_effect=self.screen_stub(False)), \
                    patch("pcc.pipeline.probe", side_effect=AssertionError("Probe must not start")):
                result = pipeline(object(), self.config(), out)
            self.assertEqual(result["decision"], "stop_negative_screen")
            self.assertEqual(result["stages"][1]["status"], "skipped")
            self.assertFalse((out / "probe-started.json").exists())
            self.assertFalse((out / "probe").exists())

    def test_stage_failure_or_missing_artifact_prevents_completion(self):
        cases = [(RuntimeError("screen failed"), None), (self.screen_stub(missing_marker=True), None),
                 (self.screen_stub(), RuntimeError("probe failed"))]
        for screen_effect, probe_effect in cases:
            with tempfile.TemporaryDirectory() as temp, self.subTest(screen=screen_effect):
                out = Path(temp) / "pipeline"
                with patch("pcc.pipeline.screen", side_effect=screen_effect), \
                        patch("pcc.pipeline.probe", side_effect=probe_effect) as probe:
                    with self.assertRaises((RuntimeError, FileNotFoundError)):
                        pipeline(object(), self.config(), out)
                self.assertFalse((out / "complete.json").exists())
                self.assertTrue((out / "failure.json").exists())
                if probe_effect is None:
                    probe.assert_not_called()

    def test_existing_output_cannot_resume_or_overwrite_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "complete.json").write_text("preserve")
            with patch("pcc.pipeline.screen", side_effect=AssertionError("Must not launch")):
                with self.assertRaises(FileExistsError):
                    pipeline(object(), self.config(), out)
            self.assertEqual((out / "complete.json").read_text(), "preserve")

    def test_logging_failure_cannot_publish_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "pipeline"
            def log(message, **kwargs):
                if message.startswith("PIPELINE complete:"):
                    raise BrokenPipeError("injected log failure")
            with patch("pcc.pipeline.screen", side_effect=self.screen_stub(False)), patch("builtins.print", side_effect=log):
                with self.assertRaises(BrokenPipeError):
                    pipeline(object(), self.config(), out)
            self.assertTrue((out / "failure.json").exists())
            self.assertFalse((out / "complete.json").exists())

    def test_runner_executes_real_tiny_pipeline_then_next_job(self):
        # Patch only model loading and budgets in a child process. The CLI,
        # screen selection, training, pipeline, and generic runner are real.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            from test_sampled_data import sampled_fixture
            train, tokenizer, _ = sampled_fixture(root / "train")
            dev, _, _ = sampled_fixture(root / "val", sharded=False)
            tokenizer.save_pretrained(root / "tokenizer")
            config = root / "pipeline.json"
            config.write_text(json.dumps({"model_path": "tiny-unit-fixture-only",
                "train_data": str(train), "val_data": str(dev), "microbatch": 1}))
            bootstrap = root / "tiny_cli.py"
            bootstrap.write_text("""# Synthetic CPU test fixture; not a scientific launch.
import sys
sys.path.insert(0, sys.argv.pop(1))
import torch
from pcc import data, model, screen, probe, protocol
from pcc.diagnostics import tiny_backbone
import pcc.__main__ as cli
tiny = tiny_backbone()
tiny.model.to(torch.bfloat16)
from pathlib import Path
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parent / 'tokenizer', local_files_only=True)
model.load_local = lambda *args: (tiny, tokenizer)
data.CONTEXT = screen.CONTEXT = probe.CONTEXT = 12
screen.TOKENS_PER_UPDATE = probe.TOKENS_PER_UPDATE = 24
screen.SCREEN_UPDATES = probe.SCREEN_UPDATES = probe.FULL_UPDATES = 2
screen.SCREEN_DEV_TOKENS = probe.SCREEN_DEV_TOKENS = probe.PROBE_EVAL_TOKENS = 24
protocol.CONTEXT = 12
protocol.TOKENS_PER_UPDATE = 24
protocol.SCREEN_UPDATES = protocol.FULL_UPDATES = 2
protocol.SCREEN_DEV_TOKENS = protocol.PROBE_EVAL_TOKENS = 24
probe.CALIBRATION_TOKENS = 25
probe.FULL_WARMUP = 1
cli.os.cpu_count = lambda: 1
cli.main()
""")
            jobs = [{"name": "pcc", "argv": ["{python}", "-u", str(bootstrap), str(ROOT), "pipeline",
                "--config", str(config), "--output", "{run_dir}/pcc"], "timeout_seconds": 120,
                "required_outputs": [{"path": "pcc/complete.json", "json_equals": {"status": "ok"}}]},
                {"name": "after-pcc", "argv": ["{python}", "-c",
                    "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); "
                    "r=json.loads((p/'pcc/complete.json').read_text()); "
                    "assert r['status']=='ok'; "
                    "(p/'after.json').write_text(json.dumps({'status':'ok','pcc_decision':r['decision']}))",
                    "{run_dir}"], "required_outputs": [{"path": "after.json", "json_equals": {"status": "ok"}}]}]
            manifest = root / "jobs.json"
            manifest.write_text(json.dumps({"jobs": jobs}))
            out = root / "run"
            with patch.object(runner, "require_free", side_effect=AssertionError("CPU suite queried GPU")):
                try:
                    code = runner.run_jobs(runner.load_jobs(manifest), ROOT, out)
                except Exception:
                    self.fail((out / "pcc.log").read_text())
            self.assertEqual(code, 0)
            report = json.loads((out / "complete.json").read_text())
            self.assertEqual([record["name"] for record in report["jobs"]], ["pcc", "after-pcc"])
            self.assertTrue(all(record["status"] == "ok" for record in report["jobs"]))
            self.assertTrue((out / "pcc/screen/complete.json").exists())
            checked = json.loads((out / "pcc/input-check.json").read_text())
            self.assertEqual(checked["streams"]["train"]["input_tokens"], 48)
            self.assertEqual(checked["streams"]["dev"]["input_tokens"], 24)
            self.assertFalse(checked["test_data_inspected"])
            self.assertFalse((out / "pcc/failure.json").exists())

    def test_runner_stops_after_invalid_pipeline_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = root / "bad.json"
            bad.write_text('{"updates": 1}')
            jobs = [{"name": "invalid-pcc", "argv": ["{python}", "-m", "pcc", "pipeline",
                "--config", str(bad), "--output", "{run_dir}/pcc"],
                "required_outputs": [{"path": "pcc/complete.json"}]},
                {"name": "must-not-start", "argv": ["{python}", "-c", "raise AssertionError('Unexpected launch')"],
                 "required_outputs": [{"path": "unexpected.json"}]}]
            out = root / "run"
            with self.assertRaisesRegex(RuntimeError, "exited with code"):
                runner.run_jobs(jobs, ROOT, out)
            self.assertFalse((out / "must-not-start.log").exists())
            self.assertFalse((out / "complete.json").exists())
            self.assertEqual(json.loads((out / "run.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
