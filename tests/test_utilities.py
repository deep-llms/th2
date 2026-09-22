"""Offline CPU tests. All NVIDIA/Dropbox interactions are mocked."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import HTTPHandler, build_opener

import run_experiments as runner
from scripts import dropbox_access as dropbox
from scripts import gpu_status as gpu
from scripts import verify_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)


class JobTests(TemporaryTest):
    def job(self, name="example"):
        return {"name": name, "argv": ["{python}", "scripts/example_job.py",
                "--output", "{run_dir}/result.json"], "timeout_seconds": 5,
                "required_outputs": [{"path": "result.json", "json_equals": {"status": "ok", "value": 42}}]}

    def config(self, jobs):
        path = self.root / "jobs.json"
        path.write_text(json.dumps({"jobs": jobs}))
        return path

    def test_cpu_success_and_checksum(self):
        run_dir = self.root / "deep" / "run"
        with patch.object(runner, "require_free", side_effect=AssertionError("CPU job queried GPU")):
            self.assertEqual(runner.run_jobs([self.job()], ROOT, run_dir), 0)
        report = json.loads((run_dir / "complete.json").read_text())
        self.assertEqual(report["status"], "ok")
        artifact = report["jobs"][0]["artifacts"][0]
        self.assertEqual(artifact["sha256"], hashlib.sha256((run_dir / "result.json").read_bytes()).hexdigest())
        self.assertTrue((run_dir / "example.log").is_file())

    def test_existing_directory_refused_without_modification(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "sentinel").write_text("preserve")
        with self.assertRaises(FileExistsError):
            runner.run_jobs([self.job()], ROOT, run_dir)
        self.assertEqual((run_dir / "sentinel").read_text(), "preserve")

    def test_nonzero_stops_sequence(self):
        bad = self.job("bad")
        bad["argv"] = [sys.executable, "-c", "raise SystemExit(7)"]
        run_dir = self.root / "run"
        with self.assertRaisesRegex(RuntimeError, "code 7"):
            runner.run_jobs([bad, self.job("later")], ROOT, run_dir)
        self.assertFalse((run_dir / "complete.json").exists())
        self.assertFalse((run_dir / "later.log").exists())
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["status"], "failed")

    def test_later_job_cannot_claim_previous_output(self):
        later = self.job("later")
        later["argv"] = ["{python}", "-c", "pass"]
        run_dir = self.root / "run"
        with self.assertRaisesRegex(ValueError, "Output already exists before later"):
            runner.run_jobs([self.job(), later], ROOT, run_dir)
        self.assertFalse((run_dir / "complete.json").exists())
        self.assertFalse((run_dir / "later.log").exists())
        self.assertEqual(json.loads((run_dir / "result.json").read_text())["value"], 42)
        report = json.loads((run_dir / "run.json").read_text())
        self.assertEqual([j["status"] for j in report["jobs"]], ["ok", "failed"])

    def test_freshness_rejects_files_directories_and_dangling_links(self):
        (self.root / "file").write_text("preserve")
        (self.root / "directory").mkdir()
        (self.root / "link").symlink_to(self.root / "missing")
        for name in ("file", "directory", "link"):
            job = self.job()
            job["required_outputs"] = [{"path": name}]
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                runner.require_fresh_outputs(job, self.root)
        self.assertEqual((self.root / "file").read_text(), "preserve")
        self.assertTrue((self.root / "directory").is_dir())
        self.assertTrue((self.root / "link").is_symlink())

    def test_validator_can_read_previous_output_and_write_new_report(self):
        later = self.job("validator")
        later["argv"] = ["{python}", "-c",
            "import json,pathlib,sys; data=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
            "assert data['value']==42; pathlib.Path(sys.argv[2]).write_text(json.dumps({'validated':True}))",
            "{run_dir}/result.json", "{run_dir}/validation.json"]
        later["required_outputs"] = [{"path": "validation.json", "json_equals": {"validated": True}}]
        run_dir = self.root / "run"
        self.assertEqual(runner.run_jobs([self.job(), later], ROOT, run_dir), 0)
        report = json.loads((run_dir / "complete.json").read_text())
        self.assertEqual([j["status"] for j in report["jobs"]], ["ok", "ok"])

    def test_cli_stale_output_failure_is_nonzero(self):
        later = self.job("later")
        later["argv"] = ["{python}", "-c", "pass"]
        config = self.config([self.job(), later])
        run_dir = self.root / "run"
        result = subprocess.run([sys.executable, str(ROOT / "run_experiments.py"),
            "--config", str(config), "--project-dir", str(ROOT), "--run-dir", str(run_dir)],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Output already exists before later", result.stderr)
        self.assertFalse((run_dir / "complete.json").exists())

    def test_missing_or_wrong_artifact_fails(self):
        for case in ("missing", "wrong"):
            job = self.job()
            if case == "missing":
                job["argv"] = [sys.executable, "-c", "print('finished')"]
            else:
                job["required_outputs"][0]["json_equals"]["value"] = -1
            run_dir = self.root / case
            with self.assertRaises(ValueError):
                runner.run_jobs([job], ROOT, run_dir)
            self.assertFalse((run_dir / "complete.json").exists())

    def test_timeout_only_stops_own_child(self):
        job = self.job()
        job["argv"] = [sys.executable, "-c", "import time; time.sleep(60)"]
        job["timeout_seconds"] = 0.2
        run_dir = self.root / "run"
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.run_jobs([job], ROOT, run_dir)
        self.assertFalse((run_dir / "complete.json").exists())
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["status"], "failed")

    def test_busy_gpu_never_launches_or_kills(self):
        job = self.job()
        job["gpus"] = [2, 4]
        with patch.object(runner, "require_free", side_effect=RuntimeError("busy")), \
             patch.object(runner.subprocess, "Popen") as launch, \
             patch.object(runner.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                runner.run_jobs([job], ROOT, self.root / "run")
            launch.assert_not_called()
            kill.assert_not_called()

    def test_physical_gpu_mapping_and_postcheck(self):
        job = self.job()
        job["gpus"] = [2, 4]
        job["argv"] = ["{python}", "-c",
            "import os,json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({'gpus':os.environ['CUDA_VISIBLE_DEVICES']}))",
            "{run_dir}/result.json"]
        job["required_outputs"] = [{"path": "result.json", "json_equals": {"gpus": "2,4"}}]
        with patch.object(runner, "require_free", return_value=[]) as check:
            runner.run_jobs([job], ROOT, self.root / "run")
            self.assertEqual(check.call_count, 2)
            check.assert_called_with([2, 4])

    def test_postcheck_failure_has_no_success_marker(self):
        job = self.job()
        job["gpus"] = [0]
        run_dir = self.root / "run"
        with patch.object(runner, "require_free", side_effect=[[], RuntimeError("still busy")]):
            with self.assertRaisesRegex(RuntimeError, "still busy"):
                runner.run_jobs([job], ROOT, run_dir)
        self.assertFalse((run_dir / "complete.json").exists())

    def test_bad_manifest_rejected(self):
        mutations = [lambda j: j.update(name="../bad"),
                     lambda j: j.update(argv="echo bad"),
                     lambda j: j.update(gpus=[0, 0]),
                     lambda j: j.update(gpus=[True]),
                     lambda j: j.update(timeout_seconds=-1),
                     lambda j: j.update(timeout_seconds=float("nan")),
                     lambda j: j.update(required_outputs=[]),
                     lambda j: j.update(required_outputs=[{"path": "../outside"}]),
                     lambda j: j.update(required_outputs=[{"path": "complete.json"}]),
                     lambda j: j.update(required_outputs=[{"path": "result.json", "sha256": int("1" * 64)}])]
        for mutate in mutations:
            job = self.job()
            mutate(job)
            with self.assertRaises(ValueError):
                runner.load_jobs(self.config([job]))
        with self.assertRaises(ValueError):
            runner.load_jobs(self.config([self.job(), self.job()]))

    def test_symlink_escape_rejected(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "escape").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            runner.output_path(run_dir, "escape/secret")

    def test_unowned_process_group_is_never_signaled(self):
        proc = unittest.mock.Mock(pid=234)
        proc.poll.return_value = None
        with patch.object(runner.os, "getpgid", return_value=1), \
             patch.object(runner.os, "killpg") as kill:
            with self.assertRaises(RuntimeError):
                runner.stop_owned_process(proc)
            kill.assert_not_called()

    def test_pid_one_is_never_signaled(self):
        proc = unittest.mock.Mock(pid=1)
        proc.poll.return_value = None
        with patch.object(runner.os, "killpg") as kill:
            with self.assertRaises(RuntimeError):
                runner.stop_owned_process(proc)
            kill.assert_not_called()

    def test_list_has_no_run_side_effect(self):
        config = self.config([self.job()])
        with patch.object(sys, "argv", ["run_experiments.py", "--config", str(config), "--list"]), \
             contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner, "run_jobs") as run:
            self.assertEqual(runner.main(), 0)
            run.assert_not_called()
        self.assertEqual({p.name for p in self.root.iterdir()}, {"jobs.json"})


class GpuTests(unittest.TestCase):
    ROWS = [["0", "uuid0", "GPU A", "40960", "0", "0"],
            ["1", "uuid1", "GPU B", "40960", "500", "50"]]

    def test_subset_can_be_free_while_other_gpu_busy(self):
        with patch.object(gpu, "query", side_effect=[self.ROWS, [["uuid1", "234"]]]):
            self.assertEqual(gpu.require_free([0])[0]["index"], 0)

    def test_busy_and_unknown_uuid_fail_closed(self):
        for app in (["uuid1", "234"], ["unknown-mig", "234"]):
            with patch.object(gpu, "query", side_effect=[self.ROWS, [app]]):
                with self.assertRaises(RuntimeError):
                    gpu.require_free([1])

    def test_query_failure_not_free(self):
        with patch.object(gpu, "query", side_effect=RuntimeError("query failed")):
            with self.assertRaises(RuntimeError):
                gpu.require_free()

    def test_unknown_gpu_is_error(self):
        with patch.object(gpu, "query", return_value=self.ROWS):
            with self.assertRaises(ValueError):
                gpu.snapshot([9])

    def test_duplicate_uuid_is_error(self):
        rows = [list(row) for row in self.ROWS]
        rows[1][1] = rows[0][1]
        with patch.object(gpu, "query", return_value=rows):
            with self.assertRaises(RuntimeError):
                gpu.snapshot()


class ManifestTests(TemporaryTest):
    def test_roundtrip_and_corruption(self):
        (self.root / "file.bin").write_bytes(b"hello")
        data = manifest.create(self.root, ["file.bin"])
        self.assertEqual(manifest.verify(self.root, data), 1)
        (self.root / "file.bin").write_bytes(b"jello")
        with self.assertRaises(ValueError):
            manifest.verify(self.root, data)

    def test_extra_file_and_traversal(self):
        (self.root / "file.bin").write_bytes(b"hello")
        data = manifest.create(self.root, ["file.bin"])
        (self.root / "extra.bin").touch()
        self.assertEqual(manifest.verify(self.root, data), 1)
        with self.assertRaises(ValueError):
            manifest.verify(self.root, data, strict=True)
        with self.assertRaises(ValueError):
            manifest.create(self.root, ["../outside"])
        with self.assertRaises(ValueError):
            manifest.create(self.root, ["file.bin", "file.bin"])


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}


class DropboxTests(TemporaryTest):
    def prepare_http_request(self, req):
        # Exercise urllib's real header preparation without opening a socket.
        handler = HTTPHandler()
        handler.add_parent(build_opener())
        return handler.do_request_(req)

    def test_download_transport_has_no_form_header_or_body(self):
        def transport(req, timeout):
            prepared = self.prepare_http_request(req)
            self.assertEqual(prepared.get_method(), "POST")
            self.assertEqual(prepared.full_url,
                "https://content.dropboxapi.com/2/sharing/get_shared_link_file")
            self.assertIsNone(prepared.data)
            self.assertIsNone(prepared.get_header("Content-type"))
            self.assertEqual(json.loads(prepared.get_header("Dropbox-api-arg"))["path"], "/result")
            return Response(b"verified", 8)
        target = self.root / "result"
        with patch.object(dropbox, "urlopen", side_effect=transport) as opened:
            dropbox.download("fake", "https://www.dropbox.com/example", "/result", target)
        opened.assert_called_once()
        self.assertEqual(target.read_bytes(), b"verified")

    def test_refresh_transport_remains_form_encoded(self):
        def transport(req, timeout):
            prepared = self.prepare_http_request(req)
            self.assertEqual(prepared.get_header("Content-type"), "application/x-www-form-urlencoded")
            self.assertIn(b"grant_type=refresh_token", prepared.data)
            return Response(b'{"access_token":"fake-access"}')
        credentials = {"app_key": "fake-key", "app_secret": "fake-secret", "refresh_token": "fake-refresh"}
        with patch.dict(os.environ, {"DROPBOX_ACCESS_TOKEN": ""}), \
             patch.object(dropbox, "read_fields", return_value=credentials), \
             patch.object(dropbox, "urlopen", side_effect=transport):
            self.assertEqual(dropbox.access_token(Path("unused")), "fake-access")

    def test_shared_link_preserves_key_without_printing_it(self):
        fields = {"runner": "https://www.dropbox.com/scl/fo/example?rlkey=placeholder&dl=0"}
        with patch.object(dropbox, "read_fields", return_value=fields):
            url = dropbox.shared_link(Path("unused"), "runner")
        self.assertIn("rlkey=placeholder", url)
        self.assertNotIn("dl=", url)

    def test_pagination_and_subfolder_paths(self):
        pages = [Response(json.dumps({"entries": [{".tag": "file", "name": "a"}],
                    "has_more": True, "cursor": "c1"}).encode()),
                 Response(json.dumps({"entries": [{".tag": "file", "name": "b"}],
                    "has_more": False}).encode())]
        with patch.object(dropbox, "request", side_effect=pages) as request:
            entries = list(dropbox.list_folder("fake-token", "https://www.dropbox.com/example", "/exports"))
            self.assertFalse(json.loads(request.call_args_list[0].args[1])["recursive"])
            self.assertTrue(request.call_args_list[1].args[0].endswith("/continue"))
        self.assertEqual([e["path"] for e in entries], ["/exports/a", "/exports/b"])

    def test_download_hash_and_no_overwrite(self):
        target = self.root / "result"
        data = b"complete-result"
        expected = hashlib.sha256(data).hexdigest()
        with patch.object(dropbox, "request", return_value=Response(data, len(data))):
            result = dropbox.download("fake", "https://www.dropbox.com/example", "/result", target, expected)
        self.assertEqual(result["sha256"], expected)
        self.assertEqual(target.read_bytes(), data)
        with self.assertRaises(dropbox.AccessError):
            dropbox.download("fake", "unused", "/result", target)

    def test_bad_hash_or_partial_download_does_not_publish(self):
        for index, (size, checksum) in enumerate([(99, None), (3, "0" * 64)]):
            target = self.root / str(index)
            with patch.object(dropbox, "request", return_value=Response(b"abc", size)):
                with self.assertRaises(dropbox.AccessError):
                    dropbox.download("fake", "unused", "/result", target, checksum)
            self.assertFalse(target.exists())
            self.assertEqual(list(self.root.glob("*.part")), [])

    def test_existing_dangling_symlink_is_not_overwritten(self):
        target = self.root / "link"
        target.symlink_to(self.root / "missing")
        with self.assertRaises(dropbox.AccessError):
            dropbox.download("fake", "unused", "/result", target)


if __name__ == "__main__":
    unittest.main()
