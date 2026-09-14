import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("export_results", Path(__file__).parents[1]/"scripts/export_pilot_results.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class BundleTests(unittest.TestCase):
    def test_snapshot_hashes_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            def put(p, text='{"success": true}'):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            for name in export.RUNS:
                root = base/name
                put(root/"complete.json")
                put(root/"validated_common.json")
                put(root/"model.pt", "must not export")
                put(root/"status.next.json", "partial")
                put(base/(name+".handoff.log"), "snapshot log")
            for stage, arms in ((1, ("base","contextual","isolated","shuffled","shallow","delta","grad")),
                                (2, ("base","contextual","isolated","shuffled","grad"))):
                root = base/f"ccm_stage{stage}_seed17_20260913_a01"
                put(root/"validated_panel.json")
                for arm in arms:
                    put(root/"eval"/arm/"metrics.json")
                    put(root/"eval"/arm/"segments.jsonl")
                    put(root/f"validated_eval_{arm}.json")
                for right in ("base","grad","isolated","shuffled"):
                    put(root/"reports"/f"contextual_vs_{right}.json")
            out = base/"export"
            export.pack(base, out)
            m = json.loads((out/"manifest.json").read_text())
            pieces = b"".join((out/p["path"]).read_bytes() for p in m["parts"])
            self.assertEqual(hashlib.sha256(pieces).hexdigest(), m["archive_sha256"])
            with tarfile.open(out/"results.tar.gz") as t:
                self.assertEqual(set(t.getnames()), {x["path"] for x in m["files"]})
                for item in m["files"]:
                    data = t.extractfile(item["path"]).read()
                    self.assertEqual(len(data), item["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
                    self.assertFalse(item["path"].endswith((".pt", ".next.json")))
            with self.assertRaises(FileExistsError):
                export.pack(base, out)

    def test_missing_run_refused_without_creating_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises(ValueError):
                export.pack(base, base/"export")
            self.assertFalse((base/"export").exists())
