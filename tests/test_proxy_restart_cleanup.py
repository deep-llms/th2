import os
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
from scripts.clean_proxy_screen_restart import delete_caches, inventory, require_unused


class CleanupTests(unittest.TestCase):
    def setUp(self):
        # Dev host has other users; exercise real open-file checks on our process.
        patcher = patch("scripts.clean_proxy_screen_restart.require_unused",
                        side_effect=lambda roots: require_unused(roots, [Path("/proc")/str(os.getpid())]))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_generated_cache_files_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/'shard').mkdir()
            for name in ('data-00000.arrow','state.json','cache-a.arrow','tmp-b'):
                (root/'shard'/name).write_text('source or cache')
            _, sources = inventory([root])
            deleted = delete_caches([root], sources)
            self.assertEqual(len(deleted), 2)
            self.assertEqual(sorted(p.name for p in (root/'shard').iterdir()), ['data-00000.arrow','state.json'])

    def test_changed_source_refuses_all_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, cache = root/'data-0.arrow', root/'cache-a.arrow'
            source.write_text('a'); cache.write_text('b')
            _, sources = inventory([root])
            source.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'inventory changed'):
                delete_caches([root], sources)
            self.assertTrue(cache.exists())

    def test_symlink_and_open_file_refuse_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, cache = root/'data-0.arrow', root/'cache-a.arrow'
            source.write_text('a'); cache.write_text('b')
            _, sources = inventory([root])
            cache.unlink(); cache.symlink_to(source)
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                delete_caches([root], sources)
            cache.unlink(); cache.write_text('b')
            with cache.open():
                with self.assertRaisesRegex(RuntimeError, 'Open file'):
                    delete_caches([root], sources)
            self.assertTrue(cache.exists())
