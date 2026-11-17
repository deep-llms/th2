import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.a2_dot_rebuild_data import (recipe, plan_for, read, download_command,
    repack_e2, verify_generated, verify_corpus)
from scripts.a2_prepare_corpora import sha


class RebuildTests(unittest.TestCase):
    def test_source_selection_and_download_coverage(self):
        cfg = recipe(); old = read(cfg['plan']); urls = []
        for corpus, n in cfg['raw_counts'].items():
            plan = plan_for(cfg, corpus)
            for a, b in zip(plan['corpora'][corpus]['files'], old['corpora'][corpus]['files'][:n], strict=True):
                self.assertEqual({k:v for k,v in a.items() if k!='local'}, {k:v for k,v in b.items() if k!='local'})
            self.assertEqual(plan['train_tokens'], old['train_tokens'])
            self.assertEqual(plan['tokenizer_files'], old['tokenizer_files'])
            urls += [f['url'] for f in old['corpora'][corpus]['files'][:n]]
        found = []
        for i in range(26):
            cmd = download_command(cfg, i)
            self.assertNotIn('\n\n', cmd)
            raw = cmd.split('#datasets\n')[1].splitlines()
            self.assertLessEqual(len(raw), 10)
            found += [line.split()[1] for line in raw]
        self.assertEqual(found, urls)
        self.assertEqual(len(set(found)), 260)
        with self.assertRaises(ValueError): download_command(cfg, 26)

    def test_repack_keeps_explicit_boundaries_and_seed_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src=root/'source';src.mkdir()
            ids=np.arange(5*512, dtype='<u4').reshape(5,512)%300+1
            ids[:, 10]=0  # Literal EOS tokens without document starts must stay non-boundaries.
            starts=np.zeros_like(ids,dtype='u1');starts[0,0]=1
            ids[:,99]=0;starts[:,100]=1
            ids.tofile(src/'heal.bin');starts.tofile(src/'heal.starts')
            result=repack_e2(src,root/'out',[('train','heal',4*512+9)])['train']
            order=np.random.default_rng(0).permutation(4)
            got=np.fromfile(root/'out/train.bin',dtype='<u4').reshape(4,512)
            bound=np.fromfile(root/'out/train.starts',dtype='u1').reshape(4,512)
            expected=starts[order].copy();expected[:,0]=1
            np.testing.assert_array_equal(got,ids[order]);np.testing.assert_array_equal(bound,expected)
            self.assertTrue(np.all(bound[:,11]==0));self.assertEqual(result['dropped_to_512'],9)
            self.assertEqual(result['token_sha256'],sha(root/'out/train.bin'))
            with self.assertRaises(FileExistsError):repack_e2(src,root/'out',[('train','heal',512)])

    def test_historical_hash_and_prefix_fail_closed(self):
        cfg=recipe(); expected=read(cfg['historical_dclm'])['corpus_expected']
        verify_corpus(expected, expected)
        bad=copy.deepcopy(expected);bad['streams']['heal']['token_sha256']='0'*64
        with self.assertRaises(ValueError):verify_corpus(bad,expected)
        bad=copy.deepcopy(expected);bad['source_prefixes'][0]['tokens']+=1
        with self.assertRaises(ValueError):verify_corpus(bad,expected)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'good');expect={'x':{'bytes':4,'sha256':sha(p)}}
            verify_generated(d,expect);p.write_bytes(b'evil')
            with self.assertRaises(ValueError):verify_generated(d,expect)


if __name__=='__main__':unittest.main()
