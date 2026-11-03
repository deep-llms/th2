import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficient_llm.a2_depth.common import SPEC_SHA, read, write
from efficient_llm.a2_depth.reconstruction import arm_order, rebuilt_manifest
from efficient_llm.a2_depth.stream import verify


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReconstructionTests(unittest.TestCase):
    def fixture(self, root):
        import numpy as np
        stream = root/'stream'; stream.mkdir()
        (stream/'heal.bin').write_bytes(np.arange(40,dtype='<u4').tobytes())
        (stream/'heal.starts').write_bytes(bytes([1,0,0,0])*10)
        m = dict(status='complete',streams={'heal':dict(rows=10,length=4,
            token_sha256=digest(stream/'heal.bin'),starts_sha256=digest(stream/'heal.starts'))},
            tokenizer_sha256='tok',eos_id=0,order='seed2030',excluded_original_tail_tokens=0,
            dropped_after_shuffle=0,document_count=10,documents_sha256='docs',
            document_order_sha256='order',heldout_documents_sha256='held',document_dtype=[],
            T7={'passed':True},T18={'sequences':10})
        write(stream/'manifest.json',m)
        history=dict(data_manifest=copy.deepcopy(read(stream/'manifest.json')),
                     provenance={'data_sha256':digest(stream/'manifest.json')})
        write(stream/'manifest.json',dict(m,elapsed_seconds=17,new_path_provenance='new node'))
        receipt=dict(status='verified',spec_sha256=SPEC_SHA,
            original_manifest_sha256=history['provenance']['data_sha256'],
            new_manifest_sha256=digest(stream/'manifest.json'),stream=str(stream),
            full_and_prefix_content_equal=True,document_order_equal=True,
            selected_sequences=3124992,prefix_sequences=781248,
            tokens={'full_sha256':m['streams']['heal']['token_sha256']},
            boundaries={'full_sha256':m['streams']['heal']['starts_sha256']})
        path=root/'rebuild.json';write(path,receipt)
        return stream,history,dict(path=str(path),sha256=digest(path))

    def test_new_provenance_keeps_old_anchor_and_checks_physical_tail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stream,history,lock=self.fixture(root);original=copy.deepcopy(history)
            good=verify(stream,history,root/'good',prefix_rows=3,selected_rows=8,reconstruction=lock)
            self.assertTrue(good['passed']);self.assertEqual(history,original)
            self.assertEqual(good['historical_manifest_sha256'],history['provenance']['data_sha256'])
            self.assertNotEqual(good['rebuilt_manifest_sha256'],good['historical_manifest_sha256'])
            data=bytearray((stream/'heal.bin').read_bytes());data[-1]^=1;(stream/'heal.bin').write_bytes(data)
            bad=verify(stream,history,root/'bad',prefix_rows=3,selected_rows=8,reconstruction=lock)
            self.assertFalse(bad['passed'])
            self.assertEqual(bad['reader_prefix_sha256'],good['reader_prefix_sha256'])

    def test_changed_receipt_and_rebound_different_document_order_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stream,history,lock=self.fixture(root)
            p=Path(lock['path']);receipt=read(p);receipt['original_manifest_sha256']='wrong';write(p,receipt)
            with self.assertRaises(ValueError):rebuilt_manifest(stream,history,lock)
            lock['sha256']=digest(p)
            with self.assertRaises(ValueError):rebuilt_manifest(stream,history,lock)
            receipt['original_manifest_sha256']=history['provenance']['data_sha256']
            manifest=read(stream/'manifest.json');manifest['document_order_sha256']='changed';write(stream/'manifest.json',manifest)
            receipt['new_manifest_sha256']=digest(stream/'manifest.json');write(p,receipt);lock['sha256']=digest(p)
            with self.assertRaisesRegex(ValueError,'document_order'):rebuilt_manifest(stream,history,lock)

    def test_x72_first_notes_and_all_arms_then_downstream(self):
        import scripts.a2_depth_partc_run as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=dict(results_root=d,heal_streams={'dclm':{'path':'fixture'}},stream_history={},
                arm_order=['X72','P300','P1T'],arm_order_authorization='User request 03Nov2026')
            arms=[];notes=[]
            def run_arm(cfg,config,arm,checks):arms.append(arm);return [dict(arm=arm,returncode=0)]
            def finalize(cfg,**kw):notes.append(kw.get('note_arm'));return dict(coverage_complete=True)
            with patch('efficient_llm.a2_depth.stream.verify',return_value={'passed':True}),patch.object(runner,'run_arm',side_effect=run_arm),patch('scripts.a2_depth_partc_report.finalize',side_effect=finalize),patch.object(runner,'require_free_gpus'),patch('scripts.a2_depth_downstream.run',return_value={'coverage_complete':True}) as downstream:
                runner.execute(cfg,'config',root/'done.json')
            self.assertEqual(arms,['X72','P300','P1T']);self.assertEqual(notes,arms+[None]);downstream.assert_called_once()
            self.assertTrue(read(root/'done.json')['training_coverage_complete'])

    def test_invalid_coverage_and_failed_reconstruction_never_start_an_arm(self):
        import scripts.a2_depth_partc_run as runner
        for order in (['X72'],['X72','P300','P300'],['P1T','X72','P300']):
            with self.assertRaises(ValueError):arm_order({'arm_order':order,'arm_order_authorization':'user'})
        with self.assertRaises(ValueError):arm_order({'arm_order':['X72','P300','P1T']})
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(results_root=d,heal_streams={'dclm':{'path':'fixture'}},stream_history={},stream_reconstruction={})
            with patch('efficient_llm.a2_depth.stream.verify',return_value={'passed':False}),patch.object(runner,'run_arm') as arm:
                with self.assertRaisesRegex(ValueError,'failed DT6'):runner.execute(cfg,'config',Path(d)/'done.json')
                arm.assert_not_called()
