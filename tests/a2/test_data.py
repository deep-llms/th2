import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from efficient_llm.a2.data import prepare,PackedRows

class Tokenizer:
    eos_token_id=0
    def __call__(self,texts,**kwargs):
        assert kwargs==dict(add_special_tokens=False,truncation=False)
        return {'input_ids':[[ord(c) for c in t] for t in texts]}

class DataTests(unittest.TestCase):
    def prepare(self,path,train=None,validation=None,**kw):
        train=train or [{'text':f'doc-{i}'} for i in range(30)]
        validation=validation or [{'text':f'heldout-{i}'} for i in range(6)]
        return prepare(train,validation,Tokenizer(),Tokenizer(),path,provenance={'fixture':True},
            length=4,reserved_qwen=80,heal_tokens=31,tune_rows=3,small_rows=2,**kw)

    def test_reproducible_continuous_packing(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            ma=self.prepare(a,batch_size=1); mb=self.prepare(b,batch_size=7)
            self.assertEqual(ma,mb)
            records=[json.loads(line) for line in (a/'documents.jsonl').read_text().splitlines()]
            reserved=[r['index'] for r in records if r['split']=='train']
            self.assertEqual(reserved,np.random.default_rng(2028).permutation(30)[:len(reserved)].tolist())
            heal=[r for r in records if r['group']=='heal']
            expected=[token for r in heal for token in [*[ord(c) for c in f'doc-{r["index"]}'],0]]
            rows=PackedRows(a,'heal')
            actual=[int(x) for i in range(len(rows)) for x in rows[i]['input_ids']]
            self.assertEqual(actual,expected[:len(actual)])
            self.assertLess(len(expected)-len(actual),4)
            self.assertEqual(ma['streams']['tuning']['rows'],3)
            self.assertEqual(len(set(ma['small_indices'])),2)
            with self.assertRaises(FileExistsError): self.prepare(a)
            with (a/'heal.bin').open('r+b') as f: f.write(b'xxxx')
            with self.assertRaises(ValueError): PackedRows(a,'heal')

    def test_overlap_fails(self):
        with tempfile.TemporaryDirectory() as d:
            index=int(np.random.default_rng(2028).permutation(30)[0])
            with self.assertRaisesRegex(ValueError,'overlap'):
                self.prepare(Path(d)/'a',validation=[{'text':f'doc-{index}'}]*8)

    def test_indexed_counts_and_secondary(self):
        from efficient_llm.a2.data import prepare_secondary
        class Indexed(list):column_names=['text','a2_qwen_tokens_with_eod']
        train=Indexed([{'text':f'doc-{i}','a2_qwen_tokens_with_eod':len(f'doc-{i}')+1} for i in range(30)])
        with tempfile.TemporaryDirectory() as d:
            result=self.prepare(Path(d)/'indexed',train=train)
            self.assertEqual(result['qwen_counts_source'],'verified_published_index')
            train[0]['a2_qwen_tokens_with_eod']+=1
            with self.assertRaisesRegex(ValueError,'convention differ'):
                self.prepare(Path(d)/'bad-index',train=train)
            secondary=prepare_secondary([{'text':'abcde'},{'text':'xyz'}],Tokenizer(),Path(d)/'secondary',
                provenance={'fixed_shard':'fixture'},length=4,rows=2)
            self.assertEqual(secondary['streams']['final']['tokens'],8)
            rows=PackedRows(Path(d)/'secondary','final')
            self.assertEqual(rows[0]['input_ids'].tolist(),[97,98,99,100])
            self.assertEqual(rows[1]['input_ids'].tolist(),[101,0,120,121])

    def test_published_pool_layout(self):
        from datasets import Dataset
        import pyarrow as pa
        import pyarrow.parquet as pq
        from efficient_llm.a2.data import load_pool,sha256
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);subset=root/'subsets/qwen3_0.6b_base_en_30B';manifest={'shards':{}}
            for split in ('train','validation'):
                manifest['shards'][split]=[]
                for shard in range(2):
                    path=subset/split/f'shard_{shard:06}'
                    Dataset.from_dict({'text':[f'{split}-{shard}']}).save_to_disk(path)
                    count=len(f'{split}-{shard}')
                    pq.write_table(pa.table({'content_tokens':[count],'tokens_with_eod':[count+1]}),path/'document_index.parquet')
                    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in path.iterdir() if p.is_file()}
                    manifest['shards'][split].append({'files':files})
            (subset/'manifest.json').write_text(json.dumps(manifest))
            pool,_=load_pool(root)
            self.assertEqual(pool['train']['text'],['train-0','train-1'])
            self.assertEqual(pool['train']['a2_qwen_tokens_with_eod'],[8,8])

    def test_t14_retry_preserves_permutation(self):
        from efficient_llm.a2.data import training_order
        count=12160
        changed=training_order(count,'-t14b')
        self.assertFalse(np.array_equal(changed,np.arange(count)))
        self.assertTrue(np.array_equal(changed,training_order(count,'-t14b-rerun')))
        self.assertTrue(np.array_equal(np.sort(changed),np.arange(count)))
        self.assertTrue(np.array_equal(training_order(count,'-t14a-rerun'),np.arange(count)))


    def test_unused_overlap_does_not_change_registered_split(self):
        order=np.random.default_rng(2028).permutation(30)
        train=[{'text':f'doc-{i}'} for i in range(30)]
        train[int(order[9])]=dict(train[int(order[0])])
        with tempfile.TemporaryDirectory() as d:
            result=self.prepare(Path(d)/'a',train=train)
            self.assertGreater(result['counts']['unused'],0)
            self.assertTrue(result['disjoint_by_id_and_content'])
