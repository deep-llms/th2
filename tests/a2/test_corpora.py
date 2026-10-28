import hashlib,json,struct,tempfile,time,unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from scripts.a2_prepare_corpora import group_for,shard_worker,sha,assemble,verify_packed

TOKENIZER=Path('temp/a2_neox_tokenizer')

class CorpusTests(unittest.TestCase):
    def fixture(self,root,index,texts):
        raw=root/f'raw{index}.parquet';pq.write_table(pa.table({'text':texts}),raw)
        record=dict(index=index,path=raw.name,local=str(raw),bytes=raw.stat().st_size,sha256=sha(raw))
        out=root/f'part{index}'
        task=(record,TOKENIZER/'tokenizer.json',sha(TOKENIZER/'tokenizer.json'),out,time.monotonic()+10)
        return task,shard_worker(task)

    def test_actual_tokenizer_packing_split_resume_and_tamper(self):
        texts={g:[] for g in ('heal','tuning','final')}
        i=0
        while min(map(len,texts.values()))<3:
            text=f'Corpus fixture document number {i}. Some Unicode: café 世界.';g,_=group_for(text)
            if len(texts[g])<3:texts[g].append(text)
            i+=1
        rawtexts=[texts[g][j] for j in range(3) for g in texts]
        rawtexts+=['',texts['heal'][0],'<|endoftext|> hello']
        hf=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parts=[]
            for index,rows in enumerate((rawtexts[:4],rawtexts[4:])):
                task,saved=self.fixture(root,index,rows);parts.append((task[3],saved))
                self.assertEqual(shard_worker(task),saved)
                expected={g:[] for g in texts};starts={g:[] for g in texts}
                for t in rows:
                    if not t.strip():continue
                    g,_=group_for(t);ids=hf(t,add_special_tokens=False,truncation=False)['input_ids']+[0]
                    expected[g]+=ids;starts[g]+=[1]+[0]*(len(ids)-1)
                for g in texts:
                    np.testing.assert_array_equal(np.fromfile(task[3]/(g+'.bin'),dtype='<u4'),expected[g])
                    np.testing.assert_array_equal(np.fromfile(task[3]/(g+'.starts'),dtype='u1'),starts[g])
                self.assertEqual((task[3]/'documents.bin').stat().st_size,49*sum(bool(t.strip()) for t in rows))
            plan=dict(length=8,train_tokens=65,tuning_rows=2,final_rows=2,tokenizer_files={'tokenizer.json':sha(TOKENIZER/'tokenizer.json')})
            out=root/'packed';m=assemble(parts,out,plan,'fixture','fixed-plan')
            self.assertEqual(m['streams']['heal']['tokens'],64)
            for g in texts:
                want=np.concatenate([np.fromfile(p/(g+'.bin'),dtype='<u4') for p,_ in parts])[:m['streams'][g]['tokens']]
                np.testing.assert_array_equal(np.fromfile(out/(g+'.bin'),dtype='<u4'),want)
            self.assertEqual(assemble(parts,out,plan,'fixture','fixed-plan'),m)
            (out/'heal.bin').write_bytes(b'\x01'+(out/'heal.bin').read_bytes()[1:])
            with self.assertRaises(ValueError):verify_packed(out,m)
            p=parts[0][0]/'heal.bin';p.write_bytes(b'bad')
            # Completed-shard reuse must check bytes, not trust marker existence.
            first_task=(parts[0][1]['source'],TOKENIZER/'tokenizer.json',sha(TOKENIZER/'tokenizer.json'),parts[0][0],time.monotonic()+10)
            with self.assertRaises(ValueError):shard_worker(first_task)

    def test_partition_is_content_based_and_incomplete_source_refused(self):
        self.assertEqual(group_for('same text'),group_for('same text'))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'raw';raw.write_bytes(b'bad')
            record=dict(index=0,local=str(raw),bytes=3,sha256='0'*64)
            with self.assertRaises(ValueError):shard_worker((record,TOKENIZER/'tokenizer.json',sha(TOKENIZER/'tokenizer.json'),root/'out',time.monotonic()+5))
            self.assertFalse((root/'out').exists())


    def test_real_cpu_process_pool_to_native_reader(self):
        import subprocess,sys
        found={}
        i=0
        while len(found)<3:
            text=('This is a document for offline corpus preparation. '*1600)+str(i)
            group,_=group_for(text)
            found.setdefault(group,text);i+=1
        with tempfile.TemporaryDirectory(dir='temp') as d:
            root=Path(d).resolve();raw=root/'source.parquet';pq.write_table(pa.table({'text':list(found.values())}),raw)
            record=dict(index=0,path='source.parquet',local=str(raw),bytes=raw.stat().st_size,sha256=sha(raw))
            plan=dict(root=str(root/'data'),length=4096,train_tokens=8192,tuning_rows=2,final_rows=2,eos_id=0,
                tokenizer_path=str(TOKENIZER.resolve()),tokenizer_files={'tokenizer.json':sha(TOKENIZER/'tokenizer.json')},
                corpora={'fixture':dict(workers=1,files=[record])})
            path=root/'plan.json';path.write_text(json.dumps(plan))
            r=subprocess.run([sys.executable,'-m','scripts.a2_prepare_corpora','--plan',str(path)],capture_output=True,text=True,timeout=90)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            ready=json.loads((root/'data/complete.json').read_text());self.assertEqual(ready['status'],'complete')
            self.assertEqual(ready['corpora']['fixture']['streams']['heal']['tokens'],8192)
