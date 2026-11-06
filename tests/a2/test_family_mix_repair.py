import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer,models,pre_tokenizers
from efficient_llm.a2_family.common import read,write,sha,SPEC_SHA
from scripts.a2_family_prepare_mix import prepare_source,read_texts
from scripts.a2_family_repair_mix import repair,verify_source,corrected_plan


class MixtureRepair(unittest.TestCase):
    def test_production_corrections_keep_weights_and_exact_historical_dclm_prefix(self):
        old=read('resources/a2_family_mixture_plan_20261107.json')
        fix=read('resources/a2_family_mixture_content_correction_20261107.json')
        new=corrected_plan(old,fix)
        self.assertEqual([(e['name'],e['effective_weight']) for e in old['sources']],[(e['name'],e['effective_weight']) for e in new['sources']])
        a=next(e for e in old['sources'] if e['name']=='dclm');b=next(e for e in new['sources'] if e['name']=='dclm')
        self.assertEqual(b['files'],a['files'][:216]);self.assertEqual(len(b['files']),216)
        for e in new['sources']:
            if e['name'].startswith('stack-edu-'):
                self.assertEqual(e['text_columns'],['content'])
                self.assertTrue(e['repo']!='HuggingFaceTB/stack-edu')
    def test_actual_parquet_language_filter_keeps_original_row_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'code.parquet'
            pq.write_table(pa.table({'content':['javascript','python1','python2'],'path':['x.js','y.py','z.py']}),p)
            rows=list(read_texts([p],['content'],['.py']))
            self.assertEqual([(r[1],r[2]) for r in rows],[(1,'python1'),(2,'python2')])
            with self.assertRaises(ValueError):list(read_texts([p],['text'],['.py']))

    def test_repair_reuses_good_bytes_and_rebuilds_exact_weighted_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();cx=root/'cx';primary=root/'primary'
            tok=Tokenizer(models.WordLevel(vocab={'[EOS]':0,'[UNK]':1,'a':2},unk_token='[UNK]'))
            tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(root/'tok.json'))
            for split in ('tuning','final'):
                d=write(cx/split/'documents.json',dict(documents=[]))
                write(cx/split/'manifest.json',dict(documents_sha256=d['sha256']))
            write(primary/'complete.json',dict(status='complete'))
            for seed in ('2030','2031'):
                d=primary/seed;d.mkdir();(d/'consumed_documents.jsonl').write_text('')
                write(d/'manifest.json',dict(streams={'heal':{'exposure_sha256':sha(d/'consumed_documents.jsonl')}}))
            sources=[]
            for name,column in [('good','text'),('stackexchange','ThreadText')]:
                raw=root/(name+'.parquet');pq.write_table(pa.table({column:[' '.join(['a']*127)]*40}),raw)
                sources.append(dict(name=name,effective_weight=.5,weight=.5,text_columns=['text'],files=[dict(local=str(raw),path=raw.name,bytes=raw.stat().st_size,sha256=sha(raw))]))
            plan=dict(spec_sha256=SPEC_SHA,tokenizer={'local':str(root/'tok.json')},eos_id=0,primary_fwe_root=str(primary),sources=sources,prepared_tokens=8192,flags=[],missing_sources=[])
            pp=root/'plan.json';write(pp,plan)
            good=prepare_source(plan,sources[0],old/'sources/good',4096,set());token_path=Path(read(good['path'])['token_file']);before=sha(token_path)
            error={'stackexchange':'ValueError: No registered text column'}
            write(old/'progress.json',dict(ready={'good':good},failed=error))
            write(old/'complete.json',dict(status='incomplete',ready={'good':good},failed=error))
            fix=dict(original_plan_sha256=sha(pp),source_overrides={'stackexchange':dict(text_columns=['ThreadText'])},download_groups={})
            fp=root/'fix.json';write(fp,fix)
            repair(pp,old,root/'new',cx,fp)
            final=read(root/'new/complete.json');stream=read(final['stream']['path'])
            self.assertEqual(final['status'],'complete');self.assertEqual(final['sources']['good'],good)
            self.assertEqual(sha(token_path),before);self.assertEqual(stream['shares'],{'good':.5,'stackexchange':.5})
            self.assertTrue(all(c['passed'] for c in stream['checks']))
            self.assertEqual(stream['streams']['heal']['tokens'],8192)
            self.assertEqual(read(final['effective_plan']['path']),corrected_plan(read(pp),fix))
            token_path.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_source(good,4096)
            fix['source_overrides']['stackexchange']['effective_weight']=.7
            with self.assertRaises(ValueError):corrected_plan(read(pp),fix)
