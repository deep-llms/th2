"""Real harness few-shot prompts, truncation audit and cross-arm validation."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from eval.benchmarks import eval_benchmarks, local_dataset_paths
from eval.eval_checkpoint import json_default
from eval.models import EvaluationModel
from scripts.check_downstream_eval import check_fewshot, validate
from tests.test_eval import make_model
from tests.test_train import fixture


class FewshotTests(unittest.TestCase):
    def test_audit_matches_real_harness_and_rejects_changed_prompts(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root)
            data=root/'data';data.mkdir();files=[];splits={}
            for split,n in [('train',30),('validation',3)]:
                path=data/(split+'.parquet')
                Dataset.from_list([dict(goal=('2 '*12)+str(i),sol1='3 4',sol2='5 6',label=i%2) for i in range(n)]).to_parquet(path)
                files.append(dict(path=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                splits[split]=[path.name]
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,repositories=[
                dict(path='.',tasks=['piqa'],files=files,load_configs={'piqa':dict(loader='parquet',data_files=splits)})])))
            manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
            data_check=root/'data-check.json';data_check.write_text(json.dumps(dict(tasks={'piqa':{'rows':3}},manifest_sha256=manifest_hash)))
            tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
            for shots in (5,10,25):
                audit_path=root/f'audit-{shots}.json'
                args=argparse.Namespace(dataset_root=str(data),dataset_manifest=str(manifest),tokenizer=str(root/'model'),
                    tasks=['piqa'],num_fewshot=shots,seed=42,max_length=8,smoke_limit=2,output=str(audit_path))
                check_fewshot(args)
                audit=json.loads(audit_path.read_text())
                self.assertEqual(audit['tasks']['piqa']['truncated_documents'],3)
                for limit in (None,2):
                    comparison=root/f'eval-{shots}-{limit}';comparison.mkdir();mapping={}
                    for arm in ('A','P6-iso'):
                        model=EvaluationModel(make_model(arm),8)
                        result=eval_benchmarks(model,tokenizer,['piqa'],num_fewshot=shots,batch_size=2,
                            device='cpu',english_only=True,seed=42,limit=limit,dataset_paths=local_dataset_paths(data,manifest))
                        dest=comparison/arm;dest.mkdir();mapping['/'+arm]=str(dest)
                        (dest/'eval_metadata.json').write_text(json.dumps(dict(status='completed',step=2500,arm=arm,
                            checkpoint='/'+arm,attention_backend='fa4',context_length=8,dataset_manifest_sha256=manifest_hash,
                            arguments=dict(limit=limit,seed=42,num_fewshot=shots))))
                        (dest/'eval_benchmarks.json').write_text(json.dumps(result['results'],default=json_default))
                        (dest/'eval_benchmarks_full.json').write_text(json.dumps({k:v for k,v in result.items() if k!='samples'},default=json_default))
                        (dest/'eval_samples.jsonl').write_text('\n'.join(json.dumps(dict(task='piqa',**s),default=json_default) for s in result['samples']['piqa']))
                    (comparison/'checkpoints.json').write_text(json.dumps(mapping))
                    check=argparse.Namespace(directory=str(comparison),data_check=str(data_check),count=2,limit=limit,
                        seed=42,num_fewshot=shots,fewshot_audit=str(audit_path),output=str(comparison/'passed.json'))
                    validate(check)
                    self.assertEqual(json.loads((comparison/'passed.json').read_text())['status'],'passed')
                    p=comparison/'P6-iso/eval_samples.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
                    rows[0]['arguments'][1][0]+=' changed second-choice prompt'
                    p.write_text('\n'.join(json.dumps(x) for x in rows))
                    check.output=str(comparison/'bad.json')
                    with self.assertRaisesRegex(ValueError,'Prompt audit mismatch'):validate(check)


if __name__=='__main__':unittest.main()
