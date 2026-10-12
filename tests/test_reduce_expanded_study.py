"""Single-seed continuation preserves completed fits and rejects changed evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from run_experiments import load_jobs, verify_outputs
from scripts.expanded_evaluation_study import make
from scripts.finetune_study import make_jobs, select
from scripts.reduce_expanded_study import build, digest, verify_reuse


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class ReducedStudyTests(unittest.TestCase):
    def test_single_seed_fixed_rate_and_explicit_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jobs=make_jobs(root/'plan',Path.cwd(),'/data','/manifest',{'A':'/A'},
                           tasks=('glue_cola',),seeds=(42,),rates=(1e-5,))
            names=[j['name'] for j in jobs]
            self.assertEqual([n for n in names if n.startswith('search-')],['search-glue_cola-A-1e-05'])
            self.assertFalse(any(n.startswith('confirm-') for n in names))
            self.assertEqual([n for n in names if n.startswith('test-')],['test-glue_cola-A-42'])
            self.assertEqual(jobs[-1]['argv'][-2:],['--seeds','42'])
            result=dict(status='completed',smoke=False,evaluate_run=None,learning_rate=1e-5,
                        data={'task':'glue_cola'},source={'arm':'A'},seed=42,global_step=10,world_size=8,
                        max_length=512,attention_backend='fa4',metrics={'eval_matthews_correlation':.2})
            put(root/'fit/result.json',result)
            with self.assertRaises(ValueError):select([str(root/'fit')],root/'bad.json')
            select([str(root/'fit')],root/'selected.json',rates=(1e-5,))
            self.assertEqual(json.loads((root/'selected.json').read_text())['learning_rate'],1e-5)
            with self.assertRaises(ValueError):
                make_jobs(root/'invalid',Path.cwd(),'/d','/m',{'A':'/A'},seeds=(43,),rates=(1e-5,))

    def fixture(self, root):
        src=root/'old';run=src/'supervised/run';fit=run/'finetune'
        make(src,Path.cwd(),dict(A10='/A10',A20='/A20',P20='/P20'),'/data','/manifest','/protocol','/position',None)
        outer=load_jobs(src/'jobs.json');inner=load_jobs(src/'finetune-plan/jobs.json')
        put(run/'jobs.snapshot.json',dict(jobs=outer));put(fit/'jobs.snapshot.json',dict(jobs=inner))
        def completed(job, folder):
            for spec in job['required_outputs']:
                put(folder/spec['path'],spec.get('json_equals',{}))
            return dict(name=job['name'],status='ok',returncode=0,artifacts=verify_outputs(job,folder))
        put(run/'run.json',dict(status='failed',jobs=[completed(j,run) for j in outer[:4]]))
        records=[]
        for j in inner:
            if j['name'].startswith(('gate-','smoke-','verify-reload-')):
                records.append(completed(j,fit))
            elif j['name']=='search-glue_cola-A-1e-05':
                result=fit/j['required_outputs'][0]['path'];weights=result.parent/'best/model.safetensors'
                weights.parent.mkdir(parents=True);weights.write_bytes(b'fixture weights')
                put(result,dict(status='completed',seed=42,learning_rate=1e-5,smoke=False,
                                evaluate_run=None,weight_sha256=digest(weights)))
                records.append(dict(name=j['name'],status='ok',returncode=0,artifacts=verify_outputs(j,fit)))
            elif j['name']=='search-glue_cola-A-3e-05':
                records.append(dict(name=j['name'],status='failed'))
        put(fit/'run.json',dict(status='failed',jobs=records))
        terminal=dict(finished_at='fixture',training_status='failed',burn={})
        put(src/'supervised/supervisor.json',terminal)
        put(src/'reduce-seed42-stop.json',dict(previous_supervisor_final=terminal,
                                             burn_handoff=dict(collective_progress_verified=True)))
        return src

    def test_reuse_and_failed_output_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=self.fixture(root);dest=root/'new'
            result=build(src,dest,Path.cwd())
            self.assertEqual(result['remaining_fits'],19)
            jobs=load_jobs(dest/'finetune-plan/jobs.json');names=[j['name'] for j in jobs]
            self.assertFalse(any(n.startswith(('gate-','smoke-','verify-reload-','confirm-')) for n in names))
            self.assertNotIn('search-glue_cola-A-1e-05',names)
            self.assertNotIn('search-glue_cola-A-3e-05',names)
            self.assertEqual(sum(n.startswith('test-') for n in names),22)
            selector=next(j for j in jobs if j['name']=='select-glue_cola-A')
            self.assertIn(str(src/'supervised/run/finetune/search/glue_cola/A/lr-1e-05'),selector['argv'])
            self.assertEqual(load_jobs(dest/'jobs.json')[0]['name'],'verify-reused-artifacts')
            verify_reuse(dest/'reuse.json',root/'verified.json')
            weights=src/'supervised/run/finetune/search/glue_cola/A/lr-1e-05/best/model.safetensors'
            weights.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Reused artifact changed'):
                verify_reuse(dest/'reuse.json',root/'invalid.json')
            with self.assertRaisesRegex(ValueError,'Invalid reusable fit'):
                build(src,root/'new2',Path.cwd())


if __name__=='__main__':unittest.main()
