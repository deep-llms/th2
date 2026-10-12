"""Offline tests of the separate paired protocol with real harness artifacts."""
import copy
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_train import fixture  # CPU/offline environment before HF imports.
from tests.test_eval import make_model
from tests.test_proxy_memory import config as model_config
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from safetensors.torch import save_file
from lm_eval.api.model import LM
from eval.paired import protocol, artifacts, stats
from eval.paired.analysis import precision


class OracleLM(LM):
    def loglikelihood(self,requests):
        return [(-float(sum(map(ord,r.args[0][-8:]+r.args[1]))%43+1)/10,True) for r in requests]
    def loglikelihood_rolling(self,requests):return [-len(r.args[0])/3 for r in requests]
    def generate_until(self,requests):raise AssertionError('No generative paired tasks')


def snapshot(root,name,rows,splits):
    path=root/(name+'.parquet');Dataset.from_list(rows).to_parquet(path)
    from eval.models import file_hash
    return dict(path='.',tasks=[name],files=[dict(path=path.name,bytes=path.stat().st_size,sha256=file_hash(path))],
                load_configs={name:dict(loader='parquet',data_files={s:[path.name] for s in splits})})


def checkpoint(root,arm='P6-iso'):
    model=make_model(arm);ckpt=root/'checkpoint-20000';ckpt.mkdir()
    tok=AutoTokenizer.from_pretrained(root/'model');tok.save_pretrained(ckpt)
    save_file({k:v.clone() for k,v in model.state_dict().items()},ckpt/'model.safetensors')
    (root/'train_config.json').write_text(json.dumps(dict(model_config=model_config().to_dict(),
        model=dict(tokenizer_name=str(ckpt)),data=dict(block_size=64),training=dict(seed=42,bf16=True),
        pilot=dict(arm=arm,proxy_screen=True,consumer=2,deep_target=8,lm_chunk=3,causal_attention=False,
            **{'proxy_'+k:v for k,v in asdict(model.settings).items()}))))
    return ckpt


class StatisticsTests(unittest.TestCase):
    def test_exact_self_constant_and_nonfinite(self):
        result=stats.paired(np.zeros(10),100,42)
        self.assertEqual(result['p'],1.);self.assertEqual(result['std_effect'],0)
        self.assertEqual(result['ci'],[0.,0.])
        for d in ([1.,1.],np.full(20,.1),[np.nan,0.],[np.inf,0.],[],[1.]):
            with self.assertRaises(ValueError):stats.paired(d,100)
        for n in range(2,10):
            d=np.random.default_rng(n).normal(size=n)
            null=np.array([np.mean(d*s) for s in itertools.product([-1,1],repeat=n)])
            expected=np.mean(np.abs(null)>=abs(d.mean())*(1-1e-12))
            result=stats.paired(d,100,n)
            self.assertEqual(result['p'],expected);self.assertAlmostEqual(result['p_std'],expected)

    def test_cluster_units_and_joint_covariance(self):
        x=np.array([1.,2.,-2.,-3.,4.,8.]);groups=['a','a','b','b','c','c']
        null=np.array([np.mean(x*np.repeat(s,2)) for s in itertools.product([-1,1],repeat=3)])
        result=stats.paired(x,500,42,groups)
        self.assertEqual(result['units'],3)
        self.assertEqual(result['p'],np.mean(np.abs(null)>=abs(x.mean())*(1-1e-12)))
        rng=np.random.default_rng(3);x=rng.normal(2,.1,100);y=x*.3
        result=stats.progress_ratio([(y,x)],500,7)
        np.testing.assert_allclose(result['ci'],[.3,.3],atol=1e-14)
        result=stats.joint_contrast([(x,y),(x,y)],500,7)
        self.assertAlmostEqual(result['contrast'],0.)
        np.testing.assert_allclose(result['ci'],[0.,0.],atol=1e-10)
        with self.assertRaises(ValueError):stats.progress_ratio([(x,y)],100,clusters=[])
        with self.assertRaises(ValueError):stats.joint_contrast([(x,y),(x,y)],100,clusters=[None])
        result=stats.progress_ratio([(x,np.zeros(100))],100)
        self.assertFalse(result['estimable']);self.assertIsNone(result['ratio'])

    def test_story_prefix_components_and_ratio(self):
        texts=['abcde','ab','abcd','xyz','xy','z','ab']
        ids=artifacts.story_clusters(texts)
        self.assertEqual(len(set(ids)),3)
        self.assertEqual(ids[0],ids[1]);self.assertEqual(ids[3],ids[4])
        result=stats.ratio([2.,6.,4.],[1.,3.,2.],100,42)
        self.assertEqual(result['estimate'],2.);self.assertEqual(result['ci'],[2.,2.])
        with self.assertRaises(ValueError):stats.ratio([1.,2.],[0.,2.])

    def test_precision_detects_constant_repeat_offset(self):
        def obj(x):return dict(ids=list(range(10)),sizes={'x':10},rule='ll',records=[dict(identity=i) for i in range(10)],
                              metrics=dict(M1=np.asarray(x)),clusters=None)
        a=obj(np.zeros(10));b=obj(np.linspace(0,.2,10))
        self.assertTrue(precision(a,b,a,b,a,b,1000)['passed'])
        shifted=obj(np.full(10,.01))
        self.assertFalse(precision(a,b,a,b,shifted,b,1000)['passed'])
        self.assertFalse(precision(a,b,shifted,b,shifted,b,1000)['passed'])
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            precision(a,obj(np.full(10,np.nan)),a,b,a,b,1000)

    def test_precision_enforces_numeric_paths(self):
        def obj(x,path):
            return dict(ids=list(range(10)),sizes={'x':10},rule='ll',
                records=[dict(identity=i) for i in range(10)],metrics=dict(M1=np.asarray(x)),clusters=None,
                metadata=dict(checkpoint_sha256='weights',tokenizer_sha256='tok',dataset_manifest_sha256='data',
                    precision_path=path,attention_backend='sdpa',weights_dtype='torch.float32',
                    autocast_dtype='None' if path=='reference' else 'torch.bfloat16',
                    matmul_tf32=False,cudnn_tf32=False,batch_size=2,seed=42,attention_runtime=dict(backend='sdpa')))
        ra=obj(np.zeros(10),'reference');rb=obj(np.linspace(0,.2,10),'reference')
        pa=obj(np.zeros(10),'production');pb=obj(np.linspace(0,.2,10),'production')
        self.assertTrue(precision(ra,rb,pa,pb,pa,pb,1000)['passed'])
        for field,value in [('matmul_tf32',True),('cudnn_tf32',True),('weights_dtype','torch.bfloat16')]:
            changed=copy.deepcopy(ra);changed['metadata'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'reference'):
                precision(changed,rb,pa,pb,pa,pb,1000)
        for field,value in [('precision_path','reference'),('autocast_dtype','None')]:
            changed=copy.deepcopy(pa);changed['metadata'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'production'):
                precision(ra,rb,changed,pb,changed,pb,1000)
        changed=copy.deepcopy(pa);changed['metadata']['matmul_tf32']=True
        with self.assertRaisesRegex(ValueError,'Repeatability'):
            precision(ra,rb,pa,pb,changed,pb,1000)

    def test_position_profiles_joint_reference_and_empty_bucket(self):
        from eval.paired.position import analyze
        from eval.models import file_hash
        frozen=dict(protocol_sha256='fixture',resamples=100,analysis_seed=42,
                    position=dict(documents=20,length=8,buckets=[[1,3],[3,5],[5,8],[8,9]]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directories={};base=np.random.default_rng(3).normal(3,.1,(20,8));base[:,0]=np.nan
            for arm,offset in [('A10',.4),('A20',0.),('P20',-.2)]:
                dest=root/arm;dest.mkdir();directories[arm]=dest
                np.save(dest/'losses.npy',base+offset)
                protocol.write_json(dest/'metadata.json',dict(status='completed',protocol_sha256='fixture',
                    T8_max_abs_error=0.,rows=[dict(document_id=str(i),token_sha256=str(i)) for i in range(20)],
                    tokenizer_sha256='tok',sample_sha256='sample',policy='fixture',losses_sha256=file_hash(dest/'losses.npy'),
                    precision_path='reference',weights_dtype='torch.float32',autocast_dtype='None',softmax_dtype='float32',
                    matmul_tf32=False,cudnn_tf32=False,batch_size=1))
            result=analyze(directories,frozen)
            self.assertIsNone(result['pairs']['P20-A20']['buckets']['8:9'])
            self.assertIsNone(result['difference_of_profiles']['8:9'])
            for bucket in ('1:3','3:5','5:8'):
                self.assertAlmostEqual(result['pairs']['P20-A20']['buckets'][bucket]['normalized_profile']['ratio'],1.)
                self.assertAlmostEqual(result['difference_of_profiles'][bucket]['estimate'],0.)
            original=artifacts.load_json(root/'P20'/'metadata.json')
            changed=copy.deepcopy(original);changed['precision_path']='production'
            (root/'P20'/'metadata.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'numeric settings'):analyze(directories,frozen)
            changed=copy.deepcopy(original);changed['rows'][0]['token_sha256']='changed'
            (root/'P20'/'metadata.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'identities'):analyze(directories,frozen)

    def test_position_native_loss_check_rejects_nan(self):
        from types import SimpleNamespace
        from eval.paired.position import losses
        class Model:
            autocast_dtype=None
            def __call__(self,ids):
                return SimpleNamespace(logits=torch.zeros(*ids.shape,4))
            def wrapped(self,*args,**kwargs):
                return dict(lm_sum=torch.tensor(float('nan')),lm_count=torch.tensor(2))
        with self.assertRaisesRegex(ValueError,'T8'):
            losses(Model(),torch.tensor([[1,2,3]]))


class QueueTests(unittest.TestCase):
    def test_precision_sample_selection_is_frozen_and_stratified(self):
        from types import SimpleNamespace
        from eval.paired.run import freeze_selection
        frozen=dict(precision=dict(per_task=10,per_mmlu_subject=3))
        def inventory(task):return ['subject_a','subject_b'] if task=='mmlu_continuation' else ['arc_easy']
        def load(configs,**kwargs):return {n:SimpleNamespace(eval_docs=list(range(30))) for n in inventory(configs[0])}
        with patch('eval.paired.protocol.PRECISION',['arc_easy','mmlu_continuation']), \
             patch('eval.benchmarks.local_dataset_paths',return_value={}), \
             patch('eval.paired.protocol.configs',side_effect=lambda t,m:[t]), \
             patch('eval.paired.protocol.inventory',side_effect=inventory), \
             patch('eval.paired.protocol.manager',return_value=None), \
             patch('lm_eval.tasks.get_task_dict',side_effect=load):
            selection,sizes=freeze_selection(frozen,'data','manifest')
            self.assertEqual((selection,sizes),freeze_selection(frozen,'data','manifest'))
            self.assertEqual(len(selection['arc_easy']['arc_easy']),10)
            self.assertNotEqual(selection['arc_easy']['arc_easy'],list(range(10)))
            for ids in selection['mmlu_continuation'].values():self.assertEqual(len(set(ids)),3)

    def test_full_suite_report_fixed_sequence_and_progress_reference(self):
        from eval.paired.analysis import suite
        from eval.paired import audit
        from eval.models import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frozen=dict(protocol_sha256='fixture',resamples=100,analysis_seed=42)
            calibration=root/'calibration.json'
            protocol.write_json(calibration,dict(full_protocol=True,passed=True,statistics_sha256=file_hash(stats.__file__),calibration_code_sha256=file_hash(audit.__file__)))
            plan=dict(protocol_sha256='fixture',manifest_sha256='data',calibration=str(calibration),calibration_sha256=file_hash(calibration),
                      checkpoints={a:dict(sha256=a) for a in ('A10','A20','P20')})
            plan['plan_sha256']=protocol.digest(plan);protocol.write_json(root/'plan.json',plan)
            protocol.write_json(root/'precision.json',dict(protocol_sha256='fixture',plan_sha256=plan['plan_sha256'],suite_Z_path='production',tasks={t:dict(passed=True) for t in protocol.PRECISION}))
            def read(directory,task,*args,**kwargs):
                arm=Path(directory).parts[-3];rng=np.random.default_rng(int(protocol.digest(task)[:8],16))
                base=rng.normal(-3,1,80);change=rng.normal(.1,.2,80)
                scores=base+(change if arm=='P20' else -change if arm=='A10' else 0.)
                metrics=dict(M1=scores,ACC=(scores>-3).astype(float))
                if task!='lambada_openai':metrics.update(M2=.3+.01*scores,M3=scores*2)
                return dict(ids=list(range(80)),sizes={task:80},rule=protocol.RULES[task],
                    records=[dict(identity=i) for i in range(80)],metrics=metrics,clusters=None,chance=.25,
                    metadata=dict(checkpoint_sha256=arm,tokenizer_sha256='tokenizer',dataset_manifest_sha256='data',
                                  precision_path='production',attention_backend='sdpa'))
            with patch('eval.paired.artifacts.read',side_effect=read):
                result=suite(root,frozen,'Z')
            self.assertEqual(len(result['averages']['M1']['tasks']),14)
            self.assertEqual(len(result['averages']['M2']['tasks']),13)
            self.assertTrue(result['confirmatory']['H1_rejected']);self.assertTrue(result['confirmatory']['H2_tested'])
            self.assertAlmostEqual(result['progress_ratios']['M1']['ratio'],1.)
            means=result['averages']['M1'];self.assertAlmostEqual(means['B']-means['A'],means['mean'])
            json.dumps(result,allow_nan=False)
            def clustered_read(*args,**kwargs):
                obj=read(*args,**kwargs);obj['clusters']=[str(i//4) for i in range(80)]
                return obj
            with patch('eval.paired.artifacts.read',side_effect=clustered_read):
                probes=suite(root,frozen,'probes')
            for table in probes['tasks'].values():
                self.assertNotIn('admission_description',table);self.assertNotIn('symmetric_filter',table)
                self.assertEqual(table['metrics']['M1']['units'],20)
            self.assertEqual(len(probes['contrasts']),2)
            def wrong_path(*args,**kwargs):
                obj=clustered_read(*args,**kwargs);obj['metadata']['precision_path']='reference';return obj
            with patch('eval.paired.artifacts.read',side_effect=wrong_path),self.assertRaisesRegex(ValueError,'precision path'):
                suite(root,frozen,'probes')

    def test_precision_fallback_sequence_resume_and_changed_data_rejection(self):
        from eval.paired.run import execute
        from eval.models import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identities={}
            for arm in ('A10','A20','P20'):
                ck=root/arm;ck.mkdir();(ck/'model.safetensors').write_bytes(arm.encode());recipe=ck/'train_config.json';recipe.write_text('{}')
                identities[arm]=dict(path=str(ck),sha256=file_hash(ck/'model.safetensors'),recipe=str(recipe),recipe_sha256=file_hash(recipe))
            for f in ('manifest','calibration'):(root/f).write_text('{}')
            study=root/'study';study.mkdir()
            value=dict(protocol='frozen',protocol_sha256='hash',output=str(study),
                checkpoints=identities,manifest=str(root/'manifest'),manifest_sha256=file_hash(root/'manifest'),
                calibration=str(root/'calibration'),calibration_sha256=file_hash(root/'calibration'),
                dataset_root=str(root),batch_size=2,device='cpu',legacy_f=None,
                selection={'piqa':{'piqa':[0,2]}},sizes={'piqa':{'piqa':3}},
                position_sample='sample',sample_sha256='sample-hash')
            value['plan_sha256']=protocol.digest(value);plan=study/'plan.json';protocol.write_json(plan,value)
            calls=[]
            def launch(cmd,check):
                calls.append(cmd);command=cmd[3];opt=dict(zip(cmd[4::2],cmd[5::2]))
                dest=Path(opt['--output']);dest.mkdir(parents=True)
                arm=Path(opt['--checkpoint']).name
                meta=dict(status='completed',checkpoint_sha256=identities[arm]['sha256'],
                    recipe_sha256=identities[arm]['recipe_sha256'],dataset_manifest_sha256=value['manifest_sha256'],
                    precision_path=opt['--path'],batch_size=2,protocol_sha256='hash')
                if command=='position':
                    (dest/'losses.npy').write_bytes(b'fixture')
                    meta.update(sample_sha256='sample-hash',tokenizer_sha256='tok',batch_size=1,
                                T8_max_abs_error=0.,losses_sha256=file_hash(dest/'losses.npy'))
                protocol.write_json(dest/('metadata.json' if command=='position' else 'eval_metadata.json'),meta)
            def read(directory,*args,**kwargs):
                meta=artifacts.load_json(Path(directory)/'eval_metadata.json')
                if meta['status']!='completed':raise ValueError('Incomplete result')
                return dict(metadata=meta)
            with patch('eval.paired.protocol.read',return_value={'protocol_sha256':'hash'}), \
                 patch('eval.paired.protocol.tasks',return_value={'piqa':0}), \
                 patch('eval.paired.protocol.PRECISION',['piqa']), \
                 patch('eval.paired.artifacts.read',side_effect=read), \
                 patch('eval.paired.analysis.precision',return_value={'passed':False}), \
                 patch('eval.paired.position.read_sample',return_value={'sample_sha256':'sample-hash','tokenizer_sha256':'tok'}), \
                 patch('eval.paired.run.subprocess.run',side_effect=launch):
                with patch('eval.paired.position.read_sample',return_value={'sample_sha256':'changed'}):
                    with self.assertRaisesRegex(ValueError,'Position sample changed'):execute(plan)
                    self.assertEqual(calls,[])  # Reject changed input before any GPU job.
                execute(plan)
                self.assertEqual(len(calls),19)
                for cmd in calls[6:]:
                    opts=dict(zip(cmd[4::2],cmd[5::2]))
                    expected='production' if opts.get('--suite') in ('F','paloma') else 'reference'
                    self.assertEqual(opts['--path'],expected)
                execute(plan);self.assertEqual(len(calls),19)  # Completed task reuse, no duplicate execution.
                pos=study/'A20'/'position'/'metadata.json'
                original=artifacts.load_json(pos);meta=copy.deepcopy(original);meta['recipe_sha256']='wrong'
                pos.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError,'Position job'):execute(plan)
                pos.write_text(json.dumps(original))
                bad=study/'A20'/'Z'/'piqa'/'eval_metadata.json'
                meta=artifacts.load_json(bad);meta['dataset_manifest_sha256']='wrong';bad.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError,'queued job'):execute(plan)
                # Deferring the gated descriptive corpus must not skip Z or probes.
                deferred=copy.deepcopy(value);other=root/'deferred';other.mkdir()
                deferred['output']=str(other);deferred['deferred_suites']=['paloma']
                deferred['plan_sha256']=protocol.digest({k:v for k,v in deferred.items() if k!='plan_sha256'})
                protocol.write_json(other/'plan.json',deferred);calls.clear()
                execute(other/'plan.json');self.assertEqual(len(calls),17)
                actual=[dict(zip(c[4::2],c[5::2])).get('--suite') for c in calls]
                self.assertNotIn('paloma',actual);self.assertIn('Z',actual);self.assertIn('probes',actual)


class PairedHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_custom_data_audit_and_snapshot_profile(self):
        from eval.paired.audit import data
        from scripts.prepare_expanded_benchmarks import specifications
        specs=specifications(protocol.CUSTOM,profile='paired')
        self.assertEqual(specs['babi_cf_qa3'],('RMT-team/babilong-1k-samples','0k',['qa3']))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[]
            for name in protocol.CUSTOM:
                if name=='social_iqa_norm':rows=[dict(context='One',question='Why?',answerA='a',answerB='long answer',answerC='yes',label='2')]*3
                elif name=='commonsense_qa_cf':rows=[dict(question='What?',choices=dict(text=['a','bb','ccc','dddd','eeeee'],label=['A','B','C','D','E']),answerKey='D')]*3
                else:rows=[dict(input=t,question='Where is Mary?',target='kitchen') for t in ['Mary moved.','Mary moved. John moved.','Another story.']]
                entries.append(snapshot(root,name,rows,specs[name][2]))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,repositories=entries)))
            result=data(root,manifest,dict(protocol_sha256='fixture'))
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['tasks']['babi_cf_qa1']['stories'],2)
            self.assertTrue(result['tasks']['social_iqa_norm']['canonical_social_acc_equivalent'])

    def test_all_task_templates_actual_artifacts_and_guards(self):
        from lm_eval.tasks import get_task_dict
        from lm_eval import evaluator
        from eval.eval_checkpoint import json_default
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);configs=[];mapping={}
            rows={
                'arc_easy':dict(question='Which?',choices=dict(text=['a','long answer','xyz','b'],label=['A','B','C','D']),answerKey='B'),
                'hellaswag':dict(ctx_a='Person',ctx_b='does',activity_label='test',endings=['a','long answer','xyz','b'],label='1'),
                'piqa':dict(goal='go',sol1='here',sol2='there',label=1),
                'winogrande':dict(sentence='The _ ran home.',option1='cat',option2='dog',answer='2'),
                'xnli_en':dict(premise='One',hypothesis='Two',label=0),
                'xstorycloze_en':dict(input_sentence_1='One',input_sentence_2='Two',input_sentence_3='Three',input_sentence_4='Four',sentence_quiz1='Five',sentence_quiz2='Six',answer_right_ending=1),
                'lambada_openai':dict(text='A big car'),
                'sciq':dict(support='context',question='Which?',distractor1='bad',distractor2='b',distractor3='abc',correct_answer='right'),
                'openbookqa':dict(question_stem='Which?',choices=dict(text=['a','bb','ccc','dddd'],label=['A','B','C','D']),answerKey='C'),
                'social_iqa_norm':dict(context='One',question='Why?',answerA='a',answerB='long answer',answerC='yes',label='2'),
                'commonsense_qa_cf':dict(question='What?',choices=dict(text=['a','bb','ccc','dddd','eeeee'],label=['A','B','C','D','E']),answerKey='D'),
                'paws_en':dict(sentence1='One',sentence2='Two',label=1)}
            rows['arc_challenge']=rows['arc_easy']
            tasks=[*protocol.Z,'paws_en',*protocol.PROBES,*protocol.inventory('paloma')]
            for task in tasks:
                for name in protocol.inventory(task):
                    if task=='mmlu_continuation':row=dict(question='Which?',choices=['a','bb','ccc','dddd'],answer=2)
                    elif task=='blimp':row=dict(sentence_good='A good car.',sentence_bad='Car a good.')
                    elif task in protocol.PROBES:row=dict(input='Mary went home.',question='Where is Mary?',target='kitchen')
                    elif task.startswith('paloma_'):row=dict(text='One naïve test')
                    else:row=rows[name]
                    file=root/(name+'.jsonl');file.write_text('\n'.join(json.dumps(row) for _ in range(3)))
                    mapping[name]=dict(dataset_path='json',dataset_name=None,dataset_kwargs=dict(data_files={s:str(file) for s in ('train','test','validation','dev','val','eval','qa1','qa2','qa3')}))
                configs+=protocol.configs(task,mapping)
            for cfg in configs:cfg['num_fewshot']=2 if cfg['task'] in protocol.PROBES else 0
            loaded=get_task_dict(configs,task_manager=protocol.manager())
            for name,t in loaded.items():t.set_fewshot_seed(42)
            result=evaluator.evaluate(OracleLM(),loaded,bootstrap_iters=0,log_samples=True)
            result=json.loads(json.dumps(result,default=json_default,allow_nan=False))
            for task in tasks:
                samples={n:result['samples'][n] for n in protocol.inventory(task)}
                a=artifacts.validate(result,samples,task,2 if task in protocol.PROBES else 0)
                differences=artifacts.align(a,a)
                for d in differences.values():self.assertTrue((d==0).all())
                if task not in protocol.PROBES and not task.startswith('paloma_'):
                    self.assertIn('M1',a['metrics'])
            sample=result['samples']['arc_easy'][0];config=result['configs']['arc_easy']
            parsed=artifacts.parse(sample,config,'char');ll=np.array(parsed['ll']);gold=1
            n=np.array([1,11,3,1]);scores=ll/n
            self.assertEqual(parsed['metrics']['M1'],ll[gold]/n[gold])
            self.assertAlmostEqual(parsed['metrics']['M2'],np.exp(scores[gold])/np.exp(scores).sum())
            samples={'arc_easy':result['samples']['arc_easy']}
            for mutation in ('count','missing_counts','missing_group','target','nan','empty','missing_hash','duplicate'):
                r=copy.deepcopy(result);s=copy.deepcopy(samples)
                if mutation=='count':r['n-samples']['arc_easy']['effective']=2
                if mutation=='missing_counts':del r['n-samples']
                if mutation=='missing_group':
                    with self.assertRaises(ValueError):artifacts.validate(r,{},'blimp',0)
                    continue
                if mutation=='target':s['arc_easy'][0]['target']=90
                if mutation=='nan':s['arc_easy'][0]['filtered_resps'][0][0]=float('nan')
                if mutation=='empty':s['arc_easy'][0]['arguments'][0][1]=' '
                if mutation=='missing_hash':del s['arc_easy'][0]['target_hash']
                if mutation=='duplicate':s['arc_easy'][1]=s['arc_easy'][0]
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):artifacts.validate(r,s,'arc_easy',0)
            a=artifacts.validate(result,samples,'arc_easy',0)
            for key in ('doc_hash','prompt_hash','target_hash','gold','arguments','doc','target'):
                b=copy.deepcopy(a);b['records'][0]['identity'][key]='different'
                with self.assertRaises(ValueError):artifacts.align(a,b)

    def test_real_entrypoint_reference_subset_and_position_loss(self):
        from eval.paired.run import evaluate,load_model
        from eval.paired.position import losses,freeze,read_sample
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);ckpt=checkpoint(root)
            raw=root/'raw';raw.mkdir()
            rows=[dict(context='2',question='3',answerA='4',answerB='5 6',answerC='7',label=str(i%3+1)) for i in range(4)]
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,repositories=[snapshot(raw,'social_iqa_norm',rows,['train','validation'])])))
            frozen_path=root/'protocol.json';frozen=protocol.freeze(frozen_path)
            protocol.read(frozen_path)
            selected={'social_iqa_norm':[1,3]}
            evaluate(ckpt,'social_iqa_norm','Z',root/'result',frozen_path,raw,manifest,'cpu',2,'reference',selected)
            result=artifacts.read(root/'result','social_iqa_norm',0,frozen,selection=selected)
            self.assertEqual(result['ids'],[('social_iqa_norm',1),('social_iqa_norm',3)])
            self.assertEqual(result['metadata']['harness_n_samples']['social_iqa_norm']['effective'],4)
            self.assertEqual(result['metadata']['weights_dtype'],'torch.float32')
            evaluate(ckpt,'social_iqa_norm','Z',root/'production',frozen_path,raw,manifest,'cpu',2,'production',selected)
            production=artifacts.read(root/'production','social_iqa_norm',0,frozen,selection=selected)
            self.assertTrue(precision(result,result,production,production,production,production,4)['passed'])
            with self.assertRaises(ValueError):artifacts.read(root/'result','social_iqa_norm',0,frozen)
            with self.assertRaises(ValueError):evaluate(ckpt,'social_iqa_norm','Z',root/'result',frozen_path,raw,manifest,'cpu',2,'reference',selected)
            model,tok,meta=load_model(ckpt,'cpu','reference')
            ids=torch.tensor([[2,3,4,5,6,7,8,9],[8,7,6,5,4,3,2,1]])
            before={k:v.clone() for k,v in model.state_dict().items()}
            values,error=losses(model,ids);self.assertEqual(values.shape,(2,7));self.assertLess(error,1e-4)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            tiny=copy.deepcopy(frozen);tiny['position'].update(documents=2,length=8)
            sample=freeze([dict(text='2 3')]+[dict(text=' '.join(map(str,x.tolist()))) for x in ids],tok,root/'sample.json',tiny)
            self.assertEqual(sample['rows'][0]['input_ids'],ids[0].tolist())
            read_sample(root/'sample.json',tiny)
            sample['rows'][0]['input_ids'][0]=5;(root/'sample.json').write_text(json.dumps(sample))
            with self.assertRaises(ValueError):read_sample(root/'sample.json',tiny)
            altered=copy.deepcopy(frozen);altered['seed']=43;(root/'protocol.json').write_text(json.dumps(altered))
            with self.assertRaises(ValueError):protocol.read(root/'protocol.json')


if __name__=='__main__':unittest.main()
