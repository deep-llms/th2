"""Expanded benchmark scoring, data leakage, generation and real Trainer checks."""
from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from tests.test_eval import make_model
from tests.test_train import fixture
from eval.models import EvaluationModel
from eval.generation import generate_answer
from eval.benchmarks import resolve_tasks, task_configs
from eval.finetune import metrics, supervised_splits, document_key, encode_pairs, PairClassifier, PairCollator, GradientCheck
from eval.qa import features, SpanClassifier, SpanCollator, decode_spans, answer_scores, select_threshold, score_answers, grouped_splits


class ExpandedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_inventory_and_xcsr_labels(self):
        from scripts.prepare_expanded_benchmarks import normalize_xcsr, specifications
        self.assertEqual(len(resolve_tasks(['mmlu'], True)),57)
        self.assertEqual(len(resolve_tasks(['blimp'], True)),67)
        names=resolve_tasks(['truthfulqa','lambada','xcsr','social_iqa','boolq','openbookqa','gsm8k'],True)
        self.assertEqual(len(names),9)
        self.assertEqual(specifications(['boolq'])['boolq'],('aps/super_glue','boolq',['train','validation']))
        row=dict(lang='en',id='x',question=dict(stem='Why?',choices=dict(label=['A','B'],text=['yes','no'])),answerKey='B')
        self.assertEqual(normalize_xcsr(row)['label'],1)
        with self.assertRaisesRegex(ValueError,'hidden'): normalize_xcsr(dict(row,answerKey=''))
        config=task_configs(['xcsqa_en'],{'xcsqa_en':dict(dataset_path='parquet',dataset_kwargs={})})[0]
        self.assertEqual(config['dataset_path'],'parquet')
        self.assertEqual(config['test_split'],'validation')

    def test_greedy_matches_direct_custom_forward_and_preserves_state(self):
        class Tokenizer:
            eos_token_id=31
            def encode(self,text,**kw):return [2,3]
            def decode(self,ids,**kw):return ','.join(map(str,ids))
        tok=Tokenizer()
        for arm in ('A','P6-iso','P7-simple','P3-block'):
            model=EvaluationModel(make_model(arm),8)
            before={k:v.clone() for k,v in model.state_dict().items()}
            ids=torch.tensor([[2,3]]); expected=[]
            for _ in range(3):
                x=int(model(ids).logits[:,-1].argmax(-1))
                if x==31:break
                expected.append(x);ids=torch.cat((ids,torch.tensor([[x]])),1)
            self.assertEqual(generate_answer(model,tok,'prompt',3),tok.decode(expected))
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'exceeds context'):generate_answer(model,tok,'prompt',7)

    def test_generation_stop_and_eos(self):
        class Tokenizer:
            eos_token_id=9
            def encode(self,*a,**k):return [1]
            def decode(self,ids,**k):return ''.join({2:'hi',3:'STOP',4:'extra'}.get(x,'?') for x in ids)
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.config=SimpleNamespace(max_position_embeddings=20);self.eval()
            def forward(self,input_ids,**k):
                scores=torch.zeros(1,input_ids.shape[1],10);scores[:,-1,[2,3,4,9][min(input_ids.shape[1]-1,3)]]=10
                return SimpleNamespace(logits=scores)
        self.assertEqual(generate_answer(Model(),Tokenizer(),'p',5,['STOP']),'hi')
        self.assertEqual(generate_answer(Model(),Tokenizer(),'p',5),'hiSTOPextra')

    def test_glue_metrics_splits_and_single_sentence(self):
        logits=np.array([[4,0],[0,4],[0,4],[4,0]])
        labels=np.array([0,1,0,0])
        self.assertAlmostEqual(metrics((logits,labels),'glue_mrpc')['accuracy_f1'],(.75+2/3)/2)
        from sklearn.metrics import matthews_corrcoef
        self.assertEqual(metrics((logits,labels),'glue_cola')['matthews_correlation'],matthews_corrcoef(labels,logits.argmax(-1)))
        rows=[dict(sentence=f'text {i}',label=i%2) for i in range(40)]
        raw=DatasetDict(train=Dataset.from_list(rows),validation=Dataset.from_list(rows[:2]),test=Dataset.from_list([dict(unused='hidden')]))
        splits=supervised_splits(raw,'glue_sst2')
        sets=[{r['sentence'] for r in splits[s]} for s in ('train','validation','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(splits['test']),2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);tok=AutoTokenizer.from_pretrained(root/'model')
            encoded=encode_pairs(dict(sentence=['2 3'],label=[1]),tok,('sentence',),32,'glue_sst2')
            self.assertEqual(encoded['input_ids'][0][-1],tok.eos_token_id)
            model=PairClassifier(make_model('P6-iso'),2,42)
            result=model(**PairCollator(tok.eos_token_id)([dict(input_ids=encoded['input_ids'][0],labels=1)]))
            result['loss'].backward();check=GradientCheck();check.on_pre_optimizer_step(None,None,None,model=model)
            self.assertEqual(check.report['status'],'passed')

    def test_qa_windows_offsets_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);tok=AutoTokenizer.from_pretrained(root/'model')
            context=' '.join(['2']*70+['3']+['4']*20)
            examples=[dict(id='a',question='5',context=context,answers=dict(text=['3'],answer_start=[140])),
                      dict(id='b',question='5',context='2 4',answers=dict(text=[],answer_start=[]))]
            encoded,meta=features(examples,tok,32,stride=8)
            self.assertGreater(len(encoded),2)
            self.assertTrue(any(r['start_positions']!=m['null'] for r,m in zip(encoded,meta) if m['id']=='a'))
            for r,m in zip(encoded,meta):
                if r['start_positions']!=m['null']:
                    self.assertEqual(context[m['offsets'][r['start_positions']][0]:m['offsets'][r['end_positions']][1]],'3')
            rows=[encoded[0],encoded[-1]]
            for arm in ('A','P6-iso','P7-simple'):
                model=SpanClassifier(make_model(arm),42)
                result=model(**SpanCollator(tok.eos_token_id)(rows));result['loss'].backward()
                check=GradientCheck();check.on_pre_optimizer_step(None,None,None,model=model)
                self.assertEqual(check.report['status'],'passed')
            logits=np.full((len(encoded),32,2),-5.)
            for i,(row,m) in enumerate(zip(encoded,meta)):
                logits[i,row['start_positions'],0]=5
                logits[i,row['end_positions'],1]=5
            decoded=decode_spans(logits,meta,examples)
            threshold=select_threshold(decoded,examples)
            self.assertEqual(score_answers(decoded,examples,threshold),dict(exact_match=1.,f1=1.))
            self.assertEqual(answer_scores('The, blue car!',['blue car']),(1.,1.))

    def test_qa_group_split_and_threshold_ties(self):
        rows=[dict(id=str(i),context=f'context {i//2}',question=str(i),answers=dict(text=[],answer_start=[])) for i in range(50)]
        raw=DatasetDict(train=Dataset.from_list(rows),validation=Dataset.from_list(rows[:2]))
        splits=grouped_splits(raw)
        sets=[set(splits[k]['context']) for k in ('train','validation','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        examples=[dict(id='a',answers=dict(text=['x'])),dict(id='b',answers=dict(text=[]))]
        decoded={'a':dict(text='x',difference=1.),'b':dict(text='y',difference=1.)}
        self.assertEqual(score_answers(decoded,examples,select_threshold(decoded,examples))['f1'],.5)

    def test_span_numerical_gate(self):
        from scripts.finetune_study import numerical_check, check_numerical_limits
        from tests.test_fa4_baseline import reference_kernel
        with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'})):
            for arm in ('A','P6-iso','P7-simple'):
                model=SpanClassifier(make_model(arm,'fa4'),42)
                inputs=SpanCollator(31)([dict(input_ids=[2,3,4,31],start_positions=1,end_positions=2),
                                         dict(input_ids=[5,6,31],start_positions=2,end_positions=2)])
                report=numerical_check(model,inputs)
                check_numerical_limits(report)
                self.assertLess(report['probability_max_absolute'],1e-5)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_benchmark_aggregation(self):
        from eval.reporting import benchmark_averages, glue_average
        scores={'truthfulqa_mc1':{'acc,none':.2},'truthfulqa_mc2':{'acc,none':.6},'boolq':{'acc,none':.8}}
        self.assertAlmostEqual(benchmark_averages(scores)['groups']['multiple_choice']['average'],.6)
        with self.assertRaises(ValueError):benchmark_averages({'mmlu_anatomy':{'acc,none':.5}})
        with self.assertRaises(ValueError):glue_average({'glue_cola':.5})


class ExpandedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_real_harness_templates_and_scores(self):
        from lm_eval.tasks import TaskManager, get_task_dict
        from lm_eval import evaluator
        from eval.generation import harness_model_class
        from eval.benchmarks import expanded_groups
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);tok=AutoTokenizer.from_pretrained(root/'model')
            tok.pad_token=tok.eos_token
            configs=[]
            rows={
                'openbookqa':dict(question_stem='2',choices=dict(text=['3','4','5','6'],label=['A','B','C','D']),answerKey='A'),
                'social_iqa':dict(context='2',question='3',answerA='4',answerB='5',answerC='6',label='1'),
                'boolq':dict(passage='2',question='3',label=1),
                'truthfulqa_mc1':dict(question='2',mc1_targets=dict(choices=['3','4'],labels=[1,0]),mc2_targets=dict(choices=['3','4','5'],labels=[1,1,0])),
                'truthfulqa_mc2':dict(question='2',mc1_targets=dict(choices=['3','4'],labels=[1,0]),mc2_targets=dict(choices=['3','4','5'],labels=[1,1,0])),
                'lambada_openai':dict(text='2 3 4'),
                'xcsqa_en':dict(question='2',choices=['3','4'],label=0),
                'xcodah_en':dict(question='',choices=['2 3','2 4'],label=0),
                'gsm8k':dict(question='2',answer='3 #### 4')}
            for name in expanded_groups()['mmlu']:rows[name]=dict(question='2',choices=['3','4','5','6'],answer=0)
            for name in expanded_groups()['blimp']:rows[name]=dict(sentence_good='2 3',sentence_bad='3 2')
            for name,row in rows.items():
                file=root/(name+'.jsonl');file.write_text(json.dumps(row)+'\n')
                config=task_configs([name],{name:dict(dataset_path='json',dataset_name=None,
                    dataset_kwargs=dict(data_files={s:str(file) for s in ('train','validation','test','dev')},keep_in_memory=True))})[0]
                config['num_fewshot']=0
                if name=='gsm8k':config['generation_kwargs']=dict(until=['Question:'],max_gen_toks=2,do_sample=False,temperature=0)
                configs.append(config)
            from eval.benchmarks import benchmark_task_manager
            task_dict=get_task_dict(configs,task_manager=benchmark_task_manager())
            self.assertEqual(len(task_dict),len(rows))
            model=EvaluationModel(make_model('P6-iso'),8)
            model.config.max_position_embeddings=512
            lm=harness_model_class()(pretrained=model,tokenizer=tok,backend='causal',device='cpu',batch_size=2,max_length=512,softmax_dtype='float32')
            # Exercise every template's actual request and metric processing.
            result=evaluator.evaluate(lm,task_dict,limit=1,bootstrap_iters=0,log_samples=True)
            self.assertEqual(set(result['results']),set(rows))
            self.assertIn('exact_match,strict-match',result['results']['gsm8k'])
            for name in ('xcsqa_en','xcodah_en'):
                task=task_dict[name];doc=task.eval_docs[0]
                self.assertEqual(task.process_results(doc,[(-1.,True),(-2.,False)])['acc'],1.)
                self.assertEqual(task.process_results(doc,[(-2.,False),(-1.,True)])['acc'],0.)
            for name,task in task_dict.items():
                doc=task.eval_docs[0]
                if name.startswith('truthfulqa_mc'):
                    scores=task.process_results(doc,[(-1.,False)]*len(task.doc_to_choice(doc)))
                    self.assertAlmostEqual(scores['acc'],1. if name.endswith('mc1') else 2/3)
            lambada=task_dict['lambada_openai']
            self.assertEqual(lambada.doc_to_target(lambada.eval_docs[0]),' 4')
            self.assertEqual(lambada.process_results(lambada.eval_docs[0],[(-2.,True)])['acc'],1.)

    def test_glue_and_qa_real_trainer_reload(self):
        from dataclasses import asdict
        import hashlib
        from datasets import ClassLabel
        from safetensors.torch import save_file
        from tests.test_proxy_memory import config as model_config
        from eval.finetune import TASKS, GLUE_TASKS, run, score_name
        from scripts.finetune_study import verify_reload
        for task in ['glue_'+x for x in GLUE_TASKS]+['squad_v2']:
            with self.subTest(task=task),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixture(root);backbone=make_model('P6-iso')
                ckpt=root/'checkpoint-2500';ckpt.mkdir()
                tok=AutoTokenizer.from_pretrained(root/'model');tok.save_pretrained(ckpt)
                save_file({k:v.clone() for k,v in backbone.state_dict().items()},ckpt/'model.safetensors')
                (ckpt/'trainer_state.json').write_text(json.dumps(dict(global_step=2500)))
                (root/'train_config.json').write_text(json.dumps(dict(model_config=model_config().to_dict(),
                    model=dict(tokenizer_name=str(ckpt)),data=dict(block_size=64),training=dict(seed=42,bf16=False),
                    pilot=dict(arm='P6-iso',proxy_screen=True,consumer=2,deep_target=8,lm_chunk=3,causal_attention=False,
                        **{'proxy_'+k:v for k,v in asdict(backbone.settings).items()}))))
                raw=root/'raw';raw.mkdir();files=[];splits={}
                split_names=['train','validation_matched','validation_mismatched'] if task.startswith('glue_mnli_') else ['train','validation']
                for split in split_names:
                    rows=[]
                    for i in range(40 if split=='train' else 8):
                        if task=='squad_v2':
                            row=dict(id=f'{split}-{i}',question='5',context=f'2 3 4 {split} {i}',
                                     answers=dict(text=['3'] if i%2 else [],answer_start=[2] if i%2 else []))
                        else:
                            _,fields,labels=TASKS[task]
                            row={f:f'2 {split} {i} {f}' for f in fields}
                            row['label']=float(i%6) if task=='glue_stsb' else i%len(labels)
                        rows.append(row)
                    ds=Dataset.from_list(rows)
                    if task not in ('squad_v2','glue_stsb'):ds=ds.cast_column('label',ClassLabel(names=TASKS[task][2]))
                    path=raw/(split+'.parquet');ds.to_parquet(path)
                    files.append(dict(path=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    splits[split]=[path.name]
                manifest=root/'manifest.json'
                aliases=['glue_mnli_matched','glue_mnli_mismatched'] if task.startswith('glue_mnli_') else [task]
                manifest.write_text(json.dumps(dict(schema_version=1,repositories=[dict(path='.',tasks=aliases,files=files,load_configs={name:dict(loader='parquet',data_files=splits) for name in aliases})])))
                cfg=dict(checkpoint=str(ckpt),task=task,dataset_root=str(raw),dataset_manifest=str(manifest),
                    max_length=32,preprocessing_num_workers=1,attention_backend='sdpa',output_dir=str(root/'fit'),
                    use_cpu=True,report_to='none',seed=42,data_seed=42,max_steps=2,learning_rate=1e-3,
                    per_device_train_batch_size=2,per_device_eval_batch_size=2,eval_strategy='steps',eval_steps=1,
                    save_strategy='steps',save_steps=1,load_best_model_at_end=True,metric_for_best_model=score_name(task),
                    greater_is_better=True,save_only_model=True,save_total_limit=0,disable_tqdm=True,smoke=True)
                config=root/'config.json';config.write_text(json.dumps(cfg));run(config)
                cfg.update(output_dir=str(root/'reload'),evaluate_run=str(root/'fit'),load_best_model_at_end=False,eval_strategy='no',save_strategy='no')
                config.write_text(json.dumps(cfg));run(config)
                verify_reload(root/'fit',root/'reload',root/'verified.json')
                self.assertEqual(json.loads((root/'verified.json').read_text())['status'],'passed')
                if task == 'glue_mnli_matched':
                    cfg.update(task='glue_mnli_mismatched',output_dir=str(root/'mnli-mm'))
                    config.write_text(json.dumps(cfg));run(config)
                    fitted=json.loads((root/'fit/result.json').read_text())
                    mm=json.loads((root/'mnli-mm/result.json').read_text())
                    self.assertEqual(mm['weight_sha256'],fitted['weight_sha256'])
                    self.assertEqual(mm['data']['train_order_sha256'],fitted['data']['train_order_sha256'])


class ExpandedProtocolTests(unittest.TestCase):
    def test_sequential_protocol_and_single_mnli_fit(self):
        from scripts.finetune_study import make_jobs, SUPPORTED_TASKS
        from eval.suite import commands
        with tempfile.TemporaryDirectory() as tmp:
            tasks=[t for t in SUPPORTED_TASKS if t.startswith('glue_')]+['squad_v2']
            jobs=make_jobs(tmp,Path.cwd(),'/data','/manifest',{'A':'/A','P6-iso':'/P6'},tasks=tasks,
                           checkpoint_steps={'A':20000,'P6-iso':20000})
            names=[j['name'] for j in jobs]
            self.assertFalse(any(n.startswith('search-glue_mnli_mismatched') for n in names))
            self.assertIn('test-glue_mnli_mismatched-A-42',names)
            first_test=next(i for i,n in enumerate(names) if n.startswith('test-'))
            self.assertTrue(all(not n.startswith(('search-','confirm-')) for n in names[first_test:]))
            cfg=json.loads((Path(tmp)/'configs/test-glue_mnli_mismatched-A-43.json').read_text())
            self.assertIn('confirm/glue_mnli_matched/A',cfg['evaluate_run'])
            self.assertIn('glue_mnli_matched-A',cfg['selection_file'])
            for task in tasks:
                self.assertIn('gate-A-'+task,names)
                self.assertIn('verify-reload-A-'+task,names)
            plans=commands(['/A','/B'],'/out','/data','/manifest')
            self.assertEqual(len(plans),10)
            self.assertEqual([x['group'] for x in plans[:5]],['zero','five','ten','twentyfive','math'])
            self.assertTrue(all('--seed' in x['argv'] for x in plans))

    def test_suite_checks_counts_and_matching_prompts(self):
        from eval.suite import collect
        with tempfile.TemporaryDirectory() as tmp, patch('eval.suite.PROTOCOLS', {'zero':(0,['truthfulqa'])}):
            root=Path(tmp);checkpoints=['/A','/P6']
            for i,ckpt in enumerate(checkpoints):
                dest=root/f'{i:02d}'/'zero';dest.mkdir(parents=True)
                (dest/'eval_metadata.json').write_text(json.dumps(dict(status='completed',checkpoint=ckpt,
                    checkpoint_sha256=ckpt,tokenizer_sha256='tokenizer',dataset_manifest_sha256='data',attention_backend='fa4',
                    arguments=dict(num_fewshot=0,seed=42,limit=1))))
                names=['truthfulqa_mc1','truthfulqa_mc2']
                (dest/'eval_benchmarks.json').write_text(json.dumps({n:{'acc,none':.5} for n in names}))
                (dest/'eval_benchmarks_full.json').write_text(json.dumps({'n-samples':{n:dict(original=817,effective=1) for n in names},'n-shot':{n:0 for n in names}}))
                (dest/'eval_samples.jsonl').write_text('\n'.join(json.dumps(dict(task=n,doc_id=0,doc={'question':'why'},arguments=['why','yes'],target=0)) for n in names))
            result=collect(root,checkpoints,limit=1)
            self.assertTrue(result['diagnostic'])
            self.assertEqual(result['checkpoints']['/A']['summary']['groups']['multiple_choice']['average'],.5)
            path=root/'01/zero/eval_samples.jsonl'
            path.write_text(path.read_text().replace('yes','no'))
            with self.assertRaisesRegex(ValueError,'prompts differ'):collect(root,checkpoints,limit=1)

    def test_fixed_qa_threshold_and_shared_mnli_protocol(self):
        from eval.qa import QAMetrics
        from eval.finetune import same_finetuning_task
        examples=[dict(id='a',context='x',answers=dict(text=['x']))]
        metadata=[dict(id='a',offsets=[[0,1],None],null=1)]
        metric=QAMetrics(metadata,examples,threshold=123.)
        metric(SimpleNamespace(predictions=np.array([[[2.,2.],[1.,1.]]])))
        self.assertEqual(metric.threshold,123.)
        self.assertEqual(same_finetuning_task('glue_mnli_matched','glue_mnli_mismatched'),True)
        self.assertEqual(same_finetuning_task('glue_rte','glue_wnli'),False)


if __name__=='__main__':unittest.main()
