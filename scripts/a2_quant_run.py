"""A2 Q coordinator: finite registered work, no scientific failure aborts peers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.a2_quant_io import lock, write
from scripts.a2_remaining import read_locked
from scripts.a2_stage1b_option2 import Journal
from efficient_llm.a2.quant_stats import PARENTS, nll_report


def collect_nll(root):
    values={};records=[]
    for path in sorted((Path(root)/'workers').glob('*/*/*/*-final.json')):
        row=json.loads(path.read_text())
        values.setdefault(row['dataset'],{}).setdefault(row['compression'],{})[row['parent']]=row['per_sequence_nll']
    for path in sorted((Path(root)/'workers').glob('*/*/*/runs.jsonl')):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    return values,records


def make_decision(cfg, root):
    values,records=collect_nll(root)
    result=nll_report(values)
    result['spec_sha256']=cfg['spec_sha256']
    result['calibration_disjointness_passed']=cfg['calibration_passed']
    result['calibration_dependent_results_unverified']=not cfg['calibration_passed']
    result['verification']={}
    for path in sorted((root/'workers').glob('*/*/*/tests.json')):
        checks=json.loads(path.read_text());phase,ds,parent=path.parts[-4:-1]
        result['verification'][f'{ds}/{parent}/{phase}']=dict(
            passed=all(c['passed'] for c in checks),failed_checks=[c['test'] for c in checks if not c['passed']])
    write(root/'decision_quant.json',result)
    (root/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return result


def downstream_statistics(replays):
    from eval.a2_stage1_stats import aligned_accuracy_arrays, paired_accuracy_intervals
    models=tuple(p+'-parent' for p in PARENTS)+tuple(p+'-quantized' for p in PARENTS)
    contrasts={}
    for i,p in enumerate(PARENTS):
        row=[0]*8;row[i]=1;row[i+4]=-1;contrasts[p+'_drop']=tuple(row)
    from efficient_llm.a2.quant_stats import CONTRASTS
    for name in ('L_ann','L_stab','R_short'):
        w=CONTRASTS[name];contrasts[name]=tuple(w)+tuple(-x for x in w)
    tasks=aligned_accuracy_arrays(replays,models=models)
    return paired_accuracy_intervals(tasks,models=models,contrasts=contrasts)


def run(cfg, cfg_lock, root):
    root=Path(root);started=time.monotonic();journal=Journal(root)
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','TOKENIZERS_PARALLELISM':'false',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    for name in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(name,None)
    def command(argv, log, gpu=None):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            result=subprocess.run(argv,env={**env,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)},
                                  stdout=stream,stderr=subprocess.STDOUT,check=False)
        journal.record('process:'+str(log.relative_to(root)),result.returncode==0,'returncode='+str(result.returncode))
        return result.returncode
    def phase(name, datasets, bits=None):
        items=[(ds,p) for ds in datasets for p in PARENTS]
        def lane(index):
            for ds,p in items[index::8]:
                target=root/'workers'/name/ds/p
                argv=[sys.executable,'-u','-m','scripts.a2_quant_worker','--config',cfg_lock['path'],
                    '--config-sha256',cfg_lock['sha256'],'--dataset',ds,'--parent',p,
                    '--phase',name,'--output',str(target)]
                journal.attempt(name+':'+ds+':'+p,lambda argv=argv,ds=ds,p=p:
                    command(argv,root/'logs'/name/(ds+'-'+p+'.log'),index))
        print('QUANT_PHASE',name,flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lane,range(8)))
    from scripts.a2_quant_inputs import pooled_followup
    with ThreadPoolExecutor(max_workers=1) as cpu:
        followup=cpu.submit(journal.attempt,'pooled_followup',lambda:pooled_followup(cfg,root))
        phase('baseline',('dclm','fwe','nem'))
        phase('rtn',('dclm','fwe','nem'))
        phase('gptq',('dclm','fwe'))
        phase('cal1',('dclm','fwe'))
        interim=journal.attempt('interim_labels',lambda:make_decision(cfg,root))
        if interim:
            labels={ds:{c:{k:v['label'] for k,v in r['contrasts'].items() if k in ('L_ann','L_stab','R_short')}
                        for c,r in rows.items()} for ds,rows in interim['datasets'].items() if ds!='nem'}
            write(root/'labels_note.json',dict(spec_sha256=cfg['spec_sha256'],labels=labels,
                calibration_disjointness_passed=cfg['calibration_passed'],continuing='NEM GPTQ then downstream'))
            print('QUANT_LABELS_NOTE',json.dumps(labels),flush=True)
        phase('gptq',('nem',))
        followup.result()
    decision=journal.attempt('all_nll_statistics',lambda:make_decision(cfg,root))
    choice=None
    if decision:
        means={}
        for bits in (3,4):
            try:
                damages=[decision['datasets'][ds]['gptq'+str(bits)]['damage'][p]['damage']
                         for ds in ('dclm','fwe') for p in PARENTS]
                means[bits]=sum(damages)/8
            except KeyError:journal.record('bitwidth_coverage:'+str(bits),False,'Missing final NLL; use a complete registered bitwidth if available')
        if means:choice=min(means,key=lambda b:(abs(means[b]-.10),b))
        write(root/'bitwidth.json',dict(spec_sha256=cfg['spec_sha256'],mean_damage=means,selected_bits=choice,
            rule='closest to 0.10 over all eight DCLM/FWE parents; exact tie -> lower bitwidth'))
    scores={};replays={};independent={}
    for ds in ('dclm','fwe'):
        for p in ('S72','P300','P1T'):
            name=ds+'/'+p+'-parent'
            scores[name]=cfg['prior_downstream'][ds]['scores'][p+'-parent']
            replays[name]=cfg['prior_downstream'][ds]['replays'][p+'-parent']
    selection=[(ds,p,choice) for ds in ('dclm','fwe') for p in PARENTS] if choice else []
    selection.extend((ds,'S188',None) for ds in ('dclm','fwe'))
    def score_lane(index):
        for ds,p,bits in selection[index::8]:
            name=p+('-quantized' if bits else '-parent');target=root/'downstream'/ds/name
            argv=[sys.executable,'-u','-m','scripts.a2_quant_worker','--config',cfg_lock['path'],
                '--config-sha256',cfg_lock['sha256'],'--dataset',ds,'--parent',p,
                '--phase','downstream','--output',str(target)]
            if bits:argv.extend(['--bits',str(bits)])
            def work():
                command(argv,root/'logs/downstream'/(ds+'-'+name+'.log'),index)
                item=lock(target/'scores.json');read_locked(item)
                return item
            item=journal.attempt('score:'+ds+'/'+name,work)
            if item:scores[ds+'/'+name]=item
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(score_lane,range(8)))
    common=cfg['downstream_common']
    def replay(name,item,folder):
        target=root/folder/(name+'.json')
        argv=[common['eval_python'],'-m','eval.a2_requests','replay','--tokenizer',common['tokenizer'],
            '--input-spec',common['input_spec']['path'],'--dataset-root',common['dataset_root'],
            '--manifest',common['manifest']['path'],'--max-length','4096','--tasks',*common['tasks'],
            '--prepared',cfg['prepared']['path'],'--scored',item['path'],'--output',str(target)]
        command(argv,root/'logs'/folder/(name.replace('/','-')+'.log'))
        replay_value=json.loads(target.read_text())
        replay_value['a2_quant_spec_sha256']=cfg['spec_sha256']
        write(target,replay_value)
        result=lock(target)
        from eval.a2_suite import collect_files
        identity=read_locked(item)['checkpoint_identity']
        collect_files({name:identity},cfg['prepared'],{name:item},{name:result})
        return result
    # CPU harness imports remain isolated in the existing eval interpreter.
    for name,item in sorted(scores.items()):
        if name not in replays:
            result=journal.attempt('primary_replay:'+name,lambda n=name,v=item:replay(n,v,'replays'))
            if result:replays[name]=result
    paired={}
    for ds in ('dclm','fwe'):
        result=journal.attempt('downstream_CI:'+ds,lambda ds=ds:downstream_statistics(
            {n.split('/')[1]:read_locked(v) for n,v in replays.items() if n.startswith(ds+'/')}))
        if result:paired[ds]=result
    write(root/'downstream_paired.json',dict(spec_sha256=cfg['spec_sha256'],datasets=paired))
    for name,item in sorted(scores.items()):
        result=journal.attempt('independent_replay:'+name,lambda n=name,v=item:replay(n,v,'independent_replays'))
        if result:independent[name]=result
    verified={}
    for ds in ('dclm','fwe'):
        def compare(ds=ds):
            actual=downstream_statistics({n.split('/')[1]:read_locked(v) for n,v in independent.items() if n.startswith(ds+'/')})
            equal=actual==paired[ds]
            journal.record('independent_paired:'+ds,equal,'Full fresh replay, prompt/score checks and bootstrap recomputation')
            return equal
        verified[ds]=journal.attempt('independent_statistics:'+ds,compare)
    write(root/'downstream.json',dict(spec_sha256=cfg['spec_sha256'],selected_bits=choice,
          scores=scores,replays=replays,independent_replays=independent,verified=verified))
    values,records=collect_nll(root)
    worker_checks=[]
    for file in sorted(root.glob('workers/*/*/*/tests.json')):
        worker_checks.extend(json.loads(file.read_text()))
    for file in sorted(root.glob('downstream/*/*/tests.json')):
        worker_checks.extend(json.loads(file.read_text()))
    journal.record('registered_NLL_coverage',len(records)==160,'Expected 80 models x two sets; actual='+str(len(records)))
    journal.record('registered_downstream_coverage',len(scores)==16 and len(replays)==16,
                   'Ten new scores plus six reused parent scores; actual='+str(len(scores)))
    # Independent file-level NLL recomputation, not just exit markers.
    def verify_nll():
        import numpy as np
        for item in records:
            row=read_locked(item['per_sequence_file'])
            if float(np.asarray(row['per_sequence_nll'],dtype=np.float64).mean())!=item['nll']:
                raise ValueError('NLL file differs from run record')
        recomputed=nll_report(values)
        if decision and recomputed['datasets']!=decision['datasets']:raise ValueError('Sequence bootstrap differs')
        return True
    journal.attempt('independent_NLL_recomputation',verify_nll)
    all_checks=dict(spec_sha256=cfg['spec_sha256'],coordinator=journal.events,workers=worker_checks,
                    preparation=json.loads((root/'preparation_checks.json').read_text()))
    write(root/'tests.json',all_checks)
    issues=[x for x in journal.events if not x['passed']]+[x for x in worker_checks if not x['passed']]
    done=dict(spec_sha256=cfg['spec_sha256'],execution_complete=True,registered_models=80,
        nll_files=len(records),downstream_models=len(scores),elapsed_seconds=time.monotonic()-started,
        node_hours=(time.monotonic()-started)/3600,coverage_complete=len(records)==160 and len(scores)==16
        and len(replays)==16 and all(v is True for v in verified.values()),
        flagged_checks=len(issues),calibration_disjointness_passed=cfg['calibration_passed'],
        decision=lock(root/'decision_quant.json') if decision else None,tests=lock(root/'tests.json'))
    write(root/'complete.json',done)
    from scripts.a2_quant_report import render
    journal.attempt('render_final_report',lambda:render(root,cfg))
    return done


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args();item=dict(path=a.config,sha256=a.config_sha256);cfg=read_locked(item)
    done=run(cfg,item,Path(cfg['results_root']))
    write(Path(a.receipt),dict(execution_complete=True,result=lock(Path(cfg['results_root'])/'complete.json'),
        coverage_complete=done['coverage_complete'],spec_sha256=cfg['spec_sha256']))


if __name__=='__main__':main()
