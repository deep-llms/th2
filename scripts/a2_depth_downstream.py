"""Three C students, exact DCLM parent reuse, paired bootstrap and fresh replay.

The native owner calls this only after C training releases all GPUs. Unusable
artifacts are flagged; independent models and replays continue without a pause.
"""
import os
from pathlib import Path
import subprocess
import sys
from efficient_llm.a2_depth.common import Checks,write
from scripts.a2_quant_io import lock
from scripts.a2_remaining import read_locked
from eval.a2_request_bridge import digest
from eval.a2_suite import collect_files
from eval.a2_stage1_stats import MODELS,aligned_accuracy_arrays,paired_accuracy_intervals


def paired(replays):
    return paired_accuracy_intervals(aligned_accuracy_arrays(replays),resamples=10000,seed=0)


def verify_replay(primary, fresh):
    import numpy as np
    a=aligned_accuracy_arrays(primary);b=aligned_accuracy_arrays(fresh)
    return all(np.array_equal(a[t][m],b[t][m]) for t in a for m in MODELS)


def run(cfg,config):
    from scripts.a2_depth_partc_run import require_free_gpus
    root=Path(cfg['results_root'])/'downstream';root.mkdir(parents=True,exist_ok=False)
    checks=Checks(root/'checks.json');scores={};identities={};replays={};fresh={}
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'WANDB_MODE':'offline','TOKENIZERS_PARALLELISM':'false','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
    for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(k,None)
    def command(args,label,gpu=None):
        with (root/(label+'.log')).open('x') as f:
            proc=subprocess.run(args,env={**env,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)},
                                stdout=f,stderr=subprocess.STDOUT)
        checks.record('process:'+label,proc.returncode==0,proc.returncode)
        return proc.returncode
    def admit(name,item,replay):
        value=read_locked(item);ident=value['checkpoint_identity']
        if value['batch_size']!=2:raise ValueError('Registered batch 2 required')
        collect_files({name:ident},cfg['prepared'],{name:item},{name:replay})
        scores[name]=item;identities[name]=ident;replays[name]=replay
    parents=cfg['prior_downstream']
    for name in MODELS[:3]:
        checks.attempt('parent_reuse:'+name,lambda name=name:admit(name,parents['scores'][name],parents['replays'][name]))
    # Three lanes are unnecessary for this small screen; serial allocation makes
    # ownership explicit and allows the other arms to continue after a failure.
    for arm in ('P300','P1T','X72'):
        require_free_gpus()
        name=arm+'-student';out=root/'scores'/name
        command([sys.executable,'-u','-m','scripts.a2_depth_score','--config',str(config),
                 '--arm',arm,'--output',str(out)],name+'-score',0)
        def accept_score():
            item=lock(out/'scores.json');value=read_locked(item)
            prepared=read_locked(cfg['prepared'])
            if value['batch_size']!=2 or value['prepared_sha256']!=digest(prepared) or len(value['scores'])!=len(prepared['requests']):
                raise ValueError('Incomplete or mismatched scores')
            scores[name]=item;identities[name]=value['checkpoint_identity']
        checks.attempt('score:'+name,accept_score)
        # Import every scientific flag, even if production scoring succeeded.
        def worker_checks():
            for event in read_locked(lock(out/'checks.json'))['checks']:
                checks.record(name+':'+event['test'],event['passed'],event.get('detail'))
        checks.attempt('worker_checks:'+name,worker_checks)
    common=cfg['downstream_common']
    def replay(name,item,folder):
        for k in ('input_spec','manifest'):read_locked(common[k])
        target=root/folder/(name+'.json')
        command([common['eval_python'],'-m','eval.a2_requests','replay','--tokenizer',common['tokenizer'],
            '--input-spec',common['input_spec']['path'],'--dataset-root',common['dataset_root'],
            '--manifest',common['manifest']['path'],'--max-length','4096','--tasks',*common['tasks'],
            '--prepared',cfg['prepared']['path'],'--scored',item['path'],'--output',str(target)],name+'-'+folder)
        output=lock(target)
        collect_files({name:identities[name]},cfg['prepared'],{name:item},{name:output})
        return output
    for name,item in scores.items():
        if name not in replays:
            value=checks.attempt('replay:'+name,lambda name=name,item=item:replay(name,item,'replays'))
            if value:replays[name]=value
        value=checks.attempt('fresh_replay:'+name,lambda name=name,item=item:replay(name,item,'independent_replays'))
        if value:fresh[name]=value
    eligible={n:identities[n] for n in scores if n in replays}
    result=checks.attempt('collect',lambda:collect_files(eligible,cfg['prepared'],
         {n:scores[n] for n in eligible},{n:replays[n] for n in eligible})) if eligible else None
    primary=checks.attempt('paired_bootstrap',lambda:paired({n:read_locked(v) for n,v in replays.items()}))
    independent=checks.attempt('fresh_paired_bootstrap',lambda:paired({n:read_locked(v) for n,v in fresh.items()}))
    arrays_equal=checks.attempt('fresh_item_accuracy',lambda:verify_replay(
        {n:read_locked(v) for n,v in replays.items()},{n:read_locked(v) for n,v in fresh.items()}))
    checks.record('fresh_item_accuracy_equal',arrays_equal is True,'Every aligned primary accuracy, including reused parents')
    exact=primary is not None and primary==independent and arrays_equal is True
    checks.record('independent_paired_replay',exact,'Fresh harness replay plus independent bootstrap recomputation')
    coverage=set(eligible)==set(MODELS) and set(fresh)==set(MODELS) and exact
    done=dict(execution_complete=True,coverage_complete=coverage,results=result,paired=primary,
        independent_paired=independent,scores=scores,replays=replays,independent_replays=fresh,
        production_batch_size=2,flags=[e for e in checks.items if not e['passed']])
    write(root/'complete.json',done)
    return done
