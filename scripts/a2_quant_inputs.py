"""Read-only input receipts and calibration-document audit for A2 Q."""
import itertools
import json
from pathlib import Path
from scripts.a2_quant_io import lock, write
from scripts.a2_remaining import read_locked


def calibrations(root):
    root=Path(root);docs={k:[] for k in ('cal0','cal1','tuning','final')};offset=0
    with (root/'documents.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);group=row['group']
            if group=='heal':
                end=offset+row['gpt_neox_tokens']
                for name,lo,hi in [('cal0',0,128*4096),('cal1',128*4096,256*4096)]:
                    if offset<hi and end>lo:docs[name].append(dict(row,token_start=offset,token_stop=end))
                offset=end
            elif group in ('tuning','final'):docs[group].append(row)
    overlaps={}
    for a,b in itertools.combinations(docs,2):
        if a not in ('cal0','cal1'):continue
        ids=lambda key:{(r['split'],r['index']) for r in docs[key]}
        hashes=lambda key:{r['text_sha256'] for r in docs[key]}
        overlaps[a+':'+b]=dict(ids=sorted(ids(a)&ids(b)),content_hashes=sorted(hashes(a)&hashes(b)))
    return dict(sequence_ids={'cal0':list(range(128)),'cal1':list(range(128,256))},
        documents=docs,overlaps=overlaps,
        passed=all(not r['ids'] and not r['content_hashes'] for r in overlaps.values()),
        on_overlap='Keep exact preregistered sequence prefixes; flag QT6/calibration-dependent results. No silent resampling.')


def prepare(cfg, root, record):
    from efficient_llm.a2.data import PackedRows
    root=Path(root);base=Path(cfg['original_inputs']);source=Path(__file__).resolve().parents[1]
    cfg=dict(cfg)
    record('spec_identity',lock(source/'docs/A2_quant_spec.md')['sha256']==cfg['spec_sha256'],cfg['spec_sha256'])
    assets=read_locked(cfg['assets'])
    record('twelve_parent_manifest',len(assets)==12,dict(keys=sorted(assets)))
    sec=json.loads((base/'secondary_map.json').read_text())
    cfg['sets']={}
    verified={}
    for ds in ('dclm','fwe','nem'):
        own=(cfg['heldout'][ds],'tuning') if ds!='nem' else (sec[ds]['path'],'final')
        cfg['sets'][ds]={}
        for name,(directory,stream) in {'final':(str(base/'data'),'final'),'own_corpus':own}.items():
            manifest=lock(Path(directory)/'manifest.json')
            key=(directory,stream)
            if key not in verified:
                try:
                    rows=PackedRows(directory,stream,verify=True)
                    verified[key]=dict(sequences=len(rows),length=rows.length)
                    record('data:'+directory+':'+stream,rows.length==4096,verified[key])
                except Exception as error:
                    verified[key]=dict(error=repr(error));record('data:'+directory+':'+stream,False,verified[key])
            cfg['sets'][ds][name]=dict(root=directory,stream=stream,manifest=manifest,**verified[key])
    PackedRows(base/'data','heal',count=256,verify=True)
    cal=calibrations(base/'data');cal['spec_sha256']=cfg['spec_sha256']
    write(root/'manifests/calibration.json',cal)
    record('QT6',cal['passed'],dict(overlaps=cal['overlaps'],action=cal['on_overlap']))
    cfg['calibration_passed']=cal['passed']
    cfg['calibration_manifest']=lock(root/'manifests/calibration.json')
    cfg['data_checks']=verified_to_list(verified)
    write(root/'manifests/assets.json',dict(spec_sha256=cfg['spec_sha256'],assets=cfg['assets'],parents=assets))
    reference=json.loads((source/'resources/a2_quant_gptq_reference_20261031.json').read_text())
    from importlib.metadata import version
    import torch
    reference.update(spec_sha256=cfg['spec_sha256'],packages={k:version(k) for k in
        ('torch','transformers','accelerate','numpy','safetensors')},cuda=torch.version.cuda,
        implementation_sha256=lock(source/'efficient_llm/a2/quant.py')['sha256'],
        tf32=False,grid='exact per-row min/max; no zero range expansion; constant groups preserved and flagged',
        parameters=dict(group=128,block=128,damping=.01,actorder=False,true_sequential=True,
                        calibration_sequences=128,calibration_tokens=128*4096))
    write(root/'manifests/gptq_implementation.json',reference)
    return cfg


def verified_to_list(mapping):
    return [dict(root=k[0],stream=k[1],**v) for k,v in mapping.items()]


def pooled_followup(cfg, root):
    from eval.a2_stage1_stats import aligned_accuracy_arrays, MODELS
    from eval.a2_request_bridge import digest
    from efficient_llm.a2.quant_stats import pooled
    import math
    replays={ds:{n:read_locked(item) for n,item in cfg['prior_downstream'][ds]['replays'].items()}
             for ds in ('dclm','fwe')}
    combined={ds+'-'+n:r for ds,rows in replays.items() for n,r in rows.items()}
    # This single alignment also validates identical prompts across both corpora.
    aligned_accuracy_arrays(combined,models=tuple(combined))
    tasks={ds:aligned_accuracy_arrays(rows) for ds,rows in replays.items()}
    result=pooled(tasks)
    checks=[]
    for metric in ('mc_mean','lambada_accuracy'):
        for name,row in result['metrics'][metric].items():
            for ds in ('dclm','fwe'):
                prior=cfg['prior_paired'][ds][metric][name]['estimate']
                delta=row[ds]-prior
                checks.append(dict(metric=metric,contrast=name,dataset=ds,difference=delta,
                                   passed=math.isclose(row[ds],prior,rel_tol=0,abs_tol=1e-15)))
                row[ds]=prior  # publish the exact archived serialization, after recomputing.
            row['pooled']=(row['dclm']+row['fwe'])/2
    result.update(spec_sha256=cfg['spec_sha256'],checks=checks,
                  passed=all(x['passed'] for x in checks),inputs=cfg['prior_downstream'])
    write(Path(root)/'pooled_downstream.json',result)
    return result
