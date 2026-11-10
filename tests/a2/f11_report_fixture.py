"""Full 42-row temperature/LM report+archive fixture; no GPU or real results."""
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
from eval.a2_family_f11 import write,read,GUARD_GRID,choose_temperature
from eval.a2_family_f10 import select_models
from eval.a2_family_f9_v110 import select
from eval.a2_family_f11_report import report
from scripts.a2_family_f11_backup import backup
from scripts.a2_quant_io import lock
from efficient_llm.diagnostic_data import sha256
from scripts.a2_family_f11_queue import descriptors

def run():
    entries=sum(select(read('resources/a2_family_f9_plan_20261110.json')['models'])[0],[])
    _,groups=select_models(entries,read('resources/a2_family_f10_plan_20261111.json'));entries=sum(groups,[])
    with tempfile.TemporaryDirectory() as d:
        base=Path(d);root=base/'result';root.mkdir();previous=base/'old.md';previous.write_text('PRESERVED F10\n')
        for e in entries:
            f=root/'calibration'/e['model_id'];value=e['final_nll'];arr={str(t):[value+(t-100)**2/10000+i/100 for i in range(3)] for t in GUARD_GRID}
            means={t:sum(v)/3 for t,v in arr.items()};sel=choose_temperature(means)
            for split in ('tuning','final'):write(f/(split+'.json'),dict(per_sequence_nll=arr,nll=means,per_sequence_entropy=[1,2,3],mean_entropy=2.,token_counts=[4095]*3,tokens=12285,sequences=3))
            write(f/'result.json',dict(model=e,selection=sel,FT17=dict(expected=means['100'],difference=0.,passed=True),flags=[],complete=True))
        for item in descriptors():
            f=root/'heals'/item['run_id'];orders=[]
            for rank in range(8):
                p=f/('sample-order-rank'+str(rank)+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('w') as stream:
                    for b in range(3051*2):stream.write(json.dumps({'sample_ids':list(range(b*32+rank*4,b*32+rank*4+4))})+'\n')
                orders.append(lock(p))
            def score(v):return dict(loss=v,per_sequence_nll=[v-.01,v,v+.01],sequences=3,tokens=12285)
            x=.03 if item['arm']=='P1T' else 0.
            write(f/'result.json',dict(arm=item['arm'],raw_order_files=orders,baseline=score(3.),pre={'final':score(3.5+x)},scores={'final':score(3.1+x)},delta_post=.1+x,flags=[]))
            (f/'final').mkdir();(f/'final/model.safetensors').write_bytes(b'weights-not-in-scientific-archive')
        cfg={'entries':entries,'previous_report':lock(previous),'output_root':str(base),'f10':{'output':str(base/'f10')}}
        # Downstream numerical reconstruction is covered by reused F9/F10 tests;
        # this fixture exercises report completeness, ordering, both NLL tables,
        # true 8-rank trace audit and the real archive byte verification.
        down={'accuracy':{'contrasts':[{'E':{'MC8':{'estimate':0.,'ci95':[0.,0.]}}}]},'complete':True}
        with patch('eval.a2_family_f11_report.downstream',return_value=down):report(cfg,root)
        complete=read(root/'report_complete.json');assert complete['complete'],complete
        text=(root/'REPORT_FULL_v1.13.md').read_text();assert text.startswith('PRESERVED F10\n');assert text.index('| Pair | E_LM')<text.index('| Corpus | Pair')
        assert len(read(root/'calibration.json')['models'])==42
        assert abs(read(root/'LM_comparison.json')['E_LM']['estimate']-.03)<1e-12
        source=base/'source';source.mkdir();(source/'code.py').write_text('# fixture')
        cfg['source_manifest']=write(base/'source-manifest.json',dict(files={'code.py':sha256(source/'code.py')}))
        for k in ('cpu_proof','f10_config_lock','preparation_proof','data_manifest','assets'):cfg[k]=write(base/(k+'.json'),{})
        write(base/'config.json',cfg)
        arc=base/'f10/backup/family_f10_scientific.tar.gz';arc.parent.mkdir(parents=True);arc.write_bytes(b'old F10 archive')
        write(arc.parent/'verified.json',dict(verified=True,archive_sha256=sha256(arc)))
        proof=read(backup(root,cfg)['path']);assert proof['verified']
        assert not any('model.safetensors' in name for name in proof['files'])
        assert sum('sample-order-rank' in name for name in proof['files'])==16
        print('F11_FULL_REPORT_BACKUP_FIXTURE_PASS',complete,'bytes',(root/'REPORT_FULL_v1.13.md').stat().st_size,'archive_members',proof['members'])

if __name__=='__main__':run()
