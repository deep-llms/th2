#1 +60+a
#th2-export-tiered-groupreduce-complete-results-20260905-a01
set -euo pipefail
TASK_BASE=/mnt/local/_outputs/@PROJECT@
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
cd /mnt/local/@PROJECT@
/mnt/local/conda-py311/envs/eval/bin/python3.11 - "$TASK_BASE" <<'PY'
from pathlib import Path
import hashlib,io,json,math,sys,tarfile

base=Path(sys.argv[1])
run='tiered_c512_groupreduce_lb_10k_20260905_a01'
stem='tiered_c512_groupreduce_lb_eval_finetune_complete_20260905'
arms={'tiered_ranklift_c512_s10000':'tiered_ranklift_lb_t4_c512',
      'groupreduce_lb_t4_s10000':'groupreduce_matched_lb_t4'}
ft=base/f'finetune_{run}'
marker=base/f'eval_finetune_{run}.complete'
assert 'status=success' in marker.read_text()
assert 'finetune_jobs=18' in marker.read_text()
expected_eval_tasks={
 'hellaswag':{'hellaswag','hellaswag_ar','hellaswag_de','hellaswag_ru','hellaswag_vi'},
 'arc_easy':{'arc_easy','arc_ar','arc_de','arc_ru','arc_vi','arc_zh'},
 'xnli':{'xnli_en','xnli_vi','xnli_zh','xnli_de','xnli_ru','xnli_ar'},
}
members=[]
for label,arm in arms.items():
    checkpoint=base/arm/'checkpoint-10000'
    ppl=json.loads((checkpoint/'eval_ppl.json').read_text())
    bench=json.loads((checkpoint/'eval_benchmarks.json').read_text())
    assert set(ppl)=={'en','vi','zh','ru','de','ar'}
    assert len(bench)==26
    for r in ppl.values():
        assert r['num_tokens']>0 and all(math.isfinite(float(r[k])) for k in ('loss','perplexity'))
    for r in bench.values():
        assert math.isfinite(float(r.get('acc,none',r.get('acc'))))
    members.append(base/arm/'train_config.json')
    members.extend(checkpoint/n for n in ('config.json','trainer_state.json','eval.log','eval_ppl.json','eval_benchmarks.json'))
    for task,expected in expected_eval_tasks.items():
        for seed in (42,123,456):
            prefix=f'{task}_{label}_seed{seed}'
            path=ft/(prefix+'.json')
            r=json.loads(path.read_text())
            assert r['checkpoint']==str(checkpoint) and r['task']==task and int(r['seed'])==seed
            assert r['epochs']==3 and math.isclose(float(r['lr']),2e-5)
            assert set(r['eval_results'])==expected
            assert math.isfinite(float(r['train_time_s']))
            for scores in r['eval_results'].values():
                assert math.isfinite(float(scores['acc']))
                if scores.get('acc_norm') is not None: assert math.isfinite(float(scores['acc_norm']))
            model=ft/'models'/prefix/'model_state.pt'
            assert model.is_file() and model.stat().st_size>0
            members.extend((path,ft/(prefix+'.log')))
    print('RESULT_SET_VALID',arm,'PPL=6 benchmarks=26 finetune=9')
members.extend((ft/'summary.md',marker,base/f'eval_parallel_{run}.log',base/'logs'/f'eval_finetune_{run}.log'))
assert len(list(ft.glob('*.json')))==18
assert len(list(ft.glob('*.log')))==18
members=sorted(members)
assert len(members)==len(set(members))==52,len(members)
for p in members: assert p.is_file() and not p.is_symlink() and p.stat().st_size>0,p
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
manifest=''.join(f'{sha(p)}  {p.relative_to(base).as_posix()}\n' for p in members)
export=base/'result_exports';export.mkdir(exist_ok=True)
archive=export/(stem+'.tar.gz')
checksum=export/(stem+'.tar.gz.sha256')
listing=export/(stem+'.files')
for p in (archive,checksum,listing): assert not p.exists(),f'refusing overwrite: {p}'
with tarfile.open(archive,'w:gz') as tar:
    for p in members: tar.add(p,arcname=p.relative_to(base).as_posix(),recursive=False)
    payload=manifest.encode()
    info=tarfile.TarInfo('MEMBER_SHA256SUMS');info.size=len(payload)
    tar.addfile(info,io.BytesIO(payload))
with tarfile.open(archive,'r:gz') as tar:
    assert len(tar.getmembers())==53
    for p in members:
        name=p.relative_to(base).as_posix()
        h=hashlib.sha256(tar.extractfile(name).read()).hexdigest()
        assert h==sha(p),name
assert archive.stat().st_size<250_000_000,archive.stat().st_size
checksum.write_text(f'{sha(archive)}  {archive.name}\n')
listing.write_text(''.join(f'{p.relative_to(base).as_posix()}\n' for p in members)+'MEMBER_SHA256SUMS\n')
print('EXPORT_MEMBERS=53 payload_files=52')
print('ARCHIVE_BYTES',archive.stat().st_size)
print(checksum.read_text())
print('TH2 COMPLETE PAIRED RESULTS EXPORT READY')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 RESULT EXPORT COMPLETE; GPU BURNS UNTOUCHED'
