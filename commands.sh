#1 +30+a
#th2-6y3o-efficient-deployment-verify-20261026-a03
set -euo pipefail
python3 - <<'REMOTE'
import ast, datetime, hashlib, json, os, re, socket, subprocess
from pathlib import Path
ROOT=Path('/mnt/local/deep-llms_th2')
OUT=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-migration-20261026-a01')
EXPECTED={'.gitignore': '781f45d5f693442be7fecc5ad175f870546a1ccefe3607b86e9026465af93c31', 'AGENTS.md': 'd4fbc0c60fa575499fe7b5adc72757e5ad58692fafae45cc7a6269ea94cb418c', 'README.md': '87e80e82586c46624af62593c0f6d4d72abd987900e3c21f0659aa8b3274a377', 'docs/B200_REAL_DATA_PILOT_20261017.md': 'ace19c0097ed7b490cc8ea23d63bdbd812b66884e421f6fd78862af80126cde2', 'docs/B200_SMOKE_20261017.md': '98296c050b5fa4f0bc5a7075721090a25efe256f19f77abf1b43d8f9c527d118', 'docs/CURRENT_TASK.md': '8cf238c1c616734089e7b2d6a1d63ee66aeaad7fc5f09b48194e771d58b29fcc', 'docs/EFFICIENT_LLM_RESEARCH_20261026.md': '1a7d9c13f83fe95fee0bee3ef8d9b3e2b04e6f5153e7937c00d53fe94c34d996', 'docs/FIRST_ARCHITECTURE_HYPOTHESIS.md': 'f4e7eebabf075a6d08284a0f90b5420e430138b6f63f0c3c426deb891e032bce', 'docs/GIT_PUSH.md': '71348345d0279986fcd633c248f93eef54761fa61ed23ca11f9888c1b757e86a', 'docs/GPU_SAFETY.md': 'd81f945d5a0ac6d41d211242d5801e70d255767753591370271548f1b84f9d21', 'docs/JOBS.md': '5bfec9a7be4e864f9bcef293ea3571354e154c16bab101debd4631255e9b6fba', 'docs/PROJECT_IDEA.md': '61689d0983c407fab6a8b989e901fd4e8cf20deaa1b1a48a58ccf71a004bd16e', 'docs/PROJECT_NOTES.md': '6d350108aac603a5fad341e727d2aceb74d4f4a725d8debec3bfbe38e403b23e', 'docs/TRAINING.md': '2309d58ac2c4b1313c89e730c73f8f69e393225b65962c9b409ffb3acd5cf423', 'docs/VALIDATION.md': '84e45d5bb18b37ffaac71ed947ac3d4b9a16fdfb4a4644132f3a744d2dda8e77', 'docs/commands.md': '644467835c1dbac7bb59c5536f84f1a6cf1a314e29f0520cc9ccd9877d2ea77b', 'docs/results/efficient_llm_migration_20261026.json': '82b5cc890a49dfa239286869bb8ec2f3996eefb0e25063d4cf9298e37ba5bb48', 'docs/results/template_b200_smoke_20261017.json': 'd65a6b2794945c3d7d346a9e6ca0bd81641e024e586d93528b3cbb76ab4c4eb6', 'docs/results/template_realdata_pilot_20261017.json': '190c6c7d90ec489c56fcf4840c6c3aa75ca3e4e0a4ccb0f414cd9aa19ca4c3e8', 'eval/packed_ppl.py': '103d1c5256da259a5c199079c108b056797cceef84e63a8f0601b42dae8203bd', 'jobs.example.json': '27dcf04212e850a497776c9fcbd2d2d126b37b1e405cbf3011d2124d81cef341', 'jobs.training.example.json': '904e5b95f12c296e96f78410d00720d39673ffbb07564769b8367b23f22188f6', 'packed_evaluation.py': 'e8ad9b65831ed44522670b229e8724e9e3ff69163c79f901371fb20c274877e7', 'packed_fa4.py': '0e2fb01603c7fdcb1ac40d4fb85c4838b127d5fc54d643ff6728bce3d08304be', 'packed_training.py': '5d592ecdd3575633feb164ab2408925ddea23dd010f9b9dd701300ddf6cf7af0', 'project.example.json': '7988cbd87cab4280ee321d081f363a9e92fe3a5404ad3cd50aba538fad94f5cd', 'requirements.txt': '36801f82a3c93dc4f31d077ed9ec2f4ce32861f69321eafe87bbd36ac235f554', 'resources/accelerate_config.example.yaml': '17daff31ff51043c933231fdd2ebdda8dc0014f3c59e10c5a8121a4e652144d6', 'resources/llm_pretrain_burn.py': '2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174', 'resume_recipe.py': '0aee4eb8ff915202a8b665f836fb64b5f3c1cc3d6ded36720975833340eea2e9', 'run_experiments.py': '752c7913adbd4b2b1e58ae21a535b20462f2622be00ce5f48a9e76ec80e3172d', 'scripts/dropbox_access.py': 'de747b2b5771aac7d996708f75287c5a711b085c0cdf191a5905e7c32c1dc6ab', 'scripts/example_job.py': '724707158ab7409aa5e99a0f345bb77bb897678754796921a0b55da98cb52add', 'scripts/gpu_status.py': '63506e500bd24c4b105234116cd601543e1cdf5bd50c599b22550730b31c38eb', 'scripts/train_qwen3_0.6b_baseline.sh': '1e25d634d3d3193bf62a08b46460e5d1f6b9465bcc95abf86a5d427711c8b5c6', 'scripts/training_results.py': 'de631f088ec98a573e5044245cb63412d177fb654a623eb278cf315bf697ded3', 'scripts/verify_manifest.py': '4be05466e3a0144cbb6930438e0c94df8597e7ae725c27ff20fe3db340db13c7', 'tests/ml/ddp_isolation_probe.py': '347a047302c2561dbb6c5944df7d12735cf5f1c277a9bdbe2ff391ab5de57bc1', 'tests/ml/ddp_periodic_eval_probe.py': '825e5f7d911390a8308cb62657fa83b4e3cb980c2a5ace29b4f0501a2cb57bb7', 'tests/ml/ddp_resume_probe.py': '0517b628ae9a431175bc4cf2669db5b004df3214aa54ec30693e22886331ac0a', 'tests/ml/fa4_gpu_probe.py': '8af2b1b499f5ae66723b10d4cec8d4ade0712aca87a55d66bced3eb443b36c72', 'tests/ml/gpu_workflow_probe.py': '01d0c09610f90d385c3ec3e7d4e0ce3337f5bf1185a8e99955e6d0519b78716a', 'tests/ml/real_data_pilot.py': 'd0cace4c1d8c79a3ee2240487bfc7b3c298b608b0d4ccea0e98e012db06ea859', 'tests/ml/test_baseline.py': '14e9db8f8e14810692020eb42c257dd7507bdaf933892b1910013fe7ae595a53', 'tests/ml/test_document_isolation.py': '8d7e0607d45486279ffbf894c915f0d696a7bb7fd35f2744bdc002573db7ec4f', 'tests/ml/test_fa4.py': '00f3b0bf9c5381834933dec639707e44a3daf5e261aab3680789bdefc62c9b85', 'tests/ml/test_packed_ppl.py': '598c8ceddd4fd59ba75bfb220974732655233a64b3566c06322d73addb81f69f', 'tests/ml/test_periodic_eval.py': 'f5b923769ee08311e516b2a11a6a61f7ba94fc1ba9dcc08e1784b32ad243902d', 'tests/ml/test_resume_recipe.py': '9d49893bc900bbf25cdb32fb007651b908509f3d78a12c5c1f1b90f3c12b07a1', 'tests/ml/test_stopping.py': 'eee30013bccbf7d99ca0ec749de2dc82f7c04852ff12383ac113cf9dd1b95be5', 'tests/ml/test_training_data.py': 'e7b2cb8c56ef134aea13a7a21d23f668bd44df1adcc542c76023b7fc98aa030c', 'tests/ml/test_training_workflow.py': 'a5f6c45a7b62261929ef51c1b7ff39a775f9c927604cb6c140f2543a556d8f91', 'tests/test_training_results.py': 'f7e4177043468d19029bcbf0a03debf942e7d3d610273a14be13ca8251b5370c', 'tests/test_utilities.py': 'd235ddf5f02549ed8e9b5ee74aeb94c8495c35d8f81395f7183bba345046cc49', 'train.py': '87fcf8302164aace3255facda0d9bd278e201288589ac4e27c6693e49b96d82b', 'training_data.py': '37982be7fe0ddf3a068c7d625125eae6da3ea2073a9e88a97f02474d86ff7028'}
def emit(event,**data):
    print(event+' '+json.dumps(data,sort_keys=True),flush=True)
if socket.gethostname()!='thiennh-p6-6y3o-worker-0' or ROOT.is_symlink() or ROOT.resolve()!=ROOT or Path.cwd()!=ROOT:
    raise RuntimeError('Unexpected deployment target')
cleanup=json.loads((OUT/'cleanup_result.json').read_text())
for rel,digest in EXPECTED.items():
    p=ROOT/rel
    if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Missing/mismatched deployed source: '+rel)
    if p.suffix=='.py': ast.parse(p.read_text(),filename=rel)
allowed_top={Path(rel).parts[0] for rel in EXPECTED}|{'commands.sh','_run_log_','_dl_active.txt','_install_active.txt','_previous_run_status.log'}
if {p.name for p in ROOT.iterdir()}!=allowed_top: raise RuntimeError('Unexpected top-level leftovers')
for forbidden in ['bdot','configs','queued','packed_fa2.py','envs','temp']:
    if (ROOT/forbidden).exists(): raise RuntimeError('Old/private directory survived: '+forbidden)
# Recheck exactly the same GPU worker identities; no GPU computation or signals.
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in apps.splitlines():
    pid,uuid=line.split(',');pid=int(pid)
    fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    rows.append({'pid':pid,'gpu_uuid':uuid.strip(),'start_ticks':int(fields[19])})
rows.sort(key=lambda r:r['pid'])
if rows!=cleanup['gpu_processes']: raise RuntimeError('GPU ownership changed')
# Verify communication progress from the live launcher's existing log.
log=Path('/proc/581838/fd/1').resolve(strict=True)
if not log.is_relative_to(Path('/mnt/local/_outputs/deep-llms_th2')) or not log.is_file(): raise RuntimeError('Unexpected burn log')
with log.open('rb') as handle:
    head=handle.read(128*1024).decode(errors='replace')
    handle.seek(max(0,log.stat().st_size-128*1024))
    tail=handle.read().decode(errors='replace')
def burn_records(text, event):
    return [dict(re.findall(r"([a-z_]+)=([^\s]+)", s)) for s in text.splitlines(keepends=True)
            if s.startswith(event+' ') and s.endswith('\n')]
ready=burn_records(head,'gpu_burn_ready')
progress=burn_records(tail,'gpu_burn_progress')
if {int(r['rank']) for r in ready}!=set(range(8)) or any(int(r['world_size'])!=8 or float(r['collective_probe_sum'])!=36 for r in ready):
    raise RuntimeError('Incomplete eight-rank burn readiness')
if len(progress)<2 or int(progress[-1]['completed_cycles'])<=int(progress[-2]['completed_cycles']) or float(progress[-1]['completed_collective_payload_gib'])<=float(progress[-2]['completed_collective_payload_gib']):
    raise RuntimeError('Burn progress not advancing')
emit('BURN_LOG',path=str(log),ready=ready,latest_progress=progress[-2:])
metadata_code="import importlib.metadata as m,json,sys; names=['torch','transformers','accelerate','datasets','flash-attn','flash-attn-4']; print(json.dumps({'python':sys.executable,'versions':{n:m.version(n) for n in names if list(m.distributions(name=n))}},sort_keys=True))"
metadata=json.loads(subprocess.check_output(['/mnt/local/conda-py311/envs/train/bin/python3.11','-c',metadata_code],text=True,timeout=60))
emit('EXISTING_TRAIN_ENV',**metadata)
model=Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6')
if (model/'config.json').is_file():
    config=json.loads((model/'config.json').read_text())
    emit('AVAILABLE_MODEL',path=str(model),config={k:config.get(k) for k in ['model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','hidden_size','head_dim']})
raw=Path('/mnt/local/_data/deep-llms_th2/raw')
emit('AVAILABLE_DATA',path=str(raw),entries=sorted(p.name for p in raw.iterdir()))
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'code_root':str(ROOT),'verified_files':len(EXPECTED),'source_sha256':EXPECTED,'old_code_absent':True,'gpu_processes_unchanged':True,'gpu_processes':rows,'train_env':metadata,'burn_progress':progress[-2:],'burn_ready':ready}
(OUT/'deployment_result.json').write_text(json.dumps(result,indent=2)+'\n')
emit('DEPLOYMENT_COMPLETE',**{k:v for k,v in result.items() if k not in ['source_sha256','burn_progress','burn_ready']})
REMOTE
