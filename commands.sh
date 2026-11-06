#1 +40+a
#th2-wc69-depth-resume-20261106-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'RESUME'
from pathlib import Path
from types import SimpleNamespace
import json
from scripts.a2_remaining import read_locked
from scripts.a2_quant_io import lock,write
from scripts.train_then_burn import digest
from scripts.a2_depth_continue import continue_run,validate_existing
from efficient_llm.a2_depth.resume import select_checkpoint
from efficient_llm.a2_depth.heal_long import run_identity
from efficient_llm.a2_depth.recovery import completed_arm
base=Path('/mnt/local/_outputs/deep-llms_th2');root=base/'wc69-a2-depth-partc-20261103-a01'
pf=read_locked(dict(path=str(base/'wc69-a2-depth-resume-preflight-20261106-a01.json'),sha256='bbb9c3d0dc28b8637fad183d852ce75cc7c1cba06753faee2fa3186fb1575aca'))
assert pf['pause_22_sha'] and pf['selected_matches_pause'] and pf['data_actual_hash_pass'] and not pf['rejected']
assert read_locked(pf['why_decision'])['coverage_complete']
record,cfg,frozen=validate_existing(root,Path(pf['depth_launch']['input_plan']['path']),'e2af735a59127f1116dbf8d671fe5aa95f1bb12f')
assert record==pf['depth_launch'] and cfg==pf['depth_config']
assert cfg['arm_order']==['X72','P300','P1T']
pause=read_locked(pf['pause_receipt']);out=Path(cfg['results_root'])/'heals/ST1-dclm-P300-S1-12.8B-lr1e-3'
selected,rejected=select_checkpoint(out/'trainer',run_identity(cfg,'P300'),8)
assert str(selected)==pf['selected'] and selected.name=='checkpoint-24000' and not rejected
for name,sha in pause['hashes'].items():assert digest(selected/name)==sha,name
completed_arm(Path(cfg['results_root'])/'heals/ST1-dclm-X72-S1-12.8B-lr1e-3',run_identity(cfg,'X72'))
proof=Path('resources/a2_depth_resume_patch_20261106.json').resolve()
assert json.loads(proof.read_text())['original_source_manifest']==record['source_manifest']
handoff=base/'wc69-a2-depth-resume-handoff-20261106-a01.json';assert not handoff.exists()
write(handoff,dict(mode='burn',predecessor=pf['predecessor'],preflight=lock(base/'wc69-a2-depth-resume-preflight-20261106-a01.json'),full_why_backup_sha256='70cfa5bf71266078d89776d6683c891888271f7039caaf362a99f904cd3e7a93'))
continue_run(SimpleNamespace(output=root,plan=Path(record['input_plan']['path']),code_commit='e2af735a59127f1116dbf8d671fe5aa95f1bb12f',session='a2_depth_partc_wc69_resume_20261106_a01',recovery_handoff=handoff,recovery_patch=proof))
print('DEPTH_RESUME_LAUNCH_RECORD',json.dumps(json.loads((root/'continuations.json').read_text())),flush=True)
RESUME
