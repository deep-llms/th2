#1 +120+a
#th2-wc69-a2-why-rest-launch-20261105-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CHECK'
import json
from pathlib import Path
from scripts.a2_remaining import read_locked
from scripts.a2_prepare_corpora import verify_packed,sha
from scripts.launch_a2_depth_partb import predecessor
from efficient_llm.a2_why.common import write
cfg=json.loads(Path('resources/a2_why_rest_plan_20261105_a01.json').read_text())
out=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-rest-20261105-a01')
assert not out.exists()
predecessor(cfg,out)
pause=read_locked(dict(path='/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-pause-20261104-a01/pause/saved_state.json',sha256='bc8a633435bab3f1d410a9bec7c3cc341cc25a3b65fa6a47bad1d17d9d431faa'))
assert pause['global_step']==24000 and len(pause['hashes'])==22
for name,digest in pause['hashes'].items():
    assert sha(Path(pause['checkpoint'])/name)==digest, name
print('PAUSE_24000_ALL_22_SHA_PASS',flush=True)
data=read_locked(cfg['why_data']);assert data['status']=='complete'
verified=[]
for ds,streams in data['streams'].items():
    for seed,stream in streams.items():
        manifest=read_locked(dict(path=stream['path']+'/manifest.json',sha256=stream['sha256']))
        verify_packed(Path(stream['path']),manifest)
        verified.append(dict(dataset=ds,seed=seed,manifest_sha256=stream['sha256']))
        print('WHY_STREAM_ACTUAL_BYTES_PASS',ds,seed,flush=True)
predecessor(cfg,out)
write(str(out)+'.preflight.json',dict(pause_22_sha=True,verified_streams=verified,caveat=cfg['data_caveat']))
CHECK
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_why \
 --plan resources/a2_why_rest_plan_20261105_a01.json \
 --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-why-rest-20261105-a01 \
 --session a2_why_rest_wc69_20261105_a01 \
 --stage rest --code-commit f3f99379ac7d45a2558b52b1b4e0d69fadba20dc
