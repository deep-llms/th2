#1 +30+a
#th2-q359-ce7454-odd-eval-fix-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /mnt/local/conda-py311/envs/ce7454_env/bin/python - <<'PATCH'
from pathlib import Path
import hashlib,importlib.util,json,shutil,torch
from datetime import datetime,timezone
torch.set_num_threads(2)
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-band-20261016-a01')
p=root/'code/research_band_stem.py'
q=Path('ce7454/src/research_band_stem.py')
expected='b73b13e490c4de67f0268a56ea10ef3c41c62aa30c31686549696f7f3e2098d9'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
assert json.loads((root/'controller.json').read_text())['status']=='running'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('old',p);new=module('new',q)
for transform in ('haar','phases'):
 torch.manual_seed(0);a=old.BandStemResNet(transform).eval()
 torch.manual_seed(0);b=new.BandStemResNet(transform).eval()
 with torch.no_grad():
  x=torch.randn(1,3,512,512);torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
  for h,w in ((201,200),(200,201),(201,203)):
   x=torch.randn(1,3,h,w);pad=torch.nn.functional.pad(x,(0,w%2,0,h%2),mode='replicate')
   torch.testing.assert_close(b(x),a(pad),rtol=0,atol=0)
 print('PASSED',transform,'native512 exact; odd shapes equal explicit extension',flush=True)
backup=p.with_name('research_band_stem.training.py')
assert not backup.exists()
shutil.copy2(p,backup)
tmp=p.with_suffix('.patch.tmp');tmp.write_bytes(q.read_bytes());tmp.replace(p)
receipt={'at':datetime.now(timezone.utc).isoformat(),'status':'passed','old_sha256':expected,
 'new_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'native512_outputs_exact':True,
 'scope':'Odd evaluation shapes only. Loaded trainers and native512 function unchanged. Original training source manifest retained; this receipt records the evaluation compatibility overlay.'}
(root/'odd_evaluation_patch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PATCH
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
