#1 +60+a
#th2-78gg-export-sampling-env-20260926-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY_ENV'
import sys,json,platform,hashlib,importlib.metadata,struct
from pathlib import Path
from datetime import datetime,timezone
import transformers,datasets,pyarrow,tokenizers
from transformers import AutoTokenizer
import pyarrow.parquet as pq
out=Path('/mnt/local/_outputs/deep-llms_th2/data_preparation/sampling_environment_20260926_a01.json')
assert not out.exists()
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
conda=[]
for p in sorted((Path(sys.prefix)/'conda-meta').glob('*.json')):
 d=json.loads(p.read_text());conda.append({k:d.get(k) for k in ['name','version','build','subdir']})
model=Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base-da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
probes={}
for lang in ['en','vi','zh','ru','de','ar']:
 p=sorted((Path('/mnt/local/_data/deep-llms_th2/data/raw')/lang).glob('*.parquet'))[0]
 texts=next(pq.ParquetFile(p).iter_batches(batch_size=16,columns=['text'])).column(0).to_pylist()
 ids=tok(texts,add_special_tokens=False)['input_ids']
 probes[lang]={'file':p.name,'documents':len(texts),'text_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),'tokens':sum(map(len,ids)),'token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
report={'host':platform.node(),'utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'python_full':sys.version,'platform':platform.platform(),'packages':dict(sorted(packages.items())),'conda_packages':conda,'tokenizer_class':type(tok).__name__,'tokenizer_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model.glob('*.json')},'prepare_data_sha256':hashlib.sha256(Path('prepare_data.py').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(Path('resources/culturax_raw_manifest.tsv').read_bytes()).hexdigest(),'tokenization_probes':probes}
out.write_text(json.dumps(report,indent=2))
print('SAMPLING_ENVIRONMENT_JSON_BEGIN')
print(json.dumps(report,indent=2))
print('SAMPLING_ENVIRONMENT_JSON_END')
PY_ENV
