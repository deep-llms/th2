"""CPU-only, audited addition of dev/test predictions to future endpoint evals."""
import argparse
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import socket


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
    if socket.gethostname()!='thiennh-p6-q359-worker-0':raise RuntimeError('Wrong machine')
    ready=json.loads((a.cache/'ready.json').read_text())
    if ready['status']!='passed' or ready['manifest_sha256']!=a.manifest_sha256:
        raise ValueError('Wrong evaluation cache')
    source=Path(__file__).resolve().parent
    text=(source/'evaluate_deadline.py').read_text()
    needle="p.add_argument('--external-cache', type=Path, default=None)"
    if text.count(needle)!=1:raise ValueError('Unexpected evaluator source')
    rendered=text.replace(needle,"p.add_argument('--external-cache', type=Path, default=Path("+repr(str(a.cache))+"))")
    base=Path('/mnt/local/_outputs/deep-llms_th2')
    targets=[(base/'q359-ce7454-scratch-20261016-a04','evaluate_deadline_original.py'),
             (base/'q359-ce7454-band-20261016-a01','evaluate_deadline.py')]
    expected='71861e6b559065b21d67d2c0f9b545942b2a6a32ae9b03b189b06e94113516cf'
    for root,name in targets:
        target=root/'code'/name
        if digest(target)!=expected:raise ValueError('Evaluator differs from expected pre-patch source')
        if list(root.glob('**/*_evaluation/results.json')):raise ValueError('Evaluation already started/completed')
        if (root/'external_evaluation_patch.json').exists():raise FileExistsError('Already installed')
    for root,name in targets:
        target=root/'code'/name
        backup=target.with_suffix('.pre_external.py')
        if backup.exists():raise FileExistsError(backup)
        shutil.copy2(target,backup)
        shutil.copy2(source/'predict_external.py',root/'code/predict_external.py')
        tmp=target.with_suffix('.patch.tmp');tmp.write_text(rendered);py_compile.compile(str(tmp),doraise=True)
        tmp.replace(target)
        receipt={'status':'passed','evaluator':str(target),'before_sha256':expected,
                 'after_sha256':digest(target),'cache':str(a.cache),'manifest_sha256':a.manifest_sha256,
                 'splits':{'dev':900,'test_official':1500},'training_changed':False,
                 'scope':'Future fixed-endpoint eval adds single-pass dev and official-test predictions; local labels score them after retrieval.'}
        (root/'external_evaluation_patch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
