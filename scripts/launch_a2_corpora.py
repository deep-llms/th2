"""Launch only offline CPU tokenization; never acquire GPUs or alter burns."""
import argparse,hashlib,json,os,shlex,shutil,socket,subprocess,sys
from pathlib import Path
from importlib.metadata import version


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('output','plan'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('session','host','code-commit'):p.add_argument('--'+k,required=True)
    a=p.parse_args();source=Path(__file__).resolve().parents[1]
    if socket.gethostname()!=a.host or a.output.exists():raise ValueError('Wrong host or existing launch output')
    if subprocess.run(['tmux','has-session','-t',a.session],capture_output=True).returncode==0:raise ValueError('Session already exists')
    plan=json.loads(a.plan.read_text());root=Path(plan['root'])
    if not root.resolve().is_relative_to(Path('/mnt/local/_data/deep-llms_th2')):raise ValueError('Unexpected data root')
    if shutil.disk_usage(root.parent).free<1_000_000_000_000:raise ValueError('Need 1TB free')
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):raise ValueError('Pinned commit required')
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'source';identities={}
    names=['scripts/a2_prepare_corpora.py','efficient_llm/__init__.py','efficient_llm/a2/__init__.py',
           'efficient_llm/a2/data.py','efficient_llm/diagnostic_data.py']
    for name in names:
        target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target);identities[name]=sha(target)
    frozen_plan=frozen/'plan.json';shutil.copyfile(a.plan,frozen_plan)
    (a.output/'source-manifest.json').write_text(json.dumps(dict(code_commit=a.code_commit,files=identities,
        plan_sha256=sha(frozen_plan),runtime={k:version(k) for k in ('tokenizers','pyarrow','numpy')},
        cpu_only=True),indent=2)+'\n')
    argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
        'TOKENIZERS_PARALLELISM=false','RAYON_NUM_THREADS=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'MKL_NUM_THREADS=1','PYTHONPATH='+str(frozen),'nice','-n','10',sys.executable,'-u',
        str(frozen/'scripts/a2_prepare_corpora.py'),'--plan',str(frozen_plan),'--deadline-hours','48']
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'prepare.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('CORPUS_CPU_PREPARATION_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()
