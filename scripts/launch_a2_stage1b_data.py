"""Freeze and launch the authorized Stage1b FWE CPU document shuffle; GPU work is separate."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys
from pathlib import Path
from scripts.a2_prepare_corpora import sha,write_json

def main():
 p=argparse.ArgumentParser()
 for n in ('output','data-output'):p.add_argument('--'+n,type=Path,required=True)
 for n in ('session','code-commit'):p.add_argument('--'+n,required=True)
 a=p.parse_args();source=Path(__file__).resolve().parents[1]
 if socket.gethostname()!='thiennh-p6-6y3o-worker-0' or a.output.exists() or a.data_output.exists():raise ValueError('Wrong node/existing output')
 if len(a.code_commit)!=40:raise ValueError('Source commit required')
 if subprocess.run(['tmux','has-session','-t',a.session],capture_output=True).returncode==0:raise ValueError('Existing session')
 if shutil.disk_usage(a.data_output.parent).free<170_000_000_000:raise ValueError('Insufficient new shuffle space')
 a.output.mkdir();frozen=a.output/'source';files={}
 for name in ('scripts/a2_stage1_data.py','scripts/a2_stage1b_data.py','scripts/a2_prepare_corpora.py','efficient_llm/__init__.py','efficient_llm/a2/__init__.py','efficient_llm/a2/data.py','efficient_llm/diagnostic_data.py','docs/A2_stage1_spec.md','docs/A2_stage1b_fwe_spec.md','resources/a2_stage1b_fwe_release_20261030.json','docs/results/a2_corpora_20261029/fwe_final/verification.json','resources/a2_corpora_30b_20261029.json'):
  target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target);files[name]=sha(target)
 write_json(a.output/'source-manifest.json',dict(code_commit=a.code_commit,files=files,cpu_only=True))
 argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONPATH='+str(frozen),'nice','-n','10',sys.executable,'-u','-m','scripts.a2_stage1b_data','--output',str(a.data_output)]
 command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'prepare.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],check=True)
 subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
 print('STAGE1B_SHUFFLE_SUBMITTED',a.output,flush=True)
if __name__=='__main__':main()
