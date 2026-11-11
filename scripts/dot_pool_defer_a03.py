"""Owned CPU handoff: keep E1 calibration, deploy corrected E2 scheduler afterward."""
import argparse,json,os,shlex,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
from scripts.launch_dot_pool import BASE,HOST
from scripts.gpu_process import inspect,process,ownership,pidfd_open,pidfd_send_signal
from efficient_llm.dot_pool.data import sha,write

PENDING=BASE/'pending-a03'
OLD=BASE/'execution-a02'
RUN=BASE/'run-a02'
GUARD=Path('/mnt/local/_gpu_guard/DISABLED')

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if socket.gethostname()!=HOST:raise ValueError('Host changed')
 if not a.execute:
  if PENDING.exists() or (BASE/'execution-a03').exists():raise ValueError('Duplicate handoff')
  marker=json.loads(GUARD.read_text());ident=process(marker['owner_pid'])
  if marker['output']!=str(OLD/'supervised'):raise ValueError('Wrong supervisor')
  argv=(Path('/proc')/str(ident['pid'])/'cmdline').read_bytes().split(b'\0')
  if b'scripts.train_then_burn' not in argv or str(OLD/'supervised').encode() not in argv:raise ValueError('Owner invocation changed')
  source=Path(__file__).resolve().parents[1]
  # Exact E1 numerical identity: only the E2 decay denominator changes in trainer.
  for rel in ['efficient_llm/dot_pool/model.py','efficient_llm/dot_pool/data.py']:
   if sha(source/rel)!=sha(OLD/'source'/rel):raise ValueError('E1 numerical source changed')
  oldtrain=(OLD/'source/efficient_llm/dot_pool/train.py').read_text()
  if oldtrain.replace('max(0,(total-step)/(step-start))','max(0,(total-step)/(total-start))')!=(source/'efficient_llm/dot_pool/train.py').read_text():raise ValueError('Trainer changes beyond E2 fix')
  PENDING.mkdir();frozen=PENDING/'source';files=list(source.glob('*.py'))
  for d in ['efficient_llm','scripts','eval','tests']:files+=list((source/d).rglob('*.py'))
  files += [source/r for r in ['resources/llm_pretrain_burn.py','resources/downstream_english_20261023.json','resources/a2_inputs_20261028.json','docs/DOT_pool_pilot_spec_v1.2.md','envs/dot_pool.txt']]
  hashes={}
  for f in sorted(set(files)):
   rel=f.relative_to(source);out=frozen/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,out);hashes[str(rel)]=sha(out)
  write(PENDING/'source-manifest.json',dict(commit=a.commit,files=hashes))
  write(PENDING/'owner.json',dict(identity=ident,guard=marker))
  env={**os.environ,'PYTHONPATH':str(frozen),'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}
  cmd='exec '+shlex.join([sys.executable,'-u','-m','scripts.dot_pool_defer_a03','--execute','--commit',a.commit])+' >'+shlex.quote(str(PENDING/'handoff.log'))+' 2>&1'
  subprocess.run(['tmux','new-session','-d','-s','dot_pool_20261112_handoff_a03','-c',str(frozen),cmd],env=env,check=True)
  print('DOT_A03_HANDOFF_QUEUED',json.dumps(dict(owner=ident['pid'],source_manifest_sha256=sha(PENDING/'source-manifest.json'))),flush=True);return
 owner=json.loads((PENDING/'owner.json').read_text());pid=owner['identity']['pid'];reuse=False
 write(PENDING/'status.json',dict(state='waiting_for_completed_e1_calibration',pid=os.getpid(),utc=time.time()))
 while True:
  for rel,expected in json.loads((PENDING/'source-manifest.json').read_text())['files'].items():
   if sha(PENDING/'source'/rel)!=expected:raise ValueError('Handoff source changed')
  terminal=json.loads((OLD/'supervised/supervisor.json').read_text())
  if terminal.get('finished_at'):break
  if process(pid)!=owner['identity'] or json.loads(GUARD.read_text())!=owner['guard']:raise ValueError('Ownership changed')
  status=json.loads((RUN/'status.json').read_text()) if (RUN/'status.json').exists() else {'phase':'phase0_tests'}
  ready=all((RUN/f'cal-e1-lr{i}/train_complete.json').exists() for i in range(2))
  if ready or status['phase'] not in ['phase0_tests','phase0_production_overfit','phase0_e1_lr','project_failure_recorded']:
   if not status['phase'].startswith('phase0'):raise ValueError('Already left calibration stage; refusing stop')
   if ready:
    from scripts.dot_pool_calibration_reuse import validated_score
    proofs=[]
    try:
     for i in range(2):
      cal=RUN/f'cal-e1-lr{i}';cfg=json.loads((cal/'config.json').read_text());_,proof=validated_score(cal,cfg);proofs.append(proof)
     write(PENDING/'reuse-proof.json',dict(e1_lr=proofs,verified=True));reuse=True
    except (ValueError,OSError,KeyError) as error:
     write(PENDING/'reuse-proof.json',dict(e1_lr=proofs,verified=False,error=repr(error),action='retain evidence and recalibrate in corrected queue'));reuse=False
   # Calibration workers may exit between snapshots; retry reads without signaling.
   stopped=False
   for attempt in range(20):
    fd=None
    try:
     before=inspect();records={pid:owner['identity']}
     for worker in before['workers']:
      cur=worker;seen=set()
      while cur!=pid:
       if cur<=1 or cur in seen:raise ValueError('Foreign GPU worker')
       seen.add(cur);item=process(cur);records[cur]=item;cur=item['ppid']
     fd=pidfd_open(pid)
     stable=ownership(inspect()['gpus'])==ownership(before['gpus']) and all(process(i)==r for i,r in records.items())
     if stable:
      if json.loads(GUARD.read_text())!=owner['guard']:raise ValueError('Guard owner changed')
      write(PENDING/'stop-proof.json',dict(before=before,identities=records,signal_target=pid,reuse=reuse,phase=status,utc=time.time()))
      pidfd_send_signal(fd,signal.SIGTERM);stopped=True;break
    except (ProcessLookupError,FileNotFoundError):pass
    finally:
     if fd is not None:os.close(fd)
    time.sleep(1)
   if not stopped:raise RuntimeError('No stable ownership snapshot; no signal sent')
   write(PENDING/'status.json',dict(state='waiting_native_handoff',reuse=reuse,utc=time.time()));break
  time.sleep(2)
 for _ in range(90):
  terminal=json.loads((OLD/'supervised/supervisor.json').read_text())
  if terminal.get('finished_at') and not GUARD.exists():break
  time.sleep(5)
 else:raise RuntimeError('Native handoff unverified; no launch')
 if terminal.get('handoff_error'):raise RuntimeError(terminal['handoff_error'])
 verified=json.loads((OLD/'supervised/burn-verified.json').read_text())
 if not verified.get('collective_progress_verified'):raise RuntimeError('Burn unverified')
 argv=[sys.executable,'-u','-m','scripts.launch_dot_pool','--commit',a.commit,'--name','a03','--predecessor','a02']
 if reuse:argv+=['--reuse-e1-calibration']
 subprocess.run(argv,check=True)
 write(PENDING/'status.json',dict(state='a03_submitted',reuse=reuse,utc=time.time(),submitted=json.loads((BASE/'execution-a03/submitted.json').read_text())))
if __name__=='__main__':main()
