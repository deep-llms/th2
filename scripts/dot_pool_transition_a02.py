"""One-time authorized calibration reschedule; only the pinned a01 supervisor is signaled."""
import argparse,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from scripts.gpu_process import inspect,process,pidfd_open,pidfd_send_signal,ownership
from efficient_llm.dot_pool.data import sha,write
from scripts.launch_dot_pool import BASE,HOST

PID=168897
START=432302041
COMMAND='d6b9663465ce87f68b2e5c4bca76f02a9d715bfed61ff7ff18c1b124e6ce59d7'

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 old=BASE/'execution-a01';run=BASE/'run-a01';guard=Path('/mnt/local/_gpu_guard/DISABLED')
 proof=BASE/'transition-a01-a02.json'
 if socket.gethostname()!=HOST or proof.exists() or (BASE/'execution-a02').exists():raise ValueError('Wrong host or duplicate transition')
 status=json.loads((run/'status.json').read_text())
 if status['phase']!='phase0_e1_lr' or list(run.glob('E[12]-*')):raise ValueError('Calibration already advanced; no stop')
 marker=json.loads(guard.read_text())
 if marker['owner_pid']!=PID or marker['output']!=str(old/'supervised'):raise ValueError('Guard owner differs')
 source=json.loads((old/'source-manifest.json').read_text())
 if any(sha(old/'source'/r)!=h for r,h in source['files'].items()):raise ValueError('Frozen source changed')
 before=inspect();ident=process(PID)
 if ident['start_ticks']!=START or ident['command_sha256']!=COMMAND:raise ValueError('Supervisor identity changed')
 args=(Path('/proc')/str(PID)/'cmdline').read_bytes().split(b'\0')
 if b'scripts.train_then_burn' not in args or str(old/'supervised').encode() not in args:raise ValueError('Not expected native supervisor')
 if not before['workers'] or not set(before['workers'])<= {169333,169334}:raise ValueError('GPU workers changed; no stop')
 records={PID:ident}
 for worker in before['workers']:
  cur=worker;seen=set()
  while cur!=PID:
   if cur<=1 or cur in seen:raise ValueError('Worker not owned')
   seen.add(cur);item=process(cur);records[cur]=item;cur=item['ppid']
 fd=pidfd_open(PID)
 try:
  if ownership(inspect()['gpus'])!=ownership(before['gpus']) or any(process(i)!=r for i,r in records.items()):raise ValueError('Ownership changed before signal')
  if json.loads(guard.read_text())!=marker or json.loads((run/'status.json').read_text())['phase']!='phase0_e1_lr':raise ValueError('Phase/guard changed')
  write(proof,dict(reason='Repack two calibration fits onto four GPUs each, same global batch; preserve all partial logs; fix explicit E2 step table',before=before,identities=records,signal_target=PID,utc=time.time(),state='signal_requested'))
  pidfd_send_signal(fd,signal.SIGTERM)
 finally:os.close(fd)
 print('DOT_OWNED_SUPERVISOR_STOP_REQUESTED',PID,flush=True)
 # Native supervisor alone performs descendant cleanup, free checks and burn handoff.
 for _ in range(84):
  time.sleep(5);terminal=json.loads((old/'supervised/supervisor.json').read_text())
  if terminal.get('finished_at'):
   if terminal.get('handoff_error'):raise RuntimeError('Native handoff failed: '+terminal['handoff_error'])
   if not guard.exists():break
 else:raise RuntimeError('Native handoff not yet verified; no launch')
 verified=json.loads((old/'supervised/burn-verified.json').read_text())
 if not verified.get('collective_progress_verified'):raise RuntimeError('Native burn unverified')
 result=json.loads(proof.read_text());result.update(state='native_handoff_verified',terminal=terminal,verified=verified);write(proof,result)
 print('DOT_NATIVE_HANDOFF_VERIFIED',flush=True)
 subprocess.run([sys.executable,'-u','-m','scripts.launch_dot_pool','--commit',a.commit,'--name','a02','--predecessor','a01'],check=True)
 result.update(state='a02_submitted',submitted=json.loads((BASE/'execution-a02/submitted.json').read_text()));write(proof,result)
if __name__=='__main__':main()
