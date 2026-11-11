"""Validate completed E1 LR evidence before reusing its scores across an E2-only fix."""
import json
from pathlib import Path
from efficient_llm.dot_pool.data import sha
from efficient_llm.dot_pool import SPEC_SHA

def validated_score(root,cfg):
 root=Path(root);old=json.loads((root/'config.json').read_text());done=json.loads((root/'train_complete.json').read_text());evaluation=json.loads((root/'final_eval.json').read_text())
 for key in ['study','arm','seed','steps','lr','batch','microbatch','model','data','data_ready_sha256','spec_sha256']:
  if old.get(key)!=cfg.get(key):raise ValueError('Calibration differs: '+key)
 if old['actual_world']!=4 or old['spec_sha256']!=SPEC_SHA:raise ValueError('Calibration rank/spec mismatch')
 complete=json.loads((root/'final_state/complete.json').read_text())
 if complete['step']!=cfg['steps'] or complete['world']!=4:raise ValueError('Calibration checkpoint incomplete')
 for name,record in complete['files'].items():
  f=root/'final_state'/name
  if Path(name).name!=name or f.stat().st_size!=record['bytes'] or sha(f)!=record['sha256']:raise ValueError('Calibration checkpoint hash mismatch')
 if not all(f'rank{i}.pt' in complete['files'] for i in range(4)) or 'state.pt' not in complete['files']:raise ValueError('Incomplete rank state')
 if done['step']!=cfg['steps'] or done.get('flags') or done.get('benchmark'):raise ValueError('Calibration training incomplete/flagged')
 if evaluation['step']!=cfg['steps'] or evaluation.get('flags'):raise ValueError('Calibration final metrics incomplete/flagged')
 metric=evaluation['metrics']['id_dev'];score=metric['accuracy']
 if not (0<=score<=1) or metric['targets']!=1000:raise ValueError('Invalid ID-dev score')
 return score,dict(origin=str(root),files={name:sha(root/name) for name in ['config.json','train_complete.json','final_eval.json','final_state/complete.json']},checkpoint_members_verified=True,score=score)
