"""Frozen Stage1 v3.3 arms, LR selection and screening decisions."""
from dataclasses import dataclass,asdict
from decimal import Decimal
import math

SPEC_SHA='9ebe0d0a377f0107f7058bf5e4899b5380cbe5f4f518b09091ac533d2c9f254e'
ARMS=('P300','P1T','X72')
GRID=(1e-4,3e-4,1e-3)
STEPS={'0.2B':762,'0.8B':3051,'3.2B':12207}

@dataclass(frozen=True)
class Run:
 arm:str
 budget:str
 lr:float
 def __post_init__(self):
  if self.arm not in ARMS or self.budget not in STEPS or self.lr not in (*GRID,3e-5,3e-3):raise ValueError('Unregistered Stage1 run')
 @property
 def parent(self):return 'S72' if self.arm=='X72' else self.arm
 @property
 def teacher(self):return 'P300' if self.arm=='X72' else self.arm
 @property
 def steps(self):return STEPS[self.budget]
 @property
 def run_id(self):
  m,e=format(self.lr,'.8e').split('e');lr=m.rstrip('0').rstrip('.')+'e'+str(int(e))
  return f'ST1-dclm-{self.arm}-S1-{self.budget}-lr{lr}'
 def record(self):return dict(asdict(self),dataset='dclm',set='S1',part='Stage1',suffix='',run_id=self.run_id,parent=self.parent,teacher=self.teacher,steps=self.steps,heal_tokens=self.steps*262144)


def grid_choice(rows,lrs=GRID,arms=('P300','P1T')):
 if len(rows)!=len(lrs)*len(arms) or {(r['arm'],r['lr']) for r in rows}!={(a,lr) for a in arms for lr in lrs}:raise ValueError('Incomplete/duplicate LR grid')
 means={}
 for lr in lrs:
  pair=[r for r in rows if r['lr']==lr]
  if all(r.get('status')=='complete' and not r.get('diverged') and math.isfinite(r['val_tune_post']) for r in pair):means[lr]=sum(r['val_tune_post'] for r in pair)/len(pair)
 if not means:raise ValueError('Every grid LR diverged')
 best=min(means,key=lambda lr:(means[lr],lr));low=min(means.values());chosen=min(lr for lr,val in means.items() if val<=low+.001+1e-12)
 edge=3e-5 if best==1e-4 else 3e-3 if best==1e-3 else None
 return dict(chosen_lr=chosen,best_lr=best,means=means,edge_lr=edge if tuple(lrs)==GRID else None)


def decision(value,kind):
 finite=value is not None and math.isfinite(value);d=Decimal(str(value)) if finite else None
 if kind=='length':clauses={'INCOMPLETE':not finite,'persists':finite and d>=Decimal('.008'),'reversed':finite and d<=Decimal('-.008'),'erased':finite and abs(d)<Decimal('.003')}
 elif kind=='state':clauses={'INCOMPLETE':not finite,'anneal first':finite and d>=Decimal('.01'),'prune before you anneal':finite and d<=Decimal('-.01'),'no effect':finite and abs(d)<Decimal('.003')}
 else:raise ValueError('Unknown decision')
 return dict(value=value,clauses=clauses,label=next((k for k,v in clauses.items() if v),'grey'))


def remaining(lr,include_x=True):
 arms=ARMS if include_x else ARMS[:2]
 return [Run(a,b,lr) for b in ('3.2B','0.8B') for a in arms]+([Run('X72','0.2B',lr)] if include_x else [])
