"""FineWeb-Edu replication v1.1, with explicit interim boundary."""
from dataclasses import dataclass
from efficient_llm.a2_stage1.protocol import Run as Stage1Run, SPEC_SHA, GRID, ARMS, decision, grid_choice
REPLICATION_SPEC_SHA='b2121a4f2859f99db421273082c6be035dc5e833af6d1d071f5ed4cd381ed483'
HARD_SECONDS=24*3600
REPORT_SECONDS=20*3600
@dataclass(frozen=True)
class Run(Stage1Run):
 @property
 def run_id(self):return super().run_id.replace('ST1-dclm-','ST1-fwe-',1)
 def record(self):return dict(super().record(),dataset='fwe',part='Stage1b',run_id=self.run_id)
def phase_runs(phase,lr,include_x=True):
 if phase=='part1':return [Run(a,'3.2B',lr) for a in ARMS[:2]]
 if phase=='part2':return ([Run('X72','3.2B',lr)] if include_x else [])+[Run(a,'0.8B',lr) for a in (ARMS if include_x else ARMS[:2])]+([Run('X72','0.2B',lr)] if include_x else [])
 raise ValueError('Unknown registered part')
def cross_dataset_label(label):
 return {'persists':'replicated','grey':'weaker on FineWeb-Edu','erased':'not replicated','reversed':'not replicated','INCOMPLETE':'INCOMPLETE'}[label]
