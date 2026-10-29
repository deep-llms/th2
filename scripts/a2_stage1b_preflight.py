"""Stage1b preflight report, matching Stage1 tests with 20/24h limits."""
import argparse,json,math
from pathlib import Path
from efficient_llm.a2.data import write_json
from efficient_llm.a2.protocol import lr_for_update,warmup_steps
from efficient_llm.diagnostic_data import sha256
def read(p):return json.loads(Path(p).read_text())

def finalize_preflight(root,inputs):
 inputs=read(inputs);prov=inputs['provenance'];parents={p:read(root/(p+'.json')) for p in ('P300','P1T','S72')}
 numerical=read(root/'numerical.json');t8=read(root/'throughput.json');t17=read(root/'t17/T17.json')
 records=[*parents.values(),numerical,t8,t17]
 if any(r['provenance']!=prov for r in records):raise ValueError('Mixed preflight provenance')
 if not numerical['passed'] or not t8['passed'] or not t17['passed']:raise ValueError('Required numerical/throughput/T17 failed')
 if not all(parents[p]['T1']['passed'] and parents[p]['sanity']['passed'] for p in ('P300','P1T')):raise ValueError('Annealed parent T1 failed')
 eligible=parents['S72']['T1']['passed'] and parents['S72']['sanity']['passed'] and inputs['T3']['passed']
 step=t8['step_seconds'];count=51108 if eligible else 35088
 # Include loading/full validation/export, possible one extra edge pair,
 # downstream and cleanup explicitly; measured fit timings refine later ETAs.
 seconds=(count+1524)*step+(15 if eligible else 12)*180+3600
 tests=dict(T1={p:v['T1'] for p,v in parents.items()},T3=inputs['T3'],T5=numerical['T5_loss'],T12=numerical,
  T7=inputs['data_manifest']['T7'],T8=t8,T17=t17,T16=dict(passed=read(root/'t17/result.json')['roundtrip_difference']<=1e-6),
  T6={b:{str(i):lr_for_update(i,n,3e-4) for i in (1,warmup_steps(n),math.ceil(n/2),n-1,n)} for b,n in [('0.2B',762),('0.8B',3051),('3.2B',12207)]},
  T18=dict(inputs['data_manifest']['T18'],parent_loss={p:dict(fwe=v['baseline']['secondary']['loss'],culturax=v['baseline']['final']['loss'],flag=v['baseline']['secondary']['loss']>=v['baseline']['final']['loss']) for p,v in parents.items() if v['baseline'] is not None}))
 report=dict(status='passed',provenance=prov,eligible_x72=eligible,parents=parents,tests=tests,step_seconds=step,
  annealing_gain=None if parents['S72']['baseline'] is None else parents['S72']['baseline']['final']['loss']-parents['P300']['baseline']['final']['loss'],
  estimated_remaining_seconds=seconds,report_before_launch=seconds+inputs.get('preflight_cap_seconds',7200)>20*3600,
  preflight_artifacts={str(p.relative_to(root)):sha256(p) for p in [*(root/(k+'.json') for k in parents),root/'numerical.json',root/'throughput.json',root/'t17/T17.json',root/'t17/result.json']})
 write_json(root/'tests.json',report);write_json(root/'parents.json',parents)
 print('STAGE1B_PREFLIGHT_PASSED',json.dumps(dict(eligible_x72=eligible,step_seconds=step,estimated_remaining_hours=seconds/3600)),flush=True)
 return report


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();finalize_preflight(a.output,a.inputs)
if __name__=='__main__':main()
