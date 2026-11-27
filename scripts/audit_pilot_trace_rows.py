"""Recover sparse pilot trace row IDs and compare only authenticated equal streams."""
from pathlib import Path
import hashlib,json
import numpy as np
from scripts.audit_pilot_exclusion_inventory import read_export,ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
 payload,proof=read_export(ROOT/'results/warm_order_20261118')
 source=json.loads(payload['execution/source.json']);trainer='efficient_llm/warm_order/train.py'
 assert sha(ROOT/trainer)==source[trainer]
 rp=ROOT/'results/layer_dependency_20261120/router_packed_rows.json';r=json.loads(rp.read_text());arrays=ROOT/r['local_output'];assert sha(arrays)==r['output_sha256']
 recp=ROOT/'results/layer_dependency_20261120/e2_order_recovered.json';assert sha(recp)==r['recovery_proof_sha256'];rec=json.loads(recp.read_text())
 ready_path=ROOT/'results/a2_dot_data_20261118/ready_verified.json';assert sha(ready_path)==rec['ready_receipt_sha256']
 ready=json.loads(ready_path.read_text())['ready']['dot_e2'];stream=ready['streams']['train'];assert r['stream']==ready['path']+'/train'
 order_path=ROOT/rec['output'];assert sha(order_path)==r['order_sha256']==rec['actual_sha256'];order=np.load(order_path,mmap_mode='r',allow_pickle=False)
 seeds=[];members={}
 with np.load(arrays,allow_pickle=False) as z:
  for seed in [0,1]:
   merged={};pin=None;arm_sizes={}
   for arm in ['D','R','W']:
    prefix=f'runs/{arm}-s{seed}'
    for suffix in ['recipe.json','trace.jsonl']:members[prefix+'/'+suffix]=proof['files'][prefix+'/'+suffix]
    c=json.loads(payload[prefix+'/recipe.json'])['config'];assert c['seed']==seed and c['arm']==arm and c['batch']==32 and c['steps']==12000
    assert c['data']==r['stream'] and c['expected_inputs'][c['data']+'.bin']==stream['token_sha256'] and c['expected_inputs'][c['data']+'.starts']==stream['starts_sha256']
    this_pin=c['expected_inputs'][c['indices']];assert pin in [None,this_pin];pin=this_pin
    trace=[json.loads(x) for x in payload[prefix+'/trace.jsonl'].splitlines()];seen=set();arm_sizes[arm]=len(trace)
    for row in trace:
     step=row['step'];i=row['first_row'];assert type(step)==int and 1<=step<=12000 and type(i)==int and 0<=i<len(order)
     position=(step-1)*32;assert position not in seen;seen.add(position)
     if position in merged:assert merged[position]['e2_destination_row']==i
     else:merged[position]=dict(array_position=position,e2_destination_row=i,packed_heal_row=int(order[i]),observed_arms=[])
     merged[position]['observed_arms'].append(arm)
   ids=np.array([v['e2_destination_row'] for v in merged.values()],dtype=np.int64)
   matches={}
   for name in r['selections']:
    dest=z[name+'_e2_row'];np.testing.assert_array_equal(order[dest],z[name+'_packed_heal_row'])
    matches[name]=np.intersect1d(ids,dest).tolist()
   seeds.append(dict(seed=seed,selected_array_sha256=pin,consumed_entries=384000,logged_records_by_arm=arm_sizes,observed_positions=len(merged),coverage=len(merged)/384000,all_three_arms_observed_positions=sum(len(v['observed_arms'])==3 for v in merged.values()),same_stream_router_matches=matches,observations=[merged[p] for p in sorted(merged)]))
 return dict(scope='Authenticated pilot sparse first-row observations only; all shared positions agree across arms.',archive_sha256=proof['archive_sha256'],verified_members=members,trainer_sha256=source[trainer],router_proof_sha256=sha(rp),router_arrays_sha256=sha(arrays),order_sha256=sha(order_path),stream=r['stream'],token_sha256=stream['token_sha256'],starts_sha256=stream['starts_sha256'],audit_script_sha256=sha(Path(__file__)),seeds=seeds,complete_training_arrays_recovered=False,original_documents_mapped=False,quality_impact_measured=False,production_ready=False,limits=['Additional R-s0 step40 observation is not independently logged by D/W.','Same row values follow matching selected-array pins; complete bytes still unavailable.','Sparse logged positions cannot estimate full exposure or prove disjointness.','Router-fit comparisons are not final test contamination; direct dev/final streams are not joined.'])
if __name__=='__main__':
 d=audit();(ROOT/'results/layer_dependency_20261120/pilot_trace_row_observations.json').write_text(json.dumps(d,indent=2)+'\n')
 for s in d['seeds']:print('seed',s['seed'],'positions',s['observed_positions'],'all-arms',s['all_three_arms_observed_positions'],'matches',s['same_stream_router_matches'])
