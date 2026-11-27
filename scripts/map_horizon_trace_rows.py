"""Join sparse trace observations with authenticated router rows in the same stream."""
from pathlib import Path
import hashlib,json
import numpy as np
from scripts.audit_horizon_trace_rows import audit,ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 observed=audit()
 rp=ROOT/'results/layer_dependency_20261120/router_packed_rows.json';r=json.loads(rp.read_text())
 npz=ROOT/r['local_output'];assert sha(npz)==r['output_sha256']
 recovery_path=ROOT/'results/layer_dependency_20261120/e2_order_recovered.json'
 assert sha(recovery_path)==r['recovery_proof_sha256']
 recovery=json.loads(recovery_path.read_text());order_path=ROOT/recovery['output']
 assert sha(order_path)==recovery['actual_sha256']==r['order_sha256']
 ready_path=ROOT/'results/a2_dot_data_20261118/ready_verified.json'
 assert sha(ready_path)==recovery['ready_receipt_sha256']
 ready=json.loads(ready_path.read_text())['ready']['dot_e2'];stream=ready['streams']['train']
 assert r['stream']==ready['path']+'/train'
 b=ROOT/'results/horizon_20261119/export_a01/verified'
 for arm in ['D','R','WS','WL']:
  for seed in [0,1]:
   rel=f'runs/{arm}-s{seed}/train/recipe.json';p=b/rel
   assert sha(p)==observed['verified_members'][rel]
   c=json.loads(p.read_text())['config'];assert c['data']==r['stream']
   assert c['expected_inputs'][c['data']+'.bin']==stream['token_sha256']
   assert c['expected_inputs'][c['data']+'.starts']==stream['starts_sha256']
 order=np.load(order_path,mmap_mode='r',allow_pickle=False);assert order.shape==(3906250,)
 results=[]
 with np.load(npz,allow_pickle=False) as z:
  for s in observed['seeds']:
   ids=np.array([v['e2_destination_row'] for v in s['observations']],dtype=np.int64)
   matches={}
   for name in r['selections']:
    dest=z[name+'_e2_row'];packed=z[name+'_packed_heal_row']
    np.testing.assert_array_equal(order[dest],packed)
    both=np.intersect1d(ids,dest)
    matches[name]=[dict(e2_destination_row=int(i),packed_heal_row=int(order[i])) for i in both]
   results.append(dict(seed=s['seed'],observed_positions=len(ids),comparisons=matches,packed_row_observations=[dict(array_position=v['array_position'],e2_destination_row=v['e2_destination_row'],packed_heal_row=int(order[v['e2_destination_row']])) for v in s['observations']]))
 out=dict(scope='Partial Horizon training-trace rows versus previously authenticated pilot/Horizon router selections on identical E2 train token/starts hashes only.',audit_script_sha256=sha(Path(__file__)),horizon_trace_audit_script_sha256=observed['audit_script_sha256'],horizon_archive_sha256=observed['archive_sha256'],router_proof_sha256=sha(rp),router_arrays_sha256=sha(npz),order_sha256=sha(order_path),stream=r['stream'],token_sha256=stream['token_sha256'],starts_sha256=stream['starts_sha256'],seeds=results,original_document_mapping=False,complete_exclusion_union=False,quality_impact_measured=False,limits=['Positive matches establish reuse of these specific E2 rows, not general overlap rates.','Pilot fit is router-training data, not final evaluation.','Zero matches in sparse Horizon-fit/valid comparisons do not establish full row or document disjointness.','Original boundaries/text hashes and full selected training arrays remain missing.'])
 (ROOT/'results/layer_dependency_20261120/horizon_trace_router_matches.json').write_text(json.dumps(out,indent=2)+'\n')
 for s in results:print('seed',s['seed'],{k:len(v) for k,v in s['comparisons'].items()})
if __name__=='__main__':main()
