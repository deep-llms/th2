"""Consolidate known E2 exposure rows; deliberately not a complete exclusion set."""
from pathlib import Path
import hashlib,json
import numpy as np
from scripts.audit_horizon_trace_rows import audit as horizon_audit,ROOT
from scripts.audit_pilot_trace_rows import audit as pilot_audit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=ROOT/'results/layer_dependency_20261120'
 hp=b/'horizon_trace_row_observations.json';pp=b/'pilot_trace_row_observations.json'
 h=horizon_audit();p=pilot_audit();assert h==json.loads(hp.read_text()) and p==json.loads(pp.read_text())
 rp=b/'router_packed_rows.json';r=json.loads(rp.read_text());npz=ROOT/r['local_output'];assert sha(npz)==r['output_sha256']
 recp=b/'e2_order_recovered.json';assert sha(recp)==r['recovery_proof_sha256'];rec=json.loads(recp.read_text());op=ROOT/rec['output'];assert sha(op)==r['order_sha256']==rec['actual_sha256']
 order=np.load(op,mmap_mode='r',allow_pickle=False)
 assert p['stream']==r['stream'] and p['order_sha256']==sha(op)
 groups={};scope={}
 with np.load(npz,allow_pickle=False) as z:
  for name,desc in r['selections'].items():
   ids=z[name+'_e2_row'];assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==desc['destination']['ordered_int64le_sha256']
   np.testing.assert_array_equal(order[ids],z[name+'_packed_heal_row'])
   groups[name]=ids.copy();scope[name]='Complete IDs for this previously authenticated router selection only.'
  assert set(z['horizon_monitor_e2_row'].tolist())<=set(groups['horizon_valid'].tolist())
 for family,proof in [('horizon_trace',h),('pilot_trace',p)]:
  for seed in proof['seeds']:
   name=f'{family}_s{seed["seed"]}';groups[name]=np.array([v['e2_destination_row'] for v in seed['observations']],dtype='<i8')
   scope[name]='Sparse logged minibatch-first rows only; full selected training array unavailable.'
 # Verify Horizon stream pins as well; matching integer IDs alone is insufficient.
 for arm in ['D','R','WS','WL']:
  for seed in [0,1]:
   rel=f'runs/{arm}-s{seed}/train/recipe.json';path=ROOT/'results/horizon_20261119/export_a01/verified'/rel
   assert sha(path)==h['verified_members'][rel];c=json.loads(path.read_text())['config'];assert c['data']==p['stream']
   assert c['expected_inputs'][c['data']+'.bin']==p['token_sha256'] and c['expected_inputs'][c['data']+'.starts']==p['starts_sha256']
 union=np.unique(np.concatenate(list(groups.values())));assert np.array_equal(union,np.array(sorted(set().union(*(set(v.tolist()) for v in groups.values()))),dtype='<i8'))
 packed=order[union];assert len(np.unique(packed))==len(union)
 membership=np.zeros(len(union),dtype=np.uint16);arrays={'known_e2_rows':union,'known_packed_heal_rows':packed};labels=[]
 for bit,(name,ids) in enumerate(groups.items()):
  flag=1<<bit;membership[np.searchsorted(union,ids)]|=flag
  assert np.array_equal(union[(membership&flag)!=0],np.unique(ids))
  arrays[name+'_e2_rows']=ids;labels.append(dict(name=name,bit=bit,observations=len(ids),unique_rows=len(np.unique(ids)),scope=scope[name]))
 arrays['source_membership_mask']=membership;assert (membership>0).all()
 out=ROOT/'temp/layer_dependency_review/recovered_indices/known_exposure_rows.npz'
 if out.exists():
  with np.load(out,allow_pickle=False) as z:
   assert set(z.files)==set(arrays)
   for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
 else:
  with out.open('xb') as f:np.savez_compressed(f,**arrays)
 router=np.unique(np.concatenate([groups[k] for k in r['selections']]))
 traces=np.unique(np.concatenate([v for k,v in groups.items() if '_trace_' in k]))
 proof=dict(scope='Known rows in one E2 train stream only; lower-bound historical exposure inventory, not a fresh reservation.',input_proofs={str(x.relative_to(ROOT)):sha(x) for x in [hp,pp,rp,recp]},token_sha256=p['token_sha256'],starts_sha256=p['starts_sha256'],order_sha256=sha(op),groups=labels,monitor_alias='Horizon monitor128 is contained in horizon_valid; not a separate group.',known_router_unique_rows=len(router),known_trace_unique_rows=len(traces),shared_router_trace_rows=len(np.intersect1d(router,traces)),additional_trace_rows=len(np.setdiff1d(traces,router)),known_union_rows=len(union),output=str(out.relative_to(ROOT)),output_sha256=sha(out),output_bytes=out.stat().st_size,audit_script_sha256=sha(Path(__file__)),document_union_verified=False,complete_historical_union=False,production_ready=False,excluded=['Unlogged training rows and remaining historical experiments.','Direct dev/final streams, which must not receive the E2 train permutation.','Original document boundaries/text identities and duplicate-document occurrences.'])
 (b/'known_exposure_rows.json').write_text(json.dumps(proof,indent=2)+'\n')
 print({k:proof[k] for k in ['known_router_unique_rows','known_trace_unique_rows','shared_router_trace_rows','additional_trace_rows','known_union_rows','output_bytes']})
if __name__=='__main__':main()
