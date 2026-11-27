"""Recover sparse recorded training-row observations; never infer missing rows."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
 base=ROOT/'results/horizon_20261119';b=base/'export_a01/verified';archive=base/'export_a01/results.tar.gz'
 assert sha(archive)==json.loads((base/'node_handoff_verified.json').read_text())['archive_sha256']
 with tarfile.open(archive,'r:gz') as t:
  m=next(m for m in t if m.name.removeprefix('./')=='export-members.json');members=json.load(t.extractfile(m))
 verified={}
 def read(rel,lines=False):
  p=b/rel;assert sha(p)==members[rel],rel;verified[rel]=members[rel]
  return [json.loads(x) for x in p.read_text().splitlines()] if lines else json.loads(p.read_text())
 source=read('execution/source.json')
 rel='efficient_llm/warm_order/horizon_train.py';assert sha(ROOT/rel)==source[rel]
 seeds=[]
 for seed in [0,1]:
  reference=None;array_sha=None
  for arm in ['D','R','WS','WL']:
   stem=f'runs/{arm}-s{seed}/train';c=read(stem+'/recipe.json')['config'];trace=read(stem+'/trace.jsonl',True)
   assert c['seed']==seed and c['arm']==arm and c['steps']==80000 and c['batch']==32
   pin=c['expected_inputs'][c['indices']]
   assert array_sha in [None,pin];array_sha=pin
   observations={}
   for r in trace:
    step=r['step'];row=r['first_row'];assert type(step)==int and 1<=step<=80000 and type(row)==int and 0<=row<3906250
    position=(step-1)*32
    assert position not in observations
    observations[position]=row
   if reference is None:reference=observations
   else:assert reference==observations,'Arms disagree on logged training rows'
  seeds.append(dict(seed=seed,selected_training_array_sha256=array_sha,total_consumed_array_entries=80000*32,observed_positions=len(reference),observed_unique_row_ids=len(set(reference.values())),coverage_of_array_positions=len(reference)/(80000*32),observations=[dict(array_position=p,e2_destination_row=r) for p,r in sorted(reference.items())]))
 return dict(scope='Sparse first-row observations from eight authenticated Horizon traces, equal across four arms per seed; no inferred unsaved batch entries.',archive_sha256=sha(archive),verified_members=verified,trainer_source_sha256=source[rel],audit_script_sha256=sha(Path(__file__)),seeds=seeds,complete_training_arrays_recovered=False,complete_exclusion_union=False,original_documents_mapped=False,production_ready=False,limitations=['Recorded row identifiers only, not token bytes or document IDs.','A logged first row does not determine the other31 minibatch rows or unlogged minibatches.','Absence from this sparse sample does not establish an unexposed row/document.','Selected-array hashes remain unverified against actual complete array bytes.'])
if __name__=='__main__':
 result=audit();out=ROOT/'results/layer_dependency_20261120/horizon_trace_row_observations.json';out.write_text(json.dumps(result,indent=2)+'\n')
 for s in result['seeds']:print('seed',s['seed'],'positions',s['observed_positions'],'unique rows',s['observed_unique_row_ids'],'coverage',s['coverage_of_array_positions'])
