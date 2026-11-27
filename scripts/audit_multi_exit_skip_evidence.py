from pathlib import Path
import json,hashlib,datetime
import numpy as np
root=Path('results/multi_exit_20261119')
b=root/'export_a01/verified'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=root/'export_a01/results.tar.gz'
assert sha(archive)==json.loads((root/'LOCAL_REPLAY.json').read_text())['archive_sha256']
members=json.loads((b/'export-members.json').read_text())
def checked(rel):
 p=b/rel;assert sha(p)==members[rel],rel
 return p
source=json.loads(checked('execution/source.json').read_text())
sources={}
for rel in ['efficient_llm/warm_order/model.py','efficient_llm/warm_order/multi_exit.py','efficient_llm/warm_order/multi_exit_train.py','efficient_llm/warm_order/multi_exit_eval.py']:
 assert sha(Path(rel))==source[rel],rel
 sources[rel]=source[rel]
rows=[]
for p in sorted(b.glob('runs/*/dclm/summary.npz')):
 rel=str(p.relative_to(b));checked(rel)
 trace_rel=str(p.parent.parent.relative_to(b)/'train/trace.jsonl')
 trace=[json.loads(x) for x in checked(trace_rel).read_text().splitlines()]
 with np.load(p,allow_pickle=False) as z:
  assert z['costs'].shape==(3475,8)
  rows.append(dict(run=p.parent.parent.name,summary_sha256=members[rel],costs_shape=list(z['costs'].shape),summary_fields=sorted(z.files),trace_sha256=members[trace_rel],trace_records=len(trace),trace_fields=sorted(set().union(*(r.keys() for r in trace)))))
assert len(rows)==8
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Local archived DCLM summaries and train traces from eight multi-exit arms only; no forward/model loading or fitting. Archive matches existing verified replay SHA; selected members and current source match archived pins.','archive_sha256':sha(archive),'source_hashes':sources,'runs':rows,'finding':'Evaluation costs cover eight full-depth four-middle-block permutations, not short exits. Training trace short_loss is the mean over depths1/2/3 and minibatch, from evolving weights. It does not preserve a terminal-checkpoint per-example depth3-minus-depth4 contrast. These inspected artifacts cannot supply matched native-last-block deletion losses.','limits':'Not a claim about every node cache or every historical experiment; no new empirical effect size, original-document mapping or fresh-state evidence.'}
out['audit_script']='scripts/audit_multi_exit_skip_evidence.py'
out['audit_script_sha256']=sha(Path(out['audit_script']))
Path('results/layer_dependency_20261120/multi_exit_skip_evidence_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS archive/source/member hashes; inspected8DCLM summaries3475x8 and8training traces. No matched terminal skip losses in this scope.')
