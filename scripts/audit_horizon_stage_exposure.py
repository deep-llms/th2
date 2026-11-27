"""Audit archived path counts only; no model imports, inference or remote access."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def audit():
    base=ROOT/'results/horizon_20261119'
    archive=base/'export_a01/results.tar.gz'
    assert sha(archive)==json.loads((base/'node_handoff_verified.json').read_text())['archive_sha256']
    with tarfile.open(archive,'r:gz') as t:
        member=next(m for m in t if m.name.removeprefix('./')=='export-members.json')
        manifest=json.load(t.extractfile(member))
    b=base/'export_a01/verified'
    verified={}
    def read(rel):
        p=b/rel;digest=sha(p);assert digest==manifest[rel],rel
        verified[rel]=digest
        return json.loads(p.read_text())
    pins=read('execution/source.json')
    sources={name:sha(ROOT/name) for name in ['efficient_llm/warm_order/model.py','efficient_llm/warm_order/progressive.py','efficient_llm/warm_order/horizon_train.py']}
    assert all(pins[n]==s for n,s in sources.items())
    # Only the pure schedule function is executed; no imports or model classes.
    orders=tuple(tuple((start+direction*j)%4 for j in range(4)) for direction in (1,-1) for start in range(4))
    tree=ast.parse((ROOT/'efficient_llm/warm_order/progressive.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='path_for')
    ns={'ORDERS':orders}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pinned_schedule>','exec'),ns)
    results=[]
    for arm in ['D','R','WS','WL']:
        for seed in [0,1]:
            stem=f'runs/{arm}-s{seed}/train'
            config=read(stem+'/recipe.json')['config']
            done=read(stem+'/complete.json')
            assert config['arm']==done['arm']==arm and config['seed']==done['seed']==seed
            assert config['steps']==done['step']==80000
            expected=Counter(','.join(map(str,ns['path_for'](arm,s,80000))) for s in range(80000))
            assert dict(expected)==done['counts'],stem
            depths=Counter();edges=Counter();slots=[[0]*4 for _ in range(4)]
            for key,count in done['counts'].items():
                path=tuple(map(int,key.split(',')));depths[len(path)]+=count
                for slot,block in enumerate(path):slots[block][slot]+=count
                for u,v in zip(path,path[1:]):edges[f'{u+1}->{v+1}']+=count
            results.append(dict(arm=arm,seed=seed,updates_by_depth=dict(sorted(depths.items())),block_by_slot_update_counts=slots,directed_pair_updates=dict(sorted(edges.items())),missing_pairs_1based=[f'{u+1}->{v+1}' for u in range(4) for v in range(4) if u!=v and f'{u+1}->{v+1}' not in edges]))
    return dict(scope='Archived terminal counts reconciled to pinned schedule; counts are update exposures, not distinct examples, gradients, useful contribution or effect sizes.',archive_sha256=sha(archive),verified_members=verified,verified_sources=sources,audit_script_sha256=sha(Path(__file__)),runs=results,production_ready=False,model_executed=False,limits='No weights/current node inspected; repeated windows and original-document disjointness unresolved. No causal attribution of failed routing.')

if __name__=='__main__':
    result=audit()
    out=ROOT/'results/layer_dependency_20261120/horizon_stage_exposure.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['runs'][::2]:print(r['arm'],r['updates_by_depth'],r['block_by_slot_update_counts'][0],r['directed_pair_updates'],r['missing_pairs_1based'])
