"""CPU result-only archive verification and independent accuracy replay."""
import argparse,csv,hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np


def unpack_items(blob,schema):
    result={}
    with np.load(io.BytesIO(blob),allow_pickle=False) as archive:
        a={key:archive[key] for key in archive.files}
        for task,meta in schema.items():
            v=lambda key:a[task+'_'+key]
            rows=[]
            for i in range(meta['items']):
                count=int(v('count')[i]);ll=v('ll')[i,:count].tolist()
                r=dict(id=int(v('id')[i]),identity=v('identity')[i].tobytes().hex(),ll=ll,lengths=v('lengths')[i,:count].tolist(),correct=int(v('correct')[i]))
                if task=='lambada_openai':r.update(greedy=int(v('greedy')[i]),target_nll=-ll[0])
                else:r.update(chars=v('chars')[i,:count].tolist(),gold=int(v('gold')[i]),acc=int(v('acc')[i]),acc_norm=int(v('acc_norm')[i]))
                rows.append(r)
            result[task]=dict(metric=meta['metric'],accuracy=meta['accuracy'],items=rows)
    return result


def verify(archive,proof,out,expected_counts=None):
    archive=Path(archive);proof=json.loads(Path(proof).read_text());out=Path(out)
    sha=lambda x:hashlib.sha256(x).hexdigest()
    if out.exists() or archive.stat().st_size!=proof['bytes'] or sha(archive.read_bytes())!=proof['archive_sha256']:raise ValueError('Archive identity/output')
    content={}
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        if sum(m.size for m in members)>300_000_000:raise ValueError('Expanded cap')
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.suffix=='.pt' or m.name in content or m.name not in proof['files']:raise ValueError('Unsafe member')
            value=tar.extractfile(m).read()
            if sha(value)!=proof['files'][m.name]:raise ValueError('Member checksum')
            content[m.name]=value
    if set(content)!=set(proof['files']):raise ValueError('Missing member')
    report=json.loads(content['REPORT.json']);plan=json.loads(content['plan.json']);values={};tasks=expected_counts or {'hellaswag':10042,'piqa':1838,'arc_easy':2376,'lambada_openai':5153};count=0
    if len(plan['models'])!=12:raise ValueError('Model coverage')
    for model in plan['models']:
        name=model['name'];items=(json.loads(content[f'runs/{name}/items.json']) if f'runs/{name}/items.json' in content else unpack_items(content[f'runs/{name}/items.npz'],json.loads(content[f'runs/{name}/items_schema.json'])));values[name]=items
        done=json.loads(content[f'runs/{name}/complete.json']);details=json.loads(content[f'runs/{name}/details.json'])
        if done['requests_sha256']!=plan['requests_sha256'] or details['checkpoint_hashes']!=model['hashes']:raise ValueError('Identity')
        for t,n in tasks.items():
            x=items[t];rows=x['items']
            if len(rows)!=n or len({r['id'] for r in rows})!=n:raise ValueError('Item coverage')
            correct=[]
            for row in rows:
                if t=='lambada_openai':c=int(row['greedy']);assert abs(row['target_nll']+row['ll'][0])<1e-10
                else:
                    ll=np.asarray(row['ll']);chars=np.asarray(row['chars']);g=row['gold']
                    if not np.isfinite(ll).all() or (chars<=0).any():raise ValueError('Nonfinite/length')
                    acc=int(ll.argmax()==g);norm=int((ll/chars).argmax()==g);assert acc==row['acc'] and norm==row['acc_norm']
                    c=norm if x['metric']=='acc_norm' else acc
                if c!=row['correct']:raise ValueError('Accuracy replay')
                correct.append(c);count+=1
            value=sum(correct)/n
            if abs(value-x['accuracy'])>1e-12:raise ValueError('Aggregate')
            summary=next(r for r in report['rows'] if r['model']==name and r['task']==t)
            if abs(value-summary['accuracy'])>1e-12:raise ValueError('Report row')
    for model,contrasts in report['comparisons'].items():
        baseline=('pilot-D' if model.startswith('pilot-') else 'local-C')+'-'+model.split('-')[-1]
        for task,want in contrasts.items():
            a=values[model][task]['items'];b=values[baseline][task]['items'];assert [(r['id'],r['identity']) for r in a]==[(r['id'],r['identity']) for r in b]
            delta=np.array([x['correct']-y['correct'] for x,y in zip(a,b)]);counts=[int((delta==k).sum()) for k in [-1,0,1]]
            assert abs(float(delta.mean())-want['delta'])<1e-12 and counts[0]==want['losses'] and counts[2]==want['wins']
            rng=np.random.default_rng(901);draw=rng.multinomial(len(delta),np.array(counts)/len(delta),size=2000);q=np.quantile((draw[:,2]-draw[:,0])/len(delta),[.025,.975]);np.testing.assert_allclose(q,[want['lo'],want['hi']],atol=1e-12,rtol=0)
    out.mkdir(parents=True)
    for name,b in content.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    result=dict(status='PASS',models=12,task_model_rows=48,items_replayed=count,members=len(content),weights_loaded=False,paired_intervals=sum(len(x) for x in report['comparisons'].values()))
    (out/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--proof',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(verify(a.archive,a.proof,a.out)))
