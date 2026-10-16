"""Score remote compact prediction receipts with local filename-joined labels."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '99ff5ec0bcb187ec7678ae436b0cc0cb7d9349d5431ca1bee8b39b8fc48cc42e'


def score(log, manifest_path, dev_labels, test_labels):
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError('Wrong evaluation manifest')
    manifest = json.loads(raw)
    labels = {'dev':json.loads(dev_labels.read_text()),
              'test_official':json.loads(test_labels.read_text())}
    results = []
    for line in log.read_text().splitlines():
        if not line.startswith('INTERIM_RESULT '):
            continue
        record = json.loads(line.removeprefix('INTERIM_RESULT '))
        if record['status'] != 'passed' or record['receipt']['manifest_sha256'] != MANIFEST_SHA:
            raise ValueError('Unverified result')
        row = {k:record[k] for k in ('model','epoch','training_crop','checkpoint_sha256','inference_seconds')}
        row.update(arm=record.get('arm',record['model']),cycle=record.get('cycle',0),metrics={})
        for split,count in [('dev',900),('test_official',1500)]:
            names = [r['filename'] for r in manifest['splits'][split]['images']]
            lab = labels[split]['labels'];pred = record['predictions'][split]
            if len(names)!=count or len(pred)!=count or set(names)!=set(lab):
                raise ValueError('Filename/count mismatch')
            if any(type(p) is not int or p not in (0,1,2) for p in pred):
                raise ValueError('Invalid predictions')
            truth = [lab[n][0] if isinstance(lab[n],list) else lab[n] for n in names]
            if any(t not in (0,1,2) for t in truth):raise ValueError('Invalid labels')
            conf = [[0]*3 for _ in range(3)]
            for t,p in zip(truth,pred):conf[t][p]+=1
            correct = sum(conf[i][i] for i in range(3))
            row['metrics'][split] = {'n':count,'correct':correct,'accuracy':correct/count,
                                    'confusion':conf,'label_note':labels[split].get('note'),
                                    'labels_sha256':hashlib.sha256((dev_labels if split=='dev' else test_labels).read_bytes()).hexdigest()}
        results.append(row)
    if not results:raise ValueError('No completed evaluation in log')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('log','manifest','dev-labels','test-labels','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();rows=score(a.log,a.manifest,a.dev_labels,a.test_labels)
    with a.output.open('x') as f:json.dump({'results':rows},f,indent=2)
    for r in rows:
        print(r['arm'],r['epoch'],'dev',r['metrics']['dev']['accuracy'],'test',r['metrics']['test_official']['accuracy'])
