"""Summarize matched terminal DCLM rows from the long architecture comparison."""
import argparse
from itertools import permutations
import json
from pathlib import Path

import numpy as np


def compute(directory):
    paths=list(permutations(range(4)))
    labels=['path'+''.join(map(str,path)) for path in paths]
    odd=np.array([sum(a>b for i,a in enumerate(path) for b in path[i+1:])%2==1 for path in paths])
    result={}
    common_ids=None
    for seed in (0,1):
        panels={}
        for arm in ('direct','additive','reader'):
            d=directory/f's{seed}'/arm
            summary=json.loads((d/'summary.json').read_text())
            if summary['arm']!=arm or summary['seed']!=seed or summary['rows']!=512:
                raise ValueError('Mismatched terminal score panel')
            with np.load(d/'scores.npz',allow_pickle=False) as raw:
                ids=raw['row_ids']
                if len(np.unique(ids))!=512 or ids.shape!=(512,):
                    raise ValueError('Invalid rows')
                if common_ids is None:
                    common_ids=ids.copy()
                np.testing.assert_array_equal(ids,common_ids)
                if set(summary['nll'])!=set(raw.files)-{'row_ids'} or len(summary['nll'])!=36:
                    raise ValueError('Wrong case inventory')
                for key,expected in summary['nll'].items():
                    values=raw[key]
                    if values.shape!=(512,) or not np.isfinite(values).all():
                        raise ValueError('Invalid losses')
                    np.testing.assert_allclose(values.mean(),expected,atol=1e-10,rtol=0)
                full=np.stack([raw[key] for key in labels])
                skip={key:float(np.mean(raw[key]-raw[key.split('-')[0]]))
                      for key in raw.files if '-skip' in key}
            panels[arm]=dict(full=full,skip=skip)
        seed_report={}
        for arm,d in panels.items():
            arr=d['full']
            paired=(arr[odd]-panels['direct']['full'][odd]).mean(0)
            seed_report[arm]=dict(seen=float(arr[~odd].mean()),unseen=float(arr[odd].mean()),
                unseen_minus_direct=float(paired.mean()),median_row_path_range=float(np.median(np.ptp(arr[odd],axis=0))),
                rows_better_than_direct=int((paired<0).sum()),paired_row_quantiles=np.quantile(paired,[.1,.5,.9]).tolist(),
                skip_cost=d['skip'])
        result[str(seed)]=seed_report
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    print(json.dumps(compute(p.parse_args().directory),indent=2,allow_nan=False))
