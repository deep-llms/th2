"""Owner-approved v1.7: F8 first, then F7; no reruns of F0–F6."""

def runs():
    values=[]
    for cut in (3,9):
        for arm in ('P300','P1T'):
            values.append(dict(run_id=f'Y3-dclm-{arm}-L{cut}-3.2B',arm=arm,
                               removed=list(range(22-cut,22)),phase='F8'))
    for arm in ('S72','S188'):
        values.append(dict(run_id=f'ST1-dclm-{arm}-S1-3.2B-lr1e-3',arm=arm,
                           removed=list(range(16,22)),phase='F7'))
    return [dict(r,dataset='dclm',budget='3.2B',steps=12207,tokens=12207*64*4096,
                 lr=.001,stream_seed=2030,seed=0,variant='drop') for r in values]


def capacity(es):
    if any(k not in es for k in (3,6,9,12)):return 'unverified'
    if es[3]<es[6]<es[9]<es[12]:return 'A supported'
    if es[12]-es[3]<.003:return 'A not supported'
    return 'mixed'
