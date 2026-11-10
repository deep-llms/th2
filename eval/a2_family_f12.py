"""Family v1.16 registered F12 identities and readout helpers."""
from eval.a2_family_f9 import read
from scripts.a2_quant_io import write as atomic_write
from eval.a2_family_f11 import document_bootstrap,interval,lm_reading
SPEC_SHA='3438a50a40a2570d500cd2799c4afbc34906ba341555c0e7d04d5cc030a9fc38'
def write(path,value):return atomic_write(path,dict(value,spec_sha256=SPEC_SHA))
def descriptors():
    return [dict(run_id=f'W-dclm-{a}-FFN5440-{b}',dataset='dclm',arm=a,budget=b,steps=n,stream_seed=2030,phase='F12b',keep=5440)
            for b,n in [('0.2B',762),('0.8B',3051),('3.2B',12207)] for a in ('P300','P1T')]
def budget_reading(first,last):
    delta=last-first
    return 'flat' if abs(delta)<max(.2*abs(first),.003) else 'changing upward' if delta>0 else 'changing downward'
