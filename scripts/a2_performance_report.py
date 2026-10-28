"""Validate and summarize discarded A2 performance probe artifacts, offline."""
import argparse,json,math,statistics
from pathlib import Path
from efficient_llm.a2.performance import SUITES
from efficient_llm.a2.data import write_json


def summarize(root,suite='initial'):
    variants=SUITES[suite]
    root=Path(root);results={}
    for name,option in variants.items():
        path=root/name/'result.json'
        if not path.is_file():continue
        r=json.loads(path.read_text())
        if (r['status']!='complete' or r.get('suite','initial')!=suite or r['variant']!=name or r['options']!=option.record()
            or not r['coverage']['exact_order'] or r['coverage']['examples']!=1024
            or r['coverage']['world_size']!=8 or not r['teacher_no_gradient']
            or len(r['all_step_seconds'])!=16 or len(r['lr_updates'])!=16):raise ValueError('Probe contract differs: '+name)
        steady=r['all_step_seconds'][5:]
        if any(not math.isfinite(v) or v<=0 for v in steady):raise ValueError('Invalid timing')
        if abs(statistics.mean(steady)-r['steady_mean_seconds'])>1e-8:raise ValueError('Timing summary mismatch')
        g=r['first_gradient'];passed=g['relative_l2']<=.02 and g['cosine']>=.9998 and abs(r['validation_delta'])<=.002
        if passed!=r['numerical_passed']:raise ValueError('Numerical status mismatch')
        results[name]=r
    if not results:return dict(status='incomplete',completed=0,rows=[])
    ref=results['reference'];repeat=results.get('reference_repeat')
    for r in results.values():
        if any(r[k]!=ref[k] for k in ('assets_sha256','data_sha256','lr_updates','code_commit')):raise ValueError('Comparison identities differ')
        if abs((r['validation_nll']-ref['validation_nll'])-r['validation_delta'])>1e-8:raise ValueError('Validation delta mismatch')
    timing_ref=repeat or ref
    rows=[dict(variant=n,numerical_passed=r['numerical_passed'],
        step_seconds=r['steady_mean_seconds'],median_seconds=r['steady_median_seconds'],
        speedup=timing_ref['steady_mean_seconds']/r['steady_mean_seconds'],tokens_per_second=r['tokens_per_second'],
        peak_gib=r['peak_allocated_bytes']/2**30,cold_compute_seconds=r['cold_first_update_compute_seconds'],
        gradient_relative_l2=r['first_gradient']['relative_l2'],gradient_cosine=r['first_gradient']['cosine'],
        validation_delta=r['validation_delta']) for n,r in results.items()]
    stable=bool(repeat and repeat['numerical_passed'] and abs(repeat['steady_mean_seconds']/ref['steady_mean_seconds']-1)<=.1)
    usable=[r for r in rows if r['numerical_passed'] and r['variant'] not in ('reference','reference_repeat') and r['speedup']>=1.05]
    return dict(status='complete' if len(results)==len(variants) else 'incomplete',completed=len(results),
        repeat_stable=stable,timing_reference='reference_repeat' if repeat else 'reference',rows=rows,
        fastest_screened_candidate=min(usable,key=lambda r:r['step_seconds'])['variant'] if stable and usable else None,
        note='Screening only; promotion requires reviewing profile, cold overhead, numerical limits and real integration. No default mutation.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--suite',choices=SUITES,default='initial');p.add_argument('--output',required=True)
    a=p.parse_args();r=summarize(a.root,a.suite);write_json(a.output,r);print(json.dumps(r))


if __name__=='__main__':main()
