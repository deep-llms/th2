"""Independent CPU verification, recording every outcome without terminating."""
import argparse
from pathlib import Path
from eval.a2_run import write
from eval.a2_suite import collect_files
from eval.a2_stage1b_stats import paired_report,PART_MODELS
from eval.a2_stage1_stats import aligned_accuracy_arrays
from scripts.a2_remaining import read_locked
from scripts.a2_stage1b_option2 import Journal,require


def verify(inputs,root):
    Path(root).mkdir(parents=True,exist_ok=True)
    journal=Journal(root)
    ids={n:read_locked(v) for n,v in inputs['identities'].items()}
    scores=inputs['scores'];original=inputs['replays'];fresh=inputs['independent_replays']
    def coverage():
        require(set(scores)==set(original)==set(fresh)==set(PART_MODELS['final']),'Missing registered model')
        return True
    journal.attempt('coverage',coverage)
    for name in PART_MODELS['final']:
        def model(name=name):
            a=collect_files({name:ids[name]},inputs['prepared'],{name:scores[name]},{name:original[name]})
            b=collect_files({name:ids[name]},inputs['prepared'],{name:scores[name]},{name:fresh[name]})
            require(a==b,'Fresh replay disagrees with recorded aggregates')
            x,y=read_locked(original[name]),read_locked(fresh[name])
            require(x['samples']==y['samples'],'Fresh replay samples differ')
            return True
        journal.attempt('model:'+name,model)
    def paired():
        replay={n:read_locked(v) for n,v in fresh.items()}
        actual=paired_report(replay,'final')
        require(actual==read_locked(inputs['paired']),'Recomputed bootstrap differs')
        return True
    journal.attempt('paired_bootstrap_10000_seed0',paired)
    def batch():
        decision=read_locked(inputs['decision'])
        arrays=aligned_accuracy_arrays({k:read_locked(v) for k,v in decision['inputs']['replays'].items()},models=('1','2'))
        diffs={t:float(a['2'].mean()-a['1'].mean()) for t,a in arrays.items()}
        mc=sum(v for t,v in diffs.items() if t!='lambada_openai')/8
        require(abs(mc)<=.001+1e-15 and abs(diffs['lambada_openai'])<=.001+1e-15,'Accuracy rule changed')
        require(scores['P300-parent']==decision['inputs']['scores']['2'],'Parent batch2 score was not reused')
        # Retain full-set failure as a diagnostic, never require it to pass.
        return True
    journal.attempt('accuracy_rule_and_exact_parent_reuse',batch)
    return dict(passed=all(e['passed'] for e in journal.events),checks=journal.events,
        status='independently_checked',models=len(scores),resamples=10000,seed=0,
        full_set_ll_scope='diagnostic_only')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','inputs-sha256','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    try:value=verify(read_locked(dict(path=a.inputs,sha256=a.inputs_sha256)),out.parent/'verification_audit')
    except Exception as error:value=dict(passed=False,status='verification_error',error=repr(error))
    write(out,value)


if __name__=='__main__':main()
