"""Four consumer-aware hypotheses; reuse the validated confirmation workflow."""
import argparse
from pathlib import Path

from scripts import proxy_confirmation_queue as common
from scripts.proxy_overnight_queue import read

CASES = (
    ('consumer-only', 'P7-simple-consumer-only', .1, 42),
    ('consumer-cosine', 'P7-simple-consumer', .1, 42),
    ('consumer-targetgrad', 'P7-simple-consumer-targetgrad', .1, 42),
    ('targetgrad-low', 'P7-simple-targetgrad', .01, 42),
)
EVAL_PYTHON = common.EVAL_PYTHON


def predecessor_reference(root, guard=Path('/mnt/local/_gpu_guard/DISABLED')):
    root=Path(root)/'supervised'
    state=read(root/'supervisor.json');run=read(root/'run/run.json')
    burn=read(root/'burn-verified.json');scope=read(root/'run/downstream-scope.json')
    from scripts.proxy_screen_evaluation import validate_scope
    validate_scope(scope)
    result=read(root/'run/downstream/full/summary.json')
    if (state.get('training_status')!='ok' or state.get('training_returncode')!=0
            or not state.get('finished_at') or run['status']!='ok'
            or len(run['jobs'])!=17 or any(j['status']!='ok' for j in run['jobs'])
            or not burn.get('collective_progress_verified')
            or not state.get('burn',{}).get('collective_progress_verified')
            or burn['session']!=state['burn']['session'] or guard.exists()
            or scope.get('kind')!='confirmation3'
            or result['status']!='passed' or result['diagnostic'] is not False
            or set(result['checkpoints'])!=set(scope['checkpoints'])):
        raise ValueError('Confirmation queue and burn handoff must be complete; GPUs untouched')
    return scope['reference']


def make(recipe_path, output, rows, predecessor):
    return common.make(recipe_path,output,rows,predecessor,cases=CASES,
                       module='scripts.proxy_consumer_queue')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('summarize','evaluate','preflight'))
    p.add_argument('--root',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--stage',choices=('numerics','smoke','full'))
    a=p.parse_args()
    if a.mode=='summarize':common.summarize(a.root,a.reference,cases=CASES)
    elif a.mode=='preflight':common.preflight(a.root,a.reference)
    elif a.stage:common.evaluate(a.root,a.reference,a.stage,cases=CASES,scope_kind='consumer4')
    else:p.error('Evaluation requires a stage')
