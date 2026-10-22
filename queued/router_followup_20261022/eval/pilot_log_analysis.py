"""CPU-only analysis of retained pilot logs and small checkpoint buffers."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from bdot.model import BDOTForCausalLM
from eval.packed_ppl import write_result


def replay_balance_sign(load, rate):
    """Exercise the actual update function with the observed aggregate shares.

    Each synthetic step gets the same module-count vector. Its mean share is
    therefore exactly the observed total share, with the correct uniform target.
    This tests the implementation, not unlogged historical per-update beta.
    """
    if torch.distributed.is_initialized():
        raise ValueError('Log analysis must run in a standalone CPU process')
    size = len(load)
    probe = SimpleNamespace(options=SimpleNamespace(routing='learned', balance_rate=rate, S=size),
        balance_bias=torch.zeros(size), completed_steps=torch.tensor(0),
        load_counts=torch.tensor([load]*size, dtype=torch.long),
        metric_totals=torch.zeros(9,dtype=torch.float64), training_totals=torch.zeros(5,dtype=torch.float64))
    BDOTForCausalLM.finish_update(probe)
    expected = rate*(1/size-torch.tensor(load,dtype=torch.float64)/sum(load)).sign()
    return probe.balance_bias.double(), expected


def analyze(run):
    run = Path(run)
    log = run/'metrics.jsonl'
    records = [json.loads(l) for l in log.read_text().splitlines()]
    config = json.loads((run/'config.json').read_text())['bdot']
    size,rate = config['S'],config['balance_rate']
    train = {r['step']:r for r in records if 'module_0_decisions_last_update' in r}
    validation = [r for r in records if 'eval_spike' in r]
    sign_checks=[]
    for step,r in train.items():
        counts=[r[f'module_{m}_decisions_last_update'] for m in range(size)]
        actual,expected=replay_balance_sign(counts,rate)
        sign_checks.append(dict(step=step,passed=bool(torch.allclose(actual,expected,atol=1e-9,rtol=1e-6)),
                                expected_direction=expected.sign().tolist()))
    snapshots=[]
    for step in (1000,2500,4768):
        cp=run/f'checkpoint-{step}'
        buffers={}
        for path in sorted(cp.glob('*.safetensors')):
            with safe_open(path,framework='pt',device='cpu') as f:
                for key in ('balance_bias','completed_steps','training_totals'):
                    if key in f.keys():buffers[key]=f.get_tensor(key)
        if int(buffers['completed_steps'])!=step:raise ValueError('Checkpoint step mismatch')
        last=max(s for s in train if s<=step);r=train[last]
        counts=[r[f'module_{m}_decisions_last_update'] for m in range(size)]
        totals=buffers['training_totals'].tolist()
        snapshots.append(dict(step=step,beta=buffers['balance_bias'].tolist(),
            beta_range=float(buffers['balance_bias'].max()-buffers['balance_bias'].min()),
            logged_load_step=last,module_counts_last_logged_update=counts,
            module_share_last_logged_update=[c/sum(counts) for c in counts],
            cumulative_z_mean_square=totals[2]/totals[3],
            recent_z_mean_square=r['router_z_loss_last_update']))
    # Every tenth update was logged. Bound the missing signs instead of
    # misrepresenting a 10-update beta difference as one update's response.
    bounds=[]
    previous_step,previous_beta=0,torch.zeros(size,dtype=torch.float64)
    for snap in snapshots:
        end=snap['step']; known=[r for r in sign_checks if previous_step<r['step']<=end]
        known_change=rate*sum((torch.tensor(r['expected_direction'],dtype=torch.float64) for r in known),
                              torch.zeros(size,dtype=torch.float64))
        missing=end-previous_step-len(known)
        delta=torch.tensor(snap['beta'],dtype=torch.float64)-previous_beta
        residual=delta-known_change
        bounds.append(dict(start=previous_step,end=end,logged_updates=len(known),unlogged_updates=missing,
            observed_beta_change=delta.tolist(),known_sign_change=known_change.tolist(),
            maximum_unknown_change_per_module=missing*rate,
            compatible_with_logged_signs=bool((residual.abs()<=missing*rate+1e-3).all())))
        previous_step,previous_beta=end,torch.tensor(snap['beta'],dtype=torch.float64)
    return dict(status='ok',run=str(run),metrics_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        balance_rate=rate,target_share=1/size,checkpoint_snapshots=snapshots,
        sign_replay=dict(passed=all(r['passed'] for r in sign_checks),updates=len(sign_checks),
                         failures=[r for r in sign_checks if not r['passed']]),
        historical_sign_bounds=bounds,
        sign_scope='Actual finish_update replay on every logged module-count vector; beta was not logged each update. '
                   'Checkpoint intervals only test compatibility with unknown-update bounds, not exact historical signs.',
        spikes=[r for r in validation if r['eval_spike']],
        transitions=[r for r in validation if r.get('eval_spike_reset')],
        validation_observations=len(validation),
        z_loss_series=[dict(step=r['step'],mean_square_logsumexp=r['router_z_loss_last_update'],
                            interval_z_loss=r.get('router_z_loss')) for r in train.values()],
        mean_logsumexp_history='Not logged. Mean square logsumexp is not mean logsumexp; checkpoint evaluations report the latter separately.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();write_result(a.output,analyze(a.run))


if __name__=='__main__':main()
