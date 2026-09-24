"""Paired fixed-dev analysis for the matched frozen-backbone student stage."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .distill import file_hash, settings
from .joint_training import require
from .screen import write_json
from .statistics import paired_bootstrap


def report_results(args, config):
    from dataclasses import asdict
    from .joint import load_inputs
    args.output.mkdir(parents=True, exist_ok=False)
    rows, curves = [], []
    common_code = None
    try:
        for seed in range(2):
            _, dev, fingerprints = load_inputs(config,args.data_dir,seed)
            expected_counts=(dev.valid[:,:-1] & dev.valid[:,1:])
            if dev.segments is not None:
                expected_counts &= dev.segments[:,:-1] == dev.segments[:,1:]
            expected_counts=expected_counts.sum(-1)
            audit_dir=args.audit_root/f'audit-seed-{seed}'
            audit=json.loads((audit_dir/'complete.json').read_text())
            require(audit['status']=='ok' and audit['ready_for_students'] and audit['inputs']==fingerprints,
                    'Invalid feedback audit')
            with np.load(audit_dir/'eval.npz',allow_pickle=False) as values:
                counts=values['target_counts'].copy()
                require(np.array_equal(counts,expected_counts)
                        and np.array_equal(values['sequence_indices'],np.arange(len(counts))), 'Audit targets/order changed')
                losses={'FeedbackOff':values['off_loss_sums'].copy(),'Deep':values['deep_loss_sums'].copy()}
            require(all(abs(float(losses[arm].sum()/counts.sum())-audit['nll'][arm])<1e-12
                        for arm in ('FeedbackOff','Deep')), 'Audit NLL receipt mismatch')
            identities=[]
            for arm in ('LM','PCC'):
                directory=args.runs_dir/f'seed-{seed}-{arm}'
                require(not (directory/'failure.json').exists(),'Student has a failure record')
                identity=json.loads((directory/'identity.json').read_text());identities.append(identity)
                complete=json.loads((directory/'complete.json').read_text())
                require(identity['purpose']=='correction_distillation' and identity['arm']==arm
                        and identity['seed_index']==seed and identity['inputs']==fingerprints
                        and identity['config']==config and identity['parent']==audit['parent']
                        and identity['audit_sha256']==file_hash(audit_dir/'complete.json')
                        and identity['settings']==asdict(settings(config))
                        and identity['sigma_delta']==audit['calibration']['sigma_delta']
                        and identity['world_size']==8 and identity['physical_gpus']==list(range(8)),
                        'Mismatched student identity')
                if common_code is None:
                    common_code=identity['code']
                require(identity['code']==common_code,'Mixed student source versions')
                require(complete['status']=='ok' and complete['updates']==settings(config).updates
                        and complete['input_tokens']==settings(config).updates*settings(config).tokens_per_update
                        and complete['world_size']==8 and complete['teacher_unchanged']
                        and complete['checkpoint_verified'],'Incomplete student run')
                saved=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
                require(saved['format']=='distill-v1' and saved['identity']==identity
                        and saved['update']==settings(config).updates and len(saved['rank_rng'])==8,
                        'Incorrect final student checkpoint')
                digest=hashlib.sha256()
                for name,tensor in saved['student'].items():
                    require(tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),'Nonfinite student weights')
                    digest.update(name.encode());digest.update(tensor.contiguous().numpy())
                require(digest.hexdigest()==complete['replicas_sha256'],'Final student weight checksum mismatch')
                require(bool(saved['optimizer']['state']),'Missing final optimizer state')
                for value in saved['optimizer']['state'].values():
                    require(float(value['step'])==settings(config).updates,'Incomplete final optimizer state')
                    require(all(t.dtype==torch.float32 and bool(torch.isfinite(t).all())
                                for t in (value['exp_avg'],value['exp_avg_sq'])),'Invalid final optimizer moments')
                train_log=[json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
                validation=[json.loads(line) for line in (directory/'validation.jsonl').read_text().splitlines()]
                require(train_log==saved['history']['train'] and validation==saved['history']['validation'],
                        'Student logs and checkpoint history differ')
                require([r['update'] for r in train_log]==list(range(1,settings(config).updates+1))
                        and all(r['input_tokens']==r['update']*settings(config).tokens_per_update for r in train_log),
                        'Student budget mismatch')
                require(all(r['lambda_corr']==(1. if arm=='PCC' else 0.) and r['sigma_delta']==identity['sigma_delta']
                            for r in train_log),'Student objectives changed')
                curves.extend({'seed_index':seed,'arm':arm,**r} for r in validation)
                with np.load(directory/'eval.npz',allow_pickle=False) as values:
                    require(np.array_equal(values['target_counts'],counts)
                            and np.array_equal(values['sequence_indices'],np.arange(len(counts))), 'Student eval alignment changed')
                    losses[arm]=values['loss_sums'].copy()
                    require(abs(float(losses[arm].sum()/counts.sum())-complete['nll'])<1e-12,'Student NLL receipt mismatch')
            for key in ('settings','adapter_seed','data_order_seed','inputs','config','parent','audit_sha256',
                        'sigma_delta','code','world_size','physical_gpus','mixed_precision','torch'):
                require(identities[0][key]==identities[1][key],f'Unmatched student pair: {key}')
            original=args.joint_root/'runs'/f'seed-{seed}-Base/eval.npz'
            with np.load(original,allow_pickle=False) as values:
                require(np.array_equal(values['target_counts'],counts)
                        and np.array_equal(values['sequence_indices'],np.arange(len(counts))),'Original baseline eval changed')
                losses['OriginalBase']=values['loss_sums'].copy()
            stats=paired_bootstrap(losses,counts,resamples=2000,seed=20260922)
            contrasts={}
            for control in ('LM','FeedbackOff','OriginalBase','Deep'):
                contrasts[control]={'pcc_minus_control':stats['nll']['PCC']-stats['nll'][control],
                    'ci95':np.quantile(stats['samples']['PCC']-stats['samples'][control],[.025,.975]).tolist()}
            gain=stats['nll']['FeedbackOff']-stats['nll']['Deep']
            recovery=(stats['nll']['FeedbackOff']-stats['nll']['PCC'])/gain if gain>0 else None
            gates={'beats_matched_lm':contrasts['LM']['ci95'][1]<0,
                   'beats_feedback_off':contrasts['FeedbackOff']['ci95'][1]<0,
                   'beats_original_baseline':contrasts['OriginalBase']['ci95'][1]<0,
                   'recovers_quarter':recovery is not None and recovery>=.25}
            rows.append({'seed_index':seed,'nll':stats['nll'],'contrasts':contrasts,
                         'recovery_ratio':recovery,'gates':gates,'passed':all(gates.values())})
        with (args.output/'validation-curves.csv').open('x') as f:
            writer=csv.DictWriter(f,fieldnames=['seed_index','arm','update','nll','target_tokens'])
            writer.writeheader();writer.writerows(curves)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for seed,axis in enumerate(axes):
            for arm in ('LM','PCC'):
                values=[r for r in curves if r['seed_index']==seed and r['arm']==arm]
                axis.plot([r['update'] for r in values],[r['nll'] for r in values],label=arm)
            # Monitor is a smaller dev prefix; use only monitor curves in this plot.
            axis.set(title=f'Seed {seed+1}',xlabel='Student update',ylabel='Fixed monitor NLL')
            axis.legend()
        fig.tight_layout();fig.savefig(args.output/'validation-curves.png',dpi=160);plt.close(fig)
        passed=all(r['passed'] for r in rows)
        result={'status':'ok','stage':'distill-v1','seeds':rows,'exploratory':True,
                'decision':'recommend_target_semantics_control' if passed else 'stop_no_consistent_student_recovery',
                'test_unlocked':False,'automatic_followup_training':False}
        text=['# Correction-distillation results','',result['decision'],'',
              'Fixed-dev exploratory comparison; no new held-out test or automatic follow-up.','',
              '| Seed | Feedback off | Deep teacher | LM control | PCC | Recovery | Pass |',
              '|---|---:|---:|---:|---:|---:|---|']
        for row in rows:
            n=row['nll'];recovery=row['recovery_ratio']
            text.append(f"| {row['seed_index']+1} | {n['FeedbackOff']:.8f} | {n['Deep']:.8f} | {n['LM']:.8f} | "
                        f"{n['PCC']:.8f} | {recovery:.4f} | {row['passed']} |")
        (args.output/'RESULTS.md').write_text('\n'.join(text)+'\n')
        write_json(args.output/'complete.json',result)
        return result
    except BaseException as error:
        write_json(args.output/'failure.json',{'status':'failed','error':str(error)})
        raise
