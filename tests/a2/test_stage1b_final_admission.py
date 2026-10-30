import hashlib
import tempfile
import unittest
from pathlib import Path
from eval.a2_run import write,lock
from eval.a2_stage1b_final_plan import admit_final_regression,ADDITIONS
from scripts.launch_a2_stage1b_final_downstream import budget
from eval.a2_stage1b_batch_compare import ll_differences
from eval.a2_request_bridge import digest

ROOT=Path(__file__).resolve().parents[2]
POLICY=lock(ROOT/'resources/a2_stage1b_continuation_20261030.json')
TIME=lock(ROOT/'resources/a2_stage1b_time_policy_20261030.json')


class FinalAdmission(unittest.TestCase):
    def test_report_only_ledger_can_exceed_previous_cap(self):
        r=budget(100000,600,10800,TIME)
        self.assertTrue(r['former_24h_exceeded']);self.assertTrue(r['report_due'])
        self.assertIsNone(r['timeout_seconds']);self.assertIsNone(r['hard_deadline'])
        with self.assertRaises(ValueError):budget(0,0,7200,dict(TIME,sha256='0'*64))
        with self.assertRaises(ValueError):budget(float('nan'),0,7200,TIME)

    def test_only_registered_adapters_may_differ_from_actual_passed_regression(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';candidate=root/'candidate'
            for p in (src/'eval',src/'efficient_llm',candidate/'eval',candidate/'efficient_llm'):p.mkdir(parents=True)
            (src/'eval/core.py').write_text('core=1\n');(candidate/'eval/core.py').write_bytes((src/'eval/core.py').read_bytes())
            for name in ADDITIONS:(candidate/name).write_text('# registered new adapter\n')
            sm=write(root/'source-manifest.json',dict(files={'eval/core.py':hashlib.sha256((src/'eval/core.py').read_bytes()).hexdigest()}))
            base=root/'run';base.mkdir()
            report=write(base/'regression.json',dict(status='compared',passed=True))
            complete=write(base/'complete.json',dict(status='ok',models=1,additional_results=dict(regression=report)))
            terminal=write(base/'terminal.json',dict(status='ok',jobs=[dict(name='stage1b-regression',returncode=0,status='ok',
                artifacts=[dict(path=n,sha256=lock(base/n)['sha256']) for n in ('regression.json','complete.json')])]))
            bundle=write(root/'bundle.json',dict(source_manifest=sm,complete=complete,run_complete=terminal,
                supervisor=write(root/'supervisor.json',dict(training_status='ok',training_returncode=0)),
                burn=write(root/'burn.json',dict(collective_progress_verified=True)),training_predecessor=write(root/'prior.json',{})))
            self.assertTrue(admit_final_regression(bundle,candidate,POLICY)['unchanged'])
            (candidate/'eval/core.py').write_text('core=2\n')
            with self.assertRaises(ValueError):admit_final_regression(bundle,candidate,POLICY)
            (candidate/'eval/core.py').write_bytes((src/'eval/core.py').read_bytes())
            (candidate/'eval/unregistered.py').write_text('')
            with self.assertRaises(ValueError):admit_final_regression(bundle,candidate,POLICY)
            (candidate/'eval/unregistered.py').unlink()
            (src/'eval/core.py').write_text('original tampered\n')
            with self.assertRaises(ValueError):admit_final_regression(bundle,candidate,POLICY)

    def test_full_request_ll_statistics_and_padding_association(self):
        n=78259
        requests=[dict(context=[1]*(2 if i%2==0 else 1),continuation=[2]) for i in range(n)]
        prepared=dict(requests=requests,protocol=dict(max_length=4096))
        a=dict(batch_size=1,prepared_sha256=digest(prepared),request_sha256=digest(requests),
            checkpoint_identity_sha256='identity',likelihood_requests=n,precision='p',max_length=4096,packages={},
            scores=[[-1.,False] for _ in range(n)])
        b=dict(a,batch_size=2,scores=[list(x) for x in a['scores']]);b['scores'][0][0]-=.25
        r=ll_differences(prepared,a,b)
        self.assertEqual(r['summary']['requests'],n);self.assertEqual(r['summary']['max_abs'],.25)
        self.assertEqual(r['summary']['mean_abs'],.25/n)
        self.assertFalse(r['differences_only_in_padded_pairs']);self.assertEqual(r['largest']['index'],0)
        with self.assertRaises(ValueError):ll_differences(prepared,a,dict(b,request_sha256='changed'))
        with self.assertRaises(ValueError):ll_differences(prepared,a,dict(b,scores=b['scores'][:-1]))


if __name__=='__main__':unittest.main()

class CombinedPlan(unittest.TestCase):
    def test_final_plan_requires_both_terminal_parts_and_immutable_prior_fits(self):
        from unittest.mock import patch
        from eval.a2_stage1b_final_plan import validate_plan
        from eval.a2_stage1b_stats import PART_MODELS
        from eval.downstream_suite import COUNTS
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=write(root/'manifest.json',{})
            protocol=dict(tasks=list(COUNTS),max_length=4096,limit=None,seed=42,num_fewshot=0,
                manifest_sha256=manifest['sha256'],tokenizer_files={})
            requests=[dict(context=[1],continuation=[2])]*78259
            prepared=write(root/'prepared.json',dict(protocol=protocol,requests=requests,
                request_sha256=digest(requests),protocol_sha256=digest(protocol)))
            bundles={};ids={};terminal={}
            assets=write(root/'assets.json',{})
            for phase in ('part1','part2'):
                locks={name:write(root/(name+'.json'),dict(name=name)) for name in PART_MODELS[phase]}
                ids.update(locks);stage=write(root/(phase+'-stage.json'),dict(phase=phase))
                terminal[stage['path']]=({},'same-reference',{'old-fit':{'final':3.0},**({'new-fit':{}} if phase=='part2' else {})},{})
                bundles[phase]=write(root/(phase+'-bundle.json'),dict(identities=locks,source_locks=dict(stage=stage,assets=assets)))
            plan=dict(schema=3,phase='final',identities={n:ids[n] for n in PART_MODELS['final']},
                identity_bundles=bundles,batch_size=2,policy=POLICY,time_policy=TIME,
                regression=write(root/'regression.json',{}),prepared=prepared,manifest=manifest,
                input_spec=write(root/'input.json',dict(tokenizer_provenance=dict(gpt_neox=dict(files={})))),
                failed_gate=write(root/'failed.json',dict(passed=False,batch_max_nats_per_token=.011164188385009766)))
            def part_validate(sub):
                return {n:dict(name=n) for n in sub['identities']}
            with patch('eval.a2_stage1b_final_plan.PREPARED_SHA',prepared['sha256']), \
                 patch('eval.a2_stage1b_final_plan.validate_part_plan',side_effect=part_validate), \
                 patch('eval.a2_stage1b_final_plan.validate_terminal_bundle',side_effect=lambda item,a:terminal[item['path']]):
                self.assertEqual(set(validate_plan(plan)),set(PART_MODELS['final']))
                second=terminal[str(root/'part2-stage.json')][2];second['old-fit']['final']=4.0
                with self.assertRaises(ValueError):validate_plan(plan)
                second['old-fit']['final']=3.0
                plan['identities']['P300-parent']=ids['P1T-parent']
                with self.assertRaises(ValueError):validate_plan(plan)
