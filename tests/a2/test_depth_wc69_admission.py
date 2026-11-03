import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts import a2_depth_wc69_admission as gate


class WC69AdmissionTests(unittest.TestCase):
    def fixture(self):
        host='thiennh-p6-wc69-worker-0'
        snapshot=dict(host=host,gpus=[dict(index=i,uuid='gpu'+str(i),name='NVIDIA B200',pids=[503+i]) for i in range(8)])
        identities={str(i):dict(pid=i,ppid=436,start_ticks=i,command_sha256=str(i)) for i in range(503,511)}
        identities['436']=dict(pid=436,ppid=1,start_ticks=123,command_sha256='parent')
        cfg=dict(predecessor={'historical':'B'},node_restart=dict(host=host,inspection=copy.deepcopy(snapshot),known_burn_identities=copy.deepcopy(identities)))
        return cfg,snapshot,identities

    def mocks(self,snapshot,identities):
        stack=ExitStack()
        stack.enter_context(patch.object(gate.socket,'gethostname',return_value=snapshot['host']))
        stack.enter_context(patch.object(gate,'inspect',return_value=snapshot))
        stack.enter_context(patch.object(gate,'process',side_effect=lambda pid:identities[str(pid)]))
        stack.enter_context(patch.object(gate,'argv',return_value=['python','/tmp/llm_pretrain_burn.py']))
        stack.enter_context(patch.object(gate,'digest',side_effect=lambda p:gate.GUARD_HASH if p.endswith('gpu_guard.sh') else gate.APPROVED_BURNS[p]))
        stack.enter_context(patch.object(gate.Path,'exists',return_value=False))
        return stack

    def test_known_operator_burn_accepted_without_signals(self):
        cfg,snapshot,identities=self.fixture()
        with self.mocks(snapshot,identities):
            result=gate.admission(cfg,Path('/mnt/local/_outputs/deep-llms_th2/new-run'))
        self.assertEqual(result['identities'],identities)
        self.assertEqual(result['historical_predecessor'],cfg['predecessor'])

    def test_changed_worker_start_topology_guard_or_node_refused(self):
        for change in ('start','pid','uuid','guard','host','launcher'):
            with self.subTest(change=change):
                cfg,snapshot,identities=self.fixture()
                if change=='start':identities['503']['start_ticks']+=1
                if change=='pid':snapshot['gpus'][0]['pids']=[999]
                if change=='uuid':snapshot['gpus'][0]['uuid']='other'
                if change=='host':snapshot['host']='other-node'
                with self.mocks(snapshot,identities),ExitStack() as extra:
                    if change=='guard':extra.enter_context(patch.object(gate.Path,'exists',return_value=True))
                    if change=='launcher':extra.enter_context(patch.object(gate,'argv',return_value=['unrelated-training']))
                    with self.assertRaises(ValueError):gate.admission(cfg,Path('/mnt/local/_outputs/deep-llms_th2/new-run'))

    def test_new_launcher_rejects_data_or_proof_before_output_or_process_actions(self):
        from scripts import launch_a2_depth_partc as launcher
        from efficient_llm.a2_depth.common import write
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';plan=root/'plan.json'
            cfg=dict(node_restart={'host':'thiennh-p6-wc69-worker-0'},arm_order=['X72','P300','P1T'],arm_order_authorization='user',
                     heal_streams={'dclm':{'path':'data','sha256':'expected'}},stream_history={},stream_reconstruction={},
                     cpu_validation={'sha256':'proof-sha'},cpu_validation_resource='proof.json')
            write(plan,cfg)
            for manifest_hash,proof in [('wrong',{}),('expected',{'status':'FAIL','files':{}})]:
                with patch('sys.argv',['launcher','--plan',str(plan),'--output',str(out),'--session','fixture','--code-commit','fixture']),patch.object(launcher.socket,'gethostname',return_value=cfg['node_restart']['host']),patch('efficient_llm.a2_depth.reconstruction.rebuilt_manifest',return_value=manifest_hash),patch.object(launcher,'read_locked',return_value=proof),patch.object(launcher,'digest',return_value='proof-sha'),patch.object(launcher.subprocess,'run') as process:
                    with self.assertRaises(ValueError):launcher.main()
                    process.assert_not_called();self.assertFalse(out.exists())

    def test_wc69_free_continuation_requires_same_hardware_and_uses_own_burn(self):
        from scripts import a2_depth_continue as continuation
        from efficient_llm.a2_depth.common import write
        cfg,snapshot,_=self.fixture()
        for gpu in snapshot['gpus']:gpu['pids']=[]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);initial=write(root/'inspection.json',snapshot)
            with patch.object(continuation.socket,'gethostname',return_value=snapshot['host']),patch.object(continuation,'inspect',return_value=snapshot),patch.object(continuation.Path,'exists',return_value=False),patch.object(continuation,'digest',return_value=continuation.GUARD_HASH):
                result=continuation.admission(root,{'initial_inspection':initial},cfg,{'mode':'free'})
                self.assertEqual(result['predecessor']['burn_script'],str(root/'source/resources/llm_pretrain_burn.py'))
                snapshot['gpus'][0]['uuid']='replacement-gpu'
                with self.assertRaises(ValueError):continuation.admission(root,{'initial_inspection':initial},cfg,{'mode':'free'})

    def test_new_launcher_uses_native_operator_burn_and_freezes_cpu_proof(self):
        from scripts import launch_a2_depth_partc as launcher
        from efficient_llm.a2_depth.common import write, read
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';out=Path(d)/'out'
            (source/'resources').mkdir(parents=True)
            (source/'resources/accelerate_b200.yaml').write_text('fixture')
            proof=write(source/'resources/proof.json',dict(status='PASS',files={'resources/accelerate_b200.yaml':launcher.digest(source/'resources/accelerate_b200.yaml')}))
            cfg=dict(node_restart={'host':'thiennh-p6-wc69-worker-0'},arm_order=['X72','P300','P1T'],arm_order_authorization='user',
                heal_streams={'dclm':{'path':'data','sha256':'expected'}},stream_history={},stream_reconstruction={},
                cpu_validation=proof,cpu_validation_resource='resources/proof.json',predecessor={'burn_script':'historical-only'})
            plan=source/'resources/plan.json';write(plan,cfg);calls=[]
            def run(args,**kwargs):
                calls.append(args)
                return SimpleNamespace(returncode=1 if args[:2]==['tmux','has-session'] else 0,stdout='',stderr='')
            def freeze(src,output,commit,**kwargs):
                self.assertIn('resources/proof.json',kwargs['extra_files'])
                write(output/'source-manifest.json',dict(files={},code_commit=commit))
                return source
            with patch('sys.argv',['launcher','--plan',str(plan),'--output',str(out),'--session','fixture','--code-commit','fixture']),patch.object(launcher,'__file__',str(source/'scripts/launcher.py')),patch.object(launcher.socket,'gethostname',return_value=cfg['node_restart']['host']),patch('efficient_llm.a2_depth.reconstruction.rebuilt_manifest',return_value='expected'),patch.object(gate,'admission',return_value={'inspection':{}}) as admit,patch.object(launcher,'predecessor') as old,patch.object(launcher,'freeze_source',side_effect=freeze),patch.object(launcher,'prepare',side_effect=lambda cfg,root,*args:write(root/'config/plan.json',cfg)),patch.object(launcher.subprocess,'run',side_effect=run),patch('accelerate.commands.config.config_args.default_config_file',str(Path(d)/'cache.yaml')),patch('accelerate.commands.config.config_args.load_config_from_file',return_value=SimpleNamespace(num_processes=8,mixed_precision='bf16',use_cpu=False)):
                launcher.main()
            old.assert_not_called();self.assertEqual(admit.call_count,2)
            command=next(args[-1] for args in calls if args[:2]==['tmux','new-session'])
            self.assertIn('scripts.train_then_burn',command)
            self.assertNotIn('--predecessor-burn-script',command)
            record=read(out/'launch.json');self.assertEqual(record['input_plan'],lock(plan))
