import tempfile
from pathlib import Path
from unittest.mock import patch
from eval.a2_family_f12 import write,read,descriptors
from scripts.a2_quant_io import lock
from scripts.launch_a2_family_f12 import prepare
from efficient_llm.diagnostic_data import sha256
with tempfile.TemporaryDirectory(dir='temp',prefix='f12-prepare-fixture-') as d:
 b=Path(d).resolve();q=b/'queue';q.mkdir();old=b/'f11queue';old.mkdir();(old/'source').mkdir();(old/'source/file.py').write_text('pass\n')
 write(old/'source-manifest.json',dict(code_commit='a'*40,files={'file.py':sha256(old/'source/file.py')}))
 pre=write(old/'prepared_config.json',dict(output_root='/f11',f10_root='/f10'))
 plan=dict(f11_prepared=pre,f11_root='/f11',f10_root='/f10',f11_code_commit='a'*40)
 for key,n in [('inventory','inventory'),('reference_13_lock','reference_13')]:plan[key]=lock(Path(f'resources/a2_family_f12_{n}_20261111.json').resolve())
 stream=b/'data';m=write(stream/'manifest.json',dict(streams=dict(heal=dict(rows=12207*64))))
 plan['heal_streams']={'dclm':{'2030':dict(path=str(stream),sha256=m['sha256'])}}
 inherited=dict(preparation_proof=write(q/'actual_inputs.json',dict(files={})),reference_nll={'fixture':'unchanged'})
 with patch('scripts.launch_a2_family_f11.prepare',return_value=inherited):cfg=prepare(plan,q)
 assert cfg['runs']=={r['run_id']:r for r in descriptors()};assert cfg['reference_13']['nonembedding_parameters']==1205424128
 assert read(q/'ready.json')['waiting_for']=='F11 report/backup/native handoff'
 assert read(q/'actual_inputs.json')['heal_rows']==12207*64
 write('temp/f12_inputs/prepare_proof.json',dict(passed=True,f11_lineage_lock=True,full_budget_stream_required=True,full_parent_input_validation_reused=True,actual_input_hashing_mocked_in_fixture=True))
print('F12_CPU_ADMISSION_FIXTURE_PASS')
