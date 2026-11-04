"""Offline CPU preparation; restore FWE, reconstruct historical pool, create Y1 streams."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
from importlib.metadata import version
from scripts.a2_depth_wc69_restore import restore_files
from scripts.a2_rebuild_wc69_data import compare_content
from scripts.a2_stage1_data import prepare as shuffle
from scripts.a2_why_streams import prepare as new_streams
from scripts.a2_prepare_corpora import sha
from efficient_llm.a2_why.common import read,write,Checks,SPEC_SHA

OUT=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-data-20261104-a01')
DATA=Path('/mnt/local/_data/deep-llms_th2/a2_why_fwe_rebuild_20261104')
SHUFFLE=Path('/mnt/local/_data/deep-llms_th2/a2_why_fwe_shuffle2030_20261104')
CX='/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/data'


def work():
    if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
    checks=Checks(OUT/'tests.json')
    for package,pin in [('numpy','2.4.6'),('tokenizers','0.22.1'),('pyarrow','26.0.0')]:checks.record('runtime:'+package,version(package)==pin,version(package))
    asset=read('resources/a2_inputs_20261028.json')['assets'];depth=read('resources/a2_depth_wc69_plan_20261103_a01.json')
    records={}
    for key in ('fwe/P300','fwe/P1T'):
        item=asset[key]
        for directory,hashes in ((item['weights_dir'],item['files']),(item['code_dir'],item['code_files'])):
            for name,digest in hashes.items():
                dst=Path(directory)/name;src=Path('/mnt/local/_backup/deep-llms_th2')/dst.relative_to('/mnt/local/_models/deep-llms_th2')
                records[str(dst)]=dict(source=str(src),destination=str(dst),sha256=digest)
        for label in ('parent','pre-S1'):
            item=depth['reference_nll'][key+'/'+label];dst=Path(item['path'])
            src=Path('/mnt/local/_backup/deep-llms_th2')/dst.relative_to('/mnt/local/_outputs/deep-llms_th2')
            records[str(dst)]=dict(source=str(src),destination=str(dst),sha256=item['sha256'])
    files=restore_files(list(records.values()));write(OUT/'restore.json',dict(files=files,status='verified'))
    # DCLM already matches historical full hashes. New permutations need no retokenization.
    streams={'dclm':new_streams('/mnt/local/_outputs/deep-llms_th2/wc69-a2-data-rebuild-20261103-a02/plan.json',
        'dclm','/mnt/local/_data/deep-llms_th2/a2_dclm_shuffle2030_rebuild_20261103',OUT/'dclm',CX)}
    write(OUT/'dclm_ready.json',dict(streams=streams['dclm'],status='complete'))
    plan=read('resources/a2_corpora_30b_20261029.json');plan['root']=str(DATA)
    plan['corpora']={'fineweb_edu':plan['corpora']['fineweb_edu']};plan['corpora']['fineweb_edu']['files']=plan['corpora']['fineweb_edu']['files'][:44]
    plan_path=OUT/'corpus_plan.json';write(plan_path,plan)
    subprocess.run([sys.executable,'-u','-m','scripts.a2_prepare_corpora','--plan',str(plan_path),'--deadline-hours','inf'],check=True)
    packed=DATA/'packed/fineweb_edu';actual=read(packed/'manifest.json');expected=read('docs/results/a2_corpora_20261029/fwe_final/fwe_manifest.json')
    compare_content(actual,expected,['streams','tokenizer_sha256','eos_id','small_indices','partition','packing'])
    prefix=lambda m:[(x['group'],x['source_index'],x['tokens']) for x in m['source_prefixes']]
    checks.record('historical_FWE_prefix',prefix(actual)==prefix(expected))
    shuffle(plan_path,sha(packed/'manifest.json'),SHUFFLE,'docs/A2_stage1_spec.md',corpus='fineweb_edu',replication_spec='docs/A2_stage1b_fwe_spec.md')
    shuffled=read(SHUFFLE/'manifest.json');old=read('docs/results/a2_stage1b_fwe_20261030/data_final/manifest.json')
    compare_content(shuffled,old,['streams','tokenizer_sha256','eos_id','document_count','documents_sha256','document_order_sha256','heldout_documents_sha256','excluded_original_tail_tokens','dropped_after_shuffle'])
    write(OUT/'historical_fwe_identity.json',dict(status='verified',old_manifest_sha256=sha('docs/results/a2_stage1b_fwe_20261030/data_final/manifest.json'),new_manifest_sha256=sha(SHUFFLE/'manifest.json'),content_hashes_equal=True))
    streams['fwe']=new_streams(plan_path,'fineweb_edu',SHUFFLE,OUT/'fwe',CX)
    streams['fwe']['2030']=dict(path=str(SHUFFLE),sha256=sha(SHUFFLE/'manifest.json'))
    streams['dclm']['2030']=depth['heal_streams']['dclm']
    write(OUT/'complete.json',dict(status='complete',streams=streams,checks=checks.items))
    print('WHY_DATA_COMPLETE',flush=True)

if __name__=='__main__':
    try:work()
    except Exception as e:
        write(OUT/'failure.json',dict(error=repr(e),action='Preserve bytes; continue independent registered work'))
        raise
