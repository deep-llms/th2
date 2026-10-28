"""Offline CPU request preparation/replay; native GPU scoring is a separate stage.

Uses the unchanged canonical nine-task templates. This command loads no model,
never emits placeholder task scores, and refuses protocol/request mismatches.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline')

from eval.a2_request_bridge import RequestsCaptured, digest, make_bridge
from eval.kv_harness import file_hash, json_default


def validate_prepared(prepared, protocol):
    if (prepared.get('schema') != 1 or prepared.get('protocol') != protocol
            or prepared.get('request_sha256') != digest(prepared['requests'])
            or prepared.get('protocol_sha256') != digest(protocol)):
        raise ValueError('Prepared requests or fixed evaluation protocol changed')


def validate_scored(prepared, scored):
    if (scored.get('prepared_sha256') != digest(prepared)
            or scored.get('request_sha256') != prepared['request_sha256']
            or scored.get('max_length') != prepared['protocol']['max_length']
            or scored.get('precision') != 'fp32_parameters_bf16_cuda_autocast'
            or scored.get('likelihood_requests') != len(prepared['requests'])):
        raise ValueError('Native score artifact does not match prepared protocol')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare','replay'))
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--input-spec', default='resources/a2_inputs_20261028.json')
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--limit', type=int)
    p.add_argument('--prepared');p.add_argument('--scored')
    p.add_argument('--output', required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('Output exists')
    if not 2<=a.max_length<=4096 or a.limit is not None and a.limit<1:p.error('Invalid context/limit')
    if len(a.tasks)!=len(set(a.tasks)):p.error('Duplicate tasks')
    if (a.mode=='replay') != bool(a.prepared and a.scored):p.error('Replay requires prepared and scored files')
    if a.mode=='prepare' and (a.prepared or a.scored):p.error('Prepare does not consume scores')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:p.error('CPU bridge expects one process')
    spec=json.loads(Path(a.input_spec).read_text())
    token_hashes=spec['tokenizer_provenance']['gpt_neox']['files']
    for name,expected in token_hashes.items():
        if file_hash(Path(a.tokenizer)/name)!=expected:raise ValueError('Tokenizer checksum mismatch')
    from transformers import AutoTokenizer
    from lm_eval import evaluator
    from eval.benchmark_data import local_dataset_paths, task_configs, benchmark_task_manager
    from eval.downstream_suite import COUNTS
    if set(a.tasks)-COUNTS.keys():p.error('Task outside the existing nine-task screen')
    paths=local_dataset_paths(a.dataset_root,a.manifest)
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    protocol=dict(tasks=a.tasks,max_length=a.max_length,limit=a.limit,num_fewshot=0,seed=42,
        manifest_sha256=file_hash(a.manifest),tokenizer_files=token_hashes,
        packages={k:version(k) for k in ('lm-eval','transformers','tokenizers','datasets','torch')})
    prepared=scored=None
    if a.mode=='replay':
        prepared=json.loads(Path(a.prepared).read_text());scored=json.loads(Path(a.scored).read_text())
        validate_prepared(prepared,protocol);validate_scored(prepared,scored)
    bridge=make_bridge(tok,prepared=prepared,scored=scored)
    try:
        result=evaluator.simple_evaluate(model=bridge,tasks=task_configs(a.tasks,paths),
            num_fewshot=0,random_seed=42,numpy_random_seed=42,torch_random_seed=42,
            fewshot_random_seed=42,log_samples=True,apply_chat_template=False,limit=a.limit,
            bootstrap_iters=0,task_manager=benchmark_task_manager())
    except RequestsCaptured:
        if a.mode!='prepare' or bridge.captured is None:raise
        result={**bridge.captured,'protocol':protocol,'protocol_sha256':digest(protocol)}
    else:
        if a.mode!='replay' or bridge.calls!=1:raise ValueError('Expected request capture or complete replay')
        if set(result['results'])!=set(a.tasks):raise ValueError('Unexpected task result set')
        for task in a.tasks:
            expected=COUNTS[task] if a.limit is None else min(a.limit,COUNTS[task])
            if (len(result['samples'][task])!=expected
                    or len({row['doc_id'] for row in result['samples'][task]})!=expected
                    or result['n-shot'][task]!=0
                    or result['n-samples'][task]!=dict(original=COUNTS[task],effective=expected)):
                raise ValueError('Incomplete benchmark sample coverage')
        result['a2_metadata']=dict(protocol=protocol,prepared_sha256=digest(prepared),
            scored_file_sha256=file_hash(a.scored),request_sha256=prepared['request_sha256'])
    out.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive write: a failed or concurrent task cannot overwrite existing evidence.
    encoded=json.dumps(result,default=json_default,allow_nan=False)+'\n'
    with out.open('x') as stream:stream.write(encoded)


if __name__=='__main__':main()
