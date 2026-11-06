"""F0: evaluate 1.3B with the original Depth students' exact scoring module."""
import argparse
import importlib.util
import inspect
from pathlib import Path
import torch
from transformers import set_seed
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from efficient_llm.a2.model import assert_native
from efficient_llm.a2.data import PackedRows
from .common import read,write,sha,Checks

SCORE_SHA='db818b8b124d6a097072713294f140cb14cf4c3d7495b351dbcd8f5e2e701d26'
MODEL_SHA='f1a826eed8897a4df548c26d37741fb7b5d5cf3d170db2a7b079cd9f30da512d'
CONFIG_SHA='194d816263ec0279f5804f41041ecd29a645c330b5893636d38dc7cb94747885'


def run(cfg,dataset):
    from accelerate import PartialState
    state=PartialState();rank=state.process_index
    out=Path(cfg['results_root'])/'reference'/dataset
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    state.wait_for_everyone();checks=Checks(out/f'tests-rank{rank}.json')
    entry=next(x for x in cfg['models']['checkpoints'] if x['label']=='f0_'+dataset)
    directory=Path(entry['path']);module_path=Path(cfg['original_depth_source'])/'efficient_llm/a2/measure.py'
    # Wrong code is an admission failure, never silently replace the estimator.
    if sha(module_path)!=SCORE_SHA:raise ValueError('Original student scoring code unavailable/different')
    spec=importlib.util.spec_from_file_location('efficient_llm.a2._family_original_score',module_path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    cls=get_class_from_dynamic_module('modeling_opensci.OpensciForCausalLM',str(directory),local_files_only=True)
    checks.record('FT5_model_code',sha(inspect.getfile(cls))==MODEL_SHA)
    checks.record('FT5_config_code',sha(inspect.getfile(cls.config_class))==CONFIG_SHA)
    model_cfg=cls.config_class.from_dict(read(directory/'config.json'));model_cfg.use_cache=False;model_cfg.attention_dropout=0.
    set_seed(0)
    model,loaded=cls.from_pretrained(str(directory),config=model_cfg,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    checks.record('complete_model_load',not (set(loaded['missing_keys'])-{'lm_head.weight'} or loaded['unexpected_keys'] or loaded.get('mismatched_keys') or loaded.get('error_msgs')),loaded)
    model.to(state.device).eval();checks.attempt('FT5_native_architecture',lambda:assert_native(model,full=False))
    embedding=model.get_input_embeddings().weight
    nonembedding=sum(p.numel() for p in model.parameters() if p is not embedding)
    checks.record('FT5_eval_code_identity',sha(module_path)==SCORE_SHA,dict(path=str(module_path),sha256=SCORE_SHA))
    scores={name:checks.attempt('score:'+name,lambda name=name:original.score(model,PackedRows(cfg['opensci_cx_root'],name,verify=False))) for name in ('tuning','final')}
    flags=[None]*state.num_processes;torch.distributed.all_gather_object(flags,[x for x in checks.items if not x['passed']])
    if rank==0:
        write(out/'result.json',dict(dataset=dataset,model=entry['repo'],revision=entry['commit'],config=model_cfg.to_dict(),nonembedding_parameters=nonembedding,
                                   total_parameters=sum(p.numel() for p in model.parameters()),scores=scores,evaluation_code_sha256=SCORE_SHA,
                                   pretraining_tokens=300_000_000_000,flags=[f for rows in flags for f in rows]))
        for name,value in scores.items():
            if value is not None:write(out/'perseq'/(name+'.json'),value)
    state.wait_for_everyone()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dataset',choices=['dclm','fwe'],required=True);a=p.parse_args();run(read(a.config),a.dataset)
