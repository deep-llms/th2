"""Single model F9 scorer; native OpenSci kernel unchanged, SmolLM3 own tokenizer."""
import argparse
import inspect
from pathlib import Path
from importlib.metadata import version
from eval.a2_family_f9 import read,write
from efficient_llm.diagnostic_data import sha256
from eval.a2_request_bridge import digest

def load(entry,device):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    for path,h in entry['file_hashes'].items():
        if sha256(path)!=h:raise ValueError('FT12 actual file changed: '+path)
    directory=entry['path'];cfg=read(Path(directory)/'config.json')
    if entry['family']=='smollm3':
        cls=AutoModelForCausalLM
        kwargs=dict(use_cache=False)
    else:
        cls=get_class_from_dynamic_module('modeling_opensci.OpensciForCausalLM',
                                         entry['code_dir'],local_files_only=True)
        if sha256(inspect.getfile(cls))!=entry['expected_code']['modeling_opensci.py'] or sha256(inspect.getfile(cls.config_class))!=entry['expected_code']['configuration_opensci.py']:
            raise ValueError('Imported model class differs from pinned code')
        config=cls.config_class.from_dict(cfg);config.use_cache=False;config.attention_dropout=0.
        kwargs=dict(config=config)
    model,info=cls.from_pretrained(directory,local_files_only=True,torch_dtype=torch.float32,
        attn_implementation='sdpa',output_loading_info=True,**kwargs)
    if set(info['missing_keys'])-{'lm_head.weight'} or info['unexpected_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Incomplete model load: '+repr(info))
    if info['missing_keys'] and model.lm_head.weight.data_ptr()!=model.get_input_embeddings().weight.data_ptr():
        raise ValueError('Missing untied output-head weights')
    keep=cfg.get('a2_family_kept_layers',cfg.get('a2_kept_layers'))
    if keep is not None:
        if len(keep)!=len(model.model.layers):raise ValueError('Saved layer map invalid')
        for layer,original in zip(model.model.layers,keep,strict=True):layer.self_attn.layer_idx=original
    if entry['family']!='smollm3':
        from efficient_llm.a2.model import assert_native
        assert_native(model)
    return model.to(device).eval(),info

def smol_score(model,requests,batch_size,max_length=4096):
    """Same sum-loglikelihood estimator/right padding as A2, native Smol forward."""
    import torch
    from efficient_llm.a2.model import autocast
    if model.config.model_type!='smollm3' or model.config._attn_implementation!='sdpa' or model.config.use_cache:
        raise ValueError('Expected native SmolLM3 SDPA without cache')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Require FP32 parameters')
    rows=[];truncated=0
    for i,r in enumerate(requests):
        context=r['context'];target=r['continuation']
        if not context or not 1<=len(target)<=max_length:raise ValueError('Invalid prepared request')
        if any(type(t)!=int or not 0<=t<model.config.vocab_size for t in context+target):raise ValueError('Token outside vocabulary')
        tokens=context+target;truncated+=len(tokens)>max_length+1
        rows.append((i,tokens[-(max_length+1):][:-1],target))
    rows.sort(key=lambda r:-len(r[1]));scores=[None]*len(rows)
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            ids=torch.zeros((len(batch),len(batch[0][1])),dtype=torch.long,device=model.device)
            for j,(_,inputs,_) in enumerate(batch):ids[j,:len(inputs)]=torch.tensor(inputs,device=model.device)
            with autocast(model.device):
                logits=model(input_ids=ids,use_cache=False,position_ids=torch.arange(ids.shape[1],device=ids.device).expand_as(ids)).logits
            for j,(index,inputs,target) in enumerate(batch):
                selected=logits[j,len(inputs)-len(target):len(inputs)].float()
                wanted=torch.tensor(target,device=ids.device)
                ll=selected.log_softmax(-1).gather(1,wanted[:,None]).sum()
                if not torch.isfinite(ll):raise FloatingPointError('Nonfinite likelihood')
                scores[index]=(float(ll),bool(torch.equal(selected.argmax(-1),wanted)))
            del logits
    return dict(scores=scores,likelihood_requests=len(rows),truncated_requests=truncated,
                precision='fp32_parameters_bf16_cuda_autocast',max_length=max_length)

def main():
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--prepared',required=True)
    p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,choices=[1,2],required=True)
    p.add_argument('--spot-check',action='store_true');p.add_argument('--load-only',action='store_true');a=p.parse_args()
    from transformers import set_seed
    set_seed(42)
    entry=read(a.entry);prepared=read(a.prepared);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    model,info=load(entry,'cuda:0')
    write(out/'load.json',dict(complete=True,loading_info=info,identity=entry))
    if a.spot_check:
        try:
            if entry['family']=='smollm3':
                from efficient_llm.a2_family.evaluate import score_docs
                result=score_docs(model,Path(entry['cx_root'])/'final')
            else:
                from efficient_llm.a2.measure import score
                from efficient_llm.a2.data import PackedRows
                result=score(model,PackedRows(entry['cx_root'],'final',verify=True),distributed=False)
            expected=entry.get('final_nll')
            write(out/'FT12_spot.json',dict(result=result,expected=expected,
                passed=expected is not None and abs(result['loss']-expected)<=.002))
        except Exception as e:write(out/'FT12_spot.json',dict(passed=False,error=repr(e)))
    if a.load_only:
        write(out/'complete.json',dict(execution_complete=True,load_verified=True));return
    if entry['family']=='smollm3':value=smol_score(model,prepared['requests'],a.batch_size)
    else:
        from eval.a2_likelihood import score_requests
        value=score_requests(model,prepared['requests'],batch_size=a.batch_size,max_length=4096)
    value.update(batch_size=a.batch_size,prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
        checkpoint_identity=entry,checkpoint_identity_sha256=digest(entry),
        packages={k:version(k) for k in ('torch','transformers','tokenizers')})
    score_lock=write(out/'scores.json',value)
    write(out/'complete.json',dict(execution_complete=True,scores=score_lock))

if __name__=='__main__':main()
