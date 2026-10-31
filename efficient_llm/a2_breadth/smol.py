"""One document per sequence, native SmolLM3, immutable original RoPE identity."""
import argparse
from contextlib import contextmanager
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import read,write,sha,Checks,blocks
from efficient_llm.a2.model import autocast
from efficient_llm.a2.measure import full_fp32

@contextmanager
def removed_layers(model,removed):
    layers=model.model.layers;n=model.config.num_hidden_layers
    keep=[i for i in range(len(layers)) if i not in removed]
    model.model.layers=torch.nn.ModuleList([layers[i] for i in keep]);model.config.num_hidden_layers=len(keep)
    try:yield keep
    finally:model.model.layers=layers;model.config.num_hidden_layers=n

def encode_document(tokenizer,text):
    # A2 convention: no automatic BOS/template; explicit document EOS before truncation.
    return (tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']+[tokenizer.eos_token_id])[:4096]

def prepare(cfg):
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    root=Path(cfg['results_root']);checks=Checks(root/'document_checks.json')
    pool=Path(cfg['source_documents']['root']);files=cfg['source_documents']['files']
    for rel,expected in files.items():
        checks.record('ST3_source:'+rel,sha(pool/rel)==expected['sha256'],dict(bytes=(pool/rel).stat().st_size))
    ds=load_from_disk(str(pool/cfg['source_documents']['shard']))
    tok=AutoTokenizer.from_pretrained(cfg['downloads']['checkpoints'][0]['path'],local_files_only=True)
    all_ids=[];offsets=[0];docs=[]
    for i,row in enumerate(ds):
        ids=encode_document(tok,row['text']);all_ids.extend(ids);offsets.append(len(all_ids))
        docs.append(dict(index=i,text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),tokens=len(ids),predicted_tokens=max(len(ids)-1,0)))
    target=root/'documents';target.mkdir(exist_ok=True)
    np.save(target/'tokens.npy',np.asarray(all_ids,dtype=np.int32));np.save(target/'offsets.npy',np.asarray(offsets,dtype=np.int64))
    write(target/'documents.json',dict(documents=docs))
    checks.record('ST3_no_cross_document_attention',True,'Separate forward per document, no packing or shared KV cache')
    checks.record('ST3_nonempty_predictions',all(d['predicted_tokens']>0 for d in docs),dict(empty=[d['index'] for d in docs if not d['predicted_tokens']]))
    m=dict(documents=len(ds),tokens=len(all_ids),predicted_tokens=sum(d['predicted_tokens'] for d in docs),
        tokenizer_files={f['name']:f['sha256'] for f in cfg['downloads']['checkpoints'][0]['files'] if 'token' in f['name']},
        token_sha256=sha(target/'tokens.npy'),offset_sha256=sha(target/'offsets.npy'),
        convention='add_special_tokens=False + one EOS, then truncate to4096; one document/sequence; preserve validation order')
    write(root/'manifests/documents.json',m)

@torch.no_grad()
def score_docs(model,tokens,offsets,progress=None):
    model.eval();device=next(model.parameters()).device;sums=[];counts=[]
    for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        ids=torch.tensor(np.array(tokens[start:end],dtype=np.int64),device=device)[None]
        count=max(ids.shape[1]-1,0);total=torch.zeros((),dtype=torch.float64,device=device)
        if count:
            with autocast(device):
                h=model.model(ids,use_cache=False,return_dict=True).last_hidden_state
                for off in range(0,count,128):
                    hi=min(off+128,count);logits=model.lm_head(h[:,off:hi]).float()
                    loss=F.cross_entropy(logits.flatten(0,1),ids[:,off+1:hi+1].flatten(),reduction='none')
                    total+=loss.double().sum()
        sums.append(float(total));counts.append(count)
        if progress and (i+1)%500==0:write(progress,dict(documents=i+1,total=len(offsets)-1))
    nll=[v/c if c else 0. for v,c in zip(sums,counts)]
    return dict(loss=sum(sums)/sum(counts),per_sequence_nll=nll,per_sequence_tokens=counts,
        tokens=sum(counts),sequences=len(counts),precision='FP32 parameters, BF16 autocast decoder/head, FP32 CE, FP64 sum',
        zero_prediction_documents=[i for i,c in enumerate(counts) if not c])

def run(cfg,index):
    from transformers import AutoModelForCausalLM
    root=Path(cfg['results_root']);entry=cfg['downloads']['checkpoints'][index];name=entry['revision']
    out=root/'smol'/name;out.mkdir(parents=True,exist_ok=False);checks=Checks(out/'tests.json');started=time.monotonic()
    for f in entry['files']:checks.record('asset:'+f['name'],sha(Path(entry['path'])/f['name'])==f['sha256'])
    model,loading=AutoModelForCausalLM.from_pretrained(entry['path'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa',use_cache=False,output_loading_info=True)
    checks.record('ST1_complete_load',not any(loading.values()),loading)
    model=model.to('cuda').eval()
    n=len(model.model.layers);flags=[dict(original_layer=i,use_rope=bool(l.self_attn.use_rope),layer_idx=l.self_attn.layer_idx) for i,l in enumerate(model.model.layers)]
    token_identity=lambda e:{f['name']:f['sha256'] for f in e['files'] if 'token' in f['name']}
    checks.record('ST1_shared_tokenizer',token_identity(entry)==token_identity(cfg['downloads']['checkpoints'][0]),token_identity(entry))
    checks.record('ST1_configuration',True,dict(config=model.config.to_dict(),layers=flags))
    tokens=np.load(root/'documents/tokens.npy',mmap_mode='r');offsets=np.load(root/'documents/offsets.npy')
    ids=torch.tensor(np.array(tokens[:min(32,int(offsets[1]))],dtype=np.int64),device='cuda')[None]
    with torch.no_grad(),full_fp32():
        original=model(ids,use_cache=False).logits
        with removed_layers(model,[]):actual=model(ids,use_cache=False).logits
    checks.record('ST2_no_removal_bitwise_fp32',torch.equal(original,actual),dict(max_difference=float((original-actual).abs().max())))
    for block,removed in dict(parent=[],**blocks(n)).items():
        try:
            with removed_layers(model,removed) as keep:
                checks.record('ST2_original_RoPE_NoPE:'+block,all(l.self_attn.layer_idx==i and bool(l.self_attn.use_rope)==flags[i]['use_rope'] for i,l in zip(keep,model.model.layers)),dict(kept=keep,removed=removed))
                result=score_docs(model,tokens,offsets,out/'progress.json')
            checks.record('finite:'+block,np.isfinite(result['per_sequence_nll']).all())
            write(root/'perseq'/name/(block+'.json'),dict(result,checkpoint=entry['commit'],revision=name,block=block,removed=removed))
        except Exception as e:checks.record('evaluation:'+block,False,dict(error=repr(e),action='continue next registered block'))
    write(out/'complete.json',dict(execution_complete=True,elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--index',type=int);p.add_argument('--prepare',action='store_true');a=p.parse_args();cfg=read(a.config)
    if a.prepare:prepare(cfg)
    else:run(cfg,a.index)
