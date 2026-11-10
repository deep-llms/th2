"""F10 GPU worker: fixed native models, no network and no optimization."""
import argparse
from pathlib import Path
from importlib.metadata import version
from eval.a2_family_f10 import write
from eval.a2_family_f9 import read
from eval.a2_family_f9_score import load, smol_score
from eval.a2_request_bridge import digest
from eval.a2_family_f10_tasks import answer_metrics, likelihood_metrics


def greedy(model, tokenizer, prompts, batch_size, max_new_tokens=32, use_cache=True):
    """Greedy newline/EOS stop; left padding, per-item positions, native cache.

    An unconfigured DynamicCache supports the original retained layer indices
    on pruned models. No attention layer identity or model config is rewritten.
    """
    import torch
    from transformers import DynamicCache
    from efficient_llm.a2.model import autocast
    results=[None]*len(prompts)
    order=sorted(range(len(prompts)),key=lambda i:-len(prompts[i]))
    with torch.inference_mode():
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size];seqs=[prompts[i][-4064:] for i in indices]
            if any(not s for s in seqs):raise ValueError('Empty generation prompt')
            width=max(map(len,seqs));n=len(seqs)
            ids=torch.full((n,width),tokenizer.eos_token_id,dtype=torch.long,device=model.device)
            mask=torch.zeros_like(ids)
            for i,s in enumerate(seqs):
                ids[i,-len(s):]=torch.tensor(s,device=model.device);mask[i,-len(s):]=1
            tokens=[[] for _ in seqs];finished=[False]*n;reasons=['max_tokens']*n
            cache=DynamicCache() if use_cache else None
            full=ids
            for step in range(max_new_tokens):
                positions=(mask.cumsum(-1)-1).clamp_min(0)
                current=full if step==0 or not use_cache else full[:,-1:]
                positions=positions if current.shape[1]!=1 else positions[:,-1:]
                with autocast(model.device):
                    out=model(input_ids=current,attention_mask=mask,position_ids=positions,
                              past_key_values=cache,use_cache=use_cache,logits_to_keep=1)
                logits=out.logits[:,-1,:]
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite generation logits')
                next_ids=logits.argmax(-1)
                if use_cache:cache=out.past_key_values
                for i,token in enumerate(next_ids.tolist()):
                    if finished[i]:continue
                    tokens[i].append(token)
                    decoded=tokenizer.decode(tokens[i],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    if token==tokenizer.eos_token_id:finished[i]=True;reasons[i]='eos'
                    elif '\n' in decoded:finished[i]=True;reasons[i]='newline'
                if all(finished):break
                full=torch.cat([full,next_ids[:,None]],dim=1)
                mask=torch.cat([mask,torch.ones((n,1),dtype=mask.dtype,device=mask.device)],dim=1)
            for i,index in enumerate(indices):
                text=tokenizer.decode(tokens[i],skip_special_tokens=True,clean_up_tokenization_spaces=False).split('\n',1)[0]
                results[index]=dict(text=text,token_ids=tokens[i],stop_reason=reasons[i])
            del cache,out
            if start%200==0:print('TRIVIAQA_GENERATED',min(start+batch_size,len(order)),'/',len(order),flush=True)
    return results


def main():
    p=argparse.ArgumentParser()
    for k in ('entry','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--prepared');p.add_argument('--tokenizer')
    p.add_argument('--mode',choices=['ft14','nine','new','generation'],required=True)
    p.add_argument('--batch-size',type=int,choices=[1,2],default=2)
    p.add_argument('--generation-batch-size',type=int,choices=[1,2],default=2)
    p.add_argument('--generation-reuse')
    a=p.parse_args()
    from transformers import set_seed,AutoTokenizer
    set_seed(42)
    entry=read(a.entry);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    model,info=load(entry,'cuda:0')
    keep=None
    if entry.get('oneshot'):
        if entry['family']=='smollm3':from efficient_llm.a2_family.model import prune
        else:from efficient_llm.a2.model import prune
        keep=prune(model,entry['removed_layers'])
    write(out/'load.json',dict(loading_info=info,model=entry,kept_layers=keep,
                              packages={k:version(k) for k in ('torch','transformers','tokenizers')}))
    if a.mode=='ft14':
        if not entry.get('oneshot'):raise ValueError('FT14 requires one-shot model')
        if entry['family']=='smollm3':
            from efficient_llm.a2_family.evaluate import score_docs
            result=score_docs(model,Path(entry['cx_root'])/'final')
        else:
            from efficient_llm.a2.measure import score
            from efficient_llm.a2.data import PackedRows
            result=score(model,PackedRows(entry['cx_root'],'final',verify=True),distributed=False)
        expected=entry['oneshot_expected_nll']
        write(out/'FT14.json',dict(result=result,expected=expected,difference=result['loss']-expected,
                                  passed=abs(result['loss']-expected)<=.002,model=entry))
    else:
        prepared=read(a.prepared)
        if prepared['request_sha256']!=digest(prepared['requests']):raise ValueError('Prepared requests changed')
        if a.mode!='generation':
            if entry['family']=='smollm3':scores=smol_score(model,prepared['requests'],a.batch_size)
            else:
                from eval.a2_likelihood import score_requests
                scores=score_requests(model,prepared['requests'],batch_size=a.batch_size,max_length=4096)
            scores.update(batch_size=a.batch_size,prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
                          checkpoint_identity=entry,checkpoint_identity_sha256=digest(entry))
            write(out/'scores.json',scores)
            if a.mode=='new':
                values={task:likelihood_metrics(rows,scores['scores'],task) for task,rows in prepared['tasks'].items()}
                write(out/'likelihood_items.json',dict(tasks=values,task_identity=prepared['task_identity']))
        if a.mode in ('new','generation') and not entry.get('oneshot'):
            if a.generation_reuse:
                old=read(a.generation_reuse)
                if (old['prepared_sha256']!=digest(prepared) or old['checkpoint_identity_sha256']!=digest(entry)
                    or old['batch_size']!=a.generation_batch_size):raise ValueError('Generation reuse mismatch')
                write(out/'generation_items.json',dict(old,reused_from=a.generation_reuse))
            else:
                tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
                rows=prepared['tasks']['triviaqa']
                generations=greedy(model,tok,[r['prompt_tokens'] for r in rows],a.generation_batch_size)
                records=[dict(g,doc_id=r['doc_id'],**answer_metrics(g['text'],r['aliases'])) for r,g in zip(rows,generations,strict=True)]
                write(out/'generation_items.json',dict(items=records,batch_size=a.generation_batch_size,
                      task_identity=prepared['task_identity']['triviaqa'],prepared_sha256=digest(prepared),
                      checkpoint_identity_sha256=digest(entry),
                      exact_match=sum(r['exact_match'] for r in records)/len(records),token_f1=sum(r['token_f1'] for r in records)/len(records)))
    write(out/'complete.json',dict(execution_complete=True,mode=a.mode,model=entry['model_id']))


if __name__=='__main__':main()
