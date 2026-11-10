"""Offline canonical F10 task preparation and transparent per-item metrics."""
from collections import Counter
import re
import string
from pathlib import Path
import numpy as np
from eval.a2_family_f10 import SPEC_SHA, write
from eval.a2_family_f9 import read
from eval.a2_request_bridge import digest, make_bridge
from efficient_llm.diagnostic_data import sha256


def normalize_answer(text):
    # Standard TriviaQA normalization, also used by token F1.
    text = text.lower().replace('_', ' ')
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_metrics(prediction, aliases):
    if not aliases or any(not isinstance(a,str) for a in aliases):
        raise ValueError('Missing gold aliases')
    prediction = normalize_answer(prediction)
    exact, f1 = 0., 0.
    for alias in aliases:
        target = normalize_answer(alias)
        exact = max(exact, float(prediction == target))
        a, b = prediction.split(), target.split()
        common = sum((Counter(a) & Counter(b)).values())
        score = 2 * common / (len(a)+len(b)) if a and b else float(a == b)
        f1 = max(f1, score)
    return dict(exact_match=exact, token_f1=f1)


def likelihood_metrics(rows, scores, task):
    result=[]
    for row in rows:
        ll=np.asarray([scores[i][0] for i in row['requests']],dtype=float)
        lengths=np.asarray(row['character_lengths'],dtype=float)
        if not np.isfinite(ll).all() or len(ll)!=len(lengths) or (lengths<=0).any():
            raise ValueError('Invalid answer score or character length')
        gold=row['gold'];s=ll/lengths;softmax=np.exp(s-s.max());softmax/=softmax.sum()
        record=dict(doc_id=row['doc_id'],subject=row.get('subject'),conf=float(s[gold]),
                    raw_loglikelihoods=ll.tolist(),character_lengths=lengths.tolist(),
                    gold=gold,correctness=float(np.argmax(s)==gold) if task=='mmlu' else None,
                    acc=float(np.argmax(ll)==gold) if task=='mmlu' else None,
                    p_correct=float(softmax[gold]) if task=='mmlu' else None)
        result.append(record)
    return result


def prepare(manifest, tokenizer, output, code_commit):
    from importlib.metadata import version
    import lm_eval
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks._yaml_loader import load_yaml
    from transformers import AutoTokenizer
    if version('lm-eval')!='0.4.13' or not re.fullmatch('[0-9a-f]{40}',code_commit):
        raise ValueError('Require pinned harness and actual code commit')
    cfg=read(manifest)
    if cfg['spec_sha256']!=SPEC_SHA or Path(output).exists():
        raise ValueError('Wrong spec or existing preparation')
    paths={};hashes={}
    for repo in cfg['repositories']:
        found={}
        for item in repo['files']:
            p=Path(item['local_dir'])/Path(item['path']).name
            if not p.is_file() or p.stat().st_size!=item['bytes'] or sha256(p)!=item['sha256']:
                raise ValueError('Missing or changed input: '+str(p))
            found[item['path']]=str(p);hashes[str(p)]=item['sha256']
        paths[repo['tag']]=found
    tokens=read('resources/a2_inputs_20261028.json')['tokenizer_provenance']['gpt_neox']['files']
    for name,h in tokens.items():
        if sha256(Path(tokenizer)/name)!=h:raise ValueError('Tokenizer differs')
    tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True)
    bridge=make_bridge(tok)
    requests=[];rows={'mmlu':[],'triviaqa':[]};sources={};fewshots=[]
    def request(context, target, generation_context=False):
        a,b=bridge._encode_pair(context,target)
        if generation_context:
            # Both confidence and generation share this exact prompt prefix.
            a=a[-4064:]
        index=len(requests)
        requests.append(dict(text=[context,target],context=a or [tok.eos_token_id],continuation=b))
        return index
    taskroot=Path(lm_eval.__file__).parent/'tasks'
    subjects=sorted(Path(p).parts[0] for p in paths['mmlu'])
    if len(subjects)!=57 or len(set(subjects))!=57:raise ValueError('Expected 57 subjects')
    for subject in subjects:
        yaml=taskroot/'mmlu/continuation'/('mmlu_'+subject+'.yaml')
        template=load_yaml(yaml,resolve_func=True);sources[str(yaml)]=sha256(yaml)
        data=next(p for name,p in paths['mmlu'].items() if name.startswith(subject+'/'))
        config={**template,'dataset_path':'parquet','dataset_name':None,
                'dataset_kwargs':{'data_files':{'test':[data]}},'num_fewshot':0,'fewshot_split':None}
        task=ConfigurableTask(config=config);task.set_fewshot_seed(42);task.build_all_requests()
        instances=task.instances
        for offset in range(0,len(instances),4):
            group=instances[offset:offset+4];doc=group[0].doc
            if len(group)!=4 or len({i.doc_id for i in group})!=1:raise ValueError('MMLU request grouping')
            choices=task.doc_to_choice(doc);gold=int(task.doc_to_target(doc))
            indices=[request(*i.arguments) for i in group]
            rows['mmlu'].append(dict(doc_id=subject+':'+str(group[0].doc_id),subject=subject,
                                    choices=choices,gold=gold,character_lengths=[len(c) for c in choices],requests=indices))
    yaml=taskroot/'triviaqa/default.yaml';sources[str(yaml)]=sha256(yaml)
    source=taskroot/'mmlu/continuation/_continuation_template_yaml';sources[str(source)]=sha256(source)
    template=load_yaml(yaml,resolve_func=True)
    config={**template,'dataset_path':'parquet','dataset_name':None,
            'dataset_kwargs':{'data_files':{split:[paths['triviaqa']['rc.nocontext/'+split+'-00000-of-00001.parquet']] for split in ('train','validation')}},
            'num_fewshot':5,'fewshot_split':'train',
            'generation_kwargs':{'until':['\n'],'do_sample':False,'temperature':0.,'max_gen_toks':32}}
    task=ConfigurableTask(config=config);task.set_fewshot_seed(42)
    original_sample=task.sampler.sample
    def tracked_sample(*args,**kwargs):
        docs=original_sample(*args,**kwargs)
        fewshots.append([dict(question=d['question'],first_gold=task.doc_to_target(d)[0]) for d in docs])
        return docs
    task.sampler.sample=tracked_sample;task.build_all_requests()
    for inst in task.instances:
        aliases=task.doc_to_target(inst.doc)
        if not isinstance(aliases,list) or not aliases:raise ValueError('TriviaQA gold aliases absent')
        context=inst.arguments[0];index=request(context,' '+aliases[0],True)
        # encode_pair must not steal a non-whitespace token from the generation prompt.
        if requests[index]['context']!=tok.encode(context,add_special_tokens=False)[-4064:]:
            raise ValueError('Confidence/generation prompt tokenization differs')
        rows['triviaqa'].append(dict(doc_id=str(inst.doc_id),question_id=inst.doc.get('question_id'),aliases=aliases,
                                    gold=0,character_lengths=[len(aliases[0])],requests=[index],
                                    prompt_tokens=requests[index]['context']))
    if len(rows['mmlu'])!=14042 or len(rows['triviaqa'])!=len(task.validation_docs()) or len(fewshots)!=len(rows['triviaqa']):
        raise ValueError('Task coverage mismatch')
    output=Path(output);output.mkdir(parents=True)
    prepared=dict(requests=requests,request_sha256=digest(requests),tasks=rows,
                  max_length=4096,generation_max_prompt_tokens=4064,
                  task_identity={k:digest(v) for k,v in rows.items()})
    target=write(output/'prepared.json',prepared)
    examples=write(output/'fewshot_examples.json',dict(seed=42,examples=fewshots))
    write(output/'FT16.json',dict(recorded_before_scoring=True,code_commit=code_commit,
          counts={k:len(v) for k,v in rows.items()},subjects=subjects,source_files=sources,
          packages={k:version(k) for k in ('lm-eval','datasets','transformers','tokenizers')},
          task_versions={'triviaqa':3.,'mmlu_continuation':1.},input_hashes=hashes,tokenizer_hashes=tokens,
          manifest_sha256=sha256(manifest),prepared=target,fewshot_examples=examples,
          triviaqa_overrides=config['generation_kwargs'],triviaqa_num_fewshot=5,fewshot_seed=42,
          mmlu_num_fewshot=0,prompt_policy='canonical lm-eval 0.4.13 task templates; MMLU answer texts, no letters',
          metric_policy='TriviaQA standard lower/punctuation/articles/underscore/whitespace normalization; max EM/F1 over aliases; first alias for confidence',
          conf_character_length='correct answer excluding target delimiter',
          registered_task_source_urls=['https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13/lm_eval/tasks/triviaqa',
                                       'https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13/lm_eval/tasks/mmlu/continuation']))
    print('F10_FT16_READY', {k:len(v) for k,v in rows.items()},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for k in ('manifest','tokenizer','output','code-commit'):p.add_argument('--'+k,required=True)
    a=p.parse_args();prepare(a.manifest,a.tokenizer,a.output,a.code_commit)
