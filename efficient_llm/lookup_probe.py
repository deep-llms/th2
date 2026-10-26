"""Seeded synthetic associative lookup sanity check, not a benchmark reproduction."""
import random
import torch

COLORS=['red','blue','green','yellow','black','white','orange','purple']


def build(tokenizer, lengths=(2048,8192), samples=16, seed=9042):
    encode=lambda text:tokenizer.encode(text,add_special_tokens=False)
    choices=[encode(' '+word) for word in COLORS]
    if any(len(x)!=1 for x in choices) or len({x[0] for x in choices})!=8:
        raise ValueError('Lookup choices must be distinct single tokens')
    candidate_ids=[x[0] for x in choices]
    header=encode('Complete the lookup using the record with the matching id.\n'
        'Record id=EXAMPLE1; value: circle.\nLookup id=EXAMPLE1; value: circle.\n'
        'Record id=EXAMPLE2; value: square.\nLookup id=EXAMPLE2; value: square.\n\nRecords:\n')
    filler=encode(' These routine notes describe the meeting agenda and its schedule. The report continues with ordinary details.\n')
    if set(filler)&set(candidate_ids):raise ValueError('Filler contains a candidate token')
    rows=[]
    for length in lengths:
        for depth in ('near','10%','50%','90%'):
            for sample in range(samples):
                # Identical records/answer across context lengths and locations.
                rng=random.Random(seed+sample);keys=rng.sample(range(100000,999999),8)
                correct=sample%8;other=[i for i in range(8) if i!=correct];rng.shuffle(other)
                prefix=header+encode(''.join(f'Record id=K{keys[i]}; value: {COLORS[i]}.\n' for i in other)+'\nNotes:\n')
                needle=encode(f'\nRecord id=K{keys[correct]}; value: {COLORS[correct]}.\n')
                question=encode(f'\nLookup id=K{keys[correct]}; value:')
                if needle.count(candidate_ids[correct])!=1 or candidate_ids[correct] in prefix:
                    raise ValueError('Target token is ambiguous or leaked outside target record')
                start=length-len(question)-len(needle)-8 if depth=='near' else int(length*int(depth[:-1])/100)
                before=start-len(prefix);after=length-start-len(needle)-len(question)
                if min(before,after)<0:raise ValueError('Context too short for lookup layout')
                repeat=lambda n:(filler*((n+len(filler)-1)//len(filler)))[:n]
                ids=prefix+repeat(before)+needle+repeat(after)+question
                position=start+needle.index(candidate_ids[correct])
                if len(ids)!=length or (depth=='near' and length-position>128) or (depth!='near' and length-position<=128):
                    raise ValueError('Lookup placement contract violated')
                rows.append(dict(tokens=torch.tensor(ids),label=correct,length=length,depth=depth,
                    sample=sample,target_token_position=position,query_id=f'K{keys[correct]}'))
    return dict(protocol='synthetic_record_lookup_v1',seed=seed,candidates=COLORS,
        candidate_ids=candidate_ids,rows=rows,chance_accuracy=1/8,
        notes=['Balanced eight-way next-token choice, no answer-length bias or test-time fitting.',
               'Same records/labels at each context/location; cells are correlated, not independent replications.',
               'Interpret distant-context failure only alongside dense and near-target controls.'])
