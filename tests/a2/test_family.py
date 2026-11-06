import copy
import tempfile
import unittest
import numpy as np
import torch
from efficient_llm.a2_family import protocol as p
from efficient_llm.a2_family.model import document_attention, prune, save_student, reload_student

class FamilyProtocol(unittest.TestCase):
    def test_registered_grid_and_extensions(self):
        grid=p.grid();self.assertEqual(len(grid),6)
        self.assertEqual({x['lr'] for x in grid},{2.5e-5,5e-5,1e-4})
        self.assertEqual(p.removed_layers(36),list(range(24,33)))
        self.assertEqual(p.extension_arms(.02,.03,.01,.01),{'F3':['k6','k20'],'F4':['k64']})
        self.assertEqual(p.extension_arms(.02,.02,.01,.02),{'F3':[],'F4':['k64','k20']})
        self.assertFalse(p.changed(.01,.013))
        rows={(a,lr):n for a in ('k6','k20') for lr,n in [(2.5e-5,1.),(5e-5,2.),(1e-4,3.)]}
        self.assertTrue(p.choose_lr(rows)['extend_down'])
        rows.update({(a,1.25e-5):.9 for a in ('k6','k20')})
        self.assertFalse(p.choose_lr(rows)['extend_down'])
        self.assertEqual(p.choose_lr(rows)['lr'],1.25e-5)
    def test_log_budget_interpolation(self):
        self.assertAlmostEqual(p.break_even([.2,.8],[3.,1.],2.)['budget'],.4)
        self.assertEqual(p.break_even([.2,.8],[3.,2.5],2.)['gap'],.5)

class SmolIsolation(unittest.TestCase):
    def model(self):
        from transformers import SmolLM3Config,SmolLM3ForCausalLM
        torch.manual_seed(7)
        cfg=SmolLM3Config(vocab_size=32,pad_token_id=0,bos_token_id=1,eos_token_id=2,hidden_size=32,intermediate_size=64,
                         num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
                         head_dim=8,no_rope_layers=[1,0,1,0],max_position_embeddings=64,
                         tie_word_embeddings=True,attention_dropout=0.,use_cache=False)
        cfg._attn_implementation='sdpa'
        return SmolLM3ForCausalLM(cfg).float().eval()
    def test_ft1_mask_independence_and_roundtrip(self):
        model=self.model();ids=torch.tensor([[1,2,3,4,5,6]])
        starts=torch.tensor([[1,0,0,1,0,0]],dtype=torch.bool)
        mask,pos=document_attention(starts)
        self.assertEqual(pos.tolist(),[[0,1,2,0,1,2]])
        with torch.no_grad():
            ref=model(ids).logits
            unchanged=copy.deepcopy(model);prune(unchanged,[])
            self.assertTrue(torch.equal(ref,unchanged(ids).logits))
            out=model(ids,attention_mask=mask,position_ids=pos).logits
            changed=ids.clone();changed[:,:3]=torch.tensor([10,11,12])
            other=model(changed,attention_mask=mask,position_ids=pos).logits
            self.assertTrue(torch.equal(out[:,3:],other[:,3:]))
            alone=model(ids[:,3:]).logits
            torch.testing.assert_close(out[:,3:],alone,rtol=1e-6,atol=1e-7)
        prune(model,[1]);self.assertEqual([x.self_attn.use_rope for x in model.model.layers],[1,1,0])
        self.assertEqual([x.self_attn.layer_idx for x in model.model.layers],[0,2,3])
        for parameter in model.parameters():parameter.data.copy_(parameter.data.bfloat16().float())
        with tempfile.TemporaryDirectory() as d:
            save_student(model,d+'/model');loaded=reload_student(d+'/model')
            with torch.no_grad():self.assertTrue(torch.equal(model(ids).logits,loaded(ids).logits))
        self.assertEqual([x.self_attn.use_rope for x in loaded.model.layers],[1,1,0])

class TextRecovery(unittest.TestCase):
    def test_hash_mismatch_and_clipped_tail_never_accepted(self):
        import hashlib,json
        from pathlib import Path
        from unittest.mock import patch
        from scripts import a2_family_recover_cx as r
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            h=lambda b:hashlib.sha256(b).hexdigest()
            docs=[dict(group='tuning',split='train',index=1,gpt_neox_tokens=3,text_sha256=h(b'exact')),
                  dict(group='tuning',split='train',index=2,gpt_neox_tokens=3,text_sha256=h(b'wrong')),
                  dict(group='final',split='validation',index=0,gpt_neox_tokens=4,text_sha256=h(b'tail'))]
            (data/'documents.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in docs))
            streams={}
            for name,ids in [('tuning',[1,2,0,1,2,0]),('final',[1,2])]:
                raw=np.array(ids,dtype='<u4').tobytes();(data/(name+'.bin')).write_bytes(raw);streams[name]=dict(token_sha256=h(raw))
            (data/'manifest.json').write_text(json.dumps(dict(streams=streams)));(root/'tokenizer.json').write_text('{}')
            with patch.object(r,'CX_MANIFEST',h((data/'manifest.json').read_bytes())),patch.object(r,'CX_DOCUMENTS',h((data/'documents.jsonl').read_bytes())),patch.object(r,'TOKENIZER',h(b'{}')),patch.object(r,'Tokenizer') as tok:
                tok.from_file.return_value.decode.return_value='exact';r.recover(data,root/'tokenizer.json',root/'out')
            receipt=json.loads((root/'out/recovery.json').read_text())
            self.assertEqual(receipt['recovered'],dict(tuning=1,final=0));self.assertEqual(len(receipt['missing']),2)
            self.assertEqual(len((root/'out/exact_documents.jsonl').read_text().splitlines()),1)

class FweRetokenization(unittest.TestCase):
    def test_exact_source_order_hashes_and_partial_last_document(self):
        from pathlib import Path
        import json,hashlib
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer,models,pre_tokenizers
        from unittest.mock import patch
        from scripts import a2_family_prepare_fwe as f
        from efficient_llm.a2_family.common import SPEC_SHA
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();model=root/'model';model.mkdir()
            h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            texts=['a b','c d e','f','g h'];pq.write_table(pa.table({'text':texts}),root/'raw.parquet')
            vocab={'[EOS]':0,'[UNK]':1,**{c:i+2 for i,c in enumerate('abcdefgh')}}
            tok=Tokenizer(models.WordLevel(vocab=vocab,unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(model/'tokenizer.json'))
            pool=np.zeros(4,dtype=f.INDEX);pool['row']=np.arange(4);pool['tokens']=100 # force prefix expansion
            pool['hash']=[hashlib.sha256(t.encode()).digest() for t in texts];pool.tofile(old/'documents.bin')
            order=np.array([2,0,3,1],dtype='<u8');order.tofile(old/'document_order.bin');(old/'manifest.json').write_text('{}')
            raw=dict(local=str(root/'raw.parquet'),path='raw.parquet',bytes=(root/'raw.parquet').stat().st_size,sha256=h(root/'raw.parquet'))
            (root/'plan.json').write_text(json.dumps(dict(corpora={'fineweb_edu':{'files':[raw]}})))
            (root/'downloads.json').write_text(json.dumps(dict(spec_sha256=SPEC_SHA,checkpoints=[dict(label='k6',path=str(model),config={'eos_token_id':0},files=[dict(name='tokenizer.json',bytes=(model/'tokenizer.json').stat().st_size,sha256=h(model/'tokenizer.json'))])])))
            with patch.object(f,'PLAN_SHA',h(root/'plan.json')),patch.object(f,'MANIFEST_SHA',h(old/'manifest.json')),patch.object(f,'POOL_SHA',h(old/'documents.bin')),patch.object(f,'ORDER_SHA',h(old/'document_order.bin')),patch.object(f,'TARGETS',{2030:7,2031:4}):
                f.prepare(root/'plan.json',old,root/'downloads.json',root/'out')
            expected=[]
            for i in order:expected.extend(tok.encode(texts[int(i)]).ids+[0])
            self.assertEqual(np.fromfile(root/'out/2030/heal.bin',dtype='<u4').tolist(),expected[:7])
            self.assertEqual(np.fromfile(root/'out/2030/heal.starts',dtype='u1').tolist(),[1,0,1,0,0,1,0])
            trace=[json.loads(x) for x in (root/'out/2030/consumed_documents.jsonl').read_text().splitlines()]
            self.assertEqual([x['pool_index'] for x in trace],[2,0,3]);self.assertEqual(trace[-1]['consumed_tokens'],2)
            self.assertEqual(trace[-1]['full_document_tokens'],3)
            self.assertEqual((root/'out/document_order_2030.bin').read_bytes(),(old/'document_order.bin').read_bytes())

class CxAssembly(unittest.TestCase):
    def test_fallback_preserves_exact_doc_order_and_breadth_token_bytes(self):
        import json,hashlib
        from pathlib import Path
        import pyarrow as pa
        from tokenizers import Tokenizer,models,pre_tokenizers
        from scripts.a2_family_prepare_cx import prepare
        from efficient_llm.a2_family.common import SPEC_SHA
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=root/'rec';rec.mkdir();h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            texts=['a b','c d'];rows=[dict(group='final',split='validation',index=i,sequence_index=i,text_sha256=hashlib.sha256(t.encode()).hexdigest(),text=t) for i,t in enumerate(texts)]
            (rec/'exact_documents.jsonl').write_text(json.dumps(rows[0])+'\n');(rec/'recovery.json').write_text(json.dumps(dict(counts={'tuning':0,'final':2})))
            with pa.OSFile(str(root/'raw.arrow'),'wb') as f:
                table=pa.table({'text':texts})
                with pa.ipc.new_stream(f,table.schema) as w:w.write_table(table)
            tok=Tokenizer(models.WordLevel(vocab={'[UNK]':1,'[EOS]':0,'a':2,'b':3,'c':4,'d':5},unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(root/'tokenizer.json'))
            source=dict(name='raw.arrow',local=str(root/'raw.arrow'),bytes=(root/'raw.arrow').stat().st_size,sha256=h(root/'raw.arrow'))
            tf=dict(local=str(root/'tokenizer.json'),bytes=(root/'tokenizer.json').stat().st_size,sha256=h(root/'tokenizer.json'))
            missing=dict(rows[1],source='raw.arrow',local_row=1);missing.pop('text')
            cfg=dict(spec_sha256=SPEC_SHA,recovery_root=str(rec),recovery_document_sha256=h(rec/'exact_documents.jsonl'),files=[source],missing=[missing],shared_tokenizer=tf,tokenizer_eos_id=0)
            (root/'inputs.json').write_text(json.dumps(cfg));np.save(root/'tokens.npy',np.array([2,3,0,4,5,0],dtype=np.int32));np.save(root/'offsets.npy',np.array([0,3,6],dtype=np.int64))
            ref=dict(documents=2,tokens=6,predicted_tokens=4,token_sha256=h(root/'tokens.npy'),offset_sha256=h(root/'offsets.npy'));(root/'ref.json').write_text(json.dumps(ref))
            prepare(root/'inputs.json',root/'out',root/'ref.json')
            actual=json.loads((root/'out/complete.json').read_text());self.assertFalse(actual['unverified']);self.assertTrue(actual['checks'][0]['passed'])
            self.assertEqual(json.loads((root/'out/final/documents.json').read_text())['documents'][1]['text_sha256'],rows[1]['text_sha256'])
