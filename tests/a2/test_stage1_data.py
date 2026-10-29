import hashlib,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models
from scripts.a2_stage1_data import prepare,INDEX
from scripts.a2_prepare_corpora import sha

class Stage1DataTests(unittest.TestCase):
 def fixture(self,root,overlap=False):
  tok=root/'tok';tok.mkdir();Tokenizer(models.WordLevel({'<|endoftext|>':0,'x':1,'y':2},unk_token='x')).save(str(tok/'tokenizer.json'))
  part=root/'parts/dclm/00000';part.mkdir(parents=True)
  docs=[(0,hashlib.sha256(b'a').digest(),3,0),(1,hashlib.sha256(b'b').digest(),4,0),(2,hashlib.sha256(b'c').digest(),3,0),(3,hashlib.sha256(b'a' if overlap else b'd').digest(),4,1)]
  (part/'documents.bin').write_bytes(b''.join(struct.pack('<Q32sQB',*d) for d in docs))
  rec={'index':0};plan={'root':str(root),'corpora':{'dclm':{'files':[rec]}},'tokenizer_path':str(tok),'tokenizer_files':{'tokenizer.json':sha(tok/'tokenizer.json')}}
  (root/'plan.json').write_text(json.dumps(plan));(root/'spec.md').write_text('spec')
  saved={'source':rec,'files':{'documents.bin':{'bytes':(part/'documents.bin').stat().st_size,'sha256':sha(part/'documents.bin')}}};(part/'complete.json').write_text(json.dumps(saved))
  packed=root/'packed/dclm';packed.mkdir(parents=True)
  # Last source document cut after its first token; must be excluded before shuffle.
  ids=np.array([1,1,0,2,2,2,0,1],dtype='<u4');starts=np.array([1,0,0,1,0,0,0,1],dtype='u1');ids.tofile(packed/'heal.bin');starts.tofile(packed/'heal.starts')
  np.array([1,1,1,0],dtype='<u4').tofile(packed/'tuning.bin');np.array([1,0,0,0],dtype='u1').tofile(packed/'tuning.starts')
  streams={g:dict(tokens=n,rows=n//2,length=2,token_sha256=sha(packed/(g+'.bin')),starts_sha256=sha(packed/(g+'.starts'))) for g,n in [('heal',8),('tuning',4)]}
  m=dict(status='complete',plan_sha256=sha(root/'plan.json'),tokenizer_sha256=plan['tokenizer_files']['tokenizer.json'],streams=streams,source_prefixes=[dict(group=g,source_index=0,tokens=n,shard_receipt_sha256=sha(part/'complete.json')) for g,n in [('heal',8),('tuning',4)]])
  (packed/'manifest.json').write_text(json.dumps(m));return sha(packed/'manifest.json')
 def test_shuffle_complete_docs_nested_prefix_and_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);h=self.fixture(p);prepare(p/'plan.json',h,p/'out',p/'spec.md',length=2,minimum=2)
   m=json.loads((p/'out/manifest.json').read_text());self.assertEqual(m['excluded_original_tail_tokens'],1);self.assertTrue(m['T7']['passed'])
   order=np.random.default_rng(2030).permutation(2);docs=[np.array([1,1,0]),np.array([2,2,2,0])]
   expected=np.concatenate([docs[i] for i in order])[:6];actual=np.fromfile(p/'out/heal.bin',dtype='<u4');np.testing.assert_array_equal(actual,expected)
   self.assertEqual(len(np.fromfile(p/'out/documents.bin',dtype=INDEX)),2)
   with self.assertRaises(ValueError):prepare(p/'plan.json','0'*64,p/'bad',p/'spec.md',length=2,minimum=2)
 def test_content_overlap_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);h=self.fixture(p,True)
   with self.assertRaisesRegex(ValueError,'overlap'):prepare(p/'plan.json',h,p/'out',p/'spec.md',length=2,minimum=2)
