"""CPU-only admission/repacking of verified A2 DCLM; no GPU or network use."""
import argparse,json,shutil,socket,time
from pathlib import Path
import numpy as np
from efficient_llm.dot_pool.data import make_kg,sha,write
from efficient_llm.dot_pool import SPEC_SHA

def prepare(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=False)
 if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
 src=Path('/mnt/local/_data/deep-llms_th2/a2_dclm_rebuild_20261103/packed/dclm')
 expected='b6b85acc4f0548d66464d6c3e2bf70e567b2abb29f5346f4385afe2443f13285'
 if sha(src/'manifest.json')!=expected:raise ValueError('DCLM source manifest differs')
 manifest=json.loads((src/'manifest.json').read_text());recipe=json.loads(Path('resources/a2_data_rebuild_wc69_20261103.json').read_text())
 if manifest['streams']!=recipe['corpus_expected']['streams']:raise ValueError('DCLM stream historical identities differ')
 tok=Path('/mnt/local/_models/deep-llms_th2/a2/dclm/P300/tokenizer.json')
 if sha(tok)!=manifest['tokenizer_sha256']:raise ValueError('Tokenizer does not match source')
 from tokenizers import Tokenizer
 tokenizer=Tokenizer.from_file(str(tok));assert tokenizer.get_vocab_size()==50277 and tokenizer.token_to_id('<|endoftext|>')==0
 write(root/'started.json',dict(host=socket.gethostname(),source=str(src),source_sha256=expected,partition=manifest['partition'],tokenizer=str(tok)))
 make_kg(root/'e1');out=root/'e2';out.mkdir();proof={}
 for dest,name,limit in [('train','heal',2000000000),('val','final',10000000)]:
  info=manifest['streams'][name];print('DOT_VERIFY_SOURCE',name,flush=True)
  for suffix,key in [('bin','token_sha256'),('starts','starts_sha256')]:
   if sha(src/(name+'.'+suffix))!=info[key]:raise ValueError('Source bytes differ: '+name+'.'+suffix)
  count=limit//512;tokens=np.memmap(src/(name+'.bin'),mode='r',dtype='<u4').reshape(-1,512);starts=np.memmap(src/(name+'.starts'),mode='r',dtype='u1').reshape(-1,512)
  if len(tokens)<count:raise ValueError('Insufficient source coverage')
  order=np.random.default_rng(0).permutation(count);np.save(out/(dest+'_order.npy'),order)
  boundaries=0;maximum=0
  with (out/(dest+'.bin')).open('xb') as ft,(out/(dest+'.starts')).open('xb') as fs:
   for j in range(0,count,4096):
    ix=order[j:j+4096];a=np.array(tokens[ix]);b=np.array(starts[ix]);maximum=max(maximum,int(a.max()))
    if np.any(b[:,1:].astype(bool)&(a[:,:-1]!=0)):raise ValueError('Document starts not preceded by EOS')
    boundaries+=int(b.sum());b[:,0]=1;a.tofile(ft);b.tofile(fs)
  if maximum>=50277:raise ValueError('Out-of-vocabulary data')
  proof[dest]=dict(source=str(src/name),source_identity=info,rows=count,tokens=count*512,dropped_to_512=limit-count*512,source_document_starts=boundaries,maximum_id=maximum,order_sha256=sha(out/(dest+'_order.npy')),token_sha256=sha(out/(dest+'.bin')),starts_sha256=sha(out/(dest+'.starts')))
  write(out/'progress.json',dict(streams=proof));print('DOT_STREAM_READY',dest,count,flush=True)
 write(out/'manifest.json',dict(status='complete',streams=proof,source_manifest_sha256=expected,tokenizer_sha256=sha(tok),eos_id=0,partition=manifest['partition'],packing='first prefix, reshape512 preserving explicit boundaries; row permutation seed0; reset fragment positions',train_val_disjoint='inherited content-hash partition in verified A2 corpus manifest'))
 write(root/'ready.json',dict(ready=True,e1=str(root/'e1'),e2=str(out),tokenizer=str(tok.parent),data_manifest_sha256=sha(out/'manifest.json'),kg_manifest_sha256=sha(root/'e1/manifest.json')))
 print('DOT_DATA_READY',json.dumps(json.loads((root/'ready.json').read_text())),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();prepare(a.output)
