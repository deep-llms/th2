import argparse,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('downloads',type=Path);p.add_argument('target',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
m=json.loads((a.downloads/'manifest.json').read_text());original=json.loads((a.target/'retrieval_manifest.json').read_text())
assert m['format']=='a2_replay_shared_prompts_v1'
assert m['original_manifest_sha256']==sha(a.target/'retrieval_manifest.json')
expected={x['member']:x for x in original['groups']['replays']['files']}
assert set(x['member'] for x in m['files'])==set(expected)
archive=a.downloads/'replays_shared.jsonl.gz'
with archive.open('xb') as f:
 for part in m['parts']:
  path=a.downloads/Path(part['path']).name
  assert path.stat().st_size==part['bytes'] and sha(path)==part['sha256']
  f.write(path.read_bytes())
assert sha(archive)==m['archive_sha256'] and archive.stat().st_size==m['archive_bytes']
restored=[]
with gzip.open(archive,'rt',encoding='utf-8') as stream:
 header=json.loads(next(stream));assert header['type']=='shared_prompts';shared=header['value']
 for line in stream:
  e=json.loads(line);assert e['type']=='replay';v=e['value'];record=expected[e['member']]
  assert e['sha256']==record['sha256'] and e['bytes']==record['bytes']
  for task,samples in v['samples'].items():
   for sample in samples:
    for key in ('doc','arguments','target'):
     assert sample[key] is None
     sample[key]=shared[task][str(sample['doc_id'])][key]
  text=(json.dumps(v,indent=2,allow_nan=False) if e['encoding']=='indent2' else json.dumps(v,allow_nan=False))+'\n'
  raw=text.encode();assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
  target=a.target/e['member'];assert target.resolve().is_relative_to(a.target.resolve())
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(raw)
  restored.append({'member':e['member'],'sha256':e['sha256'],'bytes':e['bytes']})
  print('RESTORED_EXACT',e['member'],flush=True)
assert len(restored)==len(expected)==32
with a.receipt.open('x') as f:json.dump({'files':restored,'count':32,'all_original_bytes_sha256_exact':True,'transport_archive_sha256':m['archive_sha256']},f,indent=2);f.write('\n')
