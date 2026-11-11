"""Make a bounded, deduplicated metadata-only kit; no modification of backups."""
import hashlib, io, json, os, socket, tarfile
from pathlib import Path

def sha(data): return hashlib.sha256(data).hexdigest()
def main():
 assert socket.gethostname()=='thiennh-p6-wc69-worker-0'
 b=Path('/mnt/local/_backup/deep-llms_th2')
 out=Path('/mnt/local/_outputs/deep-llms_th2/wc69-rebuild-metadata-export-20261111-a01');out.mkdir(exist_ok=False)
 old=b/'wc69-preservation-20261111-a01'
 ledger=old/'files.jsonl';assert sha(ledger.read_bytes())=='7cc191bd479f37fe300b8cc63c00309a9139aaa4da752cef9092ceadc44addce'
 expected={r['destination']:r for r in map(json.loads,ledger.read_text().splitlines())}
 roots=[p for p in b.iterdir() if p.is_dir() and p.name.startswith(('wc69-a2-data-rebuild-','wc69-a2-why-data-','wc69-a2-family-preparation-','wc69-a2-family-mixture-'))]
 roots += [old/'data_metadata',old/'source/scripts',old/'source/resources',old/'source/efficient_llm',old/'source/envs']
 selected={};omitted=[];objects={};total=0
 def add(p,explicit=False):
  nonlocal total
  rel=str(p.relative_to(b));assert not p.is_symlink() and p.is_file()
  for a in p.parents:
   assert not a.is_symlink()
   if a==b:break
  if rel in selected:return
  size=p.stat().st_size
  if size>(20<<20 if explicit else 2<<20):
   prior=expected.get(str(p),{});omitted.append(dict(path=rel,bytes=size,sha256=prior.get('sha256'),reason='large metadata; retained on node'));return
  data=p.read_bytes();h=sha(data);prior=expected.get(str(p))
  if prior:assert h==prior['sha256'] and len(data)==prior['bytes'],rel
  selected[rel]=dict(bytes=len(data),sha256=h)
  if h not in objects:
   objects[h]=data;total+=len(data);assert total<=128<<20,'Metadata unpacked cap exceeded'
 for root in roots:
  assert root.is_dir(),str(root)
  for d,ds,fs in os.walk(root,followlinks=False):
   for n in ds:assert not (Path(d)/n).is_symlink()
   for n in fs:
    p=Path(d)/n
    if p.suffix in ('.json','.jsonl','.sha256','.py','.txt','.yaml','.yml','.toml'):
     if any(w in n.lower() for w in ('credential','secret','hf_token','instruction')):continue
     add(p)
    elif any(w in n.lower() for w in ('order','offset','indices','permutation','documents')):
     prior=expected.get(str(p),{});omitted.append(dict(path=str(p.relative_to(b)),bytes=p.stat().st_size,sha256=prior.get('sha256'),reason='binary/large index retained on node; regenerate with recorded recipe'))
 for n in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
  p=b/'a2_family/shared_tokenizer'/n
  if p.exists():add(p,True)
 for n in ['plan.json','deleted.jsonl','complete.json','BACKUP_RETENTION_20261111.md','a2_prune_backup_inputs.py']:
  add(b/'wc69-input-prune-20261111-a01'/n,True)
 assert any(p.endswith('/mixture/effective_plan.json') for p in selected)
 assert 'a2_family/shared_tokenizer/tokenizer.json' in selected
 index=dict(schema='a2-rebuild-metadata-content-addressed-v1',backup_root=str(b),files=selected,omitted=omitted,
  limitations=['Large document orders/indices/traces remain on node; regenerate using exact seeds and code and verify recorded hashes.',
  'Exact public source bytes must remain available. No full independent mixture rebuild is claimed.',
  'Use with local artifacts/a2_reproduction_20261031.tar.gz and the project Git history.'],
  unique_objects=len(objects),unique_unpacked_bytes=total)
 archive=out/'rebuild_metadata.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=6) as t:
  for name,data in [('index.json',json.dumps(index,sort_keys=True,indent=2).encode()),*[(f'objects/{h}',data) for h,data in sorted(objects.items())]]:
   ti=tarfile.TarInfo(name);ti.size=len(data);ti.mode=0o644;ti.mtime=0;t.addfile(ti,io.BytesIO(data))
 assert archive.stat().st_size<=16<<20,'Compressed metadata cap exceeded; DO NOT EXPORT'
 with tarfile.open(archive) as t:
  for h in objects:assert sha(t.extractfile('objects/'+h).read())==h
 proof=dict(complete=True,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive.read_bytes()),
  paths=len(selected),objects=len(objects),unpacked_bytes=total,omitted_paths=len(omitted),source_roots=[str(p) for p in roots])
 (out/'verification.json').write_text(json.dumps(proof,indent=2))
 print('REBUILD_METADATA_READY',json.dumps(proof),flush=True)
if __name__=='__main__':main()
