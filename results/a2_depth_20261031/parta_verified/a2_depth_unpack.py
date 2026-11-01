from pathlib import Path
import hashlib,json,tarfile
m=json.loads(Path('temp/a2_depth_export_manifest.json').read_text());blob=Path('temp/a2_depth_results.part000');raw=blob.read_bytes()
assert len(raw)==m['archive_bytes'] and hashlib.sha256(raw).hexdigest()==m['archive_sha256']
root=Path('results/a2_depth_20261031/artifacts');root.mkdir(parents=True,exist_ok=False)
expected={r['path']:r for r in m['files']}
with tarfile.open(blob,'r:gz') as tar:
 members=tar.getmembers();assert len(members)==len(expected)
 for member in members:
  assert member.isfile() and member.name in expected and not member.name.startswith('/')
  target=root/member.name;assert target.resolve().is_relative_to(root.resolve())
  content=tar.extractfile(member).read();r=expected[member.name]
  assert len(content)==r['bytes'] and hashlib.sha256(content).hexdigest()==r['sha256'],member.name
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
rows=[json.loads(l.split(' ',1)[1]) for l in Path('temp/a2_depth_parta_health05.log').read_text().splitlines() if l.startswith('PERSEQ ')]
assert len(rows)==156
for r in rows:assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
out=Path('results/a2_depth_20261031/parta_verified');out.mkdir(exist_ok=True)
(out/'download_verification.json').write_text(json.dumps(dict(spec_sha256=m['spec_sha256'],files=len(expected),arrays=156,all_member_sha256_verified=True,all_arrays_match_terminal_receipt=True,archive_bytes=len(raw),archive_sha256=m['archive_sha256'],local_originals=str(root),export_commit='edb350309f6e0543f179ba63c1df6a775a69d128'),indent=2)+'\n')
print('Downloaded and SHA verified',len(expected),'files,',len(rows),'arrays')
