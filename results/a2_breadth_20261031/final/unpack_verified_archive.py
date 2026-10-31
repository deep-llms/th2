from pathlib import Path
import hashlib,json,tarfile,shutil
m=json.loads(Path('temp/a2_breadth_export_manifest.json').read_text());blob=Path('temp/a2_breadth_results.part000');raw=blob.read_bytes()
assert len(raw)==m['archive_bytes'];assert hashlib.sha256(raw).hexdigest()==m['archive_sha256']
root=Path('results/a2_breadth_20261031/artifacts');root.mkdir(parents=True,exist_ok=False)
expected={x['path']:x for x in m['files']}
with tarfile.open(blob,'r:gz') as tar:
 members=tar.getmembers();assert len(members)==len(expected)
 for member in members:
  assert member.isfile() and member.name in expected and not member.name.startswith('/')
  target=root/member.name;assert target.resolve().is_relative_to(root.resolve())
  content=tar.extractfile(member).read();record=expected[member.name]
  assert len(content)==record['bytes'] and hashlib.sha256(content).hexdigest()==record['sha256'],member.name
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
source=root/'results/a2_breadth_20261031';receipt=json.loads(Path('results/a2_breadth_20261031/terminal_0712/perseq_hashes.json').read_text())
for record in receipt:
 assert hashlib.sha256((root/record['path']).read_bytes()).hexdigest()==record['sha256']
assert len(list((source/'perseq').glob('*/*.json')))==56
out=Path('results/a2_breadth_20261031/final');out.mkdir(exist_ok=True)
shutil.copy2('temp/a2_breadth_export_manifest.json',out/'export_manifest.json')
(out/'download_verification.json').write_text(json.dumps({'files':len(expected),'all_member_sha256_verified':True,'all56_perseq_match_health_receipt':True,'archive_sha256':m['archive_sha256'],'archive_bytes':len(raw),'local_originals':str(source),'export_commit':'a98de30119c3e0856b818249e513174403f1649f'},indent=2)+'\n')
print('Verified',len(expected),'files and56perseq; originals',source)
