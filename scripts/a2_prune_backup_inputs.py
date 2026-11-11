"""Two-stage, exact-file backup input pruning. Never touches original files."""
import argparse, concurrent.futures, hashlib, json, os, re, socket, stat
from pathlib import Path

LEDGER_SHA = '7cc191bd479f37fe300b8cc63c00309a9139aaa4da752cef9092ceadc44addce'
PATTERNS = [
 r'wc69-a2-family-mixture-20261107-a0[12]/mixture/sources/[^/]+/(text\.utf8|tokens\.bin)',
 r'wc69-a2-family-mixture-20261107-a02/mixture/stream/heal\.(bin|starts)',
 r'wc69-a2-family-preparation-20261107-a02/fwe/selected_tokens\.bin',
 r'wc69-a2-family-preparation-20261107-a02/fwe/203[01]/heal\.(bin|starts)',
 r'wc69-a2-family-preparation-20261107-a02/cx/(final|tuning)/tokens\.npy',
 r'wc69-a2-why-data-20261104-a01/(dclm|fwe)/203[12]/heal\.(bin|starts)',
]

def allowed(rel):
 return not any(x in ('..', '.') for x in rel.split('/')) and any(re.fullmatch(p, rel) for p in PATTERNS)

def sha(p):
 h = hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
 return h.hexdigest()

def regular(p, root):
 p.relative_to(root)
 for q in (p, *p.parents):
  if q.is_symlink(): raise ValueError('Symlink: ' + str(q))
  if q == root: break
 s = p.stat()
 if not stat.S_ISREG(s.st_mode): raise ValueError('Nonregular: ' + str(p))
 return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_mode]

def inventory(root, audit):
 out = {}
 for d, ds, fs in os.walk(root, followlinks=False):
  for n in ds:
   if (Path(d)/n).is_symlink(): raise ValueError('Directory symlink')
  for n in fs:
   p = Path(d)/n
   if p.is_relative_to(audit): continue
   out[str(p.relative_to(root))] = regular(p, root)
 return out

def write(p, obj):
 with p.open('x') as f:
  json.dump(obj, f, sort_keys=True, indent=2); f.flush(); os.fsync(f.fileno())

def verify_row(row, root, originals):
 rel = row['relative']; assert allowed(rel)
 p = root/rel; src = originals/rel
 assert str(src) == row['source']
 before = regular(p, root); original = regular(src, originals)
 assert before[:2] != original[:2], 'Hardlinked original'
 assert before[2] == original[2] == row['bytes']
 assert sha(p) == row['sha256'] and sha(src) == row['sha256'], 'Content mismatch'
 assert regular(p, root) == before and regular(src, originals) == original
 return dict(row, backup_stat=before, original_stat=original)

def preflight(rows, root, originals):
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  verified = list(pool.map(lambda r: verify_row(r, root, originals), rows))
 return verified

def execute(plan, root, originals, audit):
 # No unlink can occur until ALL candidates and ALL protected identities pass.
 rows = plan['rows']; assert len({r['relative'] for r in rows}) == len(rows)
 assert inventory(root, audit) == plan['inventory'], 'Backup changed since plan'
 for r in rows:
  assert allowed(r['relative'])
  assert regular(root/r['relative'], root) == r['backup_stat']
  assert regular(originals/r['relative'], originals) == r['original_stat']
 with (audit/'deleted.jsonl').open('x') as journal:
  for r in rows:
   p = root/r['relative']; src = originals/r['relative']
   assert regular(p, root) == r['backup_stat']
   assert regular(src, originals) == r['original_stat']
   journal.write(json.dumps(dict(state='intent', **r))+'\n'); journal.flush(); os.fsync(journal.fileno())
   p.unlink()
   journal.write(json.dumps(dict(state='deleted', **r))+'\n'); journal.flush(); os.fsync(journal.fileno())
 expected = dict(plan['inventory'])
 for r in rows:
  del expected[r['relative']]
  assert regular(originals/r['relative'], originals) == r['original_stat']
 after = inventory(root, audit)
 assert after == expected, 'Unexpected change in retained files'
 proof = dict(complete=True, removed_files=len(rows), removed_bytes=sum(r['bytes'] for r in rows),
  retained_files=len(after), retained_bytes=sum(s[2] for s in after.values()),
  protected_file_identity_check='exact path/dev/inode/size/mtime/mode equality',
  candidate_hash_check='both original and backup matched historical SHA256 before deletion',
  original_files_untouched=True, original_ledger_historical=True,
  preserved_legacy_culturax=True, durable_off_node_backup=False,
  plan_sha256=sha(audit/'plan.json'), journal_sha256=sha(audit/'deleted.jsonl'))
 write(audit/'complete.json', proof)
 print('PRUNE_COMPLETE', json.dumps(proof), flush=True)
 return proof

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('mode', choices=['plan','execute']); ap.add_argument('--plan-sha'); a=ap.parse_args()
 assert socket.gethostname() == 'thiennh-p6-wc69-worker-0'
 root=Path('/mnt/local/_backup/deep-llms_th2'); originals=Path('/mnt/local/_outputs/deep-llms_th2')
 audit=root/'wc69-input-prune-20261111-a01'; old=root/'wc69-preservation-20261111-a01'
 assert sha(old/'files.jsonl') == LEDGER_SHA
 if a.mode == 'plan':
  audit.mkdir(exist_ok=False)
  # Preserve both executable recipes and actual corrected mixture plan.
  witnesses=[old/'source/scripts'/n for n in ['a2_family_prepare_mix.py','a2_family_repair_mix.py','a2_family_prepare_fwe.py','a2_why_prepare.py']]
  witnesses += [root/'wc69-a2-family-mixture-20261107-a02/mixture/effective_plan.json',
   root/'wc69-a2-why-data-20261104-a01/corpus_plan.json',
   root/'a2_family/shared_tokenizer/tokenizer.json']
  witness_hashes={str(p.relative_to(root)): sha(p) for p in witnesses}
  rows=[]
  for line in (old/'files.jsonl').read_text().splitlines():
   r=json.loads(line); rel=str(Path(r['destination']).relative_to(root))
   if allowed(rel): rows.append(dict(relative=rel, source=r['source'], bytes=r['bytes'], sha256=r['sha256']))
  assert rows and len({r['relative'] for r in rows}) == len(rows)
  before=inventory(root,audit)
  print('PRUNE_PREFLIGHT',len(rows),sum(r['bytes'] for r in rows),flush=True)
  rows=preflight(rows,root,originals)
  assert inventory(root,audit)==before
  plan=dict(rows=rows,inventory=before,witness_hashes=witness_hashes,ledger_sha256=LEDGER_SHA)
  write(audit/'plan.json',plan)
  for r in rows: print('CANDIDATE',json.dumps({k:r[k] for k in ('relative','bytes','sha256')}),flush=True)
  print('PRUNE_PLAN_READY',json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),sha256=sha(audit/'plan.json'),witness_hashes=witness_hashes)),flush=True)
 else:
  assert a.plan_sha and sha(audit/'plan.json')==a.plan_sha
  plan=json.loads((audit/'plan.json').read_text())
  for rel,h in plan['witness_hashes'].items(): assert sha(root/rel)==h
  execute(plan,root,originals,audit)

if __name__=='__main__': main()
