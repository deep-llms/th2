#1 +60+a
#th2-q359-expanded-smoke-filters-20261012-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
python3 -u - <<'PY'
from pathlib import Path
from collections import Counter
import json
r=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a02')
for name in ('supervised/supervisor.json','supervised/burn.log'):
 p=r/name
 if p.exists():print(name,p.read_text()[-8000:])
for path in sorted((r/'supervised/run/expanded-smoke').glob('*/*/eval_samples.jsonl')):
 rows=[json.loads(x) for x in path.read_text().splitlines()];counts=Counter((x['task'],x['doc_id']) for x in rows)
 duplicate={key:n for key,n in counts.items() if n>1}
 print('SAMPLE_FILE',str(path.relative_to(r)),'rows',len(rows),'unique_docs',len(counts),'filters',dict(Counter(x.get('filter','none') for x in rows)))
 if duplicate:
  key=next(iter(duplicate));print('FIRST_MULTIPLE_RECORD',key,[(x.get('filter'),sorted(x)) for x in rows if (x['task'],x['doc_id'])==key])
PY
