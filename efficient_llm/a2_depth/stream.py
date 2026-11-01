"""DT6: anchor the unchanged 30B stream to historical Stage1 byte hashes."""
import hashlib
from pathlib import Path
import numpy as np
from efficient_llm.a2.data import PackedRows
from efficient_llm.diagnostic_data import sha256
from .common import Checks, write


def hashes(path, prefix_bytes):
    full=hashlib.sha256();prefix=hashlib.sha256();size=0
    with Path(path).open('rb') as handle:
        while block:=handle.read(8*1024*1024):
            full.update(block)
            if size<prefix_bytes:prefix.update(block[:prefix_bytes-size])
            size+=len(block)
    return dict(full_sha256=full.hexdigest(),prefix_sha256=prefix.hexdigest(),bytes=size)


def verify(root, historical, output, *, prefix_rows=781248, selected_rows=3124992):
    """CPU only; reads existing files, never repacks, reorders or truncates them.

    Historical inputs recorded a full-stream hash, not a prefix-only hash. Full
    byte identity authenticates today's prefix as the historical Stage1 prefix;
    separately hashing the actual PackedRows view verifies the C reader/order.
    """
    root,output=Path(root),Path(output)
    checks=Checks(output/'DT6_checks.json');result={}
    info=historical['data_manifest']['streams']['heal']
    def inspect():
        checks.record('DT6_manifest',sha256(root/'manifest.json')==historical['provenance']['data_sha256'])
        before={name:(root/name).stat() for name in ('heal.bin','heal.starts')}
        tokens=hashes(root/'heal.bin',prefix_rows*info['length']*4)
        starts=hashes(root/'heal.starts',prefix_rows*info['length'])
        checks.record('DT6_historical_full_tokens',tokens['full_sha256']==info['token_sha256'],tokens)
        checks.record('DT6_historical_full_boundaries',starts['full_sha256']==info['starts_sha256'],starts)
        checks.record('DT6_physical_lengths',tokens['bytes']==info['rows']*info['length']*4 and starts['bytes']==info['rows']*info['length'])
        rows=PackedRows(root,'heal',count=selected_rows,verify=False)
        checks.record('DT6_count',len(rows)==selected_rows and rows.total>=selected_rows and len(rows)>=prefix_rows,
                      dict(selected=len(rows),available=rows.total,required=selected_rows))
        checks.record('DT6_sequential_order',np.array_equal(rows.order,np.arange(selected_rows)))
        view=hashlib.sha256()
        for begin in range(0,prefix_rows,4096):
            indices=rows.order[begin:min(begin+4096,prefix_rows)]
            view.update(rows.ids[indices].astype('<u4',copy=False).tobytes(order='C'))
        checks.record('DT6_reader_prefix',view.hexdigest()==tokens['prefix_sha256'],dict(prefix_sequences=prefix_rows,
                      reader_sha256=view.hexdigest(),physical_prefix_sha256=tokens['prefix_sha256']))
        checks.record('DT6_files_stable',all((s.st_size,s.st_mtime_ns,s.st_ino)==
            ((root/name).stat().st_size,(root/name).stat().st_mtime_ns,(root/name).stat().st_ino) for name,s in before.items()))
        result.update(tokens=tokens,boundaries=starts,reader_prefix_sha256=view.hexdigest(),
                      selected_sequences=len(rows),available_sequences=rows.total)
    checks.attempt('DT6_read_stream',inspect)
    result.update(passed=bool(checks.items) and all(c['passed'] for c in checks.items),
        checks=checks.items,flags=[c for c in checks.items if not c['passed']],
        historical_manifest_sha256=historical['provenance']['data_sha256'],
        historical_full_token_sha256=info['token_sha256'],prefix_sequences=prefix_rows,
        method='Historical full-stream hash authenticates the prefix; separate PackedRows view hash checks the unchanged sequential reader. No historical prefix-only digest was recorded.',
        stream=str(root),action='record all failures and continue registered independent work')
    write(output/'DT6.json',result)
    return result
