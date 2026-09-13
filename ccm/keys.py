"""Exact causal bigram routing. No dictionary lookup in model forward."""
from pathlib import Path
import sqlite3
import numpy as np
import torch
from .contracts import require, digest_json, write_json, read_json, file_hash


def position_keys(ids, special_ids):
    """One immutable segment -> keys and eligibility at hidden positions."""
    ids = np.asarray(ids, dtype=np.int64)
    require(ids.ndim == 1 and np.all((ids >= 0) & (ids < 2**31)), "Invalid token IDs")
    keys = np.zeros(len(ids), dtype=np.int64)
    valid = np.zeros(len(ids), dtype=bool)
    if len(ids) >= 3:
        keys[1:-1] = (ids[:-2] << 32) | ids[1:-1]
        valid[1:-1] = ~np.isin(ids[:-2], special_ids) & ~np.isin(ids[1:-1], special_ids)
    return keys, valid


class Vocabulary:
    def __init__(self, keys, counts, metadata):
        self.keys = np.asarray(keys, dtype=np.int64)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.metadata = metadata
        require(len(self.keys) > 0 and len(self.keys) == len(self.counts), "Empty/invalid vocabulary")
        require(len(np.unique(self.keys)) == len(self.keys), "Duplicate keys")
        require(np.all(self.counts > 0), "Nonpositive counts")
        order = np.lexsort((self.keys, -self.counts))
        require(np.array_equal(order, np.arange(len(order))), "Wrong frequency/tie slot order")
        self.sort = np.argsort(self.keys)
        self.sorted_keys = self.keys[self.sort]
        self.hash = digest_json(dict(keys=self.keys.tolist(), counts=self.counts.tolist(), metadata=metadata))

    def route(self, ids, special_ids):
        keys, eligible = position_keys(ids, special_ids)
        i = np.searchsorted(self.sorted_keys, keys)
        safe = np.minimum(i, len(self.keys)-1)
        hit = eligible & (i < len(self.keys)) & (self.sorted_keys[safe] == keys)
        slots = np.full(len(ids), -1, dtype=np.int32)
        slots[hit] = self.sort[safe[hit]].astype(np.int32)
        return slots, eligible

    def save(self, path):
        path = Path(path)
        with path.open("xb") as f:
            np.savez(f, keys=self.keys, counts=self.counts)
        write_json(str(path)+".json", dict(metadata=self.metadata, hash=self.hash, sha256=file_hash(path)))

    @classmethod
    def load(cls, path):
        info = read_json(str(path)+".json")
        require(file_hash(path) == info["sha256"], "Vocabulary file checksum mismatch")
        with np.load(path, allow_pickle=False) as a:
            v = cls(a["keys"], a["counts"], info["metadata"])
        require(v.hash == info["hash"], "Vocabulary semantic hash mismatch")
        return v


def count_vocabulary(segments, special_ids, slots, db_path, metadata):
    """Disk-backed counts; bounded RAM even for a billion-token compiler."""
    require(not Path(db_path).exists(), "Count database already exists")
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE counts (key INTEGER PRIMARY KEY, n INTEGER NOT NULL)")
    pending = []
    for segment in segments:
        keys, eligible = position_keys(segment["tokens"], special_ids)
        unique, n = np.unique(keys[eligible], return_counts=True)
        pending.extend(zip(map(int, unique), map(int, n)))
        if len(pending) >= 100000:
            db.executemany("INSERT INTO counts VALUES (?,?) ON CONFLICT(key) DO UPDATE SET n=n+excluded.n", pending)
            db.commit()
            pending.clear()
    db.executemany("INSERT INTO counts VALUES (?,?) ON CONFLICT(key) DO UPDATE SET n=n+excluded.n", pending)
    db.commit()
    rows = db.execute("SELECT key,n FROM counts ORDER BY n DESC,key ASC LIMIT ?", (slots,)).fetchall()
    db.close()
    require(len(rows) == slots, "Not enough distinct eligible bigrams for locked capacity")
    return Vocabulary([r[0] for r in rows], [r[1] for r in rows], metadata)


def safe_gather(table, slots):
    require(slots.dtype in (torch.int32, torch.int64), "Slot IDs must be integer")
    require(bool(((slots >= -1) & (slots < table.shape[0])).all()), "Out-of-range slot")
    hit = slots >= 0
    values = table[slots.clamp_min(0).long()]
    return values.masked_fill(~hit.unsqueeze(-1), 0)
