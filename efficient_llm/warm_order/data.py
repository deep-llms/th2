"""Select 512-token single-document windows without modifying restored data."""
from pathlib import Path
import hashlib
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

class Rows:
    def __init__(self, stem, length=512):
        self.ids=np.memmap(str(stem)+'.bin',mode='r',dtype='<u4').reshape(-1,length)
        self.starts=np.memmap(str(stem)+'.starts',mode='r',dtype='u1').reshape(-1,length)
        if self.ids.shape != self.starts.shape:raise ValueError('Boundary shape mismatch')
    def eligible(self):
        chunks=[]
        for i in range(0,len(self.ids),8192):
            a=self.starts[i:i+8192]
            chunks.append(np.flatnonzero(~a[:,1:].any(1))+i)
        return np.concatenate(chunks)
    def batch(self, ix, device):
        if np.any(self.starts[ix,1:]):raise ValueError('Cross-document row')
        return torch.as_tensor(np.array(self.ids[ix],dtype=np.int64),device=device)
