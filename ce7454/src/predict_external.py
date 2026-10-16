"""Single-pass dev/test inference; labels remain on dev machine for scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from train import normalize


@torch.no_grad()
def predict_external(model, cache, output, checkpoint_sha256, batch_size=32):
    cache,output=Path(cache),Path(output)
    ready=json.loads((cache/'ready.json').read_text())
    if ready['status']!='passed' or ready['purpose']!='evaluation_only':
        raise ValueError('Unverified evaluation data')
    output.mkdir(parents=True,exist_ok=False)
    device=next(model.parameters()).device
    model.eval()
    receipt={'status':'passed','checkpoint_sha256':checkpoint_sha256,
             'manifest_sha256':ready['manifest_sha256'],'splits':{},'tta':False,'labels_used':False}
    for split,n in [('dev',900),('test_official',1500)]:
        meta=cache/(split+'_meta.json')
        if hashlib.sha256(meta.read_bytes()).hexdigest()!=ready['splits'][split]['meta_sha256']:
            raise ValueError('Metadata changed')
        files=json.loads(meta.read_text())['filenames']
        images=np.load(cache/(split+'_images.npy'),mmap_mode='r')
        if len(files)!=n or len(set(files))!=n or images.shape!=(n,512,512,3) or images.dtype!=np.uint8:
            raise ValueError('Wrong evaluation split')
        logits=np.empty((n,3),dtype=np.float32)
        for start in range(0,n,batch_size):
            x=torch.from_numpy(np.array(images[start:start+batch_size])).permute(0,3,1,2).to(device)
            x=normalize(x).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits[start:start+len(x)]=model(x).float().cpu().numpy()
        if not np.isfinite(logits).all():raise ValueError('Nonfinite predictions')
        pred=logits.argmax(1)
        data={'split':split,'checkpoint_sha256':checkpoint_sha256,
              'filenames':files,'logits':logits.tolist(),'predictions':pred.tolist()}
        dest=output/(split+'_predictions.json')
        dest.write_text(json.dumps(data,allow_nan=False)+'\n')
        receipt['splits'][split]={'n':n,'prediction_counts':np.bincount(pred,minlength=3).tolist(),
             'file':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        print('EXTERNAL_PREDICTIONS',split,receipt['splits'][split],flush=True)
    (output/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def score_predictions(predictions, labels):
    names=predictions['filenames']; raw=labels['labels']
    if len(set(names))!=len(names) or set(names)!=set(raw):
        raise ValueError('Prediction/label filename mismatch')
    truth=np.asarray([raw[f][0] if isinstance(raw[f],list) else raw[f] for f in names])
    logits=np.asarray(predictions['logits']); pred=np.asarray(predictions['predictions'])
    if logits.shape!=(len(names),3) or not np.isfinite(logits).all() or not np.array_equal(pred,logits.argmax(1)):
        raise ValueError('Invalid logits/predictions')
    if not np.isin(truth,[0,1,2]).all():raise ValueError('Invalid labels')
    return {'n':len(names),'correct':int((pred==truth).sum()),'accuracy':float((pred==truth).mean()),
            'confusion':np.bincount(truth*3+pred,minlength=9).reshape(3,3).tolist(),
            'label_note':labels.get('note'),'checkpoint_sha256':predictions['checkpoint_sha256']}
