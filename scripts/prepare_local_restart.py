"""Fixed local restart split from the already verified CulturaX parquet shard."""
import argparse
import hashlib,json
from pathlib import Path
from datasets import Dataset,concatenate_datasets
from pcc.joint import prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-inputs',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    source=json.loads(args.source_inputs.read_text())
    path=Path(source['parquet_path'])
    with path.open('rb') as f:
        sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha != 'b83d23dc436670adc95adf3111fc330902aebfce671775a831309c614a34004e':
        raise ValueError('Local CulturaX shard checksum mismatch')
    raw=Dataset.from_parquet(str(path)).select_columns(['text'])
    if len(raw) != 1106987:
        raise ValueError('Local CulturaX row count mismatch')
    train=concatenate_datasets([raw.select(range(20000)),raw.select(range(30000,610000))])
    dev=raw.select(range(1096987,1106987))
    train.save_to_disk(str(root/'train'))
    dev.save_to_disk(str(root/'dev'))
    config={'experiment':'joint-local-v3','model_path':source['model_path'],'train_data':str(root/'train'),'val_data':str(root/'dev'),'microbatch':1}
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (root/'source.json').write_text(json.dumps({'source':source,'sha256':sha,'train_rows':[[0,20000],[30000,610000]],'validation_rows':[[1096987,1106987]],'document_overlap':False,'previous_local_validation_excluded_from_train':True,'new_validation_slice':True,'scope':'new exploratory local experiment, not B200 split reproduction'},indent=2)+'\n')
    prepare(config,root/'inputs')
    print('LOCAL_INPUTS_READY',flush=True)


if __name__ == '__main__':
    main()
