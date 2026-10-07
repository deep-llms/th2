"""Supervised English sentence-pair adaptation with the standard HF Trainer.

Supports A, P6/P6-iso and P7-simple families. Each pair is one causal document,
ending in EOS; classification reads that EOS hidden state. Padding is an
isolated dummy document. This does not call the inference-only eval adapter.
"""
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline')
os.environ.setdefault('NCCL_NVLS_ENABLE', '0')

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainerCallback, TrainingArguments, set_seed

from deep_kv.model import Context
from deep_kv import P6_VARIANTS, SIMPLE_MEMORY_ARMS
from deep_kv.training import save_module
from eval.benchmarks import local_dataset_paths
from eval.models import load_checkpoint, file_hash

TASKS = {'paws': ('paws_en', ('sentence1', 'sentence2'), ['different', 'paraphrase']),
         'nli': ('xnli_en', ('premise', 'hypothesis'), ['entailment', 'neutral', 'contradiction']),
         'stsb': ('stsb', ('sentence1', 'sentence2'), ['similarity']),
         'boolq': ('boolq', ('passage', 'question'), ['False', 'True'])}
ARMS = ('A', 'P6', 'P6-iso') + P6_VARIANTS + SIMPLE_MEMORY_ARMS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@dataclass
class TaskArguments:
    checkpoint: str
    task: str
    dataset_root: str
    dataset_manifest: str
    max_length: int = 512
    preprocessing_num_workers: int = 8
    attention_backend: str = 'fa4'
    selection_file: str | None = None
    evaluate_run: str | None = None
    smoke: bool = False


class PairClassifier(nn.Module):
    """Task-only, end-to-end fine-tuning; no proxy target/statistics updates."""
    def __init__(self, backbone, labels, seed):
        super().__init__()
        if backbone.arm not in ARMS:
            raise ValueError('Unsupported supervised arm: ' + backbone.arm)
        self.wrapped = backbone
        self.config = backbone.backbone.config
        # Plain task mode flag is deliberately not persisted in pretraining state.
        # This wrapper always re-enables it when reconstructing a fine-tuned model.
        if backbone.arm in ('P6-iso',) + P6_VARIANTS + SIMPLE_MEMORY_ARMS:
            for head in backbone.heads.values():
                head.task_finetuning = True
        # The language-model output projection is not used. Preserve tied input
        # embeddings as trainable; freeze only an independent, unused LM head.
        embedding = backbone.backbone.model.embed_tokens.weight
        if backbone.backbone.lm_head.weight is not embedding:
            backbone.backbone.lm_head.weight.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.score = nn.Linear(self.config.hidden_size, labels, bias=True)
            nn.init.normal_(self.score.weight, std=self.config.initializer_range)
            nn.init.zeros_(self.score.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError('Expected matching [batch, length] IDs and mask')
        valid = attention_mask.bool()
        lengths = valid.sum(-1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        if not bool((lengths >= 2).all()) or not torch.equal(valid, positions < lengths[:, None]):
            raise ValueError('Each pair needs at least two tokens and contiguous right padding')
        # Proxy backbone expects all packed slots valid. The padding gets its own
        # document, so no real token can see it; no loss/statistics use padding.
        segments = (~valid).long()
        positions = torch.where(valid, positions, positions - lengths[:, None])
        context = Context(input_ids, torch.ones_like(valid), positions, segments)
        hidden = self.wrapped.hidden_states(context)[0]
        last = hidden[torch.arange(len(hidden), device=hidden.device), lengths - 1]
        logits = self.score(last).float()
        out = {'logits': logits}
        if labels is not None:
            out['loss'] = (F.mse_loss(logits.squeeze(-1), labels.float())
                           if self.score.out_features == 1 else F.cross_entropy(logits, labels))
        return out


class PairCollator:
    def __init__(self, pad_id, regression=False):
        self.pad_id = pad_id
        self.regression = regression

    def __call__(self, rows):
        length = ((max(len(x['input_ids']) for x in rows) + 7) // 8) * 8
        ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            n = len(row['input_ids'])
            ids[i, :n] = torch.tensor(row['input_ids'])
            mask[i, :n] = 1
        return dict(input_ids=ids, attention_mask=mask,
                    labels=torch.tensor([x['labels'] for x in rows],
                                        dtype=torch.float32 if self.regression else torch.long))


def encode_pairs(batch, tokenizer, fields, max_length, task_name=None):
    # Tokenize the two strings separately, then longest-first truncate. This
    # prevents a long first sentence from removing the second one entirely.
    first = tokenizer(batch[fields[0]], add_special_tokens=False)['input_ids']
    second = tokenizer(batch[fields[1]], add_special_tokens=False)['input_ids']
    template = ('Passage: ', '\nQuestion: ', '\nAnswer:') if task_name == 'boolq' else (
        'Sentence 1: ', '\nSentence 2: ', '\nSimilarity:' if task_name == 'stsb' else '\nRelationship:')
    prefix, middle, suffix = [tokenizer.encode(t, add_special_tokens=False) for t in template]
    suffix += [tokenizer.eos_token_id]
    budget = max_length - len(prefix) - len(middle) - len(suffix)
    if budget < 2:
        raise ValueError('Context too short for sentence-pair template')
    rows, truncated = [], []
    for a, b in zip(first, second):
        truncated.append(len(a) + len(b) > budget)
        if len(a) + len(b) > budget:
            na = min(len(a), max(budget // 2, budget - len(b)))
            nb = min(len(b), budget - na)
            a, b = a[:na], b[:nb]
        rows.append(prefix + a + middle + b + suffix)
    return dict(input_ids=rows, labels=batch['label'], truncated=truncated)


def document_key(row, fields, symmetric):
    pair = [' '.join(row[f].split()) for f in fields]
    if symmetric:
        pair.sort()
    return hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode()).hexdigest()


def supervised_splits(raw, task_name):
    """Freeze train-derived development data; public validation is final holdout.

    BoolQ official test labels are hidden. Never load/use that split here.
    Group identical inputs before a fixed 10% split,
    so repeated inputs cannot cross training/development boundaries.
    """
    from datasets import DatasetDict
    if task_name == 'stsb':
        fields = TASKS[task_name][1]
        heldout = {document_key(row, fields, True) for row in raw['test']}
        keep = [i for i,row in enumerate(raw['validation']) if document_key(row, fields, True) not in heldout]
        return DatasetDict(train=raw['train'], validation=raw['validation'].select(keep), test=raw['test'])
    if task_name != 'boolq':
        return raw
    fields = TASKS[task_name][1]
    symmetric = False
    heldout = {document_key(row, fields, symmetric) for row in raw['validation']}
    keys = [document_key(row, fields, symmetric) for row in raw['train']]
    groups = sorted(set(keys)-heldout, key=lambda key: hashlib.sha256(('split-42:'+key).encode()).hexdigest())
    if len(groups) < 10:
        raise ValueError('Too few distinct training groups for fixed development split')
    development = set(groups[:max(1, len(groups)//10)])
    train = [i for i,key in enumerate(keys) if key not in heldout and key not in development]
    dev = [i for i,key in enumerate(keys) if key in development]
    return DatasetDict(train=raw['train'].select(train), validation=raw['train'].select(dev), test=raw['validation'])


def prepare_data(task, tokenizer, train_args):
    if task.task not in TASKS or task.max_length % 8 or task.max_length > 2048:
        raise ValueError('Unsupported task/context length')
    name, fields, label_names = TASKS[task.task]
    # All ranks verify input hashes. Dataset maps are serialized across ranks by
    # main_process_first and reuse HF cache exactly as in the training pipeline.
    mappings = local_dataset_paths(task.dataset_root, task.dataset_manifest)
    config = mappings[name]
    with train_args.main_process_first(desc='local supervised data and tokenization'):
        raw = load_dataset(config['dataset_path'], **config['dataset_kwargs'])
        if task.task == 'stsb':
            for split in ('train', 'validation', 'test'):
                labels = np.asarray(raw[split]['score'])
                if not np.isfinite(labels).all() or not ((labels >= 0) & (labels <= 1)).all():
                    raise ValueError('Pinned STS-B source requires finite normalized 0–1 scores')
                raw[split] = raw[split].map(lambda batch: {'label':[5.0*x for x in batch['score']]},
                    batched=True, remove_columns=['score'], desc='Restore STS-B 0–5 scale')
        elif raw['train'].features['label'].names != (['0', '1'] if task.task == 'paws' else label_names):
            raise ValueError('Unexpected dataset label ordering')
        source_train_rows = len(raw['train'])
        source_dev_rows = len(raw['validation'])
        raw = supervised_splits(raw, task.task)
        # Hold out validation and test sentence pairs. Inspect text identity only,
        # never use test labels to filter, tune, or select hyperparameters.
        heldout = {document_key(r, fields, task.task in ('paws', 'stsb'))
                   for split in ('validation', 'test') for r in raw[split]}
        keep, digest = [], hashlib.sha256()
        for i, row in enumerate(raw['train']):
            key = document_key(row, fields, task.task in ('paws', 'stsb'))
            if key not in heldout:
                keep.append(i)
                digest.update((key + ':' + str(row['label']) + '\n').encode())
        original = len(raw['train'])
        raw['train'] = raw['train'].select(keep)
        splits = ['test'] if task.evaluate_run else ['train', 'validation']
        encoded, counts, truncation, order_hashes = {}, {}, {}, {}
        for split in splits:
            # Reload smoke uses development examples, never test scores.
            source_split = 'validation' if task.smoke and task.evaluate_run else split
            part = raw[source_split]
            if task.smoke:
                part = part.select(range(min(len(part), 256 if split == 'train' else 32)))
            if split == 'train' and not task.smoke:
                order_hashes[split] = digest.hexdigest()
            else:
                order = hashlib.sha256()
                for row in part:
                    order.update((document_key(row, fields, False)+':'+str(row['label'])+'\n').encode())
                order_hashes[split] = order.hexdigest()
            part = part.map(encode_pairs, batched=True, num_proc=task.preprocessing_num_workers,
                            fn_kwargs=dict(tokenizer=tokenizer, fields=fields, max_length=task.max_length, task_name=task.task),
                            remove_columns=part.column_names, desc=f'Tokenize {task.task}/{split}')
            counts[split] = len(part)
            truncation[split] = sum(part['truncated'])
            encoded[split] = part.remove_columns('truncated')
    info = dict(task=task.task, label_names=label_names, rows=counts, truncated_rows=truncation,
                train_rows_before_filter=original, train_rows_after_filter=len(keep),
                heldout_overlap_removed=original-len(keep), train_order_sha256=digest.hexdigest(),
                dataset_manifest_sha256=file_hash(task.dataset_manifest),
                fingerprints={k: v._fingerprint for k, v in encoded.items()},
                split_order_sha256=order_hashes,
                evaluation_split='validation' if task.smoke or not task.evaluate_run else 'test',
                tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest())
    if task.task == 'stsb':
        info.update(split_policy='official train/dev/test; source score multiplied by 5; dev/test exact pairs removed from dev',
                    development_overlap_removed=source_dev_rows-len(raw['validation']),
                    final_source_split='test', final_rows=len(raw['test']))
    if task.task == 'boolq':
        info.update(split_policy='train-derived 10% grouped development, fixed split-42; public validation final holdout',
                    source_train_rows=source_train_rows, development_rows=len(raw['validation']),
                    final_source_split='validation', final_rows=len(raw['test']),
                    heldout_overlap_removed=source_train_rows-len(raw['train'])-len(raw['validation']))
    return encoded, info


def metrics(prediction):
    logits, labels = map(np.asarray, prediction)
    if not np.isfinite(logits).all() or not np.isfinite(labels).all():
        raise ValueError('Nonfinite task predictions/labels')
    if logits.ndim == 2 and logits.shape[1] == 1:
        from scipy.stats import rankdata
        predicted = logits[:, 0]
        if predicted.shape != labels.shape or len(labels) < 2:
            raise ValueError('Invalid regression prediction shape/count')
        def correlation(a, b):
            # Defined protocol for a constant predictor: no correlation skill.
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        pearson = correlation(predicted, labels)
        spearman = correlation(rankdata(predicted), rankdata(labels))
        return dict(pearson=pearson, spearman=spearman, correlation=(pearson+spearman)/2,
                    mse=float(np.mean((predicted-labels)**2)))
    return {'accuracy': float((logits.argmax(-1) == labels).mean())}


class GradientCheck(TrainerCallback):
    """Fail before first update if any task-connected parameter has no gradient."""
    def __init__(self):
        self.report = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.report is not None:
            return
        missing, bad, norms = [], [], {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                missing.append(name)
            elif not bool(torch.isfinite(p.grad).all()):
                bad.append(name)
            else:
                group = 'proxy' if '.heads.' in name else 'classifier' if name.startswith('score.') else 'backbone'
                norms[group] = norms.get(group, 0.) + float(p.grad.float().square().sum())
        if missing or bad or any(v <= 0 for v in norms.values()):
            raise ValueError(f'Invalid task gradients: missing={missing}, nonfinite={bad}, norms={norms}')
        self.report = dict(status='passed', squared_gradient_norms=norms,
                           task_gradient_parameters=sum(p.requires_grad for p in model.parameters()))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in (logs or {}).values()):
            raise ValueError('Nonfinite Trainer metric')


class SupervisedTrainer(Trainer):
    # Reuse the proven tied-weight-safe nn.Module saving implementation.
    _save = save_module

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False


def run(config_path):
    task, args = HfArgumentParser((TaskArguments, TrainingArguments)).parse_json_file(str(Path(config_path).resolve()))
    if args.gradient_accumulation_steps != 1:
        raise ValueError('This matched per-example protocol requires GAS=1')
    if args.report_to or args.gradient_checkpointing:
        raise ValueError('Protocol uses local Trainer logs and no activation checkpointing')
    if args.resume_from_checkpoint:
        raise ValueError('This offline study starts fresh from a pretraining checkpoint')
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Fine-tuning output must be fresh')
    set_seed(args.seed)
    adapter, tok_source, source = load_checkpoint(task.checkpoint, 'cpu', attention_backend='sdpa')
    if source['arm'] not in ARMS or source['step'] != 2500 and not task.smoke:
        raise ValueError('Unexpected source checkpoint arm/step')
    if task.max_length > source['context_length']:
        raise ValueError('Downstream context exceeds the pretrained context limit')
    # Construct strictly on CPU, then select the requested runtime kernel before
    # Trainer moves the model to the local rank's device.
    wrapped = adapter.wrapped
    if task.attention_backend == 'fa4':
        from deep_kv.fa4 import load_kernel
        if args.device.type != 'cuda' or not args.bf16:
            raise ValueError('FA4 requires CUDA BF16')
        wrapped.fa4_kernel, wrapped.fa4_metadata = load_kernel()
    elif task.attention_backend != 'sdpa':
        raise ValueError('Unknown attention backend')
    wrapped.attention_backend = task.attention_backend
    tokenizer = AutoTokenizer.from_pretrained(tok_source, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define EOS')
    data, data_info = prepare_data(task, tokenizer, args)
    model = PairClassifier(wrapped, len(TASKS[task.task][2]), args.seed)
    del adapter
    selection = None
    if task.selection_file:
        selection = json.loads(Path(task.selection_file).read_text())
        if selection['arm'] != source['arm'] or selection['task'] != task.task or selection['status'] != 'selected':
            raise ValueError('Wrong LR selection')
        args.learning_rate = selection['learning_rate']
    if task.evaluate_run:
        saved = json.loads((Path(task.evaluate_run)/'result.json').read_text())
        if saved['source']['checkpoint_sha256'] != source['checkpoint_sha256'] or saved['data']['task'] != task.task:
            raise ValueError('Fine-tuned model/source/task mismatch')
        if saved['data']['tokenizer_sha256'] != data_info['tokenizer_sha256']:
            raise ValueError('Fine-tuned tokenizer mismatch')
        if saved['data']['dataset_manifest_sha256'] != data_info['dataset_manifest_sha256']:
            raise ValueError('Fine-tuned data mismatch')
        if saved['seed'] != args.seed or saved['learning_rate'] != args.learning_rate or saved['max_length'] != task.max_length:
            raise ValueError('Fine-tuned seed/rate/context mismatch')
        weights = Path(task.evaluate_run)/'best/model.safetensors'
        if file_hash(weights) != saved['weight_sha256']:
            raise ValueError('Fine-tuned weights changed since training')
        model.load_state_dict(load_file(str(weights)), strict=True)
    callback = GradientCheck()
    initial_buffers = {k: v.clone() for k, v in wrapped.named_buffers()}
    trainer = SupervisedTrainer(model=model, args=args, processing_class=tokenizer,
        train_dataset=data.get('train'), eval_dataset=data.get('validation'),
        data_collator=PairCollator(tokenizer.eos_token_id, regression=task.task == 'stsb'), compute_metrics=metrics,
        callbacks=[callback])
    # Trainer initializes its own output directory; only rank zero writes shared artifacts.
    if trainer.is_world_process_zero():
        write_json(output/'protocol.json', dict(task=vars(task), training=args.to_dict(), source=source, data=data_info,
                    adaptation='task-only end-to-end supervised learning; no auxiliary loss; fixed normalization buffers'))
    start = time.monotonic()
    if task.evaluate_run:
        prediction = trainer.predict(data['test'], metric_key_prefix='test')
        results = prediction.metrics
        if len(prediction.label_ids) != len(data['test']):
            raise ValueError('Incorrect test example count')
        if trainer.is_world_process_zero():
            np.savez_compressed(output/'predictions.npz', logits=prediction.predictions, labels=prediction.label_ids)
        global_step, gradient_report = saved['global_step'], saved['gradients']
    else:
        trained = trainer.train()
        results = trainer.evaluate()
        results.update({k: v for k, v in trained.metrics.items() if k.startswith('train_')})
        trainer.save_model(str(output/'best'))
        trainer.save_state()
        global_step, gradient_report = trainer.state.global_step, callback.report
        if gradient_report is None or trainer.state.best_model_checkpoint is None:
            raise ValueError('Missing gradient check or development-selected checkpoint')
    for name, value in wrapped.named_buffers():
        if not torch.equal(value.detach().cpu(), initial_buffers[name]):
            raise ValueError('Fine-tuning changed normalization/rotary buffers: '+name)
    if trainer.is_world_process_zero():
        write_json(output/'result.json', dict(status='completed', source=source, data=data_info,
             seed=args.seed, learning_rate=args.learning_rate, global_step=global_step,
             best_model_checkpoint=trainer.state.best_model_checkpoint, metrics=results,
             gradients=gradient_report, elapsed_seconds=time.monotonic()-start,
             world_size=args.world_size, max_length=task.max_length, smoke=task.smoke,
             attention_backend=task.attention_backend, evaluate_run=task.evaluate_run,
             weight_sha256=file_hash(output/'best/model.safetensors') if not task.evaluate_run else file_hash(weights)))
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    run(sys.argv[1])
