"""Completion-only training derived from the exact official evaluation task."""
import torch
from torch.utils.data import Dataset

TASK_CONFIGS = {
    'hellaswag': dict(max_length=256, epochs=3, lr=2e-5, batch_size=16),
    'arc_easy': dict(max_length=256, epochs=3, lr=2e-5, batch_size=32),
    'xnli': dict(max_length=256, epochs=3, lr=2e-5, batch_size=32),
}


def format_example(task, doc):
    if task.OUTPUT_TYPE != 'multiple_choice' or task.multiple_input:
        raise ValueError('Only ordinary multiple-choice generative training is supported')
    prompt = task.doc_to_text(doc)
    choices = task.doc_to_choice(doc)
    target = task.doc_to_target(doc)
    if isinstance(target, str) and target.isdigit():
        target = int(target)
    if not isinstance(prompt, str) or type(target) is not int or not 0 <= target < len(choices):
        raise ValueError('Expected string prompt and one integer answer index')
    completion = task.config.target_delimiter + choices[target]
    return prompt, completion


def encode_example(prompt, completion, tokenizer, max_length):
    if max_length < 2 or tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Need max_length >= 2 and pad/EOS IDs')
    # Match HFLM's causal _encode_pair, including trailing-context whitespace.
    if prompt:
        spaces = len(prompt) - len(prompt.rstrip())
        if spaces:
            completion = prompt[-spaces:] + completion
            prompt = prompt[:-spaces]
    encode = lambda text: tokenizer(text, add_special_tokens=False)['input_ids']
    if prompt:
        context = encode(prompt)
        continuation = encode(prompt+completion)[len(context):]
    else:
        continuation = encode(completion)
        if not continuation:
            return None
        prefix = tokenizer.eos_token_id
        if continuation[0] == prefix:
            context, continuation = continuation[:1], continuation[1:]
        else:
            context = [prefix]
    if not context or not continuation:
        return None
    ids = (context+continuation)[:max_length]
    labels = ([-100]*len(context)+continuation)[:max_length]
    if not any(label != -100 for label in labels[1:]):
        return None
    pads = max_length-len(ids)
    return dict(input_ids=torch.tensor(ids+[tokenizer.pad_token_id]*pads),
                labels=torch.tensor(labels+[-100]*pads),
                attention_mask=torch.tensor([1]*len(ids)+[0]*pads))


class GenerativeDataset(Dataset):
    def __init__(self, task, tokenizer, max_length=256):
        if not task.has_training_docs():
            raise ValueError('Selected task/language has no training split; never substitute validation/test')
        docs = task.training_docs()
        self.source_count, self.skipped = len(docs), 0
        self.data_fingerprint = docs._fingerprint
        self.items = []
        for doc in docs:
            prompt, completion = format_example(task, doc)
            item = encode_example(prompt, completion, tokenizer, max_length)
            if item is None:
                self.skipped += 1
            else:
                self.items.append(item)
        if not self.items:
            raise ValueError('No scored training examples remain after tokenization/truncation')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
