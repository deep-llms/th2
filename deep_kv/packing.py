"""Document boundaries for Qwen text packing (no model or GPU imports)."""

DOCUMENT_END_TOKEN = "<|endoftext|>"
QWEN_DOCUMENT_END_ID = 151643
PACKING_POLICY = "document_map_eod_v1"


def document_end_id(tokenizer):
    # tokenizer.eos_token_id can refer to the chat turn marker <|im_end|>.
    token_id = tokenizer.get_vocab().get(DOCUMENT_END_TOKEN)
    if token_id is None or tokenizer.encode(DOCUMENT_END_TOKEN, add_special_tokens=False) != [token_id]:
        raise ValueError("Packing requires a single <|endoftext|> document-end token")
    return token_id


def tokenize_documents(texts, tokenizer, end_id):
    """Append one explicit boundary after every document before chunking."""
    rows = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [list(row) + [end_id] for row in rows]


def preprocessing_policy(context_length, end_id=QWEN_DOCUMENT_END_ID):
    return {"policy": PACKING_POLICY, "map_batch_size": 1000, "num_proc": 1,
            "context_length": context_length, "add_special_tokens": False,
            "document_end_token": DOCUMENT_END_TOKEN, "document_end_token_id": end_id,
            "document_end_policy": "append_one_per_document",
            "remainder": "drop_per_document_map_batch"}


def require_current_preprocessing(metadata, context_length):
    if metadata.get("preprocessing") != preprocessing_policy(context_length):
        raise ValueError("Packed inputs use an old or incompatible document-end policy; "
                         "prepare fresh inputs from the saved text splits")


def tokenize_batch(examples, tokenizer, end_id):
    ids = tokenize_documents(examples["text"], tokenizer, end_id)
    return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def group_texts(examples, block_size):
    from itertools import chain
    concatenated = {key: list(chain(*rows)) for key, rows in examples.items()}
    total = len(concatenated["input_ids"]) // block_size * block_size
    result = {key: [tokens[i:i + block_size] for i in range(0, total, block_size)]
              for key, tokens in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_dataset(raw_dataset, tokenizer, block_size, training_args, *,
                       num_proc=None, overwrite_cache=False):
    """The train.py two-map CLM pipeline, cached by Hugging Face Datasets."""
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized = raw_dataset.map(
            tokenize_batch, fn_kwargs={"tokenizer": tokenizer, "end_id": document_end_id(tokenizer)},
            batched=True, num_proc=num_proc, remove_columns=raw_dataset.column_names,
            load_from_cache_file=not overwrite_cache, desc="Running tokenizer on dataset")
    with training_args.main_process_first(desc="grouping texts together"):
        packed = tokenized.map(
            group_texts, fn_kwargs={"block_size": block_size}, batched=True, num_proc=num_proc,
            load_from_cache_file=not overwrite_cache, desc=f"Grouping texts in chunks of {block_size}")
    return packed
