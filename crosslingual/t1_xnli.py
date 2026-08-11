"""T1 — zero-shot cross-lingual transfer probe (XNLI).

Freeze the model; mean-pool hidden states at chosen layers; build NLI pair
features [u; v; |u-v|; u*v]; train logistic regression on ENGLISH XNLI only;
evaluate unchanged on all 6 languages. The literal "train on one language,
test on another" transfer test.

Ported from cross_lingual_embeddings_hub diagnostics/test_t1_transfer_probe.py
(built there but never run). Caveat from its spec: at the embedding layer
mean-pooling is bag-of-words and XNLI sits at chance for every model — read
this test at mid/late layers only.
"""

import numpy as np
import torch

XNLI_LANGS = ["en", "vi", "zh", "ru", "de", "ar"]


@torch.no_grad()
def mean_pool_at_layer(model, tokenizer, texts, layer_idx, device,
                       batch_size=32, max_length=128):
    all_embs = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[start:start + batch_size],
                           return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hs = (outputs.hidden_states[0] if layer_idx is None
              else outputs.hidden_states[layer_idx + 1])
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embs.append(pooled.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def build_pair_features(premise_embs, hypothesis_embs):
    diff = np.abs(premise_embs - hypothesis_embs)
    prod = premise_embs * hypothesis_embs
    return np.concatenate([premise_embs, hypothesis_embs, diff, prod], axis=1)


def load_xnli(lang, split, max_examples=None):
    from datasets import load_dataset
    split_name = {"train": "train", "dev": "validation"}.get(split, "test")
    ds = load_dataset("facebook/xnli", lang, split=split_name)
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    examples = []
    for row in ds:
        label = row["label"]
        if isinstance(label, str):
            label = label_map.get(label, label)
        examples.append((row["premise"], row["hypothesis"], label))
        if max_examples and len(examples) >= max_examples:
            break
    return examples


def encode_examples(model, tokenizer, examples, layer_idx, device,
                    batch_size=32, max_length=128):
    premises = [ex[0] for ex in examples]
    hypotheses = [ex[1] for ex in examples]
    labels = np.array([ex[2] for ex in examples])
    p = mean_pool_at_layer(model, tokenizer, premises, layer_idx, device,
                           batch_size, max_length)
    h = mean_pool_at_layer(model, tokenizer, hypotheses, layer_idx, device,
                           batch_size, max_length)
    return build_pair_features(p, h), labels


def run_t1(model, tokenizer, device, layers=(14, 27), seeds=(42, 43, 44),
           max_train=20000, batch_size=32, max_length=128):
    """Returns {layer_name: {mean: {...}, std: {...}, per_seed: [...]}}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    train_examples = load_xnli("en", "train", max_examples=max_train)
    dev_examples = load_xnli("en", "dev")
    test_data = {lang: load_xnli(lang, "test") for lang in XNLI_LANGS}

    results = {}
    for layer_idx in layers:
        layer_name = ("embedding" if layer_idx is None
                      else f"layer_{layer_idx}")
        X_train, y_train = encode_examples(
            model, tokenizer, train_examples, layer_idx, device,
            batch_size, max_length)
        X_dev, y_dev = encode_examples(
            model, tokenizer, dev_examples, layer_idx, device,
            batch_size, max_length)
        X_tests, y_tests = {}, {}
        for lang in XNLI_LANGS:
            X_tests[lang], y_tests[lang] = encode_examples(
                model, tokenizer, test_data[lang], layer_idx, device,
                batch_size, max_length)

        seed_results = []
        for seed in seeds:
            clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            clf.fit(X_train, y_train)
            per_lang = {lang: float(accuracy_score(
                            y_tests[lang], clf.predict(X_tests[lang])))
                        for lang in XNLI_LANGS}
            seed_results.append({
                "en_train_acc": float(accuracy_score(
                    y_train, clf.predict(X_train))),
                "en_dev_acc": float(accuracy_score(
                    y_dev, clf.predict(X_dev))),
                "per_lang": per_lang,
            })

        accs = {lang: [r["per_lang"][lang] for r in seed_results]
                for lang in XNLI_LANGS}
        results[layer_name] = {
            "mean": {
                "en_dev_acc": float(np.mean(
                    [r["en_dev_acc"] for r in seed_results])),
                "per_lang": {l: float(np.mean(a)) for l, a in accs.items()},
            },
            "std": {
                "per_lang": {l: float(np.std(a)) for l, a in accs.items()},
            },
            "per_seed": seed_results,
        }
    return results
