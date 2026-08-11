"""T7 — generative code-switch continuation (behavioral, logits-level).

Templated translation prompts in both directions; metric = mean
log P(correct translation token) - log P(random same-language token).

Ported from cross_lingual_embeddings_hub
diagnostics/test_t7_generative_codeswitch.py. Caveat from the source
project: margins were all negative and barely separated arms at the 0.6B /
10K-step scale — a weak discriminator; keep expectations low.
"""

import torch
import torch.nn.functional as F

from crosslingual.utils import NON_EN_LANGS

LANG_NAMES = {"vi": "Vietnamese", "zh": "Chinese", "ru": "Russian",
              "de": "German", "ar": "Arabic"}

TEMPLATES_EN_TO_X = [
    '"{word}" in {lang_name} is',
    'the {lang_name} word for "{word}" is',
    '"{word}" translates to {lang_name} as',
]
TEMPLATES_X_TO_EN = [
    '"{word}" in English is',
    '"{word}" =',
    '"{word}" means',
]


def build_prompts(en_word, tgt_word, lang, tokenizer):
    lang_name = LANG_NAMES.get(lang, lang)
    prompts = []
    tgt_ids = tokenizer(tgt_word, add_special_tokens=False)["input_ids"]
    if len(tgt_ids) == 1:
        for template in TEMPLATES_EN_TO_X:
            prompts.append((template.format(word=en_word, lang_name=lang_name),
                            tgt_ids[0], "en_to_x"))
    en_ids = tokenizer(en_word, add_special_tokens=False)["input_ids"]
    if len(en_ids) == 1:
        for template in TEMPLATES_X_TO_EN:
            prompts.append((template.format(word=tgt_word),
                            en_ids[0], "x_to_en"))
    return prompts


@torch.no_grad()
def get_next_token_logprob(model, tokenizer, prompt, target_token_id, device):
    inputs = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=False).to(device)
    logits = model(**inputs).logits
    log_probs = F.log_softmax(logits[0, -1].float(), dim=-1)
    return log_probs[target_token_id].item()


def find_random_control_token(exclude_word, answer_lang, translations,
                              tokenizer):
    for other_en, other_trans in translations:
        if answer_lang == "en":
            if other_en == exclude_word:
                continue
            ids = tokenizer(other_en, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                return ids[0]
        else:
            if answer_lang not in other_trans:
                continue
            other_word = other_trans[answer_lang]
            if other_word == exclude_word:
                continue
            if other_word.lower() == other_en.lower():
                continue
            ids = tokenizer(other_word, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                return ids[0]
    return None


def run_t7(model, tokenizer, translations, device, max_pairs=500):
    results = {}
    for lang in NON_EN_LANGS:
        margins, m_en_to_x, m_x_to_en = [], [], []
        n_tested = 0
        for en_word, trans in translations:
            if lang not in trans:
                continue
            tgt_word = trans[lang]
            if en_word.lower() == tgt_word.lower():
                continue
            prompts = build_prompts(en_word, tgt_word, lang, tokenizer)
            if not prompts:
                continue
            rand_en_to_x = find_random_control_token(
                tgt_word, lang, translations, tokenizer)
            rand_x_to_en = find_random_control_token(
                en_word, "en", translations, tokenizer)

            for prompt_text, target_id, direction in prompts:
                random_id = (rand_en_to_x if direction == "en_to_x"
                             else rand_x_to_en)
                if random_id is None:
                    continue
                margin = (get_next_token_logprob(
                              model, tokenizer, prompt_text, target_id, device)
                          - get_next_token_logprob(
                              model, tokenizer, prompt_text, random_id, device))
                margins.append(margin)
                (m_en_to_x if direction == "en_to_x" else m_x_to_en).append(
                    margin)
                n_tested += 1
            if n_tested >= max_pairs:
                break

        if margins:
            mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
            results[f"en-{lang}"] = {
                "mean_margin": mean(margins),
                "mean_margin_en_to_x": mean(m_en_to_x),
                "mean_margin_x_to_en": mean(m_x_to_en),
                "n_prompts": len(margins),
            }
    return results
