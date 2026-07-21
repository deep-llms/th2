"""Core benchmark evaluation using lm-evaluation-harness."""

import os

import lm_eval
from lm_eval.models.huggingface import HFLM


TASK_CONFIGS = {
    "xnli": [
        "xnli_en", "xnli_vi", "xnli_zh", "xnli_ru", "xnli_de", "xnli_ar",
    ],
    "belebele": [
        "belebele_eng_Latn", "belebele_vie_Latn", "belebele_zho_Hans",
        "belebele_rus_Cyrl", "belebele_deu_Latn", "belebele_arb_Arab",
    ],
    "xcopa": [
        "xcopa_vi", "xcopa_zh",
    ],
    "xstorycloze": [
        "xstorycloze_en", "xstorycloze_ar", "xstorycloze_ru", "xstorycloze_zh",
    ],
    "paws-x": [
        "paws_en", "paws_de", "paws_zh",
    ],
    "hellaswag": [
        "hellaswag", "hellaswag_ar", "hellaswag_de", "hellaswag_ru", "hellaswag_vi",
    ],
}


def patch_lm_eval_dataset_paths():
    tasks_dir = os.path.join(os.path.dirname(lm_eval.__file__), "tasks")
    patches = [
        (os.path.join(tasks_dir, "xnli", "xnli_common_yaml"),
         "dataset_path: xnli\n", "dataset_path: facebook/xnli\n"),
        (os.path.join(tasks_dir, "xcopa", "default_et.yaml"),
         "dataset_path: xcopa\n", "dataset_path: cambridgeltl/xcopa\n"),
        (os.path.join(tasks_dir, "paws-x", "pawsx_template_yaml"),
         "dataset_path: paws-x\n", "dataset_path: google-research-datasets/paws-x\n"),
    ]
    for filepath, old, new in patches:
        if not os.path.isfile(filepath):
            continue
        with open(filepath) as f:
            content = f.read()
        if old in content:
            content = content.replace(old, new)
            with open(filepath, "w") as f:
                f.write(content)
            print(f"  Patched {filepath}")


def eval_benchmarks(model, tokenizer, task_groups=None, num_fewshot=0, batch_size=16, device="cuda"):
    patch_lm_eval_dataset_paths()

    if task_groups is None:
        task_groups = list(TASK_CONFIGS.keys())

    task_list = []
    for group in task_groups:
        task_list.extend(TASK_CONFIGS[group])

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
    )

    return results


def print_benchmark_results(results):
    from lm_eval.utils import make_table
    print(make_table(results))
