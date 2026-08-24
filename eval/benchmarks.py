"""Core benchmark evaluation using lm-evaluation-harness."""

import fcntl
import os
import re
import stat
import tempfile

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


_DATASET_REPOSITORIES = {
    "facebook/xnli": "facebook/xnli",
    "facebook/belebele": "facebook/belebele",
    "cambridgeltl/xcopa": "cambridgeltl/xcopa",
    "juletxara/xstory_cloze": "juletxara/xstory_cloze",
    "google-research-datasets/paws-x": "google-research-datasets/paws-x",
    "Rowan/hellaswag": "Rowan/hellaswag",
    "alexandrainst/m_hellaswag": "alexandrainst/m_hellaswag",
}

_DATASET_PATH_PATCHES = [
    ("xnli/xnli_common_yaml", "facebook/xnli", {"xnli"}),
    ("belebele/_default_template_yaml", "facebook/belebele", set()),
    ("xcopa/default_et.yaml", "cambridgeltl/xcopa", {"xcopa"}),
    ("xstorycloze/default_ar.yaml", "juletxara/xstory_cloze", set()),
    ("paws-x/pawsx_template_yaml", "google-research-datasets/paws-x", {"paws-x"}),
    ("hellaswag/hellaswag.yaml", "Rowan/hellaswag", set()),
    *[
        (
            f"okapi/hellaswag_multilingual/hellaswag_{lang}.yaml",
            "alexandrainst/m_hellaswag",
            set(),
        )
        for lang in ("ar", "de", "ru", "vi")
    ],
]


def _replace_dataset_path(filepath, target, accepted_values):
    with open(filepath, encoding="utf-8") as handle:
        content = handle.read()

    matches = re.findall(r"^dataset_path:\s*(\S+)\s*$", content, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one dataset_path in {filepath}, found {len(matches)}"
        )

    current = matches[0]
    if current == target:
        return False
    if current not in accepted_values and not os.path.isabs(current):
        raise RuntimeError(
            f"Refusing unexpected dataset_path in {filepath}: {current!r}"
        )

    replacement = re.sub(
        r"^dataset_path:\s*\S+\s*$",
        f"dataset_path: {target}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    file_mode = stat.S_IMODE(os.stat(filepath).st_mode)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(filepath)}.", dir=os.path.dirname(filepath)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(replacement)
        os.chmod(temporary_path, file_mode)
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return True


def patch_lm_eval_dataset_paths(dataset_root=None):
    """Point the configured lm-eval tasks at Hub IDs or local snapshots.

    Set ``LM_EVAL_DATASET_ROOT`` to the directory containing the preserved Hub
    namespace (for example, ``<root>/facebook/xnli``) for fully offline eval.
    The package task files are shared by parallel evaluator processes, so all
    changes are serialized and installed atomically.
    """
    if dataset_root is None:
        dataset_root = os.environ.get("LM_EVAL_DATASET_ROOT")
    if dataset_root:
        dataset_root = os.path.abspath(dataset_root)
        missing = [
            os.path.join(dataset_root, relative_path)
            for relative_path in _DATASET_REPOSITORIES.values()
            if not os.path.isdir(os.path.join(dataset_root, relative_path))
        ]
        if missing:
            raise FileNotFoundError(
                "Missing local lm-eval dataset snapshots: " + ", ".join(missing)
            )

    tasks_dir = os.path.join(os.path.dirname(lm_eval.__file__), "tasks")
    lock_path = os.path.join(tempfile.gettempdir(), "sparse_embedding_lm_eval_paths.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        for relative_file, repository, aliases in _DATASET_PATH_PATCHES:
            filepath = os.path.join(tasks_dir, relative_file)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Missing lm-eval task configuration: {filepath}")
            target = (
                os.path.join(dataset_root, _DATASET_REPOSITORIES[repository])
                if dataset_root
                else repository
            )
            accepted_values = {repository, *aliases}
            if _replace_dataset_path(filepath, target, accepted_values):
                print(f"  Patched {filepath} -> {target}")


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
