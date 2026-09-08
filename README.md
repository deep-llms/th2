# Stagewise widening transformer

English from-scratch capacity-allocation experiments using **Qwen3 with six
layers**, plus independent vocabulary interfaces and stagewise-width arms.

Start with [the experiment guide](docs/CAPACITY_EXPERIMENTS.md).
The old sparse-embedding sampled text datasets are reused; train.py follows
its Hugging Face Trainer and cached multiprocess preprocessing workflow.
No GPT-2 resampling or custom binary-token dataset is required.

- Architecture: capacity_allocation/modeling.py
- Training: train.py
- Data loading/packing: capacity_allocation/data.py
- Held-out PPL: evaluate_capacity.py; per-language PPL + benchmarks: eval/eval_parallel.py
- Task fine-tuning: finetune/run_all.py; English defaults and multilingual selection: docs/EVALUATION.md
- CPU tests: python -m unittest discover -s tests -v

Historical research drafts are retained for context; their GPT-2/Llama setup
is superseded by the current experiment guide. commands.sh defaults to #0:
publishing source does not authorize training or remote GPU actions.
