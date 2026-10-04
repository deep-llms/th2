"""Eight-process BF16 CPU check of train.py with microbatch 16, accumulation 4."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import load_file
from transformers import TrainingArguments
from test_train import fixture, fixture_target_counts
import train
from deep_kv.model import DeepKV
from deep_kv import ARMS, BOTTLENECK_ARMS, code_loss_weight


def check_masked_distributed_update(root, arm):
    """Eight-rank GAS update against a single global-batch SGD reference.

    Different padding/segment/label masks give each loss its own denominator,
    and each rank a different number of eligible positions.
    """
    from datasets import Dataset
    from transformers import default_data_collator
    from deep_kv.training import DeepKVTrainer
    from deep_kv.model import Context
    from test_deep_kv import model
    ids = torch.arange(16 * 6).reshape(16, 6) % 31
    valid = torch.arange(6)[None] < (2 + torch.arange(16) % 5)[:, None]
    segments = (torch.arange(6)[None] // 3).expand_as(ids)
    positions = (torch.arange(6)[None] % 3).expand_as(ids)
    labels = ids.clone()
    labels[::3, 2] = -100
    batch = Context(ids, valid, positions, segments, labels)
    reference = model(arm)
    with torch.no_grad():
        reference.aux.out.weight.fill_(.03)
    out = reference(batch)
    loss = (out['lm_sum'] / out['lm_count'] + out['extractor_sum'] / out['extractor_count']
            + code_loss_weight(arm) * out['align_sum'] / out['align_count'])
    loss.backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    data = Dataset.from_dict({'input_ids': ids.tolist(), 'attention_mask': valid.tolist(),
                              'segments': segments.tolist(), 'position_ids': positions.tolist(),
                              'labels': labels.tolist()})
    actual = model(arm)
    with torch.no_grad():
        actual.aux.out.weight.fill_(.03)
    arguments = TrainingArguments(output_dir=str(root / arm / 'masked-ddp'), use_cpu=True,
        report_to='none', max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        optim='sgd', learning_rate=.01, max_grad_norm=0, remove_unused_columns=False,
        ddp_find_unused_parameters=False, save_strategy='no', logging_strategy='no', disable_tqdm=True)
    trainer = DeepKVTrainer(model=actual, args=arguments, train_dataset=data, data_collator=default_data_collator)
    trainer.train()
    maximum = 0.0
    for name, value in actual.state_dict().items():
        expected = reference.state_dict()[name]
        torch.testing.assert_close(value, expected, atol=2e-7, rtol=2e-6, msg=name)
        maximum = max(maximum, (value - expected).abs().max().item())
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list("ABCD"))
    args = parser.parse_args()
    root = args.output.resolve()
    torch.set_num_threads(1)
    state = TrainingArguments(output_dir=str(root), use_cpu=True, report_to="none").distributed_state
    if state.num_processes != 8:
        raise ValueError("Run this check with eight CPU processes")
    if state.is_main_process:
        root.mkdir(parents=True, exist_ok=False)
        cfg = fixture(root, world=8, bf16=True, microbatch=16, accumulation=4)
        (root / "fixture.json").write_text(json.dumps(cfg))
    state.wait_for_everyone()
    cfg = json.loads((root / "fixture.json").read_text())
    target_count, consumer_count = fixture_target_counts(root, cfg)
    results = {}
    original_factory = DeepKV.from_scratch
    for arm in args.arms:
        masked_delta = check_masked_distributed_update(root, arm) if arm in BOTTLENECK_ARMS else None
        dtypes = set()
        def check_dtype(module, inputs, output):
            dtypes.add(str(output.dtype))
            assert output.dtype == torch.bfloat16
        def make_model(*args, **kwargs):
            model = original_factory(*args, **kwargs)
            model.backbone.model.layers[0].self_attn.q_proj.register_forward_hook(check_dtype)
            return model
        for mode, stop in (("full", None), ("resumed", 2), ("resumed", None)):
            config = {**cfg, "arm": arm, "output_dir": str(root / arm / mode)}
            if stop is not None:
                config["stop_after"] = stop
            path = root / "invocation.json"
            if state.is_main_process:
                path.write_text(json.dumps(config))
            state.wait_for_everyone()
            with patch("sys.argv", ["train.py", str(path)]), patch.object(DeepKV, "from_scratch", side_effect=make_model):
                train.main()
            state.wait_for_everyone()
        if state.is_main_process:
            full, resumed = (load_file(root / arm / p / "model.safetensors") for p in ("full", "resumed"))
            delta = max((full[k] - resumed[k]).abs().max().item() for k in full)
            for key in full:
                torch.testing.assert_close(full[key], resumed[key], atol=2e-6, rtol=2e-5)
            result = json.loads((root / arm / "resumed/result.json").read_text())
            assert result["global_step"] == 3 and result["evaluation"]["eval_rows"] == 5
            assert result["input_tokens"] == 3 * 8 * 16 * 4 * 8
            if arm in ("F", "G"):
                evaluation = result["evaluation"]
                assert evaluation["eval_route_queries"] == target_count
                expected = (evaluation["eval_lm_loss"] + .3 * evaluation["eval_loss_route"]
                            + (.3 * evaluation["eval_loss_msg"] if arm == "G" else 0))
                assert abs(evaluation["eval_loss"] - expected) < 1e-10
                assert arm != "F" or evaluation["eval_loss_msg"] == 0
                assert "eval_loss_k" not in evaluation
            if arm in BOTTLENECK_ARMS:
                evaluation = result['evaluation']
                assert evaluation['eval_loss'] == evaluation['eval_lm_loss']
                assert evaluation['eval_extractor_targets'] == (consumer_count if arm.startswith('Consumer') else target_count)
                expected = (evaluation['eval_lm_loss'] + evaluation[
                    'eval_loss_use' if arm.startswith('Consumer') else 'eval_loss_extract']
                    + code_loss_weight(arm) * evaluation['eval_loss_align'])
                assert abs(evaluation['eval_objective'] - expected) < 1e-10
            assert dtypes
            results[arm] = {"maximum_parameter_difference": delta, "updates": 3, "eval_rows": 5,
                            "projection_dtypes": sorted(dtypes), "masked_ddp_maximum_difference": masked_delta}
        state.wait_for_everyone()
    if state.is_main_process:
        (root / "resume_verified.json").write_text(json.dumps({"status": "ok", "world_size": 8,
            "cpu_only": True, "bf16": True, "microbatch": 16, "accumulation": 4, "arms": results}, indent=2))


if __name__ == "__main__":
    main()
