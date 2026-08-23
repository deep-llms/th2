import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pyarrow as pa
from datasets import load_from_disk

import prepare_data


def _make_local_fixture(tmp_path, lang="vi", count=5):
    raw_dir = tmp_path / "raw"
    lang_dir = raw_dir / lang
    lang_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.tsv"

    manifest_lines = []
    for index in range(count):
        content = f"parquet-placeholder-{index}".encode()
        filename = f"{lang}_part_{index:05d}.parquet"
        (lang_dir / filename).write_bytes(content)
        manifest_lines.append(
            f"{hashlib.sha256(content).hexdigest()}\t{len(content)}\t{lang}/{filename}"
        )

    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    args = SimpleNamespace(
        raw_dir=str(raw_dir),
        manifest=str(manifest_path),
        langs=[lang],
    )
    return args, lang_dir, manifest_path


def test_build_local_plan_accepts_exact_manifested_files(tmp_path):
    args, _, _ = _make_local_fixture(tmp_path)

    plan = prepare_data.build_local_plan(args)

    assert list(plan) == ["vi"]
    assert len(plan["vi"]) == 5
    assert [path.rsplit("/", 1)[-1] for path in plan["vi"]] == sorted(
        path.rsplit("/", 1)[-1] for path in plan["vi"]
    )


def test_build_local_plan_rejects_missing_extra_or_wrong_size(tmp_path):
    args, lang_dir, _ = _make_local_fixture(tmp_path)
    missing_path = lang_dir / "vi_part_00004.parquet"
    missing_path.unlink()
    with pytest.raises(RuntimeError, match="missing="):
        prepare_data.build_local_plan(args)

    missing_path.write_bytes(b"parquet-placeholder-4")
    extra_path = lang_dir / "vi_part_99999.parquet"
    extra_path.write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="extra="):
        prepare_data.build_local_plan(args)

    extra_path.unlink()
    (lang_dir / "vi_part_00000.parquet").write_bytes(b"wrong-size")
    with pytest.raises(RuntimeError, match="Size mismatch"):
        prepare_data.build_local_plan(args)


def test_sample_only_dry_run_never_constructs_hf_filesystem(tmp_path, monkeypatch):
    args, _, manifest_path = _make_local_fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data.py",
            "sample",
            "--dry-run",
            "--langs",
            "vi",
            "--raw-dir",
            args.raw_dir,
            "--manifest",
            str(manifest_path),
        ],
    )

    with patch.object(
        prepare_data,
        "HfFileSystem",
        side_effect=AssertionError("sample-only mode contacted Hugging Face"),
    ):
        prepare_data.main()


def test_sample_loads_local_tokenizer_path_without_changing_output_identity(tmp_path):
    args = SimpleNamespace(
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_path="/models/Qwen3-0.6B",
        local_files_only=True,
        data_dir=str(tmp_path),
        raw_dir=str(tmp_path / "raw"),
        flush_every=1,
        tokenize_batch_size=2,
    )

    with patch.object(prepare_data.AutoTokenizer, "from_pretrained") as load_tokenizer:
        prepare_data.sample_by_token_count(args, {})

    load_tokenizer.assert_called_once_with(
        "/models/Qwen3-0.6B",
        local_files_only=True,
    )


def test_sample_only_writes_disjoint_train_and_eval_outputs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    lang_dir = raw_dir / "vi"
    lang_dir.mkdir(parents=True)
    parquet_path = lang_dir / "vi_part_00000.parquet"
    texts = [f"document-{index}" for index in range(6)]
    prepare_data.pq.write_table(pa.table({"text": texts}), parquet_path)

    monkeypatch.setitem(
        prepare_data.LANG_CONFIG,
        "vi",
        {"target_tokens": 3, "eval_tokens": 2, "num_files": 1},
    )

    batch_lengths = []

    class OneTokenPerDocument:
        def __call__(self, batch, add_special_tokens=False):
            assert add_special_tokens is False
            batch_lengths.append(len(batch))
            return {"input_ids": [[1] for _ in batch]}

    args = SimpleNamespace(
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_path="/models/Qwen3-0.6B",
        local_files_only=True,
        data_dir=str(tmp_path / "processed"),
        raw_dir=str(raw_dir),
        flush_every=1,
        tokenize_batch_size=2,
    )

    with patch.object(
        prepare_data.AutoTokenizer,
        "from_pretrained",
        return_value=OneTokenPerDocument(),
    ):
        prepare_data.sample_by_token_count(args, {"vi": [str(parquet_path)]})

    output_root = tmp_path / "processed" / "Qwen_Qwen3-0.6B"
    train = load_from_disk(output_root / "train" / "vi" / "shard_0000")
    evaluation = load_from_disk(output_root / "eval" / "vi")
    train_texts = set(train["text"])
    eval_texts = set(evaluation["text"])

    assert len(train_texts) == 3
    assert len(eval_texts) == 2
    assert train_texts.isdisjoint(eval_texts)
    assert batch_lengths == [2, 2, 2]


def test_sample_rejects_partial_language_output(tmp_path):
    raw_dir = tmp_path / "raw"
    lang_dir = raw_dir / "vi"
    lang_dir.mkdir(parents=True)
    parquet_path = lang_dir / "vi_part_00000.parquet"
    prepare_data.pq.write_table(pa.table({"text": ["document"]}), parquet_path)

    data_dir = tmp_path / "processed"
    (data_dir / "Qwen_Qwen3-0.6B" / "train" / "vi").mkdir(parents=True)
    args = SimpleNamespace(
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_path="/models/Qwen3-0.6B",
        local_files_only=True,
        data_dir=str(data_dir),
        raw_dir=str(raw_dir),
        flush_every=1,
        tokenize_batch_size=2,
    )

    with patch.object(prepare_data.AutoTokenizer, "from_pretrained"):
        with pytest.raises(RuntimeError, match="Partial output detected"):
            prepare_data.sample_by_token_count(args, {"vi": [str(parquet_path)]})
