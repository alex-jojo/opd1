"""Save pooled actor/reference hidden states for rubric probe training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.gpt_rollout_scorer import RUBRIC_NAMES


FORMAT_VERSION = 1
HIDDEN_KEYS = (
    "rubric_probe_student_last",
    "rubric_probe_student_mean",
    "rubric_probe_teacher_last",
    "rubric_probe_teacher_mean",
)
ALLOWED_RUBRIC_SCORES = tuple(1.0 + 0.5 * index for index in range(7))


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    return value


def _row_value(batch, key: str, row_idx: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return default
    try:
        return _json_safe(values[row_idx])
    except (IndexError, KeyError, TypeError):
        return default


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def extract_rubric_score_vector(rubric_scores: Any) -> list[float] | None:
    """Extract scores in the one canonical seven-rubric order."""

    if not isinstance(rubric_scores, dict):
        return None
    vector = []
    for rubric_name in RUBRIC_NAMES:
        rubric = rubric_scores.get(rubric_name)
        value = rubric.get("score") if isinstance(rubric, dict) else rubric
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(score) or not any(abs(score - allowed) < 1.0e-6 for allowed in ALLOWED_RUBRIC_SCORES):
            return None
        vector.append(score)
    return vector


def rubric_scores_to_labels(rubric_scores: torch.Tensor) -> torch.Tensor:
    """Map 1.0, 1.5, ..., 4.0 scores to class ids 0, 1, ..., 6."""

    return ((rubric_scores - 1.0) * 2).long()


def _problem_and_ground_truth(batch, tokenizer, row_idx: int) -> tuple[str, str, str]:
    extra_info = _mapping(_row_value(batch, "extra_info", row_idx, {}))
    reward_model = _mapping(_row_value(batch, "reward_model", row_idx, {}))

    prompt_ids = batch.batch["prompts"][row_idx]
    attention_mask = batch.batch.get("attention_mask")
    if attention_mask is not None:
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = int(attention_mask[row_idx, :prompt_length].sum().item())
        prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids[:0]
    prompt = tokenizer.decode(prompt_ids.detach().cpu().tolist(), skip_special_tokens=True)
    problem = _text(extra_info.get("problem") or extra_info.get("question") or prompt)

    response_ids = batch.batch["responses"][row_idx]
    response_mask = batch.batch["response_mask"][row_idx].to(dtype=torch.bool)
    valid_response_ids = response_ids[response_mask]
    response = tokenizer.decode(valid_response_ids.detach().cpu().tolist(), skip_special_tokens=True)

    ground_truth = _text(reward_model.get("ground_truth") or extra_info.get("answer") or "")
    return problem, response, ground_truth


def _problem_id(batch, row_idx: int, problem: str) -> str:
    explicit = _row_value(batch, "problem_id", row_idx)
    extra_info = _mapping(_row_value(batch, "extra_info", row_idx, {}))
    explicit = explicit or extra_info.get("problem_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    normalized = " ".join(problem.split())
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _sample_id(batch, row_idx: int, global_step: int) -> str:
    extra_info = _mapping(_row_value(batch, "extra_info", row_idx, {}))
    source_id = (
        _row_value(batch, "sample_id", row_idx)
        or extra_info.get("sample_id")
        or _row_value(batch, "request_id", row_idx)
        or _row_value(batch, "uid", row_idx)
        or extra_info.get("index")
        or row_idx
    )
    # row_idx keeps IDs unique when rollout.n > 1 repeats the same prompt UID.
    return f"step-{global_step}:row-{row_idx}:{source_id}"


def _validate_hidden_tensors(batch, config: Any) -> None:
    missing = [key for key in HIDDEN_KEYS if key not in batch.batch]
    if missing:
        raise RuntimeError(
            "Rubric-probe collection is enabled but pooled hidden states are missing: "
            f"{missing}. This collector currently requires the FSDP actor/ref path."
        )

    batch_size = len(batch)
    expected_student = int(_get(config, "expected_student_hidden_size", 2048))
    expected_teacher = int(_get(config, "expected_teacher_hidden_size", 2560))
    expected_sizes = {
        "rubric_probe_student_last": expected_student,
        "rubric_probe_student_mean": expected_student,
        "rubric_probe_teacher_last": expected_teacher,
        "rubric_probe_teacher_mean": expected_teacher,
    }
    for key, expected_size in expected_sizes.items():
        tensor = batch.batch[key]
        if tensor.ndim != 2 or tensor.shape[0] != batch_size:
            raise RuntimeError(f"{key} must have shape [batch, hidden], got {tuple(tensor.shape)}")
        if expected_size > 0 and tensor.shape[1] != expected_size:
            raise RuntimeError(
                f"{key} hidden size is {tensor.shape[1]}, expected {expected_size}. "
                "Check the configured student/teacher checkpoints or override the expected hidden size."
            )


def _eligible_rows(batch) -> tuple[list[int], list[list[float]], dict[str, int]]:
    rows: list[int] = []
    score_vectors: list[list[float]] = []
    stats = {"total": len(batch), "non_orig": 0, "padding": 0, "empty_response": 0, "invalid_rubric": 0}
    response_lengths = batch.batch["response_mask"].sum(dim=-1).detach().cpu().tolist()

    for row_idx in range(len(batch)):
        if bool(_row_value(batch, "g_opd_padding_sample", row_idx, False)):
            stats["padding"] += 1
            continue
        if str(_row_value(batch, "g_opd_sample_kind", row_idx, "orig") or "orig") != "orig":
            stats["non_orig"] += 1
            continue
        if int(response_lengths[row_idx]) <= 0:
            stats["empty_response"] += 1
            continue
        scores = extract_rubric_score_vector(_row_value(batch, "gpt_rollout_rubric_scores", row_idx))
        if scores is None:
            stats["invalid_rubric"] += 1
            continue
        rows.append(row_idx)
        score_vectors.append(scores)

    stats["saved"] = len(rows)
    return rows, score_vectors, stats


def _metadata_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("tensor_row", pa.int64()),
            ("sample_id", pa.string()),
            ("problem_id", pa.string()),
            ("global_step", pa.int64()),
            ("student_checkpoint", pa.string()),
            ("teacher_checkpoint", pa.string()),
            ("response_length", pa.int64()),
            ("problem", pa.string()),
            ("student_response", pa.string()),
            ("ground_truth", pa.string()),
            ("rubric_scores", pa.list_(pa.float32(), 7)),
            ("rubric_labels", pa.list_(pa.int64(), 7)),
            ("gpt_score_100", pa.float32()),
            ("problem_domain", pa.string()),
            ("difficulty_3", pa.string()),
            ("old_log_prob_mean", pa.float32()),
            ("ref_log_prob_mean", pa.float32()),
            ("teacher_score", pa.float32()),
            ("data_source", pa.string()),
            ("ended_with_eos", pa.bool_()),
            ("gpt_model", pa.string()),
            ("rubric_prompt_version", pa.string()),
        ]
    )


def _schema_document(config: Any) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "rubric_order": list(RUBRIC_NAMES),
        "raw_score_values": list(ALLOWED_RUBRIC_SCORES),
        "label_formula": "((rubric_scores - 1.0) * 2).long()",
        "teacher_input": "The teacher encodes the same student response; it does not generate a separate answer.",
        "pooling": {
            "last": "final model layer at the last valid response token",
            "mean": "masked mean of the final model layer over valid response tokens only",
            "prompt_included": False,
        },
        "tensors": {
            "student_last": [None, int(_get(config, "expected_student_hidden_size", 2048))],
            "student_mean": [None, int(_get(config, "expected_student_hidden_size", 2048))],
            "teacher_last": [None, int(_get(config, "expected_teacher_hidden_size", 2560))],
            "teacher_mean": [None, int(_get(config, "expected_teacher_hidden_size", 2560))],
            "rubric_scores": [None, 7],
            "rubric_labels": [None, 7],
        },
    }


def _ensure_schema(output_dir: Path, config: Any) -> None:
    schema_path = output_dir / "schema.json"
    document = _schema_document(config)
    if schema_path.exists():
        with schema_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != document:
            raise RuntimeError(f"Existing rubric-probe schema does not match this run: {schema_path}")
        return

    temporary = output_dir / f".schema-{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, schema_path)


def save_rubric_probe_batch(
    *,
    batch,
    tokenizer,
    config: Any,
    global_step: int,
    student_checkpoint: str,
    teacher_checkpoint: str,
) -> dict[str, int]:
    """Save one training-step shard and return collection counters."""

    output_value = _get(config, "output_dir", None)
    if output_value is None or str(output_value).strip().lower() in {"", "none", "null"}:
        raise ValueError("trainer.rubric_probe_data.output_dir is required when collection is enabled")

    _validate_hidden_tensors(batch, config)
    selected_rows, score_vectors, stats = _eligible_rows(batch)
    if not selected_rows:
        return stats

    output_dir = Path(str(output_value)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_schema(output_dir, config)

    selected_tensor_rows = torch.tensor(selected_rows, dtype=torch.long)
    hidden_dtype_name = str(_get(config, "hidden_dtype", "float16")).lower()
    hidden_dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if hidden_dtype_name not in hidden_dtypes:
        raise ValueError("rubric_probe_data.hidden_dtype must be float16/fp16 or bfloat16/bf16")
    hidden_dtype = hidden_dtypes[hidden_dtype_name]

    rubric_scores = torch.tensor(score_vectors, dtype=torch.float32)
    rubric_labels = rubric_scores_to_labels(rubric_scores)
    tensor_payload = {
        "student_last": batch.batch["rubric_probe_student_last"]
        .detach()
        .cpu()
        .index_select(0, selected_tensor_rows)
        .to(hidden_dtype)
        .contiguous(),
        "student_mean": batch.batch["rubric_probe_student_mean"]
        .detach()
        .cpu()
        .index_select(0, selected_tensor_rows)
        .to(hidden_dtype)
        .contiguous(),
        "teacher_last": batch.batch["rubric_probe_teacher_last"]
        .detach()
        .cpu()
        .index_select(0, selected_tensor_rows)
        .to(hidden_dtype)
        .contiguous(),
        "teacher_mean": batch.batch["rubric_probe_teacher_mean"]
        .detach()
        .cpu()
        .index_select(0, selected_tensor_rows)
        .to(hidden_dtype)
        .contiguous(),
        "rubric_scores": rubric_scores.contiguous(),
        "rubric_labels": rubric_labels.contiguous(),
    }

    response_mask = batch.batch["response_mask"].detach().cpu().to(dtype=torch.float32)
    token_counts = response_mask.sum(dim=-1).clamp_min(1.0)
    old_means = (
        (batch.batch["old_log_probs"].detach().cpu().to(torch.float32) * response_mask).sum(dim=-1)
        / token_counts
    )
    ref_means = (
        (batch.batch["ref_log_prob"].detach().cpu().to(torch.float32) * response_mask).sum(dim=-1)
        / token_counts
    )
    responses = batch.batch["responses"].detach().cpu()
    response_mask_bool = response_mask.to(dtype=torch.bool)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    save_text = bool(_get(config, "save_text", True))
    prompt_version = str(_get(config, "rubric_prompt_version", "math_7rubric_v1"))
    default_gpt_model = str(_get(config, "gpt_model", ""))

    metadata_rows = []
    for tensor_row, (row_idx, scores, labels) in enumerate(
        zip(selected_rows, rubric_scores.tolist(), rubric_labels.tolist(), strict=True)
    ):
        problem, student_response, ground_truth = _problem_and_ground_truth(batch, tokenizer, row_idx)
        response_length = int(response_mask_bool[row_idx].sum().item())
        valid_positions = torch.nonzero(response_mask_bool[row_idx], as_tuple=False).squeeze(-1)
        final_token_id = int(responses[row_idx, int(valid_positions[-1])].item())
        old_mean = float(old_means[row_idx].item())
        ref_mean = float(ref_means[row_idx].item())
        gpt_score_100 = _row_value(batch, "gpt_rollout_score_100", row_idx)
        metadata_rows.append(
            {
                "tensor_row": tensor_row,
                "sample_id": _sample_id(batch, row_idx, int(global_step)),
                "problem_id": _problem_id(batch, row_idx, problem),
                "global_step": int(global_step),
                "student_checkpoint": str(student_checkpoint),
                "teacher_checkpoint": str(teacher_checkpoint),
                "response_length": response_length,
                "problem": problem if save_text else "",
                "student_response": student_response if save_text else "",
                "ground_truth": ground_truth if save_text else "",
                "rubric_scores": scores,
                "rubric_labels": labels,
                "gpt_score_100": None if gpt_score_100 is None else float(gpt_score_100),
                "problem_domain": _text(_row_value(batch, "gpt_rollout_problem_domain", row_idx)) or None,
                "difficulty_3": _text(_row_value(batch, "gpt_rollout_difficulty_3", row_idx)) or None,
                "old_log_prob_mean": old_mean,
                "ref_log_prob_mean": ref_mean,
                "teacher_score": ref_mean - old_mean,
                "data_source": _text(_row_value(batch, "data_source", row_idx)) or None,
                "ended_with_eos": eos_token_id is not None and final_token_id == int(eos_token_id),
                "gpt_model": _text(_row_value(batch, "gpt_rollout_model", row_idx, default_gpt_model)) or None,
                "rubric_prompt_version": prompt_version,
            }
        )

    final_shard_dir = output_dir / f"step_{int(global_step):08d}"
    if final_shard_dir.exists():
        raise FileExistsError(
            f"Rubric-probe shard already exists for global_step={global_step}: {final_shard_dir}. "
            "Refusing to overwrite or silently duplicate samples."
        )
    temporary_dir = output_dir / f".{final_shard_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary_dir.mkdir()
    try:
        from safetensors.torch import save_file
        import pyarrow as pa
        import pyarrow.parquet as pq

        save_file(
            tensor_payload,
            str(temporary_dir / "tensors.safetensors"),
            metadata={
                "format_version": str(FORMAT_VERSION),
                "global_step": str(global_step),
                "rubric_order": json.dumps(list(RUBRIC_NAMES), ensure_ascii=False),
            },
        )
        table = pa.Table.from_pylist(metadata_rows, schema=_metadata_schema())
        pq.write_table(table, temporary_dir / "metadata.parquet", compression="zstd")
        manifest = {
            "format_version": FORMAT_VERSION,
            "global_step": int(global_step),
            "num_samples": len(selected_rows),
            "hidden_dtype": str(tensor_payload["student_last"].dtype).removeprefix("torch."),
            "tensor_file": "tensors.safetensors",
            "metadata_file": "metadata.parquet",
            "collection_stats": stats,
        }
        with (temporary_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        temporary_dir.rename(final_shard_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return stats


def pop_rubric_probe_hidden(batch) -> None:
    """Remove transient pooled hiddens before PPO advantage/update code."""

    keys = [key for key in HIDDEN_KEYS if key in batch.batch]
    if keys:
        batch.pop(batch_keys=keys)
