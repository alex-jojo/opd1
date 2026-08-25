from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from torch import nn

from verl.trainer.ppo.gpt_rollout_scorer import RUBRIC_NAMES
from verl.trainer.ppo.rubric_probe_data import (
    extract_rubric_score_vector,
    rubric_scores_to_labels,
    save_rubric_probe_batch,
)
from verl.workers.actor.hidden_state_pooling import (
    LastHiddenStateCapture,
    find_final_norm_module,
    pool_response_hidden,
)


class _FakeBatch(SimpleNamespace):
    def __len__(self):
        return self.batch["responses"].shape[0]


class _FakeTokenizer:
    eos_token_id = 2

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


def _rubrics(scores):
    return {name: {"score": score, "weight": 0.0, "reason": "test"} for name, score in zip(RUBRIC_NAMES, scores, strict=True)}


def test_pool_response_hidden_uses_only_valid_response_tokens():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[5.0, 6.0], [88.0, 88.0], [7.0, 8.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]])

    last, mean = pool_response_hidden(hidden, mask)

    torch.testing.assert_close(last, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))
    torch.testing.assert_close(mean, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))


def test_final_norm_hook_captures_only_final_layer_output():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.norm = nn.LayerNorm(3)

        def forward(self, values):
            return self.model.norm(values)

    model = TinyModel()
    name, norm = find_final_norm_module(model)
    values = torch.tensor([[[1.0, 2.0, 3.0]]])
    with LastHiddenStateCapture(norm) as capture:
        expected = model(values)

    assert name == "model.norm"
    torch.testing.assert_close(capture.take(), expected)


def test_rubric_order_and_class_mapping_are_fixed():
    scores = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    vector = extract_rubric_score_vector(_rubrics(scores))
    labels = rubric_scores_to_labels(torch.tensor([vector], dtype=torch.float32))

    assert vector == scores
    assert labels.tolist() == [[0, 1, 2, 3, 4, 5, 6]]


def test_save_rubric_probe_batch_writes_aligned_atomic_shard(tmp_path):
    valid_scores = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    batch = _FakeBatch(
        batch={
            "prompts": torch.tensor([[0, 10], [0, 11], [0, 12]]),
            "responses": torch.tensor([[1, 2, 0], [3, 2, 0], [0, 0, 0]]),
            "attention_mask": torch.tensor(
                [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 0]]
            ),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
            "old_log_probs": torch.tensor([[-2.0, -4.0, 0.0], [-1.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
            "ref_log_prob": torch.tensor([[-1.0, -5.0, 0.0], [-2.0, -2.0, 0.0], [0.0, 0.0, 0.0]]),
            "rubric_probe_student_last": torch.arange(9, dtype=torch.bfloat16).reshape(3, 3),
            "rubric_probe_student_mean": torch.arange(9, 18, dtype=torch.bfloat16).reshape(3, 3),
            "rubric_probe_teacher_last": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "rubric_probe_teacher_mean": torch.arange(12, 24, dtype=torch.bfloat16).reshape(3, 4),
        },
        non_tensor_batch={
            "g_opd_sample_kind": np.array(["orig", "reroll_hint", "padding"], dtype=object),
            "g_opd_padding_sample": np.array([False, False, True], dtype=object),
            "gpt_rollout_rubric_scores": np.array(
                [_rubrics(valid_scores), _rubrics(valid_scores), None], dtype=object
            ),
            "gpt_rollout_score_100": np.array([75.0, 75.0, None], dtype=object),
            "gpt_rollout_problem_domain": np.array(["algebra_symbolic", "algebra_symbolic", None], dtype=object),
            "gpt_rollout_difficulty_3": np.array(["medium", "medium", None], dtype=object),
            "gpt_rollout_model": np.array(["gpt-test", "gpt-test", None], dtype=object),
            "data_source": np.array(["unit", "unit", "unit"], dtype=object),
            "uid": np.array(["sample-a", "sample-b", "padding"], dtype=object),
            "extra_info": np.array(
                [{"problem": "x + 1 = 2", "problem_id": "problem-a"}, {}, {}], dtype=object
            ),
            "reward_model": np.array([{"ground_truth": "1"}, {}, {}], dtype=object),
        },
    )
    config = {
        "output_dir": str(tmp_path),
        "hidden_dtype": "float16",
        "expected_student_hidden_size": 3,
        "expected_teacher_hidden_size": 4,
        "save_text": True,
        "rubric_prompt_version": "test-v1",
    }

    stats = save_rubric_probe_batch(
        batch=batch,
        tokenizer=_FakeTokenizer(),
        config=config,
        global_step=12,
        student_checkpoint="student-checkpoint",
        teacher_checkpoint="teacher-checkpoint",
    )

    assert stats == {
        "total": 3,
        "non_orig": 1,
        "padding": 1,
        "empty_response": 0,
        "invalid_rubric": 0,
        "saved": 1,
    }
    shard = tmp_path / "step_00000012"
    tensors = load_file(shard / "tensors.safetensors")
    assert tensors["student_last"].shape == (1, 3)
    assert tensors["student_last"].dtype == torch.float16
    assert tensors["teacher_mean"].shape == (1, 4)
    assert tensors["rubric_scores"].tolist() == [valid_scores]
    assert tensors["rubric_labels"].tolist() == [[0, 1, 2, 3, 4, 5, 6]]

    metadata = pq.read_table(shard / "metadata.parquet").to_pylist()
    assert len(metadata) == 1
    assert metadata[0]["sample_id"] == "step-12:row-0:sample-a"
    assert metadata[0]["problem_id"] == "problem-a"
    assert metadata[0]["response_length"] == 2
    assert metadata[0]["ended_with_eos"] is True
    assert metadata[0]["old_log_prob_mean"] == -3.0
    assert metadata[0]["ref_log_prob_mean"] == -3.0
    assert metadata[0]["teacher_score"] == 0.0
    assert metadata[0]["rubric_prompt_version"] == "test-v1"
