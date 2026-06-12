import importlib.util
import json
from pathlib import Path

import pytest


def _load_scorer_module():
    module_path = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "gpt_rollout_scorer.py"
    spec = importlib.util.spec_from_file_location("gpt_rollout_scorer_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


scorer = _load_scorer_module()


def _rubric_scores(scores):
    return {
        name: {"score": score, "weight": scorer.RUBRIC_WEIGHTS[name], "reason": "ok"}
        for name, score in zip(scorer.RUBRIC_NAMES, scores, strict=True)
    }


def test_score_one_computes_total_from_rubrics_without_model_total(monkeypatch):
    rubric_scores = _rubric_scores([2.5, 1.5, 1.5, 2.5, 2.5, 2.5, 1.5])
    api_payload = {
        "rubric_scores": rubric_scores,
        "overall_comment": "The rubric details imply a locally computed total.",
        "revision_suggestion": "This should be cleared because the recomputed score passes.",
    }

    monkeypatch.setattr(
        scorer,
        "_post_json",
        lambda *args, **kwargs: {"output_text": json.dumps(api_payload)},
    )

    result = scorer._score_one(
        api_url="https://example.test/responses",
        api_key="test-key",
        model="test-model",
        problem="problem",
        response="response",
        ground_truth="answer",
        timeout=1,
        retries=0,
        max_output_tokens=1024,
        reasoning_effort=None,
        request_idx=1,
        request_count=1,
        verbose=False,
    )

    assert result["weighted_score_1_to_4"] == pytest.approx(2.075)
    assert result["score_100"] == pytest.approx(51.875)
    assert result["score"] == pytest.approx(0.51875)
    assert result["revision_suggestion"] == ""
    assert result["error"] == ""

def test_score_schema_does_not_ask_gpt_for_total_scores():
    schema = scorer._score_schema()

    assert "weighted_score_1_to_4" not in schema["properties"]
    assert "final_score_100" not in schema["properties"]
    assert schema["required"] == ["rubric_scores", "overall_comment", "revision_suggestion"]

