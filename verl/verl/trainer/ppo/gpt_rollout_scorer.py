import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


RUBRIC_NAMES = (
    "Mathematical Rigor",
    "Answer Correctness and Verifiability",
    "Expression Fluency",
    "Expression Conciseness",
    "Solution Reasonableness",
)

TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off", ""}
_LOG_LOCK = threading.Lock()

EVALUATION_PROMPT_TEMPLATE = """You are a mathematical solution quality evaluation model. Your task is to score a given solution based on the provided [Problem], [Ground Truth Answer], and [Solution to Evaluate].
Please note: the [Ground Truth Answer] contains only the final correct answer and does not include a reference solution or intermediate reasoning. Therefore, you should not require the evaluated solution to match any specific solution method. Instead, you should independently judge whether the mathematical reasoning is valid, whether the final answer is correct, and whether the explanation is clear and reasonable.
Please score the solution strictly according to the following 5 rubrics. Each rubric should receive a numeric score from 1.0 to 4.0, allowing only 0.5-point increments:
4.0 points: Excellent performance, with almost no obvious issues.
3.0 points: Generally good, with only minor flaws.
2.0 points: Contains clear issues, but still has some reasonable content.
1.0 point: Poor performance with no effective mathematical content, content that is completely impossible to evaluate, or content unrelated to the problem.
Use 0.5-point scores, such as 2.5 or 3.5, when the quality falls between two adjacent anchor levels.

Scoring principles:
Do not automatically give a high score just because the final answer is correct; the reasoning process must also be reasonable.
Do not penalize a solution simply because it differs from common methods, as long as it is mathematically correct and reasonable.
Do not write a new complete solution; only evaluate the given solution itself.
If the reasoning process is correct but the final answer is wrong, lower the score for "Answer Correctness and Verifiability", and adjust other scores appropriately based on the source of the error.
If the final answer is correct but the reasoning is wrong, severely incomplete, or appears to rely on guessing, lower the scores for "Mathematical Rigor" and "Solution Reasonableness".
If the solution contains multiple mutually contradictory answers, lower the score for "Answer Correctness and Verifiability".
If the solution contains partially correct key intermediate expressions, transformations, formulas, computations, or reasoning steps that are relevant to the problem, give at least 2.0 for the applicable rubric(s), even if the solution is incomplete or has later mistakes.
If the final percentage score is below 45, provide a concise revision suggestion. If the final score is greater than or equal to 45, do not provide a revision suggestion.
When writing the revision_suggestion, it should help guide and correct the model's reasoning process rather than directly revealing the correct final answer.

Rubric 1: Mathematical Rigor
Weight: 25%
Evaluate whether the mathematical reasoning in the solution is rigorous and reliable.
Assign 1.0 only when there is no effective mathematical content, the content is completely impossible to evaluate, or the response is unrelated to the problem.
Scoring criteria:
Whether the reasoning chain is valid, and whether each step follows naturally from the problem statement, definitions, theorems, or previous steps.
Whether there are issues such as skipped steps, circular reasoning, unsupported intuition-based claims, or reasoning backward from the conclusion.
Whether key conditions are fully used, such as integer constraints, positivity, ranges, uniqueness, extrema, divisibility, distinctness, and other restrictions.
For problems requiring case analysis, whether all cases are covered and impossible cases are reasonably eliminated.
If some key intermediate reasoning, equations, transformations, or case setup is mathematically valid and relevant, the score for this rubric should be at least 2.0. Give 1.0 only when the solution has no valid mathematical reasoning, is impossible to evaluate, or is unrelated to the problem.

Rubric 2: Answer Correctness and Verifiability
Weight: 20%
Evaluate whether the final answer is correct, clear, and easy to verify.
Assign 1.0 only when there is no effective mathematical content, the content is completely impossible to evaluate, or the response is unrelated to the problem.
Scoring criteria:
Whether the final answer matches the ground truth answer, especially regarding signs, integer form, modular results, or any special output format required by the problem.
Whether the final answer is clear and unique, without multiple mutually contradictory answers.
Whether the answer follows naturally from the preceding derivation, rather than being suddenly stated or guessed.
Whether the final answer is expressed clearly, avoiding ambiguity, redundancy, or forms that are difficult to parse.
If the final answer is wrong but the solution includes a clearly correct, relevant, and verifiable intermediate result, or the mistake is only a small final arithmetic, sign, simplification, or formatting error, the score for this rubric should be at least 2.0. Give 1.0 only when there is no meaningful answer to verify, the stated answer is unrelated to the problem, or the response is impossible to evaluate.

Rubric 3: Expression Fluency
Weight: 15%
Evaluate whether the solution process is natural, coherent, and easy to read.
Scoring criteria:
Whether the solution proceeds in a natural order, such as understanding the problem, establishing relationships, deriving or computing results, and reaching a conclusion.
Whether variables, symbols, and intermediate conclusions are consistent throughout, avoiding cases where the same symbol refers to different objects.
Whether sentences, formulas, and paragraphs connect smoothly, allowing the reader to follow the solution.
Whether the solution avoids disorganized trial-and-error, repeated revisions, sudden shifts in direction, or unexplained formula dumping.

Rubric 4: Expression Conciseness
Weight: 10%
Evaluate whether the solution is concise and effective, without obvious redundancy.
Scoring criteria:
Whether the solution includes only necessary reasoning and avoids irrelevant background information, small talk, or excessive explanation.
Whether it avoids repeating the same conclusion, repeating computations, or including clearly redundant intermediate steps.
Whether it completes the proof or computation through a relatively short path while maintaining rigor.
For simple problems, whether it avoids unnecessary overcomplication; for difficult problems, whether it avoids using large amounts of ineffective enumeration to obscure the core idea.

Rubric 5: Solution Reasonableness
Weight: 30%
Evaluate whether the chosen solution method is appropriate for the problem, whether it captures the key structure, and whether it is relatively strong among possible solution methods.
Scoring criteria:
Whether the chosen method fits the problem type, such as algebraic manipulation, case analysis, construction, counting, number-theoretic reasoning, geometric relationships, and so on.
Whether the solution captures the core structure of the problem, rather than relying on blind enumeration, guessing, or inefficient computation.
Whether it uses a relatively natural, stable, and less error-prone solution path; if the method is clearly roundabout or unnecessarily complicated, lower the score.
Whether the solution reflects reusable mathematical thinking, rather than an accidental operation tailored only to a single answer.

Total Score Calculation
Each rubric receives a numeric score from 1.0 to 4.0 in 0.5-point increments.
The weighted score is calculated as follows:
Weighted Score =
Mathematical Rigor * 0.25
Answer Correctness and Verifiability * 0.20
Expression Fluency * 0.15
Expression Conciseness * 0.10
Solution Reasonableness * 0.30
The final weighted score ranges from 1.0 to 4.0.
The percentage score is:
Final Score = Weighted Score / 4 * 100
If final_score_100 < 45, provide a concise revision suggestion in the revision_suggestion field.
If final_score_100 >= 45, set the revision_suggestion field to an empty string "".

Input Format
[Problem]
{problem}

[Ground Truth Answer]
{ground_truth}

[Solution to Evaluate]
{solution}

Output Format
Please strictly output in the requested JSON schema and do not output any extra text."""

REROLL_CONTEXT_SUMMARY_PROMPT_TEMPLATE = """Aggressively compress the reroll context below into one concise paragraph of 512 to 768 words so it can be appended to a student model prompt.

Requirements:
- Aggressively compress the [Previous Solution] and [GPT Feedback on Previous Solution] together in one pass.
- Compress them into one concise paragraph of 512 to 768 words.
- Return a single reroll_context string that still contains both section headers, in this order:
  [Previous Solution]
  [GPT Feedback on Previous Solution]
- Do not include the problem statement or the final "solve again" instruction in reroll_context.
- Do not solve the math problem again.
- Do not improve, correct, or polish the previous solution beyond compression.
- If the previous solution is wrong, vague, repetitive, disorganized, or overly verbose, preserve that quality and meaning.
- Preserve all important mistakes, contradictions, final answers, and reasoning choices from the previous solution.
- Preserve actionable GPT feedback, low rubric scores, and revision suggestions.
- The entire returned reroll_context must be one concise paragraph of 512 to 768 words and at most {target_tokens} student-tokenizer tokens.

[Reroll Context To Compress: Previous Solution + GPT Feedback]
{context}

Return only JSON matching the schema."""


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
    return bool(value)


def _debug_print(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_LOCK:
        print(f"[gpt_rollout_score] {timestamp} {message}", flush=True)


def _format_error(error: str, max_chars: int = 240) -> str:
    return str(error).replace("\n", " ")[:max_chars]


def _to_list(value: Any, length: int, default: Any) -> list[Any]:
    if value is None:
        return [default for _ in range(length)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    value = list(value)
    if len(value) < length:
        value.extend(default for _ in range(length - len(value)))
    return value


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


TRUNCATION_MARKER = "\n...[truncated]...\n"


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return list(token_ids)


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _truncate_middle_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text

    token_ids = _encode_text(tokenizer, text)
    if len(token_ids) <= max_tokens:
        return text

    marker_ids = _encode_text(tokenizer, TRUNCATION_MARKER)
    if len(marker_ids) >= max_tokens:
        return _decode_token_ids(tokenizer, token_ids[:max_tokens])

    kept_tokens = max_tokens - len(marker_ids)
    head = kept_tokens // 3
    tail = kept_tokens - head
    tail_ids = token_ids[-tail:] if tail > 0 else []
    return _decode_token_ids(tokenizer, token_ids[:head] + marker_ids + tail_ids)


def _extract_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _post_json(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _rubric_score_schema(weight: float) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number", "enum": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]},
            "weight": {"type": "number", "enum": [weight]},
            "reason": {"type": "string"},
        },
        "required": ["score", "weight", "reason"],
        "additionalProperties": False,
    }


def _score_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rubric_scores": {
                "type": "object",
                "properties": {
                    "Mathematical Rigor": _rubric_score_schema(0.25),
                    "Answer Correctness and Verifiability": _rubric_score_schema(0.20),
                    "Expression Fluency": _rubric_score_schema(0.15),
                    "Expression Conciseness": _rubric_score_schema(0.10),
                    "Solution Reasonableness": _rubric_score_schema(0.30),
                },
                "required": list(RUBRIC_NAMES),
                "additionalProperties": False,
            },
            "weighted_score_1_to_4": {"type": "number", "minimum": 1, "maximum": 4},
            "final_score_100": {"type": "number", "minimum": 25, "maximum": 100},
            "overall_comment": {"type": "string"},
            "revision_suggestion": {"type": "string"},
        },
        "required": [
            "rubric_scores",
            "weighted_score_1_to_4",
            "final_score_100",
            "overall_comment",
            "revision_suggestion",
        ],
        "additionalProperties": False,
    }


def _reroll_context_summary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reroll_context": {"type": "string"},
        },
        "required": ["reroll_context"],
        "additionalProperties": False,
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "false", "0", "default"}:
        return None
    return text


def _model_supports_reasoning_effort(model: str) -> bool:
    normalized = str(model).lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def _add_reasoning_effort(payload: dict[str, Any], model: str, reasoning_effort: str | None) -> None:
    reasoning_effort = _optional_str(reasoning_effort)
    if reasoning_effort is not None and _model_supports_reasoning_effort(model):
        payload["reasoning"] = {"effort": reasoning_effort}


def _score_one(
    *,
    api_url: str,
    api_key: str,
    model: str,
    problem: str,
    response: str,
    ground_truth: str,
    timeout: float,
    retries: int,
    max_output_tokens: int,
    reasoning_effort: str | None,
    request_idx: int,
    request_count: int,
    verbose: bool,
) -> dict[str, Any]:
    user_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        problem=problem,
        ground_truth=ground_truth,
        solution=response,
    )
    payload = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "Evaluate the supplied mathematical solution according to the user's rubric. "
                    "Return only JSON matching the schema."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "math_solution_quality_score",
                "strict": True,
                "schema": _score_schema(),
            }
        },
        "max_output_tokens": max_output_tokens,
    }
    _add_reasoning_effort(payload, model, reasoning_effort)

    last_error = ""
    total_attempts = retries + 1
    for attempt in range(total_attempts):
        attempt_start = time.time()
        if verbose:
            _debug_print(
                f"request {request_idx}/{request_count} attempt {attempt + 1}/{total_attempts} start "
                f"model={model} prompt_chars={len(problem)} response_chars={len(response)} "
                f"ground_truth_chars={len(ground_truth)} timeout={timeout:g}s max_output_tokens={max_output_tokens}"
            )
        try:
            api_response = _post_json(api_url, api_key, payload, timeout)
            text = _extract_response_text(api_response)
            parsed = json.loads(text)
            score_100 = max(0.0, min(100.0, float(parsed["final_score_100"])))
            if verbose:
                _debug_print(
                    f"request {request_idx}/{request_count} attempt {attempt + 1}/{total_attempts} done "
                    f"score_100={score_100:.1f} elapsed={time.time() - attempt_start:.1f}s"
                )
            return {
                "score": score_100 / 100.0,
                "score_100": score_100,
                "weighted_score_1_to_4": float(parsed["weighted_score_1_to_4"]),
                "rubric_scores": parsed["rubric_scores"],
                "reason": str(parsed.get("overall_comment", "")),
                "revision_suggestion": str(parsed.get("revision_suggestion", "")),
                "error": "",
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            last_error = str(exc)

        if verbose:
            status = "retry" if attempt < retries else "failed"
            _debug_print(
                f"request {request_idx}/{request_count} attempt {attempt + 1}/{total_attempts} {status} "
                f"elapsed={time.time() - attempt_start:.1f}s error={_format_error(last_error)}"
            )
        if attempt < retries:
            sleep_seconds = min(2**attempt, 8)
            if verbose:
                _debug_print(f"request {request_idx}/{request_count} sleep_before_retry={sleep_seconds}s")
            time.sleep(sleep_seconds)

    return {
        "score": None,
        "score_100": None,
        "weighted_score_1_to_4": None,
        "rubric_scores": None,
        "reason": "",
        "revision_suggestion": "",
        "error": last_error,
    }


def score_rollouts_with_gpt(batch: Any, tokenizer: Any, config: Any) -> dict[str, list[Any]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when trainer.gpt_rollout_score.enable=True")

    base_url = _get(config, "base_url", None) or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_url = base_url.rstrip("/") + "/responses"
    model = _get(config, "model", "chat-latest") or "chat-latest"
    reasoning_effort = _get(config, "reasoning_effort", os.environ.get("GPT_ROLLOUT_SCORE_REASONING_EFFORT", None))
    timeout = float(_get(config, "timeout", 60))
    retries = int(_get(config, "retries", 2))
    max_workers = max(1, int(_get(config, "max_workers", 8)))
    max_prompt_tokens = int(_get(config, "max_prompt_chars", 2048))
    max_response_tokens = int(_get(config, "max_response_chars", 4096))
    max_output_tokens = int(_get(config, "max_output_tokens", 1024))
    verbose = _to_bool(_get(config, "verbose", os.environ.get("GPT_ROLLOUT_SCORE_VERBOSE", "0")))

    prompt_ids = batch.batch["prompts"]
    response_ids = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    prompt_len = prompt_ids.shape[-1]
    batch_size = len(batch)

    worker_count = min(max_workers, max(1, batch_size))
    if verbose:
        _debug_print(
            f"batch start requests={batch_size} max_workers={worker_count} model={model} "
            f"timeout={timeout:g}s retries={retries} max_output_tokens={max_output_tokens} "
            f"max_prompt_tokens={max_prompt_tokens} max_response_tokens={max_response_tokens} "
            f"reasoning_effort={_optional_str(reasoning_effort) or 'default'} api_url={api_url}"
        )

    extra_infos = _to_list(batch.non_tensor_batch.get("extra_info"), batch_size, {})
    reward_models = _to_list(batch.non_tensor_batch.get("reward_model"), batch_size, {})

    requests: list[dict[str, Any]] = []
    for i in range(batch_size):
        valid_prompt_len = int(attention_mask[i, :prompt_len].sum().item())
        valid_response_len = int(attention_mask[i, prompt_len:].sum().item())

        valid_prompt_ids = prompt_ids[i, -valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[i, :0]
        valid_response_ids = response_ids[i, :valid_response_len]

        prompt_text = tokenizer.decode(valid_prompt_ids.detach().cpu().tolist(), skip_special_tokens=True)
        response_text = tokenizer.decode(valid_response_ids.detach().cpu().tolist(), skip_special_tokens=True)

        extra_info = extra_infos[i] if isinstance(extra_infos[i], dict) else {}
        reward_model = reward_models[i] if isinstance(reward_models[i], dict) else {}
        problem = extra_info.get("problem") or extra_info.get("question") or prompt_text
        ground_truth = reward_model.get("ground_truth") or extra_info.get("answer") or ""

        requests.append(
            {
                "api_url": api_url,
                "api_key": api_key,
                "model": model,
                "problem": _truncate_middle_tokens(_to_text(problem), tokenizer, max_prompt_tokens),
                "response": _truncate_middle_tokens(response_text, tokenizer, max_response_tokens),
                "ground_truth": _truncate_middle_tokens(_to_text(ground_truth), tokenizer, max_prompt_tokens),
                "timeout": timeout,
                "retries": retries,
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": reasoning_effort,
                "request_idx": i + 1,
                "request_count": batch_size,
                "verbose": verbose,
            }
        )

    results: list[dict[str, Any] | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {executor.submit(_score_one, **request): idx for idx, request in enumerate(requests)}
        for done_count, future in enumerate(as_completed(future_to_idx), start=1):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "score": None,
                    "score_100": None,
                    "weighted_score_1_to_4": None,
                    "rubric_scores": None,
                    "reason": "",
                    "revision_suggestion": "",
                    "error": str(exc),
                }
            results[idx] = result
            if verbose:
                score = result["score_100"]
                score_text = "error" if score is None else f"{float(score):.1f}"
                error_text = result["error"]
                if error_text:
                    _debug_print(
                        f"batch progress done={done_count}/{batch_size} request_idx={idx + 1} "
                        f"score_100={score_text} error={_format_error(error_text)}"
                    )
                else:
                    _debug_print(
                        f"batch progress done={done_count}/{batch_size} request_idx={idx + 1} "
                        f"score_100={score_text}"
                    )

    results = [
        result
        if result is not None
        else {
            "score": None,
            "score_100": None,
            "weighted_score_1_to_4": None,
            "rubric_scores": None,
            "reason": "",
            "revision_suggestion": "",
            "error": "missing GPT rollout scoring result",
        }
        for result in results
    ]

    if verbose:
        valid_count = sum(result["score"] is not None for result in results)
        _debug_print(f"batch result_summary valid_count={valid_count} error_count={batch_size - valid_count}")

    return {
        "scores": [result["score"] for result in results],
        "scores_100": [result["score_100"] for result in results],
        "weighted_scores_1_to_4": [result["weighted_score_1_to_4"] for result in results],
        "rubric_scores": [result["rubric_scores"] for result in results],
        "reasons": [result["reason"] for result in results],
        "revision_suggestions": [result["revision_suggestion"] for result in results],
        "errors": [result["error"] for result in results],
        "models": [model for _ in results],
    }


def summarize_reroll_context_with_gpt(
    *,
    context: str,
    target_tokens: int,
    config: Any,
    request_idx: int,
    verbose: bool = False,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when reroll context summarization is enabled")

    base_url = _get(config, "base_url", None) or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_url = base_url.rstrip("/") + "/responses"
    model = _get(config, "reroll_summary_model", None) or _get(config, "model", "chat-latest") or "chat-latest"
    reasoning_effort = _get(config, "reasoning_effort", os.environ.get("GPT_ROLLOUT_SCORE_REASONING_EFFORT", None))
    timeout = float(_get(config, "timeout", 60))
    retries = int(_get(config, "retries", 2))
    max_output_tokens = int(_get(config, "reroll_summary_max_output_tokens", 1024))
    user_prompt = REROLL_CONTEXT_SUMMARY_PROMPT_TEMPLATE.format(
        target_tokens=target_tokens,
        context=context,
    )
    payload = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "Compress reroll context for a math training pipeline. Preserve the previous solution's "
                    "meaning, quality, flaws, and the GPT feedback. Return only JSON matching the schema."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reroll_context_summary",
                "strict": True,
                "schema": _reroll_context_summary_schema(),
            }
        },
        "max_output_tokens": max_output_tokens,
    }
    _add_reasoning_effort(payload, model, reasoning_effort)

    last_error = ""
    total_attempts = retries + 1
    for attempt in range(total_attempts):
        attempt_start = time.time()
        if verbose:
            _debug_print(
                f"reroll_summary request_idx={request_idx} attempt={attempt + 1}/{total_attempts} "
                f"model={model} context_chars={len(context)} target_tokens={target_tokens} "
                f"max_output_tokens={max_output_tokens} reasoning_effort={_optional_str(reasoning_effort) or 'default'}"
            )
        try:
            api_response = _post_json(api_url, api_key, payload, timeout)
            text = _extract_response_text(api_response)
            parsed = json.loads(text)
            if verbose:
                _debug_print(
                    f"reroll_summary request_idx={request_idx} done elapsed={time.time() - attempt_start:.1f}s"
                )
            return {
                "reroll_context": str(parsed["reroll_context"]),
                "model": model,
                "error": "",
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            last_error = str(exc)

        if verbose:
            status = "retry" if attempt < retries else "failed"
            _debug_print(
                f"reroll_summary request_idx={request_idx} attempt={attempt + 1}/{total_attempts} {status} "
                f"elapsed={time.time() - attempt_start:.1f}s error={_format_error(last_error)}"
            )
        if attempt < retries:
            time.sleep(min(2**attempt, 8))

    return {
        "reroll_context": "",
        "model": model,
        "error": last_error,
    }
