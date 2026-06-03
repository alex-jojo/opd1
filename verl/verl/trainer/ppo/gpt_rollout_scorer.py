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
Please score the solution strictly according to the following 5 rubrics. Each rubric should receive an integer score from 1 to 4:
4 points: Excellent performance, with almost no obvious issues.
3 points: Generally good, with only minor flaws.
2 points: Contains clear issues, but still has some reasonable content.
1 point: Poor performance, with serious errors, missing reasoning, or content that cannot be effectively evaluated.

Scoring principles:
Do not automatically give a high score just because the final answer is correct; the reasoning process must also be reasonable.
Do not penalize a solution simply because it differs from common methods, as long as it is mathematically correct and reasonable.
Do not write a new complete solution; only evaluate the given solution itself.
If the reasoning process is correct but the final answer is wrong, lower the score for "Answer Correctness and Verifiability", and adjust other scores appropriately based on the source of the error.
If the final answer is correct but the reasoning is wrong, severely incomplete, or appears to rely on guessing, lower the scores for "Mathematical Rigor" and "Solution Reasonableness".
If the solution contains multiple mutually contradictory answers, lower the score for "Answer Correctness and Verifiability".
If the final percentage score is below 50, provide a concise revision suggestion. If the final score is greater than or equal to 50, do not provide a revision suggestion.

Rubric 1: Mathematical Rigor
Weight: 25%
Evaluate whether the mathematical reasoning in the solution is rigorous and reliable.
Scoring criteria:
Whether the reasoning chain is valid, and whether each step follows naturally from the problem statement, definitions, theorems, or previous steps.
Whether there are issues such as skipped steps, circular reasoning, unsupported intuition-based claims, or reasoning backward from the conclusion.
Whether key conditions are fully used, such as integer constraints, positivity, ranges, uniqueness, extrema, divisibility, distinctness, and other restrictions.
For problems requiring case analysis, whether all cases are covered and impossible cases are reasonably eliminated.

Rubric 2: Answer Correctness and Verifiability
Weight: 20%
Evaluate whether the final answer is correct, clear, and easy to verify.
Scoring criteria:
Whether the final answer matches the ground truth answer, especially regarding signs, integer form, modular results, or any special output format required by the problem.
Whether the final answer is clear and unique, without multiple mutually contradictory answers.
Whether the answer follows naturally from the preceding derivation, rather than being suddenly stated or guessed.
Whether the final answer is expressed clearly, avoiding ambiguity, redundancy, or forms that are difficult to parse.

Rubric 3: Expression Fluency
Weight: 15%
Evaluate whether the solution process is natural, coherent, and easy to read.
Scoring criteria:
Whether the solution proceeds in a natural order, such as understanding the problem, establishing relationships, deriving or computing results, and reaching a conclusion.
Whether variables, symbols, and intermediate conclusions are consistent throughout, avoiding cases where the same symbol refers to different objects.
Whether sentences, formulas, and paragraphs connect smoothly, allowing the reader to follow the solution.
Whether the solution avoids disorganized trial-and-error, repeated revisions, sudden shifts in direction, or unexplained formula dumping.

Rubric 4: Expression Conciseness
Weight: 15%
Evaluate whether the solution is concise and effective, without obvious redundancy.
Scoring criteria:
Whether the solution includes only necessary reasoning and avoids irrelevant background information, small talk, or excessive explanation.
Whether it avoids repeating the same conclusion, repeating computations, or including clearly redundant intermediate steps.
Whether it completes the proof or computation through a relatively short path while maintaining rigor.
For simple problems, whether it avoids unnecessary overcomplication; for difficult problems, whether it avoids using large amounts of ineffective enumeration to obscure the core idea.

Rubric 5: Solution Reasonableness
Weight: 25%
Evaluate whether the chosen solution method is appropriate for the problem, whether it captures the key structure, and whether it is relatively strong among possible solution methods.
Scoring criteria:
Whether the chosen method fits the problem type, such as algebraic manipulation, case analysis, construction, counting, number-theoretic reasoning, geometric relationships, and so on.
Whether the solution captures the core structure of the problem, rather than relying on blind enumeration, guessing, or inefficient computation.
Whether it uses a relatively natural, stable, and less error-prone solution path; if the method is clearly roundabout or unnecessarily complicated, lower the score.
Whether the solution reflects reusable mathematical thinking, rather than an accidental operation tailored only to a single answer.

Total Score Calculation
Each rubric receives an integer score from 1 to 4.
The weighted score is calculated as follows:
Weighted Score =
Mathematical Rigor * 0.25
Answer Correctness and Verifiability * 0.20
Expression Fluency * 0.15
Expression Conciseness * 0.15
Solution Reasonableness * 0.25
The final weighted score ranges from 1 to 4.
The percentage score is:
Final Score = Weighted Score / 4 * 100
If final_score_100 < 50, provide a concise revision suggestion in the revision_suggestion field.
If final_score_100 >= 50, set the revision_suggestion field to an empty string "".

Input Format
[Problem]
{problem}

[Ground Truth Answer]
{ground_truth}

[Solution to Evaluate]
{solution}

Output Format
Please strictly output in the requested JSON schema and do not output any extra text."""


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


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


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
            "score": {"type": "integer", "minimum": 1, "maximum": 4},
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
                    "Expression Conciseness": _rubric_score_schema(0.15),
                    "Solution Reasonableness": _rubric_score_schema(0.25),
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
    model = _get(config, "model", "gpt-4.1-mini")
    timeout = float(_get(config, "timeout", 60))
    retries = int(_get(config, "retries", 2))
    max_workers = max(1, int(_get(config, "max_workers", 8)))
    max_prompt_chars = int(_get(config, "max_prompt_chars", 8000))
    max_response_chars = int(_get(config, "max_response_chars", 16000))
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
            f"timeout={timeout:g}s retries={retries} max_output_tokens={max_output_tokens} api_url={api_url}"
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
                "problem": _truncate_middle(_to_text(problem), max_prompt_chars),
                "response": _truncate_middle(response_text, max_response_chars),
                "ground_truth": _truncate_middle(_to_text(ground_truth), max_prompt_chars),
                "timeout": timeout,
                "retries": retries,
                "max_output_tokens": max_output_tokens,
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
