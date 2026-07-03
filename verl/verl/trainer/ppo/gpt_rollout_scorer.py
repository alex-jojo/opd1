import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


RUBRIC_NAMES = (
    "Problem Understanding and Constraint Use",
    "Mathematical Rigor",
    "Answer Correctness and Verifiability",
    "Exploration and Exploitation",
    "Solution Reasonableness",
    "Expression Fluency",
    "Expression Conciseness",
)
RUBRIC_WEIGHTS = {
    "Problem Understanding and Constraint Use": 0.15,
    "Mathematical Rigor": 0.15,
    "Answer Correctness and Verifiability": 0.20,
    "Exploration and Exploitation": 0.20,
    "Solution Reasonableness": 0.125,
    "Expression Fluency": 0.10,
    "Expression Conciseness": 0.075,
}
PROBLEM_DOMAINS = (
    "geometry_visual",
    "algebra_symbolic",
    "discrete_counting_process",
    "arithmetic_number_modeling",
)
DIFFICULTY_3_VALUES = ("easy", "medium", "hard")

TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off", ""}
_LOG_LOCK = threading.Lock()

EVALUATION_PROMPT_TEMPLATE = """You are a mathematical solution quality evaluation model. Your task is to score a given solution based on the provided [Problem], [Ground Truth Answer], and [Solution to Evaluate].
Please note: the [Ground Truth Answer] contains only the final correct answer and does not include a reference solution or intermediate reasoning. Therefore, you should not require the evaluated solution to match any specific solution method. Instead, you should independently judge whether the solution understands the problem, reasons in a mathematically meaningful way, explores useful directions when needed, reaches a correct or partially useful conclusion, and explains the process clearly.
Please score the solution strictly according to the following 7 rubrics. Each rubric should receive a numeric score from 1.0 to 4.0, allowing only 0.5-point increments:
4.0 points: Excellent performance, with almost no obvious issues.
3.0 points: Generally good, with only minor flaws.
2.0 points: Contains clear issues, but still has some reasonable content.
1.0 point: Poor performance with no effective mathematical content, content that is completely impossible to evaluate, or content unrelated to the problem.
Use 0.5-point scores, such as 2.5 or 3.5, when the quality falls between two adjacent anchor levels.

Scoring principles:
Do not automatically give a high score just because the final answer is correct; the reasoning process must also be reasonable.
Do not automatically give a low score just because the final answer is wrong; meaningful setup, useful intermediate results, or relevant exploration should receive credit.
Do not require a fully formal proof when the solution already shows correct and useful mathematical reasoning.
Do not penalize a solution simply because it differs from common methods, as long as the method is mathematically reasonable.
Do not reward long, random, or repetitive exploration if it does not use the problem structure.
Do not write a new complete solution; only evaluate the given solution itself.
If the final answer is correct but the reasoning is clearly wrong, guessed, or unsupported, lower the scores for Mathematical Rigor, Exploration and Exploitation, and Solution Reasonableness.
If the final answer is wrong but the solution contains correct and relevant intermediate work, give partial credit in the applicable rubrics.
If the solution contains multiple contradictory answers, lower the score for Answer Correctness and Verifiability.

Revision Suggestion Rules
After assigning the rubric scores, estimate the final weighted percentage. If it would be below 50, revision_suggestion is mandatory and must be non-empty. If it would be at least 50, set revision_suggestion to an empty string "".
The revision_suggestion should help the next attempt reason better, but it must not reveal the correct final answer or give a complete solution path.
Write the revision_suggestion in 1 to 2 short sentences and no more than 256 characters.
For a below-50 solution, never leave revision_suggestion blank; this field is used as supervision for a second rollout.
When the solution already has a useful setup, equation, case split, intermediate result, or promising idea, the suggestion should guide the solver to continue from that useful part, check the weak step, and complete the reasoning more carefully.
When the solution is mostly off-track, based on a wrong interpretation, or stuck in unhelpful computation, the suggestion should gently point toward a different broad direction that fits the problem structure, such as using constraints, trying cases, looking for an invariant, setting up an equation, drawing a diagram, bounding quantities, or checking special cases.
When the solution has the right general method but loses accuracy near the end, the suggestion should focus on verifying the final computation, sign, condition, format, or answer extraction.
The revision_suggestion should not mention rubric names, scores, weights, or evaluation policy.
The revision_suggestion should not say "the correct answer is..." or include the ground truth answer.
The revision_suggestion should not provide a hidden shortcut that directly determines the final answer.
The revision_suggestion should not merely say "try again" or "be more rigorous"; it should name a concrete reasoning action.

Problem Classification Rules
Classify the problem itself, ignoring the quality of the submitted solution. Use exactly one problem_domain:
- geometry_visual: synthetic geometry, coordinate geometry, areas, angles, circles, polygons, diagrams, grids, spatial/visual constructions.
- algebra_symbolic: equations, inequalities, functions, expressions, complex numbers, absolute values, symbolic casework, algebraic structure.
- discrete_counting_process: counting, probability, arrangements, finite state processes, recurrences, sequences, digit/time/card/grid counting.
- arithmetic_number_modeling: number theory, divisibility, modular/integer constraints, ratios, rates, units, prices, recipes, word-problem modeling.

Classify difficulty_3 from the shortest reasonable solution to the problem itself:
- easy: direct formula, direct count, direct substitution, or simple proportion; usually 1-2 core reasoning moves.
- medium: needs one non-obvious setup, transformation, recurrence, case split, finite enumeration, or standard theorem; usually 3-6 core reasoning moves.
- hard: needs multiple linked constraints, auxiliary construction, complex angle/area work, multi-case reasoning, or a nonlocal insight.

Rubric 1: Problem Understanding and Constraint Use
Weight: 15%
Evaluate whether the solution understands the problem and uses the important information in the problem statement.
Scoring criteria:
Whether the solution identifies the target quantity or conclusion.
Whether it correctly introduces relevant variables, equations, diagrams, cases, or conditions.
Whether it uses key constraints such as integer conditions, positivity, ranges, parity, divisibility, uniqueness, extrema, distinctness, or geometric relationships.
Whether it avoids solving a different, simplified, or misread version of the problem.
If the solution correctly sets up meaningful information from the problem, this rubric should usually receive at least 2.0 even if the later solution is incomplete.

Rubric 2: Mathematical Rigor
Weight: 15%
Evaluate whether the mathematical steps in the solution are valid and reliable.
Scoring criteria:
Whether algebraic transformations, equations, inequalities, counting arguments, case analysis, or geometric claims follow from previous steps.
Whether the solution avoids unsupported jumps, circular reasoning, contradictions, or invalid assumptions.
Whether important claims are justified enough to be believable.
Whether errors are small and local, or whether they break the whole solution.
This rubric should focus on the correctness of the reasoning that is present, not on demanding a fully formal proof.

Rubric 3: Answer Correctness and Verifiability
Weight: 20%
Evaluate whether the final answer is correct, clear, and easy to verify.
Scoring criteria:
Whether the final answer matches the ground truth answer, including signs, integer form, modular result, units, or required format.
Whether the final answer is stated clearly and uniquely.
Whether the answer follows from the preceding reasoning rather than appearing suddenly or being guessed.
Whether a wrong answer is caused by a small arithmetic, sign, simplification, or formatting error, or by a deeper reasoning failure.
If the final answer is wrong but the solution includes correct, relevant, and verifiable intermediate results, the score for this rubric should not automatically be 1.0.

Rubric 4: Exploration and Exploitation
Weight: 20%
Evaluate whether the solution chooses a reasonable amount of searching or direct solving based on how difficult the problem appears to be for the model, as reflected by the model's own output.
Scoring criteria:
For difficult problems, whether the solution tries relevant approaches such as examples, cases, transformations, equations, lemmas, bounds, or structural observations.
Whether the exploration is connected to the actual problem rather than random guessing or unrelated computation.
Whether the solution notices useful intermediate results and tries to build on them.
Whether the solution changes direction when an approach becomes unhelpful.
For easier problems, whether the solution avoids unnecessary wandering and uses a direct path.
Whether the solution can turn progress into a conclusion when enough information has been found.
High scores should go to solutions that either explore difficult problems in useful ways or solve easier problems efficiently. Low scores should go to solutions that give up too early, guess blindly, wander without purpose, or fail to use progress that it already made.

Rubric 5: Solution Reasonableness
Weight: 12.5%
Evaluate whether the overall method is appropriate for the problem.
Scoring criteria:
Whether the chosen method fits the problem type, such as algebra, number theory, counting, construction, case analysis, inequalities, or geometry.
Whether the solution focuses on the core structure of the problem rather than only surface features.
Whether the method is reasonably efficient for the problem difficulty.
Whether brute force, enumeration, or computation is used only when it is reasonable.
Whether the solution reflects reusable mathematical thinking rather than accidental manipulation.

Rubric 6: Expression Fluency
Weight: 10%
Evaluate whether the solution is coherent and easy to follow.
Scoring criteria:
Whether the solution proceeds in a natural order.
Whether variables, symbols, and intermediate conclusions are used consistently.
Whether formulas and sentences connect smoothly.
Whether the reader can understand why the solution moves from one step to the next.
Whether the solution avoids disorganized restarts, sudden unexplained shifts, or formula dumping.

Rubric 7: Expression Conciseness
Weight: 7.5%
Evaluate whether the solution is concise and avoids unnecessary content.
Scoring criteria:
Whether the solution avoids irrelevant background, small talk, or repeated statements.
Whether it avoids repeating the same computation or conclusion.
Whether it uses a reasonably short path while keeping enough explanation.
Whether it avoids excessive enumeration when a clearer structure is available.
For difficult problems, do not penalize necessary exploration, but penalize repetitive or unhelpful wandering.

Total Score Calculation
Each rubric receives a numeric score from 1.0 to 4.0 in 0.5-point increments.
The weighted score is calculated as follows:

Weighted Score =
Problem Understanding and Constraint Use * 0.15
+ Mathematical Rigor * 0.15
+ Answer Correctness and Verifiability * 0.20
+ Exploration and Exploitation * 0.20
+ Solution Reasonableness * 0.125
+ Expression Fluency * 0.10
+ Expression Conciseness * 0.075

The final weighted score ranges from 1.0 to 4.0.
The percentage score is:
Final Score = Weighted Score / 4 * 100

Do not output weighted_score_1_to_4 or final_score_100. The local evaluator code will compute both fields from rubric_scores.
If the locally computed final score would be below 50, provide a concise revision suggestion in the revision_suggestion field. It must be non-empty and no more than 256 characters.
If the locally computed final score would be at least 50, set the revision_suggestion field to an empty string "".

Input Format
[Problem]
{problem}

[Ground Truth Answer]
{ground_truth}

[Solution to Evaluate]
{solution}

Output Format
Please strictly output in the requested JSON schema and do not output any extra text.
The JSON must contain only rubric_scores, revision_suggestion, problem_domain, and difficulty_3 at the top level. Do not include weighted_score_1_to_4, final_score_100, or overall_comment."""

REVISION_SUGGESTION_PROMPT_TEMPLATE = """You are writing a concise hint for a second attempt at a math problem.
The previous evaluator scored the solution below 50/100 but did not provide a usable hint. Produce only one revision_suggestion.

Rules:
The revision_suggestion must be non-empty and no more than 256 characters.
Write 1 to 2 short sentences.
Do not reveal the correct final answer or include the ground truth answer.
Do not give a complete solution path.
Do not mention rubric names, scores, weights, or evaluation policy.
Do not merely say "try again" or "be more rigorous"; name a concrete reasoning action.
If the solution has useful partial work, guide the next attempt to continue from it and check the weak step.
If the solution is off-track, point toward a broad problem-appropriate direction such as using constraints, cases, equations, invariants, bounds, or a diagram.

[Problem]
{problem}

[Ground Truth Answer]
{ground_truth}

[Low-Scoring Solution]
{solution}

[Evaluator Notes]
{rubric_feedback}

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


def _weighted_score_from_rubrics(rubric_scores: dict[str, Any]) -> float:
    weighted_score = 0.0
    for name in RUBRIC_NAMES:
        weighted_score += float(rubric_scores[name]["score"]) * RUBRIC_WEIGHTS[name]
    return round(max(1.0, min(4.0, weighted_score)), 6)


def _score_100_from_weighted_score(weighted_score: float) -> float:
    return round(max(25.0, min(100.0, weighted_score / 4.0 * 100.0)), 6)


def _compute_scores_from_rubrics(rubric_scores: dict[str, Any]) -> tuple[float, float]:
    weighted_score = _weighted_score_from_rubrics(rubric_scores)
    return weighted_score, _score_100_from_weighted_score(weighted_score)


def _fallback_revision_suggestion(rubric_scores: dict[str, Any]) -> str:
    fallback_by_rubric = {
        "Problem Understanding and Constraint Use": (
            "Re-read the problem, identify the target and key constraints, then set up variables, cases, or equations that directly use them."
        ),
        "Mathematical Rigor": (
            "Check each inference for justification, especially the step where the argument changes direction or assumes an unstated condition."
        ),
        "Answer Correctness and Verifiability": (
            "Verify the final computation and substitute the result back into the original conditions before extracting the final answer."
        ),
        "Exploration and Exploitation": (
            "Use the most promising partial result to continue systematically; if it stalls, try cases, bounds, or an invariant tied to the constraints."
        ),
        "Solution Reasonableness": (
            "Pause at the main conclusion and test whether it is compatible with the problem conditions, edge cases, and expected size or sign."
        ),
        "Expression Fluency": (
            "Rewrite the reasoning in a clear sequence of claims and checks so the next step follows from the previous one."
        ),
        "Expression Conciseness": (
            "Remove unrelated exploration and focus on the shortest chain from the setup to a verifiable final answer."
        ),
    }
    weakest_name = "Problem Understanding and Constraint Use"
    weakest_score = float("inf")
    for name in RUBRIC_NAMES:
        try:
            score = float(rubric_scores[name]["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if score < weakest_score:
            weakest_score = score
            weakest_name = name
    return fallback_by_rubric[weakest_name][:256]


def _revision_suggestion_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "revision_suggestion": {"type": "string"},
        },
        "required": ["revision_suggestion"],
        "additionalProperties": False,
    }


def _score_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rubric_scores": {
                "type": "object",
                "properties": {
                    name: _rubric_score_schema(RUBRIC_WEIGHTS[name]) for name in RUBRIC_NAMES
                },
                "required": list(RUBRIC_NAMES),
                "additionalProperties": False,
            },
            "revision_suggestion": {"type": "string"},
            "problem_domain": {"type": "string", "enum": list(PROBLEM_DOMAINS)},
            "difficulty_3": {"type": "string", "enum": list(DIFFICULTY_3_VALUES)},
        },
        "required": [
            "rubric_scores",
            "revision_suggestion",
            "problem_domain",
            "difficulty_3",
        ],
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


def _normalize_revision_suggestion(value: Any) -> str:
    return str(value or "").strip()[:256].strip()


def _normalize_problem_domain(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in PROBLEM_DOMAINS else None


def _normalize_difficulty_3(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in DIFFICULTY_3_VALUES else None


def _rubric_feedback_for_hint(rubric_scores: dict[str, Any]) -> str:
    feedback: list[tuple[float, str]] = []
    for name in RUBRIC_NAMES:
        rubric = rubric_scores.get(name, {}) if isinstance(rubric_scores, dict) else {}
        try:
            score = float(rubric.get("score"))
        except (TypeError, ValueError):
            continue
        reason = str(rubric.get("reason", "")).replace("\n", " ").strip()
        if reason:
            feedback.append((score, reason[:240]))
    feedback.sort(key=lambda item: item[0])
    if not feedback:
        return "The evaluator found the solution below 50/100 but did not provide detailed notes."
    return "\n".join(f"- score {score:g}: {reason}" for score, reason in feedback[:3])


def _suggest_revision_one(
    *,
    api_url: str,
    api_key: str,
    model: str,
    problem: str,
    response: str,
    ground_truth: str,
    rubric_scores: dict[str, Any],
    timeout: float,
    retries: int,
    max_output_tokens: int,
    reasoning_effort: str | None,
    request_idx: int,
    request_count: int,
    verbose: bool,
) -> tuple[str, str]:
    user_prompt = REVISION_SUGGESTION_PROMPT_TEMPLATE.format(
        problem=problem,
        ground_truth=ground_truth,
        solution=response,
        rubric_feedback=_rubric_feedback_for_hint(rubric_scores),
    )
    payload = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "Write one concise revision suggestion for a low-scoring math solution. "
                    "Return only JSON matching the schema."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "math_solution_revision_suggestion",
                "strict": True,
                "schema": _revision_suggestion_schema(),
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
                f"hint request {request_idx}/{request_count} attempt {attempt + 1}/{total_attempts} start "
                f"model={model} timeout={timeout:g}s max_output_tokens={max_output_tokens}"
            )
        try:
            api_response = _post_json(api_url, api_key, payload, timeout)
            text = _extract_response_text(api_response)
            parsed = json.loads(text)
            revision_suggestion = _normalize_revision_suggestion(parsed.get("revision_suggestion", ""))
            if revision_suggestion:
                if verbose:
                    _debug_print(
                        f"hint request {request_idx}/{request_count} attempt {attempt + 1}/{total_attempts} done "
                        f"chars={len(revision_suggestion)} elapsed={time.time() - attempt_start:.1f}s"
                    )
                return revision_suggestion, ""
            last_error = "empty revision_suggestion"
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body}"
        except Exception as exc:
            last_error = str(exc)

        if verbose:
            status = "retry" if attempt < retries else "failed"
            _debug_print(
                f"hint request {request_idx}/{request_count} attempt {attempt + 1}/{total_attempts} {status} "
                f"elapsed={time.time() - attempt_start:.1f}s error={_format_error(last_error)}"
            )
        if attempt < retries:
            sleep_seconds = min(2**attempt, 8)
            if verbose:
                _debug_print(f"hint request {request_idx}/{request_count} sleep_before_retry={sleep_seconds}s")
            time.sleep(sleep_seconds)

    return "", last_error


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
    hint_retries: int,
    hint_max_output_tokens: int,
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
            rubric_scores = parsed["rubric_scores"]
            problem_domain = _normalize_problem_domain(parsed.get("problem_domain"))
            difficulty_3 = _normalize_difficulty_3(parsed.get("difficulty_3"))
            if problem_domain is None:
                raise ValueError(f"invalid problem_domain: {parsed.get('problem_domain')!r}")
            if difficulty_3 is None:
                raise ValueError(f"invalid difficulty_3: {parsed.get('difficulty_3')!r}")
            weighted_score, score_100 = _compute_scores_from_rubrics(rubric_scores)
            if verbose:
                _debug_print(
                    f"request {request_idx}/{request_count} attempt {attempt + 1}/{total_attempts} done "
                    f"score_100={score_100:.1f} elapsed={time.time() - attempt_start:.1f}s"
                )
            revision_suggestion = _normalize_revision_suggestion(parsed.get("revision_suggestion", ""))
            revision_suggestion_source = "gpt" if revision_suggestion else ""
            if score_100 >= 50.0:
                revision_suggestion = ""
                revision_suggestion_source = ""
            else:
                if not revision_suggestion:
                    revision_suggestion, hint_error = _suggest_revision_one(
                        api_url=api_url,
                        api_key=api_key,
                        model=model,
                        problem=problem,
                        response=response,
                        ground_truth=ground_truth,
                        rubric_scores=rubric_scores,
                        timeout=timeout,
                        retries=hint_retries,
                        max_output_tokens=hint_max_output_tokens,
                        reasoning_effort=reasoning_effort,
                        request_idx=request_idx,
                        request_count=request_count,
                        verbose=verbose,
                    )
                    if revision_suggestion:
                        revision_suggestion_source = "gpt_hint_retry"
                    else:
                        revision_suggestion = _fallback_revision_suggestion(rubric_scores)
                        revision_suggestion_source = "local_fallback"
                        if verbose:
                            _debug_print(
                                f"hint request {request_idx}/{request_count} local_fallback "
                                f"error={_format_error(hint_error)}"
                            )

            return {
                "score": score_100 / 100.0,
                "score_100": score_100,
                "weighted_score_1_to_4": weighted_score,
                "rubric_scores": rubric_scores,
                "reason": "",
                "revision_suggestion": revision_suggestion,
                "revision_suggestion_source": revision_suggestion_source,
                "problem_domain": problem_domain,
                "difficulty_3": difficulty_3,
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
        "revision_suggestion_source": "",
        "problem_domain": None,
        "difficulty_3": None,
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
    hint_retries = max(0, int(_get(config, "hint_retries", 0)))
    hint_max_output_tokens = int(_get(config, "hint_max_output_tokens", 256))
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
            f"hint_retries={hint_retries} hint_max_output_tokens={hint_max_output_tokens} "
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
                "hint_retries": hint_retries,
                "hint_max_output_tokens": hint_max_output_tokens,
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
                    "revision_suggestion_source": "",
                    "problem_domain": None,
                    "difficulty_3": None,
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
            "revision_suggestion_source": "",
            "problem_domain": None,
            "difficulty_3": None,
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
        "revision_suggestion_sources": [result["revision_suggestion_source"] for result in results],
        "problem_domains": [result["problem_domain"] for result in results],
        "difficulty_3": [result["difficulty_3"] for result in results],
        "errors": [result["error"] for result in results],
        "models": [model for _ in results],
    }
