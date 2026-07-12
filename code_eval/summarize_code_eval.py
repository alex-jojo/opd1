#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().rstrip('%')
        if not value or value.upper() == 'NA':
            return None
        try:
            out = float(value)
            if out > 1.0:
                out /= 100.0
            return out
        except ValueError:
            return None
    return None


def _fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'NA'
    return f'{100.0 * float(value):.2f}'


def summarize_lcb(eval_all_file: Path):
    if not eval_all_file or not eval_all_file.exists():
        return None, None, None
    rows = json.loads(eval_all_file.read_text())
    if not rows:
        return None, None, None

    per_problem_avg = []
    per_problem_any = []
    per_problem_maj = []
    for row in rows:
        graded = row.get('graded_list') or []
        vals = [bool(x) for x in graded]
        if not vals:
            continue
        correct = sum(vals)
        total = len(vals)
        per_problem_avg.append(correct / total)
        per_problem_any.append(1.0 if correct > 0 else 0.0)
        per_problem_maj.append(1.0 if correct > total / 2 else 0.0)

    if not per_problem_avg:
        return None, None, None
    avg_at_8 = sum(per_problem_avg) / len(per_problem_avg)
    p_at_8 = sum(per_problem_any) / len(per_problem_any)
    maj_at_8 = sum(per_problem_maj) / len(per_problem_maj)
    return avg_at_8, p_at_8, maj_at_8


def summarize_evalplus(result_file: Path | None, k: int = 8, require_plus: bool = True):
    if not result_file or not result_file.exists():
        return None, None, None

    data = json.loads(result_file.read_text())
    if not isinstance(data, dict):
        return None, None, None

    eval_rows = data.get('eval') or {}
    if not isinstance(eval_rows, dict):
        return None, None, None

    per_problem_avg = []
    per_problem_any = []
    per_problem_maj = []
    for task_results in eval_rows.values():
        if not isinstance(task_results, list):
            continue
        vals = []
        for result in task_results[:k]:
            if not isinstance(result, dict):
                continue
            passed = result.get('base_status') == 'pass'
            if require_plus:
                passed = passed and result.get('plus_status') == 'pass'
            vals.append(passed)
        if len(vals) < k:
            continue
        correct = sum(vals)
        total = len(vals)
        per_problem_avg.append(correct / total)
        per_problem_any.append(1.0 if correct > 0 else 0.0)
        per_problem_maj.append(1.0 if correct > total / 2 else 0.0)

    if not per_problem_avg:
        return None, None, None
    avg_at_k = sum(per_problem_avg) / len(per_problem_avg)
    p_at_k = sum(per_problem_any) / len(per_problem_any)
    maj_at_k = sum(per_problem_maj) / len(per_problem_maj)
    return avg_at_k, p_at_k, maj_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--lcb-eval-all-file', type=Path, required=True)
    parser.add_argument('--humaneval-result-file', type=Path, default=None)
    parser.add_argument('--mbpp-result-file', type=Path, default=None)
    parser.add_argument('--output-file', type=Path, required=True)
    args = parser.parse_args()

    lcb_avg, lcb_p, lcb_maj = summarize_lcb(args.lcb_eval_all_file)
    humaneval_avg, humaneval_p, humaneval_maj = summarize_evalplus(args.humaneval_result_file, require_plus=True)
    mbpp_avg, mbpp_p, mbpp_maj = summarize_evalplus(args.mbpp_result_file, require_plus=False)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'model',
        'LiveCodeBenchv5 AVG@8',
        'LiveCodeBenchv5 P@8',
        'LiveCodeBenchv5 maj@8',
        'HumanEval+ AVG@8',
        'HumanEval+ P@8',
        'HumanEval+ maj@8',
        'MBPP AVG@8',
        'MBPP P@8',
        'MBPP maj@8',
        'lcb_eval_all_file',
        'humaneval_result_file',
        'mbpp_result_file',
    ]
    with args.output_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'model': args.model_name,
            'LiveCodeBenchv5 AVG@8': _fmt(lcb_avg),
            'LiveCodeBenchv5 P@8': _fmt(lcb_p),
            'LiveCodeBenchv5 maj@8': _fmt(lcb_maj),
            'HumanEval+ AVG@8': _fmt(humaneval_avg),
            'HumanEval+ P@8': _fmt(humaneval_p),
            'HumanEval+ maj@8': _fmt(humaneval_maj),
            'MBPP AVG@8': _fmt(mbpp_avg),
            'MBPP P@8': _fmt(mbpp_p),
            'MBPP maj@8': _fmt(mbpp_maj),
            'lcb_eval_all_file': str(args.lcb_eval_all_file),
            'humaneval_result_file': '' if args.humaneval_result_file is None else str(args.humaneval_result_file),
            'mbpp_result_file': '' if args.mbpp_result_file is None else str(args.mbpp_result_file),
        })
    print(args.output_file)


if __name__ == '__main__':
    main()
