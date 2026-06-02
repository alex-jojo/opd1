import argparse
import os
import torch
import json
import copy

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from math_verify import parse, verify


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return string[idx: right_brace_idx + 1]


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left): -1]
    except Exception:
        return None


def apply_chat_template(toker, messages, chat_template=None, enable_thinking=False):
    if chat_template is None:
        input_prompt = toker.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
    else:
        input_prompt = chat_template.format(prompt=messages[0]["content"])

    return input_prompt


def safe_verify_answer(gold_answer, pred_answer):
    if pred_answer is None:
        return False

    try:
        return verify(
            parse("\\boxed{" + str(gold_answer) + "}"),
            parse("\\boxed{" + str(pred_answer) + "}"),
        )
    except Exception:
        return False


def same_answer(ans1, ans2):
    if ans1 is None or ans2 is None:
        return False

    try:
        return verify(
            parse("\\boxed{" + str(ans1) + "}"),
            parse("\\boxed{" + str(ans2) + "}"),
        )
    except Exception:
        return str(ans1).strip() == str(ans2).strip()


def majority_vote_answer(pred_answers):
    """
    对 pred_answers 做多数投票。

    - 会用 math_verify 把数学等价答案归为一类。
    - None 不参与投票。
    - 如果票数相同，选择最早出现的答案。
    """
    groups = []

    for idx, ans in enumerate(pred_answers):
        if ans is None:
            continue

        matched = False

        for group in groups:
            if same_answer(ans, group["rep"]):
                group["count"] += 1
                matched = True
                break

        if not matched:
            groups.append(
                {
                    "rep": ans,
                    "count": 1,
                    "first_idx": idx,
                }
            )

    if len(groups) == 0:
        return None

    groups.sort(key=lambda x: (-x["count"], x["first_idx"]))
    return groups[0]["rep"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--n", type=int, default=8)

    parser.add_argument("--begin_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")

    args = parser.parse_args()

    toker = AutoTokenizer.from_pretrained(args.model_path)

    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path)

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=args.max_num_seqs,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
        seed=args.seed,
    )

    with open(args.input_file, "r", encoding="utf-8") as file:
        input_data = [json.loads(line) for line in file]

    if args.begin_idx >= 0 and args.end_idx >= 0:
        input_data = input_data[args.begin_idx: args.end_idx]

    prompts = []

    for item in input_data:
        problem = item["problem"]
        answer = str(item["answer"])
        item["answer"] = answer

        problem = (
            problem
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )

        prompts.append(problem)

    chat_template = None

    prompt_token_ids = [
        apply_chat_template(
            toker,
            [{"role": "user", "content": prompt}],
            chat_template=chat_template,
            enable_thinking=args.enable_thinking,
        )
        for prompt in prompts
    ]

    generations = llm.generate(prompt_token_ids, sampling_params=sampling_params)

    res_data = []

    for i in range(len(input_data)):
        d = copy.deepcopy(input_data[i])

        responses = []
        boxed_answers = []
        acc_list = []

        for j in range(len(generations[i].outputs)):
            response = generations[i].outputs[j].text.strip()
            responses.append(response)

            boxed_answer = remove_boxed(last_boxed_only_string(response))
            boxed_answers.append(boxed_answer)

            acc = safe_verify_answer(d["answer"], boxed_answer)
            acc_list.append(acc)

        majority_answer = majority_vote_answer(boxed_answers)
        majority_acc = safe_verify_answer(d["answer"], majority_answer)

        d["pred_answers"] = boxed_answers
        d["responses"] = responses
        d["acc_list"] = acc_list
        d["majority_answer"] = majority_answer
        d["majority_acc"] = majority_acc
        d["model"] = args.model_name

        res_data.append(d)

    k = args.n

    total_preds = 0
    correct_preds = 0

    pass_at_k_count = 0
    maj_at_k_count = 0

    avg_length = 0.0

    for d in res_data:
        accs = d.get("acc_list", [])
        responses = d.get("responses", [])

        total_preds += len(accs)
        correct_preds += sum(1 for acc in accs if acc)

        if any(accs):
            pass_at_k_count += 1

        if d.get("majority_acc", False):
            maj_at_k_count += 1

        if len(responses) > 0:
            per_problem_length = 0.0

            for response in responses:
                length = len(toker.encode(response, add_special_tokens=False))
                per_problem_length += length

            per_problem_length /= len(responses)
            avg_length += per_problem_length

    num_problems = len(res_data)

    mean_at_k = correct_preds / total_preds if total_preds > 0 else 0.0
    pass_at_k = pass_at_k_count / num_problems if num_problems > 0 else 0.0
    maj_at_k = maj_at_k_count / num_problems if num_problems > 0 else 0.0
    avg_length = avg_length / num_problems if num_problems > 0 else 0.0

    print(f"dataset: {args.input_file}")
    print(f"model: {args.model_name}")
    print(f"Total problems: {num_problems}")
    print(f"Samples per problem: {k}")
    print(f"Total predictions: {total_preds}")
    print(f"Accurate predictions: {correct_preds}")
    print(f"mean@{k}: {mean_at_k:.4f} ({mean_at_k * 100:.2f}%)")
    print(f"pass@{k}: {pass_at_k:.4f} ({pass_at_k * 100:.2f}%)")
    print(f"maj@{k}: {maj_at_k:.4f} ({maj_at_k * 100:.2f}%)")
    print(f"avg_length: {avg_length:.4f}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as file:
        for d in res_data:
            file.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
