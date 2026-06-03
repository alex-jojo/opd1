#!/usr/bin/env python3
import argparse
import csv
import fnmatch
import json
from pathlib import Path


DATASET_ORDER = {
    "aime24": 0,
    "aime25": 1,
    "aime26": 2,
    "amc23": 3,
    "hmmt26": 4,
    "math500": 5,
}


def pct(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def matches_any(value, patterns):
    if not patterns:
        return True
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def bool_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok"}
    return False


def summarize_jsonl(path, name_source):
    dataset = path.parent.name
    file_model = path.stem
    record_model = None

    n = 0
    total_preds = 0
    correct_preds = 0
    pass_count = 0
    maj_count = 0
    maj_total = 0
    k_values = set()
    bad_lines = 0
    missing_acc = 0
    missing_maj = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            n += 1

            if record_model is None and item.get("model"):
                record_model = str(item["model"])

            acc_list = item.get("acc_list")
            if not isinstance(acc_list, list):
                missing_acc += 1
                continue

            accs = [bool_value(x) for x in acc_list]
            k_values.add(len(accs))
            total_preds += len(accs)
            correct_preds += sum(accs)

            if any(accs):
                pass_count += 1

            if "majority_acc" in item:
                maj_total += 1
                if bool_value(item.get("majority_acc")):
                    maj_count += 1
            else:
                missing_maj += 1

    if name_source == "record" and record_model:
        model = record_model
    else:
        model = file_model

    mean_at_k = correct_preds / total_preds if total_preds else 0.0
    pass_at_k = pass_count / n if n else 0.0
    maj_at_k = maj_count / maj_total if maj_total else None

    if n == 0:
        status = "empty"
    elif bad_lines:
        status = f"bad-json:{bad_lines}"
    elif missing_acc:
        status = f"missing-acc:{missing_acc}"
    elif missing_maj:
        status = f"missing-maj:{missing_maj}"
    elif len(k_values) > 1:
        status = "varying-k"
    else:
        status = "ok"

    if not k_values:
        k = 0
    elif len(k_values) == 1:
        k = next(iter(k_values))
    else:
        k = "/".join(str(x) for x in sorted(k_values))

    return {
        "dataset": dataset,
        "model": model,
        "n": n,
        "k": k,
        "mean_at_k": mean_at_k,
        "pass_at_k": pass_at_k,
        "maj_at_k": maj_at_k,
        "status": status,
        "path": str(path),
    }


def collect_rows(output_dir, datasets, models, name_source):
    rows = []

    for path in output_dir.glob("*/*.jsonl"):
        dataset = path.parent.name
        file_model = path.stem

        if not matches_any(dataset, datasets):
            continue
        if not matches_any(file_model, models):
            continue

        rows.append(summarize_jsonl(path, name_source))

    rows.sort(
        key=lambda row: (
            row["model"],
            DATASET_ORDER.get(row["dataset"], 10_000),
            row["dataset"],
        )
    )
    return rows


def print_table(rows):
    headers = ["dataset", "model", "n", "k", "mean@k", "pass@k", "maj@k", "status"]

    display_rows = [
        [
            row["dataset"],
            row["model"],
            str(row["n"]),
            str(row["k"]),
            pct(row["mean_at_k"]),
            pct(row["pass_at_k"]),
            pct(row["maj_at_k"]),
            row["status"],
        ]
        for row in rows
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in display_rows))
        if display_rows
        else len(headers[i])
        for i in range(len(headers))
    ]

    numeric_cols = {2, 3, 4, 5, 6}

    def fmt(values):
        out = []
        for i, value in enumerate(values):
            if i in numeric_cols:
                out.append(value.rjust(widths[i]))
            else:
                out.append(value.ljust(widths[i]))
        return "  ".join(out)

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in display_rows:
        print(fmt(row))


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "n",
                "k",
                "mean_at_k",
                "pass_at_k",
                "maj_at_k",
                "status",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_output_dir = script_dir / "eval_outputs"

    parser = argparse.ArgumentParser(
        description="Summarize math eval jsonl outputs into mean@k/pass@k/maj@k."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory containing eval_outputs/<dataset>/<model>.jsonl.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset filters. Wildcards are supported, e.g. aime*.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional model filename filters. Wildcards are supported.",
    )
    parser.add_argument(
        "--name-source",
        choices=["file", "record"],
        default="file",
        help="Use jsonl filename or the first record's model field as display name.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path. Defaults to <output-dir>/summary.csv.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Print the table only; do not write summary.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    csv_path = args.csv.resolve() if args.csv else output_dir / "summary.csv"

    rows = collect_rows(output_dir, args.datasets, args.models, args.name_source)

    if rows:
        print_table(rows)
    else:
        print(f"No eval jsonl files found in {output_dir}")

    if not args.no_csv:
        write_csv(rows, csv_path)
        print()
        print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
