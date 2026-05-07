#!/usr/bin/env python3
"""
Compute additional When2Call metrics from LM-Eval-Harness sample logs.

This mirrors the NVIDIA When2Call additional metrics script while avoiding
heavy dependencies (sklearn/pandas), so it can run in constrained environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i} in {path}") from exc
    if not rows:
        raise ValueError(f"No JSON rows found in {path}")
    return rows


def _find_samples_jsonl(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    candidates = sorted(input_path.rglob("samples*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No samples*.jsonl files found under {input_path}. "
            "Pass --samples_path directly if your file has a custom name."
        )
    return candidates[-1]


def _infer_label_names(first_row: dict[str, Any]) -> list[str]:
    answers = first_row.get("doc", {}).get("answers", {})
    if isinstance(answers, dict) and answers:
        return [str(k) for k in answers.keys()]
    return []


def _infer_tool_call_index(label_names: list[str]) -> int:
    for i, label in enumerate(label_names):
        l = label.lower()
        if "tool" in l and "call" in l:
            return i
    # NVIDIA script assumes index 1 is "tool call".
    return 1


def _extract_target_index(row: dict[str, Any]) -> int | None:
    # Format used by some LM-Eval tasks.
    macro_f1 = row.get("macro_f1", [])
    if isinstance(macro_f1, list) and len(macro_f1) >= 2:
        try:
            return int(macro_f1[0])
        except (TypeError, ValueError):
            return None

    # Common MCQ sample format.
    if "target_index" in row:
        try:
            return int(row["target_index"])
        except (TypeError, ValueError):
            pass

    if "target" in row:
        try:
            return int(row["target"])
        except (TypeError, ValueError):
            return None
    return None


def _extract_pred_index(row: dict[str, Any]) -> int | None:
    # Format used by some LM-Eval tasks.
    macro_f1 = row.get("macro_f1", [])
    if isinstance(macro_f1, list) and len(macro_f1) >= 2:
        try:
            return int(macro_f1[1])
        except (TypeError, ValueError):
            return None

    # MCQ format: choose highest log-likelihood among filtered responses.
    filtered = row.get("filtered_resps")
    if not isinstance(filtered, list) or not filtered:
        return None

    best_idx: int | None = None
    best_score: float | None = None
    for i, item in enumerate(filtered):
        if not isinstance(item, list) or not item:
            continue
        try:
            score = float(item[0])
        except (TypeError, ValueError):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _collect_gold_pred(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    gold: list[int] = []
    pred: list[int] = []
    for row in rows:
        g = _extract_target_index(row)
        p = _extract_pred_index(row)
        if g is None or p is None:
            continue
        gold.append(g)
        pred.append(p)
    if not gold:
        raise ValueError("No valid (gold, predicted) pairs found in sample rows.")
    return gold, pred


def _confusion_matrix(gold: list[int], pred: list[int], num_labels: int) -> list[list[int]]:
    m = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for g, p in zip(gold, pred):
        if 0 <= g < num_labels and 0 <= p < num_labels:
            m[g][p] += 1
    return m


def _labeled_confusion_dict(matrix: list[list[int]], label_names: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for g_idx, row in enumerate(matrix):
        g_label = label_names[g_idx]
        out[f"true:{g_label}"] = {f"pred:{label_names[p_idx]}": int(v) for p_idx, v in enumerate(row)}
    return out


def _format_confusion_matrix(matrix: list[list[int]], label_names: list[str]) -> str:
    headers = [f"pred:{name}" for name in label_names]
    row_labels = [f"true:{name}" for name in label_names]

    col_widths = [max(len(headers[i]), 6) for i in range(len(headers))]
    for col_idx in range(len(headers)):
        for row in matrix:
            col_widths[col_idx] = max(col_widths[col_idx], len(str(row[col_idx])))

    row_label_width = max(max(len(x) for x in row_labels), len(""), 8)
    header_line = " " * (row_label_width + 2) + "  ".join(
        h.ljust(col_widths[i]) for i, h in enumerate(headers)
    )
    body_lines = []
    for row_name, row in zip(row_labels, matrix):
        vals = "  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
        body_lines.append(f"{row_name.ljust(row_label_width)}  {vals}")
    return "\n".join([header_line, *body_lines])


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _compute_classification_metrics(
    matrix: list[list[int]], label_names: list[str]
) -> dict[str, Any]:
    num_labels = len(label_names)
    total = sum(sum(row) for row in matrix)
    diag_sum = sum(matrix[i][i] for i in range(num_labels))

    per_category: dict[str, dict[str, float | int]] = {}
    macro_f1 = 0.0
    macro_acc = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i, label in enumerate(label_names):
        tp = matrix[i][i]
        fn = sum(matrix[i][j] for j in range(num_labels) if j != i)
        fp = sum(matrix[j][i] for j in range(num_labels) if j != i)
        tn = total - tp - fp - fn

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        # One-vs-rest accuracy for this category.
        acc = _safe_div(tp + tn, total)
        support = tp + fn

        per_category[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": acc,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        macro_f1 += f1
        macro_acc += acc
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = _safe_div(macro_f1, num_labels)
    macro_acc = _safe_div(macro_acc, num_labels)

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2.0 * micro_precision * micro_recall, micro_precision + micro_recall)
    micro_acc = _safe_div(diag_sum, total)

    return {
        "per_category": per_category,
        "macro_avg": {
            "f1": macro_f1,
            "acc": macro_acc,
        },
        "micro_avg": {
            "f1": micro_f1,
            "acc": micro_acc,
        },
        "overall_accuracy": micro_acc,
        "num_samples": total,
    }


def _format_per_category_metrics(per_category: dict[str, dict[str, float | int]]) -> str:
    headers = ["label", "f1", "acc", "precision", "recall", "support"]
    rows = []
    for label, m in per_category.items():
        rows.append(
            [
                label,
                f"{float(m['f1']):.6f}",
                f"{float(m['acc']):.6f}",
                f"{float(m['precision']):.6f}",
                f"{float(m['recall']):.6f}",
                str(int(m["support"])),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            col_widths[i] = max(col_widths[i], len(value))

    out = ["  ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))]
    for row in rows:
        out.append("  ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def calculate_hallucination_rate(rows: list[dict[str, Any]], tool_call_index: int) -> float:
    filtered = []
    for row in rows:
        doc = row.get("doc", {})
        correct_answer = doc.get("correct_answer")
        tools = doc.get("tools", [])
        if correct_answer == "cannot_answer" and isinstance(tools, list) and len(tools) == 0:
            filtered.append(row)

    if not filtered:
        return 0.0

    hallucinated = 0
    for row in filtered:
        predicted_index = _extract_pred_index(row)
        if predicted_index is None:
            continue
        if predicted_index == tool_call_index:
            hallucinated += 1

    return hallucinated / len(filtered)


def compute_additional_metrics(samples_input: Path | str) -> dict[str, Any]:
    samples_path = _find_samples_jsonl(Path(samples_input).expanduser().resolve())
    rows = _read_jsonl(samples_path)

    label_names = _infer_label_names(rows[0])
    gold, pred = _collect_gold_pred(rows)
    max_index = max(max(gold), max(pred))

    if not label_names:
        label_names = [str(i) for i in range(max_index + 1)]
    elif len(label_names) <= max_index:
        for i in range(len(label_names), max_index + 1):
            label_names.append(str(i))

    matrix = _confusion_matrix(gold, pred, num_labels=len(label_names))
    labelled_confusion = _labeled_confusion_dict(matrix, label_names)
    classification_metrics = _compute_classification_metrics(matrix, label_names)
    tool_call_index = _infer_tool_call_index(label_names)
    hallucination_rate = calculate_hallucination_rate(rows, tool_call_index=tool_call_index)

    return {
        "samples_path": str(samples_path),
        "tool_call_index": tool_call_index,
        "hallucination_rate": hallucination_rate,
        "confusion_matrix": labelled_confusion,
        "classification_metrics": classification_metrics,
        "_debug": {
            "matrix": matrix,
            "label_names": label_names,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute additional When2Call metrics from LM-Eval-Harness sample logs "
            "(hallucination rate + confusion matrix)."
        )
    )
    parser.add_argument(
        "--samples_path",
        type=str,
        required=True,
        help="Path to samples JSONL file, or an output directory containing samples*.jsonl.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save computed metrics as JSON.",
    )
    args = parser.parse_args()

    result = compute_additional_metrics(args.samples_path)
    samples_path = Path(result["samples_path"])
    matrix = result["_debug"]["matrix"]
    label_names = result["_debug"]["label_names"]
    classification_metrics = result["classification_metrics"]
    tool_call_index = int(result["tool_call_index"])
    hallucination_rate = float(result["hallucination_rate"])

    print(f"Samples: {samples_path}")
    print(f"Tool-call index used for hallucination rate: {tool_call_index}")
    print(f"Hallucination rate: {hallucination_rate:.6f}")
    print("Confusion matrix:")
    print(_format_confusion_matrix(matrix, label_names))
    print("Per-category metrics:")
    print(_format_per_category_metrics(classification_metrics["per_category"]))
    print(
        "Averages: "
        f"macro_f1={classification_metrics['macro_avg']['f1']:.6f}, "
        f"macro_acc={classification_metrics['macro_avg']['acc']:.6f}, "
        f"micro_f1={classification_metrics['micro_avg']['f1']:.6f}, "
        f"micro_acc={classification_metrics['micro_avg']['acc']:.6f}"
    )
    print("\nJSON:")
    result_for_print = {k: v for k, v in result.items() if k != "_debug"}
    print(json.dumps(result_for_print, indent=2))

    if args.save_json:
        out = Path(args.save_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result_for_print, indent=2), encoding="utf-8")
        print(f"\nSaved metrics JSON to: {out}")


if __name__ == "__main__":
    main()
