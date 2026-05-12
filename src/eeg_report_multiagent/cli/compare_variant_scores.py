from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from eeg_report_multiagent.io import read_split_rows, report_id_from_row


SCORE_COLUMNS = [
    "bleu-1",
    "bleu-4",
    "bleu-1-smooth",
    "bleu-4-smooth",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "perplexity",
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "meteor",
]


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_row_indices(path: Path | None, total_rows: int) -> List[int]:
    if path is None:
        return list(range(total_rows))
    out: List[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(int(line))
    return out


def _parse_variant(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise ValueError("Variant must use NAME=/path/to/results_dir syntax")
    name, path = text.split("=", 1)
    if not name.strip():
        raise ValueError("Variant name is empty")
    return name.strip(), Path(path)


def _index_scores(results_dir: Path) -> Dict[str, Dict[str, str]]:
    rows = _read_csv_dicts(results_dir / "overall_scores.csv")
    return {row.get("deidentified_name", ""): row for row in rows if row.get("deidentified_name")}


def compare_variant_scores(
    data_root: Path,
    site: str,
    split_type: str,
    split: str,
    row_indices_file: Path | None,
    variants: List[Tuple[str, Path]],
) -> Dict[str, Any]:
    split_rows = read_split_rows(data_root=data_root, site=site, split=split, split_type=split_type)
    selected_indices = _read_row_indices(row_indices_file, len(split_rows))
    score_indices = {name: _index_scores(path) for name, path in variants}

    rows: List[Dict[str, Any]] = []
    for row_index in selected_indices:
        split_row = split_rows[row_index]
        report_id = report_id_from_row(split_row)
        item: Dict[str, Any] = {
            "row_index": row_index,
            "report_id": report_id,
            "patient_id": split_row.get("BDSPPatientID", ""),
            "target_eeg_sections": split_row.get("Extracted_EEG_sections", ""),
        }
        for name, _ in variants:
            score = score_indices[name].get(report_id, {})
            item[f"{name}_score_row_exists"] = "true" if score else "false"
            item[f"{name}_nonzero_text_metric"] = "true" if any(
                _float_or_zero(score.get(col)) > 0.0 for col in ["bleu-1", "rouge1", "rougeL", "meteor"]
            ) else "false"
            for col in SCORE_COLUMNS:
                item[f"{name}_{col}"] = score.get(col, "")
        rows.append(item)

    summary: Dict[str, Any] = {
        "rows": len(rows),
        "row_indices_file": str(row_indices_file) if row_indices_file else None,
        "variants": {},
    }
    for name, _ in variants:
        metric_means = {}
        for col in SCORE_COLUMNS:
            values = [_float_or_zero(row.get(f"{name}_{col}")) for row in rows if row.get(f"{name}_score_row_exists") == "true"]
            metric_means[col] = mean(values) if values else 0.0
        summary["variants"][name] = {
            "score_row_exists": sum(1 for row in rows if row.get(f"{name}_score_row_exists") == "true"),
            "nonzero_text_metric": sum(1 for row in rows if row.get(f"{name}_nonzero_text_metric") == "true"),
            "means": metric_means,
        }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CELM-style score CSVs for selected split rows.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--row-indices-file", default=None)
    parser.add_argument("--variant", action="append", required=True, help="NAME=/path/to/results_dir containing overall_scores.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    result = compare_variant_scores(
        data_root=Path(args.data_root),
        site=args.site,
        split_type=args.split_type,
        split=args.split,
        row_indices_file=Path(args.row_indices_file) if args.row_indices_file else None,
        variants=[_parse_variant(item) for item in args.variant],
    )
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json) if args.output_json else output_csv.with_suffix(".json")
    rows = result["rows"]
    _write_csv(output_csv, rows, list(rows[0].keys()) if rows else [])
    _write_json(output_json, result)
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json), "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
