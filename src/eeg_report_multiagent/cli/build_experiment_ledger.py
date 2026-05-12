from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from eeg_report_multiagent.io import read_split_rows, report_id_from_row


CELM_SCORE_COLUMNS = [
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
    if not path.exists():
        return []
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


def _float_or_zero(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _index_celm_scores(celm_results_dir: Path | None) -> Dict[str, Dict[str, str]]:
    if celm_results_dir is None:
        return {}
    rows = _read_csv_dicts(celm_results_dir / "overall_scores.csv")
    return {row.get("deidentified_name", ""): row for row in rows if row.get("deidentified_name")}


def _index_our_batch(our_batch_root: Path | None) -> Dict[str, Dict[str, str]]:
    if our_batch_root is None:
        return {}
    rows = _read_csv_dicts(our_batch_root / "batch_summary.csv")
    return {row.get("report_id", ""): row for row in rows if row.get("report_id")}


def build_ledger(
    data_root: Path,
    site: str,
    split_type: str,
    split: str,
    celm_results_dir: Path | None,
    our_batch_root: Path | None,
) -> List[Dict[str, Any]]:
    split_rows = read_split_rows(data_root=data_root, site=site, split=split, split_type=split_type)
    celm_scores = _index_celm_scores(celm_results_dir)
    our_scores = _index_celm_scores(our_batch_root / "celm_results" if our_batch_root else None)
    our_rows = _index_our_batch(our_batch_root)
    celm_generated_dir = celm_results_dir / "generated_reports_json" if celm_results_dir else None
    our_generated_dir = our_batch_root / "celm_results" / "generated_reports_json" if our_batch_root else None

    ledger: List[Dict[str, Any]] = []
    for row_index, row in enumerate(split_rows):
        report_id = report_id_from_row(row)
        celm_score = celm_scores.get(report_id, {})
        our_score = our_scores.get(report_id, {})
        our_row = our_rows.get(report_id, {})
        celm_generated_exists = bool(
            celm_generated_dir and (celm_generated_dir / f"GENERATED_REPORT_{report_id}.json").exists()
        )
        our_generated_exists = bool(
            our_generated_dir and (our_generated_dir / f"GENERATED_REPORT_{report_id}.json").exists()
        )
        celm_nonzero = any(_float_or_zero(celm_score.get(col)) > 0 for col in ["bleu-1", "rouge1", "rougeL", "meteor"])
        our_nonzero = any(_float_or_zero(our_score.get(col)) > 0 for col in ["bleu-1", "rouge1", "rougeL", "meteor"])

        item: Dict[str, Any] = {
            "row_index": row_index,
            "report_id": report_id,
            "patient_id": row.get("BDSPPatientID", ""),
            "split": split,
            "visit_type": row.get("VisitTypeDSC", ""),
            "number_of_sessions": row.get("NumberOfSessions", ""),
            "processed_eeg_paths": row.get("Processed_EEG_Paths", ""),
            "target_eeg_sections": row.get("Extracted_EEG_sections", ""),
            "celm_generated_exists": _bool_str(celm_generated_exists),
            "celm_score_row_exists": _bool_str(bool(celm_score)),
            "celm_nonzero_text_metric": _bool_str(celm_nonzero),
            "our_B_status": our_row.get("status", "not_started"),
            "our_B_artifact_dir": our_row.get("artifact_dir", ""),
            "our_B_generated_exists": _bool_str(our_generated_exists),
            "our_B_score_row_exists": _bool_str(bool(our_score)),
            "our_B_nonzero_text_metric": _bool_str(our_nonzero),
            "our_B_llm_review_status": our_row.get("llm_review_status", ""),
            "our_B_weak_evidence_records": our_row.get("weak_evidence_records", ""),
            "our_B_missing_slot_records": our_row.get("missing_slot_records", ""),
            "our_B_do_not_claim_records": our_row.get("do_not_claim_records", ""),
            "our_B_claim_constraint_records": our_row.get("claim_constraint_records", ""),
            "our_B_overall_pass": our_row.get("overall_pass", ""),
            "our_B_input_contract_pass": our_row.get("input_contract_pass", ""),
            "our_B_error": our_row.get("error_message", ""),
        }
        for col in CELM_SCORE_COLUMNS:
            item[f"celm_{col}"] = celm_score.get(col, "")
        for col in CELM_SCORE_COLUMNS:
            item[f"our_B_{col}"] = our_score.get(col, "")
        ledger.append(item)
    return ledger


def summarize_ledger(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "celm_generated_exists": sum(1 for r in rows if r.get("celm_generated_exists") == "true"),
        "celm_nonzero_text_metric": sum(1 for r in rows if r.get("celm_nonzero_text_metric") == "true"),
        "our_B_ok": sum(1 for r in rows if r.get("our_B_status") in {"ok", "skipped_existing"}),
        "our_B_error": sum(1 for r in rows if r.get("our_B_status") == "error"),
        "our_B_not_started": sum(1 for r in rows if r.get("our_B_status") in {"", "not_started"}),
        "our_B_score_row_exists": sum(1 for r in rows if r.get("our_B_score_row_exists") == "true"),
        "our_B_nonzero_text_metric": sum(1 for r in rows if r.get("our_B_nonzero_text_metric") == "true"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an experiment ledger by joining split CSV, CELM scores, and our batch outputs")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--celm-results-dir", default=None, help="Directory containing CELM overall_scores.csv and generated_reports_json/")
    parser.add_argument("--our-batch-root", default=None, help="Our batch root containing batch_summary.csv and celm_results/")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows = build_ledger(
        data_root=Path(args.data_root),
        site=args.site,
        split_type=args.split_type,
        split=args.split,
        celm_results_dir=Path(args.celm_results_dir) if args.celm_results_dir else None,
        our_batch_root=Path(args.our_batch_root) if args.our_batch_root else None,
    )
    fieldnames = list(rows[0].keys()) if rows else []
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json) if args.output_json else output_csv.with_suffix(".json")
    summary = summarize_ledger(rows)
    _write_csv(output_csv, rows, fieldnames)
    _write_json(output_json, {"summary": summary, "rows": rows})
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
