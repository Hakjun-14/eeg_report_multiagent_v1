from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _case_key(row: Dict[str, str]) -> str:
    return str(row.get("case_id") or f"row_{int(row.get('row_index', 0)):06d}_{row.get('report_id', '')}")


def _select_top(
    selected: Dict[str, str],
    rows: List[Dict[str, Any]],
    criterion: str,
    key: str,
    limit: int,
    min_value: float | None = None,
) -> None:
    for row in sorted(rows, key=lambda r: _as_float(r.get(key)), reverse=True):
        if min_value is not None and _as_float(row.get(key)) < min_value:
            continue
        if len(selected) >= limit:
            return
        case_id = str(row["case_id"])
        selected.setdefault(case_id, criterion)
        return


def build_human_review_subset(
    audit_dir: Path,
    comparison_root: Path,
    output_dir: Path,
    max_cases: int,
    forced_rows: List[int],
) -> Dict[str, Any]:
    case_rows = _read_csv(audit_dir / "clinical_audit_case_summary.csv")
    card_rows = _read_csv(audit_dir / "clinical_audit_claim_cards.csv")

    by_case_variant: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    by_case_cards: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        by_case_variant[_case_key(row)][str(row.get("variant"))] = row
    for row in card_rows:
        by_case_cards[str(row.get("case_id"))].append(row)

    candidates: List[Dict[str, Any]] = []
    for case_id, variants in by_case_variant.items():
        any_row = next(iter(variants.values()))
        cards = by_case_cards.get(case_id, [])
        decisions = defaultdict(int)
        slots = defaultdict(int)
        for card in cards:
            decisions[str(card.get("decision"))] += 1
            slots[str(card.get("slot"))] += 1
        celm = variants.get("CELM", {})
        old = variants.get("Our_Upgrade_LLMProp", {})
        filtered = variants.get("Our_Upgrade_Filtered", {})
        candidate = {
            "case_id": case_id,
            "row_index": _as_int(any_row.get("row_index")),
            "report_id": any_row.get("report_id", ""),
            "patient_id": any_row.get("patient_id", ""),
            "celm_rougeL": _as_float(celm.get("rougeL")),
            "celm_concept_f1": _as_float(celm.get("concept_f1_mean")),
            "celm_critical_cards": _as_int(celm.get("critical_cards")),
            "old_debug_leakage": _as_int(old.get("debug_leakage")),
            "filtered_debug_leakage": _as_int(filtered.get("debug_leakage")),
            "filtered_critical_cards": _as_int(filtered.get("critical_cards")),
            "filtered_major_cards": _as_int(filtered.get("major_cards")),
            "filtered_false_negative": _as_int(filtered.get("over_cautious_false_negative")),
            "filtered_under_specified": _as_int(filtered.get("under_specified")),
            "filtered_concept_f1": _as_float(filtered.get("concept_f1_mean")),
            "filtered_rougeL": _as_float(filtered.get("rougeL")),
            "possible_leakage_cards": decisions.get("possible_leakage_or_memorization", 0),
            "numeric_cards": slots.get("numeric_quantitation", 0),
            "debug_surface_cards": slots.get("debug_surface_separation", 0),
            "pdr_cards": slots.get("pdr_frequency", 0),
            "morphology_cards": slots.get("epileptiform_morphology", 0),
            "localization_cards": slots.get("localization_laterality", 0),
        }
        candidate["review_priority_score"] = (
            4 * candidate["possible_leakage_cards"]
            + 3 * candidate["filtered_critical_cards"]
            + 2 * candidate["filtered_major_cards"]
            + candidate["old_debug_leakage"]
            + candidate["numeric_cards"]
            + candidate["morphology_cards"]
            + candidate["localization_cards"]
        )
        candidates.append(candidate)

    selected: Dict[str, str] = {}
    for row_index in forced_rows:
        for row in candidates:
            if row["row_index"] == row_index:
                selected.setdefault(row["case_id"], "forced_anchor_case")
                break

    criteria = [
        ("possible_leakage_or_memorization", "possible_leakage_cards", 1.0),
        ("highest_filtered_critical_burden", "filtered_critical_cards", 1.0),
        ("highest_old_debug_leakage_pre_filter", "old_debug_leakage", 1.0),
        ("largest_numeric_provenance_issue", "numeric_cards", 1.0),
        ("largest_morphology_failure_burden", "morphology_cards", 1.0),
        ("largest_localization_failure_burden", "localization_cards", 1.0),
        ("highest_celm_metric_with_errors", "celm_rougeL", 0.8),
        ("lowest_filtered_concept_f1", "negative_filtered_concept_f1", None),
    ]
    for row in candidates:
        row["negative_filtered_concept_f1"] = -float(row["filtered_concept_f1"])
    while len(selected) < max_cases:
        before = len(selected)
        for criterion, key, min_value in criteria:
            if len(selected) >= max_cases:
                break
            _select_top(
                selected,
                [r for r in candidates if r["case_id"] not in selected],
                criterion,
                key,
                max_cases,
                min_value=min_value,
            )
        if len(selected) == before:
            break

    selected_rows = []
    for row in sorted(candidates, key=lambda r: (r["row_index"], r["report_id"])):
        if row["case_id"] not in selected:
            continue
        row = dict(row)
        row["selection_reason"] = selected[row["case_id"]]
        row["audit_markdown_path"] = str(audit_dir / "per_case_markdown" / f"{row['case_id']}.md")
        row["comparison_json_path"] = str(comparison_root / "per_case_json" / f"{row['case_id']}.json")
        selected_rows.append(row)

    _write_csv(output_dir / "human_review_subset.csv", selected_rows)
    _write_json(output_dir / "human_review_subset.json", selected_rows)

    packet_lines = [
        "# Human Review Subset",
        "",
        "Purpose: pre-specified clinical review subset for provenance-aware model debugging.",
        "GT/reference text is evaluation-only; reviewers should judge clinical correctness and traceability, not just lexical metrics.",
        "",
        "| Row | Report | Reason | CELM RougeL | Filtered F1 | Old Debug | Filtered Critical | Key slots |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    review_cases_dir = output_dir / "review_cases"
    review_cases_dir.mkdir(parents=True, exist_ok=True)
    for row in selected_rows:
        key_slots = ", ".join(
            slot
            for slot in ["pdr", "morphology", "localization", "numeric"]
            if row.get(f"{slot}_cards" if slot != "numeric" else "numeric_cards", 0)
        )
        packet_lines.append(
            f"| {row['row_index']} | {row['report_id']} | {row['selection_reason']} | "
            f"{row['celm_rougeL']:.3f} | {row['filtered_concept_f1']:.3f} | "
            f"{row['old_debug_leakage']} | {row['filtered_critical_cards']} | {key_slots or 'mixed'} |"
        )
        src = Path(row["audit_markdown_path"])
        if src.exists():
            shutil.copy2(src, review_cases_dir / src.name)
    (output_dir / "README.md").write_text("\n".join(packet_lines) + "\n", encoding="utf-8")

    return {
        "audit_dir": str(audit_dir),
        "comparison_root": str(comparison_root),
        "output_dir": str(output_dir),
        "selected_cases": len(selected_rows),
        "forced_rows": forced_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a balanced human-review subset from clinical provenance audit artifacts.")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--forced-row", action="append", type=int, default=[])
    args = parser.parse_args()
    summary = build_human_review_subset(
        audit_dir=Path(args.audit_dir),
        comparison_root=Path(args.comparison_root),
        output_dir=Path(args.output_dir),
        max_cases=args.max_cases,
        forced_rows=args.forced_row,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
