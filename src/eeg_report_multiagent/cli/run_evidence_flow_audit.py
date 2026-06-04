from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from eeg_report_multiagent.modules.evidence_flow_auditor import CLINICAL_SLOT_SPECS, EvidenceFlowAuditor
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.evidence_flow import EvidenceFlowAggregate, EvidenceFlowAuditResult, SlotFlowRecord
from eeg_report_multiagent.schemas.report import AtomicClaimPlan


ANCHOR_CASE_IDS = [
    "row_000189_NeuroReport_13450252291_593274052_20180914",
    "row_000548_NeuroReport_13341375710_527779290_20190820",
    "row_000783_NeuroReport_13324097354_322634096_20180129",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _load_case(row_dir: Path, variant: str) -> EvidenceFlowAuditResult:
    evidence_board = EvidenceBoard.model_validate_json((row_dir / "evidence_board.json").read_text(encoding="utf-8"))
    shared_board = evidence_board.ensure_shared_evidence_board()
    claims = [AtomicClaimPlan.model_validate(item) for item in _read_json(row_dir / "atomic_claim_plan.json")]
    if (row_dir / "celm_section_texts.json").exists():
        final_report = _read_json(row_dir / "celm_section_texts.json")
    else:
        final_report = _read_json(row_dir / "celm_generated_report.json")
    return EvidenceFlowAuditor().audit_case(
        case_id=row_dir.name,
        variant=variant,
        evidence_board=evidence_board,
        shared_evidence_board=shared_board,
        atomic_claims=claims,
        final_report=final_report,
    )


def _record_row(record: SlotFlowRecord) -> Dict[str, Any]:
    return {
        "case_id": record.case_id,
        "clinical_slot": record.clinical_slot,
        "section_name": record.section_name,
        "measurement_exists": record.measurement_exists,
        "evidence_item_exists": record.evidence_item_exists,
        "evidence_ids": "|".join(record.evidence_ids),
        "evidence_type_counts": json.dumps(record.evidence_type_counts, sort_keys=True),
        "reportability_counts": json.dumps(record.reportability_counts, sort_keys=True),
        "atomic_claim_exists": record.atomic_claim_exists,
        "atomic_claim_ids": "|".join(record.atomic_claim_ids),
        "surface_action_counts": json.dumps(record.surface_action_counts, sort_keys=True),
        "surfaced_in_final_prose": record.surfaced_in_final_prose,
        "suppression_reasons": "|".join(record.suppression_reasons),
        "useful_but_suppressed": record.useful_but_suppressed,
        "notes": record.notes or "",
    }


def _aggregate_rows(aggregate: EvidenceFlowAggregate, audits: Iterable[EvidenceFlowAuditResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reason_by_slot = aggregate.per_slot_suppression_reason_counts
    for spec in CLINICAL_SLOT_SPECS:
        stats = aggregate.per_slot_availability.get(spec.name, {})
        reasons = reason_by_slot.get(spec.name, {})
        top_reason = max(reasons.items(), key=lambda item: item[1])[0] if reasons else ""
        rows.append({
            "clinical_slot": spec.name,
            "measurement_rate": stats.get("measurement_rate", 0.0),
            "evidence_item_rate": stats.get("evidence_item_rate", 0.0),
            "claim_rate": stats.get("claim_rate", 0.0),
            "surface_rate": stats.get("surface_rate", 0.0),
            "useful_suppressed_rate": stats.get("useful_suppressed_rate", 0.0),
            "top_suppression_reason": top_reason,
            "recommendation": _slot_recommendation(stats, top_reason),
        })
    del audits
    return rows


def _slot_recommendation(stats: Dict[str, float], top_reason: str) -> str:
    if stats.get("measurement_rate", 0.0) == 0:
        return "Stage 3A: improve evidence extraction for this slot."
    if stats.get("evidence_item_rate", 0.0) < stats.get("measurement_rate", 0.0) * 0.75:
        return "Stage 3B: repair Measurement to EvidenceItem conversion."
    if "no_atomic_claim_generated" in top_reason:
        return "Stage 3D: refine AtomicClaimPlan generation."
    if "atomic_claim_blocked" in top_reason or "surface_policy_rejected" in top_reason:
        return "Stage 3C/E: calibrate reportability and SurfacePolicy with evidence weighting."
    if stats.get("surface_rate", 0.0) == 0 and stats.get("claim_rate", 0.0) > 0:
        return "Stage 3E/F: inspect gate plus section rendering."
    return "Monitor; no immediate slot-specific repair identified."


def _render_case_markdown(audit: EvidenceFlowAuditResult) -> str:
    lines = [f"# {audit.case_id} Evidence Flow Audit", "", f"- variant: `{audit.variant}`", f"- diagnosis: {audit.case_diagnosis}", ""]
    lines.append("## Slot Flow Table")
    headers = [
        "slot", "measurement", "evidence", "claim", "surface",
        "evidence_type_counts", "reportability_counts", "surface_action_counts", "suppression_reasons", "useful",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for record in audit.slot_records:
        lines.append("| " + " | ".join([
            record.clinical_slot,
            str(record.measurement_exists),
            str(record.evidence_item_exists),
            str(record.atomic_claim_exists),
            str(record.surfaced_in_final_prose),
            _compact(record.evidence_type_counts),
            _compact(record.reportability_counts),
            _compact(record.surface_action_counts),
            ", ".join(record.suppression_reasons),
            str(record.useful_but_suppressed),
        ]) + " |")
    lines.extend(["", "## Surfaced Slots", ""])
    lines.extend([f"- `{slot}`" for slot in audit.surfaced_slots] or ["- None"])
    lines.extend(["", "## Top Suppression Reasons", ""])
    for reason, count in sorted(audit.suppression_reason_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Useful But Suppressed Evidence", ""])
    for item in audit.useful_suppressed_evidence[:20]:
        lines.append(f"- `{item['clinical_slot']}` evidence={item['evidence_ids']} reasons={item['suppression_reasons']} note={item.get('notes','')}")
    if not audit.useful_suppressed_evidence:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _compact(mapping: Dict[str, int]) -> str:
    if not mapping:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in sorted(mapping.items()))


def _render_useful_examples(audits: Iterable[EvidenceFlowAuditResult]) -> str:
    lines = ["# Useful But Suppressed Examples", ""]
    idx = 1
    for audit in audits:
        for item in audit.useful_suppressed_evidence:
            lines.append(f"## {idx}. {audit.case_id} / {item['clinical_slot']}")
            lines.append(f"- evidence_ids: `{', '.join(item['evidence_ids'])}`")
            lines.append(f"- suppression_reasons: `{', '.join(item['suppression_reasons'])}`")
            lines.append(f"- notes: {item.get('notes', '')}")
            lines.append("")
            idx += 1
            if idx > 21:
                return "\n".join(lines)
    if idx == 1:
        lines.append("No useful-but-suppressed examples were detected by the current heuristic.")
    return "\n".join(lines)


def _render_stage3_recommendation(aggregate: EvidenceFlowAggregate, aggregate_rows: List[Dict[str, Any]]) -> str:
    lines = ["# Stage 3 Recommendation", "", f"Primary recommendation: **{aggregate.aggregate_recommendation_for_stage3}**", ""]
    lines.append("## Slot Summary")
    lines.append("| clinical_slot | measurement_rate | evidence_item_rate | claim_rate | surface_rate | top_suppression_reason | recommendation |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in aggregate_rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["clinical_slot"]),
                f"{float(row['measurement_rate']):.2f}",
                f"{float(row['evidence_item_rate']):.2f}",
                f"{float(row['claim_rate']):.2f}",
                f"{float(row['surface_rate']):.2f}",
                str(row["top_suppression_reason"]),
                str(row["recommendation"]),
            ])
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    surface_rates = [float(row["surface_rate"]) for row in aggregate_rows]
    useful_rates = [float(row["useful_suppressed_rate"]) for row in aggregate_rows]
    lines.append(f"- Mean slot surface rate: `{mean(surface_rates):.3f}`")
    lines.append(f"- Mean useful-suppressed rate: `{mean(useful_rates):.3f}`")
    lines.append("- Safety gates remain intact; this audit is diagnostic and should not be used to bypass SurfacePolicy.")
    return "\n".join(lines) + "\n"


def run_evidence_flow_audit(rows_root: Path, output_dir: Path, variant: str, limit: int | None = None) -> Dict[str, Any]:
    case_dirs = sorted(path for path in rows_root.iterdir() if path.is_dir() and (path / "evidence_board.json").exists())
    if limit is not None:
        case_dirs = case_dirs[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir = output_dir / "per_case"
    if per_case_dir.exists():
        shutil.rmtree(per_case_dir)
    per_case_dir.mkdir(parents=True, exist_ok=True)

    audits = [_load_case(path, variant) for path in case_dirs]
    auditor = EvidenceFlowAuditor()
    aggregate = auditor.aggregate_selected50(audits, variant=variant)
    aggregate_rows = _aggregate_rows(aggregate, audits)

    flat_rows: List[Dict[str, Any]] = []
    for audit in audits:
        _write_json(per_case_dir / f"{audit.case_id}_evidence_flow.json", audit)
        (per_case_dir / f"{audit.case_id}_evidence_flow.md").write_text(_render_case_markdown(audit), encoding="utf-8")
        for record in audit.slot_records:
            flat_rows.append(_record_row(record))
        if audit.case_id in ANCHOR_CASE_IDS:
            suffix = audit.case_id.split("_", 2)[1]
            (output_dir / f"row{int(suffix):03d}_gate_loss_diagnosis.md").write_text(_render_case_markdown(audit), encoding="utf-8")

    _write_csv(output_dir / "selected50_evidence_flow_records.csv", flat_rows)
    _write_csv(output_dir / "selected50_evidence_flow_aggregate.csv", aggregate_rows)
    _write_json(output_dir / "selected50_evidence_flow_aggregate.json", aggregate)
    reason_counter = Counter(reason for audit in audits for reason, count in audit.suppression_reason_counts.items() for _ in range(count))
    _write_csv(output_dir / "suppression_reason_counts.csv", [{"suppression_reason": reason, "count": count} for reason, count in sorted(reason_counter.items(), key=lambda item: (-item[1], item[0]))])
    (output_dir / "useful_but_suppressed_examples.md").write_text(_render_useful_examples(audits), encoding="utf-8")
    (output_dir / "stage3_recommendation.md").write_text(_render_stage3_recommendation(aggregate, aggregate_rows), encoding="utf-8")
    return {
        "rows_root": str(rows_root),
        "output_dir": str(output_dir),
        "case_count": len(audits),
        "aggregate_csv": str(output_dir / "selected50_evidence_flow_aggregate.csv"),
        "stage3_recommendation": aggregate.aggregate_recommendation_for_stage3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2.75 evidence availability and gate-loss audit.")
    parser.add_argument("--rows-root", default="artifacts/batch_s0001_test_Our_EvidenceGated_v1_selected50/rows")
    parser.add_argument("--output-dir", default="artifacts/stage2_75_evidence_flow_audit")
    parser.add_argument("--variant", default="Our_EvidenceGated_v1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_evidence_flow_audit(
        rows_root=Path(args.rows_root),
        output_dir=Path(args.output_dir),
        variant=args.variant,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
