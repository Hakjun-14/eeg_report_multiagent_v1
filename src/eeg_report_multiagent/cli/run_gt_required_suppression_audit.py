from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from eeg_report_multiagent.modules.gt_required_suppression_auditor import GTRequiredSuppressionAuditor
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.gt_suppression import GTSuppressionAuditResult
from eeg_report_multiagent.schemas.report import AtomicClaimPlan
from eeg_report_multiagent.schemas.shared_evidence import SharedEvidenceBoard

ANCHOR_CASE_IDS = {
    "row_000189_NeuroReport_13450252291_593274052_20180914": "row189_gt_required_audit.md",
    "row_000548_NeuroReport_13341375710_527779290_20190820": "row548_gt_required_audit.md",
    "row_000783_NeuroReport_13324097354_322634096_20180129": "row783_gt_required_audit.md",
}

DEFAULT_ROWS_ROOT = Path("artifacts/batch_s0001_test_Our_EvidenceGated_v1_selected50/rows")
DEFAULT_OUTPUT_DIR = Path("artifacts/stage2_9_gt_required_suppression_audit")
DEFAULT_HOST_EEG_ROOT = Path("/exHDD_8T/hjlee_data/eeg_data")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _resolve_gt_json(row_dir: Path, host_eeg_root: Path) -> Path:
    context = _read_json(row_dir / "study_context.json")
    metadata = context.get("metadata", {})
    raw = metadata.get("report_json_path_eval_only")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
        raw_str = str(path)
        if raw_str.startswith("/workspace/eeg_data"):
            host_path = host_eeg_root / raw_str.removeprefix("/workspace/eeg_data/")
            if host_path.exists():
                return host_path
        if raw_str.startswith("/workspace/"):
            desktop_path = Path("/home/hjlee/Desktop") / raw_str.removeprefix("/workspace/")
            if desktop_path.exists():
                return desktop_path
    report_id = metadata.get("report_id") or row_dir.name.split("_", 2)[-1]
    site = metadata.get("site", "S0001")
    fallback = host_eeg_root / "celm_s_sites_pipeline" / "matched_eeg_recordings_report" / site / report_id / f"{report_id}.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve GT report JSON for {row_dir}")


def _load_case(row_dir: Path, variant: str, host_eeg_root: Path) -> GTSuppressionAuditResult:
    evidence_board = EvidenceBoard.model_validate_json((row_dir / "evidence_board.json").read_text(encoding="utf-8"))
    if (row_dir / "shared_evidence_board.json").exists():
        shared_board = SharedEvidenceBoard.model_validate_json((row_dir / "shared_evidence_board.json").read_text(encoding="utf-8"))
    else:
        shared_board = evidence_board.ensure_shared_evidence_board()
    atomic_claims = [AtomicClaimPlan.model_validate(item) for item in _read_json(row_dir / "atomic_claim_plan.json")]
    if (row_dir / "celm_section_texts.json").exists():
        final_report = _read_json(row_dir / "celm_section_texts.json")
    else:
        final_report = _read_json(row_dir / "celm_generated_report.json")
    gt_json = _resolve_gt_json(row_dir, host_eeg_root)
    return GTRequiredSuppressionAuditor().audit_case(
        case_id=row_dir.name,
        variant=variant,
        gt_report_json=gt_json,
        evidence_board=evidence_board,
        shared_evidence_board=shared_board,
        atomic_claims=atomic_claims,
        final_report=final_report,
    )


def _claim_row(audit: GTSuppressionAuditResult, claim_id: str) -> Dict[str, Any]:
    claims = {claim.gt_claim_id: claim for claim in audit.gt_claims}
    matches = {match.gt_claim_id: match for match in audit.gt_claim_matches}
    claim = claims[claim_id]
    match = matches[claim_id]
    return {
        "case_id": audit.case_id,
        "gt_claim_id": claim.gt_claim_id,
        "section": claim.section,
        "claim_type": claim.claim_type,
        "normalized_value": json.dumps(claim.normalized_value, ensure_ascii=False, sort_keys=True) if isinstance(claim.normalized_value, (dict, list)) else claim.normalized_value,
        "unit": claim.unit or "",
        "state": claim.state or "",
        "certainty": claim.certainty,
        "source_text": claim.source_text,
        "match_stage": match.match_stage,
        "suppression_stage": match.suppression_stage,
        "category": match.category,
        "salvageability": match.salvageability,
        "matched_measurement_ids": "|".join(match.matched_measurement_ids),
        "matched_evidence_ids": "|".join(match.matched_evidence_ids),
        "matched_atomic_claim_ids": "|".join(match.matched_atomic_claim_ids),
        "surfaced_sentence": match.surfaced_sentence or "",
        "suppression_reason": match.suppression_reason,
        "rationale": match.rationale,
    }


def _render_case_markdown(audit: GTSuppressionAuditResult) -> str:
    lines = [f"# {audit.case_id} GT-Required Suppression Audit", "", f"- variant: `{audit.variant}`"]
    lines.append(f"- GT required surface rate: `{audit.gt_required_surface_rate:.3f}`")
    lines.append(f"- GT claim recovery rate: `{audit.gt_required_claim_recovery_rate:.3f}`")
    lines.append("")
    lines.append("## GT Claim Pipeline Match Table")
    headers = ["claim", "value", "GT text", "match_stage", "suppression_stage", "category", "salvageability", "matched evidence", "matched plans"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    claims = {claim.gt_claim_id: claim for claim in audit.gt_claims}
    for match in audit.gt_claim_matches:
        claim = claims[match.gt_claim_id]
        value = claim.normalized_value if not isinstance(claim.normalized_value, dict) else f"{claim.normalized_value.get('lower')}-{claim.normalized_value.get('upper')}"
        lines.append("| " + " | ".join([
            _md(claim.claim_type),
            _md(f"{value or ''} {claim.unit or ''}".strip()),
            _md(claim.source_text[:160]),
            _md(match.match_stage),
            _md(match.suppression_stage),
            _md(match.category),
            _md(match.salvageability),
            _md(", ".join(match.matched_evidence_ids[:4])),
            _md(", ".join(match.matched_atomic_claim_ids[:4])),
        ]) + " |")
    lines.extend(["", "## GT-Required Suppressed Claims", ""])
    for match in audit.gt_claim_matches:
        if match.category in {"gt_required_but_suppressed", "gt_required_but_surfacepolicy_blocked", "gt_required_but_no_atomic_claim_generated", "gt_required_but_not_converted_to_evidence_item"}:
            claim = claims[match.gt_claim_id]
            lines.append(f"- `{claim.claim_type}`: {match.suppression_reason} ({match.salvageability})")
    if not any(match.category.startswith("gt_required_but") and "missing" not in match.category for match in audit.gt_claim_matches):
        lines.append("- None")
    lines.extend(["", "## Missing GT Claims", ""])
    for match in audit.gt_claim_matches:
        if match.category == "gt_required_but_missing_from_evidence_extraction":
            claim = claims[match.gt_claim_id]
            lines.append(f"- `{claim.claim_type}`: {claim.source_text}")
    if not audit.gt_required_missing_evidence:
        lines.append("- None")
    lines.extend(["", "## Salvageable Caveat/Allow Candidates", ""])
    for match in audit.gt_claim_matches:
        if match.salvageability in {"allow_candidate", "caveat_candidate"} and match.match_stage != "surfaced":
            claim = claims[match.gt_claim_id]
            lines.append(f"- `{claim.claim_type}` -> `{match.salvageability}`; evidence={match.matched_evidence_ids}; rationale={match.rationale}")
    if not any(match.salvageability in {"allow_candidate", "caveat_candidate"} and match.match_stage != "surfaced" for match in audit.gt_claim_matches):
        lines.append("- None")
    lines.extend(["", "## Recommendations", ""])
    lines.extend([f"- {rec}" for rec in audit.recommendations])
    return "\n".join(lines) + "\n"


def _md(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _examples_markdown(title: str, rows: Iterable[Dict[str, Any]], predicate, limit: int = 30) -> str:
    lines = [f"# {title}", ""]
    count = 0
    for row in rows:
        if not predicate(row):
            continue
        count += 1
        lines.append(f"## {count}. {row['case_id']} / {row['claim_type']}")
        lines.append(f"- source: {row['source_text']}")
        lines.append(f"- match_stage: `{row['match_stage']}`")
        lines.append(f"- suppression_stage: `{row['suppression_stage']}`")
        lines.append(f"- category: `{row['category']}`")
        lines.append(f"- salvageability: `{row['salvageability']}`")
        lines.append(f"- evidence: `{row['matched_evidence_ids']}`")
        lines.append(f"- plans: `{row['matched_atomic_claim_ids']}`")
        lines.append(f"- rationale: {row['rationale']}")
        lines.append("")
        if count >= limit:
            break
    if count == 0:
        lines.append("No examples found by the current audit.")
    return "\n".join(lines)


def _render_recommendation(aggregate: Any) -> str:
    metrics = aggregate.metrics
    lines = ["# Stage 3C GT-Grounded Recommendation", "", f"Recommendation: **{aggregate.stage3_recommendation}**", ""]
    lines.append("## Aggregate Metrics")
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value:.3f}`")
    lines.extend(["", "## Category Counts", ""])
    for key, value in sorted(aggregate.category_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Interpretation", ""])
    lines.append("- This audit uses GT reports only for evaluation-time diagnosis, not as inference input.")
    lines.append("- Event candidate burden, support scores, localization ratios, and boundary/global 0.5 Hz peaks are intentionally not treated as safe matches for GT clinical claims.")
    lines.append("- Stage 3C should proceed only if GT-required claims are present upstream and blocked downstream; otherwise detector/adapter/planner gaps should be prioritized.")
    return "\n".join(lines) + "\n"


def run_gt_required_suppression_audit(
    *,
    rows_root: Path,
    output_dir: Path,
    variant: str,
    host_eeg_root: Path = DEFAULT_HOST_EEG_ROOT,
    limit: int | None = None,
) -> Dict[str, Any]:
    row_dirs = sorted(path for path in rows_root.iterdir() if path.is_dir() and (path / "evidence_board.json").exists())
    if limit is not None:
        row_dirs = row_dirs[:limit]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    per_case_dir = output_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    audits = [_load_case(row_dir, variant, host_eeg_root) for row_dir in row_dirs]
    auditor = GTRequiredSuppressionAuditor()
    aggregate = auditor.aggregate(audits, variant=variant)

    flat_rows: list[dict[str, Any]] = []
    for audit in audits:
        _write_json(per_case_dir / f"{audit.case_id}_gt_claim_audit.json", audit)
        case_md = _render_case_markdown(audit)
        (per_case_dir / f"{audit.case_id}_gt_claim_audit.md").write_text(case_md, encoding="utf-8")
        if audit.case_id in ANCHOR_CASE_IDS:
            (output_dir / ANCHOR_CASE_IDS[audit.case_id]).write_text(case_md, encoding="utf-8")
        for claim in audit.gt_claims:
            flat_rows.append(_claim_row(audit, claim.gt_claim_id))

    _write_csv(output_dir / "selected50_gt_claim_audit.csv", flat_rows)
    _write_json(output_dir / "selected50_gt_claim_audit.json", aggregate)
    (output_dir / "gt_required_suppressed_examples.md").write_text(
        _examples_markdown(
            "GT-Required Suppressed Examples",
            flat_rows,
            lambda row: row["category"] in {"gt_required_but_suppressed", "gt_required_but_surfacepolicy_blocked", "gt_required_but_no_atomic_claim_generated", "gt_required_but_not_converted_to_evidence_item"},
        ),
        encoding="utf-8",
    )
    (output_dir / "salvageable_gt_claim_examples.md").write_text(
        _examples_markdown("Salvageable GT Claim Examples", flat_rows, lambda row: row["salvageability"] in {"allow_candidate", "caveat_candidate"}),
        encoding="utf-8",
    )
    (output_dir / "detector_gap_examples.md").write_text(
        _examples_markdown("Detector Gap Examples", flat_rows, lambda row: row["match_stage"] == "no_measurement"),
        encoding="utf-8",
    )
    (output_dir / "adapter_gap_examples.md").write_text(
        _examples_markdown("Adapter Gap Examples", flat_rows, lambda row: row["suppression_stage"] == "not_converted_to_evidence_item"),
        encoding="utf-8",
    )
    (output_dir / "claim_planner_gap_examples.md").write_text(
        _examples_markdown("Claim Planner Gap Examples", flat_rows, lambda row: row["suppression_stage"] == "atomic_claim_not_generated"),
        encoding="utf-8",
    )
    (output_dir / "surface_policy_gap_examples.md").write_text(
        _examples_markdown("Surface Policy Gap Examples", flat_rows, lambda row: row["suppression_stage"] in {"atomic_claim_blocked", "surface_policy_rejected", "reportability_blocked"}),
        encoding="utf-8",
    )
    (output_dir / "stage3c_gt_grounded_recommendation.md").write_text(_render_recommendation(aggregate), encoding="utf-8")

    metrics_rows = [{"metric": key, "value": value} for key, value in aggregate.metrics.items()]
    _write_csv(output_dir / "selected50_gt_claim_metrics.csv", metrics_rows)
    _write_csv(
        output_dir / "selected50_gt_claim_category_counts.csv",
        [{"category": key, "count": value} for key, value in sorted(aggregate.category_counts.items(), key=lambda item: (-item[1], item[0]))],
    )

    return {
        "rows_root": str(rows_root),
        "output_dir": str(output_dir),
        "case_count": len(audits),
        "gt_claim_count": aggregate.num_gt_claims,
        "metrics": aggregate.metrics,
        "stage3_recommendation": aggregate.stage3_recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2.9 GT-required suppressed evidence audit.")
    parser.add_argument("--rows-root", default=str(DEFAULT_ROWS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variant", default="Our_EvidenceGated_v1")
    parser.add_argument("--host-eeg-root", default=str(DEFAULT_HOST_EEG_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_gt_required_suppression_audit(
        rows_root=Path(args.rows_root),
        output_dir=Path(args.output_dir),
        variant=args.variant,
        host_eeg_root=Path(args.host_eeg_root),
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
