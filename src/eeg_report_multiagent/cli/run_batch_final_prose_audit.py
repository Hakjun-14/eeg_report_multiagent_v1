from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

from eeg_report_multiagent.modules.final_prose_auditor import FinalProseAuditor
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas import EvidenceBoard
from eeg_report_multiagent.schemas.final_prose_audit import FinalProseAuditResult
from eeg_report_multiagent.schemas.report import AtomicClaimPlan
from eeg_report_multiagent.schemas.shared_evidence import SharedEvidenceBoard


DEFAULT_VARIANT_TRACE_ROOTS = {
    "Our_B": "artifacts/batch_s0001_test_B_selected50/rows",
    "Our_D": "artifacts/batch_s0001_test_D_selected50/rows",
    "Our_B_QFv2": "artifacts/batch_s0001_test_B_quality_floor_v2_selected50/rows",
    "Our_Upgrade_LLMProp": "artifacts/batch_s0001_test_onepass_upgrade_llmprop_selected50/rows",
    "Our_LocalizationV2Atomic": "artifacts/batch_s0001_test_localization_v2_atomic_claim_selected50/rows",
}

TEXT_ONLY_METRICS = [
    "DebugLeakCount",
    "InternalArtifactExposureRate",
    "UnsupportedNumericHeuristicRate",
    "SectionLeakageCount",
    "SectionLeakageRate",
    "SeizureGateViolationCount",
    "SeizureGateViolationRate",
    "HighRiskViolationRate",
]

FULL_TRACE_METRICS = [
    "NumericProvenanceAccuracy",
    "UnsupportedNumericRate",
    "ClaimTraceCoverage",
    "EvidenceLinkedClaimRate",
    "AuditPassRate",
    "ReportabilityViolationCount",
]

SECTION_BUCKETS = {
    "background": "Background violations",
    "epileptiform": "Epileptiform abnormalities violations",
    "events": "Events/Seizures violations",
    "seizures": "Seizures violations",
    "impression": "Impression violations",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def _jsonable(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, list):
        return [_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    return payload


def _parse_variant_root(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--trace-root must use Variant=/path/to/rows syntax")
    name, path = text.split("=", 1)
    return name.strip(), Path(path)


def _variant_sections(variant_payload: Dict[str, Any]) -> Dict[str, str]:
    sections = variant_payload.get("generated_sections") or {}
    return {str(name): str(text or "") for name, text in sections.items()}


def _case_id(row_index: int, report_id: str) -> str:
    return f"row_{row_index:06d}_{report_id}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _find_row_dir(rows_root: Path, row_index: int, report_id: str) -> Path | None:
    direct = rows_root / _case_id(row_index, report_id)
    if direct.exists():
        return direct
    matches = list(rows_root.glob(f"row_{row_index:06d}_{report_id}*"))
    return matches[0] if matches else None


def _load_trace_context(
    row_dir: Path | None,
) -> Tuple[SharedEvidenceBoard, List[AtomicClaimPlan], bool, str]:
    if row_dir is None:
        return SharedEvidenceBoard(board_id="text_only", recording_id="unknown"), [], False, "no_row_dir"
    evidence_board_path = row_dir / "evidence_board.json"
    if not evidence_board_path.exists():
        return SharedEvidenceBoard(board_id=f"text_only_{row_dir.name}", recording_id=row_dir.name), [], False, "missing_evidence_board"
    board = EvidenceBoard.model_validate_json(evidence_board_path.read_text(encoding="utf-8"))
    shared_board = board.ensure_shared_evidence_board()
    atomic_claim_path = row_dir / "atomic_claim_plan.json"
    if atomic_claim_path.exists():
        claim_plans = [AtomicClaimPlan.model_validate(item) for item in _read_json(atomic_claim_path)]
    else:
        claim_plans = ReportSynthesizer().build_atomic_claim_plan(board)
    return shared_board, claim_plans, True, "full_trace"


def _text_only_summary(result: FinalProseAuditResult) -> Dict[str, Any]:
    unsupported_numeric = len(result.unsupported_numeric_mentions)
    debug_leaks = len(result.debug_leaks)
    section_leaks = len(result.section_leakages)
    seizure_violations = len(result.seizure_gate_violations)
    high_risk = unsupported_numeric + debug_leaks + section_leaks + seizure_violations
    numeric_total = unsupported_numeric + len(result.supported_numeric_mentions)
    return {
        "DebugLeakCount": debug_leaks,
        "InternalArtifactExposureRate": 1.0 if debug_leaks else 0.0,
        "UnsupportedNumericHeuristicRate": unsupported_numeric / numeric_total if numeric_total else 0.0,
        "SectionLeakageCount": section_leaks,
        "SectionLeakageRate": result.metrics.get("SectionLeakageRate", 0.0),
        "SeizureGateViolationCount": seizure_violations,
        "SeizureGateViolationRate": result.metrics.get("SeizureGateViolationRate", 0.0),
        "HighRiskViolationRate": 1.0 if high_risk else 0.0,
    }


def _full_trace_summary(result: FinalProseAuditResult, trace_available: bool) -> Dict[str, Any]:
    if not trace_available:
        return {key: "" for key in FULL_TRACE_METRICS}
    reportability_violations = sum(
        1 for match in result.unmatched_surface_claims if match.match_status == "surface_policy_violation"
    )
    return {
        "NumericProvenanceAccuracy": result.metrics.get("NumericProvenanceAccuracy", 1.0),
        "UnsupportedNumericRate": result.metrics.get("UnsupportedNumericRate", 0.0),
        "ClaimTraceCoverage": result.metrics.get("ClaimTraceCoverage", 1.0),
        "EvidenceLinkedClaimRate": result.metrics.get("EvidenceLinkedClaimRate", 1.0),
        "AuditPassRate": result.metrics.get("AuditPassRate", 1.0),
        "ReportabilityViolationCount": reportability_violations,
    }


def _section_key(section_name: str) -> str:
    lowered = section_name.lower()
    if "background" in lowered or "description" in lowered or "detail" in lowered:
        return "background"
    if "interictal" in lowered or "epileptiform" in lowered:
        return "epileptiform"
    if "event" in lowered:
        return "events"
    if "seizure" in lowered:
        return "seizures"
    if "impression" in lowered or "interpretation" in lowered:
        return "impression"
    return "other"


def _section_counts(result: FinalProseAuditResult) -> Dict[str, int]:
    counts = {label: 0 for label in SECTION_BUCKETS.values()}
    for item in [*result.section_leakages, *result.seizure_gate_violations, *result.debug_leaks]:
        key = _section_key(item.section_name)
        label = SECTION_BUCKETS.get(key)
        if label:
            counts[label] += 1
    for item in result.unsupported_numeric_mentions:
        key = _section_key(item.numeric_mention.section_name)
        label = SECTION_BUCKETS.get(key)
        if label:
            counts[label] += 1
    return counts


def _failure_bucket_rows(
    variant: str,
    case_id: str,
    result: FinalProseAuditResult,
    *,
    trace_available: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for leak in result.debug_leaks:
        bucket = "Debug/Internal Score Exposure" if leak.leak_type.value in {"debug_score", "internal_reviewer_text", "measurement_artifact"} else "Proxy-to-Prose Leakage"
        if "localization" in leak.term or "laterality" in leak.term or "bifrontal" in leak.term:
            bucket = "Topography Collapse or Unsafe Localization Surface"
        rows.append({
            "variant": variant,
            "case_id": case_id,
            "section": leak.section_name,
            "bucket": bucket,
            "violation_type": leak.leak_type.value,
            "offending_sentence": leak.sentence,
            "rationale": f"Banned surface term: {leak.term}",
        })
    for match in result.unsupported_numeric_mentions:
        rationale = match.rationale
        if not trace_available and match.match_status.value == "no_match":
            rationale = "Text-only audit mode has no EvidenceBoard; numeric mention is flagged as unverified rather than traceability-failed."
        rows.append({
            "variant": variant,
            "case_id": case_id,
            "section": match.numeric_mention.section_name,
            "bucket": "Numeric Provenance Failure",
            "violation_type": match.match_status.value,
            "offending_sentence": match.numeric_mention.sentence,
            "rationale": rationale,
        })
    for leak in result.section_leakages:
        bucket = "Section Leakage"
        if "localization" in leak.leakage_type or "topography" in leak.leakage_type:
            bucket = "Topography Collapse or Unsafe Localization Surface"
        rows.append({
            "variant": variant,
            "case_id": case_id,
            "section": leak.section_name,
            "bucket": bucket,
            "violation_type": leak.leakage_type,
            "offending_sentence": leak.sentence,
            "rationale": leak.rationale,
        })
    for leak in result.seizure_gate_violations:
        rows.append({
            "variant": variant,
            "case_id": case_id,
            "section": leak.section_name,
            "bucket": "Seizure Gate Violation",
            "violation_type": leak.leakage_type,
            "offending_sentence": leak.sentence,
            "rationale": leak.rationale,
        })
    if trace_available:
        for match in result.unmatched_surface_claims:
            bucket = "Traceability Missing" if match.match_status == "unmatched_surface_claim" else "Proxy-to-Prose Leakage"
            rows.append({
                "variant": variant,
                "case_id": case_id,
                "section": match.section_name,
                "bucket": bucket,
                "violation_type": match.match_status,
                "offending_sentence": match.sentence,
                "rationale": match.rationale,
            })
    if not rows and result.pass_fail == "pass":
        rows.append({
            "variant": variant,
            "case_id": case_id,
            "section": "",
            "bucket": "Safe Empty Output" if not result.supported_numeric_mentions else "No High-Risk Violation",
            "violation_type": "none",
            "offending_sentence": "",
            "rationale": "No high-risk final-prose audit violation detected.",
        })
    return rows


def _aggregate_variant(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    out: List[Dict[str, Any]] = []
    for variant, items in sorted(by_variant.items()):
        agg: Dict[str, Any] = {"variant": variant, "report_count": len(items)}
        for metric in TEXT_ONLY_METRICS:
            vals = [float(item.get(metric) or 0.0) for item in items]
            agg[f"{metric}_mean"] = mean(vals) if vals else 0.0
            agg[f"{metric}_median"] = median(vals) if vals else 0.0
            agg[f"{metric}_max"] = max(vals) if vals else 0.0
        trace_items = [item for item in items if item.get("trace_available") == "true"]
        agg["trace_available_count"] = len(trace_items)
        for metric in FULL_TRACE_METRICS:
            vals = [float(item.get(metric) or 0.0) for item in trace_items if item.get(metric) != ""]
            agg[f"{metric}_mean"] = mean(vals) if vals else ""
        for label in SECTION_BUCKETS.values():
            agg[label] = sum(int(item.get(label) or 0) for item in items)
        out.append(agg)
    return out


def _bucket_counts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = Counter((row["variant"], row["bucket"]) for row in rows)
    return [
        {"variant": variant, "failure_bucket": bucket, "count": count}
        for (variant, bucket), count in sorted(counter.items())
    ]


def _render_variant_comparison(aggregate_rows: List[Dict[str, Any]], high_risk: List[Dict[str, Any]]) -> str:
    lines = ["# Stage 2.5 Batch Final-Prose Audit", ""]
    lines.append("## Overall Table")
    headers = [
        "variant",
        "report_count",
        "DebugLeakCount_mean",
        "UnsupportedNumericHeuristicRate_mean",
        "SectionLeakageCount_mean",
        "SeizureGateViolationRate_mean",
        "HighRiskViolationRate_mean",
        "trace_available_count",
        "NumericProvenanceAccuracy_mean",
        "ClaimTraceCoverage_mean",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in aggregate_rows:
        lines.append("| " + " | ".join(_fmt(row.get(header, "")) for header in headers) + " |")
    lines.extend(["", "## Top High-Risk Examples", ""])
    for idx, row in enumerate(high_risk[:10], start=1):
        lines.append(f"{idx}. `{row['variant']}` `{row['case_id']}` `{row['section']}`")
        lines.append(f"   - bucket: {row['bucket']}")
        lines.append(f"   - violation: {row['violation_type']}")
        lines.append(f"   - sentence: {row['offending_sentence'][:500]}")
        lines.append(f"   - rationale: {row['rationale']}")
    lines.extend(["", "## Short Interpretation", ""])
    if aggregate_rows:
        debug_worst = max(aggregate_rows, key=lambda r: float(r.get("DebugLeakCount_mean") or 0.0))
        numeric_worst = max(aggregate_rows, key=lambda r: float(r.get("UnsupportedNumericHeuristicRate_mean") or 0.0))
        seizure_worst = max(aggregate_rows, key=lambda r: float(r.get("SeizureGateViolationRate_mean") or 0.0))
        safest = min(aggregate_rows, key=lambda r: float(r.get("HighRiskViolationRate_mean") or 0.0))
        lines.append(f"- Most internal/debug feature exposure: `{debug_worst['variant']}`.")
        lines.append(f"- Highest unsupported numeric heuristic rate: `{numeric_worst['variant']}`.")
        lines.append(f"- Most seizure gate pressure: `{seizure_worst['variant']}`.")
        lines.append(f"- Lowest high-risk violation rate in this audit: `{safest['variant']}`.")
        lines.append("- Traceability metrics are reported only for variants with local EvidenceBoard/AtomicClaimPlan artifacts; text-only variants are not penalized for missing trace objects.")
        lines.append("- Safe fallbacks reduce high-risk leakage, but high safe-empty counts should be interpreted as possible over-suppression rather than clinical adequacy.")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_high_risk_examples(rows: List[Dict[str, Any]]) -> str:
    lines = ["# High-Risk Final-Prose Audit Examples", ""]
    for idx, row in enumerate(rows[:50], start=1):
        lines.append(f"## {idx}. {row['variant']} / {row['case_id']}")
        lines.append(f"- section: {row['section']}")
        lines.append(f"- bucket: {row['bucket']}")
        lines.append(f"- violation_type: {row['violation_type']}")
        lines.append(f"- offending_sentence: {row['offending_sentence']}")
        lines.append(f"- rationale: {row['rationale']}")
        lines.append("")
    return "\n".join(lines)


def _render_row189_summary(per_report_rows: List[Dict[str, Any]], bucket_rows: List[Dict[str, Any]]) -> str:
    row_rows = [row for row in per_report_rows if "row_000189_" in row["case_id"]]
    bucket_row189 = [row for row in bucket_rows if "row_000189_" in row["case_id"] and row["violation_type"] != "none"]
    lines = ["# Row-189 Style Final-Prose Audit Summary", ""]
    if not row_rows:
        lines.append("No row-189 case was present in the audited subset.")
        return "\n".join(lines) + "\n"
    headers = ["variant", "audit_mode", "DebugLeakCount", "unsupported_numeric_count", "SectionLeakageCount", "SeizureGateViolationCount", "HighRiskViolationRate", "trace_available"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in sorted(row_rows, key=lambda r: r["variant"]):
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.extend(["", "## Offending Examples", ""])
    for row in bucket_row189[:15]:
        lines.append(f"- `{row['variant']}` `{row['section']}` {row['bucket']}: {row['offending_sentence'][:400]}")
    lines.extend(["", "## Interpretation", ""])
    lines.append("- A low high-risk count indicates safer surface behavior, not necessarily richer clinical adequacy.")
    lines.append("- If the current SurfacePolicy output is safely empty, it should be treated as over-suppression risk to be addressed later by better evidence, not by bypassing the gate.")
    return "\n".join(lines) + "\n"


def run_batch_final_prose_audit(
    comparison_json_dir: Path,
    output_dir: Path,
    trace_roots: Dict[str, Path],
    limit: int | None = None,
) -> Dict[str, Any]:
    auditor = FinalProseAuditor()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_report_dir = output_dir / "per_report_audits"
    if per_report_dir.exists():
        shutil.rmtree(per_report_dir)
    per_report_dir.mkdir(parents=True, exist_ok=True)

    case_paths = sorted(comparison_json_dir.glob("*.json"))
    if limit is not None:
        case_paths = case_paths[:limit]

    per_report_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    for case_path in case_paths:
        case = _read_json(case_path)
        row_index = int(case.get("row_index"))
        report_id = str(case.get("report_id"))
        case_id = _case_id(row_index, report_id)
        for variant, payload in sorted((case.get("variants") or {}).items()):
            sections = _variant_sections(payload)
            trace_root = trace_roots.get(variant)
            row_dir = _find_row_dir(trace_root, row_index, report_id) if trace_root else None
            shared_board, claim_plans, trace_available, audit_mode = _load_trace_context(row_dir)
            result = auditor.audit_report(sections, shared_board, claim_plans)
            audit_payload = {
                "variant": variant,
                "case_id": case_id,
                "row_index": row_index,
                "report_id": report_id,
                "audit_mode": audit_mode if trace_available else "text_only_safety",
                "trace_available": trace_available,
                "result": result,
            }
            audit_path = per_report_dir / f"{_safe_name(variant)}_{case_id}_final_prose_audit.json"
            _write_json(audit_path, audit_payload)

            text_summary = _text_only_summary(result)
            full_summary = _full_trace_summary(result, trace_available)
            section_counts = _section_counts(result)
            row = {
                "variant": variant,
                "case_id": case_id,
                "row_index": row_index,
                "report_id": report_id,
                "audit_mode": audit_payload["audit_mode"],
                "trace_available": "true" if trace_available else "false",
                "audit_path": str(audit_path),
                "unsupported_numeric_count": len(result.unsupported_numeric_mentions),
                **text_summary,
                **full_summary,
                **section_counts,
            }
            per_report_rows.append(row)
            bucket_rows.extend(_failure_bucket_rows(variant, case_id, result, trace_available=trace_available))

    aggregate_rows = _aggregate_variant(per_report_rows)
    bucket_count_rows = _bucket_counts(bucket_rows)
    high_risk_rows = [
        row for row in bucket_rows
        if row["violation_type"] != "none" and row["bucket"] != "No High-Risk Violation"
    ]

    _write_csv(output_dir / "per_report_audit_summary.csv", per_report_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_json(output_dir / "aggregate_metrics.json", {"variants": aggregate_rows, "per_report_count": len(per_report_rows)})
    _write_csv(output_dir / "failure_bucket_counts.csv", bucket_count_rows)
    (output_dir / "variant_comparison.md").write_text(_render_variant_comparison(aggregate_rows, high_risk_rows), encoding="utf-8")
    (output_dir / "high_risk_examples.md").write_text(_render_high_risk_examples(high_risk_rows), encoding="utf-8")
    (output_dir / "row189_audit_summary.md").write_text(_render_row189_summary(per_report_rows, bucket_rows), encoding="utf-8")

    return {
        "comparison_json_dir": str(comparison_json_dir),
        "output_dir": str(output_dir),
        "case_count": len(case_paths),
        "per_report_count": len(per_report_rows),
        "variants": [row["variant"] for row in aggregate_rows],
        "aggregate_metrics_csv": str(output_dir / "aggregate_metrics.csv"),
        "variant_comparison_md": str(output_dir / "variant_comparison.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2.5 batch final-prose audit over selected comparison outputs.")
    parser.add_argument("--comparison-json-dir", default="artifacts/gt_generated_comparison_selected50_UpgradeLLMProp/per_case_json")
    parser.add_argument("--output-dir", default="artifacts/stage2_5_batch_audit")
    parser.add_argument("--trace-root", action="append", default=[], help="Variant=/path/to/rows for full traceability audit.")
    parser.add_argument("--use-default-trace-roots", action="store_true", help="Use known selected50 OURS row artifact roots when present.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    trace_roots: Dict[str, Path] = {}
    if args.use_default_trace_roots:
        trace_roots.update({name: Path(path) for name, path in DEFAULT_VARIANT_TRACE_ROOTS.items() if Path(path).exists()})
    trace_roots.update(dict(_parse_variant_root(item) for item in args.trace_root))
    summary = run_batch_final_prose_audit(
        comparison_json_dir=Path(args.comparison_json_dir),
        output_dir=Path(args.output_dir),
        trace_roots=trace_roots,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
