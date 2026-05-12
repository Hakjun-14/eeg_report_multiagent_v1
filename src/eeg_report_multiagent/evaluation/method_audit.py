from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LEAKAGE_TOP_LEVEL_KEYS = {
    "note_text",
    "report_text",
    "reference_text",
    "target_text",
    "eeg_section_text",
    "EEG_section_llm_extractions",
}


@dataclass(frozen=True)
class AuditInputs:
    artifact_dir: Path


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _as_list(payload: Any, key: Optional[str] = None) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict) and key and isinstance(payload.get(key), list):
        return [x for x in payload[key] if isinstance(x, dict)]
    return []


def _source_types(provenance: Iterable[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for prov in provenance:
        source_type = prov.get("source_type")
        if isinstance(source_type, str):
            out.add(source_type)
    return out


def _has_time_provenance(provenance: Iterable[Dict[str, Any]]) -> bool:
    for prov in provenance:
        time = prov.get("time") or {}
        if time.get("window_indices") or time.get("start_sec") is not None or time.get("end_sec") is not None:
            return True
    return False


def _has_space_provenance(provenance: Iterable[Dict[str, Any]]) -> bool:
    for prov in provenance:
        space = prov.get("space") or {}
        if space.get("channels") or space.get("region") or space.get("laterality"):
            return True
    return False


def _has_measurement_provenance(provenance: Iterable[Dict[str, Any]]) -> bool:
    for prov in provenance:
        measurement = prov.get("measurement") or {}
        if measurement.get("tool_name") or measurement.get("function_name") or measurement.get("measurement_ids"):
            return True
    return False


def _count_by(items: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(item.get(field, "<missing>")) for item in items))


def _audit_input_contract(study_context: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    top_keys = set(study_context.keys())
    metadata = study_context.get("metadata") if isinstance(study_context.get("metadata"), dict) else {}
    forbidden_top_keys = sorted(top_keys & LEAKAGE_TOP_LEVEL_KEYS)
    metadata_contains_eval_only_path = "report_json_path_eval_only" in metadata
    top_level_has_gt_path = "gt_report_json_path" in study_context or "report_json_path" in study_context
    trace_inputs = trace.get("inputs") if isinstance(trace.get("inputs"), dict) else {}
    return {
        "pass": not forbidden_top_keys and not top_level_has_gt_path,
        "forbidden_top_level_keys": forbidden_top_keys,
        "top_level_has_gt_path": top_level_has_gt_path,
        "metadata_contains_report_json_path_eval_only": metadata_contains_eval_only_path,
        "trace_input_contract": trace_inputs.get("input_contract"),
        "trace_gt_report_json_path_eval_only": trace_inputs.get("gt_report_json_path_eval_only"),
        "trace_legacy_report_alias_used": bool(trace_inputs.get("legacy_report_json_alias_used") or trace_inputs.get("legacy_report_text_alias_used")),
    }


def _audit_provenance(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    signal_findings = []
    for finding in findings:
        provenance = _as_list(finding.get("provenance"))
        if "signal" in _source_types(provenance) or str(finding.get("source_module", "")).startswith(("background", "event")):
            signal_findings.append(finding)

    missing_time: List[str] = []
    missing_space: List[str] = []
    missing_measurement: List[str] = []
    no_provenance: List[str] = []
    for finding in signal_findings:
        fid = str(finding.get("finding_id", "<missing>"))
        provenance = _as_list(finding.get("provenance"))
        if not provenance:
            no_provenance.append(fid)
            continue
        if not _has_time_provenance(provenance):
            missing_time.append(fid)
        if not _has_space_provenance(provenance):
            missing_space.append(fid)
        if not _has_measurement_provenance(provenance):
            missing_measurement.append(fid)

    denom = max(len(signal_findings), 1)
    complete = len(signal_findings) - len(set(no_provenance + missing_time + missing_space + missing_measurement))
    return {
        "signal_finding_count": len(signal_findings),
        "complete_signal_finding_count": complete,
        "complete_signal_finding_fraction": complete / denom,
        "missing_time_finding_ids": missing_time,
        "missing_space_finding_ids": missing_space,
        "missing_measurement_finding_ids": missing_measurement,
        "no_provenance_finding_ids": no_provenance,
    }


def _audit_weak_measurements(measurements: List[Dict[str, Any]], findings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    for measurement in measurements:
        name = str(measurement.get("measurement_name", ""))
        quant = measurement.get("quantitation") if isinstance(measurement.get("quantitation"), dict) else {}
        exact = quant.get("exact")
        lower = quant.get("lower")
        upper = quant.get("upper")
        if name == "background_dominant_frequency_hz" and exact in {0.5, 30.0}:
            flags.append(
                {
                    "severity": "medium",
                    "measurement_id": str(measurement.get("measurement_id")),
                    "reason": "Dominant frequency is at the configured spectral search boundary; treat as weak PDR/background evidence.",
                }
            )
        if "amplitude" in name and (upper == 0 or (lower == 0 and upper == 0)):
            flags.append(
                {
                    "severity": "medium",
                    "measurement_id": str(measurement.get("measurement_id")),
                    "reason": "Amplitude quantitation is zero-valued; check scale inference, flatline, or artifact masking.",
                }
            )
        if name.endswith("score") and exact is None:
            flags.append(
                {
                    "severity": "low",
                    "measurement_id": str(measurement.get("measurement_id")),
                    "reason": "Score measurement has no exact numeric value.",
                }
            )

    unknown_findings = [f for f in findings if str(f.get("assertion")) == "unknown"]
    if unknown_findings:
        flags.append(
            {
                "severity": "info",
                "measurement_id": "<finding-level>",
                "reason": f"{len(unknown_findings)} finding(s) have unknown assertion; this is acceptable for unavailable context but should be tracked.",
            }
        )
    return flags


def audit_artifact_dir(artifact_dir: Path) -> Dict[str, Any]:
    study_context = _read_json(artifact_dir / "study_context.json", {})
    trace = _read_json(artifact_dir / "inference_trace.json", {})
    manifest = _read_json(artifact_dir / "manifest.json", {})
    scout = _read_json(artifact_dir / "scout_summary.json", {})
    evidence_board = _read_json(artifact_dir / "evidence_board.json", {})
    background_findings = _read_json(artifact_dir / "background_findings.json", [])
    event_findings = _read_json(artifact_dir / "event_findings.json", [])
    parser_findings = _read_json(artifact_dir / "parsed_context.json", [])
    verification = _read_json(artifact_dir / "verification.json", [])
    run_log = _read_text(artifact_dir / "run.log")
    detail = _read_text(artifact_dir / "detail.txt")
    impression = _read_text(artifact_dir / "impression.txt")

    measurements = _as_list(evidence_board, "measurements")
    board_findings = _as_list(evidence_board, "findings")
    board_claims = _as_list(evidence_board, "claims")
    tool_invocations = _as_list(evidence_board, "tool_invocations")
    deliberations = _as_list(evidence_board, "deliberations")
    verification_records = _as_list(verification)
    if not verification_records:
        verification_records = _as_list(trace.get("verification") if isinstance(trace, dict) else None)

    all_findings = board_findings or (_as_list(background_findings) + _as_list(event_findings) + _as_list(parser_findings))
    windows = _as_list(manifest, "windows")

    audit = {
        "artifact_dir": str(artifact_dir),
        "input_contract": _audit_input_contract(study_context if isinstance(study_context, dict) else {}, trace if isinstance(trace, dict) else {}),
        "manifest_summary": {
            "session_id": manifest.get("session_id") if isinstance(manifest, dict) else None,
            "shape_nct": manifest.get("shape_nct") if isinstance(manifest, dict) else None,
            "window_count": len(windows),
            "source_pkl_count": len(manifest.get("source_pkl_paths", [])) if isinstance(manifest, dict) else 0,
        },
        "scout_summary": scout,
        "counts": {
            "measurements": len(measurements),
            "findings": len(all_findings),
            "claims": len(board_claims),
            "tool_invocations": len(tool_invocations),
            "agent_deliberations": len(deliberations),
            "weak_evidence_records": sum(len(_as_list(d, "weak_evidence")) for d in deliberations),
            "missing_slot_records": sum(len(_as_list(d, "missing_slots")) for d in deliberations),
            "do_not_claim_records": sum(len(_as_list(d, "do_not_claim")) for d in deliberations),
            "claim_constraint_records": sum(len(_as_list(d, "claim_constraints")) for d in deliberations),
            "background_findings_file": len(_as_list(background_findings)),
            "event_findings_file": len(_as_list(event_findings)),
            "parser_findings_file": len(_as_list(parser_findings)),
            "verification_records": len(verification_records),
        },
        "finding_assertions": _count_by(all_findings, "assertion"),
        "finding_types": _count_by(all_findings, "finding_type"),
        "tool_invocation_status": _count_by(tool_invocations, "status"),
        "claim_support": _count_by(verification_records, "support_label"),
        "provenance": _audit_provenance(all_findings),
        "weak_measurement_flags": _audit_weak_measurements(measurements, all_findings),
        "report_text_summary": {
            "detail_chars": len(detail),
            "impression_chars": len(impression),
            "detail_nonempty": bool(detail.strip()),
            "impression_nonempty": bool(impression.strip()),
        },
        "run_log_tail": run_log.strip().splitlines()[-12:],
    }
    audit["overall_pass"] = bool(
        audit["input_contract"]["pass"]
        and audit["manifest_summary"]["window_count"] > 0
        and audit["counts"]["measurements"] > 0
        and audit["counts"]["findings"] > 0
        and audit["report_text_summary"]["detail_nonempty"]
    )
    return audit


def render_audit_markdown(audit: Dict[str, Any]) -> str:
    lines = [
        "# Method Audit",
        "",
        f"- artifact_dir: `{audit.get('artifact_dir')}`",
        f"- overall_pass: `{audit.get('overall_pass')}`",
        f"- input_contract_pass: `{audit.get('input_contract', {}).get('pass')}`",
        "",
        "## Manifest",
    ]
    manifest = audit.get("manifest_summary", {})
    lines.extend(
        [
            f"- session_id: `{manifest.get('session_id')}`",
            f"- shape_nct: `{manifest.get('shape_nct')}`",
            f"- window_count: `{manifest.get('window_count')}`",
            f"- source_pkl_count: `{manifest.get('source_pkl_count')}`",
            "",
            "## Counts",
        ]
    )
    for key, value in audit.get("counts", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Provenance"])
    provenance = audit.get("provenance", {})
    for key in [
        "signal_finding_count",
        "complete_signal_finding_count",
        "complete_signal_finding_fraction",
    ]:
        lines.append(f"- {key}: `{provenance.get(key)}`")
    lines.extend(["", "## Weak Measurement Flags"])
    flags = audit.get("weak_measurement_flags", [])
    if flags:
        for flag in flags:
            lines.append(f"- {flag.get('severity')}: `{flag.get('measurement_id')}` - {flag.get('reason')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Support"])
    for key, value in audit.get("claim_support", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)
