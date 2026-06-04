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


def _audit_weak_measurements(measurements: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
    return flags


def audit_artifact_dir(artifact_dir: Path) -> Dict[str, Any]:
    study_context = _read_json(artifact_dir / "study_context.json", {})
    trace = _read_json(artifact_dir / "inference_trace.json", {})
    manifest = _read_json(artifact_dir / "manifest.json", {})
    scout = _read_json(artifact_dir / "scout_summary.json", {})
    evidence_board = _read_json(artifact_dir / "evidence_board.json", {})
    background_measurements_file = _read_json(artifact_dir / "background_measurements.json", [])
    event_measurements_file = _read_json(artifact_dir / "event_measurements.json", [])
    parser_measurements_file = _read_json(artifact_dir / "parsed_context.json", [])
    shared_evidence_board = _read_json(artifact_dir / "shared_evidence_board.json", {})
    verification = _read_json(artifact_dir / "verification.json", [])
    run_log = _read_text(artifact_dir / "run.log")
    detail = _read_text(artifact_dir / "detail.txt")
    impression = _read_text(artifact_dir / "impression.txt")

    measurements = _as_list(evidence_board, "measurements")
    board_claims = _as_list(evidence_board, "claims")
    tool_invocations = _as_list(evidence_board, "tool_invocations")
    deliberations = _as_list(evidence_board, "deliberations")
    shared_evidence_items = _as_list(shared_evidence_board, "evidence_items")
    if not shared_evidence_items and isinstance(evidence_board, dict):
        shared_evidence_items = _as_list(evidence_board.get("shared_evidence_board"), "evidence_items")
    verification_records = _as_list(verification)
    if not verification_records:
        verification_records = _as_list(trace.get("verification") if isinstance(trace, dict) else None)

    all_measurements = measurements or (
        _as_list(background_measurements_file) + _as_list(event_measurements_file) + _as_list(parser_measurements_file)
    )
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
            "measurements": len(all_measurements),
            "shared_evidence_items": len(shared_evidence_items),
            "claims": len(board_claims),
            "tool_invocations": len(tool_invocations),
            "agent_deliberations": len(deliberations),
            "weak_evidence_records": sum(len(_as_list(d, "weak_evidence")) for d in deliberations),
            "missing_slot_records": sum(len(_as_list(d, "missing_slots")) for d in deliberations),
            "do_not_claim_records": sum(len(_as_list(d, "do_not_claim")) for d in deliberations),
            "claim_constraint_records": sum(len(_as_list(d, "claim_constraints")) for d in deliberations),
            "background_measurements_file": len(_as_list(background_measurements_file)),
            "event_measurements_file": len(_as_list(event_measurements_file)),
            "parser_measurements_file": len(_as_list(parser_measurements_file)),
            "verification_records": len(verification_records),
        },
        "tool_invocation_status": _count_by(tool_invocations, "status"),
        "claim_support": _count_by(verification_records, "support_label"),
        "weak_measurement_flags": _audit_weak_measurements(all_measurements),
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
        and (
            audit["counts"]["shared_evidence_items"] > 0
            or audit["counts"]["claims"] > 0
        )
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
