from __future__ import annotations

from typing import Any, Iterable, List

from eeg_report_multiagent.schemas.agent import AgentDeliberationRecord
from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue, StatusSemantic
from eeg_report_multiagent.schemas.provenance import ProvenanceRecord, SourceType
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


_EVENT_CANDIDATE_FINDINGS = {"epileptiform_event_candidate_burden", "event_train_duration"}
_EVENT_CANDIDATE_MEASUREMENTS = {
    "event_candidate_score_distribution",
    "event_candidate_burden_ratio",
    "event_train_duration_upper_sec",
    "event_train_duration_distribution_sec",
}
_DEBUG_SCORE_MEASUREMENTS = {
    "event_morphology_support_score",
    "epileptiform_candidate_likelihood_score",
    "electrographic_seizure_likelihood_score",
    "pdr_candidate_confidence_score",
    "background_ap_organization_score",
    "slowing_score",
    "beta_excess_score",
}
_LOCALIZATION_PROXY_FINDINGS = {
    "event_laterality",
    "event_focality_bifrontal_spread",
    "event_clinical_localization",
    "event_localization_support",
    "event_peak_localization",
    "event_peak_field_support",
    "event_peak_laterality",
    "event_field_concentration",
}
_LOCALIZATION_PROXY_MEASUREMENTS = {
    "event_laterality_index",
    "event_clinical_localization_label",
    "event_localization_concentration_ratio",
    "event_peak_localization_label",
    "event_peak_field_concentration_ratio",
    "event_peak_laterality_index",
    "event_bifrontal_ratio",
    "event_field_concentration_ratio",
}
_MORPHOLOGY_PROXY_FINDINGS = {"event_morphology_class", "event_morphology_support"}
_MORPHOLOGY_PROXY_MEASUREMENTS = {"event_morphology_proxy_class", "event_morphology_proxy_score_distribution"}
_PROTOCOL_FINDINGS = {
    "protocol_state_awake",
    "protocol_state_drowsy",
    "protocol_state_sleep",
    "protocol_hyperventilation_status",
    "protocol_photic_stimulation_status",
    "protocol_ekg_availability",
    "protocol_video_availability",
    "protocol_comparison_history_presence",
}


def build_shared_evidence_board(
    *,
    recording_id: str,
    measurements: Iterable[MeasurementValue],
    findings: Iterable[Finding],
    board_id: str | None = None,
) -> SharedEvidenceBoard:
    measurement_list = list(measurements)
    finding_list = list(findings)
    measurement_index = {measurement.measurement_id: measurement for measurement in measurement_list}
    board = SharedEvidenceBoard(board_id=board_id or f"seb_{recording_id}", recording_id=recording_id)

    if not finding_list:
        for item in grouped_evidence_items_from_measurements(measurement_list):
            board.add_evidence(item)
        return board

    linked_measurements: set[str] = set()
    for finding in finding_list:
        linked = [measurement_index[mid] for mid in finding.measurement_ids if mid in measurement_index]
        if linked:
            linked_measurements.update(m.measurement_id for m in linked)
            board.add_evidence(evidence_item_from_finding(finding, linked[0]))
        else:
            board.add_evidence(evidence_item_from_finding(finding, None))

    for measurement in measurement_list:
        if measurement.measurement_id not in linked_measurements:
            board.add_evidence(evidence_item_from_measurement(measurement))
    return board


def grouped_evidence_items_from_measurements(measurements: Iterable[MeasurementValue]) -> List[EvidenceItem]:
    """Build clinically grouped evidence directly from deterministic measurements.

    This is the new runtime path. `Finding` remains loadable for old artifacts,
    but new sessions should group measurements by clinical target before claim
    planning rather than creating one finding/evidence/claim per tool output.
    """

    measurement_list = list(measurements)
    groups: dict[str, list[MeasurementValue]] = {}
    for measurement in measurement_list:
        group_key = _measurement_group_key(measurement)
        groups.setdefault(group_key, []).append(measurement)

    out: list[EvidenceItem] = []
    for group_key, group_measurements in groups.items():
        item = _grouped_evidence_item(group_key, group_measurements)
        if item is not None:
            out.append(item)
    return out


def _measurement_group_key(measurement: MeasurementValue) -> str:
    name = measurement.measurement_name
    if name.startswith("pdr_") or name.startswith("background_ap_organization") or name == "background_reactivity_status":
        return "pdr"
    if name == "background_amplitude_range_uv":
        return "background_amplitude"
    if name in {"background_dominant_frequency_hz", "slowing_score"}:
        return "background_slowing"
    if name == "beta_excess_score":
        return "excess_beta"
    if name.startswith("protocol_state") or name.startswith("state_") or name in {"sleep_architecture_status"}:
        return "state"
    if name.startswith("protocol_") or name in {"hyperventilation_status", "photic_stimulation_status"} or name.endswith("_availability") or name.endswith("_presence"):
        return "protocol"
    if name in _EVENT_CANDIDATE_MEASUREMENTS:
        return "event_candidate"
    if name in _LOCALIZATION_PROXY_MEASUREMENTS:
        return "localization"
    if name in _MORPHOLOGY_PROXY_MEASUREMENTS or name in {"event_morphology_support_score", "epileptiform_candidate_likelihood_score"}:
        return "epileptiform_morphology"
    if "seizure" in name:
        return "seizure_evidence"
    if name.startswith("relative_bandpower_"):
        return "background_bandpower_debug"
    return f"measurement:{name}"


def _grouped_evidence_item(group_key: str, measurements: list[MeasurementValue]) -> EvidenceItem | None:
    if not measurements:
        return None
    target, evidence_type, reportability, source_module, allowed = _group_classification(group_key, measurements)
    provenance = [measurement.provenance for measurement in measurements]
    value = _group_value(group_key, measurements)
    return EvidenceItem(
        evidence_id=f"evgrp_{group_key.replace(':', '_')}",
        source_module=source_module,
        evidence_type=evidence_type,
        clinical_target=target,
        value=value,
        unit=None,
        normalized_value=value,
        confidence=None,
        reliability=None,
        time_provenance=_time_dict(provenance),
        space_provenance=_space_dict(provenance),
        measurement_ids=[measurement.measurement_id for measurement in measurements],
        finding_ids=[],
        reportability=reportability,
        allowed_sections=allowed,
        rationale=None,
        caveat=None,
        debug_payload={
            "measurement_names": [measurement.measurement_name for measurement in measurements],
            "group_key": group_key,
            "legacy_finding_removed": True,
        },
        created_by="measurement_grouped_evidence_builder",
        created_at=EvidenceItem.now_iso(),
    )


def _group_classification(
    group_key: str,
    measurements: list[MeasurementValue],
) -> tuple[ClinicalTarget, EvidenceType, ClaimSurfaceAction, str, list[str]]:
    if group_key == "pdr":
        return ClinicalTarget.PDR, EvidenceType.DERIVED, ClaimSurfaceAction.CAVEAT, "background", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.SLEEP.value]
    if group_key == "background_amplitude":
        return ClinicalTarget.BACKGROUND_AMPLITUDE, EvidenceType.DERIVED, ClaimSurfaceAction.CAVEAT, "background", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value]
    if group_key == "background_slowing":
        return ClinicalTarget.BACKGROUND_SLOWING, EvidenceType.DERIVED, ClaimSurfaceAction.CAVEAT, "background", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.IMPRESSION.value]
    if group_key == "excess_beta":
        return ClinicalTarget.EXCESS_BETA, EvidenceType.DERIVED, ClaimSurfaceAction.CAVEAT, "background", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.IMPRESSION.value]
    if group_key == "background_bandpower_debug":
        return ClinicalTarget.UNCERTAINTY, EvidenceType.DEBUG, ClaimSurfaceAction.DEBUG_ONLY, "background", []
    if group_key == "state":
        return ClinicalTarget.STATE, EvidenceType.METADATA, ClaimSurfaceAction.ALLOW, "state_protocol", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.SLEEP.value]
    if group_key == "protocol":
        return ClinicalTarget.PROTOCOL, EvidenceType.METADATA, ClaimSurfaceAction.ALLOW, "state_protocol", [SectionRole.DETAIL.value, SectionRole.BACKGROUND.value]
    if group_key == "event_candidate":
        return ClinicalTarget.EVENT_CANDIDATE, EvidenceType.PROXY, ClaimSurfaceAction.DEBUG_ONLY, "event", []
    if group_key == "localization":
        return ClinicalTarget.LOCALIZATION, EvidenceType.PROXY, ClaimSurfaceAction.DEBUG_ONLY, "localization", []
    if group_key == "epileptiform_morphology":
        return ClinicalTarget.EPILEPTIFORM_MORPHOLOGY, EvidenceType.DEBUG, ClaimSurfaceAction.DEBUG_ONLY, "event", []
    if group_key == "seizure_evidence":
        return ClinicalTarget.SEIZURE_EVIDENCE, EvidenceType.DEBUG, ClaimSurfaceAction.DEBUG_ONLY, "seizure", []
    return ClinicalTarget.UNKNOWN, EvidenceType.DEBUG, ClaimSurfaceAction.DEBUG_ONLY, "unknown", []


def _group_value(group_key: str, measurements: list[MeasurementValue]) -> Any:
    by_name = {measurement.measurement_name: measurement for measurement in measurements}
    if group_key == "pdr":
        freq = by_name.get("pdr_candidate_frequency_hz")
        return {
            "frequency_hz": _numeric(freq),
            "pdr_supported": (freq.metadata.get("pdr_supported") if freq else None),
            "posterior_alpha_ratio": (freq.metadata.get("posterior_alpha_ratio") if freq else None),
            "posterior_anterior_alpha_ratio": (freq.metadata.get("posterior_anterior_alpha_ratio") if freq else None),
            "symmetry_score": _numeric(by_name.get("pdr_symmetry_score")),
            "reactivity": _status(by_name.get("background_reactivity_status")),
        }
    if group_key == "background_amplitude":
        measurement = by_name.get("background_amplitude_range_uv")
        return _range_or_value(measurement)
    if group_key in {"background_slowing", "excess_beta"}:
        return {measurement.measurement_name: _range_or_value(measurement) for measurement in measurements}
    if group_key in {"state", "protocol"}:
        return {measurement.measurement_name: _status_or_value(measurement) for measurement in measurements}
    if group_key in {"event_candidate", "localization", "epileptiform_morphology", "seizure_evidence"}:
        return {measurement.measurement_name: _debug_value(measurement) for measurement in measurements}
    return {measurement.measurement_name: _debug_value(measurement) for measurement in measurements}


def _numeric(measurement: MeasurementValue | None) -> float | None:
    if measurement is None or measurement.quantitation is None:
        return None
    return measurement.quantitation.exact


def _range_or_value(measurement: MeasurementValue | None) -> Any:
    if measurement is None:
        return None
    value, _unit = _value_and_unit(None, measurement)
    return value


def _status(measurement: MeasurementValue | None) -> str | None:
    return measurement.status_value.status.value if measurement is not None and measurement.status_value else None


def _status_or_value(measurement: MeasurementValue) -> Any:
    if measurement.status_value is not None:
        return measurement.status_value.status.value
    return _debug_value(measurement)


def _debug_value(measurement: MeasurementValue) -> Any:
    value, _unit = _value_and_unit(None, measurement)
    return value


def evidence_item_from_finding(finding: Finding, measurement: MeasurementValue | None = None) -> EvidenceItem:
    evidence_type, target, reportability, rationale, allowed = _classify_finding(finding, measurement)
    value, unit = _value_and_unit(finding, measurement)
    provenance = _linked_provenance(finding, measurement)
    return EvidenceItem(
        evidence_id=f"ev_{finding.finding_id}",
        source_module=_source_module(finding, measurement),
        evidence_type=evidence_type,
        clinical_target=target,
        value=value,
        unit=unit,
        normalized_value=_normalized_value(finding, measurement),
        confidence=None,
        reliability=None,
        time_provenance=_time_dict(provenance),
        space_provenance=_space_dict(provenance),
        measurement_ids=list(finding.measurement_ids),
        finding_ids=[finding.finding_id],
        reportability=reportability,
        allowed_sections=allowed,
        rationale=rationale,
        caveat=_caveat_for(evidence_type, reportability),
        debug_payload={
            "finding_type": finding.finding_type,
            "measurement_name": measurement.measurement_name if measurement else None,
            "assertion": finding.assertion.value,
        },
        created_by="measurement_finding_adapter",
        created_at=EvidenceItem.now_iso(),
    )


def evidence_item_from_measurement(measurement: MeasurementValue) -> EvidenceItem:
    evidence_type, target, reportability, rationale, allowed = _classify_measurement(measurement)
    value, unit = _value_and_unit(None, measurement)
    provenance = [measurement.provenance]
    return EvidenceItem(
        evidence_id=f"ev_{measurement.measurement_id}",
        source_module=_source_module(None, measurement),
        evidence_type=evidence_type,
        clinical_target=target,
        value=value,
        unit=unit,
        normalized_value=_normalized_value(None, measurement),
        confidence=None,
        reliability=None,
        time_provenance=_time_dict(provenance),
        space_provenance=_space_dict(provenance),
        measurement_ids=[measurement.measurement_id],
        finding_ids=[],
        reportability=reportability,
        allowed_sections=allowed,
        rationale=rationale,
        caveat=_caveat_for(evidence_type, reportability),
        debug_payload={"measurement_name": measurement.measurement_name},
        created_by="measurement_adapter",
        created_at=EvidenceItem.now_iso(),
    )


def append_deliberation_evidence(board: SharedEvidenceBoard, deliberation: AgentDeliberationRecord) -> None:
    """Store reviewer outputs as audit-only evidence, never clinical ground truth."""
    for idx, item in enumerate(deliberation.weak_evidence):
        board.add_evidence(
            EvidenceItem(
                evidence_id=f"ev_{deliberation.review_id}_weak_{idx}",
                source_module="llm_reviewer",
                evidence_type=EvidenceType.LLM_ASSISTED,
                clinical_target=ClinicalTarget.UNCERTAINTY,
                value=item.severity.value,
                unit=None,
                confidence=None,
                reliability=0.25,
                measurement_ids=list(item.linked_measurement_ids),
                finding_ids=list(item.linked_finding_ids),
                reportability=ClaimSurfaceAction.DEBUG_ONLY,
                rationale="LLM-assisted weak-evidence record is audit-only and cannot directly surface.",
                debug_payload={"record_type": "weak_evidence", "target_id": item.target_id},
                created_by="evidence_reviewer_adapter",
                created_at=EvidenceItem.now_iso(),
            )
        )
    for idx, item in enumerate(deliberation.missing_slots):
        board.add_evidence(
            EvidenceItem(
                evidence_id=f"ev_{deliberation.review_id}_missing_{idx}",
                source_module="llm_reviewer",
                evidence_type=EvidenceType.LLM_ASSISTED,
                clinical_target=ClinicalTarget.UNCERTAINTY,
                value=item.slot_name,
                unit=None,
                confidence=None,
                reliability=0.25,
                measurement_ids=[],
                finding_ids=list(item.linked_finding_ids),
                reportability=ClaimSurfaceAction.DEBUG_ONLY,
                rationale="LLM-assisted missing-slot record is audit-only and cannot directly surface.",
                debug_payload={"record_type": "missing_slot", "target_module": item.target_module},
                created_by="evidence_reviewer_adapter",
                created_at=EvidenceItem.now_iso(),
            )
        )
    for idx, item in enumerate(deliberation.do_not_claim):
        board.add_evidence(
            EvidenceItem(
                evidence_id=f"ev_{deliberation.review_id}_do_not_claim_{idx}",
                source_module="llm_reviewer",
                evidence_type=EvidenceType.LLM_ASSISTED,
                clinical_target=ClinicalTarget.UNCERTAINTY,
                value="do_not_claim",
                unit=None,
                confidence=None,
                reliability=0.25,
                measurement_ids=[],
                finding_ids=list(item.linked_finding_ids),
                reportability=ClaimSurfaceAction.DEBUG_ONLY,
                rationale="LLM-assisted do-not-claim record is audit-only and cannot directly surface.",
                debug_payload={"record_type": "do_not_claim", "item_id": item.item_id},
                created_by="evidence_reviewer_adapter",
                created_at=EvidenceItem.now_iso(),
            )
        )
    for idx, item in enumerate(deliberation.claim_constraints):
        board.add_evidence(
            EvidenceItem(
                evidence_id=f"ev_{deliberation.review_id}_constraint_{idx}",
                source_module="llm_reviewer",
                evidence_type=EvidenceType.LLM_ASSISTED,
                clinical_target=ClinicalTarget.UNCERTAINTY,
                value=item.target,
                unit=None,
                confidence=None,
                reliability=0.25,
                measurement_ids=[],
                finding_ids=list(item.linked_finding_ids),
                reportability=ClaimSurfaceAction.DEBUG_ONLY,
                rationale="LLM-assisted claim-constraint record is audit-only and cannot directly surface.",
                debug_payload={"record_type": "claim_constraint", "constraint_id": item.constraint_id},
                created_by="evidence_reviewer_adapter",
                created_at=EvidenceItem.now_iso(),
            )
        )


def _classify_finding(
    finding: Finding,
    measurement: MeasurementValue | None,
) -> tuple[EvidenceType, ClinicalTarget, ClaimSurfaceAction, str, List[str]]:
    mname = measurement.measurement_name if measurement else ""
    if finding.finding_type in _EVENT_CANDIDATE_FINDINGS or mname in _EVENT_CANDIDATE_MEASUREMENTS:
        return EvidenceType.PROXY, ClinicalTarget.EVENT_CANDIDATE, ClaimSurfaceAction.DEBUG_ONLY, "Event candidate screens are proxy evidence, not seizure or definite epileptiform evidence.", []
    if finding.finding_type in _LOCALIZATION_PROXY_FINDINGS or mname in _LOCALIZATION_PROXY_MEASUREMENTS:
        return EvidenceType.PROXY, ClinicalTarget.LOCALIZATION, ClaimSurfaceAction.DEBUG_ONLY, "Localization screens are proxy evidence until claim-gated with morphology and spatial provenance.", []
    if finding.finding_type in _MORPHOLOGY_PROXY_FINDINGS or mname in _MORPHOLOGY_PROXY_MEASUREMENTS:
        return EvidenceType.DEBUG, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY, ClaimSurfaceAction.DEBUG_ONLY, "Morphology screens are debug/proxy evidence until validated morphology support exists.", []
    if mname in _DEBUG_SCORE_MEASUREMENTS:
        target = ClinicalTarget.SEIZURE_EVIDENCE if "seizure" in mname else ClinicalTarget.UNCERTAINTY
        return EvidenceType.DEBUG, target, ClaimSurfaceAction.DEBUG_ONLY, "Internal score measurements are audit/debug evidence only.", []
    if finding.finding_type == "background_pdr_frequency":
        if finding.assertion == StatusSemantic.PRESENT and _has_time_or_space(_linked_provenance(finding, measurement)):
            action = ClaimSurfaceAction.ALLOW if measurement is not None and measurement.metadata.get("pdr_supported") == "true" else ClaimSurfaceAction.CAVEAT
            return EvidenceType.DERIVED, ClinicalTarget.PDR, action, "PDR evidence is derived and may surface only with valid provenance and policy gating.", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.SLEEP.value]
        return EvidenceType.PROXY, ClinicalTarget.PDR, ClaimSurfaceAction.BLOCK, "PDR evidence lacks sufficient provenance for report surface.", []
    if finding.finding_type == "background_frequency":
        return EvidenceType.PROXY, ClinicalTarget.BACKGROUND_SLOWING, ClaimSurfaceAction.DEBUG_ONLY, "Global dominant frequency is not a PDR claim and remains debug/proxy evidence.", []
    if finding.finding_type == "background_amplitude_range":
        return EvidenceType.DERIVED, ClinicalTarget.BACKGROUND_AMPLITUDE, ClaimSurfaceAction.CAVEAT, "Amplitude range can surface as caveated provenance-linked evidence.", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value]
    if finding.finding_type == "background_slowing":
        action = ClaimSurfaceAction.CAVEAT if finding.assertion == StatusSemantic.PRESENT else ClaimSurfaceAction.BLOCK
        return EvidenceType.DERIVED, ClinicalTarget.BACKGROUND_SLOWING, action, "Slowing is derived local-tool evidence and requires caveated prose.", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.IMPRESSION.value]
    if finding.finding_type == "excess_beta":
        action = ClaimSurfaceAction.CAVEAT if finding.assertion == StatusSemantic.PRESENT else ClaimSurfaceAction.BLOCK
        return EvidenceType.DERIVED, ClinicalTarget.EXCESS_BETA, action, "Excess beta is derived local-tool evidence and requires caveated prose.", [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.IMPRESSION.value]
    if finding.finding_type in _PROTOCOL_FINDINGS or finding.finding_type.startswith("protocol_"):
        action = ClaimSurfaceAction.ALLOW if finding.assertion != StatusSemantic.UNKNOWN else ClaimSurfaceAction.BLOCK
        target = ClinicalTarget.STATE if "state" in finding.finding_type else ClinicalTarget.PROTOCOL
        return EvidenceType.METADATA, target, action, "Structured protocol/context status may surface when non-unknown.", [SectionRole.DETAIL.value, SectionRole.BACKGROUND.value, SectionRole.SLEEP.value]
    return EvidenceType.DERIVED, ClinicalTarget.UNKNOWN, ClaimSurfaceAction.BLOCK, "Unmapped finding type is retained in evidence board but not reportable.", []


def _classify_measurement(measurement: MeasurementValue) -> tuple[EvidenceType, ClinicalTarget, ClaimSurfaceAction, str, List[str]]:
    name = measurement.measurement_name
    if name in _EVENT_CANDIDATE_MEASUREMENTS:
        return EvidenceType.PROXY, ClinicalTarget.EVENT_CANDIDATE, ClaimSurfaceAction.DEBUG_ONLY, "Candidate burden/duration measurements are proxy evidence.", []
    if name in _LOCALIZATION_PROXY_MEASUREMENTS:
        return EvidenceType.PROXY, ClinicalTarget.LOCALIZATION, ClaimSurfaceAction.DEBUG_ONLY, "Localization ratio/label measurements are proxy evidence.", []
    if name in _MORPHOLOGY_PROXY_MEASUREMENTS or name in _DEBUG_SCORE_MEASUREMENTS or name.startswith("relative_bandpower_"):
        return EvidenceType.DEBUG, ClinicalTarget.UNCERTAINTY, ClaimSurfaceAction.DEBUG_ONLY, "Internal measurement is debug evidence only.", []
    if name.startswith("protocol_") or name.endswith("_status"):
        return EvidenceType.METADATA, ClinicalTarget.PROTOCOL, ClaimSurfaceAction.ALLOW, "Structured metadata/status measurement may surface through a finding.", [SectionRole.DETAIL.value]
    return EvidenceType.DERIVED, ClinicalTarget.UNKNOWN, ClaimSurfaceAction.BLOCK, "Measurement requires mapped finding before report surface.", []


def _source_module(finding: Finding | None, measurement: MeasurementValue | None) -> str:
    prov = measurement.provenance if measurement is not None else None
    if prov and prov.source_type == SourceType.METADATA:
        return "metadata"
    if prov and prov.source_type == SourceType.REPORT_TEXT:
        return "state_protocol"
    if measurement and (measurement.measurement_name.startswith("event_") or "epileptiform" in measurement.measurement_name):
        return "event"
    if measurement and measurement.measurement_name.startswith("background"):
        return "background"
    return "unknown"


def _linked_provenance(finding: Finding | None, measurement: MeasurementValue | None) -> List[ProvenanceRecord]:
    if measurement is not None:
        return [measurement.provenance]
    return list(finding.provenance) if finding is not None else []


def _value_and_unit(finding: Finding | None, measurement: MeasurementValue | None) -> tuple[Any, str | None]:
    q = finding.quantitation if finding and finding.quantitation is not None else (measurement.quantitation if measurement else None)
    if q is not None:
        if q.exact is not None:
            return q.exact, q.unit
        if q.lower is not None or q.upper is not None:
            return {"lower": q.lower, "upper": q.upper}, q.unit
        if q.values:
            return list(q.values), q.unit
    if measurement is not None:
        if measurement.status_value is not None:
            return measurement.status_value.status.value, None
        if measurement.categorical_value is not None:
            return measurement.categorical_value, None
        if measurement.boolean_value is not None:
            return measurement.boolean_value, None
    return None, None


def _normalized_value(finding: Finding | None, measurement: MeasurementValue | None) -> Any:
    value, _unit = _value_and_unit(finding, measurement)
    return value


def _time_dict(provenance: List[ProvenanceRecord]) -> dict[str, Any] | None:
    windows: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    for p in provenance:
        windows.extend(p.time.window_indices)
        if p.time.start_sec is not None:
            starts.append(p.time.start_sec)
        if p.time.end_sec is not None:
            ends.append(p.time.end_sec)
    if not windows and not starts and not ends:
        return None
    return {
        "start_sec": min(starts) if starts else None,
        "end_sec": max(ends) if ends else None,
        "window_id": sorted(set(windows))[0] if windows else None,
        "window_indices": sorted(set(windows)),
        "epoch_id": None,
    }


def _space_dict(provenance: List[ProvenanceRecord]) -> dict[str, Any] | None:
    channels: list[str] = []
    regions: list[str] = []
    sides: list[str] = []
    for p in provenance:
        channels.extend(p.space.channels)
        if p.space.region:
            regions.append(p.space.region)
        if p.space.laterality:
            sides.append(p.space.laterality)
    if not channels and not regions and not sides:
        return None
    return {
        "channels": sorted(set(channels)),
        "montage": None,
        "region": sorted(set(regions))[0] if regions else None,
        "side": sorted(set(sides))[0] if sides else None,
        "electrode_maxima": sorted(set(channels))[:4],
    }


def _has_time_or_space(provenance: List[ProvenanceRecord]) -> bool:
    return bool(_time_dict(provenance) or _space_dict(provenance))


def _reliability(evidence_type: EvidenceType, reportability: ClaimSurfaceAction, provenance: List[ProvenanceRecord]) -> float | None:
    base = {
        EvidenceType.DIRECT: 0.9,
        EvidenceType.METADATA: 0.85,
        EvidenceType.DERIVED: 0.65,
        EvidenceType.LLM_ASSISTED: 0.45,
        EvidenceType.PROXY: 0.35,
        EvidenceType.DEBUG: 0.2,
    }[evidence_type]
    if not provenance:
        base -= 0.15
    if reportability in {ClaimSurfaceAction.DEBUG_ONLY, ClaimSurfaceAction.BLOCK}:
        base = min(base, 0.35)
    return max(base, 0.0)


def _caveat_for(evidence_type: EvidenceType, reportability: ClaimSurfaceAction) -> str | None:
    if reportability == ClaimSurfaceAction.CAVEAT:
        return "Surface text must preserve evidence limitations and should not upgrade certainty."
    if evidence_type in {EvidenceType.PROXY, EvidenceType.DEBUG, EvidenceType.LLM_ASSISTED}:
        return "Audit/provenance evidence only; do not surface as clinical prose without a gated atomic claim."
    return None
