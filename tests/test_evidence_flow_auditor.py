from __future__ import annotations

from eeg_report_multiagent.modules.evidence_flow_auditor import EvidenceFlowAuditor
from eeg_report_multiagent.schemas import EvidenceBoard, MeasurementValue, QuantitationValue
from eeg_report_multiagent.schemas.measurement import QuantitationKind
from eeg_report_multiagent.schemas.provenance import MeasurementProvenance, ProvenanceRecord, SourceType, SpaceProvenance, TimeProvenance
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


def _prov() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=SourceType.SIGNAL,
        source_ref="s",
        time=TimeProvenance(window_indices=[0], start_sec=0.0, end_sec=10.0),
        space=SpaceProvenance(channels=["O1", "O2"], region="posterior", laterality=None),
        measurement=MeasurementProvenance(tool_name="tool", function_name="fn"),
    )


def _measurement(mid: str, name: str, value: float, unit: str) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=mid,
        measurement_name=name,
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=value, unit=unit),
        provenance=_prov(),
    )


def test_evidence_flow_traces_surfaced_pdr_candidate() -> None:
    measurement = _measurement("m_pdr_candidate_frequency", "pdr_candidate_frequency", 9.0, "Hz")
    board = EvidenceBoard(session_id="s", measurements=[measurement])
    shared = SharedEvidenceBoard(board_id="seb", recording_id="s")
    shared.add_evidence(EvidenceItem(
        evidence_id="ev_pdr",
        source_module="background",
        evidence_type=EvidenceType.DERIVED,
        clinical_target=ClinicalTarget.PDR,
        value=9.0,
        unit="Hz",
        normalized_value=9.0,
        measurement_ids=[measurement.measurement_id],
        reportability=ClaimSurfaceAction.CAVEAT,
        allowed_sections=["background"],
        time_provenance={"start_sec": 0.0, "end_sec": 10.0},
        space_provenance={"channels": ["O1", "O2"], "region": "posterior"},
        created_by="test",
    ))
    claim = AtomicClaimPlan(
        plan_id="plan_pdr",
        section_type=ReportSectionType.DETAIL,
        claim_type="background_pdr_frequency",
        proposed_text="A posterior alpha rhythm candidate near 9.0 Hz is supported by structured evidence.",
        evidence_ids=["ev_pdr"],
        linked_measurement_ids=[measurement.measurement_id],
        surface_action=ClaimSurfaceAction.CAVEAT,
    )
    result = EvidenceFlowAuditor().audit_case(
        case_id="case",
        variant="v",
        evidence_board=board,
        shared_evidence_board=shared,
        atomic_claims=[claim],
        final_report={"BACKGROUND ACTIVITY": claim.proposed_text},
    )
    pdr = next(record for record in result.slot_records if record.clinical_slot == "pdr_frequency")

    assert pdr.measurement_exists
    assert pdr.evidence_item_exists
    assert pdr.atomic_claim_exists
    assert pdr.surfaced_in_final_prose
    assert not pdr.suppression_reasons


def test_evidence_flow_marks_useful_but_suppressed_when_claim_blocked() -> None:
    measurement = _measurement("m_pdr_candidate_frequency", "pdr_candidate_frequency", 9.0, "Hz")
    board = EvidenceBoard(session_id="s", measurements=[measurement])
    shared = SharedEvidenceBoard(board_id="seb", recording_id="s")
    shared.add_evidence(EvidenceItem(
        evidence_id="ev_pdr",
        source_module="background",
        evidence_type=EvidenceType.DERIVED,
        clinical_target=ClinicalTarget.PDR,
        value=9.0,
        unit="Hz",
        normalized_value=9.0,
        measurement_ids=[measurement.measurement_id],
        reportability=ClaimSurfaceAction.CAVEAT,
        allowed_sections=["background"],
        time_provenance={"start_sec": 0.0, "end_sec": 10.0},
        space_provenance={"channels": ["O1", "O2"], "region": "posterior"},
        created_by="test",
    ))
    claim = AtomicClaimPlan(
        plan_id="plan_pdr",
        section_type=ReportSectionType.DETAIL,
        claim_type="background_pdr_frequency",
        proposed_text="No surface-allowed structured evidence was available for this section.",
        evidence_ids=["ev_pdr"],
        surface_action=ClaimSurfaceAction.BLOCK,
    )
    result = EvidenceFlowAuditor().audit_case(
        case_id="case",
        variant="v",
        evidence_board=board,
        shared_evidence_board=shared,
        atomic_claims=[claim],
        final_report={"BACKGROUND ACTIVITY": "No surface-allowed structured evidence was available for this section."},
    )
    pdr = next(record for record in result.slot_records if record.clinical_slot == "pdr_frequency")

    assert pdr.evidence_item_exists
    assert pdr.atomic_claim_exists
    assert not pdr.surfaced_in_final_prose
    assert pdr.useful_but_suppressed
    assert "atomic_claim_blocked" in pdr.suppression_reasons


def test_evidence_flow_aggregate_recommends_extraction_when_slots_missing() -> None:
    result = EvidenceFlowAuditor().audit_case(
        case_id="empty",
        variant="v",
        evidence_board=EvidenceBoard(session_id="s"),
        shared_evidence_board=SharedEvidenceBoard(board_id="seb", recording_id="s"),
        atomic_claims=[],
        final_report={},
    )
    aggregate = EvidenceFlowAuditor().aggregate_selected50([result], variant="v")

    assert aggregate.num_cases == 1
    assert aggregate.per_slot_availability["pdr_frequency"]["measurement_rate"] == 0.0
    assert aggregate.aggregate_recommendation_for_stage3.startswith("Stage 3A")
