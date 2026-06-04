from __future__ import annotations

from eeg_report_multiagent.modules.llm_report_synthesizer import EvidenceBoardLLMReportSynthesizer
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.agent import (
    AgentDeliberationRecord,
    ClaimConstraintRecord,
    DoNotClaimRecord,
    EvidenceGapSeverity,
    MissingSlotRecord,
    WeakEvidenceRecord,
)
from eeg_report_multiagent.schemas import EvidenceBoard, MeasurementValue, QuantitationValue
from eeg_report_multiagent.schemas.measurement import QuantitationKind
from eeg_report_multiagent.schemas.provenance import MeasurementProvenance, ProvenanceRecord, SourceType, TimeProvenance


FORBIDDEN_SURFACE_SNIPPETS = [
    "candidate burden",
    "longest candidate train",
    "laterality index",
    "bifrontal spread tendency",
    "morphology screen classified",
    "morphology screen",
    "support score",
    "likelihood score",
    "field concentration ratio",
    "missing_slots",
    "weak_evidence",
    "do_not_claim",
    "claim_constraints",
    "evidence review:",
    "low-frequency boundary peak",
]


def _prov() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=SourceType.SIGNAL,
        source_ref="row_000189_like_session",
        time=TimeProvenance(window_indices=[0, 1, 2]),
        measurement=MeasurementProvenance(tool_name="surface_policy_canary", function_name="surface_policy_canary"),
    )


def _exact(mid: str, name: str, value: float, unit: str = "score") -> MeasurementValue:
    return MeasurementValue(
        measurement_id=mid,
        measurement_name=name,
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=value, unit=unit),
        provenance=_prov(),
    )


def _proxy_only_row189_like_board() -> EvidenceBoard:
    measurements = [
        _exact("m_freq", "relative_bandpower_delta", 0.5, "ratio"),
        _exact("m_burden", "event_candidate_burden_ratio", 0.22, "ratio"),
        _exact("m_train", "event_train_duration_upper_sec", 110.0, "sec"),
        _exact("m_lat", "event_laterality_index", 0.41, "index"),
        _exact("m_bifrontal", "event_bifrontal_ratio", 1.7, "ratio"),
        _exact("m_morph", "event_morphology_support_score", 1.2, "score"),
        _exact("m_field", "event_peak_field_concentration_ratio", 2.4, "ratio"),
        _exact("m_like", "epileptiform_candidate_likelihood_score", 0.8, "score"),
    ]
    return EvidenceBoard(session_id="row_000189_like", measurements=measurements)


def _raw_reviewer_text() -> AgentDeliberationRecord:
    return AgentDeliberationRecord(
        review_id="review_canary",
        reviewer_name="llm_evidence_review",
        status="ok",
        summary="Raw reviewer text must remain audit-only.",
        weak_evidence=[
            WeakEvidenceRecord(
                weakness_id="weak_raw_text",
                severity=EvidenceGapSeverity.HIGH,
                target_type="measurement",
                target_id="m_burden",
                reason="No morphology descriptor is present.",
                recommendation="Keep event evidence as a candidate.",
                linked_measurement_ids=["m_burden"],
            )
        ],
        missing_slots=[
            MissingSlotRecord(
                slot_id="slot_morphology",
                slot_name="epileptiform_morphology",
                target_module="event",
                severity=EvidenceGapSeverity.HIGH,
                reason="morphology missing_slots text should not reach the LLM report payload",
                expected_evidence="validated spike/sharp morphology",
                linked_measurement_ids=["m_burden"],
            )
        ],
        do_not_claim=[
            DoNotClaimRecord(
                item_id="no_seizure_from_burden",
                text="Do not claim seizure from event candidate burden.",
                rationale="No seizure evolution evidence is present.",
                linked_measurement_ids=["m_burden"],
            )
        ],
        claim_constraints=[
            ClaimConstraintRecord(
                constraint_id="no_lateralization",
                target="event_evidence",
                constraint="Do not surface laterality index as clinical localization.",
                rationale="Proxy localization only.",
                linked_measurement_ids=["m_lat"],
            )
        ],
    )


def _all_text(sections: dict[str, str]) -> str:
    return "\n".join(sections.values()).lower()


def test_celm_compatible_sections_do_not_surface_proxy_or_debug_terms() -> None:
    sections = ReportSynthesizer().synthesize_celm_sections(
        _proxy_only_row189_like_board(),
        [
            "EEG DESCRIPTION/DETAILS",
            "BACKGROUND ACTIVITY",
            "EPLEPTIFORM ABNORMALITIES",
            "EVENTS/SEIZURES",
            "SEIZURES",
            "IMPRESSION/INTERPRETATION",
        ],
    )

    rendered = _all_text(sections)
    for snippet in FORBIDDEN_SURFACE_SNIPPETS:
        assert snippet not in rendered
    assert "posterior dominant" not in rendered
    assert "pdr" not in rendered
    assert sections["SEIZURES"] == "Seizures: no seizure-specific evidence was produced by the current structured tools."
    assert "transient" not in sections["SEIZURES"].lower()
    assert "epileptiform" not in sections["SEIZURES"].lower()
    assert "abnormal" not in sections["IMPRESSION/INTERPRETATION"].lower()


def test_llm_payload_contains_only_surface_allowed_atomic_claim_plans() -> None:
    class CapturingAdapter:
        model = "fake"

        def __init__(self) -> None:
            self.payload = None

        def synthesize(self, evidence_payload):
            self.payload = evidence_payload
            return {
                "report_sections": [],
                "global_limitations": [],
                "raw_eeg_used": False,
                "gt_report_used": False,
            }

    adapter = CapturingAdapter()
    board = _proxy_only_row189_like_board()
    board.deliberations = [_raw_reviewer_text()]
    result = EvidenceBoardLLMReportSynthesizer(adapter=adapter).synthesize_celm_sections(
        board,
        ["EEG DESCRIPTION/DETAILS", "SEIZURES"],
    )

    assert adapter.payload is not None
    assert adapter.payload["atomic_claim_plans"] == []
    assert "measurements" not in adapter.payload
    payload_text = str(adapter.payload).lower()
    for snippet in FORBIDDEN_SURFACE_SNIPPETS:
        assert snippet not in payload_text
    assert "do not claim seizure" not in payload_text
    assert "morphology missing_slots text" not in payload_text
    assert "no morphology descriptor is present" not in payload_text
    assert result.section_texts["SEIZURES"] == "Seizures: no seizure-specific evidence was produced by the current structured tools."
