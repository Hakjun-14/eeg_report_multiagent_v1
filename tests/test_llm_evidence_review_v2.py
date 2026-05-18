from eeg_report_multiagent.modules.evidence_reviewer import EvidenceReviewModule
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas import EvidenceBoard, Finding, MeasurementValue, QuantitationValue
from eeg_report_multiagent.schemas.measurement import QuantitationKind, StatusSemantic
from eeg_report_multiagent.schemas.provenance import (
    MeasurementProvenance,
    ProvenanceRecord,
    SourceType,
    TimeProvenance,
)


class FakeAdapter:
    model = "fake-review-model"

    def __init__(self) -> None:
        self.payload = None

    def review(self, evidence_payload):
        self.payload = evidence_payload
        return {
            "summary": "LLM reviewer checked structured evidence only.",
            "evidence_gaps": [],
            "weak_evidence": [
                {
                    "weakness_id": "llm_weak_event_morphology",
                    "severity": "high",
                    "target_type": "finding",
                    "target_id": "f_event",
                    "reason": "No morphology descriptor is present.",
                    "linked_measurement_ids": ["m_event"],
                    "linked_finding_ids": ["f_event"],
                    "recommendation": "Keep event finding as a candidate.",
                }
            ],
            "missing_slots": [],
            "do_not_claim": [
                {
                    "item_id": "llm_do_not_claim_seizure",
                    "text": "Do not claim seizure from event candidate burden.",
                    "rationale": "No seizure evolution evidence is present.",
                    "linked_finding_ids": ["f_event"],
                }
            ],
            "claim_constraints": [],
            "tool_request_proposals": [
                {
                    "proposal_id": "bad_tool",
                    "target_module": "event",
                    "tool_name": "unregistered_magic_tool",
                    "rationale": "Should be rejected.",
                    "expected_measurement": "invalid",
                    "linked_gap_ids": [],
                    "linked_finding_ids": ["f_event"],
                }
            ],
        }


def _signal_prov(tool_name: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=SourceType.SIGNAL,
        source_ref="s",
        time=TimeProvenance(window_indices=[0, 1]),
        measurement=MeasurementProvenance(tool_name=tool_name, function_name=tool_name),
    )


def _board() -> EvidenceBoard:
    m_freq = MeasurementValue(
        measurement_id="m_freq",
        measurement_name="background_dominant_frequency_hz",
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=0.5, unit="Hz"),
        provenance=_signal_prov("psd_summary"),
    )
    f_freq = Finding(
        finding_id="f_freq",
        finding_type="background_frequency",
        assertion=StatusSemantic.PRESENT,
        quantitation=m_freq.quantitation,
        measurement_ids=["m_freq"],
        provenance=[m_freq.provenance],
        source_module="background_module",
    )
    m_event = MeasurementValue(
        measurement_id="m_event",
        measurement_name="event_candidate_burden_ratio",
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=0.2, unit="ratio"),
        provenance=_signal_prov("transient_candidate_score"),
    )
    f_event = Finding(
        finding_id="f_event",
        finding_type="epileptiform_event_candidate_burden",
        assertion=StatusSemantic.PRESENT,
        quantitation=m_event.quantitation,
        measurement_ids=["m_event"],
        provenance=[m_event.provenance],
        source_module="event_module",
    )
    return EvidenceBoard(session_id="s", measurements=[m_freq, m_event], findings=[f_freq, f_event])


def test_evidence_review_v2_uses_structured_payload_and_rejects_unregistered_tools() -> None:
    adapter = FakeAdapter()
    review = EvidenceReviewModule(adapter=adapter).run(_board())

    assert review.status == "ok"
    assert review.review_version == "v2"
    assert review.raw_eeg_used is False
    assert review.gt_report_used is False
    assert review.weak_evidence
    assert review.missing_slots
    assert review.do_not_claim
    assert review.claim_constraints
    assert review.rejected_tool_request_proposals[0].tool_name == "unregistered_magic_tool"
    assert adapter.payload["privacy_contract"] == {
        "contains_raw_eeg": False,
        "contains_gt_report_text": False,
        "contains_source_pkl_paths": False,
    }
    assert "raw_eeg_interpretation" in adapter.payload["review_policy"]["forbidden"]


def test_report_synthesizer_reflects_evidence_review_constraints() -> None:
    board = _board()
    board.deliberations = [EvidenceReviewModule(adapter=FakeAdapter()).run(board)]

    detail, impression, claims = ReportSynthesizer().synthesize(board)

    assert "Evidence review:" not in detail.text
    assert "event candidate burden" not in detail.text.lower()
    assert "candidates" not in impression.text.lower()
    assert "Evidence gaps to address" not in impression.text
    assert "No surface-allowed" in detail.text
    assert claims
