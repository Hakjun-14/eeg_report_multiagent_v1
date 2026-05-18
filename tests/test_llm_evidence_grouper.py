from eeg_report_multiagent.modules.llm_evidence_grouper import LLMEvidenceGrouper
from eeg_report_multiagent.schemas.measurement import MeasurementValue, QuantitationKind, QuantitationValue
from eeg_report_multiagent.schemas.provenance import MeasurementProvenance, ProvenanceRecord, SourceType, SpaceProvenance, TimeProvenance
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceType


class FakeGroupingAdapter:
    model = "fake-grouping"

    def __init__(self):
        self.payload = None

    def group(self, payload):
        self.payload = payload
        assert payload["privacy_contract"]["contains_raw_eeg"] is False
        assert payload["privacy_contract"]["contains_gt_report_text"] is False
        assert "signals" not in payload
        assert "gt_report" not in payload
        return {
            "summary": "grouped evidence",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "evidence_groups": [
                {
                    "evidence_id": "pdr_group",
                    "clinical_target": "pdr",
                    "evidence_type": "derived",
                    "value_summary": "Posterior alpha candidate from occipital frequency measurement.",
                    "linked_measurement_ids": ["m_pdr", "missing_id"],
                    "allowed_sections": ["background", "detail"],
                    "clinical_knowledge_reference": {
                        "reference_type": "required_but_not_provided",
                        "statement": "PDR requires posterior alpha support and state/reactivity context.",
                    },
                    "rationale": "Measurement has occipital spatial provenance.",
                },
                {
                    "evidence_id": "empty_group",
                    "clinical_target": "event_candidate",
                    "evidence_type": "proxy",
                    "value_summary": "Should be skipped because no valid measurement link exists.",
                    "linked_measurement_ids": ["missing_id"],
                    "allowed_sections": ["epileptiform"],
                    "clinical_knowledge_reference": {
                        "reference_type": "required_but_not_provided",
                        "statement": "Candidate burden alone is not morphology.",
                    },
                    "rationale": "No valid link.",
                },
            ],
        }


def _measurement():
    return MeasurementValue(
        measurement_id="m_pdr",
        measurement_name="pdr_candidate_frequency_hz",
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=9.5, unit="Hz"),
        provenance=ProvenanceRecord(
            source_type=SourceType.SIGNAL,
            source_ref="s",
            time=TimeProvenance(window_indices=[1], start_sec=10.0, end_sec=20.0),
            space=SpaceProvenance(channels=["O1", "O2"], region="occipital", laterality="bilateral"),
            measurement=MeasurementProvenance(tool_name="background", function_name="pdr_candidate"),
        ),
    )


def test_llm_evidence_grouper_creates_board_from_measurement_only_payload():
    adapter = FakeGroupingAdapter()
    result = LLMEvidenceGrouper(adapter=adapter).run(recording_id="s", measurements=[_measurement()])

    assert result["status"] == "ok"
    assert result["raw_eeg_used"] is False
    assert result["gt_report_used"] is False
    board = result["shared_evidence_board"]
    assert len(board.evidence_items) == 1
    item = board.evidence_items[0]
    assert item.evidence_id == "ev_llm_pdr_group"
    assert item.clinical_target == ClinicalTarget.PDR
    assert item.evidence_type == EvidenceType.DERIVED
    assert item.reportability == ClaimSurfaceAction.CAVEAT
    assert item.measurement_ids == ["m_pdr"]
    assert item.finding_ids == []
    assert item.space_provenance["region"] == "occipital"
