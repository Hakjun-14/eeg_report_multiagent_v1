from eeg_report_multiagent.modules.llm_report_synthesizer import EvidenceBoardLLMReportSynthesizer
from eeg_report_multiagent.schemas import EvidenceBoard, MeasurementValue, QuantitationValue
from eeg_report_multiagent.schemas.measurement import QuantitationKind
from eeg_report_multiagent.schemas.provenance import (
    MeasurementProvenance,
    ProvenanceRecord,
    SourceType,
    TimeProvenance,
)


class FakeReportAdapter:
    model = "fake-report-model"

    def __init__(self) -> None:
        self.payload = None

    def synthesize(self, evidence_payload):
        self.payload = evidence_payload
        return {
            "report_sections": [
                {
                    "section_name": "EEG DESCRIPTION/DETAILS",
                    "section_text": "Structured evidence suggests background slowing; this remains an assistive observation pending EEG review.",
                    "supporting_evidence_ids": ["evgrp_background_slowing"],
                    "evidence_limitations": ["surface-policy-gated evidence only"],
                },
                {
                    "section_name": "IMPRESSION/INTERPRETATION",
                    "section_text": "Structured evidence suggests background slowing; this remains an assistive observation pending EEG review.",
                    "supporting_evidence_ids": ["evgrp_background_slowing"],
                    "evidence_limitations": ["neurologist review required"],
                },
            ],
            "global_limitations": ["No raw EEG was provided to the LLM."],
            "raw_eeg_used": False,
            "gt_report_used": False,
            "_response_id": "resp_fake",
        }


def _board() -> EvidenceBoard:
    prov = ProvenanceRecord(
        source_type=SourceType.SIGNAL,
        source_ref="session",
        time=TimeProvenance(window_indices=[0, 1]),
        measurement=MeasurementProvenance(tool_name="slowing_score", function_name="slowing_score"),
    )
    measurement = MeasurementValue(
        measurement_id="m_slow",
        measurement_name="slowing_score",
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=0.8, unit="ratio"),
        provenance=prov,
    )
    return EvidenceBoard(session_id="s", measurements=[measurement])


def test_d_synthesizer_uses_evidence_board_only_payload() -> None:
    adapter = FakeReportAdapter()
    result = EvidenceBoardLLMReportSynthesizer(adapter=adapter).synthesize_celm_sections(
        _board(),
        ["EEG DESCRIPTION/DETAILS", "IMPRESSION/INTERPRETATION"],
    )

    assert result.section_texts["EEG DESCRIPTION/DETAILS"].startswith("Structured evidence")
    assert result.trace["raw_eeg_used"] is False
    assert result.trace["gt_report_used"] is False
    assert "atomic_claim_plans" in adapter.payload
    assert adapter.payload["atomic_claim_plans"][0]["claim_type"] == "background_slowing"
    assert "measurements" not in adapter.payload
    assert "values_preview" not in str(adapter.payload)
    assert "support score" not in str(adapter.payload).lower()
    assert adapter.payload["privacy_contract"] == {
        "contains_raw_eeg": False,
        "contains_gt_report_text": False,
        "contains_source_pkl_paths": False,
        "contains_full_measurements": False,
        "contains_full_evidence_items": False,
        "contains_debug_scores": False,
    }
    assert "reference_gt_report_text" in adapter.payload["forbidden_inputs"]


def test_d_synthesizer_sanitizes_forbidden_model_surface_text() -> None:
    class BadSurfaceAdapter(FakeReportAdapter):
        def synthesize(self, evidence_payload):
            self.payload = evidence_payload
            return {
                "report_sections": [
                    {
                        "section_name": "EEG DESCRIPTION/DETAILS",
                        "section_text": "The candidate burden and field concentration ratio are high.",
                    }
                ],
                "global_limitations": [],
                "raw_eeg_used": False,
                "gt_report_used": False,
            }

    result = EvidenceBoardLLMReportSynthesizer(adapter=BadSurfaceAdapter()).synthesize_celm_sections(
        _board(),
        ["EEG DESCRIPTION/DETAILS"],
    )

    assert "candidate burden" not in result.section_texts["EEG DESCRIPTION/DETAILS"].lower()
    assert "field concentration ratio" not in result.section_texts["EEG DESCRIPTION/DETAILS"].lower()
    assert "No surface-allowed" in result.section_texts["EEG DESCRIPTION/DETAILS"]
