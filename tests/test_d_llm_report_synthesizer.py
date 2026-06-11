from eeg_report_multiagent.modules.llm_report_synthesizer import EvidenceBoardLLMReportSynthesizer
from eeg_report_multiagent.schemas import EvidenceBoard, MeasurementValue, QuantitationValue
from eeg_report_multiagent.schemas.measurement import QuantitationKind
from eeg_report_multiagent.schemas.provenance import (
    MeasurementProvenance,
    ProvenanceRecord,
    SourceType,
    TimeProvenance,
)
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType


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
        measurement=MeasurementProvenance(tool_name="background_amplitude", function_name="background_amplitude"),
    )
    measurement = MeasurementValue(
        measurement_id="m_amp",
        measurement_name="background_amplitude_range_uv",
        quantitation=QuantitationValue(kind=QuantitationKind.RANGE, lower=70.0, upper=90.0, unit="uV"),
        provenance=prov,
    )
    return EvidenceBoard(session_id="s", measurements=[measurement])


def test_d_synthesizer_uses_evidence_board_only_payload() -> None:
    adapter = FakeReportAdapter()
    clinical_context = {
        "patient_history_and_eeg_description": "Patient history available to align with CELM-style prompting.",
        "metadata": {"indication": "spell characterization"},
        "gt_report_text_included": False,
    }
    result = EvidenceBoardLLMReportSynthesizer(adapter=adapter).synthesize_celm_sections(
        _board(),
        ["EEG DESCRIPTION/DETAILS", "IMPRESSION/INTERPRETATION"],
        clinical_context=clinical_context,
    )

    assert result.section_texts["EEG DESCRIPTION/DETAILS"].startswith("Structured evidence")
    assert result.trace["raw_eeg_used"] is False
    assert result.trace["gt_report_used"] is False
    assert "atomic_claim_plans" in adapter.payload
    assert adapter.payload["clinical_context"] == clinical_context
    assert "section_descriptions" in adapter.payload
    assert "background activity" in adapter.payload["section_descriptions"]["EEG DESCRIPTION/DETAILS"].lower()
    assert adapter.payload["atomic_claim_plans"][0]["claim_type"] == "background_amplitude"
    assert adapter.payload["atomic_claim_plans"][0]["linked_reportable_evidence"]
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


def test_d_synthesizer_prefers_supplied_atomic_claim_plan_and_links_evidence_view() -> None:
    adapter = FakeReportAdapter()
    claim_plan = [
        AtomicClaimPlan(
            plan_id="p_llm_background_amplitude",
            section_type=ReportSectionType.DETAIL,
            claim_type="background_amplitude",
            proposed_text="LLM-planned background amplitude claim with linked evidence.",
            evidence_ids=["evgrp_background_amplitude"],
            linked_measurement_ids=["m_amp"],
            surface_action=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            required_evidence=["background_amplitude_measurement"],
            missing_evidence=[],
            rationale="LLM claim planning output should be reused by D synthesis.",
        )
    ]

    result = EvidenceBoardLLMReportSynthesizer(adapter=adapter).synthesize_celm_sections(
        _board(),
        ["EEG DESCRIPTION/DETAILS"],
        claim_plan_override=claim_plan,
    )

    assert result.trace["claim_plan_source"] == "artifact_atomic_claim_plan"
    assert adapter.payload["claim_plan_source"] == "artifact_atomic_claim_plan"
    payload_claim = adapter.payload["atomic_claim_plans"][0]
    assert payload_claim["plan_id"] == "p_llm_background_amplitude"
    assert payload_claim["proposed_text"] == "LLM-planned background amplitude claim with linked evidence."
    linked = payload_claim["linked_reportable_evidence"]
    assert linked[0]["evidence_id"] == "evgrp_background_amplitude"
    assert linked[0]["clinical_target"] == "background_amplitude"
    assert linked[0]["value"]["background_amplitude_range_uv"] == {"lower": 70.0, "upper": 90.0}
    assert linked[0]["value"]["background_amplitude_typical_uv"] is None
    assert payload_claim["surface_value_requirements"] == ["preserve background amplitude range: 70.0-90.0 uV"]
    assert linked[0]["unit"] == "uV"
    assert "debug_payload" not in linked[0]
    assert result.section_texts["EEG DESCRIPTION/DETAILS"].startswith("Structured evidence")


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
