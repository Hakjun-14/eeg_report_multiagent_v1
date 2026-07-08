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
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType


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

    def synthesize_from_evidence_view(self, evidence_payload):
        self.payload = evidence_payload
        return {
            "report_sections": [
                {
                    "section_name": "EEG DESCRIPTION/DETAILS",
                    "section_text": "The background amplitude ranged from 70 to 90 uV.",
                    "supporting_evidence_ids": ["evgrp_background_amplitude"],
                    "evidence_limitations": ["diagnostic evidence-direct synthesis"],
                }
            ],
            "global_limitations": ["Evidence-direct diagnostic mode."],
            "raw_eeg_used": False,
            "gt_report_used": False,
            "_response_id": "resp_fake_evidence_direct",
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
            surface_safe_values=[
                {
                    "evidence_id": "evgrp_background_amplitude",
                    "clinical_target": "background_amplitude",
                    "value": {"background_amplitude_typical_uv": 80.0},
                    "unit": "uV",
                }
            ],
            must_render_values=["background_amplitude_typical=80.0 uV"],
            numeric_claims=[
                {
                    "slot": "background_amplitude_typical",
                    "value": 80.0,
                    "unit": "uV",
                    "evidence_id": "evgrp_background_amplitude",
                    "render_required": True,
                    "render_text": "80.0 uV",
                    "source": "test",
                }
            ],
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
    assert payload_claim["surface_safe_values"][0]["value"]["background_amplitude_typical_uv"] == 80.0
    assert payload_claim["must_render_values"] == ["background_amplitude_typical=80.0 uV"]
    assert payload_claim["surface_value_requirements"] == ["background_amplitude_typical=80.0 uV"]
    assert payload_claim["numeric_claims"][0]["render_text"] == "80.0 uV"
    assert "ifcn_background_report_structure" in payload_claim["clinical_reference_ids"]
    assert payload_claim["clinical_references"][0]["short_rule"]
    assert adapter.payload["claim_render_checklist"][0]["numeric_claims"][0]["slot"] == "background_amplitude_typical"
    assert "ifcn_background_report_structure" in adapter.payload["claim_render_checklist"][0]["clinical_reference_ids"]
    assert adapter.payload["claim_render_checklist"][0]["clinical_reference_rules"]
    assert "ifcn_background_report_structure" in adapter.payload["surface_decisions"][0]["clinical_reference_ids"]
    assert linked[0]["unit"] == "uV"
    assert "debug_payload" not in linked[0]
    assert result.section_texts["EEG DESCRIPTION/DETAILS"].startswith("Structured evidence")


def test_evidence_direct_synthesizer_sends_surface_safe_evidence_view_only() -> None:
    adapter = FakeReportAdapter()
    result = EvidenceBoardLLMReportSynthesizer(adapter=adapter).synthesize_evidence_direct_sections(
        _board(),
        ["EEG DESCRIPTION/DETAILS"],
        clinical_context={"patient_history_and_eeg_description": "spell characterization"},
    )

    assert result.trace["synthesis_version"] == "evidence_direct_diagnostic_v1"
    assert result.trace["raw_eeg_used"] is False
    assert result.trace["gt_report_used"] is False
    assert adapter.payload["diagnostic_mode"] == "evidence_direct_report_synthesis"
    assert "evidence_for_report" in adapter.payload
    assert adapter.payload["evidence_for_report"][0]["evidence_id"] == "evgrp_background_amplitude"
    assert adapter.payload["evidence_for_report"][0]["value"]["background_amplitude_range_uv"] == {"lower": 70.0, "upper": 90.0}
    assert "atomic_claim_plans" not in adapter.payload
    assert "debug_payload" not in str(adapter.payload)
    assert result.section_texts["EEG DESCRIPTION/DETAILS"].startswith("The background amplitude")


def test_slot_checklist_synthesizer_composes_report_ready_slots() -> None:
    adapter = FakeReportAdapter()
    result = EvidenceBoardLLMReportSynthesizer(adapter=adapter).synthesize_evidence_direct_sections(
        _board(),
        ["EEG DESCRIPTION/DETAILS"],
        payload_mode="slot_checklist",
    )

    assert result.trace["synthesis_version"] == "slot_checklist_diagnostic_v1"
    assert result.trace["payload_mode"] == "slot_checklist"
    assert adapter.payload["diagnostic_mode"] == "slot_checklist_report_synthesis"
    assert "section_slot_checklist" in adapter.payload
    background_slots = adapter.payload["section_slot_checklist"]["background"]["slots"]
    assert "background_amplitude_range" in background_slots
    assert background_slots["background_amplitude_range"]["value"] == {"lower": 70.0, "upper": 90.0}
    assert background_slots["background_amplitude_range"]["evidence_ids"] == ["evgrp_background_amplitude"]
    assert adapter.payload["evidence_view_summary"]["slot_count"] >= 1
    assert result.section_texts["EEG DESCRIPTION/DETAILS"].startswith("The background amplitude")


def test_evidence_direct_synthesizer_strips_evidence_ids_from_prose() -> None:
    class EvidenceIdLeakAdapter(FakeReportAdapter):
        def synthesize_from_evidence_view(self, evidence_payload):
            self.payload = evidence_payload
            return {
                "report_sections": [
                    {
                        "section_name": "EEG DESCRIPTION/DETAILS",
                        "section_text": (
                            "The PDR is 9 Hz (supported evidence IDs: evgrp_pdr, ev_llm_m_pdr). "
                            "Amplitude is 30 uV (caveated evidence ID: evgrp_background_amplitude)."
                        ),
                    }
                ],
                "global_limitations": [],
                "raw_eeg_used": False,
                "gt_report_used": False,
            }

    result = EvidenceBoardLLMReportSynthesizer(adapter=EvidenceIdLeakAdapter()).synthesize_evidence_direct_sections(
        _board(),
        ["EEG DESCRIPTION/DETAILS"],
        payload_mode="slot_checklist",
    )

    text = result.section_texts["EEG DESCRIPTION/DETAILS"]
    assert "evidence ID" not in text
    assert "evgrp" not in text
    assert "The PDR is 9 Hz." in text
    assert "Amplitude is 30 uV." in text


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


def test_d_synthesizer_keeps_safe_spatiomorphology_evidence_view() -> None:
    synth = EvidenceBoardLLMReportSynthesizer(adapter=FakeReportAdapter())
    loc_payload = synth._surface_safe_evidence_payload(  # noqa: SLF001
        EvidenceItem(
            evidence_id="evgrp_localization_v2",
            source_module="event",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.LOCALIZATION,
            value={
                "spatial_pattern": "left posterior predominance, maximal at O1/P3",
                "electrode_maxima": ["O1", "P3"],
                "region": "posterior",
                "laterality": "left",
            },
            normalized_value={
                "spatial_pattern": "left posterior predominance, maximal at O1/P3",
                "electrode_maxima": ["O1", "P3"],
                "region": "posterior",
                "laterality": "left",
            },
            measurement_ids=["m_event_spatial_pattern_v2"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "epileptiform"],
            created_by="test",
        )
    )
    morph_payload = synth._surface_safe_evidence_payload(  # noqa: SLF001
        EvidenceItem(
            evidence_id="evgrp_morphology_v2",
            source_module="event",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.EPILEPTIFORM_MORPHOLOGY,
            value={"morphology_descriptor": "sharp_transient_like"},
            normalized_value={"morphology_descriptor": "sharp_transient_like"},
            measurement_ids=["m_event_morphology_descriptor_v2"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "epileptiform"],
            created_by="test",
        )
    )

    assert loc_payload is not None
    assert loc_payload["value"]["spatial_pattern"] == "left posterior predominance, maximal at O1/P3"
    assert morph_payload is not None
    assert morph_payload["value"] == {"morphology_descriptor": "sharp_transient_like"}


def test_d_synthesizer_normalizes_microvolt_unit_symbol() -> None:
    synth = EvidenceBoardLLMReportSynthesizer(adapter=FakeReportAdapter())

    text = synth._sanitize_section_text("EEG DESCRIPTION/DETAILS", "Amplitude ranges from 8 to 32.7 μV.")  # noqa: SLF001

    assert "32.7 µV" in text
    assert "μV" not in text
