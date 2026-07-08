from eeg_report_multiagent.modules.llm_evidence_grouper import LLMEvidenceGrouper
from eeg_report_multiagent.schemas.measurement import MeasurementRole, MeasurementValue, QuantitationKind, QuantitationValue
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
    clinical_context = {
        "patient_history_and_eeg_description": "77 y.o. patient evaluated for altered awareness.",
        "metadata": {"age": "77", "gender": "Female"},
        "gt_report_text_included": False,
    }
    result = LLMEvidenceGrouper(adapter=adapter).run(
        recording_id="s",
        measurements=[_measurement()],
        clinical_context=clinical_context,
    )

    assert result["status"] == "ok"
    assert result["raw_eeg_used"] is False
    assert result["gt_report_used"] is False
    assert adapter.payload["clinical_context"] == clinical_context
    board = result["shared_evidence_board"]
    assert len(board.evidence_items) == 1
    item = board.evidence_items[0]
    assert item.evidence_id == "ev_llm_pdr_group"
    assert item.clinical_target == ClinicalTarget.PDR
    assert item.evidence_type == EvidenceType.DERIVED
    assert item.reportability == ClaimSurfaceAction.CAVEAT
    assert item.measurement_ids == ["m_pdr"]
    assert item.space_provenance["region"] == "occipital"


class MixedPdrGroupingAdapter:
    model = "fake-grouping"

    def group(self, payload):
        return {
            "summary": "grouped mixed PDR evidence",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "evidence_groups": [
                {
                    "evidence_id": "pdr_mixed_group",
                    "clinical_target": "pdr",
                    "evidence_type": "derived",
                    "linked_measurement_ids": ["m_pdr_candidate", "m_pdr_support", "m_pdr_v2"],
                    "allowed_sections": ["background", "detail"],
                    "rationale": "PDR with support measurements.",
                }
            ],
        }


def _exact_measurement(
    measurement_id: str,
    measurement_name: str,
    value: float,
    role: MeasurementRole,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=value, unit="Hz" if "frequency" in measurement_name else "score"),
        measurement_role=role,
        provenance=ProvenanceRecord(
            source_type=SourceType.SIGNAL,
            source_ref="s",
            time=TimeProvenance(window_indices=[1]),
            space=SpaceProvenance(channels=["O1", "O2"], region="occipital", laterality="bilateral"),
            measurement=MeasurementProvenance(tool_name="background", function_name="pdr"),
        ),
    )


def _range_measurement(
    measurement_id: str,
    measurement_name: str,
    lower: float,
    upper: float,
    unit: str,
    role: MeasurementRole,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.RANGE, lower=lower, upper=upper, unit=unit),
        measurement_role=role,
        provenance=ProvenanceRecord(
            source_type=SourceType.SIGNAL,
            source_ref="s",
            time=TimeProvenance(window_indices=[1]),
            space=SpaceProvenance(channels=["O1", "O2"], region="posterior", laterality="bilateral"),
            measurement=MeasurementProvenance(tool_name="background", function_name="amplitude"),
        ),
    )


def test_llm_evidence_grouper_keeps_pdr_value_free_of_support_scores():
    result = LLMEvidenceGrouper(adapter=MixedPdrGroupingAdapter()).run(
        recording_id="s",
        measurements=[
            _exact_measurement("m_pdr_candidate", "pdr_candidate_frequency_hz", 8.3, MeasurementRole.CLINICAL_MEASUREMENT),
            _exact_measurement("m_pdr_support", "pdr_v2_support_score", 0.45, MeasurementRole.PROXY_SCORE),
            _exact_measurement("m_pdr_v2", "pdr_v2_frequency_hz", 9.5, MeasurementRole.CLINICAL_MEASUREMENT),
        ],
    )

    item = result["shared_evidence_board"].evidence_items[0]
    assert item.clinical_target == ClinicalTarget.PDR
    assert item.value["frequency_hz"] == 9.5
    assert item.value["pdr_supported"] == "true"
    assert "pdr_v2_support_score" not in item.value
    assert item.unit == "Hz"
    assert item.measurement_ids == ["m_pdr_candidate", "m_pdr_v2"]
    assert item.debug_payload["all_linked_measurement_ids"] == ["m_pdr_candidate", "m_pdr_support", "m_pdr_v2"]


class MislabelledAmplitudeGroupingAdapter:
    model = "fake-grouping"

    def group(self, payload):
        return {
            "summary": "mislabelled amplitude",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "evidence_groups": [
                {
                    "evidence_id": "amp_group",
                    "clinical_target": "background_slowing",
                    "evidence_type": "derived",
                    "value_summary": "Background amplitude range indicates moderate activity.",
                    "linked_measurement_ids": ["m_amp"],
                    "allowed_sections": ["background", "detail"],
                    "rationale": "Adapter should correct target from linked measurement semantics.",
                }
            ],
        }


def test_llm_evidence_grouper_corrects_amplitude_target_when_llm_mislabels_it():
    result = LLMEvidenceGrouper(adapter=MislabelledAmplitudeGroupingAdapter()).run(
        recording_id="s",
        measurements=[
            _range_measurement(
                "m_amp",
                "background_amplitude_range_uv",
                28.1,
                43.9,
                "uV",
                MeasurementRole.CLINICAL_MEASUREMENT,
            )
        ],
    )

    item = result["shared_evidence_board"].evidence_items[0]
    assert item.clinical_target == ClinicalTarget.BACKGROUND_AMPLITUDE
    assert item.value["background_amplitude_range_uv"] == {"lower": 28.1, "upper": 43.9}
    assert item.value["background_amplitude_typical_uv"] is None
    assert item.value["background_amplitude_peak_to_peak_typical_uv"] is None
    assert item.unit == "uV"
