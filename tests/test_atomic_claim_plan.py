from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.finding import FindingObject
from eeg_report_multiagent.schemas.measurement import MeasurementValue, QuantitationKind, QuantitationValue, StatusSemantic
from eeg_report_multiagent.schemas.provenance import ProvenanceRecord, SourceType
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction


def test_atomic_claim_plan_blocks_debug_only_proxy_scores() -> None:
    prov = ProvenanceRecord(source_type=SourceType.SIGNAL, source_ref="s")
    measurement = MeasurementValue(
        measurement_id="m_debug",
        measurement_name="event_peak_field_concentration_ratio",
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=2.1, unit="ratio"),
        provenance=prov,
    )
    finding = FindingObject(
        finding_id="f_debug",
        finding_type="event_peak_field_support",
        assertion=StatusSemantic.PRESENT,
        measurement_ids=[measurement.measurement_id],
        quantitation=measurement.quantitation,
        provenance=[prov],
    )
    board = EvidenceBoard(session_id="s", measurements=[measurement], findings=[finding])

    plans = ReportSynthesizer().build_atomic_claim_plan(board)

    assert plans[0].surface_action == ClaimSurfaceAction.DEBUG_ONLY
    assert "debug" in (plans[0].rationale or "").lower()


def test_atomic_claim_plan_keeps_peak_localization_debug_only_by_default() -> None:
    prov = ProvenanceRecord(source_type=SourceType.SIGNAL, source_ref="s")
    measurement = MeasurementValue(
        measurement_id="m_loc",
        measurement_name="event_peak_localization_label",
        categorical_value="left_temporal",
        provenance=prov,
    )
    finding = FindingObject(
        finding_id="f_loc",
        finding_type="event_peak_localization",
        assertion=StatusSemantic.PRESENT,
        measurement_ids=[measurement.measurement_id],
        provenance=[prov],
    )
    board = EvidenceBoard(session_id="s", measurements=[measurement], findings=[finding])

    plans = ReportSynthesizer().build_atomic_claim_plan(board)

    assert plans[0].surface_action == ClaimSurfaceAction.DEBUG_ONLY
    assert "definitive_epileptiform_morphology" in plans[0].missing_evidence


def test_section_synthesis_blocks_proxy_localization_even_with_support_scores() -> None:
    prov = ProvenanceRecord(source_type=SourceType.SIGNAL, source_ref="s")

    def exact(mid: str, name: str, value: float) -> MeasurementValue:
        return MeasurementValue(
            measurement_id=mid,
            measurement_name=name,
            quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=value, unit="score"),
            provenance=prov,
        )

    loc = MeasurementValue(
        measurement_id="m_loc",
        measurement_name="event_peak_localization_label",
        categorical_value="left_temporal",
        provenance=prov,
    )
    morph_class = MeasurementValue(
        measurement_id="m_morph_class",
        measurement_name="event_morphology_proxy_class",
        categorical_value="sharp_transient_candidate",
        provenance=prov,
    )
    measurements = [
        loc,
        exact("m_burden", "event_candidate_burden_ratio", 0.10),
        exact("m_peak_field", "event_peak_field_concentration_ratio", 3.0),
        exact("m_likelihood", "epileptiform_candidate_likelihood_score", 0.90),
        exact("m_morph_support", "event_morphology_support_score", 1.5),
        morph_class,
    ]
    findings = [
        FindingObject(
            finding_id="f_loc",
            finding_type="event_peak_localization",
            assertion=StatusSemantic.PRESENT,
            measurement_ids=["m_loc"],
            provenance=[prov],
        ),
        FindingObject(
            finding_id="f_burden",
            finding_type="epileptiform_event_candidate_burden",
            assertion=StatusSemantic.PRESENT,
            measurement_ids=["m_burden"],
            quantitation=measurements[1].quantitation,
            provenance=[prov],
        ),
        FindingObject(
            finding_id="f_peak_field",
            finding_type="event_peak_field_support",
            assertion=StatusSemantic.PRESENT,
            measurement_ids=["m_peak_field"],
            quantitation=measurements[2].quantitation,
            provenance=[prov],
        ),
        FindingObject(
            finding_id="f_likelihood",
            finding_type="epileptiform_candidate_likelihood",
            assertion=StatusSemantic.PRESENT,
            measurement_ids=["m_likelihood"],
            quantitation=measurements[3].quantitation,
            provenance=[prov],
        ),
        FindingObject(
            finding_id="f_morph_support",
            finding_type="event_morphology_support",
            assertion=StatusSemantic.PRESENT,
            measurement_ids=["m_morph_support"],
            quantitation=measurements[4].quantitation,
            provenance=[prov],
        ),
        FindingObject(
            finding_id="f_morph_class",
            finding_type="event_morphology_class",
            assertion=StatusSemantic.PRESENT,
            measurement_ids=["m_morph_class"],
            provenance=[prov],
        ),
    ]
    board = EvidenceBoard(session_id="s", measurements=measurements, findings=findings)

    sections = ReportSynthesizer().synthesize_celm_sections(
        board,
        ["EEG DESCRIPTION/DETAILS", "EPLEPTIFORM ABNORMALITIES"],
    )

    assert "localization screen suggested" not in sections["EEG DESCRIPTION/DETAILS"]
    assert "localization screen suggested left temporal" not in sections["EPLEPTIFORM ABNORMALITIES"]
    assert "No surface-allowed epileptiform claim" in sections["EPLEPTIFORM ABNORMALITIES"]
