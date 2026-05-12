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


def test_atomic_claim_plan_caveats_peak_localization_claim() -> None:
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

    assert plans[0].surface_action == ClaimSurfaceAction.CAVEAT
    assert "definitive_epileptiform_morphology" in plans[0].missing_evidence
