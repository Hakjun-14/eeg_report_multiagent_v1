from __future__ import annotations

from eeg_report_multiagent.modules.final_prose_auditor import FinalProseAuditor
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue, QuantitationKind, QuantitationValue, StatusSemantic, StatusValue
from eeg_report_multiagent.schemas.provenance import MeasurementProvenance, ProvenanceRecord, SourceType, SpaceProvenance, TimeProvenance
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction


def _prov(*, channels=None, region=None, side=None, source_type=SourceType.SIGNAL) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=source_type,
        source_ref="stage3c_test",
        time=TimeProvenance(window_indices=[0], start_sec=0.0, end_sec=10.0),
        space=SpaceProvenance(channels=channels or [], region=region, laterality=side),
        measurement=MeasurementProvenance(tool_name="test_tool", function_name="test_fn"),
    )


def _exact(mid: str, name: str, value: float, unit: str, prov: ProvenanceRecord | None = None) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=mid,
        measurement_name=name,
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=value, unit=unit),
        provenance=prov or _prov(),
    )


def _range(mid: str, name: str, lower: float, upper: float, unit: str, prov: ProvenanceRecord | None = None) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=mid,
        measurement_name=name,
        quantitation=QuantitationValue(kind=QuantitationKind.RANGE, lower=lower, upper=upper, unit=unit),
        provenance=prov or _prov(),
    )


def _status(mid: str, name: str, status: StatusSemantic) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=mid,
        measurement_name=name,
        status_value=StatusValue(status=status),
        provenance=_prov(source_type=SourceType.METADATA),
    )


def _finding(fid: str, ftype: str, measurement: MeasurementValue, assertion: StatusSemantic = StatusSemantic.PRESENT) -> Finding:
    return Finding(
        finding_id=fid,
        finding_type=ftype,
        assertion=assertion,
        quantitation=measurement.quantitation,
        measurement_ids=[measurement.measurement_id],
        provenance=[measurement.provenance],
        source_module="background_module" if ftype.startswith("background") else "event_module",
    )


def test_stage3c_posterior_alpha_can_be_caveated_even_when_adapter_blocks_strong_pdr() -> None:
    prov = _prov(channels=["O1", "O2"], region="occipital", side="bilateral")
    measurement = _exact("m_pdr", "pdr_candidate_frequency", 9.0, "Hz", prov)
    finding = _finding("f_pdr", "background_pdr_frequency", measurement)
    board = EvidenceBoard(session_id="s", measurements=[measurement], findings=[finding])

    plans = ReportSynthesizer().build_atomic_claim_plan(board)
    sections = ReportSynthesizer().synthesize_celm_sections(board, ["BACKGROUND ACTIVITY"])

    assert plans[0].surface_action == ClaimSurfaceAction.CAVEAT
    assert plans[0].debug_payload["stage3c_calibration"]["safe_surface_override"] is True
    assert "posterior alpha rhythm candidate near 9.0 Hz" in sections["BACKGROUND ACTIVITY"]
    audit = FinalProseAuditor().audit_report(sections, board.ensure_shared_evidence_board(), plans)
    assert audit.pass_fail == "pass"


def test_stage3c_blocks_boundary_frequency_as_pdr() -> None:
    prov = _prov(channels=["O1", "O2"], region="occipital", side="bilateral")
    measurement = _exact("m_pdr", "pdr_candidate_frequency", 0.5, "Hz", prov)
    finding = _finding("f_pdr", "background_pdr_frequency", measurement)
    board = EvidenceBoard(session_id="s", measurements=[measurement], findings=[finding])

    plans = ReportSynthesizer().build_atomic_claim_plan(board)
    sections = ReportSynthesizer().synthesize_celm_sections(board, ["BACKGROUND ACTIVITY"])

    assert plans[0].surface_action == ClaimSurfaceAction.BLOCK
    assert "0.5 Hz" not in sections["BACKGROUND ACTIVITY"]
    assert "posterior alpha" not in sections["BACKGROUND ACTIVITY"].lower()


def test_stage3c_metadata_status_can_surface_without_numeric_provenance() -> None:
    measurement = _status("m_hv", "hyperventilation_status", StatusSemantic.NOT_PERFORMED)
    finding = _finding("f_hv", "protocol_hyperventilation_status", measurement, StatusSemantic.NOT_PERFORMED)
    board = EvidenceBoard(session_id="s", measurements=[measurement], findings=[finding])

    plans = ReportSynthesizer().build_atomic_claim_plan(board)
    sections = ReportSynthesizer().synthesize_celm_sections(board, ["EEG DESCRIPTION/DETAILS"])

    assert plans[0].surface_action == ClaimSurfaceAction.ALLOW
    assert "Hyperventilation: not performed." in sections["EEG DESCRIPTION/DETAILS"]
    audit = FinalProseAuditor().audit_report(sections, board.ensure_shared_evidence_board(), plans)
    assert audit.unsupported_numeric_mentions == []
    assert audit.pass_fail == "pass"


def test_stage3c_candidate_burden_and_support_scores_remain_unsurfaced() -> None:
    burden = _exact("m_burden", "event_candidate_burden_ratio", 0.2, "ratio")
    support = _exact("m_support", "event_morphology_support_score", 1.6, "score")
    findings = [
        _finding("f_burden", "epileptiform_event_candidate_burden", burden),
        _finding("f_support", "event_morphology_support", support),
    ]
    board = EvidenceBoard(session_id="s", measurements=[burden, support], findings=findings)

    sections = ReportSynthesizer().synthesize_celm_sections(board, ["EPLEPTIFORM ABNORMALITIES", "SEIZURES"])
    rendered = "\n".join(sections.values()).lower()

    assert "candidate burden" not in rendered
    assert "support score" not in rendered
    assert "seizures: no seizure-specific evidence" in rendered


def test_stage3c_localization_ratio_alone_stays_blocked() -> None:
    ratio = _exact("m_ratio", "event_peak_field_concentration_ratio", 2.4, "ratio")
    finding = _finding("f_ratio", "event_peak_field_support", ratio)
    board = EvidenceBoard(session_id="s", measurements=[ratio], findings=[finding])

    plans = ReportSynthesizer().build_atomic_claim_plan(board)
    sections = ReportSynthesizer().synthesize_celm_sections(board, ["EPLEPTIFORM ABNORMALITIES"])

    assert plans[0].surface_action in {ClaimSurfaceAction.BLOCK, ClaimSurfaceAction.DEBUG_ONLY}
    assert "field concentration ratio" not in sections["EPLEPTIFORM ABNORMALITIES"].lower()


def test_stage3c_valid_background_amplitude_still_passes_final_audit() -> None:
    measurement = _range("m_amp", "background_amplitude_range", 30.0, 60.0, "uV")
    finding = _finding("f_amp", "background_amplitude_range", measurement)
    board = EvidenceBoard(session_id="s", measurements=[measurement], findings=[finding])

    plans = ReportSynthesizer().build_atomic_claim_plan(board)
    sections = ReportSynthesizer().synthesize_celm_sections(board, ["BACKGROUND ACTIVITY"])
    audit = FinalProseAuditor().audit_report(sections, board.ensure_shared_evidence_board(), plans)

    assert plans[0].surface_action == ClaimSurfaceAction.CAVEAT
    assert "30-60 uV" in sections["BACKGROUND ACTIVITY"]
    assert audit.pass_fail == "pass"
