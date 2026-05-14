from __future__ import annotations

from eeg_report_multiagent.modules.final_prose_auditor import FinalProseAuditor
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas import EvidenceBoard, FindingObject, MeasurementValue, QuantitationValue
from eeg_report_multiagent.schemas.final_prose_audit import NumericMatchStatus
from eeg_report_multiagent.schemas.measurement import QuantitationKind, StatusSemantic
from eeg_report_multiagent.schemas.provenance import MeasurementProvenance, ProvenanceRecord, SourceType, SpaceProvenance, TimeProvenance
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


def _prov(channels=None, region=None, side=None) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=SourceType.SIGNAL,
        source_ref="s",
        time=TimeProvenance(window_indices=[0], start_sec=0.0, end_sec=10.0),
        space=SpaceProvenance(channels=channels or [], region=region, laterality=side),
        measurement=MeasurementProvenance(tool_name="tool", function_name="fn"),
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


def _finding(fid: str, ftype: str, measurement: MeasurementValue) -> FindingObject:
    return FindingObject(
        finding_id=fid,
        finding_type=ftype,
        assertion=StatusSemantic.PRESENT,
        measurement_ids=[measurement.measurement_id],
        quantitation=measurement.quantitation,
        provenance=[measurement.provenance],
        source_module="background_module",
    )


def test_numeric_extraction_common_units_and_scores() -> None:
    text = "PDR was 9-10 Hz, amplitude approximately 180 uV and 70-90 µV. Boundary was 0.5 Hz, duration 190 sec, 10%, score 1.8."
    mentions = FinalProseAuditor().extract_numeric_mentions(text, "BACKGROUND ACTIVITY")

    raw = [m.raw_text.lower() for m in mentions]
    assert "9-10 hz" in raw
    assert "approximately 180 uv" in raw
    assert any(m.unit == "uV" and m.normalized_value == {"lower": 70.0, "upper": 90.0} for m in mentions)
    assert any(m.raw_text == "0.5 Hz" for m in mentions)
    assert any(m.raw_text == "190 sec" for m in mentions)
    assert any(m.unit == "percent" and m.value == 10.0 for m in mentions)
    assert any(m.unit == "score" and m.value == 1.8 for m in mentions)


def test_numeric_matching_exact_range_unit_debug_wrong_section() -> None:
    auditor = FinalProseAuditor()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(EvidenceItem(
        evidence_id="ev_pdr",
        source_module="background",
        evidence_type=EvidenceType.DIRECT,
        clinical_target=ClinicalTarget.PDR,
        value={"lower": 9.0, "upper": 10.0},
        normalized_value={"lower": 9.0, "upper": 10.0},
        unit="Hz",
        reportability=ClaimSurfaceAction.ALLOW,
        allowed_sections=[SectionRole.BACKGROUND.value],
        created_by="test",
    ))
    board.add_evidence(EvidenceItem(
        evidence_id="ev_debug_score",
        source_module="event",
        evidence_type=EvidenceType.DEBUG,
        clinical_target=ClinicalTarget.UNCERTAINTY,
        value=1.8,
        normalized_value=1.8,
        unit="score",
        reportability=ClaimSurfaceAction.DEBUG_ONLY,
        created_by="test",
    ))

    pdr = auditor.extract_numeric_mentions("During wakefulness, posterior dominant rhythm was 9-10 Hz.", "BACKGROUND ACTIVITY")[0]
    assert auditor.match_numeric_to_evidence(pdr, board.evidence_items).match_status == NumericMatchStatus.EXACT

    contained = auditor.extract_numeric_mentions("Posterior rhythm was 9.5 Hz.", "BACKGROUND ACTIVITY")[0]
    assert auditor.match_numeric_to_evidence(contained, board.evidence_items).match_status == NumericMatchStatus.RANGE_CONTAINED

    wrong_unit = auditor.extract_numeric_mentions("Posterior rhythm was 9-10 uV.", "BACKGROUND ACTIVITY")[0]
    assert auditor.match_numeric_to_evidence(wrong_unit, board.evidence_items).match_status == NumericMatchStatus.UNIT_MISMATCH

    debug = auditor.extract_numeric_mentions("The support score 1.8 was present.", "EPLEPTIFORM ABNORMALITIES")[0]
    assert auditor.match_numeric_to_evidence(debug, board.evidence_items).match_status == NumericMatchStatus.MATCHED_BUT_NOT_REPORTABLE

    wrong_section = auditor.extract_numeric_mentions("During wakefulness, posterior dominant rhythm was 9-10 Hz.", "SEIZURES")[0]
    assert auditor.match_numeric_to_evidence(wrong_section, board.evidence_items).match_status == NumericMatchStatus.MATCHED_BUT_WRONG_SECTION


def test_numeric_matching_accepts_display_rounded_amplitude_range() -> None:
    auditor = FinalProseAuditor()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(EvidenceItem(
        evidence_id="ev_background_dominant_frequency",
        source_module="background",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.BACKGROUND_SLOWING,
        value=0.5,
        normalized_value=0.5,
        unit="Hz",
        reportability=ClaimSurfaceAction.DEBUG_ONLY,
        created_by="test",
    ))
    board.add_evidence(EvidenceItem(
        evidence_id="ev_background_amplitude_range",
        source_module="background",
        evidence_type=EvidenceType.DERIVED,
        clinical_target=ClinicalTarget.BACKGROUND_AMPLITUDE,
        value={"lower": 0.0, "upper": 79.7996},
        normalized_value={"lower": 0.0, "upper": 79.7996},
        unit="uV",
        reportability=ClaimSurfaceAction.CAVEAT,
        allowed_sections=[SectionRole.BACKGROUND.value],
        created_by="test",
    ))

    mention = auditor.extract_numeric_mentions(
        "A provenance-linked background amplitude range is available (0.0-80 uV).",
        "BACKGROUND ACTIVITY",
    )[0]
    match = auditor.match_numeric_to_evidence(mention, board.evidence_items)

    assert match.match_status == NumericMatchStatus.EXACT
    assert match.matched_evidence_id == "ev_background_amplitude_range"


def test_debug_leak_detection_terms() -> None:
    text = "Candidate burden and bifrontal spread tendency were high; support score and likelihood score used field concentration ratio with missing_slots."
    leaks = FinalProseAuditor().detect_banned_debug_terms(text, "EEG DESCRIPTION/DETAILS")
    terms = {leak.term for leak in leaks}

    assert {"candidate burden", "support score", "likelihood score", "field concentration ratio", "missing_slots", "bifrontal spread tendency"}.issubset(terms)


def test_section_leakage_rules() -> None:
    auditor = FinalProseAuditor()

    assert auditor.detect_section_leakage("BACKGROUND ACTIVITY", "Spike-wave event detail appears in background.")
    assert auditor.detect_section_leakage("EPLEPTIFORM ABNORMALITIES", "Electrographic seizure was confirmed.")
    assert auditor.detect_section_leakage("SEIZURES", "Interictal transient candidate burden was seen.")
    assert auditor.detect_section_leakage("IMPRESSION/INTERPRETATION", "Abnormality is based on field concentration ratio.")


def test_claim_surface_matching_allowed_debug_and_missing_evidence() -> None:
    auditor = FinalProseAuditor()
    allowed = AtomicClaimPlan(
        plan_id="p_allowed",
        section_type=ReportSectionType.DETAIL,
        claim_type="background_slowing",
        proposed_text="Structured evidence suggests background slowing; this remains an assistive finding pending EEG review.",
        evidence_ids=["ev_slow"],
        surface_action=ClaimSurfaceAction.CAVEAT,
    )
    debug = AtomicClaimPlan(
        plan_id="p_debug",
        section_type=ReportSectionType.DETAIL,
        claim_type="event_candidate",
        proposed_text="Candidate burden was elevated.",
        evidence_ids=["ev_candidate"],
        surface_action=ClaimSurfaceAction.DEBUG_ONLY,
    )
    missing = AtomicClaimPlan(
        plan_id="p_missing",
        section_type=ReportSectionType.DETAIL,
        claim_type="background_slowing",
        proposed_text="Background slowing is present.",
        surface_action=ClaimSurfaceAction.CAVEAT,
    )

    ok = auditor.match_text_claims_to_atomic_plans("EEG DESCRIPTION/DETAILS", allowed.proposed_text, [allowed])
    assert ok[0].match_status == "matched_allowed_claim"
    no_plan = auditor.match_text_claims_to_atomic_plans("EEG DESCRIPTION/DETAILS", "Background slowing is present.", [])
    assert no_plan[0].match_status == "unmatched_surface_claim"
    debug_match = auditor.match_text_claims_to_atomic_plans("EEG DESCRIPTION/DETAILS", "Candidate burden was elevated.", [debug])
    assert debug_match[0].match_status == "surface_policy_violation"
    missing_match = auditor.match_text_claims_to_atomic_plans("EEG DESCRIPTION/DETAILS", "Background slowing is present.", [missing])
    assert missing_match[0].match_status == "missing_required_evidence_links"


def test_row189_style_unsafe_final_prose_fails_audit() -> None:
    board = SharedEvidenceBoard(board_id="seb", recording_id="row189")
    board.add_evidence(EvidenceItem(
        evidence_id="ev_candidate",
        source_module="event",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.EVENT_CANDIDATE,
        value=0.2,
        normalized_value=0.2,
        unit="ratio",
        reportability=ClaimSurfaceAction.DEBUG_ONLY,
        created_by="test",
    ))
    report = {
        "BACKGROUND ACTIVITY": "A 0.5 Hz dominant rhythm is present with candidate burden.",
        "EPLEPTIFORM ABNORMALITIES": "Bifrontal spread tendency with support score 1.8 and field concentration ratio 2.8.",
        "SEIZURES": "Seizures consist of spike-wave complexes lasting 190 sec.",
    }

    result = FinalProseAuditor().audit_report(report, board, [])

    assert result.pass_fail == "fail"
    assert result.debug_leaks
    assert result.unsupported_numeric_mentions
    assert result.seizure_gate_violations
    assert result.section_leakages
    assert result.metrics["AuditPassRate"] == 0.0


def test_safe_fallback_output_passes_audit() -> None:
    report = {
        "EPLEPTIFORM ABNORMALITIES": "No surface-allowed epileptiform claim was produced by the current structured evidence.",
        "SEIZURES": "Seizures: no seizure-specific evidence was produced by the current structured tools.",
    }
    result = FinalProseAuditor().audit_report(report, SharedEvidenceBoard(board_id="seb", recording_id="s"), [])

    assert result.pass_fail == "pass"
    assert result.metrics["AuditPassRate"] == 1.0


def test_safe_pdr_text_passes_only_with_valid_evidence_item_and_claim_plan() -> None:
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(EvidenceItem(
        evidence_id="ev_pdr",
        source_module="background",
        evidence_type=EvidenceType.DIRECT,
        clinical_target=ClinicalTarget.PDR,
        value={"lower": 9.0, "upper": 10.0},
        normalized_value={"lower": 9.0, "upper": 10.0},
        unit="Hz",
        space_provenance={"channels": ["O1", "O2"], "region": "occipital", "side": "bilateral", "electrode_maxima": ["O1", "O2"]},
        reportability=ClaimSurfaceAction.ALLOW,
        allowed_sections=[SectionRole.BACKGROUND.value],
        created_by="test",
    ))
    plan = AtomicClaimPlan(
        plan_id="p_pdr",
        section_type=ReportSectionType.DETAIL,
        claim_type="background_pdr_frequency",
        proposed_text="During wakefulness, posterior dominant rhythm was 9-10 Hz.",
        evidence_ids=["ev_pdr"],
        surface_action=ClaimSurfaceAction.ALLOW,
        allowed_sections=[SectionRole.BACKGROUND.value],
    )
    report = {"BACKGROUND ACTIVITY": "During wakefulness, posterior dominant rhythm was 9-10 Hz."}

    result = FinalProseAuditor().audit_report(report, board, [plan])

    assert result.pass_fail == "pass"
    assert result.supported_numeric_mentions[0].matched_evidence_id == "ev_pdr"
    assert not result.debug_leaks
    assert result.metrics["NumericProvenanceAccuracy"] == 1.0
    assert result.metrics["ClaimTraceCoverage"] == 1.0
