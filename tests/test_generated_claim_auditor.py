from eeg_report_multiagent.modules.generated_claim_auditor import GeneratedClaimAuditor, claims_match
from eeg_report_multiagent.schemas.gt_suppression import GTAtomicClaim


def _claim(claim_type: str, value, unit=None, source_text="text") -> GTAtomicClaim:
    return GTAtomicClaim(
        gt_claim_id=f"id_{claim_type}",
        case_id="case",
        section="EEG DESCRIPTION/DETAILS",
        claim_type=claim_type,
        normalized_value=value,
        unit=unit,
        source_text=source_text,
    )


def test_generated_claim_match_requires_numeric_overlap() -> None:
    gt = _claim("pdr_frequency", {"lower": 9.0, "upper": 10.0}, "Hz")
    good = _claim("pdr_frequency", {"lower": 9.0, "upper": 9.0}, "Hz")
    bad = _claim("pdr_frequency", {"lower": 8.0, "upper": 8.0}, "Hz")
    assert claims_match(gt, good)
    assert not claims_match(gt, bad)


def test_generated_claim_match_electrode_overlap() -> None:
    gt = _claim("electrode_maxima", ["F3", "F7"])
    good = _claim("electrode_maxima", ["F7", "T3"])
    bad = _claim("electrode_maxima", ["O1", "O2"])
    assert claims_match(gt, good)
    assert not claims_match(gt, bad)


def test_generated_claim_extractor_can_parse_model_output_sections(tmp_path) -> None:
    report = tmp_path / "generated.json"
    report.write_text(
        '{"report_sections": [{"section_name": "EEG DESCRIPTION/DETAILS", "section_text": "The background has a 9-10 Hz posterior dominant rhythm. Push button events: none. Seizures: none."}]}',
        encoding="utf-8",
    )
    auditor = GeneratedClaimAuditor()
    sections = auditor.extract_sections_from_generated_report(report)
    claims = auditor.extractor.extract_from_sections(sections, case_id="case")
    claim_types = {claim.claim_type for claim in claims}
    assert "pdr_frequency" in claim_types
    assert "push_button_absent" in claim_types
    assert "seizure_absent" in claim_types
