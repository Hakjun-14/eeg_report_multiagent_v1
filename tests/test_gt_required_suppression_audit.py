from pathlib import Path

from eeg_report_multiagent.modules.gt_required_suppression_auditor import GTClaimExtractor, GTRequiredSuppressionAuditor
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.gt_suppression import GTAtomicClaim
from eeg_report_multiagent.schemas.measurement import QuantitationKind, QuantitationValue
from eeg_report_multiagent.schemas.provenance import ProvenanceRecord, SourceType
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


def _prov() -> ProvenanceRecord:
    return ProvenanceRecord(source_type=SourceType.SIGNAL, source_ref="test")


def _evidence_item(**kwargs) -> EvidenceItem:
    defaults = dict(
        evidence_id="ev1",
        source_module="background",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.PDR,
        value=9.0,
        unit="Hz",
        reportability=ClaimSurfaceAction.BLOCK,
        allowed_sections=["background"],
        measurement_ids=[],
        finding_ids=[],
        created_by="test",
    )
    defaults.update(kwargs)
    return EvidenceItem(**defaults)


def test_gt_claim_extractor_row189_style_slots() -> None:
    sections = {
        "background:": "Awake background showed symmetric posterior dominant rhythm at 9-10 Hz and 70-90 uV with good reactivity. Symmetric vertex waves were present during stage II sleep.",
        "events/seizures:": "Push button events: none. Seizures: none.",
        "interictal epileptiform abnormalities:": "Sporadic generalized 180 uV spike/wave discharges with subtle left > right frontal predominance, maximal at F3/F7.",
    }
    claims = GTClaimExtractor().extract_from_sections(sections, case_id="case")
    claim_types = [claim.claim_type for claim in claims]
    assert "pdr_frequency" in claim_types
    assert "background_amplitude" in claim_types
    assert "stage_ii_sleep" in claim_types
    assert "push_button_absent" in claim_types
    assert "seizure_absent" in claim_types
    assert "epileptiform_morphology_spike_wave" in claim_types
    assert "localization_laterality" in claim_types
    # Symmetric vertex waves should not be mislabeled as PDR symmetry.
    assert claim_types.count("pdr_symmetry") == 1


def test_candidate_burden_does_not_match_gt_spike_wave() -> None:
    claim = GTAtomicClaim(
        gt_claim_id="gt1",
        case_id="case",
        section="interictal epileptiform abnormalities:",
        claim_type="epileptiform_morphology_spike_wave",
        normalized_value="spike_wave",
        source_text="generalized spike/wave discharges",
    )
    evidence = _evidence_item(
        evidence_id="ev_candidate_burden",
        source_module="event",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.EVENT_CANDIDATE,
        value=0.3,
        unit=None,
        reportability=ClaimSurfaceAction.DEBUG_ONLY,
        rationale="candidate burden only",
    )
    # Bypass file extraction to directly verify the matcher behavior.
    match = GTRequiredSuppressionAuditor().match_claim(claim, [], [], [evidence], [], {})
    assert match.match_stage == "no_measurement"
    assert match.salvageability == "detector_gap"


def test_pdr_boundary_0_5_hz_is_not_safe_match() -> None:
    claim = GTAtomicClaim(
        gt_claim_id="gt_pdr",
        case_id="case",
        section="background:",
        claim_type="pdr_frequency",
        normalized_value={"lower": 9.0, "upper": 10.0},
        unit="Hz",
        source_text="posterior dominant rhythm at 9-10 Hz",
    )
    evidence = _evidence_item(
        evidence_id="ev_boundary_frequency",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.PDR,
        value=0.5,
        unit="Hz",
        reportability=ClaimSurfaceAction.DEBUG_ONLY,
        rationale="global boundary peak",
    )
    plan = AtomicClaimPlan(
        plan_id="p_boundary",
        section_type=ReportSectionType.DETAIL,
        claim_type="pdr_frequency",
        proposed_text="0.5 Hz posterior dominant rhythm",
        evidence_ids=["ev_boundary_frequency"],
        surface_action=ClaimSurfaceAction.BLOCK,
    )
    match = GTRequiredSuppressionAuditor().match_claim(claim, [], [], [evidence], [plan], {})
    assert match.match_stage == "no_measurement"
    assert match.category == "gt_required_but_missing_from_evidence_extraction"
    assert match.salvageability == "detector_gap"


def test_upstream_blocked_gt_pdr_is_surface_policy_gap() -> None:
    claim = GTAtomicClaim(
        gt_claim_id="gt_pdr",
        case_id="case",
        section="background:",
        claim_type="pdr_frequency",
        normalized_value={"lower": 9.0, "upper": 10.0},
        unit="Hz",
        source_text="posterior dominant rhythm at 9-10 Hz",
    )
    evidence = _evidence_item(
        evidence_id="ev_pdr_candidate",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.PDR,
        value=9.0,
        unit="Hz",
        reportability=ClaimSurfaceAction.BLOCK,
        space_provenance={"region": "posterior", "channels": ["O1", "O2"]},
    )
    plan = AtomicClaimPlan(
        plan_id="p_pdr_candidate",
        section_type=ReportSectionType.DETAIL,
        claim_type="pdr_frequency",
        proposed_text="PDR candidate was 9 Hz",
        evidence_ids=["ev_pdr_candidate"],
        surface_action=ClaimSurfaceAction.BLOCK,
    )
    match = GTRequiredSuppressionAuditor().match_claim(claim, [], [], [evidence], [plan], {})
    assert match.match_stage == "atomic_claim"
    assert match.category == "gt_required_but_surfacepolicy_blocked"
    assert match.salvageability == "caveat_candidate"
