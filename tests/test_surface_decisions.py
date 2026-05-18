from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType, SurfaceDecision
from eeg_report_multiagent.schemas.section_contract import SectionRole


def _plan(plan_id: str, text: str, action: ClaimSurfaceAction = ClaimSurfaceAction.ALLOW) -> AtomicClaimPlan:
    return AtomicClaimPlan(
        plan_id=plan_id,
        section_type=ReportSectionType.DETAIL,
        claim_type="background_amplitude_range",
        proposed_text=text,
        evidence_ids=["ev1"],
        surface_action=action,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="test plan",
    )


def test_surface_decision_schema_serializes_authoritative_fields() -> None:
    decision = SurfaceDecision(
        decision_id="sd1",
        claim_id="p1",
        surface_action=ClaimSurfaceAction.CAVEAT,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="safe caveat",
        hard_deny_reasons=[],
        evidence_ids=["ev1"],
        caveat="review required",
        decided_by="surface_policy",
    )
    payload = decision.model_dump(mode="json")
    assert payload["decision_id"] == "sd1"
    assert payload["claim_id"] == "p1"
    assert payload["surface_action"] == "caveat"
    assert payload["evidence_ids"] == ["ev1"]


def test_report_text_uses_allow_caveat_surface_decisions() -> None:
    synth = ReportSynthesizer()
    plan = _plan("p1", "A provenance-linked background amplitude range is available (20-40 uV).")
    decisions = synth.build_surface_decisions([plan])
    text = synth._section_text_from_plans([plan], SectionRole.DETAIL, decisions)  # noqa: SLF001
    assert "20-40 uV" in text


def test_blocked_surface_decision_does_not_surface() -> None:
    synth = ReportSynthesizer()
    plan = _plan("p1", "This candidate burden has support score 1.8.")
    decisions = synth.build_surface_decisions([plan])
    assert decisions[0].surface_action == ClaimSurfaceAction.BLOCK
    assert "forbidden_debug_or_proxy_surface_text" in decisions[0].hard_deny_reasons
    text = synth._section_text_from_plans([plan], SectionRole.DETAIL, decisions)  # noqa: SLF001
    assert "candidate burden" not in text.lower()
    assert text == synth.surface_policy.safe_fallback_for_role(SectionRole.DETAIL)


def test_seizure_claim_without_seizure_evidence_is_hard_denied() -> None:
    synth = ReportSynthesizer()
    plan = AtomicClaimPlan(
        plan_id="p_seizure",
        section_type=ReportSectionType.DETAIL,
        claim_type="seizure_absent",
        proposed_text="Seizures: none.",
        evidence_ids=["ev_event_candidate"],
        surface_action=ClaimSurfaceAction.ALLOW,
        allowed_sections=[SectionRole.SEIZURES.value],
        rationale="unsafe test plan",
    )
    decisions = synth.build_surface_decisions([plan])
    assert decisions[0].surface_action == ClaimSurfaceAction.BLOCK
    assert "seizure_claim_without_seizure_specific_evidence" in decisions[0].hard_deny_reasons
