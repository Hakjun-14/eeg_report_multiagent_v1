from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType, SurfaceDecision
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


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
        clinical_reference_ids=["ifcn_background_report_structure"],
        caveat="review required",
        decided_by="surface_policy",
    )
    payload = decision.model_dump(mode="json")
    assert payload["decision_id"] == "sd1"
    assert payload["claim_id"] == "p1"
    assert payload["surface_action"] == "caveat"
    assert payload["evidence_ids"] == ["ev1"]
    assert payload["clinical_reference_ids"] == ["ifcn_background_report_structure"]


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


def test_surface_decision_calibrates_blocked_trace_safe_pdr_to_caveat() -> None:
    synth = ReportSynthesizer()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_pdr",
            source_module="test",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.PDR,
            value={"frequency_hz": 9.5},
            normalized_value={"frequency_hz": 9.5},
            unit="Hz",
            reportability=ClaimSurfaceAction.BLOCK,
            allowed_sections=[SectionRole.DETAIL.value],
            created_by="test",
        )
    )
    plan = AtomicClaimPlan(
        plan_id="p_pdr",
        section_type=ReportSectionType.DETAIL,
        claim_type="pdr",
        proposed_text="A posterior alpha rhythm candidate is approximately 9.5 Hz; reactivity is not confirmed.",
        evidence_ids=["ev_pdr"],
        surface_action=ClaimSurfaceAction.BLOCK,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="blocked before calibration",
        surface_safe_values=[
            {
                "evidence_id": "ev_pdr",
                "clinical_target": "pdr",
                "value": {"frequency_hz": 9.5},
                "unit": "Hz",
            }
        ],
        must_render_values=["pdr_frequency=9.5 Hz"],
    )

    decision = synth.build_surface_decisions([plan], board)[0]

    assert decision.surface_action == ClaimSurfaceAction.CAVEAT
    assert decision.decided_by == "surface_policy_calibrated"
    assert decision.hard_deny_reasons == []
    assert "acns_posterior_alpha_pdr" in decision.clinical_reference_ids
    assert decision.debug_payload["surface_calibration"]["from_surface_action"] == "block"


def test_surface_decision_calibrates_status_metadata_to_allow() -> None:
    synth = ReportSynthesizer()
    plan = AtomicClaimPlan(
        plan_id="p_protocol",
        section_type=ReportSectionType.DETAIL,
        claim_type="protocol",
        proposed_text="Structured state/protocol context: photic stimulation: not_performed.",
        evidence_ids=["ev_protocol"],
        surface_action=ClaimSurfaceAction.BLOCK,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="blocked before calibration",
        surface_safe_values=[
            {
                "evidence_id": "ev_protocol",
                "clinical_target": "protocol",
                "value": {"photic_stimulation_status": "not_performed"},
                "unit": None,
            }
        ],
        must_render_values=["photic_stimulation_status=not_performed"],
    )

    decision = synth.build_surface_decisions([plan])[0]

    assert decision.surface_action == ClaimSurfaceAction.ALLOW
    assert decision.decided_by == "surface_policy_calibrated"


def test_surface_decision_calibrates_debug_only_status_metadata_to_allow() -> None:
    synth = ReportSynthesizer()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_protocol",
            source_module="metadata",
            evidence_type=EvidenceType.METADATA,
            clinical_target=ClinicalTarget.PROTOCOL,
            value={"photic_stimulation_status": "performed"},
            normalized_value={"photic_stimulation_status": "performed"},
            reportability=ClaimSurfaceAction.ALLOW,
            allowed_sections=[SectionRole.DETAIL.value],
            created_by="test",
        )
    )
    plan = AtomicClaimPlan(
        plan_id="p_protocol_debug_only",
        section_type=ReportSectionType.DETAIL,
        claim_type="protocol",
        proposed_text="Structured state/protocol context: photic stimulation: performed.",
        evidence_ids=["ev_protocol"],
        surface_action=ClaimSurfaceAction.DEBUG_ONLY,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="LLM incorrectly marked metadata as debug_only",
    )

    decision = synth.build_surface_decisions([plan], board)[0]

    assert decision.surface_action == ClaimSurfaceAction.ALLOW
    assert decision.decided_by == "surface_policy_calibrated"
    assert decision.hard_deny_reasons == []


def test_surface_decision_calibrates_from_linked_pdr_evidence_without_plan_values() -> None:
    synth = ReportSynthesizer()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_pdr",
            source_module="test",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.PDR,
            value={"frequency_hz": 10.0},
            normalized_value={"frequency_hz": 10.0},
            unit="Hz",
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=[SectionRole.DETAIL.value],
            created_by="test",
        )
    )
    plan = AtomicClaimPlan(
        plan_id="p_pdr_linked",
        section_type=ReportSectionType.DETAIL,
        claim_type="pdr",
        proposed_text="A posterior alpha rhythm candidate is approximately 10.0 Hz; reactivity is not confirmed.",
        evidence_ids=["ev_pdr"],
        surface_action=ClaimSurfaceAction.BLOCK,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="blocked before calibration",
    )

    decision = synth.build_surface_decisions([plan], board)[0]

    assert decision.surface_action == ClaimSurfaceAction.CAVEAT
    assert decision.decided_by == "surface_policy_calibrated"


def test_surface_decision_does_not_calibrate_unknown_status() -> None:
    synth = ReportSynthesizer()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_state",
            source_module="test",
            evidence_type=EvidenceType.METADATA,
            clinical_target=ClinicalTarget.STATE,
            value={"state_awake": "unknown", "state_sleep": "unknown"},
            normalized_value={"state_awake": "unknown", "state_sleep": "unknown"},
            reportability=ClaimSurfaceAction.ALLOW,
            allowed_sections=[SectionRole.DETAIL.value],
            created_by="test",
        )
    )
    plan = AtomicClaimPlan(
        plan_id="p_state_unknown",
        section_type=ReportSectionType.DETAIL,
        claim_type="state",
        proposed_text="State information is indeterminate.",
        evidence_ids=["ev_state"],
        surface_action=ClaimSurfaceAction.BLOCK,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="blocked before calibration",
    )

    decision = synth.build_surface_decisions([plan], board)[0]

    assert decision.surface_action == ClaimSurfaceAction.BLOCK
    assert decision.decided_by == "surface_policy"


def test_surface_decision_calibration_cannot_override_hard_deny() -> None:
    synth = ReportSynthesizer()
    plan = AtomicClaimPlan(
        plan_id="p_bad",
        section_type=ReportSectionType.DETAIL,
        claim_type="pdr",
        proposed_text="A posterior dominant rhythm is 0.5 Hz.",
        evidence_ids=["ev_pdr"],
        surface_action=ClaimSurfaceAction.BLOCK,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="unsafe",
        surface_safe_values=[
            {
                "evidence_id": "ev_pdr",
                "clinical_target": "pdr",
                "value": {"frequency_hz": 0.5},
                "unit": "Hz",
            }
        ],
        must_render_values=["pdr_frequency=0.5 Hz"],
    )

    decision = synth.build_surface_decisions([plan])[0]

    assert decision.surface_action == ClaimSurfaceAction.BLOCK
    assert decision.decided_by == "surface_policy"
    assert "boundary_or_global_low_frequency_pdr_forbidden" in decision.hard_deny_reasons


def test_direct_amplitude_not_hard_denied_by_context_measurement_ids() -> None:
    synth = ReportSynthesizer()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_amp",
            source_module="test",
            evidence_type=EvidenceType.DIRECT,
            clinical_target=ClinicalTarget.BACKGROUND_AMPLITUDE,
            value={"background_amplitude_range_uv": 33.3},
            normalized_value={"background_amplitude_range_uv": 33.3},
            unit="uV",
            measurement_ids=[
                "m_background_amplitude_range",
                "m_background_ap_organization_score",
                "m_bandpower_delta",
            ],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=[SectionRole.DETAIL.value],
            created_by="test",
        )
    )
    plan = AtomicClaimPlan(
        plan_id="p_amp",
        section_type=ReportSectionType.DETAIL,
        claim_type="background_amplitude",
        proposed_text="Background amplitude is approximately 33.3 uV.",
        evidence_ids=["ev_amp"],
        surface_action=ClaimSurfaceAction.CAVEAT,
        allowed_sections=[SectionRole.DETAIL.value],
        rationale="direct amplitude evidence",
    )

    decision = synth.build_surface_decisions([plan], board)[0]

    assert decision.surface_action == ClaimSurfaceAction.CAVEAT
    assert "linked_internal_or_proxy_evidence_cannot_surface" not in decision.hard_deny_reasons


def test_safe_morphology_v2_not_hard_denied_by_proxy_measurement_id() -> None:
    synth = ReportSynthesizer()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_morph_v2",
            source_module="event",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.EPILEPTIFORM_MORPHOLOGY,
            value={"morphology_descriptor": "sharp_transient_like"},
            normalized_value={"morphology_descriptor": "sharp_transient_like"},
            measurement_ids=["m_event_morphology_descriptor_v2", "m_event_morphology_proxy_class"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=[SectionRole.DETAIL.value, SectionRole.EPILEPTIFORM.value],
            created_by="test",
        )
    )
    plan = AtomicClaimPlan(
        plan_id="p_morph_v2",
        section_type=ReportSectionType.DETAIL,
        claim_type="epileptiform_morphology",
        proposed_text="Structured morphology evidence suggests sharp transient-like features; this is not a definitive epileptiform interpretation.",
        evidence_ids=["ev_morph_v2"],
        surface_action=ClaimSurfaceAction.CAVEAT,
        allowed_sections=[SectionRole.DETAIL.value, SectionRole.EPILEPTIFORM.value],
        rationale="safe morphology v2 evidence",
    )

    decision = synth.build_surface_decisions([plan], board)[0]

    assert decision.surface_action == ClaimSurfaceAction.CAVEAT
    assert "linked_internal_or_proxy_evidence_cannot_surface" not in decision.hard_deny_reasons


def test_surface_decision_calibrates_blocked_localization_v2_to_caveat() -> None:
    synth = ReportSynthesizer()
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_loc_v2",
            source_module="event",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.LOCALIZATION,
            value={"spatial_pattern": "left posterior predominance, maximal at O1/P3"},
            normalized_value={"spatial_pattern": "left posterior predominance, maximal at O1/P3"},
            measurement_ids=["m_event_spatial_pattern_v2"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=[SectionRole.DETAIL.value, SectionRole.EPILEPTIFORM.value],
            created_by="test",
        )
    )
    plan = AtomicClaimPlan(
        plan_id="p_loc_v2",
        section_type=ReportSectionType.DETAIL,
        claim_type="localization",
        proposed_text="Spatial evidence suggests left posterior predominance, maximal at O1/P3; this remains a caveated event-field observation.",
        evidence_ids=["ev_loc_v2"],
        surface_action=ClaimSurfaceAction.BLOCK,
        allowed_sections=[SectionRole.DETAIL.value, SectionRole.EPILEPTIFORM.value],
        rationale="blocked before calibration",
    )

    decision = synth.build_surface_decisions([plan], board)[0]

    assert decision.surface_action == ClaimSurfaceAction.CAVEAT
    assert decision.decided_by == "surface_policy_calibrated"
