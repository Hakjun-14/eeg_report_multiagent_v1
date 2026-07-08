from eeg_report_multiagent.modules.llm_claim_planner import LLMClaimPlanner
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


class FakeClaimAdapter:
    model = "fake-claim"

    def __init__(self):
        self.payload = None

    def plan(self, payload):
        self.payload = payload
        assert payload["privacy_contract"]["contains_raw_eeg"] is False
        assert payload["privacy_contract"]["contains_gt_report_text"] is False
        assert "gt_report" not in payload
        return {
            "summary": "planned claims",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "atomic_claims": [
                {
                    "plan_id": "pdr_claim",
                    "claim_type": "pdr",
                    "proposed_text": "A posterior alpha rhythm candidate is approximately 9.5 Hz; reactivity is not confirmed.",
                    "evidence_ids": ["ev_pdr", "missing_ev"],
                    "surface_action": "caveat",
                    "allowed_sections": ["detail", "background"],
                    "required_evidence": ["posterior alpha frequency"],
                    "missing_evidence": ["reactivity"],
                    "rationale": "Linked to PDR evidence.",
                },
                {
                    "plan_id": "bad_claim",
                    "claim_type": "unknown",
                    "proposed_text": "No evidence link.",
                    "evidence_ids": ["missing_ev"],
                    "surface_action": "allow",
                    "allowed_sections": ["detail"],
                    "required_evidence": [],
                    "missing_evidence": [],
                    "rationale": "Should be skipped.",
                },
            ],
        }


class EmptyClaimAdapter:
    model = "fake-empty-claim"

    def plan(self, payload):
        self.payload = payload
        return {
            "summary": "missed safe claim",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "atomic_claims": [],
        }


def test_llm_claim_planner_creates_atomic_claims_from_evidence_items_only():
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_pdr",
            source_module="test",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.PDR,
            value={"frequency_hz": 9.5, "pdr_supported": "true"},
            measurement_ids=["m_pdr"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            created_by="test",
        )
    )

    clinical_context = {
        "patient_history_and_eeg_description": "Altered awareness, seizure versus syncope.",
        "metadata": {"indication": "altered awareness"},
        "gt_report_text_included": False,
    }
    adapter = FakeClaimAdapter()
    result = LLMClaimPlanner(adapter=adapter).run(board, clinical_context=clinical_context)

    assert result["status"] == "ok"
    assert result["raw_eeg_used"] is False
    assert result["gt_report_used"] is False
    assert adapter.payload["clinical_context"] == clinical_context
    plans = result["atomic_claim_plan"]
    assert len(plans) == 1
    assert plans[0].plan_id == "p_llm_pdr_claim"
    assert plans[0].surface_action == ClaimSurfaceAction.CAVEAT
    assert plans[0].evidence_ids == ["ev_pdr"]
    assert plans[0].linked_measurement_ids == ["m_pdr"]
    assert "9.5 Hz" in plans[0].proposed_text
    assert plans[0].numeric_claims == [
        {
            "slot": "pdr_frequency",
            "value": 9.5,
            "unit": "Hz",
            "evidence_id": "ev_pdr",
            "render_required": True,
            "render_text": "9.5 Hz",
            "source": "surface_safe_values",
        }
    ]


def test_llm_claim_planner_coverage_guard_preserves_safe_pdr_when_llm_omits_it():
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_pdr",
            source_module="test",
            evidence_type=EvidenceType.DIRECT,
            clinical_target=ClinicalTarget.PDR,
            value={"frequency_hz": 9.5, "pdr_supported": "true"},
            normalized_value={"frequency_hz": 9.5, "pdr_supported": "true"},
            unit="Hz",
            measurement_ids=["m_pdr"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            created_by="test",
        )
    )

    result = LLMClaimPlanner(adapter=EmptyClaimAdapter()).run(board)

    plans = result["atomic_claim_plan"]
    assert len(plans) == 1
    assert plans[0].claim_type == "pdr"
    assert plans[0].surface_action == ClaimSurfaceAction.CAVEAT
    assert plans[0].evidence_ids == ["ev_pdr"]
    assert "9.5 Hz" in plans[0].proposed_text
    assert plans[0].debug_payload["coverage_guard_for_omitted_safe_evidence"] is True
    assert plans[0].must_render_values == ["pdr_frequency=9.5 Hz"]
    assert plans[0].numeric_claims[0]["render_text"] == "9.5 Hz"


def test_llm_claim_planner_coverage_checklist_requires_state_protocol_decisions():
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_state",
            source_module="metadata",
            evidence_type=EvidenceType.METADATA,
            clinical_target=ClinicalTarget.STATE,
            value={"state_awake": "present", "state_sleep": "unknown"},
            normalized_value={"state_awake": "present", "state_sleep": "unknown"},
            measurement_ids=["m_state_awake", "m_state_sleep"],
            reportability=ClaimSurfaceAction.ALLOW,
            allowed_sections=["detail"],
            created_by="test",
        )
    )
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_protocol",
            source_module="metadata",
            evidence_type=EvidenceType.METADATA,
            clinical_target=ClinicalTarget.PROTOCOL,
            value={"photic_stimulation_status": "not_performed", "hyperventilation_status": "unknown"},
            normalized_value={"photic_stimulation_status": "not_performed", "hyperventilation_status": "unknown"},
            measurement_ids=["m_photic_status", "m_hv_status"],
            reportability=ClaimSurfaceAction.ALLOW,
            allowed_sections=["detail"],
            created_by="test",
        )
    )

    adapter = EmptyClaimAdapter()
    result = LLMClaimPlanner(adapter=adapter).run(board)

    assert "claim_coverage_checklist" in adapter.payload
    assert len(adapter.payload["claim_coverage_checklist"]) == 2
    assert adapter.payload["claim_planning_contract"]["posthoc_coverage_guard_is_enabled"] is True
    assert adapter.payload["claim_planning_contract"]["llm_must_account_for_claim_coverage_checklist"] is True

    plans = result["atomic_claim_plan"]
    assert {plan.claim_type for plan in plans} == {"state", "protocol"}
    assert all(plan.surface_action == ClaimSurfaceAction.ALLOW for plan in plans)
    assert any("state_awake=present" in plan.must_render_values for plan in plans)
    assert any("photic_stimulation_status=not_performed" in plan.must_render_values for plan in plans)
    assert all(plan.debug_payload["coverage_guard_for_omitted_safe_evidence"] is True for plan in plans)
    assert result["claim_planner_coverage"]["expected_coverage_count"] == 2
    assert result["claim_planner_coverage"]["coverage_accounted_count"] == 0
    assert result["claim_planner_coverage"]["posthoc_coverage_guard_claim_count"] == 2


def test_llm_claim_planner_coverage_trace_accepts_normalized_plan_ids():
    class CoverageDecisionAdapter:
        model = "fake-coverage-decision"

        def plan(self, payload):
            coverage_id = payload["claim_coverage_checklist"][0]["coverage_id"]
            return {
                "summary": "planned with coverage",
                "raw_eeg_used": False,
                "gt_report_used": False,
                "coverage_decisions": [
                    {
                        "coverage_id": coverage_id,
                        "evidence_ids": ["ev_pdr"],
                        "decision": "claim_created",
                        "linked_plan_id": coverage_id,
                        "reason": "Created a claim for the checklist item.",
                    }
                ],
                "atomic_claims": [
                    {
                        "plan_id": coverage_id,
                        "claim_type": "pdr",
                        "proposed_text": "A posterior alpha rhythm candidate is approximately 9.5 Hz; reactivity is not confirmed.",
                        "evidence_ids": ["ev_pdr"],
                        "surface_action": "caveat",
                        "allowed_sections": ["detail", "background"],
                        "required_evidence": [],
                        "missing_evidence": ["reactivity"],
                        "rationale": "Linked to PDR evidence.",
                    }
                ],
            }

    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_pdr",
            source_module="test",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.PDR,
            value={"frequency_hz": 9.5, "pdr_supported": "true"},
            normalized_value={"frequency_hz": 9.5, "pdr_supported": "true"},
            unit="Hz",
            measurement_ids=["m_pdr"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            created_by="test",
        )
    )

    result = LLMClaimPlanner(adapter=CoverageDecisionAdapter()).run(board)

    assert result["claim_planner_coverage"]["coverage_accounted_rate"] == 1.0
    assert result["claim_planner_coverage"]["invalid_linked_plan_coverage_ids"] == []
    assert result["atomic_claim_plan"][0].plan_id == "p_llm_coverage_ev_pdr"


def test_llm_claim_planner_coverage_guard_does_not_override_existing_block_decision():
    class BlockDecisionAdapter:
        model = "fake-block-decision"

        def plan(self, payload):
            coverage_id = payload["claim_coverage_checklist"][0]["coverage_id"]
            return {
                "summary": "blocked with coverage",
                "raw_eeg_used": False,
                "gt_report_used": False,
                "coverage_decisions": [
                    {
                        "coverage_id": coverage_id,
                        "evidence_ids": ["ev_slowing"],
                        "decision": "blocked",
                        "linked_plan_id": coverage_id,
                        "reason": "Insufficient support for a reportable slowing claim.",
                    }
                ],
                "atomic_claims": [
                    {
                        "plan_id": coverage_id,
                        "claim_type": "background_slowing",
                        "proposed_text": "",
                        "evidence_ids": ["ev_slowing"],
                        "surface_action": "block",
                        "allowed_sections": ["detail", "background"],
                        "required_evidence": [],
                        "missing_evidence": ["state support"],
                        "rationale": "Insufficient support.",
                    }
                ],
            }

    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_slowing",
            source_module="test",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.BACKGROUND_SLOWING,
            value={"slowing_present": "possible"},
            normalized_value={"slowing_present": "possible"},
            measurement_ids=["m_slowing"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            created_by="test",
        )
    )

    result = LLMClaimPlanner(adapter=BlockDecisionAdapter()).run(board)

    plans = result["atomic_claim_plan"]
    assert len(plans) == 1
    assert plans[0].surface_action == ClaimSurfaceAction.BLOCK
    assert plans[0].debug_payload.get("coverage_guard_for_omitted_safe_evidence") is None
    assert result["claim_planner_coverage"]["posthoc_coverage_guard_claim_count"] == 0


def test_llm_claim_planner_compacts_large_evidence_arrays_but_keeps_reportable_values():
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_event",
            source_module="test",
            evidence_type=EvidenceType.PROXY,
            clinical_target=ClinicalTarget.EVENT_CANDIDATE,
            value={"event_candidate_score_distribution": [float(i) for i in range(1000)]},
            measurement_ids=["m_event"],
            reportability=ClaimSurfaceAction.BLOCK,
            allowed_sections=["detail"],
            created_by="test",
        )
    )
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_pdr",
            source_module="test",
            evidence_type=EvidenceType.DIRECT,
            clinical_target=ClinicalTarget.PDR,
            value={"frequency_hz": 9.5, "pdr_supported": True},
            measurement_ids=["m_pdr"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            created_by="test",
        )
    )

    adapter = EmptyClaimAdapter()
    LLMClaimPlanner(adapter=adapter).run(board)

    payload_items = {item["evidence_id"]: item for item in adapter.payload["evidence_items"]}
    event_dist = payload_items["ev_event"]["value"]["event_candidate_score_distribution"]
    assert event_dist["count"] == 1000
    assert event_dist["min"] == 0.0
    assert event_dist["max"] == 999.0
    assert len(event_dist["preview"]) == 5
    assert payload_items["ev_pdr"]["value"]["frequency_hz"] == 9.5


def test_llm_claim_planner_closes_same_measurement_evidence_links_for_numeric_trace():
    class GroupOnlyAdapter:
        model = "fake-group-only"

        def plan(self, payload):
            return {
                "summary": "planned",
                "raw_eeg_used": False,
                "gt_report_used": False,
                "atomic_claims": [
                    {
                        "plan_id": "amp_claim",
                        "claim_type": "background_amplitude",
                        "proposed_text": "Background amplitude typical is approximately 18.87 uV and ranges from 16.21 to 22.37 uV.",
                        "evidence_ids": ["ev_llm_group_background_activity"],
                        "surface_action": "allow",
                        "allowed_sections": ["detail"],
                        "required_evidence": [],
                        "missing_evidence": [],
                        "rationale": "Linked to LLM grouped amplitude evidence.",
                    }
                ],
            }

    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="ev_llm_group_background_activity",
            source_module="llm_evidence_grouper",
            evidence_type=EvidenceType.DIRECT,
            clinical_target=ClinicalTarget.BACKGROUND_AMPLITUDE,
            value={"background_amplitude_range_uv": 18.873563766479492},
            normalized_value={"background_amplitude_range_uv": 18.873563766479492},
            unit="uV",
            measurement_ids=["m_background_amplitude_typical", "m_background_amplitude_range"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            created_by="test",
        )
    )
    board.add_evidence(
        EvidenceItem(
            evidence_id="evgrp_background_amplitude",
            source_module="background",
            evidence_type=EvidenceType.DERIVED,
            clinical_target=ClinicalTarget.BACKGROUND_AMPLITUDE,
            value={
                "background_amplitude_range_uv": {"lower": 16.208969116210938, "upper": 22.37342071533203},
                "background_amplitude_typical_uv": 18.873563766479492,
            },
            normalized_value={
                "background_amplitude_range_uv": {"lower": 16.208969116210938, "upper": 22.37342071533203},
                "background_amplitude_typical_uv": 18.873563766479492,
            },
            unit="uV",
            measurement_ids=["m_background_amplitude_range", "m_background_amplitude_typical"],
            reportability=ClaimSurfaceAction.CAVEAT,
            allowed_sections=["detail", "background"],
            created_by="test",
        )
    )

    plans = LLMClaimPlanner(adapter=GroupOnlyAdapter()).run(board)["atomic_claim_plan"]

    assert len(plans) == 1
    assert plans[0].evidence_ids == ["ev_llm_group_background_activity", "evgrp_background_amplitude"]
    assert "background_amplitude_typical=18.9 uV" in plans[0].must_render_values
    assert "background_amplitude_range=16.2-22.4 uV" not in plans[0].must_render_values
    assert plans[0].numeric_claims == [
        {
            "slot": "background_amplitude_typical",
            "value": 18.873563766479492,
            "unit": "uV",
            "evidence_id": "evgrp_background_amplitude",
            "render_required": True,
            "render_text": "18.9 uV",
            "source": "surface_safe_values",
        }
    ]


def test_llm_claim_planner_forces_v2_morphology_to_caveated_safe_text():
    class UnsafeMorphologyAdapter:
        model = "fake-morphology"

        def plan(self, payload):
            return {
                "summary": "planned",
                "raw_eeg_used": False,
                "gt_report_used": False,
                "atomic_claims": [
                    {
                        "plan_id": "morphology_claim",
                        "claim_type": "epileptiform_morphology",
                        "proposed_text": "Epileptiform morphology observed as sharp transient-like activity.",
                        "evidence_ids": ["evgrp_epileptiform_morphology_v2"],
                        "surface_action": "allow",
                        "allowed_sections": ["detail", "epileptiform"],
                        "required_evidence": [],
                        "missing_evidence": [],
                        "rationale": "LLM over-upgraded morphology evidence.",
                    }
                ],
            }

    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(
        EvidenceItem(
            evidence_id="evgrp_epileptiform_morphology_v2",
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

    plans = LLMClaimPlanner(adapter=UnsafeMorphologyAdapter()).run(board)["atomic_claim_plan"]

    assert len(plans) == 1
    assert plans[0].surface_action == ClaimSurfaceAction.CAVEAT
    assert "not a definitive epileptiform interpretation" in plans[0].proposed_text
    assert plans[0].must_render_values == ["morphology_descriptor=sharp_transient_like"]
