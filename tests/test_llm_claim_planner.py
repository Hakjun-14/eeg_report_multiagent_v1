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
