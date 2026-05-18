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

    result = LLMClaimPlanner(adapter=FakeClaimAdapter()).run(board)

    assert result["status"] == "ok"
    assert result["raw_eeg_used"] is False
    assert result["gt_report_used"] is False
    plans = result["atomic_claim_plan"]
    assert len(plans) == 1
    assert plans[0].plan_id == "p_llm_pdr_claim"
    assert plans[0].surface_action == ClaimSurfaceAction.CAVEAT
    assert plans[0].evidence_ids == ["ev_pdr"]
    assert plans[0].linked_measurement_ids == ["m_pdr"]
    assert "9.5 Hz" in plans[0].proposed_text
