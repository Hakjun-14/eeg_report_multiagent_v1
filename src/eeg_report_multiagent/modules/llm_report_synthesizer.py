from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from eeg_report_multiagent.llm import OpenAIReportSynthesisAdapter
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.modules.surface_policy import SurfacePolicy
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction


@dataclass(frozen=True)
class LLMReportSynthesisResult:
    section_texts: Dict[str, str]
    trace: Dict[str, Any]


class EvidenceBoardLLMReportSynthesizer:
    """Method D: surface-policy-gated LLM report synthesis.

    The LLM receives only AtomicClaimPlan entries that were already allowed or
    caveated by SurfacePolicy. It never receives raw EEG arrays, pkl paths,
    GT/reference report text, full measurement payloads, or debug/proxy scores.
    """

    synthesizer_name = "evidence_board_llm_report_synthesizer"
    synthesis_version = "D_v2_surface_policy_gated"

    def __init__(
        self,
        adapter: OpenAIReportSynthesisAdapter | None = None,
        report_synthesizer: ReportSynthesizer | None = None,
        surface_policy: SurfacePolicy | None = None,
    ) -> None:
        self.adapter = adapter or OpenAIReportSynthesisAdapter()
        self.surface_policy = surface_policy or SurfacePolicy()
        self.report_synthesizer = report_synthesizer or ReportSynthesizer(surface_policy=self.surface_policy)
        self.section_router = SectionRouter()

    def synthesize_celm_sections(
        self,
        board: EvidenceBoard,
        target_section_names: List[str],
    ) -> LLMReportSynthesisResult:
        payload = self._build_payload(board, target_section_names)
        model_output = self.adapter.synthesize(payload)
        section_texts = self._validate_and_align_sections(model_output, target_section_names)
        trace = {
            "synthesizer_name": self.synthesizer_name,
            "synthesis_version": self.synthesis_version,
            "model_name": self.adapter.model,
            "raw_eeg_used": bool(model_output.get("raw_eeg_used")),
            "gt_report_used": bool(model_output.get("gt_report_used")),
            "target_section_names": target_section_names,
            "global_limitations": model_output.get("global_limitations", []),
            "model_response_id": model_output.get("_response_id"),
            "model_report_sections": model_output.get("report_sections", []),
            "privacy_contract": payload["privacy_contract"],
            "surface_payload_summary": payload["surface_payload_summary"],
        }
        if trace["raw_eeg_used"] or trace["gt_report_used"]:
            raise ValueError("D synthesis violated input contract")
        return LLMReportSynthesisResult(section_texts=section_texts, trace=trace)

    def _build_payload(self, board: EvidenceBoard, target_section_names: List[str]) -> Dict[str, Any]:
        claim_plans = self.report_synthesizer.build_atomic_claim_plan(board)
        surface_decisions = self.report_synthesizer.build_surface_decisions(
            claim_plans,
            board.ensure_shared_evidence_board(),
        )
        decision_by_claim = {decision.claim_id: decision for decision in surface_decisions}
        surface_plans = [
            plan
            for plan in claim_plans
            if (
                decision_by_claim.get(plan.plan_id, self.surface_policy.decide(plan)).surface_action
                in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}
            )
            and not self.surface_policy.contains_forbidden_surface_text(plan.proposed_text)
        ]
        surface_claim_ids = {plan.plan_id for plan in surface_plans}
        return {
            "session_id": board.session_id,
            "target_section_names": target_section_names,
            "atomic_claim_plans": [self._claim_plan_payload(plan) for plan in surface_plans],
            "surface_decisions": [
                self._surface_decision_payload(decision)
                for decision in surface_decisions
                if decision.claim_id in surface_claim_ids
            ],
            "surface_payload_summary": {
                "total_atomic_claim_plans": len(claim_plans),
                "surface_allowed_or_caveated_claim_plans": len(surface_plans),
                "blocked_or_debug_only_claim_plans": len(claim_plans) - len(surface_plans),
                "audit_only_review_record_count": self._audit_record_count(board),
            },
            "style_policy": {
                "tone": "formal_clinical_eeg_report",
                "allowed_claims": "verbalize only atomic_claim_plans supplied in this payload",
                "uncertainty_handling": "preserve caveated wording from the atomic claim plan and do not upgrade certainty",
                "section_behavior": "generate each requested section once and do not add extra sections",
                "debug_surface_separation": (
                    "Do not mention internal detector scores, proxy labels, reviewer raw text, or measurements "
                    "that are not already translated into an atomic claim plan."
                ),
            },
            "forbidden_inputs": [
                "raw_eeg_arrays",
                "processed_pkl_payloads",
                "reference_gt_report_text",
                "full_measurement_payloads",
                "full_evidence_item_payloads",
                "raw_evidence_reviewer_text",
                "debug_or_proxy_score_payloads",
                "unbounded_external_tools",
            ],
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
                "contains_source_pkl_paths": False,
                "contains_full_measurements": False,
                "contains_full_evidence_items": False,
                "contains_debug_scores": False,
            },
        }

    def _claim_plan_payload(self, plan: AtomicClaimPlan) -> Dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "claim_type": plan.claim_type,
            "proposed_text": plan.proposed_text,
            "evidence_ids": plan.evidence_ids,
            "surface_action": plan.surface_action.value,
            "allowed_sections": plan.allowed_sections,
            "clinical_phrase_template_id": plan.clinical_phrase_template_id,
            "linked_measurement_ids": plan.linked_measurement_ids,
            "required_evidence": plan.required_evidence,
            "missing_evidence": plan.missing_evidence,
            "confidence": plan.confidence,
            "rationale": plan.rationale,
        }

    def _surface_decision_payload(self, decision: Any) -> Dict[str, Any]:
        return {
            "decision_id": decision.decision_id,
            "claim_id": decision.claim_id,
            "surface_action": decision.surface_action.value,
            "allowed_sections": decision.allowed_sections,
            "forbidden_sections": decision.forbidden_sections,
            "clinical_phrase_template_id": decision.clinical_phrase_template_id,
            "rationale": decision.rationale,
            "hard_deny_reasons": decision.hard_deny_reasons,
            "evidence_ids": decision.evidence_ids,
            "caveat": decision.caveat,
            "decided_by": decision.decided_by,
        }

    def _audit_record_count(self, board: EvidenceBoard) -> int:
        return sum(
            len(d.weak_evidence) + len(d.missing_slots) + len(d.do_not_claim) + len(d.claim_constraints)
            for d in board.deliberations
        )

    def _validate_and_align_sections(self, model_output: Dict[str, Any], target_section_names: List[str]) -> Dict[str, str]:
        raw_sections = model_output.get("report_sections") or []
        by_name = {
            str(item.get("section_name", "")).strip().lower(): str(item.get("section_text", "")).strip()
            for item in raw_sections
            if isinstance(item, dict)
        }
        section_texts: Dict[str, str] = {}
        for idx, target in enumerate(target_section_names):
            key = target.strip().lower()
            text = by_name.get(key)
            if text is None and idx < len(raw_sections) and isinstance(raw_sections[idx], dict):
                text = str(raw_sections[idx].get("section_text", "")).strip()
            text = text or self.surface_policy.safe_fallback_for_role(self.section_router.role_for_section(target))
            section_texts[target] = self._sanitize_section_text(target, text)
        return section_texts

    def _sanitize_section_text(self, section_name: str, text: str) -> str:
        if self.surface_policy.contains_forbidden_surface_text(text):
            return self.surface_policy.safe_fallback_for_role(self.section_router.role_for_section(section_name))
        return text
