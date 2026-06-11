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
        clinical_context: Dict[str, Any] | None = None,
        claim_plan_override: List[AtomicClaimPlan] | None = None,
    ) -> LLMReportSynthesisResult:
        payload = self._build_payload(board, target_section_names, clinical_context or {}, claim_plan_override)
        model_output = self.adapter.synthesize(payload)
        section_texts = self._validate_and_align_sections(
            model_output,
            target_section_names,
            payload["surface_payload_summary"].get("surface_allowed_or_caveated_claim_plans_by_section", {}),
        )
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
            "claim_plan_source": payload["claim_plan_source"],
        }
        if trace["raw_eeg_used"] or trace["gt_report_used"]:
            raise ValueError("D synthesis violated input contract")
        return LLMReportSynthesisResult(section_texts=section_texts, trace=trace)

    def _build_payload(
        self,
        board: EvidenceBoard,
        target_section_names: List[str],
        clinical_context: Dict[str, Any],
        claim_plan_override: List[AtomicClaimPlan] | None = None,
    ) -> Dict[str, Any]:
        claim_plans = claim_plan_override if claim_plan_override is not None else self.report_synthesizer.build_atomic_claim_plan(board)
        claim_plan_source = "artifact_atomic_claim_plan" if claim_plan_override is not None else "rule_based_fallback"
        shared_board = board.ensure_shared_evidence_board()
        surface_decisions = self.report_synthesizer.build_surface_decisions(
            claim_plans,
            shared_board,
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
            "claim_plan_source": claim_plan_source,
            "target_section_names": target_section_names,
            "section_descriptions": self._section_descriptions(target_section_names),
            "clinical_context": clinical_context,
            "atomic_claim_plans": [
                self._claim_plan_payload(plan, shared_board)
                for plan in surface_plans
            ],
            "surface_decisions": [
                self._surface_decision_payload(decision)
                for decision in surface_decisions
                if decision.claim_id in surface_claim_ids
            ],
            "surface_payload_summary": {
                "total_atomic_claim_plans": len(claim_plans),
                "surface_allowed_or_caveated_claim_plans": len(surface_plans),
                "surface_allowed_or_caveated_claim_plans_by_section": self._surface_plan_counts_by_section(
                    surface_plans,
                    surface_decisions,
                    target_section_names,
                ),
                "blocked_or_debug_only_claim_plans": len(claim_plans) - len(surface_plans),
                "audit_only_review_record_count": self._audit_record_count(board),
            },
            "style_policy": {
                "tone": "formal_clinical_eeg_report",
                "allowed_claims": "verbalize only atomic_claim_plans supplied in this payload",
                "linked_evidence_use": (
                    "Use linked_reportable_evidence only to preserve clinically reportable values, units, "
                    "and provenance for the supplied atomic claim. Do not create new claims from evidence alone."
                ),
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
                "unsafe_evidence_item_fields",
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

    def _section_descriptions(self, target_section_names: List[str]) -> Dict[str, str]:
        descriptions = {
            "EEG DESCRIPTION/DETAILS": (
                "Detailed narrative of EEG findings including background activity, sleep stages, "
                "physiologic variants, activation procedures, and abnormalities observed during the recording period."
            ),
            "IMPRESSION/INTERPRETATION": (
                "Concise clinical interpretation summarizing the most important supported EEG findings and limitations."
            ),
            "BACKGROUND ACTIVITY": (
                "Background rhythm description including PDR, organization, symmetry, amplitude, reactivity, slowing, and beta activity when supported."
            ),
            "SLEEP": (
                "Sleep and drowsiness findings including vertex waves, spindles, K-complexes, and state-dependent abnormalities when supported."
            ),
            "EPILEPTIFORM ABNORMALITIES": (
                "Interictal epileptiform findings including morphology, frequency, field, laterality, localization, state dependence, and uncertainty when supported."
            ),
            "INTERICTAL EPILEPTIFORM ABNORMALITIES": (
                "Interictal epileptiform findings including morphology, frequency, field, laterality, localization, state dependence, and uncertainty when supported."
            ),
            "EVENTS/SEIZURES": (
                "Clinical or electrographic events, push-button events, and seizure-related findings only when event or seizure-specific evidence supports them."
            ),
            "SEIZURES": (
                "Seizure presence or absence only when seizure-specific evidence or validated metadata supports the statement."
            ),
        }
        return {section: descriptions.get(section.upper(), "Requested EEG report section; use only supplied allowed/caveated claims.") for section in target_section_names}

    def _claim_plan_payload(self, plan: AtomicClaimPlan, shared_board: Any) -> Dict[str, Any]:
        linked_reportable_evidence = self._linked_reportable_evidence_payload(plan, shared_board)
        reportable_evidence_values = self._reportable_evidence_values(linked_reportable_evidence)
        return {
            "plan_id": plan.plan_id,
            "claim_type": plan.claim_type,
            "proposed_text": plan.proposed_text,
            "evidence_ids": plan.evidence_ids,
            "linked_reportable_evidence": linked_reportable_evidence,
            "reportable_evidence_values": reportable_evidence_values,
            "surface_value_requirements": self._surface_value_requirements(reportable_evidence_values),
            "surface_action": plan.surface_action.value,
            "allowed_sections": plan.allowed_sections,
            "clinical_phrase_template_id": plan.clinical_phrase_template_id,
            "linked_measurement_ids": plan.linked_measurement_ids,
            "required_evidence": plan.required_evidence,
            "missing_evidence": plan.missing_evidence,
            "confidence": plan.confidence,
            "rationale": plan.rationale,
        }

    def _reportable_evidence_values(self, evidence_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compact value view for report wording; contains no debug/proxy values."""
        out: List[Dict[str, Any]] = []
        for payload in evidence_payloads:
            value = payload.get("normalized_value")
            if value in (None, "", {}, []):
                value = payload.get("value")
            if value in (None, "", {}, []):
                continue
            out.append(
                {
                    "evidence_id": payload.get("evidence_id"),
                    "clinical_target": payload.get("clinical_target"),
                    "value": value,
                    "unit": payload.get("unit"),
                }
            )
        return out

    def _surface_value_requirements(self, evidence_values: List[Dict[str, Any]]) -> List[str]:
        """Human-readable safe numeric/status values the report LLM should preserve."""
        requirements: list[str] = []
        for item in evidence_values:
            target = str(item.get("clinical_target") or "")
            unit = item.get("unit")
            value = item.get("value")
            if target == "pdr" and isinstance(value, dict):
                freq = value.get("frequency_hz")
                if isinstance(freq, (int, float)):
                    requirements.append(f"preserve PDR candidate frequency: {float(freq):.1f} {unit or 'Hz'}")
            elif target == "background_amplitude" and isinstance(value, dict):
                typical = value.get("background_amplitude_typical_uv")
                if isinstance(typical, (int, float)) and float(typical) > 0.0:
                    requirements.append(f"preserve typical background amplitude: {float(typical):.1f} {unit or 'uV'}")
                    continue
                amp_range = value.get("background_amplitude_range_uv")
                if isinstance(amp_range, dict) and amp_range.get("upper") is not None:
                    lo = amp_range.get("lower", 0.0)
                    hi = amp_range["upper"]
                    requirements.append(f"preserve background amplitude range: {float(lo):.1f}-{float(hi):.1f} {unit or 'uV'}")
            elif isinstance(value, (int, float)) and unit:
                requirements.append(f"preserve {target} value: {float(value):.1f} {unit}")
        return requirements

    def _linked_reportable_evidence_payload(self, plan: AtomicClaimPlan, shared_board: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for evidence_id in plan.evidence_ids:
            try:
                item = shared_board.get_evidence(evidence_id)
            except KeyError:
                continue
            payload = self._surface_safe_evidence_payload(item)
            if payload:
                out.append(payload)
        return out

    def _surface_safe_evidence_payload(self, item: Any) -> Dict[str, Any] | None:
        evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        if evidence_type in {"debug", "llm_assisted"}:
            return None
        if self._is_internal_or_proxy_evidence_view(item):
            return None
        safe_value_targets = {
            "pdr",
            "background_slowing",
            "background_amplitude",
            "excess_beta",
            "state",
            "protocol",
            "seizure_evidence",
            "context",
        }
        include_value = target in safe_value_targets and evidence_type in {"direct", "derived", "metadata"}
        return {
            "evidence_id": item.evidence_id,
            "source_module": item.source_module,
            "evidence_type": evidence_type,
            "clinical_target": target,
            "value": item.value if include_value else None,
            "unit": item.unit,
            "normalized_value": item.normalized_value if include_value else None,
            "time_provenance": item.time_provenance,
            "space_provenance": item.space_provenance,
            "measurement_ids": list(item.measurement_ids),
            "report_surface_note": (
                "Use only for the linked atomic claim; do not surface proxy/debug/internal values."
            ),
        }

    def _is_internal_or_proxy_evidence_view(self, item: Any) -> bool:
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
        text = " ".join(
            [
                target,
                evidence_type,
                str(item.unit or ""),
                " ".join(item.measurement_ids),
                str(item.value),
                str(item.normalized_value),
            ]
        ).lower()
        if target in {"event_candidate", "epileptiform_morphology", "localization"}:
            if any(term in text for term in ("candidate", "likelihood", "support", "score", "ratio", "concentration", "localization_label")):
                return True
        return any(
            term in text
            for term in (
                "support_score",
                "likelihood_score",
                "candidate_score",
                "candidate_burden",
                "concentration_ratio",
                "laterality_index",
                "bifrontal_ratio",
                "organization_score",
                "bandpower",
                "slowing_score",
                "beta_excess_score",
            )
        ) or str(item.unit or "").lower() in {"score", "ratio"}

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

    def _surface_plan_counts_by_section(
        self,
        surface_plans: List[AtomicClaimPlan],
        surface_decisions: List[Any],
        target_section_names: List[str],
    ) -> Dict[str, int]:
        decision_by_claim = {decision.claim_id: decision for decision in surface_decisions}
        counts = {section_name: 0 for section_name in target_section_names}
        for section_name in target_section_names:
            role = self.section_router.role_for_section(section_name)
            for plan in surface_plans:
                decision = decision_by_claim.get(plan.plan_id, self.surface_policy.decide(plan))
                if decision.surface_action not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                    continue
                if role.value in decision.forbidden_sections:
                    continue
                if decision.allowed_sections and role.value not in decision.allowed_sections:
                    continue
                counts[section_name] += 1
        return counts

    def _validate_and_align_sections(
        self,
        model_output: Dict[str, Any],
        target_section_names: List[str],
        surface_plan_counts_by_section: Dict[str, int] | None = None,
    ) -> Dict[str, str]:
        raw_sections = model_output.get("report_sections") or []
        by_name = {
            str(item.get("section_name", "")).strip().lower(): str(item.get("section_text", "")).strip()
            for item in raw_sections
            if isinstance(item, dict)
        }
        section_texts: Dict[str, str] = {}
        for idx, target in enumerate(target_section_names):
            if surface_plan_counts_by_section is not None and surface_plan_counts_by_section.get(target, 0) <= 0:
                section_texts[target] = self.surface_policy.safe_fallback_for_role(self.section_router.role_for_section(target))
                continue
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
