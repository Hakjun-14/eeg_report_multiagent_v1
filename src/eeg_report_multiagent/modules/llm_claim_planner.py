from __future__ import annotations

from typing import Any, Dict, Iterable, List

from eeg_report_multiagent.llm import OpenAIClaimPlanningAdapter
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard
from eeg_report_multiagent.modules.surface_policy import SurfacePolicy


class LLMClaimPlanner:
    """Plan AtomicClaimPlan entries from typed EvidenceItems with an LLM."""

    def __init__(self, adapter: OpenAIClaimPlanningAdapter | None = None) -> None:
        self.adapter = adapter or OpenAIClaimPlanningAdapter()
        self.surface_policy = SurfacePolicy()

    def run(self, shared_board: SharedEvidenceBoard, clinical_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        evidence_items = shared_board.list_evidence()
        payload = self._payload(shared_board, evidence_items, clinical_context or {})
        result = self.adapter.plan(payload)
        plans = self._plans_from_result(evidence_items, result)
        return {
            "status": "ok",
            "model_name": self.adapter.model,
            "summary": str(result.get("summary", "")),
            "raw_eeg_used": bool(result.get("raw_eeg_used", False)),
            "gt_report_used": bool(result.get("gt_report_used", False)),
            "raw_result": result,
            "atomic_claim_plan": plans,
        }

    def _payload(
        self,
        shared_board: SharedEvidenceBoard,
        evidence_items: List[EvidenceItem],
        clinical_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "recording_id": shared_board.recording_id,
            "clinical_context": clinical_context,
            "allowed_surface_actions": [action.value for action in ClaimSurfaceAction],
            "allowed_sections": [role.value for role in SectionRole],
            "evidence_items": [self._evidence_payload(item) for item in evidence_items],
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
            },
        }

    def _evidence_payload(self, item: EvidenceItem) -> Dict[str, Any]:
        return {
            "evidence_id": item.evidence_id,
            "source_module": item.source_module,
            "evidence_type": getattr(item.evidence_type, "value", str(item.evidence_type)),
            "clinical_target": getattr(item.clinical_target, "value", str(item.clinical_target)),
            "value": self._compact_value(item.value),
            "unit": item.unit,
            "normalized_value": self._compact_value(item.normalized_value),
            "time_provenance": self._compact_value(item.time_provenance),
            "space_provenance": self._compact_value(item.space_provenance),
            "measurement_ids": item.measurement_ids,
            "debug_payload_summary": {
                "clinical_knowledge_reference": self._compact_value(item.debug_payload.get("clinical_knowledge_reference")),
                "value_summary": self._compact_value(item.debug_payload.get("value_summary")),
                "measurement_names": self._compact_value(item.debug_payload.get("measurement_names")),
            },
        }

    def _compact_value(self, value: Any, *, depth: int = 0) -> Any:
        """Keep reportable values while preventing large debug arrays from hitting the LLM context window."""

        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if depth >= 4:
            return self._summarize_collection(value)
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for idx, (key, nested) in enumerate(value.items()):
                if idx >= 60:
                    out["__omitted_key_count"] = len(value) - idx
                    break
                out[str(key)] = self._compact_value(nested, depth=depth + 1)
            return out
        if isinstance(value, (list, tuple)):
            if len(value) <= 12:
                return [self._compact_value(nested, depth=depth + 1) for nested in value]
            numeric = [float(x) for x in value if isinstance(x, (int, float))]
            if len(numeric) == len(value) and numeric:
                return {
                    "count": len(numeric),
                    "min": min(numeric),
                    "max": max(numeric),
                    "mean": sum(numeric) / len(numeric),
                    "preview": numeric[:5],
                }
            return {
                "count": len(value),
                "preview": [self._compact_value(nested, depth=depth + 1) for nested in value[:5]],
            }
        return str(value)

    def _summarize_collection(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return {"type": "dict", "key_count": len(value), "keys_preview": [str(key) for key in list(value)[:8]]}
        if isinstance(value, (list, tuple)):
            return {"type": "list", "count": len(value)}
        return {"type": type(value).__name__}

    def _plans_from_result(self, evidence_items: List[EvidenceItem], result: Dict[str, Any]) -> List[AtomicClaimPlan]:
        evidence_index = {item.evidence_id: item for item in evidence_items}
        allowed_sections = {role.value for role in SectionRole}
        plans: List[AtomicClaimPlan] = []
        for idx, claim in enumerate(result.get("atomic_claims", [])):
            evidence_ids = [eid for eid in claim.get("evidence_ids", []) if eid in evidence_index]
            if not evidence_ids:
                continue
            action = ClaimSurfaceAction(claim.get("surface_action", ClaimSurfaceAction.BLOCK.value))
            linked_measurement_ids = self._linked_measurement_ids(evidence_index[eid] for eid in evidence_ids)
            plan_id = self._safe_plan_id(str(claim.get("plan_id") or f"llm_claim_{idx}"))
            proposed_text = str(claim.get("proposed_text") or "")
            rewritten = False
            linked_items = [evidence_index[eid] for eid in evidence_ids]
            if action in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT} and self._should_force_safe_text(linked_items, proposed_text):
                safe_text = self._safe_text_from_evidence(linked_items)
                if safe_text:
                    proposed_text = safe_text
                    rewritten = True
            if action in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT} and self.surface_policy.contains_forbidden_surface_text(proposed_text):
                safe_text = self._safe_text_from_evidence([evidence_index[eid] for eid in evidence_ids])
                if safe_text:
                    proposed_text = safe_text
                    rewritten = True
            plans.append(
                AtomicClaimPlan(
                    plan_id=plan_id,
                    section_type=ReportSectionType.DETAIL,
                    claim_type=str(claim.get("claim_type") or "llm_planned_claim"),
                    proposed_text=proposed_text,
                    evidence_ids=evidence_ids,
                    linked_measurement_ids=linked_measurement_ids,
                    required_evidence=[str(x) for x in claim.get("required_evidence", [])],
                    missing_evidence=[str(x) for x in claim.get("missing_evidence", [])],
                    surface_action=action,
                    confidence=None,
                    rationale=str(claim.get("rationale") or "LLM planned claim from typed EvidenceItems."),
                    allowed_sections=self._safe_allowed_sections(
                        [evidence_index[eid] for eid in evidence_ids],
                        [section for section in claim.get("allowed_sections", []) if section in allowed_sections],
                    ),
                    forbidden_sections=[],
                    clinical_phrase_template_id="llm_atomic_claim",
                    debug_payload={
                        "llm_claim_planning": True,
                        "rewritten_from_unsafe_llm_text": rewritten,
                        "raw_surface_action": str(claim.get("surface_action", "")),
                    },
                )
            )
        plans.extend(self._coverage_guard_claims(evidence_items, plans))
        return plans

    def _coverage_guard_claims(
        self,
        evidence_items: List[EvidenceItem],
        existing_plans: List[AtomicClaimPlan],
    ) -> List[AtomicClaimPlan]:
        claimed_evidence_ids = {evidence_id for plan in existing_plans for evidence_id in plan.evidence_ids}
        out: List[AtomicClaimPlan] = []
        for item in evidence_items:
            target = str(getattr(item.clinical_target, "value", item.clinical_target))
            evidence_type = getattr(item.evidence_type, "value", item.evidence_type)
            if item.evidence_id in claimed_evidence_ids:
                continue
            if evidence_type not in {EvidenceType.DIRECT.value, EvidenceType.DERIVED.value, EvidenceType.METADATA.value}:
                continue
            if target not in {ClinicalTarget.PDR.value, ClinicalTarget.BACKGROUND_AMPLITUDE.value}:
                continue
            proposed_text = self._safe_text_from_evidence([item])
            if not proposed_text or self.surface_policy.contains_forbidden_surface_text(proposed_text):
                continue
            out.append(
                AtomicClaimPlan(
                    plan_id=self._safe_plan_id(f"coverage_{item.evidence_id}"),
                    section_type=ReportSectionType.DETAIL,
                    claim_type=target,
                    proposed_text=proposed_text,
                    evidence_ids=[item.evidence_id],
                    linked_measurement_ids=list(item.measurement_ids),
                    required_evidence=[],
                    missing_evidence=[],
                    surface_action=ClaimSurfaceAction.CAVEAT,
                    confidence=None,
                    rationale="Coverage guard preserved safe reportable evidence omitted by LLM claim planning.",
                    allowed_sections=self._safe_allowed_sections([item], list(item.allowed_sections)),
                    forbidden_sections=[],
                    clinical_phrase_template_id="llm_atomic_claim_coverage_guard",
                    debug_payload={
                        "llm_claim_planning": True,
                        "coverage_guard_for_omitted_safe_evidence": True,
                    },
                )
            )
        return out

    def _should_force_safe_text(self, evidence_items: List[EvidenceItem], proposed_text: str) -> bool:
        targets = {str(getattr(item.clinical_target, "value", item.clinical_target)) for item in evidence_items}
        lowered = proposed_text.lower()
        if targets & {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            return True
        if ClinicalTarget.PDR.value in targets:
            return True
        if "unknown" in lowered and "presence" in lowered:
            return True
        return False

    def _safe_allowed_sections(self, evidence_items: List[EvidenceItem], llm_sections: List[str]) -> List[str]:
        targets = {str(getattr(item.clinical_target, "value", item.clinical_target)) for item in evidence_items}
        sections = set(llm_sections)
        sections.add(SectionRole.DETAIL.value)
        if targets & {ClinicalTarget.PDR.value, ClinicalTarget.BACKGROUND_SLOWING.value, ClinicalTarget.BACKGROUND_AMPLITUDE.value}:
            sections.update({SectionRole.BACKGROUND.value, SectionRole.IMPRESSION.value})
        if targets & {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            sections.update({SectionRole.BACKGROUND.value, SectionRole.SLEEP.value})
        if targets & {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value, ClinicalTarget.EVENT_CANDIDATE.value}:
            sections.update({SectionRole.EPILEPTIFORM.value, SectionRole.EVENTS_SEIZURES.value})
        return [role.value for role in SectionRole if role.value in sections]

    def _safe_text_from_evidence(self, evidence_items: List[EvidenceItem]) -> str:
        targets = {str(getattr(item.clinical_target, "value", item.clinical_target)) for item in evidence_items}
        if ClinicalTarget.PDR.value in targets:
            freq = self._first_value(evidence_items, "frequency_hz")
            if isinstance(freq, (int, float)) and 8.0 <= float(freq) <= 13.0:
                return f"A posterior alpha rhythm candidate is approximately {float(freq):.1f} Hz; state/reactivity confirmation remains incomplete."
        if ClinicalTarget.BACKGROUND_AMPLITUDE.value in targets:
            typical_amp = self._first_value(evidence_items, "background_amplitude_typical_uv")
            if isinstance(typical_amp, (int, float)) and float(typical_amp) > 0.0:
                return f"A provenance-linked typical background amplitude is approximately {float(typical_amp):.1f} uV."
            amp = self._first_value(evidence_items, "background_amplitude_range_uv")
            if isinstance(amp, dict) and amp.get("upper") is not None:
                lo = amp.get("lower", 0)
                hi = amp["upper"]
                return f"A provenance-linked background amplitude range is available ({float(lo):g}-{float(hi):.2f} uV)."
        if ClinicalTarget.BACKGROUND_SLOWING.value in targets:
            return "Structured evidence suggests possible background slowing; this remains an assistive observation pending EEG review."
        if ClinicalTarget.STATE.value in targets or ClinicalTarget.PROTOCOL.value in targets:
            return "Structured protocol/context status is available but remains non-specific."
        return ""

    def _first_value(self, evidence_items: List[EvidenceItem], key: str) -> Any:
        for item in evidence_items:
            value = self._find_value(item.value, key)
            if value is not None:
                return value
            value = self._find_value(item.normalized_value, key)
            if value is not None:
                return value
        return None

    def _find_value(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            if key in value:
                return value[key]
            for nested in value.values():
                found = self._find_value(nested, key)
                if found is not None:
                    return found
        return None

    def _linked_measurement_ids(self, items: Iterable[EvidenceItem]) -> List[str]:
        out: list[str] = []
        for item in items:
            out.extend(item.measurement_ids)
        return sorted(set(out))

    def _safe_plan_id(self, plan_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in plan_id.strip())
        if not safe.startswith("p_"):
            safe = f"p_llm_{safe}"
        return safe
