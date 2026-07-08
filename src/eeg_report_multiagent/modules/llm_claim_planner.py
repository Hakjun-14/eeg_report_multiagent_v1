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
        coverage_trace = self._coverage_trace(
            payload.get("claim_coverage_checklist", []),
            result.get("coverage_decisions", []),
            plans,
        )
        return {
            "status": "ok",
            "model_name": self.adapter.model,
            "summary": str(result.get("summary", "")),
            "raw_eeg_used": bool(result.get("raw_eeg_used", False)),
            "gt_report_used": bool(result.get("gt_report_used", False)),
            "claim_planner_coverage": coverage_trace,
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
            "claim_coverage_checklist": self._coverage_checklist(evidence_items),
            "claim_planning_contract": {
                "llm_must_account_for_claim_coverage_checklist": True,
                "posthoc_coverage_guard_is_enabled": True,
                "also_create_decisions_for_other_clinically_relevant_evidence_items": True,
                "unsafe_or_proxy_items_must_be_blocked_not_surfaced": True,
            },
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
            },
        }

    def _coverage_trace(
        self,
        checklist: List[Dict[str, Any]],
        coverage_decisions: List[Dict[str, Any]],
        plans: List[AtomicClaimPlan],
    ) -> Dict[str, Any]:
        expected_ids = [str(item.get("coverage_id")) for item in checklist if item.get("coverage_id")]
        decisions_by_id = {
            str(item.get("coverage_id")): item
            for item in coverage_decisions
            if isinstance(item, dict) and item.get("coverage_id")
        }
        plan_ids = {plan.plan_id for plan in plans}
        decision_counts: Dict[str, int] = {}
        invalid_linked_plan_ids: List[str] = []
        for coverage_id, decision in decisions_by_id.items():
            label = str(decision.get("decision") or "")
            decision_counts[label] = decision_counts.get(label, 0) + 1
            linked_plan_id = str(decision.get("linked_plan_id") or "")
            normalized_linked_plan_id = self._safe_plan_id(linked_plan_id) if linked_plan_id else ""
            if label == "claim_created" and linked_plan_id not in plan_ids and normalized_linked_plan_id not in plan_ids:
                invalid_linked_plan_ids.append(coverage_id)
        accounted = [coverage_id for coverage_id in expected_ids if coverage_id in decisions_by_id]
        unaccounted = [coverage_id for coverage_id in expected_ids if coverage_id not in decisions_by_id]
        return {
            "expected_coverage_count": len(expected_ids),
            "coverage_decision_count": len(decisions_by_id),
            "coverage_accounted_count": len(accounted),
            "coverage_accounted_rate": (len(accounted) / len(expected_ids)) if expected_ids else 1.0,
            "coverage_decision_counts": decision_counts,
            "unaccounted_coverage_ids": unaccounted,
            "invalid_linked_plan_coverage_ids": invalid_linked_plan_ids,
            "posthoc_coverage_guard_claim_count": sum(
                1 for plan in plans if plan.debug_payload.get("coverage_guard_for_omitted_safe_evidence")
            ),
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
        coverage_checklist = self._coverage_checklist(evidence_items)
        plans: List[AtomicClaimPlan] = []
        for idx, claim in enumerate(result.get("atomic_claims", [])):
            evidence_ids = [eid for eid in claim.get("evidence_ids", []) if eid in evidence_index]
            if not evidence_ids:
                continue
            evidence_ids = self._close_evidence_ids_by_measurement(evidence_ids, evidence_index)
            action = ClaimSurfaceAction(claim.get("surface_action", ClaimSurfaceAction.BLOCK.value))
            linked_measurement_ids = self._linked_measurement_ids(evidence_index[eid] for eid in evidence_ids)
            plan_id = self._safe_plan_id(str(claim.get("plan_id") or f"llm_claim_{idx}"))
            proposed_text = str(claim.get("proposed_text") or "")
            rewritten = False
            linked_items = [evidence_index[eid] for eid in evidence_ids]
            if self._requires_caveated_surface(linked_items) and action == ClaimSurfaceAction.ALLOW:
                action = ClaimSurfaceAction.CAVEAT
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
            surface_safe_values = self._surface_safe_values(linked_items)
            must_render_values = self._must_render_values(surface_safe_values)
            numeric_claims = self._numeric_claims(surface_safe_values)
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
                    surface_safe_values=surface_safe_values,
                    must_render_values=must_render_values,
                    numeric_claims=numeric_claims,
                    debug_payload={
                        "llm_claim_planning": True,
                        "rewritten_from_unsafe_llm_text": rewritten,
                        "raw_surface_action": str(claim.get("surface_action", "")),
                        "evidence_ids_closed_by_measurement": evidence_ids != [eid for eid in claim.get("evidence_ids", []) if eid in evidence_index],
                    },
                )
            )
        plans.extend(self._coverage_guard_claims(evidence_items, plans, coverage_checklist))
        return plans

    def _close_evidence_ids_by_measurement(
        self,
        evidence_ids: List[str],
        evidence_index: Dict[str, EvidenceItem],
    ) -> List[str]:
        """Link sibling reportable evidence produced from the same measurements.

        LLM grouping can choose an LLM-created EvidenceItem while deterministic
        grouped evidence retains the numeric provenance that FinalProseAuditor
        matches. Keep the claim trace closed over same-target, same-measurement
        evidence so rendered values remain auditable without exposing proxy/debug
        items.
        """

        linked_items = [evidence_index[eid] for eid in evidence_ids if eid in evidence_index]
        linked_measurements = {mid for item in linked_items for mid in item.measurement_ids}
        linked_targets = {str(getattr(item.clinical_target, "value", item.clinical_target)) for item in linked_items}
        if not linked_measurements or not linked_targets:
            return evidence_ids
        closed = list(evidence_ids)
        seen = set(closed)
        for item in evidence_index.values():
            if item.evidence_id in seen:
                continue
            target = str(getattr(item.clinical_target, "value", item.clinical_target))
            evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
            if target not in linked_targets:
                continue
            if evidence_type not in {EvidenceType.DIRECT.value, EvidenceType.DERIVED.value, EvidenceType.METADATA.value}:
                continue
            if self._is_internal_or_proxy_evidence(item):
                continue
            if not linked_measurements.intersection(item.measurement_ids):
                continue
            closed.append(item.evidence_id)
            seen.add(item.evidence_id)
        return closed

    def _requires_caveated_surface(self, evidence_items: List[EvidenceItem]) -> bool:
        targets = {str(getattr(item.clinical_target, "value", item.clinical_target)) for item in evidence_items}
        return bool(targets & {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value})

    def _coverage_checklist(self, evidence_items: List[EvidenceItem]) -> List[Dict[str, Any]]:
        checklist: List[Dict[str, Any]] = []
        for item in evidence_items:
            target = str(getattr(item.clinical_target, "value", item.clinical_target))
            evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
            if evidence_type not in {EvidenceType.DIRECT.value, EvidenceType.DERIVED.value, EvidenceType.METADATA.value}:
                continue
            if target not in self._coverage_targets():
                continue
            surface_safe_values = self._surface_safe_values([item])
            if not surface_safe_values and target in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
                continue
            if self._is_internal_or_proxy_evidence(item):
                checklist.append(
                    {
                        "coverage_id": f"coverage_{item.evidence_id}",
                        "evidence_ids": [item.evidence_id],
                        "clinical_target": target,
                        "recommended_surface_action": ClaimSurfaceAction.BLOCK.value,
                        "allowed_sections": self._safe_allowed_sections([item], list(item.allowed_sections)),
                        "must_render_values": [],
                        "coverage_reason": "Evidence contains internal/proxy-like values; planner must explicitly block or debug_only it.",
                    }
                )
                continue
            checklist.append(
                {
                    "coverage_id": f"coverage_{item.evidence_id}",
                    "evidence_ids": [item.evidence_id],
                    "clinical_target": target,
                    "recommended_surface_action": self._recommended_surface_action(item),
                    "allowed_sections": self._safe_allowed_sections([item], list(item.allowed_sections)),
                    "must_render_values": self._must_render_values(surface_safe_values),
                    "coverage_reason": "GT-relevant evidence should receive an AtomicClaimPlan decision instead of being silently omitted.",
                }
            )
        return checklist

    def _coverage_targets(self) -> set[str]:
        return {
            ClinicalTarget.PDR.value,
            ClinicalTarget.BACKGROUND_AMPLITUDE.value,
            ClinicalTarget.BACKGROUND_SLOWING.value,
            ClinicalTarget.EXCESS_BETA.value,
            ClinicalTarget.STATE.value,
            ClinicalTarget.PROTOCOL.value,
            ClinicalTarget.SEIZURE_EVIDENCE.value,
            ClinicalTarget.LOCALIZATION.value,
            ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value,
        }

    def _recommended_surface_action(self, item: EvidenceItem) -> str:
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
        if evidence_type == EvidenceType.METADATA.value and target in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            return ClaimSurfaceAction.ALLOW.value
        if target in {ClinicalTarget.PDR.value, ClinicalTarget.BACKGROUND_AMPLITUDE.value}:
            return ClaimSurfaceAction.CAVEAT.value
        if target in {ClinicalTarget.BACKGROUND_SLOWING.value, ClinicalTarget.EXCESS_BETA.value}:
            return ClaimSurfaceAction.CAVEAT.value
        if target in {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value}:
            return ClaimSurfaceAction.CAVEAT.value
        if target == ClinicalTarget.SEIZURE_EVIDENCE.value:
            return ClaimSurfaceAction.CAVEAT.value
        return ClaimSurfaceAction.BLOCK.value

    def _coverage_guard_claims(
        self,
        evidence_items: List[EvidenceItem],
        existing_plans: List[AtomicClaimPlan],
        coverage_checklist: List[Dict[str, Any]],
    ) -> List[AtomicClaimPlan]:
        evidence_index = {item.evidence_id: item for item in evidence_items}
        existing_plan_ids = {plan.plan_id for plan in existing_plans}
        claimed_evidence_ids = {evidence_id for plan in existing_plans for evidence_id in plan.evidence_ids}
        claimed_targets = self._claimed_targets(existing_plans, evidence_index)
        out: List[AtomicClaimPlan] = []
        for coverage_item in coverage_checklist:
            coverage_plan_id = self._safe_plan_id(str(coverage_item.get("coverage_id") or ""))
            if coverage_plan_id in existing_plan_ids:
                continue
            evidence_ids = [eid for eid in coverage_item.get("evidence_ids", []) if eid in evidence_index]
            if not evidence_ids:
                continue
            if any(eid in claimed_evidence_ids for eid in evidence_ids):
                continue
            linked_items = [evidence_index[eid] for eid in evidence_ids]
            target = str(coverage_item.get("clinical_target") or "coverage_claim")
            if target in claimed_targets:
                continue
            action = ClaimSurfaceAction(str(coverage_item.get("recommended_surface_action") or ClaimSurfaceAction.BLOCK.value))
            proposed_text = self._safe_text_from_evidence(linked_items)
            if not proposed_text or self.surface_policy.contains_forbidden_surface_text(proposed_text):
                continue
            surface_safe_values = self._surface_safe_values(linked_items)
            numeric_claims = self._numeric_claims(surface_safe_values)
            out.append(
                AtomicClaimPlan(
                    plan_id=self._safe_plan_id(str(coverage_item.get("coverage_id") or f"coverage_{evidence_ids[0]}")),
                    section_type=ReportSectionType.DETAIL,
                    claim_type=target,
                    proposed_text=proposed_text,
                    evidence_ids=evidence_ids,
                    linked_measurement_ids=self._linked_measurement_ids(linked_items),
                    required_evidence=[],
                    missing_evidence=[],
                    surface_action=action,
                    confidence=None,
                    rationale="Coverage guard preserved safe reportable evidence omitted by LLM claim planning.",
                    allowed_sections=[str(section) for section in coverage_item.get("allowed_sections", [])],
                    forbidden_sections=[],
                    clinical_phrase_template_id="llm_atomic_claim_coverage_guard",
                    surface_safe_values=surface_safe_values,
                    must_render_values=self._must_render_values(surface_safe_values),
                    numeric_claims=numeric_claims,
                    debug_payload={
                        "llm_claim_planning": True,
                        "coverage_guard_for_omitted_safe_evidence": True,
                        "coverage_id": coverage_item.get("coverage_id"),
                    },
                )
            )
            claimed_targets.add(target)
        return out

    def _claimed_targets(self, existing_plans: List[AtomicClaimPlan], evidence_index: Dict[str, EvidenceItem]) -> set[str]:
        targets: set[str] = {str(plan.claim_type) for plan in existing_plans}
        for plan in existing_plans:
            for evidence_id in plan.evidence_ids:
                item = evidence_index.get(evidence_id)
                if item is None:
                    continue
                targets.add(str(getattr(item.clinical_target, "value", item.clinical_target)))
        return targets

    def _should_force_safe_text(self, evidence_items: List[EvidenceItem], proposed_text: str) -> bool:
        targets = {str(getattr(item.clinical_target, "value", item.clinical_target)) for item in evidence_items}
        lowered = proposed_text.lower()
        if targets & {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            return True
        if ClinicalTarget.PDR.value in targets:
            return True
        if targets & {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value}:
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
            best_amp = self._first_value(evidence_items, "background_amplitude_best_supported_uv")
            if isinstance(best_amp, (int, float)) and float(best_amp) > 0.0:
                return f"A provenance-linked best-supported background amplitude candidate is approximately {float(best_amp):.1f} uV."
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
            status_text = self._status_text_from_evidence(evidence_items)
            if status_text:
                return status_text
            return "Structured protocol/context status is available but remains non-specific."
        if ClinicalTarget.SEIZURE_EVIDENCE.value in targets:
            seizure_status = self._first_value(evidence_items, "electrographic_seizure_pattern_status")
            if seizure_status == "not_observed":
                return "The seizure-pattern screen did not identify a sustained evolving electrographic seizure pattern."
            if seizure_status == "present":
                return "The seizure-pattern screen suggests a sustained evolving electrographic seizure pattern; this requires clinical review."
            return "Seizure-specific structured evidence is available; interpret with the stated evidence limitations."
        if ClinicalTarget.LOCALIZATION.value in targets:
            pattern = self._first_value(evidence_items, "spatial_pattern")
            if isinstance(pattern, str) and pattern and "unknown" not in pattern.lower():
                return f"Spatial evidence suggests {pattern}; this remains a caveated event-field observation."
            field = self._first_value(evidence_items, "field_descriptor")
            if isinstance(field, str) and "field" in field.lower() and "not localizable" not in field.lower():
                return f"Spatial evidence suggests {field}; this remains a caveated event-field observation."
        if ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value in targets:
            descriptor = self._first_value(evidence_items, "morphology_descriptor")
            if descriptor == "generalized_spike_wave_like":
                return "Structured morphology evidence suggests generalized spike-wave-like features; this is not a definitive epileptiform interpretation."
            if descriptor == "spike_wave_like":
                return "Structured morphology evidence suggests spike-wave-like features; this is not a definitive epileptiform interpretation."
            if descriptor == "sharp_wave_like":
                return "Structured morphology evidence suggests sharp-wave-like features; this is not a definitive epileptiform interpretation."
            if descriptor == "sharp_transient_like":
                return "Structured morphology evidence suggests sharp transient-like features; this is not a definitive epileptiform interpretation."
            if descriptor == "nonspecific_transient_like":
                return "Structured morphology evidence suggests nonspecific transient-like features; this is not a definitive epileptiform interpretation."
        return ""

    def _status_text_from_evidence(self, evidence_items: List[EvidenceItem]) -> str:
        parts: list[str] = []
        for item in evidence_items:
            target = str(getattr(item.clinical_target, "value", item.clinical_target))
            value = item.normalized_value if item.normalized_value not in (None, "", {}, []) else item.value
            if target not in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value} or not isinstance(value, dict):
                continue
            for key, raw_value in value.items():
                if not self._is_known_status(raw_value):
                    continue
                label = key.replace("protocol_", "").replace("_status", "").replace("_availability", "").replace("_presence", "").replace("_", " ")
                parts.append(f"{label}: {raw_value}")
        if not parts:
            return ""
        return "Structured state/protocol context: " + "; ".join(parts) + "."

    def _surface_safe_values(self, evidence_items: List[EvidenceItem]) -> List[Dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in evidence_items:
            target = str(getattr(item.clinical_target, "value", item.clinical_target))
            evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
            if evidence_type not in {EvidenceType.DIRECT.value, EvidenceType.DERIVED.value, EvidenceType.METADATA.value}:
                continue
            if self._is_internal_or_proxy_evidence(item):
                continue
            value = item.normalized_value if item.normalized_value not in (None, "", {}, []) else item.value
            safe_value = self._safe_value_for_target(target, value)
            if safe_value in (None, "", {}, []):
                continue
            out.append(
                {
                    "evidence_id": item.evidence_id,
                    "clinical_target": target,
                    "value": safe_value,
                    "unit": item.unit,
                }
            )
        return out

    def _safe_value_for_target(self, target: str, value: Any) -> Any:
        if target == ClinicalTarget.PDR.value and isinstance(value, dict):
            freq = value.get("frequency_hz")
            if isinstance(freq, (int, float)) and 8.0 <= float(freq) <= 13.0:
                return {"frequency_hz": float(freq)}
            return None
        if target == ClinicalTarget.BACKGROUND_AMPLITUDE.value and isinstance(value, dict):
            safe: dict[str, Any] = {}
            best_supported = value.get("background_amplitude_best_supported_uv")
            if isinstance(best_supported, (int, float)) and float(best_supported) > 0.0:
                safe["background_amplitude_best_supported_uv"] = float(best_supported)
            typical = value.get("background_amplitude_typical_uv")
            if isinstance(typical, (int, float)) and float(typical) > 0.0:
                safe["background_amplitude_typical_uv"] = float(typical)
            amp_range = value.get("background_amplitude_range_uv")
            if isinstance(amp_range, dict) and amp_range.get("upper") is not None:
                safe["background_amplitude_range_uv"] = amp_range
            return safe or None
        if target in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value} and isinstance(value, dict):
            safe = {str(k): v for k, v in value.items() if self._is_known_status(v)}
            return safe or None
        if target == ClinicalTarget.SEIZURE_EVIDENCE.value and isinstance(value, dict):
            safe = {
                "electrographic_seizure_pattern_status": value.get("electrographic_seizure_pattern_status"),
            }
            safe = {str(k): v for k, v in safe.items() if self._is_known_status(v)}
            return safe or None
        if target == ClinicalTarget.LOCALIZATION.value and isinstance(value, dict):
            pattern = value.get("spatial_pattern")
            field = value.get("field_descriptor")
            if (
                isinstance(pattern, str)
                and pattern
                and "unknown" not in pattern.lower()
            ) or (
                isinstance(field, str)
                and "field" in field.lower()
                and "not localizable" not in field.lower()
            ):
                safe = {
                    key: value.get(key)
                    for key in ("spatial_pattern", "field_descriptor", "electrode_maxima", "region", "laterality")
                    if value.get(key) not in (None, "", "unknown")
                }
                return safe or None
            return None
        if target == ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value and isinstance(value, dict):
            waveform = value.get("event_waveform_numeric")
            if isinstance(waveform, dict):
                safe_waveform = {
                    key: waveform.get(key)
                    for key in (
                        "amplitude_peak_to_peak_typical_uv",
                        "amplitude_peak_to_peak_range_uv",
                        "dominant_frequency_hz",
                        "candidate_context_only",
                        "not_seizure_evidence",
                    )
                    if waveform.get(key) not in (None, "", {}, [])
                }
                if safe_waveform:
                    return {"event_waveform_numeric": safe_waveform}
            descriptor = value.get("morphology_descriptor")
            if descriptor in {
                "sharp_transient_like",
                "sharp_wave_like",
                "spike_wave_like",
                "generalized_spike_wave_like",
                "nonspecific_transient_like",
            }:
                return {"morphology_descriptor": descriptor}
            return None
        return None

    def _is_known_status(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in {"", "unknown", "not_available", "unavailable"}
        return True

    def _must_render_values(self, surface_safe_values: List[Dict[str, Any]]) -> List[str]:
        requirements: list[str] = []
        for item in surface_safe_values:
            target = str(item.get("clinical_target") or "")
            unit = item.get("unit")
            value = item.get("value")
            if target == ClinicalTarget.PDR.value and isinstance(value, dict):
                freq = value.get("frequency_hz")
                if isinstance(freq, (int, float)):
                    requirements.append(f"pdr_frequency={float(freq):.1f} {unit or 'Hz'}")
            elif target == ClinicalTarget.BACKGROUND_AMPLITUDE.value and isinstance(value, dict):
                best_supported = value.get("background_amplitude_best_supported_uv")
                if isinstance(best_supported, (int, float)):
                    requirements.append(f"background_amplitude_best_supported={float(best_supported):.1f} {unit or 'uV'}")
                    continue
                typical = value.get("background_amplitude_typical_uv")
                if isinstance(typical, (int, float)):
                    requirements.append(f"background_amplitude_typical={float(typical):.1f} {unit or 'uV'}")
                    continue
                amp_range = value.get("background_amplitude_range_uv")
                if isinstance(amp_range, dict) and amp_range.get("upper") is not None:
                    lo = amp_range.get("lower", 0.0)
                    hi = amp_range["upper"]
                    requirements.append(f"background_amplitude_range={float(lo):.1f}-{float(hi):.1f} {unit or 'uV'}")
            elif target in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value} and isinstance(value, dict):
                for key, raw_value in value.items():
                    requirements.append(f"{key}={raw_value}")
            elif target == ClinicalTarget.LOCALIZATION.value and isinstance(value, dict):
                pattern = value.get("spatial_pattern")
                if isinstance(pattern, str) and pattern:
                    requirements.append(f"spatial_pattern={pattern}")
                field = value.get("field_descriptor")
                if isinstance(field, str) and field:
                    requirements.append(f"field_descriptor={field}")
            elif target == ClinicalTarget.SEIZURE_EVIDENCE.value and isinstance(value, dict):
                status = value.get("electrographic_seizure_pattern_status")
                if isinstance(status, str) and status:
                    requirements.append(f"electrographic_seizure_pattern_status={status}")
            elif target == ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value and isinstance(value, dict):
                waveform = value.get("event_waveform_numeric")
                if isinstance(waveform, dict):
                    amp = waveform.get("amplitude_peak_to_peak_typical_uv")
                    freq = waveform.get("dominant_frequency_hz")
                    if isinstance(amp, (int, float)):
                        requirements.append(f"event_waveform_amplitude={float(amp):.1f} uV")
                    if isinstance(freq, (int, float)):
                        requirements.append(f"event_waveform_frequency={float(freq):.1f} Hz")
                descriptor = value.get("morphology_descriptor")
                if isinstance(descriptor, str) and descriptor:
                    requirements.append(f"morphology_descriptor={descriptor}")
        return requirements

    def _numeric_claims(self, surface_safe_values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Structured numeric contract for report synthesis.

        This is intentionally narrower than surface_safe_values. It promotes only
        clinically reportable numeric values and avoids proxy/debug quantities
        such as scores, burdens, ratios, and train durations.
        """

        claims: list[dict[str, Any]] = []
        for item in surface_safe_values:
            target = str(item.get("clinical_target") or "")
            unit = str(item.get("unit") or "")
            evidence_id = str(item.get("evidence_id") or "")
            value = item.get("value")
            if target == ClinicalTarget.PDR.value and isinstance(value, dict):
                freq = value.get("frequency_hz")
                if isinstance(freq, (int, float)) and 8.0 <= float(freq) <= 13.0:
                    claims.append(
                        self._numeric_claim(
                            slot="pdr_frequency",
                            value=float(freq),
                            unit=unit or "Hz",
                            evidence_id=evidence_id,
                            render_required=True,
                        )
                    )
            elif target == ClinicalTarget.BACKGROUND_AMPLITUDE.value and isinstance(value, dict):
                best_supported = value.get("background_amplitude_best_supported_uv")
                if isinstance(best_supported, (int, float)) and float(best_supported) > 0.0:
                    claims.append(
                        self._numeric_claim(
                            slot="background_amplitude_best_supported",
                            value=float(best_supported),
                            unit=unit or "uV",
                            evidence_id=evidence_id,
                            render_required=True,
                        )
                    )
                    continue
                typical = value.get("background_amplitude_typical_uv")
                if isinstance(typical, (int, float)) and float(typical) > 0.0:
                    claims.append(
                        self._numeric_claim(
                            slot="background_amplitude_typical",
                            value=float(typical),
                            unit=unit or "uV",
                            evidence_id=evidence_id,
                            render_required=True,
                        )
                    )
                    continue
                amp_range = value.get("background_amplitude_range_uv")
                if isinstance(amp_range, dict) and isinstance(amp_range.get("upper"), (int, float)):
                    lo = float(amp_range.get("lower", 0.0))
                    hi = float(amp_range["upper"])
                    if 0.0 <= lo <= hi:
                        claims.append(
                            self._numeric_claim(
                                slot="background_amplitude_range",
                                value={"lower": lo, "upper": hi},
                                unit=unit or "uV",
                                evidence_id=evidence_id,
                                render_required=True,
                            )
                        )
            elif target == ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value and isinstance(value, dict):
                waveform = value.get("event_waveform_numeric")
                if not isinstance(waveform, dict):
                    continue
                amp = waveform.get("amplitude_peak_to_peak_typical_uv")
                if isinstance(amp, (int, float)) and float(amp) > 0.0:
                    claims.append(
                        self._numeric_claim(
                            slot="event_waveform_amplitude",
                            value=float(amp),
                            unit="uV",
                            evidence_id=evidence_id,
                            render_required=True,
                        )
                    )
                freq = waveform.get("dominant_frequency_hz")
                if isinstance(freq, (int, float)) and float(freq) > 0.0:
                    claims.append(
                        self._numeric_claim(
                            slot="event_waveform_frequency",
                            value=float(freq),
                            unit="Hz",
                            evidence_id=evidence_id,
                            render_required=True,
                        )
                    )
        return claims

    def _numeric_claim(
        self,
        *,
        slot: str,
        value: Any,
        unit: str,
        evidence_id: str,
        render_required: bool,
    ) -> Dict[str, Any]:
        if isinstance(value, dict):
            render_text = f"{float(value['lower']):.1f}-{float(value['upper']):.1f} {unit}"
        else:
            render_text = f"{float(value):.1f} {unit}"
        return {
            "slot": slot,
            "value": value,
            "unit": unit,
            "evidence_id": evidence_id,
            "render_required": render_required,
            "render_text": render_text,
            "source": "surface_safe_values",
        }

    def _is_internal_or_proxy_evidence(self, item: EvidenceItem) -> bool:
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
        if target in {ClinicalTarget.EVENT_CANDIDATE.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value, ClinicalTarget.LOCALIZATION.value}:
            if evidence_type in {EvidenceType.DERIVED.value, EvidenceType.DIRECT.value} and any(
                safe_key in text
                for safe_key in (
                    "spatial_pattern",
                    "electrode_maxima",
                    "morphology_descriptor",
                    "event_waveform_numeric",
                    "sharp_transient_like",
                    "nonspecific_transient_like",
                )
            ):
                return False
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
