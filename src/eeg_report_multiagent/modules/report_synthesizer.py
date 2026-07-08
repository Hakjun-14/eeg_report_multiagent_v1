from __future__ import annotations

import re
from typing import Any, List, Mapping

from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.report import (
    AtomicClaimPlan,
    ClaimRecord,
    ClaimSurfaceAction,
    ReportSection,
    ReportSectionType,
    SurfaceDecision,
)
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard
from eeg_report_multiagent.modules.clinical_reference_registry import clinical_reference_ids_for_claim
from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.modules.surface_policy import SurfacePolicy


class ReportSynthesizer:
    """Template-based v1 synthesizer. Reads EvidenceBoard only."""

    def __init__(
        self,
        surface_policy: SurfacePolicy | None = None,
    ) -> None:
        self.surface_policy = surface_policy or SurfacePolicy()

    def synthesize(
        self,
        board: EvidenceBoard,
        claim_plan_override: List[AtomicClaimPlan] | None = None,
    ) -> tuple[ReportSection, ReportSection, List[ClaimRecord]]:
        claims: List[ClaimRecord] = []
        claim_plan = claim_plan_override if claim_plan_override is not None else self.build_atomic_claim_plan(board)
        shared_board = board.ensure_shared_evidence_board()
        surface_decisions = self.build_surface_decisions(claim_plan, shared_board)
        detail_lines = self._section_lines_from_plans(claim_plan, SectionRole.DETAIL, surface_decisions)

        for plan in self._surfaceable_plans(claim_plan, surface_decisions):
            claim_id = f"c_{plan.plan_id}"
            claims.append(
                ClaimRecord(
                    claim_id=claim_id,
                    section_type=plan.section_type,
                    text=plan.proposed_text,
                    linked_evidence_ids=plan.evidence_ids,
                )
            )
            if plan.evidence_ids:
                shared_board.link_to_claim(claim_id, plan.evidence_ids)
        if not detail_lines:
            detail_lines = [self.surface_policy.safe_fallback_for_role(SectionRole.DETAIL)]

        impression_lines = self._section_lines_from_plans(claim_plan, SectionRole.IMPRESSION, surface_decisions)
        if not impression_lines:
            impression_lines = [self.surface_policy.safe_fallback_for_role(SectionRole.IMPRESSION)]

        imp_text = " ".join(impression_lines)
        claims.append(
            ClaimRecord(
                claim_id="c_impression_summary",
                section_type=ReportSectionType.IMPRESSION,
                text=imp_text,
                linked_evidence_ids=[eid for plan in self._surfaceable_plans(claim_plan, surface_decisions) for eid in plan.evidence_ids],
            )
        )

        detail_section = ReportSection(
            section_type=ReportSectionType.DETAIL,
            text="\n".join(detail_lines) if detail_lines else "No detail-level evidence available.",
            claim_ids=[c.claim_id for c in claims if c.section_type == ReportSectionType.DETAIL],
        )
        impression_section = ReportSection(
            section_type=ReportSectionType.IMPRESSION,
            text=imp_text,
            claim_ids=[c.claim_id for c in claims if c.section_type == ReportSectionType.IMPRESSION],
        )
        return detail_section, impression_section, claims

    def build_surface_decisions(
        self,
        claim_plan: List[AtomicClaimPlan],
        shared_board: SharedEvidenceBoard | None = None,
    ) -> List[SurfaceDecision]:
        """Build first-class report-surface decisions for atomic claims.

        `AtomicClaimPlan.surface_action` remains as a legacy mirror, but this
        list is the authoritative object used by report synthesis and persisted
        as `surface_decisions.json`.
        """
        decisions: List[SurfaceDecision] = []
        for plan in claim_plan:
            base = self.surface_policy.decide(plan)
            hard_deny = self._hard_deny_reasons(plan, base, shared_board)
            clinical_reference_ids = clinical_reference_ids_for_claim(
                plan.claim_type,
                self._claim_targets(plan, shared_board),
            )
            action = base.surface_action
            decided_by = "surface_policy"
            rationale = base.rationale
            caveat = base.caveat
            debug_payload = dict(base.debug_payload)
            if not hard_deny:
                calibrated = self._calibrated_surface_update(plan, base, shared_board)
                if calibrated is not None:
                    action, rationale, caveat = calibrated
                    decided_by = "surface_policy_calibrated"
                    debug_payload = {
                        **debug_payload,
                        "surface_calibration": {
                            "from_surface_action": base.surface_action.value,
                            "to_surface_action": action.value,
                            "reason": "trace_safe_evidence_value_recovered",
                        },
                    }
            if hard_deny and action in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                action = ClaimSurfaceAction.BLOCK
                decided_by = "surface_policy_hard_deny"
            decisions.append(
                base.model_copy(
                    update={
                        "decision_id": f"sd_{plan.plan_id}",
                        "claim_id": plan.plan_id,
                        "surface_action": action,
                        "hard_deny_reasons": hard_deny,
                        "decided_by": decided_by,
                        "rationale": rationale,
                        "caveat": caveat,
                        "clinical_reference_ids": clinical_reference_ids,
                        "debug_payload": debug_payload,
                    }
                )
            )
        return decisions

    def _calibrated_surface_update(
        self,
        plan: AtomicClaimPlan,
        base: SurfaceDecision,
        shared_board: SharedEvidenceBoard | None,
    ) -> tuple[ClaimSurfaceAction, str, str | None] | None:
        """Recover safe claim-plan outputs without bypassing hard-deny rules."""

        if base.surface_action in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
            return None
        if not self._has_trace_safe_surface_value(plan, shared_board):
            return None
        targets = self._claim_targets(plan, shared_board)
        if targets & {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            if not self._has_non_unknown_status_value(plan, shared_board):
                return None
            return (
                ClaimSurfaceAction.ALLOW,
                f"{base.rationale} SurfaceDecision calibrated metadata/status evidence with trace-safe values.",
                None,
            )
        if ClinicalTarget.PDR.value in targets and self._has_plausible_pdr_value(plan, shared_board):
            return (
                ClaimSurfaceAction.CAVEAT,
                f"{base.rationale} SurfaceDecision calibrated plausible PDR evidence to caveated prose.",
                "Reportable only as caveated structured evidence; clinical reader confirmation may still be required.",
            )
        if ClinicalTarget.BACKGROUND_AMPLITUDE.value in targets and self._has_reportable_amplitude_value(plan, shared_board):
            return (
                ClaimSurfaceAction.CAVEAT,
                f"{base.rationale} SurfaceDecision calibrated reportable background amplitude evidence to caveated prose.",
                "Reportable only as caveated structured evidence; clinical reader confirmation may still be required.",
            )
        if targets & {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value}:
            if not self._has_safe_spatiomorphology_value(plan, shared_board):
                return None
            return (
                ClaimSurfaceAction.CAVEAT,
                f"{base.rationale} SurfaceDecision calibrated trace-safe spatiomorphology evidence to caveated prose.",
                "Reportable only as caveated structured evidence; not a definitive epileptiform or localization interpretation.",
            )
        return None

    def build_atomic_claim_plan(self, board: EvidenceBoard) -> List[AtomicClaimPlan]:
        """Plan report-surface claims from grouped EvidenceItems.

        Tools, parsers, encoders, and LLM evidence review produce measurements
        and EvidenceItems.
        """
        shared_board = board.ensure_shared_evidence_board()
        return self._build_grouped_atomic_claim_plan(shared_board)

    def _build_grouped_atomic_claim_plan(self, shared_board: SharedEvidenceBoard) -> List[AtomicClaimPlan]:
        """Plan claims from clinically grouped EvidenceItems.

        Evidence grouping happens before claim planning, so claims are no
        longer generated one-per-measurement/tool output.
        """

        plans: List[AtomicClaimPlan] = []
        for item in shared_board.evidence_items:
            action, template_id, allowed_sections, required, missing = self._grouped_claim_policy(item)
            proposed_text = self._grouped_claim_text(item, template_id, action)
            plans.append(
                AtomicClaimPlan(
                    plan_id=f"p_{item.evidence_id}",
                    section_type=ReportSectionType.DETAIL,
                    claim_type=str(getattr(item.clinical_target, "value", item.clinical_target)),
                    proposed_text=proposed_text,
                    evidence_ids=[item.evidence_id],
                    linked_measurement_ids=list(item.measurement_ids),
                    required_evidence=required,
                    missing_evidence=missing,
                    surface_action=action,
                    confidence=None,
                    rationale=self._grouped_claim_rationale(item, action),
                    allowed_sections=allowed_sections,
                    forbidden_sections=[],
                    clinical_phrase_template_id=template_id,
                    debug_payload={
                        "grouped_evidence_path": True,
                        "clinical_target": str(getattr(item.clinical_target, "value", item.clinical_target)),
                        "evidence_type": str(getattr(item.evidence_type, "value", item.evidence_type)),
                    },
                )
            )
        return plans

    def _grouped_claim_policy(
        self,
        item: EvidenceItem,
    ) -> tuple[ClaimSurfaceAction, str, List[str], List[str], List[str]]:
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        if item.evidence_type in {EvidenceType.DEBUG, EvidenceType.LLM_ASSISTED}:
            return ClaimSurfaceAction.DEBUG_ONLY, "debug_or_llm_evidence_group", [], ["typed_evidence"], ["surface_allowed_evidence"]
        if item.evidence_type == EvidenceType.PROXY:
            return ClaimSurfaceAction.BLOCK, "proxy_evidence_group", [], ["non_proxy_evidence"], ["validated_clinical_support"]
        if target == ClinicalTarget.PDR.value:
            value = item.value if isinstance(item.value, dict) else {}
            freq = value.get("frequency_hz")
            supported = str(value.get("pdr_supported", "")).lower() == "true"
            if isinstance(freq, (int, float)) and 8.0 <= float(freq) <= 13.0 and supported:
                return (
                    ClaimSurfaceAction.CAVEAT,
                    "grouped_pdr_candidate",
                    [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.SLEEP.value],
                    ["posterior_alpha_frequency", "posterior_topography", "state_reactivity_when_available"],
                    ["reactivity_or_eye_opening_attenuation"],
                )
            return ClaimSurfaceAction.BLOCK, "grouped_pdr_not_supported", [], ["posterior_alpha_frequency"], ["valid_posterior_alpha_support"]
        if target == ClinicalTarget.BACKGROUND_AMPLITUDE.value:
            return (
                ClaimSurfaceAction.CAVEAT,
                "grouped_background_amplitude",
                [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value],
                ["background_amplitude_measurement"],
                [],
            )
        if target == ClinicalTarget.BACKGROUND_SLOWING.value:
            return (
                ClaimSurfaceAction.CAVEAT,
                "grouped_background_slowing",
                [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.IMPRESSION.value],
                ["background_slowing_measurement"],
                ["clinical_reader_confirmation"],
            )
        if target == ClinicalTarget.EXCESS_BETA.value:
            return (
                ClaimSurfaceAction.CAVEAT,
                "grouped_excess_beta",
                [SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.IMPRESSION.value],
                ["beta_activity_measurement"],
                ["clinical_reader_confirmation"],
            )
        if target in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            return (
                ClaimSurfaceAction.ALLOW,
                "grouped_status_context",
                [SectionRole.DETAIL.value, SectionRole.BACKGROUND.value, SectionRole.SLEEP.value],
                ["structured_status_or_metadata"],
                [],
            )
        return ClaimSurfaceAction.BLOCK, "unmapped_grouped_evidence", [], ["mapped_clinical_target"], ["claim_template"]

    def _grouped_claim_text(
        self,
        item: EvidenceItem,
        template_id: str,
        action: ClaimSurfaceAction,
    ) -> str:
        if action in {ClaimSurfaceAction.BLOCK, ClaimSurfaceAction.DEBUG_ONLY}:
            return self.surface_policy.safe_fallback_for_role(SectionRole.OTHER)
        if template_id == "grouped_pdr_candidate":
            return (
                "A posterior alpha rhythm candidate is supported by structured evidence; "
                "state and reactivity confirmation remain incomplete."
            )
        if template_id == "grouped_background_amplitude":
            return "A provenance-linked background amplitude range is available."
        if template_id == "grouped_background_slowing":
            return "Structured evidence suggests background slowing; this remains an assistive observation pending EEG review."
        if template_id == "grouped_excess_beta":
            return "Structured evidence suggests increased beta activity; this remains an assistive observation pending EEG review."
        if template_id == "grouped_status_context":
            return self._grouped_status_text(item)
        return self.surface_policy.safe_fallback_for_role(SectionRole.OTHER)

    def _grouped_status_text(self, item: EvidenceItem) -> str:
        if not isinstance(item.value, dict):
            return "Structured protocol/context status is available."
        parts: List[str] = []
        for name, value in item.value.items():
            if value in {None, "unknown"}:
                continue
            label = name.replace("protocol_", "").replace("_status", "").replace("_availability", "").replace("_presence", "").replace("_", " ")
            parts.append(f"{label}: {value}")
        if not parts:
            return "Structured protocol/context status is available but remains non-specific."
        return "Protocol/context: " + "; ".join(parts) + "."

    def _grouped_claim_rationale(self, item: EvidenceItem, action: ClaimSurfaceAction) -> str:
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        if action == ClaimSurfaceAction.DEBUG_ONLY:
            return f"Grouped {target} evidence is debug-only and cannot surface."
        if action == ClaimSurfaceAction.BLOCK:
            return f"Grouped {target} evidence lacks enough support for report-surface prose."
        if action == ClaimSurfaceAction.CAVEAT:
            return f"Grouped {target} evidence may surface only with caveated wording."
        return f"Grouped {target} metadata/status evidence may surface."

    def synthesize_celm_sections(self, board: EvidenceBoard, target_section_names: List[str]) -> dict[str, str]:
        """Generate section-specific text for CELM-compatible evaluation outputs.

        This is intentionally downstream of the EvidenceBoard. It does not inspect raw EEG or GT text.
        """
        router = SectionRouter()
        claim_plan = self.build_atomic_claim_plan(board)
        surface_decisions = self.build_surface_decisions(claim_plan, board.ensure_shared_evidence_board())
        section_texts: dict[str, str] = {}
        for section_name in target_section_names:
            role = router.role_for_section(section_name)
            section_texts[section_name] = self._section_text_from_plans(claim_plan, role, surface_decisions)
        return section_texts

    def _surfaceable_plans(
        self,
        claim_plan: List[AtomicClaimPlan],
        surface_decisions: List[SurfaceDecision] | None = None,
    ) -> List[AtomicClaimPlan]:
        decision_by_claim = self._decision_by_claim_id(surface_decisions)
        return [
            plan
            for plan in claim_plan
            if self._decision_for_plan(plan, decision_by_claim).surface_action in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}
            and not self.surface_policy.contains_forbidden_surface_text(plan.proposed_text)
        ]

    def _section_lines_from_plans(
        self,
        claim_plan: List[AtomicClaimPlan],
        role: SectionRole,
        surface_decisions: List[SurfaceDecision] | None = None,
    ) -> List[str]:
        lines: List[str] = []
        decision_by_claim = self._decision_by_claim_id(surface_decisions)
        for plan in claim_plan:
            decision = self._decision_for_plan(plan, decision_by_claim)
            if not self._decision_allows_section(decision, role):
                continue
            if self.surface_policy.contains_forbidden_surface_text(plan.proposed_text):
                continue
            lines.append(plan.proposed_text)
        return lines

    def _section_text_from_plans(
        self,
        claim_plan: List[AtomicClaimPlan],
        role: SectionRole,
        surface_decisions: List[SurfaceDecision] | None = None,
    ) -> str:
        lines = self._section_lines_from_plans(claim_plan, role, surface_decisions)
        if lines:
            return " ".join(lines)
        return self.surface_policy.safe_fallback_for_role(role)

    def _decision_by_claim_id(self, surface_decisions: List[SurfaceDecision] | None) -> dict[str, SurfaceDecision]:
        return {decision.claim_id: decision for decision in surface_decisions or [] if decision.claim_id}

    def _decision_for_plan(
        self,
        plan: AtomicClaimPlan,
        decision_by_claim: Mapping[str, SurfaceDecision],
    ) -> SurfaceDecision:
        return decision_by_claim.get(plan.plan_id) or self.surface_policy.decide(plan)

    def _decision_allows_section(self, decision: SurfaceDecision, role: SectionRole) -> bool:
        if decision.surface_action not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
            return False
        if role.value in decision.forbidden_sections:
            return False
        return not decision.allowed_sections or role.value in decision.allowed_sections

    def _hard_deny_reasons(
        self,
        plan: AtomicClaimPlan,
        decision: SurfaceDecision,
        shared_board: SharedEvidenceBoard | None,
    ) -> List[str]:
        reasons: List[str] = []
        text = plan.proposed_text.lower()
        if self.surface_policy.contains_forbidden_surface_text(plan.proposed_text):
            reasons.append("forbidden_debug_or_proxy_surface_text")
        if self._looks_like_boundary_pdr(plan, text):
            reasons.append("boundary_or_global_low_frequency_pdr_forbidden")
        linked_evidence = self._linked_evidence_items(plan, shared_board)
        if any(item.evidence_type == EvidenceType.DEBUG for item in linked_evidence):
            reasons.append("debug_evidence_cannot_surface")
        if any(self._linked_evidence_is_internal_or_proxy(item) for item in linked_evidence):
            reasons.append("linked_internal_or_proxy_evidence_cannot_surface")
        if any(self._linked_evidence_has_boundary_global_frequency(item) for item in linked_evidence):
            reasons.append("boundary_or_global_frequency_not_reportable")
        if self._looks_like_seizure_claim(plan, text) and not self._has_seizure_specific_evidence(linked_evidence):
            reasons.append("seizure_claim_without_seizure_specific_evidence")
        return sorted(set(reasons))

    def _linked_evidence_items(self, plan: AtomicClaimPlan, shared_board: SharedEvidenceBoard | None) -> list:
        if shared_board is None:
            return []
        out = []
        for evidence_id in plan.evidence_ids:
            try:
                out.append(shared_board.get_evidence(evidence_id))
            except KeyError:
                continue
        return out

    def _claim_targets(self, plan: AtomicClaimPlan, shared_board: SharedEvidenceBoard | None) -> set[str]:
        targets = {str(plan.claim_type)}
        for item in self._linked_evidence_items(plan, shared_board):
            targets.add(str(getattr(item.clinical_target, "value", item.clinical_target)))
        for item in plan.surface_safe_values:
            target = item.get("clinical_target") if isinstance(item, dict) else None
            if target:
                targets.add(str(target))
        return targets

    def _has_trace_safe_surface_value(self, plan: AtomicClaimPlan, shared_board: SharedEvidenceBoard | None) -> bool:
        if plan.surface_safe_values or plan.must_render_values:
            return True
        return any(self._evidence_item_has_reportable_value(item) for item in self._linked_evidence_items(plan, shared_board))

    def _evidence_item_has_reportable_value(self, item: EvidenceItem) -> bool:
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
        if evidence_type not in {"direct", "derived", "metadata"}:
            return False
        if target == ClinicalTarget.PDR.value:
            return self._value_has_plausible_frequency(item.value)
        if target == ClinicalTarget.BACKGROUND_AMPLITUDE.value:
            return self._value_has_amplitude_uv(item.value, item.unit)
        if target in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            return self._value_has_non_unknown_status(item.value)
        if target in {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value}:
            return self._value_has_safe_spatiomorphology_descriptor(item.value)
        return False

    def _has_plausible_pdr_value(self, plan: AtomicClaimPlan, shared_board: SharedEvidenceBoard | None) -> bool:
        for value in self._candidate_values(plan, shared_board, ClinicalTarget.PDR.value):
            if self._value_has_plausible_frequency(value):
                return True
        return False

    def _has_reportable_amplitude_value(self, plan: AtomicClaimPlan, shared_board: SharedEvidenceBoard | None) -> bool:
        for item in plan.surface_safe_values:
            if isinstance(item, dict) and str(item.get("clinical_target")) == ClinicalTarget.BACKGROUND_AMPLITUDE.value:
                if self._value_has_amplitude_uv(item.get("value"), item.get("unit")):
                    return True
        for item in self._linked_evidence_items(plan, shared_board):
            target = str(getattr(item.clinical_target, "value", item.clinical_target))
            if target == ClinicalTarget.BACKGROUND_AMPLITUDE.value and self._value_has_amplitude_uv(item.value, item.unit):
                return True
        return False

    def _has_non_unknown_status_value(self, plan: AtomicClaimPlan, shared_board: SharedEvidenceBoard | None) -> bool:
        for target in {ClinicalTarget.STATE.value, ClinicalTarget.PROTOCOL.value}:
            for value in self._candidate_values(plan, shared_board, target):
                if self._value_has_non_unknown_status(value):
                    return True
        return False

    def _has_safe_spatiomorphology_value(self, plan: AtomicClaimPlan, shared_board: SharedEvidenceBoard | None) -> bool:
        for target in {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value}:
            for value in self._candidate_values(plan, shared_board, target):
                if self._value_has_safe_spatiomorphology_descriptor(value):
                    return True
        return False

    def _candidate_values(
        self,
        plan: AtomicClaimPlan,
        shared_board: SharedEvidenceBoard | None,
        target: str,
    ) -> list[Any]:
        values: list[Any] = []
        for item in plan.surface_safe_values:
            if isinstance(item, dict) and str(item.get("clinical_target")) == target:
                values.append(item.get("value"))
        for item in self._linked_evidence_items(plan, shared_board):
            item_target = str(getattr(item.clinical_target, "value", item.clinical_target))
            if item_target == target:
                values.append(item.value)
        return values

    def _value_has_plausible_frequency(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        freq = value.get("frequency_hz") or value.get("pdr_frequency_hz")
        return isinstance(freq, (int, float)) and 8.0 <= float(freq) <= 13.0

    def _value_has_amplitude_uv(self, value: Any, unit: Any) -> bool:
        if str(unit or "").lower() not in {"uv", "µv"}:
            return False
        if isinstance(value, (int, float)):
            return True
        if not isinstance(value, dict):
            return False
        return any(
            isinstance(value.get(key), (int, float, dict))
            for key in (
                "background_amplitude_uv",
                "background_amplitude_best_supported_uv",
                "background_amplitude_typical_uv",
                "background_amplitude_range_uv",
            )
        )

    def _value_has_non_unknown_status(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        for status in value.values():
            if status is None:
                continue
            if str(status).strip().lower() not in {"", "unknown", "not_available", "unavailable", "none"}:
                return True
        return False

    def _value_has_safe_spatiomorphology_descriptor(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        pattern = value.get("spatial_pattern")
        if isinstance(pattern, str) and pattern.strip() and "unknown" not in pattern.lower():
            return True
        field_descriptor = value.get("field_descriptor")
        if isinstance(field_descriptor, str) and "field" in field_descriptor.lower() and "not localizable" not in field_descriptor.lower():
            return True
        descriptor = value.get("morphology_descriptor")
        return descriptor in {
            "sharp_transient_like",
            "sharp_wave_like",
            "spike_wave_like",
            "generalized_spike_wave_like",
            "nonspecific_transient_like",
        }

    def _looks_like_boundary_pdr(self, plan: AtomicClaimPlan, text: str) -> bool:
        if "pdr" not in plan.claim_type and "posterior dominant" not in text and "posterior alpha" not in text:
            return False
        return bool(re.search(r"\b0(?:\.0)?\s*[-–]?\s*\.?5\s*hz\b|\b0\.5\s*hz\b", text))

    def _looks_like_seizure_claim(self, plan: AtomicClaimPlan, text: str) -> bool:
        if "seizure" not in plan.claim_type and "seizure" not in text:
            return False
        negative_safe = "no seizure-specific evidence" in text or "no surface-allowed" in text
        return not negative_safe

    def _has_seizure_specific_evidence(self, evidence_items: list) -> bool:
        return any(str(getattr(item.clinical_target, "value", item.clinical_target)) == "seizure_evidence" for item in evidence_items)

    def _linked_evidence_is_internal_or_proxy(self, item: EvidenceItem) -> bool:
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
        safe_value_targets = {
            ClinicalTarget.PDR.value,
            ClinicalTarget.BACKGROUND_AMPLITUDE.value,
            ClinicalTarget.STATE.value,
            ClinicalTarget.PROTOCOL.value,
        }
        if (
            target in {ClinicalTarget.LOCALIZATION.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value}
            and evidence_type in {"direct", "derived"}
            and self._value_has_safe_spatiomorphology_descriptor(item.value)
        ):
            return False
        text_parts = [
            target,
            evidence_type,
            str(item.unit or ""),
            str(item.value),
            str(item.normalized_value),
        ]
        # For direct/status evidence, linked measurement ids may include context
        # features used upstream; do not turn those context ids into surface
        # hard-deny reasons unless the surfaced value itself is internal.
        if target not in safe_value_targets or evidence_type not in {"direct", "derived", "metadata"}:
            text_parts.append(" ".join(item.measurement_ids))
        text = " ".join(
            text_parts
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

    def _linked_evidence_has_boundary_global_frequency(self, item: EvidenceItem) -> bool:
        text = " ".join([str(item.value), str(item.normalized_value), " ".join(item.measurement_ids)]).lower()
        if "background_dominant_frequency" not in text:
            return False
        return bool(re.search(r"\b0(?:\.0)?\s*[-–]?\s*\.?\s*5\b|\b0\.5\b", text))

    def _has_review_constraint(self, board: EvidenceBoard, needle: str) -> bool:
        needle = needle.lower()
        for deliberation in board.deliberations:
            for item in deliberation.do_not_claim:
                if needle in item.text.lower():
                    return True
            for item in deliberation.claim_constraints:
                if needle in item.constraint.lower():
                    return True
        return False

    def _background_section_text(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
        review_notes: dict[str, List[str]],
    ) -> str:
        return self._section_text_from_plans(self.build_atomic_claim_plan(board), SectionRole.BACKGROUND)

    def _event_section_text(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
        review_notes: dict[str, List[str]],
        section_role: SectionRole | str,
    ) -> str:
        if not isinstance(section_role, SectionRole):
            section_role = SectionRouter().role_for_section(section_role)
        if section_role in {SectionRole.SEIZURES, SectionRole.EVENTS_SEIZURES, SectionRole.EPILEPTIFORM}:
            return self._section_text_from_plans(self.build_atomic_claim_plan(board), section_role)
        return self._section_text_from_plans(self.build_atomic_claim_plan(board), SectionRole.DETAIL)

    def _detail_section_text(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
        review_notes: dict[str, List[str]],
    ) -> str:
        return self._section_text_from_plans(self.build_atomic_claim_plan(board), SectionRole.DETAIL)

    def _clinical_impression_text(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
    ) -> str:
        return self._section_text_from_plans(self.build_atomic_claim_plan(board), SectionRole.IMPRESSION)

    def _build_impression(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
    ) -> List[str]:
        text = self._section_text_from_plans(self.build_atomic_claim_plan(board), SectionRole.IMPRESSION)
        return [text]

    def _review_impression_constraints(self, board: EvidenceBoard) -> List[str]:
        out: List[str] = []
        constraints = []
        do_not_claim = []
        missing_slots = []
        for deliberation in board.deliberations:
            constraints.extend(deliberation.claim_constraints)
            do_not_claim.extend(deliberation.do_not_claim)
            missing_slots.extend(deliberation.missing_slots)

        if any("epileptiform" in item.text.lower() for item in do_not_claim):
            out.append(
                "Event-related clinical claims require morphology-specific evidence before report-surface use."
            )
        if any("focal" in c.constraint.lower() or "lateralized" in c.constraint.lower() for c in constraints):
            out.append(
                "No focal or lateralized conclusion is made unless channel/region provenance supports it."
            )
        if missing_slots:
            out.append("Structured evidence gaps are retained in audit artifacts.")
        return out
