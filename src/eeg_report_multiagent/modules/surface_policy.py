from __future__ import annotations

from typing import Any

from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, SurfaceDecision
from eeg_report_multiagent.schemas.section_contract import SectionRole


class SurfacePolicy:
    """Single report-surface policy for atomic claim decisions.

    Measurements and EvidenceItems must be converted into AtomicClaimPlan
    objects before this boundary. This prevents raw tool output from becoming
    clinical prose without explicit claim planning.
    """

    FORBIDDEN_SURFACE_TERMS = (
        "candidate burden",
        "burden ratio",
        "longest candidate train",
        "laterality index",
        "bifrontal spread tendency",
        "bifrontal ratio",
        "morphology screen",
        "support score",
        "likelihood score",
        "likelihood scores",
        "candidate likelihood",
        "field concentration ratio",
        "missing_slots",
        "weak_evidence",
        "do_not_claim",
        "claim_constraints",
        "evidence review:",
        "raw evidence reviewer",
        "low-frequency boundary peak",
        "low-frequency boundary spectral peak",
        "support/likelihood",
        "ratio of",
        "slowing score",
        "score of",
        "score:",
        "organization score",
        "bandpower",
        "alpha ratio",
        "symmetry score",
        "confidence score",
        "confidence assessment",
        "confidence in this assessment",
        "confidence in the determination",
        "support being marked",
        "analyzed scores",
        "concentration ratios",
    )

    def decide(
        self,
        item: AtomicClaimPlan,
    ) -> SurfaceDecision:
        if not isinstance(item, AtomicClaimPlan):
            raise TypeError("SurfacePolicy.decide() requires an AtomicClaimPlan.")
        return self._decision_from_claim_plan(item)

    def _decision_from_claim_plan(self, plan: AtomicClaimPlan) -> SurfaceDecision:
        return SurfaceDecision(
            surface_action=plan.surface_action,
            allowed_sections=plan.allowed_sections,
            forbidden_sections=plan.forbidden_sections,
            clinical_phrase_template_id=plan.clinical_phrase_template_id,
            rationale=plan.rationale or "Atomic claim plan already contains a surface decision.",
            evidence_ids=plan.evidence_ids or plan.linked_measurement_ids,
            debug_payload=plan.debug_payload,
        )

    def _decision(
        self,
        action: ClaimSurfaceAction,
        template_id: str,
        rationale: str,
        *,
        allowed: list[SectionRole] | None = None,
        forbidden: list[SectionRole] | None = None,
        evidence_ids: list[str] | None = None,
        debug_payload: dict[str, Any] | None = None,
    ) -> SurfaceDecision:
        return SurfaceDecision(
            surface_action=action,
            allowed_sections=[role.value for role in (allowed or [])],
            forbidden_sections=[role.value for role in (forbidden or [])],
            clinical_phrase_template_id=template_id,
            rationale=rationale,
            evidence_ids=evidence_ids or [],
            debug_payload=debug_payload or {},
        )

    def plan_allowed_in_section(self, plan: AtomicClaimPlan, role: SectionRole) -> bool:
        decision = self.decide(plan)
        if decision.surface_action not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
            return False
        if role.value in decision.forbidden_sections:
            return False
        return not decision.allowed_sections or role.value in decision.allowed_sections

    def contains_forbidden_surface_text(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in self.FORBIDDEN_SURFACE_TERMS)

    def safe_fallback_for_role(self, role: SectionRole) -> str:
        if role == SectionRole.SEIZURES:
            return "Seizures: no seizure-specific evidence was produced by the current structured tools."
        if role == SectionRole.EVENTS_SEIZURES:
            return "Events/seizures: no seizure-specific evidence was produced by the current structured tools."
        if role == SectionRole.EPILEPTIFORM:
            return "No surface-allowed epileptiform claim was produced by the current structured evidence."
        if role in {SectionRole.BACKGROUND, SectionRole.SLEEP}:
            return "No surface-allowed background claim was produced by the current structured evidence."
        if role == SectionRole.IMPRESSION:
            return "No surface-allowed impression claim was produced by the current structured evidence."
        return "No surface-allowed structured evidence was available for this section."
