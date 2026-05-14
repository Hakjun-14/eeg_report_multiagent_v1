from __future__ import annotations

from typing import Any

from eeg_report_multiagent.schemas.finding import FindingObject
from eeg_report_multiagent.schemas.measurement import MeasurementValue, StatusSemantic
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, SurfaceDecision
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType


class SurfacePolicy:
    """Single report-surface policy for evidence-to-prose decisions.

    Signal tools may create typed measurements and modules may create findings,
    but this policy is the boundary that decides whether an item may become
    clinical prose. Proxy/debug items can still participate in gating and audit
    artifacts, but not in report text.
    """

    DEBUG_ONLY_FINDING_TYPES = {
        "background_frequency",
        "background_pdr_support",
        "background_pdr_topography",
        "background_pdr_symmetry",
        "background_ap_organization",
        "epileptiform_event_candidate_burden",
        "event_train_duration",
        "event_laterality",
        "event_focality_bifrontal_spread",
        "event_clinical_localization",
        "event_localization_support",
        "event_peak_localization",
        "event_peak_field_support",
        "event_peak_laterality",
        "event_morphology_class",
        "event_morphology_support",
        "event_field_concentration",
        "epileptiform_candidate_likelihood",
        "electrographic_seizure_likelihood",
        "event_measurement",
        "background_measurement",
    }

    DEBUG_ONLY_MEASUREMENT_NAMES = {
        "event_candidate_score_distribution",
        "event_candidate_burden_ratio",
        "event_train_duration_upper_sec",
        "event_train_duration_distribution_sec",
        "event_laterality_index",
        "event_clinical_localization_label",
        "event_localization_concentration_ratio",
        "event_peak_localization_label",
        "event_peak_field_concentration_ratio",
        "event_peak_laterality_index",
        "event_bifrontal_ratio",
        "event_morphology_proxy_class",
        "event_morphology_proxy_score_distribution",
        "event_morphology_support_score",
        "event_field_concentration_ratio",
        "epileptiform_candidate_likelihood_score",
        "electrographic_seizure_likelihood_score",
        "pdr_candidate_confidence_score",
        "pdr_posterior_anterior_alpha_ratio",
        "pdr_symmetry_score",
        "background_ap_organization_score",
        "slowing_score",
        "beta_excess_score",
    }

    FORBIDDEN_SURFACE_TERMS = (
        "candidate burden",
        "longest candidate train",
        "laterality index",
        "bifrontal spread tendency",
        "morphology screen",
        "support score",
        "likelihood score",
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
    )

    def decide(
        self,
        item: FindingObject | MeasurementValue | AtomicClaimPlan,
        *,
        measurement: MeasurementValue | None = None,
        missing_evidence: list[str] | None = None,
        evidence_items: list[EvidenceItem] | None = None,
    ) -> SurfaceDecision:
        if isinstance(item, AtomicClaimPlan):
            return self._decision_from_claim_plan(item)
        if isinstance(item, MeasurementValue):
            return self._decision_from_measurement(item)
        evidence_gate = self._decision_from_evidence_items(evidence_items or [])
        if evidence_gate is not None:
            return evidence_gate
        return self._decision_from_finding(item, measurement=measurement, missing_evidence=missing_evidence or [])

    def _decision_from_claim_plan(self, plan: AtomicClaimPlan) -> SurfaceDecision:
        return SurfaceDecision(
            surface_action=plan.surface_action,
            allowed_sections=plan.allowed_sections,
            forbidden_sections=plan.forbidden_sections,
            clinical_phrase_template_id=plan.clinical_phrase_template_id,
            rationale=plan.rationale or "Atomic claim plan already contains a surface decision.",
            evidence_ids=plan.evidence_ids or plan.linked_finding_ids + plan.linked_measurement_ids,
            debug_payload=plan.debug_payload,
        )

    def _decision_from_evidence_items(self, evidence_items: list[EvidenceItem]) -> SurfaceDecision | None:
        if not evidence_items:
            return None
        evidence_ids = [item.evidence_id for item in evidence_items]
        if any(item.evidence_type == EvidenceType.DEBUG for item in evidence_items):
            return self._decision(
                ClaimSurfaceAction.DEBUG_ONLY,
                "debug_evidence_item",
                "Debug evidence items cannot directly surface as clinical prose.",
                evidence_ids=evidence_ids,
                debug_payload={"evidence_ids": evidence_ids},
            )
        reportabilities = {item.reportability for item in evidence_items}
        if reportabilities and reportabilities.issubset({ClaimSurfaceAction.DEBUG_ONLY}):
            return self._decision(
                ClaimSurfaceAction.DEBUG_ONLY,
                "debug_only_evidence_item",
                "Debug-only evidence items may support audit/gating but not clinical prose.",
                evidence_ids=evidence_ids,
                debug_payload={"evidence_ids": evidence_ids},
            )
        if reportabilities and reportabilities.issubset({ClaimSurfaceAction.BLOCK, ClaimSurfaceAction.DEBUG_ONLY}):
            return self._decision(
                ClaimSurfaceAction.BLOCK,
                "blocked_evidence_item",
                "Linked evidence items are blocked or debug-only.",
                evidence_ids=evidence_ids,
                debug_payload={"evidence_ids": evidence_ids},
            )
        if any(item.clinical_target == ClinicalTarget.SEIZURE_EVIDENCE for item in evidence_items):
            seizure_items = [item for item in evidence_items if item.clinical_target == ClinicalTarget.SEIZURE_EVIDENCE]
            if not any(item.reportability in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT} for item in seizure_items):
                return self._decision(
                    ClaimSurfaceAction.BLOCK,
                    "no_reportable_seizure_evidence",
                    "A seizure claim requires reportable seizure-specific evidence.",
                    evidence_ids=evidence_ids,
                    debug_payload={"evidence_ids": evidence_ids},
                )
        return None

    def _decision_from_measurement(self, measurement: MeasurementValue) -> SurfaceDecision:
        if measurement.measurement_name in self.DEBUG_ONLY_MEASUREMENT_NAMES or measurement.measurement_name.startswith("relative_bandpower_"):
            return self._decision(
                ClaimSurfaceAction.DEBUG_ONLY,
                "measurement_debug_only",
                "Measurement is a proxy/debug value and must not directly surface as report prose.",
                evidence_ids=[measurement.measurement_id],
                debug_payload={"measurement_name": measurement.measurement_name},
            )
        return self._decision(
            ClaimSurfaceAction.BLOCK,
            "measurement_needs_finding",
            "Measurements require an allowed finding/claim plan before report-surface use.",
            evidence_ids=[measurement.measurement_id],
            debug_payload={"measurement_name": measurement.measurement_name},
        )

    def _decision_from_finding(
        self,
        finding: FindingObject,
        *,
        measurement: MeasurementValue | None,
        missing_evidence: list[str],
    ) -> SurfaceDecision:
        evidence_ids = [finding.finding_id] + list(finding.measurement_ids)
        debug_payload: dict[str, Any] = {
            "finding_type": finding.finding_type,
            "missing_evidence": missing_evidence,
        }
        if measurement is not None:
            debug_payload["measurement_name"] = measurement.measurement_name

        if finding.assertion == StatusSemantic.UNKNOWN:
            return self._decision(
                ClaimSurfaceAction.BLOCK,
                "unknown_finding",
                "Unknown findings are retained in evidence artifacts, not clinical prose.",
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        if finding.finding_type in self.DEBUG_ONLY_FINDING_TYPES:
            return self._decision(
                ClaimSurfaceAction.DEBUG_ONLY,
                "proxy_or_debug_finding",
                "Proxy/debug findings may affect gating and provenance but cannot directly surface.",
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        if finding.finding_type == "background_pdr_frequency":
            if finding.assertion != StatusSemantic.PRESENT:
                return self._decision(
                    ClaimSurfaceAction.BLOCK,
                    "pdr_not_supported",
                    "PDR candidate is not supported strongly enough for surface text.",
                    evidence_ids=evidence_ids,
                    debug_payload=debug_payload,
                )
            return self._decision(
                ClaimSurfaceAction.CAVEAT,
                "background_posterior_alpha_candidate",
                "Posterior alpha evidence may surface only as a candidate because state/reactivity are incomplete.",
                allowed=[SectionRole.BACKGROUND, SectionRole.DETAIL, SectionRole.SLEEP],
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        if finding.finding_type == "background_amplitude_range":
            return self._decision(
                ClaimSurfaceAction.CAVEAT,
                "background_amplitude_range",
                "Amplitude may surface as a provenance-linked measurement with scale assumptions.",
                allowed=[SectionRole.BACKGROUND, SectionRole.DETAIL],
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        if finding.finding_type == "background_slowing":
            if finding.assertion != StatusSemantic.PRESENT:
                return self._decision(
                    ClaimSurfaceAction.BLOCK,
                    "background_slowing_not_surfaceable",
                    "Absent slowing-screen findings are retained in evidence artifacts.",
                    evidence_ids=evidence_ids,
                    debug_payload=debug_payload,
                )
            return self._decision(
                ClaimSurfaceAction.CAVEAT,
                "background_slowing_assistive",
                "Slowing evidence is a local screen result and must be caveated.",
                allowed=[SectionRole.BACKGROUND, SectionRole.DETAIL, SectionRole.IMPRESSION],
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        if finding.finding_type == "excess_beta":
            if finding.assertion != StatusSemantic.PRESENT:
                return self._decision(
                    ClaimSurfaceAction.BLOCK,
                    "background_beta_not_surfaceable",
                    "Absent beta-screen findings are retained in evidence artifacts.",
                    evidence_ids=evidence_ids,
                    debug_payload=debug_payload,
                )
            return self._decision(
                ClaimSurfaceAction.CAVEAT,
                "background_beta_assistive",
                "Beta evidence is a local screen result and must be caveated.",
                allowed=[SectionRole.BACKGROUND, SectionRole.DETAIL, SectionRole.IMPRESSION],
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        if finding.finding_type in {"background_reactivity", "sleep_architecture"}:
            return self._decision(
                ClaimSurfaceAction.ALLOW,
                finding.finding_type,
                "Status finding may surface when the status is explicitly structured and non-unknown.",
                allowed=[SectionRole.BACKGROUND, SectionRole.DETAIL, SectionRole.SLEEP],
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        if finding.finding_type.startswith("protocol_"):
            if finding.assertion == StatusSemantic.UNKNOWN:
                return self._decision(
                    ClaimSurfaceAction.BLOCK,
                    "unknown_protocol_status",
                    "Unknown protocol/status findings should not surface.",
                    evidence_ids=evidence_ids,
                    debug_payload=debug_payload,
                )
            return self._decision(
                ClaimSurfaceAction.ALLOW,
                "protocol_status",
                "Structured metadata/status findings may surface.",
                allowed=[SectionRole.DETAIL, SectionRole.BACKGROUND, SectionRole.SLEEP, SectionRole.OTHER],
                evidence_ids=evidence_ids,
                debug_payload=debug_payload,
            )

        return self._decision(
            ClaimSurfaceAction.BLOCK,
            "unmapped_finding_type",
            "Finding type is not mapped to a safe report-surface template.",
            evidence_ids=evidence_ids,
            debug_payload=debug_payload,
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
