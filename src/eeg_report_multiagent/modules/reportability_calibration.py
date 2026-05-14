from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, Field

from eeg_report_multiagent.schemas.finding import FindingObject
from eeg_report_multiagent.schemas.measurement import MeasurementValue, StatusSemantic
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction, SurfaceDecision
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


_ACTION_RANK = {
    ClaimSurfaceAction.DEBUG_ONLY: 0,
    ClaimSurfaceAction.BLOCK: 1,
    ClaimSurfaceAction.CAVEAT: 2,
    ClaimSurfaceAction.ALLOW: 3,
}


class EvidenceWeightingResult(BaseModel):
    """Compact Stage 3C reportability decision for one finding/evidence group."""

    evidence_weight_score: float = 0.0
    directness_score: float = 0.0
    provenance_score: float = 0.0
    clinical_relevance_score: float = 0.0
    conflict_penalty: float = 0.0
    missing_support_penalty: float = 0.0
    recommended_reportability: ClaimSurfaceAction
    clinical_phrase_template_id: str | None = None
    allowed_sections: list[str] = Field(default_factory=list)
    forbidden_sections: list[str] = Field(default_factory=list)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    missing_support_notes: list[str] = Field(default_factory=list)
    safe_surface_override: bool = False
    debug_payload: dict[str, Any] = Field(default_factory=dict)

    def to_surface_decision(self) -> SurfaceDecision:
        return SurfaceDecision(
            surface_action=self.recommended_reportability,
            allowed_sections=list(self.allowed_sections),
            forbidden_sections=list(self.forbidden_sections),
            clinical_phrase_template_id=self.clinical_phrase_template_id,
            rationale=self.rationale,
            evidence_ids=list(self.evidence_ids),
            debug_payload={
                "stage3c_calibration": self.model_dump(mode="json"),
                **self.debug_payload,
            },
        )


class EvidenceReportabilityCalibrator:
    """Minimal Stage 3C calibration layer.

    This layer is deliberately narrow: it may upgrade only slot-specific evidence
    to allow/caveat when the evidence has clinically meaningful provenance. It
    never turns internal scores, candidate burden, ratios, or event candidates
    into final report claims, and it never relaxes the seizure gate.
    """

    INTERNAL_SCORE_NAMES = {
        "event_candidate_burden_ratio",
        "event_train_duration_upper_sec",
        "event_train_duration_distribution_sec",
        "event_morphology_support_score",
        "event_morphology_proxy_score_distribution",
        "epileptiform_candidate_likelihood_score",
        "electrographic_seizure_likelihood_score",
        "event_laterality_index",
        "event_localization_concentration_ratio",
        "event_peak_field_concentration_ratio",
        "event_peak_laterality_index",
        "event_bifrontal_ratio",
        "event_field_concentration_ratio",
        "pdr_candidate_confidence_score",
        "pdr_posterior_anterior_alpha_ratio",
        "pdr_symmetry_score",
        "background_ap_organization_score",
        "slowing_score",
        "beta_excess_score",
    }
    EVENT_CANDIDATE_FINDINGS = {"epileptiform_event_candidate_burden", "event_train_duration"}
    LOCALIZATION_RATIO_FINDINGS = {
        "event_laterality",
        "event_focality_bifrontal_spread",
        "event_localization_support",
        "event_peak_field_support",
        "event_peak_laterality",
        "event_field_concentration",
    }
    MORPHOLOGY_PROXY_FINDINGS = {"event_morphology_class", "event_morphology_support"}

    def calibrate_decision(
        self,
        *,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        evidence_items: list[EvidenceItem],
        shared_board: SharedEvidenceBoard,
        base_decision: SurfaceDecision,
        missing_evidence: list[str],
    ) -> SurfaceDecision:
        result = self.calibrate(
            finding=finding,
            measurement=measurement,
            evidence_items=evidence_items,
            shared_board=shared_board,
            missing_evidence=missing_evidence,
        )
        if result is None:
            return self._with_calibration(base_decision, None)

        calibrated = result.to_surface_decision()
        if self._is_less_permissive(calibrated.surface_action, base_decision.surface_action):
            return calibrated
        if self._is_more_permissive(calibrated.surface_action, base_decision.surface_action):
            if result.safe_surface_override:
                return calibrated
            return self._with_calibration(base_decision, result)
        merged = calibrated
        merged.debug_payload = {**base_decision.debug_payload, **calibrated.debug_payload, "base_surface_action": base_decision.surface_action.value}
        return merged

    def calibrate(
        self,
        *,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        evidence_items: list[EvidenceItem],
        shared_board: SharedEvidenceBoard,
        missing_evidence: list[str],
    ) -> EvidenceWeightingResult | None:
        evidence_ids = [item.evidence_id for item in evidence_items]
        base_payload = {
            "finding_type": finding.finding_type,
            "measurement_name": measurement.measurement_name if measurement else None,
            "evidence_ids": evidence_ids,
        }

        hard_block = self._hard_block_reason(finding, measurement, evidence_items)
        if hard_block:
            return EvidenceWeightingResult(
                recommended_reportability=ClaimSurfaceAction.DEBUG_ONLY if "debug" in hard_block else ClaimSurfaceAction.BLOCK,
                rationale=hard_block,
                evidence_ids=evidence_ids,
                missing_support_notes=list(missing_evidence),
                debug_payload=base_payload,
            )

        if finding.finding_type.startswith("protocol_"):
            return self._metadata_status_result(finding, evidence_items, missing_evidence, base_payload)
        if finding.finding_type in {"background_reactivity", "sleep_architecture"}:
            return self._status_result(finding, evidence_items, missing_evidence, base_payload)
        if finding.finding_type == "background_pdr_frequency":
            return self._pdr_result(finding, measurement, evidence_items, missing_evidence, base_payload)
        if finding.finding_type == "background_amplitude_range":
            return self._background_amplitude_result(finding, measurement, evidence_items, missing_evidence, base_payload)
        if finding.finding_type in {"background_slowing", "excess_beta"}:
            return self._background_screen_result(finding, evidence_items, missing_evidence, base_payload)
        if finding.finding_type == "event_peak_localization":
            return self._localization_result(finding, measurement, evidence_items, shared_board, missing_evidence, base_payload)
        if finding.finding_type in self.MORPHOLOGY_PROXY_FINDINGS:
            return self._morphology_result(finding, evidence_items, shared_board, missing_evidence, base_payload)
        if "seizure" in finding.finding_type:
            return self._seizure_result(finding, evidence_items, missing_evidence, base_payload)
        return None

    def _metadata_status_result(
        self,
        finding: FindingObject,
        evidence_items: list[EvidenceItem],
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        if finding.assertion == StatusSemantic.UNKNOWN:
            return self._blocked("Unknown metadata/status evidence is not surfaced.", evidence_items, missing_evidence, payload)
        target = ClinicalTarget.STATE if "state" in finding.finding_type else ClinicalTarget.PROTOCOL
        return EvidenceWeightingResult(
            evidence_weight_score=0.80,
            directness_score=0.85,
            provenance_score=0.70,
            clinical_relevance_score=0.70,
            recommended_reportability=ClaimSurfaceAction.ALLOW,
            clinical_phrase_template_id="protocol_status",
            allowed_sections=[SectionRole.DETAIL.value, SectionRole.BACKGROUND.value, SectionRole.SLEEP.value, SectionRole.OTHER.value],
            rationale=f"Non-unknown {target.value} status is metadata/status evidence and does not require numeric provenance.",
            evidence_ids=[item.evidence_id for item in evidence_items],
            missing_support_notes=list(missing_evidence),
            safe_surface_override=True,
            debug_payload={**payload, "clinical_target": target.value},
        )

    def _status_result(
        self,
        finding: FindingObject,
        evidence_items: list[EvidenceItem],
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        if finding.assertion == StatusSemantic.UNKNOWN:
            return self._blocked("Unknown status evidence is not surfaced.", evidence_items, missing_evidence, payload)
        return EvidenceWeightingResult(
            evidence_weight_score=0.72,
            directness_score=0.75,
            provenance_score=self._provenance_score(evidence_items),
            clinical_relevance_score=0.65,
            recommended_reportability=ClaimSurfaceAction.ALLOW,
            clinical_phrase_template_id=finding.finding_type,
            allowed_sections=[SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.SLEEP.value],
            rationale="Structured non-unknown status evidence may surface as status prose.",
            evidence_ids=[item.evidence_id for item in evidence_items],
            missing_support_notes=list(missing_evidence),
            safe_surface_override=True,
            debug_payload=payload,
        )

    def _pdr_result(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        evidence_items: list[EvidenceItem],
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        value = self._numeric_value(finding, measurement)
        if value is None:
            return self._blocked("PDR frequency lacks a numeric value.", evidence_items, missing_evidence, payload)
        if value < 7.0 or value > 13.0:
            return self._blocked("Global/boundary or non-alpha frequency cannot be surfaced as PDR.", evidence_items, missing_evidence, {**payload, "frequency_hz": value})
        has_posterior = self._has_posterior_support(finding, measurement, evidence_items)
        if not has_posterior:
            return self._blocked("Plausible alpha frequency lacks posterior/occipital provenance for PDR wording.", evidence_items, missing_evidence, {**payload, "frequency_hz": value})
        supported = measurement is not None and measurement.metadata.get("pdr_supported") == "true"
        action = ClaimSurfaceAction.ALLOW if supported and not missing_evidence else ClaimSurfaceAction.CAVEAT
        return EvidenceWeightingResult(
            evidence_weight_score=0.70 if action == ClaimSurfaceAction.CAVEAT else 0.88,
            directness_score=0.65,
            provenance_score=self._provenance_score(evidence_items),
            clinical_relevance_score=0.90,
            missing_support_penalty=0.15 if missing_evidence else 0.0,
            recommended_reportability=action,
            clinical_phrase_template_id="background_posterior_alpha_candidate",
            allowed_sections=[SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.SLEEP.value],
            rationale="Posterior alpha-range evidence may surface with caveated wording unless full PDR support is available.",
            evidence_ids=[item.evidence_id for item in evidence_items],
            missing_support_notes=list(missing_evidence),
            safe_surface_override=True,
            debug_payload={**payload, "frequency_hz": value, "posterior_support": has_posterior, "pdr_supported": supported},
        )

    def _background_amplitude_result(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        evidence_items: list[EvidenceItem],
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        unit = (finding.quantitation.unit if finding.quantitation else None) or (measurement.quantitation.unit if measurement and measurement.quantitation else None)
        if (unit or "").lower() not in {"uv", "µv", "microvolt", "microvolts"}:
            return self._blocked("Background amplitude requires a clinically meaningful uV unit.", evidence_items, missing_evidence, payload)
        if not self._has_any_provenance(evidence_items, finding, measurement):
            return self._blocked("Background amplitude lacks provenance.", evidence_items, missing_evidence, payload)
        return EvidenceWeightingResult(
            evidence_weight_score=0.78,
            directness_score=0.70,
            provenance_score=self._provenance_score(evidence_items),
            clinical_relevance_score=0.70,
            recommended_reportability=ClaimSurfaceAction.CAVEAT,
            clinical_phrase_template_id="background_amplitude_range",
            allowed_sections=[SectionRole.BACKGROUND.value, SectionRole.DETAIL.value],
            rationale="Background amplitude has uV units and provenance, so it may surface as caveated measurement prose.",
            evidence_ids=[item.evidence_id for item in evidence_items],
            missing_support_notes=list(missing_evidence),
            safe_surface_override=True,
            debug_payload={**payload, "unit": unit},
        )

    def _background_screen_result(
        self,
        finding: FindingObject,
        evidence_items: list[EvidenceItem],
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        if finding.assertion != StatusSemantic.PRESENT:
            return self._blocked("Absent background screen is not surfaced.", evidence_items, missing_evidence, payload)
        if any(item.evidence_type == EvidenceType.DEBUG for item in evidence_items):
            return self._blocked("Debug-only background score is retained for audit rather than prose.", evidence_items, missing_evidence, payload)
        template = "background_slowing_assistive" if finding.finding_type == "background_slowing" else "background_beta_assistive"
        return EvidenceWeightingResult(
            evidence_weight_score=0.62,
            directness_score=0.55,
            provenance_score=self._provenance_score(evidence_items),
            clinical_relevance_score=0.65,
            recommended_reportability=ClaimSurfaceAction.CAVEAT,
            clinical_phrase_template_id=template,
            allowed_sections=[SectionRole.BACKGROUND.value, SectionRole.DETAIL.value, SectionRole.IMPRESSION.value],
            rationale="Derived background screen may surface only as caveated assistive prose.",
            evidence_ids=[item.evidence_id for item in evidence_items],
            missing_support_notes=list(missing_evidence),
            safe_surface_override=True,
            debug_payload=payload,
        )

    def _morphology_result(
        self,
        finding: FindingObject,
        evidence_items: list[EvidenceItem],
        shared_board: SharedEvidenceBoard,
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        if any(item.evidence_type == EvidenceType.DEBUG for item in evidence_items):
            return EvidenceWeightingResult(
                recommended_reportability=ClaimSurfaceAction.DEBUG_ONLY,
                rationale="Morphology proxy/debug evidence cannot surface as definite epileptiform prose.",
                evidence_ids=[item.evidence_id for item in evidence_items],
                missing_support_notes=list(missing_evidence),
                debug_payload=payload,
            )
        has_context = self._has_target(shared_board, ClinicalTarget.LOCALIZATION) or self._has_target(shared_board, ClinicalTarget.EVENT_CANDIDATE)
        if finding.assertion == StatusSemantic.PRESENT and has_context and self._has_any_space(evidence_items):
            return EvidenceWeightingResult(
                evidence_weight_score=0.50,
                directness_score=0.35,
                provenance_score=self._provenance_score(evidence_items),
                clinical_relevance_score=0.60,
                recommended_reportability=ClaimSurfaceAction.CAVEAT,
                clinical_phrase_template_id=None,
                allowed_sections=[SectionRole.EPILEPTIFORM.value, SectionRole.DETAIL.value],
                rationale="Morphology evidence remains caveated and requires a future safe template before prose.",
                evidence_ids=[item.evidence_id for item in evidence_items],
                missing_support_notes=list(missing_evidence),
                safe_surface_override=False,
                debug_payload=payload,
            )
        return self._blocked("Morphology evidence lacks validated morphology plus field/context support.", evidence_items, missing_evidence, payload)

    def _localization_result(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        evidence_items: list[EvidenceItem],
        shared_board: SharedEvidenceBoard,
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        label = measurement.categorical_value if measurement else None
        if not label or label == "unknown":
            return self._blocked("Localization label is unknown.", evidence_items, missing_evidence, payload)
        has_space = self._has_any_space(evidence_items) or (measurement is not None and bool(measurement.provenance.space.channels or measurement.provenance.space.region or measurement.provenance.space.laterality))
        has_context = self._has_target(shared_board, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY) or self._has_target(shared_board, ClinicalTarget.EVENT_CANDIDATE)
        if has_space and has_context and not self._label_is_ratio_only(label):
            return EvidenceWeightingResult(
                evidence_weight_score=0.56,
                directness_score=0.40,
                provenance_score=self._provenance_score(evidence_items),
                clinical_relevance_score=0.75,
                recommended_reportability=ClaimSurfaceAction.CAVEAT,
                clinical_phrase_template_id=None,
                allowed_sections=[SectionRole.EPILEPTIFORM.value, SectionRole.EVENTS_SEIZURES.value, SectionRole.DETAIL.value],
                rationale="Localization proxy has a label and space provenance, but requires a future safe wording template before prose.",
                evidence_ids=[item.evidence_id for item in evidence_items],
                missing_support_notes=list(missing_evidence),
                safe_surface_override=False,
                debug_payload={**payload, "label": label},
            )
        return self._blocked("Localization ratio/label lacks enough morphology or space provenance for prose.", evidence_items, missing_evidence, {**payload, "label": label})

    def _seizure_result(
        self,
        finding: FindingObject,
        evidence_items: list[EvidenceItem],
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        seizure_items = [item for item in evidence_items if item.clinical_target == ClinicalTarget.SEIZURE_EVIDENCE]
        if not seizure_items:
            return self._blocked("Seizure claim requires seizure-specific evidence or validated metadata.", evidence_items, missing_evidence, payload)
        if any(item.evidence_type in {EvidenceType.DIRECT, EvidenceType.METADATA, EvidenceType.DERIVED} for item in seizure_items):
            return EvidenceWeightingResult(
                evidence_weight_score=0.80,
                directness_score=0.80,
                provenance_score=self._provenance_score(seizure_items),
                clinical_relevance_score=0.90,
                recommended_reportability=ClaimSurfaceAction.CAVEAT,
                clinical_phrase_template_id=None,
                allowed_sections=[SectionRole.SEIZURES.value, SectionRole.EVENTS_SEIZURES.value],
                rationale="Seizure evidence exists, but no v1 safe seizure template is enabled yet.",
                evidence_ids=[item.evidence_id for item in seizure_items],
                missing_support_notes=list(missing_evidence),
                safe_surface_override=False,
                debug_payload=payload,
            )
        return self._blocked("Event candidates or debug scores cannot create seizure claims.", evidence_items, missing_evidence, payload)

    def _hard_block_reason(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        evidence_items: list[EvidenceItem],
    ) -> str | None:
        mname = measurement.measurement_name if measurement else ""
        if finding.finding_type in self.EVENT_CANDIDATE_FINDINGS:
            return "Event candidate burden/duration remains proxy/debug evidence and cannot surface directly."
        if finding.finding_type in self.LOCALIZATION_RATIO_FINDINGS:
            return "Localization ratios and spread screens are debug/proxy evidence and cannot surface directly."
        if mname in self.INTERNAL_SCORE_NAMES:
            return "Internal debug score measurement remains debug_only."
        if any(item.evidence_type == EvidenceType.DEBUG for item in evidence_items) and finding.finding_type not in self.MORPHOLOGY_PROXY_FINDINGS:
            return "Linked debug evidence item cannot directly surface as clinical prose."
        return None

    def _blocked(
        self,
        rationale: str,
        evidence_items: list[EvidenceItem],
        missing_evidence: list[str],
        payload: dict[str, Any],
    ) -> EvidenceWeightingResult:
        return EvidenceWeightingResult(
            recommended_reportability=ClaimSurfaceAction.BLOCK,
            rationale=rationale,
            evidence_ids=[item.evidence_id for item in evidence_items],
            missing_support_notes=list(missing_evidence),
            debug_payload=payload,
        )

    def _with_calibration(self, decision: SurfaceDecision, result: EvidenceWeightingResult | None) -> SurfaceDecision:
        payload = dict(decision.debug_payload)
        if result is not None:
            payload["stage3c_calibration"] = result.model_dump(mode="json")
        return decision.model_copy(update={"debug_payload": payload})

    def _is_more_permissive(self, action: ClaimSurfaceAction, base: ClaimSurfaceAction) -> bool:
        return _ACTION_RANK[action] > _ACTION_RANK[base]

    def _is_less_permissive(self, action: ClaimSurfaceAction, base: ClaimSurfaceAction) -> bool:
        return _ACTION_RANK[action] < _ACTION_RANK[base]

    def _numeric_value(self, finding: FindingObject, measurement: MeasurementValue | None) -> float | None:
        q = finding.quantitation or (measurement.quantitation if measurement else None)
        if q is None:
            return None
        if q.exact is not None:
            return q.exact
        if q.lower is not None and q.upper is not None:
            return (q.lower + q.upper) / 2.0
        return None

    def _has_posterior_support(self, finding: FindingObject, measurement: MeasurementValue | None, evidence_items: list[EvidenceItem]) -> bool:
        posterior_tokens = {"o1", "o2", "oz", "p3", "p4", "pz", "occipital", "posterior", "parietal"}
        if measurement is not None and measurement.metadata.get("pdr_supported") == "true":
            return True
        records = list(finding.provenance)
        if measurement is not None:
            records.append(measurement.provenance)
        for record in records:
            values = [record.space.region or "", record.space.laterality or "", *record.space.channels]
            if any(str(value).lower() in posterior_tokens for value in values):
                return True
        for item in evidence_items:
            space = item.space_provenance or {}
            values = [space.get("region", ""), space.get("side", ""), *(space.get("channels") or []), *(space.get("electrode_maxima") or [])]
            if any(str(value).lower() in posterior_tokens for value in values):
                return True
        return False

    def _has_any_provenance(self, evidence_items: list[EvidenceItem], finding: FindingObject, measurement: MeasurementValue | None) -> bool:
        if any(item.time_provenance or item.space_provenance for item in evidence_items):
            return True
        if finding.provenance:
            return True
        return measurement is not None and measurement.provenance is not None

    def _has_any_space(self, evidence_items: Iterable[EvidenceItem]) -> bool:
        for item in evidence_items:
            space = item.space_provenance or {}
            if space.get("channels") or space.get("electrode_maxima") or space.get("region") or space.get("side"):
                return True
        return False

    def _provenance_score(self, evidence_items: list[EvidenceItem]) -> float:
        if not evidence_items:
            return 0.0
        scores = []
        for item in evidence_items:
            score = 0.0
            if item.time_provenance:
                score += 0.35
            if item.space_provenance:
                score += 0.35
            if item.unit or item.value is not None:
                score += 0.20
            if item.source_module:
                score += 0.10
            scores.append(min(score, 1.0))
        return max(scores)

    def _has_target(self, board: SharedEvidenceBoard, target: ClinicalTarget) -> bool:
        return any(item.clinical_target == target for item in board.evidence_items)

    def _label_is_ratio_only(self, label: str) -> bool:
        lowered = label.lower()
        return "ratio" in lowered or "index" in lowered or "spread tendency" in lowered
