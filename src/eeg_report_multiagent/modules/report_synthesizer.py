from __future__ import annotations

from typing import Dict, List

from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.finding import FindingObject
from eeg_report_multiagent.schemas.measurement import MeasurementValue, QuantitationValue, StatusSemantic
from eeg_report_multiagent.schemas.report import (
    AtomicClaimPlan,
    ClaimRecord,
    ClaimSurfaceAction,
    ReportSection,
    ReportSectionType,
    SurfaceDecision,
)
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.modules.surface_policy import SurfacePolicy


class ReportSynthesizer:
    """Template-based v1 synthesizer. Reads EvidenceBoard only."""

    DEBUG_SURFACE_FINDING_TYPES = SurfacePolicy.DEBUG_ONLY_FINDING_TYPES

    def __init__(self, surface_policy: SurfacePolicy | None = None) -> None:
        self.surface_policy = surface_policy or SurfacePolicy()

    def synthesize(self, board: EvidenceBoard) -> tuple[ReportSection, ReportSection, List[ClaimRecord]]:
        claims: List[ClaimRecord] = []
        claim_plan = self.build_atomic_claim_plan(board)
        shared_board = board.ensure_shared_evidence_board()
        detail_lines = self._section_lines_from_plans(claim_plan, SectionRole.DETAIL)

        for plan in self._surfaceable_plans(claim_plan):
            claim_id = f"c_{plan.plan_id}"
            claims.append(
                ClaimRecord(
                    claim_id=claim_id,
                    section_type=plan.section_type,
                    text=plan.proposed_text,
                    linked_finding_ids=plan.linked_finding_ids,
                )
            )
            if plan.evidence_ids:
                shared_board.link_to_claim(claim_id, plan.evidence_ids)
        if not detail_lines:
            detail_lines = [self.surface_policy.safe_fallback_for_role(SectionRole.DETAIL)]

        impression_lines = self._section_lines_from_plans(claim_plan, SectionRole.IMPRESSION)
        if not impression_lines:
            impression_lines = [self.surface_policy.safe_fallback_for_role(SectionRole.IMPRESSION)]

        imp_text = " ".join(impression_lines)
        claims.append(
            ClaimRecord(
                claim_id="c_impression_summary",
                section_type=ReportSectionType.IMPRESSION,
                text=imp_text,
                linked_finding_ids=[fid for plan in self._surfaceable_plans(claim_plan) for fid in plan.linked_finding_ids],
            )
        )

        detail_section = ReportSection(
            section_type=ReportSectionType.DETAIL,
            text="\n".join(detail_lines) if detail_lines else "No detail-level findings available.",
            claim_ids=[c.claim_id for c in claims if c.section_type == ReportSectionType.DETAIL],
        )
        impression_section = ReportSection(
            section_type=ReportSectionType.IMPRESSION,
            text=imp_text,
            claim_ids=[c.claim_id for c in claims if c.section_type == ReportSectionType.IMPRESSION],
        )
        return detail_section, impression_section, claims

    def build_atomic_claim_plan(self, board: EvidenceBoard) -> List[AtomicClaimPlan]:
        """Plan report-surface claims from evidence-board findings.

        Tools, parsers, encoders, and LLM evidence review may create typed
        measurements/findings. This method is the first place where those
        evidence objects are converted into candidate clinical report claims.
        """
        shared_board = board.ensure_shared_evidence_board()
        measurement_index = {m.measurement_id: m for m in board.measurements}
        plans: List[AtomicClaimPlan] = []
        for finding in board.findings:
            measurement = self._first_measurement(finding, measurement_index)
            required, missing = self._claim_evidence_requirements(finding, measurement)
            evidence_items = [
                item
                for item in shared_board.evidence_items
                if finding.finding_id in item.finding_ids
                or any(measurement_id in item.measurement_ids for measurement_id in finding.measurement_ids)
            ]
            decision = self.surface_policy.decide(
                finding,
                measurement=measurement,
                missing_evidence=missing,
                evidence_items=evidence_items,
            )
            proposed_text = self._claim_text_from_surface_decision(finding, measurement, decision)
            evidence_ids = [item.evidence_id for item in evidence_items] or decision.evidence_ids
            plans.append(
                AtomicClaimPlan(
                    plan_id=f"p_{finding.finding_id}",
                    section_type=ReportSectionType.DETAIL,
                    claim_type=finding.finding_type,
                    proposed_text=proposed_text,
                    evidence_ids=evidence_ids,
                    linked_finding_ids=[finding.finding_id],
                    linked_measurement_ids=list(finding.measurement_ids),
                    required_evidence=required,
                    missing_evidence=missing,
                    surface_action=decision.surface_action,
                    confidence=finding.confidence,
                    rationale=decision.rationale,
                    allowed_sections=decision.allowed_sections,
                    forbidden_sections=decision.forbidden_sections,
                    clinical_phrase_template_id=decision.clinical_phrase_template_id,
                    debug_payload=decision.debug_payload,
                )
            )
        return plans

    def synthesize_celm_sections(self, board: EvidenceBoard, target_section_names: List[str]) -> dict[str, str]:
        """Generate section-specific text for CELM-compatible evaluation outputs.

        This is intentionally downstream of the EvidenceBoard. It does not inspect raw EEG or GT text.
        """
        router = SectionRouter()
        claim_plan = self.build_atomic_claim_plan(board)
        section_texts: dict[str, str] = {}
        for section_name in target_section_names:
            role = router.role_for_section(section_name)
            section_texts[section_name] = self._section_text_from_plans(claim_plan, role)
        return section_texts

    def _first_measurement(self, finding: FindingObject, measurement_index: dict[str, MeasurementValue]) -> MeasurementValue | None:
        for mid in finding.measurement_ids:
            if mid in measurement_index:
                return measurement_index[mid]
        return None

    def _format_number(self, v: float) -> str:
        av = abs(v)
        if av > 0.0 and av < 1e-3:
            return f"{v:.2e}"
        if av >= 10:
            return f"{v:.0f}"
        return f"{v:.1f}"

    def _format_quantitation(self, q: QuantitationValue | None) -> str:
        if q is None:
            return ""
        unit = f" {q.unit}" if q.unit else ""
        if q.exact is not None:
            return f"{self._format_number(q.exact)}{unit}"
        if q.lower is not None and q.upper is not None:
            return f"{self._format_number(q.lower)}-{self._format_number(q.upper)}{unit}"
        if q.upper is not None:
            return f"up to {self._format_number(q.upper)}{unit}"
        if q.lower is not None:
            return f"at least {self._format_number(q.lower)}{unit}"
        if q.values:
            return f"n={len(q.values)}{unit}"
        return ""

    def _status_phrase(self, s: StatusSemantic) -> str:
        return {
            StatusSemantic.PRESENT: "present",
            StatusSemantic.ABSENT: "absent",
            StatusSemantic.NOT_OBSERVED: "not observed",
            StatusSemantic.NOT_PERFORMED: "not performed",
            StatusSemantic.NO_RESPONSE: "no response",
            StatusSemantic.UNKNOWN: "unknown",
        }[s]

    def _surfaceable_plans(self, claim_plan: List[AtomicClaimPlan]) -> List[AtomicClaimPlan]:
        return [
            plan
            for plan in claim_plan
            if plan.surface_action in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}
            and not self.surface_policy.contains_forbidden_surface_text(plan.proposed_text)
        ]

    def _section_lines_from_plans(self, claim_plan: List[AtomicClaimPlan], role: SectionRole) -> List[str]:
        lines: List[str] = []
        for plan in claim_plan:
            if not self.surface_policy.plan_allowed_in_section(plan, role):
                continue
            if self.surface_policy.contains_forbidden_surface_text(plan.proposed_text):
                continue
            lines.append(plan.proposed_text)
        return lines

    def _section_text_from_plans(self, claim_plan: List[AtomicClaimPlan], role: SectionRole) -> str:
        lines = self._section_lines_from_plans(claim_plan, role)
        if lines:
            return " ".join(lines)
        return self.surface_policy.safe_fallback_for_role(role)

    def _claim_text_from_surface_decision(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        decision: SurfaceDecision,
    ) -> str:
        template_id = decision.clinical_phrase_template_id or "unmapped"
        if decision.surface_action in {ClaimSurfaceAction.BLOCK, ClaimSurfaceAction.DEBUG_ONLY}:
            return self.surface_policy.safe_fallback_for_role(SectionRole.OTHER)

        if template_id == "background_posterior_alpha_candidate":
            qtxt = self._format_quantitation(finding.quantitation)
            near = f" near {qtxt}" if qtxt else ""
            return (
                f"A posterior alpha rhythm candidate{near} is supported by structured evidence; "
                "state and reactivity confirmation are not available from the current evidence board."
            )
        if template_id == "background_amplitude_range":
            qtxt = self._format_quantitation(finding.quantitation)
            if qtxt:
                return f"A provenance-linked background amplitude range is available ({qtxt})."
            return "A provenance-linked background amplitude measurement is available."
        if template_id == "background_slowing_assistive":
            return "Structured evidence suggests background slowing; this remains an assistive finding pending EEG review."
        if template_id == "background_beta_assistive":
            return "Structured evidence suggests increased beta activity; this remains an assistive finding pending EEG review."
        if template_id in {"background_reactivity", "sleep_architecture"}:
            label = "Background reactivity" if finding.finding_type == "background_reactivity" else "Sleep architecture"
            return f"{label}: {self._status_phrase(finding.assertion)}."
        if template_id == "protocol_status":
            label = self._finding_label(finding, measurement)
            return f"{label}: {self._status_phrase(finding.assertion)}."

        return self.surface_policy.safe_fallback_for_role(SectionRole.OTHER)

    def _is_abnormal_finding(self, f: FindingObject) -> bool:
        if f.assertion != StatusSemantic.PRESENT:
            return False
        return f.finding_type in {"background_pdr_frequency", "background_amplitude_range", "background_slowing", "excess_beta"}

    def _finding_label(self, finding: FindingObject, measurement: MeasurementValue | None) -> str:
        mapping = {
            "background_frequency": "Background dominant frequency",
            "background_pdr_frequency": "Posterior dominant rhythm candidate",
            "background_pdr_support": "Posterior rhythm evidence",
            "background_pdr_topography": "Posterior alpha topography evidence",
            "background_pdr_symmetry": "Posterior alpha symmetry evidence",
            "background_ap_organization": "Anterior-posterior organization evidence",
            "background_reactivity": "Background reactivity",
            "sleep_architecture": "Sleep architecture",
            "background_amplitude_range": "Background amplitude range",
            "background_slowing": "Background slowing index",
            "excess_beta": "Excess beta activity",
            "epileptiform_event_candidate_burden": "Event candidate burden",
            "event_train_duration": "Longest candidate train duration",
            "event_laterality": "Event laterality index",
            "event_focality_bifrontal_spread": "Bifrontal spread tendency",
            "event_clinical_localization": "Event localization screen",
            "event_localization_support": "Event localization support",
            "event_peak_localization": "Event peak-centered localization screen",
            "event_peak_field_support": "Event peak-centered field support",
            "event_peak_laterality": "Event peak-centered laterality index",
            "event_morphology_class": "Event morphology screen class",
            "event_morphology_support": "Local morphology-feature support",
            "event_field_concentration": "Event field concentration",
            "epileptiform_candidate_likelihood": "Epileptiform-candidate likelihood",
            "electrographic_seizure_likelihood": "Electrographic seizure likelihood",
            "protocol_state_awake": "Awake state",
            "protocol_hyperventilation_status": "Hyperventilation",
            "protocol_photic_stimulation_status": "Photic stimulation",
            "protocol_ekg_availability": "EKG availability",
            "protocol_video_availability": "Video availability",
            "protocol_comparison_history_presence": "Comparison/history mention",
        }
        if finding.finding_type == "background_measurement" and measurement is not None:
            name = measurement.measurement_name
            if name.startswith("relative_bandpower_"):
                band = name.replace("relative_bandpower_", "")
                return f"Relative {band} bandpower"
        if finding.finding_type == "event_measurement" and measurement is not None:
            name = measurement.measurement_name
            if name == "event_candidate_score_distribution":
                return "Event candidate score distribution"
            if name == "event_train_duration_distribution_sec":
                return "Event train duration distribution"
        return mapping.get(finding.finding_type, finding.finding_type)

    def _categorical_for_type(self, board: EvidenceBoard, finding_type: str) -> str | None:
        measurement_index = {m.measurement_id: m for m in board.measurements}
        for finding in board.findings:
            if finding.finding_type != finding_type:
                continue
            measurement = self._first_measurement(finding, measurement_index)
            if measurement is not None:
                return measurement.categorical_value
        return None

    def _clinical_quantitation_allowed(self, finding: FindingObject) -> bool:
        if finding.finding_type == "background_pdr_frequency" and finding.assertion == StatusSemantic.PRESENT:
            return True
        if finding.finding_type == "background_amplitude_range" and finding.assertion == StatusSemantic.PRESENT:
            return True
        if finding.finding_type == "event_peak_localization" and finding.assertion == StatusSemantic.PRESENT:
            return False
        # Everything below is debug/proxy unless a future claim gate upgrades it.
        if finding.finding_type in self.DEBUG_SURFACE_FINDING_TYPES:
            return False
        return finding.assertion == StatusSemantic.PRESENT

    def _claim_evidence_requirements(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
    ) -> tuple[List[str], List[str]]:
        required: List[str] = []
        missing: List[str] = []
        if finding.finding_type == "background_pdr_frequency":
            required = [
                "posterior_or_occipital_alpha_frequency",
                "posterior_topographic_predominance",
                "awake_or_resting_state",
                "reactivity_or_eye_opening_attenuation_when_available",
            ]
            if measurement is None:
                missing.append("posterior_alpha_measurement")
            if measurement is not None and measurement.metadata.get("pdr_supported") != "true":
                missing.append("strong_posterior_alpha_support")
            missing.extend(["state_specific_eye_condition", "reactivity"])
        elif finding.finding_type == "event_peak_localization":
            required = [
                "event_peak_time_provenance",
                "peak_centered_channel_field",
                "laterality_index",
                "clinical_region_mapping",
            ]
            if measurement is None:
                missing.append("peak_topography_measurement")
            missing.append("definitive_epileptiform_morphology")
        elif finding.finding_type in {"event_morphology_class", "event_morphology_support"}:
            required = ["focused_event_waveform_features", "field_distribution", "morphology_classifier_version"]
            missing.append("validated_spike_sharp_wave_classifier")
        elif finding.finding_type in {"epileptiform_event_candidate_burden", "event_train_duration"}:
            required = ["candidate_window_score_distribution", "focused_window_provenance"]
            missing.append("morphology_specific_epileptiform_confirmation")
        elif finding.finding_type.startswith("protocol_"):
            required = ["study_metadata_or_report_context"]
            if finding.assertion == StatusSemantic.UNKNOWN:
                missing.append("recoverable_protocol_status")
        elif finding.finding_type.startswith("background_") or finding.finding_type == "excess_beta":
            required = ["signal_measurement_provenance"]
            if measurement is None:
                missing.append("linked_measurement")
        else:
            required = ["linked_structured_finding"]
            if measurement is None:
                missing.append("linked_measurement")
        return required, missing

    def _claim_surface_action(
        self,
        board: EvidenceBoard,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        missing_evidence: List[str],
    ) -> ClaimSurfaceAction:
        return self.surface_policy.decide(finding, measurement=measurement, missing_evidence=missing_evidence).surface_action

    def _event_localization_surface_allowed(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
        section_role: SectionRole,
    ) -> bool:
        """Decide whether peak-localization proxy may appear in clinical prose.

        Localization v2 is always preserved in provenance. It reaches report
        text only for event/epileptiform sections and only when multiple local
        signal proxies jointly support event-like abnormality.
        """
        if section_role not in {SectionRole.EPILEPTIFORM, SectionRole.EVENTS_SEIZURES}:
            return False
        label = self._categorical_for_type(board, "event_peak_localization")
        if not label or label == "unknown":
            return False

        burden = self._exact(self._measurement_for_type(board, "epileptiform_event_candidate_burden", measurement_index))
        peak_field = self._exact(self._measurement_for_type(board, "event_peak_field_support", measurement_index))
        epileptiform_like = self._exact(self._measurement_for_type(board, "epileptiform_candidate_likelihood", measurement_index))
        morphology_support = self._exact(self._measurement_for_type(board, "event_morphology_support", measurement_index))
        morphology_class = self._categorical_for_type(board, "event_morphology_class")

        burden_ok = burden is not None and burden >= 0.05
        field_ok = peak_field is not None and peak_field >= 2.0
        likelihood_ok = epileptiform_like is not None and epileptiform_like >= 0.70
        morphology_ok = (
            morphology_class in {"sharp_transient_candidate", "nonspecific_transient_candidate"}
            and morphology_support is not None
            and morphology_support >= 1.0
        )
        return burden_ok and field_ok and likelihood_ok and morphology_ok

    def _claim_plan_rationale(
        self,
        finding: FindingObject,
        action: ClaimSurfaceAction,
        missing_evidence: List[str],
    ) -> str:
        if action == ClaimSurfaceAction.DEBUG_ONLY:
            return "Internal proxy/debug evidence is kept out of report-surface clinical prose."
        if action == ClaimSurfaceAction.BLOCK:
            return "Finding is unknown or lacks enough structured evidence for a report-surface claim."
        if action == ClaimSurfaceAction.CAVEAT:
            missing = ", ".join(missing_evidence) if missing_evidence else "clinical confirmation"
            return f"Claim may be surfaced only as a candidate because evidence is incomplete: {missing}."
        return "Claim is allowed because it has a linked typed finding and required evidence is sufficient for v1 surface text."

    def _debug_safe_review_summary(self, board: EvidenceBoard) -> str:
        if not board.deliberations:
            return ""
        return "Structured evidence-review constraints are retained in audit artifacts and are not surfaced as raw reviewer text."

    def _findings_by_type(self, board: EvidenceBoard) -> Dict[str, FindingObject]:
        return {finding.finding_type: finding for finding in board.findings}

    def _measurement_for_type(
        self,
        board: EvidenceBoard,
        finding_type: str,
        measurement_index: dict[str, MeasurementValue],
    ) -> MeasurementValue | None:
        for finding in board.findings:
            if finding.finding_type == finding_type:
                return self._first_measurement(finding, measurement_index)
        return None

    def _finding_for_measurement_name(self, board: EvidenceBoard, measurement_name: str) -> FindingObject | None:
        measurement_to_finding: dict[str, FindingObject] = {}
        for finding in board.findings:
            for measurement_id in finding.measurement_ids:
                measurement_to_finding[measurement_id] = finding
        for measurement in board.measurements:
            if measurement.measurement_name == measurement_name:
                return measurement_to_finding.get(measurement.measurement_id)
        return None

    def _measurement_by_name(self, board: EvidenceBoard, measurement_name: str) -> MeasurementValue | None:
        for measurement in board.measurements:
            if measurement.measurement_name == measurement_name:
                return measurement
        return None

    def _exact(self, measurement: MeasurementValue | None) -> float | None:
        if measurement is None or measurement.quantitation is None:
            return None
        return measurement.quantitation.exact

    def _range_text(self, finding: FindingObject | None) -> str | None:
        if finding is None or finding.quantitation is None:
            return None
        return self._format_quantitation(finding.quantitation)

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

    def _protocol_sentence(self, findings: Dict[str, FindingObject]) -> str:
        parts: List[str] = []
        hv = findings.get("protocol_hyperventilation_status")
        photic = findings.get("protocol_photic_stimulation_status")
        awake = findings.get("protocol_state_awake")
        ekg = findings.get("protocol_ekg_availability")
        video = findings.get("protocol_video_availability")

        if awake and awake.assertion != StatusSemantic.UNKNOWN:
            parts.append(f"state was {self._status_phrase(awake.assertion)}")
        drowsy = findings.get("protocol_state_drowsy")
        sleep = findings.get("protocol_state_sleep")
        if drowsy and drowsy.assertion != StatusSemantic.UNKNOWN:
            parts.append(f"drowsiness was {self._status_phrase(drowsy.assertion)}")
        if sleep and sleep.assertion != StatusSemantic.UNKNOWN:
            parts.append(f"sleep was {self._status_phrase(sleep.assertion)}")
        if photic and photic.assertion != StatusSemantic.UNKNOWN:
            parts.append(f"photic stimulation was {self._status_phrase(photic.assertion)}")
        if hv and hv.assertion != StatusSemantic.UNKNOWN:
            parts.append(f"hyperventilation was {self._status_phrase(hv.assertion)}")
        if ekg and ekg.assertion != StatusSemantic.UNKNOWN:
            parts.append(f"EKG was {self._status_phrase(ekg.assertion)}")
        if video and video.assertion != StatusSemantic.UNKNOWN:
            parts.append(f"video was {self._status_phrase(video.assertion)}")
        if parts:
            return "Protocol/context: " + "; ".join(parts) + "."
        return "Protocol, activation, EKG, and video status were not recoverable from the provided structured context."

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

    def _detail_sentence(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        review_notes: List[str] | None = None,
    ) -> str:
        decision = self.surface_policy.decide(finding, measurement=measurement)
        return self._claim_text_from_surface_decision(finding, measurement, decision)

    def _build_impression(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
    ) -> List[str]:
        text = self._section_text_from_plans(self.build_atomic_claim_plan(board), SectionRole.IMPRESSION)
        return [text]

    def _review_notes_by_finding(self, board: EvidenceBoard) -> dict[str, List[str]]:
        notes: dict[str, List[str]] = {}
        for deliberation in board.deliberations:
            for weak in deliberation.weak_evidence:
                if weak.weakness_id == "weak_signal_spatial_provenance":
                    continue
                for fid in weak.linked_finding_ids:
                    notes.setdefault(fid, []).append("A structured weak-evidence record applies; see audit artifacts.")
            for do_not in deliberation.do_not_claim:
                for fid in do_not.linked_finding_ids:
                    notes.setdefault(fid, []).append("A structured do-not-claim record applies; see audit artifacts.")
            for constraint in deliberation.claim_constraints:
                if constraint.constraint_id == "constraint_no_focal_without_space":
                    continue
                for fid in constraint.linked_finding_ids:
                    notes.setdefault(fid, []).append("A structured claim constraint applies; see audit artifacts.")
        return notes

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
