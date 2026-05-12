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
)
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.modules.section_router import SectionRouter


class ReportSynthesizer:
    """Template-based v1 synthesizer. Reads EvidenceBoard only."""

    DEBUG_SURFACE_FINDING_TYPES = {
        "background_frequency",
        "background_pdr_support",
        "background_pdr_topography",
        "background_pdr_symmetry",
        "background_ap_organization",
        "background_slowing",
        "excess_beta",
        "epileptiform_event_candidate_burden",
        "event_train_duration",
        "event_laterality",
        "event_focality_bifrontal_spread",
        "event_morphology_support",
        "event_morphology_class",
        "event_field_concentration",
        "event_localization_support",
        "event_peak_field_support",
        "event_peak_laterality",
        "epileptiform_candidate_likelihood",
        "electrographic_seizure_likelihood",
    }

    def synthesize(self, board: EvidenceBoard) -> tuple[ReportSection, ReportSection, List[ClaimRecord]]:
        detail_lines: List[str] = []
        impression_lines: List[str] = []
        claims: List[ClaimRecord] = []
        measurement_index = {m.measurement_id: m for m in board.measurements}
        claim_plan = self.build_atomic_claim_plan(board)

        for plan in claim_plan:
            if plan.surface_action not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                continue
            detail_lines.append(plan.proposed_text)
            claims.append(
                ClaimRecord(
                    claim_id=f"c_{plan.plan_id}",
                    section_type=plan.section_type,
                    text=plan.proposed_text,
                    linked_finding_ids=plan.linked_finding_ids,
                )
            )
        if not detail_lines:
            review_summary = self._debug_safe_review_summary(board)
            if review_summary:
                detail_lines.append(review_summary)

        abnormal_findings = [f for f in board.findings if self._is_abnormal_finding(f)]
        if abnormal_findings:
            impression_lines.extend(self._build_impression(board, measurement_index))
            impression_lines.extend(self._review_impression_constraints(board))
        else:
            impression_lines.append("No clearly supported abnormal EEG finding was detected in structured evidence.")

        imp_text = " ".join(impression_lines)
        claims.append(
            ClaimRecord(
                claim_id="c_impression_summary",
                section_type=ReportSectionType.IMPRESSION,
                text=imp_text,
                linked_finding_ids=[f.finding_id for f in abnormal_findings],
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
        measurement_index = {m.measurement_id: m for m in board.measurements}
        review_notes = self._review_notes_by_finding(board)
        plans: List[AtomicClaimPlan] = []
        for finding in board.findings:
            measurement = self._first_measurement(finding, measurement_index)
            required, missing = self._claim_evidence_requirements(finding, measurement)
            action = self._claim_surface_action(board, finding, measurement, missing)
            proposed_text = self._detail_sentence(finding, measurement, review_notes.get(finding.finding_id, []))
            if action == ClaimSurfaceAction.CAVEAT and "candidate" not in proposed_text.lower():
                proposed_text = proposed_text.rstrip(".") + "; this is retained as a provenance-supported candidate."
            plans.append(
                AtomicClaimPlan(
                    plan_id=f"p_{finding.finding_id}",
                    section_type=ReportSectionType.DETAIL,
                    claim_type=finding.finding_type,
                    proposed_text=proposed_text,
                    linked_finding_ids=[finding.finding_id],
                    linked_measurement_ids=list(finding.measurement_ids),
                    required_evidence=required,
                    missing_evidence=missing,
                    surface_action=action,
                    confidence=finding.confidence,
                    rationale=self._claim_plan_rationale(finding, action, missing),
                )
            )
        return plans

    def synthesize_celm_sections(self, board: EvidenceBoard, target_section_names: List[str]) -> dict[str, str]:
        """Generate section-specific text for CELM-compatible evaluation outputs.

        This is intentionally downstream of the EvidenceBoard. It does not inspect raw EEG or GT text.
        """
        measurement_index = {m.measurement_id: m for m in board.measurements}
        review_notes = self._review_notes_by_finding(board)
        router = SectionRouter()
        section_texts: dict[str, str] = {}
        for section_name in target_section_names:
            role = router.role_for_section(section_name)
            if role == SectionRole.IMPRESSION:
                section_texts[section_name] = self._clinical_impression_text(board, measurement_index)
            elif role in {SectionRole.BACKGROUND, SectionRole.SLEEP}:
                section_texts[section_name] = self._background_section_text(board, measurement_index, review_notes)
            elif role in {SectionRole.EPILEPTIFORM, SectionRole.EVENTS_SEIZURES, SectionRole.SEIZURES}:
                section_texts[section_name] = self._event_section_text(board, measurement_index, review_notes, role)
            else:
                section_texts[section_name] = self._detail_section_text(board, measurement_index, review_notes)
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

    def _is_abnormal_finding(self, f: FindingObject) -> bool:
        if f.assertion != StatusSemantic.PRESENT:
            return False
        return f.finding_type in {
            "background_frequency",
            "background_amplitude_range",
            "background_slowing",
            "excess_beta",
            "epileptiform_event_candidate_burden",
            "event_train_duration",
            "event_laterality",
            "event_focality_bifrontal_spread",
            "event_morphology_support",
            "event_field_concentration",
        }

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
        if finding.finding_type in {
            "event_peak_field_support",
            "event_peak_laterality",
            "event_morphology_support",
            "event_field_concentration",
            "event_localization_support",
            "epileptiform_candidate_likelihood",
            "electrographic_seizure_likelihood",
        }:
            return ClaimSurfaceAction.DEBUG_ONLY
        if finding.assertion == StatusSemantic.UNKNOWN:
            return ClaimSurfaceAction.BLOCK
        if finding.finding_type == "event_peak_localization":
            return ClaimSurfaceAction.DEBUG_ONLY
        if finding.finding_type in self.DEBUG_SURFACE_FINDING_TYPES:
            return ClaimSurfaceAction.DEBUG_ONLY
        if missing_evidence and finding.finding_type in {"background_pdr_frequency", "event_morphology_class"}:
            return ClaimSurfaceAction.CAVEAT
        return ClaimSurfaceAction.ALLOW

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
        notes: List[str] = []
        for deliberation in board.deliberations:
            for weak in deliberation.weak_evidence:
                notes.append(weak.recommendation)
            for do_not in deliberation.do_not_claim:
                notes.append(do_not.text)
            for constraint in deliberation.claim_constraints:
                notes.append(constraint.constraint)
        if not notes:
            return ""
        unique_notes = []
        for note in notes:
            if note and note not in unique_notes:
                unique_notes.append(note)
        return "Evidence review: " + " ".join(unique_notes[:2])

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
        findings = self._findings_by_type(board)
        freq = self._exact(self._measurement_for_type(board, "background_frequency", measurement_index))
        pdr = self._exact(self._measurement_for_type(board, "background_pdr_frequency", measurement_index))
        pdr_support = self._exact(self._measurement_for_type(board, "background_pdr_support", measurement_index))
        pdr_topography = self._exact(self._measurement_for_type(board, "background_pdr_topography", measurement_index))
        pdr_symmetry = self._exact(self._measurement_for_type(board, "background_pdr_symmetry", measurement_index))
        organization = self._exact(self._measurement_for_type(board, "background_ap_organization", measurement_index))
        amp = self._range_text(findings.get("background_amplitude_range"))
        slowing = self._exact(self._measurement_for_type(board, "background_slowing", measurement_index))
        beta = self._exact(self._measurement_for_type(board, "excess_beta", measurement_index))

        sentences: List[str] = []
        sentences.append("Structured background evidence is summarized from local signal measurements with provenance gating.")
        if (
            pdr is not None
            and pdr_support is not None
            and pdr_support >= 0.35
            and (pdr_topography is None or pdr_topography >= 1.2)
        ):
            sentences.append(
                f"A posterior alpha/PDR candidate was estimated near {self._format_number(pdr)} Hz; this is retained as a candidate unless state and reactivity evidence are available."
            )
        elif pdr is not None:
            sentences.append(
                "A reliable PDR was not directly quantified from the available structured evidence."
            )
        if freq is not None:
            if freq <= 2.0:
                sentences.append(
                    "A low-frequency boundary spectral peak was retained as debug/provenance evidence and was not surfaced as a PDR."
                )
            else:
                sentences.append(
                    f"The available rhythm estimate supports a dominant background frequency near {self._format_number(freq)} Hz."
                )
        if amp:
            sentences.append(f"The measured background amplitude range was approximately {amp}.")
        if slowing is not None:
            if slowing >= 1.0:
                sentences.append("Low-frequency power was prominent by the configured local screen.")
            else:
                sentences.append("The configured local slowing screen did not support prominent slowing.")
        if beta is not None:
            if beta >= 0.2:
                sentences.append("Relative beta activity was elevated by the configured local screen.")
            else:
                sentences.append("No excess beta activity was supported by the configured local screen.")
        if organization is not None:
            if organization >= 0.35:
                sentences.append(
                    "Anterior-posterior organization had limited structured support."
                )
            else:
                sentences.append(
                    "A reliable anterior-posterior organization claim was not supported by the available structured evidence."
                )
        if pdr_symmetry is not None and pdr_symmetry >= 0.65:
            sentences.append("Posterior alpha symmetry had limited structured support.")

        sentences.append(
            "Eye-opening reactivity and formal sleep architecture remain nullable v1 slots unless protocol/state evidence supports them."
        )

        for finding_type in ["background_frequency", "background_slowing"]:
            finding = findings.get(finding_type)
            if finding and review_notes.get(finding.finding_id):
                sentences.append("Evidence caveat: " + review_notes[finding.finding_id][0])
                break

        return " ".join(sentences) if sentences else "No background-specific structured evidence was available."

    def _event_section_text(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
        review_notes: dict[str, List[str]],
        section_role: SectionRole | str,
    ) -> str:
        if not isinstance(section_role, SectionRole):
            section_role = SectionRouter().role_for_section(section_role)
        findings = self._findings_by_type(board)
        burden = self._exact(self._measurement_for_type(board, "epileptiform_event_candidate_burden", measurement_index))
        duration = self._range_text(findings.get("event_train_duration"))
        laterality = self._exact(
            self._measurement_for_type(board, "event_peak_laterality", measurement_index)
            or self._measurement_for_type(board, "event_laterality", measurement_index)
        )
        bifrontal = self._exact(self._measurement_for_type(board, "event_focality_bifrontal_spread", measurement_index))
        morphology = self._exact(self._measurement_for_type(board, "event_morphology_support", measurement_index))
        field = self._exact(self._measurement_for_type(board, "event_field_concentration", measurement_index))
        morphology_class = self._categorical_for_type(board, "event_morphology_class")
        localization_label = self._categorical_for_type(board, "event_peak_localization") or self._categorical_for_type(board, "event_clinical_localization")
        epileptiform_like = self._exact(self._measurement_for_type(board, "epileptiform_candidate_likelihood", measurement_index))
        seizure_like = self._exact(self._measurement_for_type(board, "electrographic_seizure_likelihood", measurement_index))
        no_definite_epileptiform = self._has_review_constraint(board, "definite epileptiform")

        sentences: List[str] = []
        if section_role == SectionRole.SEIZURES:
            if seizure_like is not None and seizure_like >= 0.70:
                sentences.append(
                    "Seizures: local evidence raised concern for possible seizure-like evolution, but a definitive seizure claim requires neurologist review of the source EEG."
                )
            else:
                sentences.append(
                    "Seizures: no electrographic seizures are confirmed by the current one-pass structured evidence."
                )
            sentences.append(
                "Transient candidate screens are not treated as seizure-specific evidence in this section."
            )
            return " ".join(sentences)

        if section_role == SectionRole.EVENTS_SEIZURES:
            sentences.append(
                "Events/seizures: no electrographic seizures are confirmed by the current one-pass structured evidence."
            )
            sentences.append(
                "Push-button event status and clinical event correlation were not recoverable from the structured context."
            )
            prefix = "Separate transient-candidate review"
        elif section_role == SectionRole.EPILEPTIFORM:
            prefix = "Epileptiform-candidate review"
            sentences.append(
                "Definite epileptiform discharges are not claimed unless morphology-specific and spatial evidence support that label."
            )
        else:
            prefix = "Event-candidate review"

        if burden is not None:
            sentences.append(f"{prefix} detected transient candidates; candidate burden is retained in provenance/debug output.")
        else:
            sentences.append(f"{prefix} did not produce a quantifiable candidate burden.")
        if duration:
            sentences.append("Candidate train duration is retained in provenance/debug output and is not surfaced as a clinical burst duration without morphology support.")
        if laterality is not None:
            if abs(laterality) < 0.2:
                sentences.append("The candidate distribution did not support a strong lateralized predominance.")
            elif laterality > 0:
                sentences.append("The candidate distribution showed a left-leaning laterality index.")
            else:
                sentences.append("The candidate distribution showed a right-leaning laterality index.")
        if (
            localization_label
            and localization_label != "unknown"
            and self._event_localization_surface_allowed(board, measurement_index, section_role)
        ):
            sentences.append(
                f"A peak-centered localization screen suggested {localization_label.replace('_', ' ')} involvement; this requires event waveform and channel-level review before clinical localization is claimed."
            )
        if bifrontal is not None and bifrontal >= 1.2:
            sentences.append("A bifrontal spread tendency was suggested by the configured spatial screen.")
        if morphology_class and morphology_class != "insufficient_morphology_evidence":
            sentences.append(
                f"A local morphology screen classified the finding as a {morphology_class.replace('_', ' ')}; this is not a definitive spike/sharp classifier."
            )
        elif morphology is not None or field is not None or epileptiform_like is not None:
            sentences.append("Morphology and field proxy values are retained in provenance/debug output, not final clinical prose.")
        if no_definite_epileptiform and section_role == SectionRole.EPILEPTIFORM:
            sentences.append(
                "Morphology-specific evidence is incomplete, so these findings should not be reported as definite epileptiform discharges without focused EEG review."
            )

        for finding_type in ["epileptiform_event_candidate_burden", "event_train_duration", "event_morphology_support"]:
            finding = findings.get(finding_type)
            if finding and review_notes.get(finding.finding_id):
                sentences.append("Evidence caveat: " + review_notes[finding.finding_id][0])
                break

        return " ".join(sentences)

    def _detail_section_text(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
        review_notes: dict[str, List[str]],
    ) -> str:
        findings = self._findings_by_type(board)
        sections = [
            self._background_section_text(board, measurement_index, review_notes),
            self._event_section_text(board, measurement_index, review_notes, SectionRole.DETAIL),
            self._protocol_sentence(findings),
        ]
        missing = self._review_impression_constraints(board)
        if missing:
            sections.append(" ".join(missing))
        return " ".join(x for x in sections if x)

    def _clinical_impression_text(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
    ) -> str:
        findings = self._findings_by_type(board)
        sentences: List[str] = []
        slowing = findings.get("background_slowing")
        event = findings.get("epileptiform_event_candidate_burden")
        beta = findings.get("excess_beta")
        morphology = findings.get("event_morphology_support")
        pdr = findings.get("background_pdr_frequency")
        seizure_like = findings.get("electrographic_seizure_likelihood")

        if pdr and pdr.assertion == StatusSemantic.PRESENT:
            pdr_txt = self._range_text(pdr) or self._format_quantitation(pdr.quantitation)
            sentences.append(f"Posterior rhythm candidate is supported near {pdr_txt}.")
        if slowing and slowing.assertion == StatusSemantic.PRESENT:
            sentences.append("Abnormal automated EEG review due to prominent background slowing features.")
        elif beta and beta.assertion == StatusSemantic.PRESENT:
            sentences.append("Automated review identified excess beta activity.")

        if event and event.assertion == StatusSemantic.PRESENT:
            event_sentence = "Event-like transient candidates were detected"
            if morphology and morphology.assertion == StatusSemantic.PRESENT:
                event_sentence += ", with local morphology-feature support"
            event_sentence += "."
            sentences.append(event_sentence)
        if seizure_like and seizure_like.assertion == StatusSemantic.ABSENT:
            sentences.append("No electrographic seizure is confirmed by the local seizure-likelihood screen.")

        if self._has_review_constraint(board, "definite epileptiform"):
            sentences.append(
                "Definite epileptiform or seizure conclusions are deferred because morphology-specific evidence is incomplete."
            )
        if not sentences:
            sentences.append("No clearly supported abnormal EEG impression was generated from the current structured evidence.")
        sentences.append("Findings are intended as assistive evidence and require neurologist review.")
        return " ".join(sentences)

    def _detail_sentence(
        self,
        finding: FindingObject,
        measurement: MeasurementValue | None,
        review_notes: List[str] | None = None,
    ) -> str:
        label = self._finding_label(finding, measurement)
        status = self._status_phrase(finding.assertion)
        qtxt = self._format_quantitation(finding.quantitation)
        if not self._clinical_quantitation_allowed(finding):
            qtxt = ""
        note = ""
        if review_notes:
            note = " Evidence review: " + " ".join(review_notes[:2])
        if qtxt:
            return f"{label}: {status} ({qtxt}).{note}"
        return f"{label}: {status}.{note}"

    def _build_impression(
        self,
        board: EvidenceBoard,
        measurement_index: dict[str, MeasurementValue],
    ) -> List[str]:
        out: List[str] = []
        abnormal_types = {f.finding_type for f in board.findings if self._is_abnormal_finding(f)}
        out.append("Structured evidence supports abnormal EEG features.")

        background = []
        event = []
        protocol = []
        for f in board.findings:
            if f.assertion != StatusSemantic.PRESENT:
                continue
            label = self._finding_label(f, self._first_measurement(f, measurement_index)).lower()
            qtxt = self._format_quantitation(f.quantitation)
            if not self._clinical_quantitation_allowed(f):
                qtxt = ""
            phrase = f"{label}"
            if qtxt:
                phrase += f" ({qtxt})"

            if f.finding_type.startswith("background_") or f.finding_type == "excess_beta":
                background.append(phrase)
            elif f.finding_type.startswith("event_") or f.finding_type.startswith("epileptiform_"):
                event.append(phrase)
            elif f.finding_type.startswith("protocol_"):
                protocol.append(phrase)

        if background:
            out.append("Background features: " + ", ".join(background[:4]) + ".")
        if event:
            out.append("Event-related features: " + ", ".join(event[:4]) + ".")
        if protocol:
            out.append("Protocol/context findings: " + ", ".join(protocol[:3]) + ".")

        if not (background or event or protocol):
            out.append("Key abnormal findings were detected but category summaries were limited.")

        out.append("Interpretation remains evidence-driven and should be reviewed with full EEG context.")
        return out

    def _review_notes_by_finding(self, board: EvidenceBoard) -> dict[str, List[str]]:
        notes: dict[str, List[str]] = {}
        for deliberation in board.deliberations:
            for weak in deliberation.weak_evidence:
                if weak.weakness_id == "weak_signal_spatial_provenance":
                    continue
                for fid in weak.linked_finding_ids:
                    notes.setdefault(fid, []).append(weak.recommendation)
            for do_not in deliberation.do_not_claim:
                for fid in do_not.linked_finding_ids:
                    notes.setdefault(fid, []).append(do_not.text)
            for constraint in deliberation.claim_constraints:
                if constraint.constraint_id == "constraint_no_focal_without_space":
                    continue
                for fid in constraint.linked_finding_ids:
                    notes.setdefault(fid, []).append(constraint.constraint)
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
                "Event-related conclusions are framed as candidates because morphology-specific evidence is incomplete."
            )
        if any("focal" in c.constraint.lower() or "lateralized" in c.constraint.lower() for c in constraints):
            out.append(
                "No focal or lateralized conclusion is made unless channel/region provenance supports it."
            )
        if missing_slots:
            slot_names = ", ".join(sorted({slot.slot_name for slot in missing_slots})[:4])
            out.append(f"Evidence gaps to address in focused review: {slot_names}.")
        return out
