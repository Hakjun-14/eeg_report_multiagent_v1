from __future__ import annotations

from typing import Dict, List, Set

from eeg_report_multiagent.schemas.section_contract import (
    SectionRole,
    SectionSlotRequirement,
    TargetReportSection,
    TargetSectionContract,
)


class SectionRouter:
    """Map CELM target sections to evidence requirements and safe synthesis policy."""

    def role_for_section(self, section_name: str) -> SectionRole:
        name = section_name.strip().upper()
        if name == "EEG DESCRIPTION/DETAILS":
            return SectionRole.DETAIL
        if name == "BACKGROUND ACTIVITY":
            return SectionRole.BACKGROUND
        if "EPLEPTIFORM" in name or "EPILEPTIFORM" in name:
            return SectionRole.EPILEPTIFORM
        if name == "EVENTS/SEIZURES":
            return SectionRole.EVENTS_SEIZURES
        if name == "SEIZURES":
            return SectionRole.SEIZURES
        if name == "SLEEP":
            return SectionRole.SLEEP
        if name == "IMPRESSION/INTERPRETATION":
            return SectionRole.IMPRESSION
        return SectionRole.OTHER

    def required_slots(self, role: SectionRole) -> List[SectionSlotRequirement]:
        if role == SectionRole.BACKGROUND:
            return [
                self._slot("background_frequency_or_pdr", ["background_pdr_frequency", "background_frequency"], "background rhythm/frequency evidence"),
                self._slot("background_amplitude", ["background_amplitude_range"], "background amplitude quantitation"),
                self._slot("slowing", ["background_slowing"], "slowing screen"),
                self._slot(
                    "organization_reactivity",
                    ["background_ap_organization", "background_reactivity"],
                    "PDR organization/reactivity is clinically central but nullable in v1",
                    nullable=True,
                ),
            ]
        if role == SectionRole.DETAIL:
            return [
                self._slot("background_frequency_or_pdr", ["background_pdr_frequency", "background_frequency"], "detail section needs background rhythm evidence"),
                self._slot("background_amplitude", ["background_amplitude_range"], "detail section needs amplitude evidence"),
                self._slot("event_candidate_burden", ["epileptiform_event_candidate_burden"], "detail section needs event screen evidence"),
                self._slot("protocol_context", ["protocol_state_awake"], "detail section should include recoverable state/context", nullable=True),
            ]
        if role == SectionRole.EPILEPTIFORM:
            return [
                self._slot("event_candidate_burden", ["epileptiform_event_candidate_burden", "epileptiform_candidate_likelihood"], "epileptiform screen burden"),
                self._slot("event_duration", ["event_train_duration"], "burst/train duration estimate"),
                self._slot("event_space", ["event_laterality", "event_focality_bifrontal_spread"], "laterality/localization evidence"),
                self._slot(
                    "morphology_confirmation",
                    ["event_morphology_support"],
                    "definite epileptiform language needs morphology-specific support",
                    nullable=True,
                ),
            ]
        if role == SectionRole.EVENTS_SEIZURES:
            return [
                self._slot("event_candidate_burden", ["epileptiform_event_candidate_burden"], "event screen burden"),
                self._slot("event_duration", ["event_train_duration"], "event duration estimate"),
                self._slot("seizure_specific_evidence", ["electrographic_seizure_likelihood"], "seizure-specific likelihood screen", nullable=True),
            ]
        if role == SectionRole.SEIZURES:
            return [
                self._slot("seizure_specific_evidence", ["electrographic_seizure_likelihood"], "seizure-specific likelihood screen", nullable=True),
            ]
        if role == SectionRole.SLEEP:
            return [
                self._slot("sleep_state", [], "sleep staging is nullable in v1", nullable=True),
            ]
        if role == SectionRole.IMPRESSION:
            return [
                self._slot("summary_abnormalities", ["background_slowing", "epileptiform_event_candidate_burden"], "impression-level abnormality summary"),
            ]
        return []

    def optional_slots(self, role: SectionRole) -> List[SectionSlotRequirement]:
        if role in {SectionRole.BACKGROUND, SectionRole.DETAIL}:
            return [
                self._slot("activation_status", ["protocol_photic_stimulation_status", "protocol_hyperventilation_status"], "activation status if recoverable", required=False, nullable=True),
                self._slot("ekg_video", ["protocol_ekg_availability", "protocol_video_availability"], "recording adjunct availability if recoverable", required=False, nullable=True),
            ]
        if role in {SectionRole.EPILEPTIFORM, SectionRole.EVENTS_SEIZURES}:
            return [
                self._slot("field_concentration", ["event_field_concentration"], "focused field concentration", required=False, nullable=True),
            ]
        return []

    def generation_policy(self, role: SectionRole) -> str:
        return {
            SectionRole.DETAIL: "Combine background, event-candidate, and protocol/context evidence; do not invent unavailable normal EEG slots.",
            SectionRole.BACKGROUND: "Report background measurements and explicitly mark PDR/organization/reactivity as unavailable when not measured.",
            SectionRole.EPILEPTIFORM: "Separate transient candidates from definite epileptiform discharges unless morphology support is present.",
            SectionRole.EVENTS_SEIZURES: "Separate event candidates from seizure confirmation; do not route candidate burden as confirmed seizures.",
            SectionRole.SEIZURES: "Do not claim electrographic seizures from transient candidate tools alone.",
            SectionRole.SLEEP: "Report sleep only if state/context evidence supports it.",
            SectionRole.IMPRESSION: "Summarize only evidence-board-supported claims.",
            SectionRole.OTHER: "Use general detail synthesis from evidence board only.",
        }[role]

    def build_contract(
        self,
        report_id: str,
        target_section_names_raw: List[str],
        target_section_names_standardized: List[str],
        eval_only_reference_json_path: str | None = None,
    ) -> TargetSectionContract:
        sections: List[TargetReportSection] = []
        for raw, standardized in zip(target_section_names_raw, target_section_names_standardized):
            role = self.role_for_section(standardized)
            sections.append(
                TargetReportSection(
                    raw_name=raw,
                    standardized_name=standardized,
                    role=role,
                    required_slots=self.required_slots(role),
                    optional_slots=self.optional_slots(role),
                    generation_policy=self.generation_policy(role),
                )
            )
        return TargetSectionContract(
            contract_id=f"section_contract_{report_id}",
            report_id=report_id,
            target_sections=sections,
            eval_only_reference_json_path=eval_only_reference_json_path,
            notes=[
                "Target section names are benchmark/inference contract metadata.",
                "GT section text is not included in this contract and must not be used for inference.",
            ],
        )

    def covered_slots(self, finding_types: Set[str], requirements: List[SectionSlotRequirement]) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for req in requirements:
            out[req.slot_name] = bool(req.finding_types and any(ft in finding_types for ft in req.finding_types))
        return out

    def _slot(
        self,
        name: str,
        finding_types: List[str],
        reason: str,
        required: bool = True,
        nullable: bool = False,
    ) -> SectionSlotRequirement:
        return SectionSlotRequirement(
            slot_name=name,
            finding_types=finding_types,
            required=required,
            nullable=nullable,
            reason=reason,
        )
