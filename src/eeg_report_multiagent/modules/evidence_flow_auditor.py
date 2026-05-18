from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.evidence_flow import EvidenceFlowAggregate, EvidenceFlowAuditResult, SlotFlowRecord
from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction
from eeg_report_multiagent.schemas.shared_evidence import EvidenceItem, EvidenceType, SharedEvidenceBoard


SUPPRESSION_REASONS = {
    "missing_measurement",
    "missing_finding",
    "not_converted_to_evidence_item",
    "proxy_or_debug_only",
    "missing_required_provenance",
    "blocked_by_clinical_rule",
    "no_atomic_claim_generated",
    "atomic_claim_blocked",
    "surface_policy_rejected",
    "wrong_section",
    "missing_state_support",
    "missing_protocol_support",
    "missing_morphology_support",
    "missing_localization_support",
    "missing_seizure_specific_evidence",
    "numeric_not_reportable",
    "internal_score_suppressed",
}


@dataclass(frozen=True)
class SlotSpec:
    name: str
    section_name: str
    measurement_keywords: tuple[str, ...]
    finding_keywords: tuple[str, ...]
    clinical_targets: tuple[str, ...]
    claim_keywords: tuple[str, ...]
    required_support: tuple[str, ...] = ()


CLINICAL_SLOT_SPECS: tuple[SlotSpec, ...] = (
    SlotSpec("pdr_frequency", "background", ("pdr_candidate_frequency",), ("background_pdr_frequency", "pdr_candidate_frequency"), ("pdr",), ("pdr_candidate_frequency", "background_pdr_frequency", "posterior alpha"), ("state", "topography")),
    SlotSpec("pdr_posterior_predominance_topography", "background", ("posterior_anterior", "pdr"), ("background_pdr_topography", "pdr"), ("pdr",), ("topography", "posterior"), ("topography",)),
    SlotSpec("pdr_symmetry", "background", ("symmetry", "pdr"), ("background_pdr_symmetry", "symmetry"), ("pdr",), ("symmetry", "pdr"), ("topography",)),
    SlotSpec("pdr_reactivity", "background", ("reactivity",), ("background_reactivity", "reactivity"), ("pdr", "state"), ("reactivity",), ("state",)),
    SlotSpec("background_amplitude", "background", ("amplitude",), ("background_amplitude",), ("background_amplitude",), ("amplitude",)),
    SlotSpec("background_slowing", "background", ("slowing", "dominant_frequency", "delta", "theta"), ("background_slowing", "background_frequency"), ("background_slowing",), ("slowing",), ("state",)),
    SlotSpec("excess_beta", "background", ("beta",), ("excess_beta",), ("excess_beta",), ("beta",)),
    SlotSpec("awake", "state_protocol", ("awake",), ("awake", "protocol_state_awake"), ("state",), ("awake",)),
    SlotSpec("drowsiness", "state_protocol", ("drowsy", "drowsiness"), ("drowsy", "drowsiness", "protocol_state_drowsy"), ("state",), ("drowsy", "drowsiness")),
    SlotSpec("sleep_stage_ii_architecture", "state_protocol", ("sleep", "spindle", "k_complex", "vertex"), ("sleep", "stage_ii", "spindle", "k_complex", "vertex"), ("state",), ("sleep", "stage")),
    SlotSpec("photic_status_response", "state_protocol", ("photic",), ("photic",), ("protocol",), ("photic",), ("protocol",)),
    SlotSpec("hyperventilation_status_response", "state_protocol", ("hyperventilation",), ("hyperventilation",), ("protocol",), ("hyperventilation",), ("protocol",)),
    SlotSpec("epileptiform_morphology", "epileptiform", ("morphology", "spike", "sharp", "spike_wave"), ("epileptiform", "morphology", "spike", "sharp", "spike_wave"), ("epileptiform_morphology",), ("epileptiform", "morphology", "spike", "sharp"), ("morphology",)),
    SlotSpec("spike_sharp_spike_wave_support", "epileptiform", ("spike", "sharp", "spike_wave", "morphology"), ("spike", "sharp", "spike_wave", "epileptiform"), ("epileptiform_morphology",), ("spike", "sharp", "spike-wave"), ("morphology",)),
    SlotSpec("event_candidate_burden", "epileptiform", ("candidate_burden", "burden"), ("candidate_burden", "epileptiform_event_candidate_burden"), ("event_candidate",), ("candidate", "burden")),
    SlotSpec("event_frequency", "epileptiform", ("event_frequency",), ("event_frequency",), ("event_candidate", "epileptiform_morphology"), ("event_frequency",), ("morphology",)),
    SlotSpec("event_amplitude", "epileptiform", ("event_amplitude",), ("event_amplitude",), ("event_candidate", "epileptiform_morphology"), ("event_amplitude",), ("morphology",)),
    SlotSpec("event_duration", "epileptiform", ("train_duration", "duration"), ("event_train_duration", "duration"), ("event_candidate",), ("duration", "train")),
    SlotSpec("uncertainty_caveat", "epileptiform", ("uncertainty", "candidate"), ("uncertainty", "candidate"), ("uncertainty", "event_candidate"), ("caveat", "candidate", "uncertain")),
    SlotSpec("localization_laterality", "localization", ("laterality", "localization", "field", "bifrontal"), ("laterality", "localization", "bifrontal"), ("localization",), ("localization", "lateral", "left", "right", "bifrontal"), ("localization", "morphology")),
    SlotSpec("electrode_maxima", "localization", ("electrode", "maxima", "channel"), ("electrode", "maxima", "channel"), ("localization",), ("electrode", "channel", "maxima"), ("localization",)),
    SlotSpec("region", "localization", ("region", "frontal", "temporal", "posterior"), ("region", "frontal", "temporal", "posterior"), ("localization",), ("frontal", "temporal", "posterior"), ("localization",)),
    SlotSpec("field_support", "localization", ("field", "concentration"), ("field", "concentration"), ("localization",), ("field",), ("localization", "morphology")),
    SlotSpec("seizure_specific_evidence", "seizure", ("seizure",), ("seizure",), ("seizure_evidence",), ("seizure",), ("seizure",)),
    SlotSpec("seizure_absence", "seizure", ("seizure",), ("seizure",), ("seizure_evidence",), ("no seizure", "seizures:"), ("seizure",)),
    SlotSpec("push_button_event_metadata", "seizure", ("push_button", "push button"), ("push_button", "push button"), ("seizure_evidence", "protocol", "context"), ("push_button", "push button")),
)


class EvidenceFlowAuditor:
    """Trace clinical slots across Measurement/Finding -> EvidenceItem -> Claim -> final prose."""

    def audit_case(
        self,
        *,
        case_id: str,
        variant: str,
        evidence_board: EvidenceBoard,
        shared_evidence_board: SharedEvidenceBoard | None,
        atomic_claims: Sequence[AtomicClaimPlan],
        final_report: Mapping[str, str] | list[dict[str, str]],
    ) -> EvidenceFlowAuditResult:
        shared_board = shared_evidence_board or evidence_board.ensure_shared_evidence_board()
        sections = self._normalize_final_report(final_report)
        slot_records = [
            self.audit_slot(
                spec.name,
                evidence_board.measurements,
                evidence_board.findings,
                shared_board,
                atomic_claims,
                sections,
                case_id=case_id,
                spec=spec,
            )
            for spec in CLINICAL_SLOT_SPECS
        ]
        reason_counts = Counter(reason for record in slot_records for reason in record.suppression_reasons)
        useful_examples = [
            {
                "case_id": case_id,
                "clinical_slot": record.clinical_slot,
                "evidence_ids": record.evidence_ids,
                "suppression_reasons": record.suppression_reasons,
                "notes": record.notes,
            }
            for record in slot_records
            if record.useful_but_suppressed
        ]
        surfaced = [record.clinical_slot for record in slot_records if record.surfaced_in_final_prose]
        missing = [
            record.clinical_slot
            for record in slot_records
            if not record.measurement_exists and not record.finding_exists and not record.evidence_item_exists
        ]
        diagnosis = self._case_diagnosis(slot_records)
        return EvidenceFlowAuditResult(
            case_id=case_id,
            variant=variant,
            slot_records=slot_records,
            suppression_reason_counts=dict(reason_counts),
            useful_suppressed_evidence=useful_examples,
            surfaced_slots=surfaced,
            missing_slots=missing,
            case_diagnosis=diagnosis,
        )

    def audit_slot(
        self,
        slot_name: str,
        measurements: Sequence[MeasurementValue],
        findings: Sequence[Finding],
        evidence_board: SharedEvidenceBoard,
        atomic_claims: Sequence[AtomicClaimPlan],
        final_report: Mapping[str, str],
        *,
        case_id: str = "",
        spec: SlotSpec | None = None,
    ) -> SlotFlowRecord:
        slot_spec = spec or self._spec_by_name(slot_name)
        measurement_hits = [m for m in measurements if self._matches_any(m.measurement_id, m.measurement_name, keywords=slot_spec.measurement_keywords)]
        finding_hits = [f for f in findings if self._matches_any(f.finding_id, f.finding_type, *(f.measurement_ids or []), keywords=slot_spec.finding_keywords)]
        evidence_hits = [
            item
            for item in evidence_board.evidence_items
            if self._evidence_matches_slot(item, slot_spec, measurement_hits, finding_hits)
        ]
        evidence_ids = [item.evidence_id for item in evidence_hits]
        claim_hits = [
            claim
            for claim in atomic_claims
            if self._claim_matches_slot(claim, slot_spec, evidence_ids, measurement_hits, finding_hits)
        ]
        surfaced, sentence = self.trace_claim_to_surface([claim.plan_id for claim in claim_hits], claim_hits, final_report)
        record = SlotFlowRecord(
            case_id=case_id,
            section_name=slot_spec.section_name,
            clinical_slot=slot_spec.name,
            measurement_exists=bool(measurement_hits),
            finding_exists=bool(finding_hits),
            evidence_item_exists=bool(evidence_hits),
            measurement_ids=[m.measurement_id for m in measurement_hits],
            finding_ids=[f.finding_id for f in finding_hits],
            evidence_ids=evidence_ids,
            evidence_type_counts=dict(Counter(str(getattr(item.evidence_type, "value", item.evidence_type)) for item in evidence_hits)),
            reportability_counts=dict(Counter(str(getattr(item.reportability, "value", item.reportability)) for item in evidence_hits)),
            atomic_claim_exists=bool(claim_hits),
            atomic_claim_ids=[claim.plan_id for claim in claim_hits],
            surface_action_counts=dict(Counter(str(getattr(claim.surface_action, "value", claim.surface_action)) for claim in claim_hits)),
            surfaced_in_final_prose=surfaced,
            final_sentence=sentence,
        )
        record.suppression_reasons = self.infer_suppression_reason(record, evidence_hits, claim_hits, slot_spec)
        record.useful_but_suppressed = self._useful_but_suppressed(record, evidence_hits)
        record.notes = self._slot_note(record)
        return record

    def trace_measurement_to_evidence(self, measurement_id: str, evidence_board: SharedEvidenceBoard) -> list[EvidenceItem]:
        return [item for item in evidence_board.evidence_items if measurement_id in item.measurement_ids]

    def trace_evidence_to_claim(self, evidence_id: str, atomic_claims: Iterable[AtomicClaimPlan]) -> list[AtomicClaimPlan]:
        return [claim for claim in atomic_claims if evidence_id in claim.evidence_ids]

    def trace_claim_to_surface(
        self,
        claim_ids: Sequence[str],
        claims: Sequence[AtomicClaimPlan],
        final_report: Mapping[str, str],
    ) -> tuple[bool, str | None]:
        del claim_ids
        sentences = self._report_sentences(final_report)
        for claim in claims:
            if claim.surface_action not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                continue
            claim_norm = self._norm(claim.proposed_text)
            if not claim_norm:
                continue
            for sentence in sentences:
                sent_norm = self._norm(sentence)
                if claim_norm in sent_norm or sent_norm in claim_norm or self._token_overlap(claim_norm, sent_norm) >= 0.55:
                    return True, sentence
        return False, None

    def infer_suppression_reason(
        self,
        record: SlotFlowRecord,
        evidence_items: Sequence[EvidenceItem] | None = None,
        atomic_claims: Sequence[AtomicClaimPlan] | None = None,
        spec: SlotSpec | None = None,
    ) -> List[str]:
        if record.surfaced_in_final_prose:
            return []
        reasons: list[str] = []
        if not record.measurement_exists:
            reasons.append("missing_measurement")
        if not record.finding_exists:
            reasons.append("missing_finding")
        if (record.measurement_exists or record.finding_exists) and not record.evidence_item_exists:
            reasons.append("not_converted_to_evidence_item")
        if record.evidence_item_exists:
            reportable = record.reportability_counts
            evidence_type = record.evidence_type_counts
            if reportable.get("debug_only", 0) or evidence_type.get("debug", 0):
                reasons.append("internal_score_suppressed")
            if reportable.get("debug_only", 0) or evidence_type.get("proxy", 0):
                reasons.append("proxy_or_debug_only")
            if not (reportable.get("allow", 0) or reportable.get("caveat", 0)):
                reasons.append("numeric_not_reportable")
        if record.evidence_item_exists and not record.atomic_claim_exists:
            reasons.append("no_atomic_claim_generated")
        if record.atomic_claim_exists:
            if record.surface_action_counts.get("block", 0):
                reasons.append("atomic_claim_blocked")
                reasons.append("blocked_by_clinical_rule")
            if record.surface_action_counts.get("debug_only", 0):
                reasons.append("surface_policy_rejected")
        if spec is not None:
            support_reasons = self._support_gap_reasons(spec, evidence_items or [], atomic_claims or [])
            reasons.extend(support_reasons)
        return sorted({reason for reason in reasons if reason in SUPPRESSION_REASONS})

    def aggregate_selected50(self, case_audits: Sequence[EvidenceFlowAuditResult], variant: str = "Our_EvidenceGated_v1") -> EvidenceFlowAggregate:
        num_cases = len(case_audits)
        per_slot_records: dict[str, list[SlotFlowRecord]] = defaultdict(list)
        for audit in case_audits:
            for record in audit.slot_records:
                per_slot_records[record.clinical_slot].append(record)
        availability: dict[str, dict[str, float]] = {}
        surface_rate: dict[str, float] = {}
        reason_counts: dict[str, dict[str, int]] = {}
        examples: list[dict] = []
        for slot, records in sorted(per_slot_records.items()):
            denom = max(len(records), 1)
            availability[slot] = {
                "measurement_rate": sum(r.measurement_exists for r in records) / denom,
                "finding_rate": sum(r.finding_exists for r in records) / denom,
                "evidence_item_rate": sum(r.evidence_item_exists for r in records) / denom,
                "claim_rate": sum(r.atomic_claim_exists for r in records) / denom,
                "surface_rate": sum(r.surfaced_in_final_prose for r in records) / denom,
                "useful_suppressed_rate": sum(r.useful_but_suppressed for r in records) / denom,
            }
            surface_rate[slot] = availability[slot]["surface_rate"]
            counter = Counter(reason for record in records for reason in record.suppression_reasons)
            reason_counts[slot] = dict(counter)
            for record in records:
                if record.useful_but_suppressed:
                    examples.append({
                        "case_id": record.case_id,
                        "clinical_slot": slot,
                        "evidence_ids": record.evidence_ids[:8],
                        "suppression_reasons": record.suppression_reasons,
                        "notes": record.notes,
                    })
        return EvidenceFlowAggregate(
            variant=variant,
            num_cases=num_cases,
            per_slot_availability=availability,
            per_slot_surface_rate=surface_rate,
            per_slot_suppression_reason_counts=reason_counts,
            useful_suppressed_top_examples=examples[:20],
            aggregate_recommendation_for_stage3=self._recommend_stage3(case_audits),
        )

    def _spec_by_name(self, slot_name: str) -> SlotSpec:
        for spec in CLINICAL_SLOT_SPECS:
            if spec.name == slot_name:
                return spec
        raise KeyError(slot_name)

    def _evidence_matches_slot(
        self,
        item: EvidenceItem,
        spec: SlotSpec,
        measurement_hits: Sequence[MeasurementValue],
        finding_hits: Sequence[Finding],
    ) -> bool:
        if self._is_reviewer_evidence(item) and spec.name != "uncertainty_caveat":
            return False
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        if target in spec.clinical_targets:
            item_text = " ".join([
                item.evidence_id,
                str(item.value),
                str(item.normalized_value),
                str(item.debug_payload),
                " ".join(item.measurement_ids),
                " ".join(item.finding_ids),
            ])
            if spec.name in {"awake", "drowsiness", "sleep_stage_ii_architecture", "photic_status_response", "hyperventilation_status_response"}:
                return self._contains_keyword(item_text, spec.measurement_keywords + spec.finding_keywords + spec.claim_keywords)
            if target in {"event_candidate", "localization", "epileptiform_morphology", "pdr", "background_amplitude", "background_slowing", "excess_beta", "seizure_evidence"}:
                return self._contains_keyword(item_text, spec.measurement_keywords + spec.finding_keywords + spec.claim_keywords) or bool(
                    set(item.measurement_ids) & {m.measurement_id for m in measurement_hits}
                ) or bool(set(item.finding_ids) & {f.finding_id for f in finding_hits})
        measurement_ids = {m.measurement_id for m in measurement_hits}
        finding_ids = {f.finding_id for f in finding_hits}
        return bool(set(item.measurement_ids) & measurement_ids) or bool(set(item.finding_ids) & finding_ids)

    def _claim_matches_slot(
        self,
        claim: AtomicClaimPlan,
        spec: SlotSpec,
        evidence_ids: Sequence[str],
        measurement_hits: Sequence[MeasurementValue],
        finding_hits: Sequence[Finding],
    ) -> bool:
        if set(claim.evidence_ids) & set(evidence_ids):
            return True
        if set(claim.linked_measurement_ids) & {m.measurement_id for m in measurement_hits}:
            return True
        if set(claim.linked_finding_ids) & {f.finding_id for f in finding_hits}:
            return True
        return self._matches_any(claim.plan_id, claim.claim_type, claim.proposed_text, keywords=spec.claim_keywords + spec.finding_keywords)

    def _is_reviewer_evidence(self, item: EvidenceItem) -> bool:
        return item.evidence_id.startswith("ev_review_") or str(getattr(item.evidence_type, "value", item.evidence_type)) == "llm_assisted"

    def _support_gap_reasons(
        self,
        spec: SlotSpec,
        evidence_items: Sequence[EvidenceItem],
        atomic_claims: Sequence[AtomicClaimPlan],
    ) -> list[str]:
        del atomic_claims
        reasons: list[str] = []
        if "state" in spec.required_support and not self._has_non_debug_target(evidence_items, "state"):
            reasons.append("missing_state_support")
        if "protocol" in spec.required_support and not self._has_non_debug_target(evidence_items, "protocol"):
            reasons.append("missing_protocol_support")
        if "morphology" in spec.required_support and not self._has_non_debug_target(evidence_items, "epileptiform_morphology"):
            reasons.append("missing_morphology_support")
        if "localization" in spec.required_support and not self._has_space_provenance(evidence_items):
            reasons.append("missing_localization_support")
        if "topography" in spec.required_support and not self._has_space_provenance(evidence_items):
            reasons.append("missing_localization_support")
        if "seizure" in spec.required_support and not self._has_non_debug_target(evidence_items, "seizure_evidence"):
            reasons.append("missing_seizure_specific_evidence")
        return reasons

    def _useful_but_suppressed(self, record: SlotFlowRecord, evidence_items: Sequence[EvidenceItem]) -> bool:
        if record.surfaced_in_final_prose or not evidence_items:
            return False
        for item in evidence_items:
            evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
            if evidence_type == "debug":
                continue
            if evidence_type == "proxy" and not self._strong_proxy_provenance(item):
                continue
            if self._internal_debug_like(item):
                continue
            if self._has_provenance_or_clinical_value(item):
                return True
        return False

    def _strong_proxy_provenance(self, item: EvidenceItem) -> bool:
        space = item.space_provenance or {}
        time = item.time_provenance or {}
        has_space = bool(space.get("channels") or space.get("region") or space.get("side") or space.get("electrode_maxima"))
        has_time = bool(time.get("window_id") is not None or time.get("start_sec") is not None or time.get("epoch_id") is not None)
        return has_space and has_time and str(getattr(item.clinical_target, "value", item.clinical_target)) in {"pdr", "localization", "epileptiform_morphology"}

    def _has_provenance_or_clinical_value(self, item: EvidenceItem) -> bool:
        has_time = bool(item.time_provenance)
        has_space = bool(item.space_provenance)
        has_value = item.value is not None and item.unit not in {"score", "ratio", "likelihood"}
        has_target = str(getattr(item.clinical_target, "value", item.clinical_target)) not in {"unknown", "uncertainty"}
        return has_target and (has_time or has_space or has_value)

    def _internal_debug_like(self, item: EvidenceItem) -> bool:
        text = " ".join([item.evidence_id, str(item.value), str(item.unit or ""), str(item.debug_payload)]).lower()
        return any(
            needle in text
            for needle in [
                "support score",
                "likelihood",
                "candidate_burden",
                "burden_ratio",
                "field_concentration",
                "laterality_index",
                "train_duration",
                "slowing_score",
                "beta_ratio",
            ]
        )

    def _has_non_debug_target(self, evidence_items: Sequence[EvidenceItem], target: str) -> bool:
        return any(
            str(getattr(item.clinical_target, "value", item.clinical_target)) == target
            and str(getattr(item.evidence_type, "value", item.evidence_type)) != "debug"
            for item in evidence_items
        )

    def _has_space_provenance(self, evidence_items: Sequence[EvidenceItem]) -> bool:
        for item in evidence_items:
            space = item.space_provenance or {}
            if space.get("channels") or space.get("region") or space.get("side") or space.get("electrode_maxima"):
                return True
        return False

    def _slot_note(self, record: SlotFlowRecord) -> str:
        if record.surfaced_in_final_prose:
            return "slot surfaced in final prose"
        if record.useful_but_suppressed:
            return "typed evidence appears potentially useful but did not surface"
        if record.evidence_item_exists and not record.atomic_claim_exists:
            return "evidence exists but no atomic claim was generated"
        if record.atomic_claim_exists:
            return "atomic claim exists but did not reach final prose"
        if record.measurement_exists or record.finding_exists:
            return "measurement/finding exists but did not become report-surface evidence"
        return "no slot-specific measurement/finding/evidence found"

    def _case_diagnosis(self, records: Sequence[SlotFlowRecord]) -> str:
        total = len(records)
        measurement_rate = sum(r.measurement_exists or r.finding_exists for r in records) / max(total, 1)
        evidence_rate = sum(r.evidence_item_exists for r in records) / max(total, 1)
        claim_rate = sum(r.atomic_claim_exists for r in records) / max(total, 1)
        surface_rate = sum(r.surfaced_in_final_prose for r in records) / max(total, 1)
        useful_count = sum(r.useful_but_suppressed for r in records)
        if evidence_rate < 0.25:
            bottleneck = "A. evidence absence / tool limitation"
        elif claim_rate < evidence_rate * 0.5:
            bottleneck = "C. AtomicClaimPlan failure"
        elif surface_rate < claim_rate * 0.5:
            bottleneck = "D. SurfacePolicy over-suppression or conservative reportability"
        elif useful_count:
            bottleneck = "B/C. EvidenceItem classification and claim planning need calibration"
        else:
            bottleneck = "F. section rendering limitation"
        return (
            f"measurement_or_finding_rate={measurement_rate:.2f}, evidence_item_rate={evidence_rate:.2f}, "
            f"claim_rate={claim_rate:.2f}, surface_rate={surface_rate:.2f}, "
            f"useful_but_suppressed={useful_count}; bottleneck={bottleneck}."
        )

    def _recommend_stage3(self, case_audits: Sequence[EvidenceFlowAuditResult]) -> str:
        records = [record for audit in case_audits for record in audit.slot_records]
        if not records:
            return "Stage 3A = evidence extraction improvement; no records were available."
        measurement_rate = sum(r.measurement_exists or r.finding_exists for r in records) / len(records)
        evidence_rate = sum(r.evidence_item_exists for r in records) / len(records)
        claim_rate = sum(r.atomic_claim_exists for r in records) / len(records)
        surface_rate = sum(r.surfaced_in_final_prose for r in records) / len(records)
        useful_rate = sum(r.useful_but_suppressed for r in records) / len(records)
        morphology_weak = self._slot_surface_rate(records, "epileptiform_morphology") == 0.0
        localization_weak = self._slot_surface_rate(records, "localization_laterality") == 0.0
        if measurement_rate < 0.35:
            return "Stage 3A = evidence extraction improvement."
        if evidence_rate < measurement_rate * 0.75:
            return "Stage 3B = adapter and EvidenceItem conversion repair."
        if useful_rate > 0.10:
            return "Stage 3C = reportability classification and evidence weighting."
        if claim_rate < evidence_rate * 0.50:
            return "Stage 3D = AtomicClaimPlan refinement."
        if surface_rate < claim_rate * 0.50:
            return "Stage 3E = SurfacePolicy calibration."
        if morphology_weak or localization_weak:
            return "Stage 3G = local EEG encoder or improved signal modules for morphology/state/localization."
        return "Stage 3F = section-specific rendering improvement."

    def _slot_surface_rate(self, records: Sequence[SlotFlowRecord], slot: str) -> float:
        sub = [record for record in records if record.clinical_slot == slot]
        return sum(record.surfaced_in_final_prose for record in sub) / max(len(sub), 1)

    def _normalize_final_report(self, final_report: Mapping[str, str] | list[dict[str, str]]) -> dict[str, str]:
        if isinstance(final_report, Mapping):
            if "report_sections" in final_report and isinstance(final_report["report_sections"], list):  # type: ignore[index]
                return self._normalize_final_report(final_report["report_sections"])  # type: ignore[arg-type]
            return {str(k): str(v or "") for k, v in final_report.items()}
        return {str(item.get("section_name", "")): str(item.get("section_text", "")) for item in final_report}

    def _report_sentences(self, final_report: Mapping[str, str]) -> list[str]:
        out: list[str] = []
        for text in final_report.values():
            out.extend([part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()])
        return out

    def _matches_any(self, *texts: str, keywords: Sequence[str]) -> bool:
        haystack = " ".join(str(text or "") for text in texts).lower()
        return self._contains_keyword(haystack, keywords)

    def _contains_keyword(self, text: str, keywords: Sequence[str]) -> bool:
        low = text.lower()
        return any(keyword.lower() in low for keyword in keywords)

    def _norm(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. -]+", " ", text.lower())).strip()

    def _token_overlap(self, a: str, b: str) -> float:
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(len(a_tokens), 1)
