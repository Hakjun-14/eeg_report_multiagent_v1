from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.finding import FindingObject
from eeg_report_multiagent.schemas.gt_suppression import GTAtomicClaim, GTClaimPipelineMatch, GTSuppressionAggregate, GTSuppressionAuditResult
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard

_NUMERIC_RE = re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*(?:[-–]\s*(?P<b>\d+(?:\.\d+)?))?\s*(?P<unit>hz|uv|µv|microvolt(?:s)?|sec(?:ond)?s?|s|%)", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ELECTRODE_RE = re.compile(r"\b(?:fpz|fp1|fp2|fz|f3|f4|f7|f8|cz|c3|c4|pz|p3|p4|o1|o2|t3|t4|t5|t6|a1|a2)(?:\s*/\s*(?:fpz|fp1|fp2|fz|f3|f4|f7|f8|cz|c3|c4|pz|p3|p4|o1|o2|t3|t4|t5|t6|a1|a2))*\b", re.I)

GT_SECTION_KEYS = ("EEG_section_llm_extractions", "EEG_sections")


class GTClaimExtractor:
    """Extract evaluation-only GT report claims into coarse clinical slots."""

    def extract_from_report_json(self, path: Path, case_id: str) -> list[GTAtomicClaim]:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        sections = self._extract_sections(payload)
        return self.extract_from_sections(sections, case_id=case_id)

    def extract_from_sections(self, sections: Mapping[str, str], *, case_id: str) -> list[GTAtomicClaim]:
        claims: list[GTAtomicClaim] = []
        seen: set[tuple[str, str, str, str]] = set()
        for section, text in sections.items():
            section_key = _norm(section)
            for sentence in _sentences(text):
                for claim_type, value, unit, state, topography, certainty in self._claims_from_sentence(section_key, sentence):
                    key = (claim_type, _norm(str(value)), unit or "", _norm(sentence))
                    if key in seen:
                        continue
                    seen.add(key)
                    claims.append(
                        GTAtomicClaim(
                            gt_claim_id=f"gt_{len(claims)+1:04d}_{claim_type}",
                            case_id=case_id,
                            section=section,
                            claim_type=claim_type,
                            normalized_value=value,
                            unit=unit,
                            state=state,
                            topography=topography,
                            certainty=certainty,
                            source_text=sentence,
                        )
                    )
        return claims

    def _extract_sections(self, payload: Mapping[str, Any]) -> dict[str, str]:
        eeg = payload.get("EEG_section_llm_extractions", payload)
        sections = eeg.get("EEG_sections", eeg) if isinstance(eeg, Mapping) else {}
        out: dict[str, str] = {}
        if isinstance(sections, Mapping):
            for name, value in sections.items():
                if isinstance(value, str) and value.strip():
                    out[str(name)] = value
                elif isinstance(value, Mapping):
                    text = value.get("section_text") or value.get("text") or value.get("content")
                    if isinstance(text, str) and text.strip():
                        out[str(name)] = text
        elif isinstance(sections, list):
            for idx, item in enumerate(sections):
                if not isinstance(item, Mapping):
                    continue
                name = item.get("section_name") or item.get("name") or f"section_{idx}"
                text = item.get("section_text") or item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    out[str(name)] = text
        return out

    def _claims_from_sentence(self, section_key: str, sentence: str) -> list[tuple[str, Any, str | None, str | None, dict[str, Any] | None, str]]:
        s = _norm(sentence)
        claims: list[tuple[str, Any, str | None, str | None, dict[str, Any] | None, str]] = []
        nums = _numeric_mentions(sentence)

        is_detail_like = any(key in section_key for key in ("detail", "description", "eeg description"))
        if "background" in section_key or is_detail_like:
            if "posterior dominant" in s or "pdr" in s:
                for num in nums:
                    if num["unit"] == "Hz":
                        claims.append(("pdr_frequency", num["value"], "Hz", _state_from_sentence(s), {"region": "posterior"}, "asserted"))
                    if num["unit"] == "uV":
                        claims.append(("background_amplitude", num["value"], "uV", _state_from_sentence(s), {"region": "posterior"}, "asserted"))
            if "symmetric" in s and ("posterior dominant" in s or "pdr" in s):
                claims.append(("pdr_symmetry", "symmetric", None, _state_from_sentence(s), {"region": "posterior"}, "asserted"))
            if "reactiv" in s:
                claims.append(("pdr_reactivity", "reactive", None, _state_from_sentence(s), {"region": "posterior"}, "asserted"))
            if "anterior-posterior" in s or "anterior posterior" in s or "organization" in s:
                claims.append(("background_organization", "organized", None, _state_from_sentence(s), None, "asserted"))
            if "slowing" in s:
                claims.append(("background_slowing", "present", None, _state_from_sentence(s), None, "asserted"))
            if "awake" in s or "wakefulness" in s:
                claims.append(("awake", "present", None, "awake", None, "asserted"))
            if "drows" in s:
                claims.append(("drowsy", "present", None, "drowsy", None, "asserted"))
            if "stage ii" in s:
                claims.append(("stage_ii_sleep", "present", None, "sleep", None, "asserted"))
            if any(word in s for word in ("vertex", "spindle", "k-complex", "k complex")):
                claims.append(("sleep_architecture", "present", None, "sleep", None, "asserted"))
            if "excess beta" in s or "beta" in s:
                claims.append(("excess_beta", "present", None, _state_from_sentence(s), None, "asserted"))
            if "hyperventilation" in s:
                status = "not_performed" if "not performed" in s else "performed"
                claims.append(("hyperventilation_status", status, None, None, None, "asserted"))
            if "photic" in s:
                status = "not_performed" if "not performed" in s else "performed"
                claims.append(("photic_status", status, None, None, None, "asserted"))
                if "driving" in s:
                    claims.append(("photic_response", "driving", None, None, None, "asserted"))
            if "no focal" in s or "no epileptiform" in s or "epileptiform abnormalities were seen" in s:
                claims.append(("epileptiform_absent", "absent", None, None, None, "asserted_absent"))
            if "no organized" in s and "seizure" in s:
                claims.append(("seizure_absent", "absent", None, None, None, "asserted_absent"))

        if any(key in section_key for key in ("epileptiform", "interictal")) or (is_detail_like and any(key in s for key in ("spike", "sharp", "epileptiform", "discharge"))):
            if "spike/wave" in s or "spike-wave" in s or "spike wave" in s:
                claims.append(("epileptiform_morphology_spike_wave", "spike_wave", None, _state_from_sentence(s), None, _certainty(s)))
            if "sharp" in s and "discharge" in s:
                claims.append(("epileptiform_morphology_sharp", "sharp", None, _state_from_sentence(s), None, _certainty(s)))
            if "generalized" in s and "discharge" in s:
                claims.append(("generalized_discharge", "present", None, _state_from_sentence(s), None, _certainty(s)))
            if "left" in s and "right" in s and (">" in s or "greater than" in s):
                claims.append(("localization_laterality", "left>right", None, _state_from_sentence(s), {"side": "left>right"}, _certainty(s)))
            if "left frontal" in s:
                claims.append(("localization_region", "left_frontal", None, _state_from_sentence(s), {"side": "left", "region": "frontal"}, _certainty(s)))
            elif "frontal" in s:
                claims.append(("localization_region", "frontal", None, _state_from_sentence(s), {"region": "frontal"}, _certainty(s)))
            electrodes = _electrodes(sentence)
            if electrodes:
                claims.append(("electrode_maxima", electrodes, None, _state_from_sentence(s), {"electrode_maxima": electrodes}, _certainty(s)))
            if "field" in s:
                claims.append(("field", "present", None, _state_from_sentence(s), None, _certainty(s)))
            if "during sleep" in s or "appear during sleep" in s or "seen during sleep" in s:
                claims.append(("event_state_sleep", "present", None, "sleep", None, _certainty(s)))
            if "irregular sleep architecture" in s:
                claims.append(("alternative_sleep_architecture", "possible", None, "sleep", None, "uncertain"))
            if "not definitively epileptiform" in s or "uncertain" in s or "unclear significance" in s or "uncertain significance" in s:
                claims.append(("uncertain_epileptiformity", "present", None, _state_from_sentence(s), None, "uncertain"))
            for num in nums:
                if num["unit"] == "uV":
                    claims.append(("event_amplitude", num["value"], "uV", _state_from_sentence(s), None, _certainty(s)))
                if num["unit"] == "sec":
                    claims.append(("event_duration", num["value"], "sec", _state_from_sentence(s), None, _certainty(s)))
                if num["unit"] == "Hz":
                    claims.append(("event_frequency", num["value"], "Hz", _state_from_sentence(s), None, _certainty(s)))

        if "event" in section_key or "seizure" in section_key or is_detail_like:
            if "push button" in s and "none" in s:
                claims.append(("push_button_absent", "absent", None, None, None, "asserted_absent"))
            if "seizures" in s and ("none" in s or "no seizure" in s):
                claims.append(("seizure_absent", "absent", None, None, None, "asserted_absent"))
            elif "electrographic seizure" in s and "absent" in s:
                claims.append(("seizure_absent", "absent", None, None, None, "asserted_absent"))
        return claims


class GTRequiredSuppressionAuditor:
    """Match GT-required atomic claims to saved pipeline objects and final prose."""

    def __init__(self) -> None:
        self.extractor = GTClaimExtractor()

    def audit_case(
        self,
        *,
        case_id: str,
        variant: str,
        gt_report_json: Path,
        evidence_board: EvidenceBoard,
        shared_evidence_board: SharedEvidenceBoard,
        atomic_claims: Sequence[AtomicClaimPlan],
        final_report: Mapping[str, str],
    ) -> GTSuppressionAuditResult:
        gt_claims = self.extractor.extract_from_report_json(gt_report_json, case_id=case_id)
        matches = [
            self.match_claim(
                claim,
                evidence_board.measurements,
                evidence_board.findings,
                shared_evidence_board.evidence_items,
                atomic_claims,
                final_report,
            )
            for claim in gt_claims
        ]
        total = len(matches) or 1
        surfaced = sum(1 for m in matches if m.match_stage == "surfaced")
        recovered = sum(1 for m in matches if m.match_stage in {"atomic_claim", "surfaced"})
        suppressed_ids = [m.gt_claim_id for m in matches if m.category == "gt_required_but_suppressed"]
        missing_ids = [m.gt_claim_id for m in matches if m.category == "gt_required_but_missing_from_evidence_extraction"]
        return GTSuppressionAuditResult(
            case_id=case_id,
            variant=variant,
            gt_claims=gt_claims,
            gt_claim_matches=matches,
            gt_required_suppressed_claims=suppressed_ids,
            gt_required_missing_evidence=missing_ids,
            gt_required_surface_rate=surfaced / total,
            gt_required_claim_recovery_rate=recovered / total,
            recommendations=self._case_recommendations(matches),
        )

    def match_claim(
        self,
        claim: GTAtomicClaim,
        measurements: Sequence[MeasurementValue],
        findings: Sequence[FindingObject],
        evidence_items: Sequence[EvidenceItem],
        atomic_claims: Sequence[AtomicClaimPlan],
        final_report: Mapping[str, str],
    ) -> GTClaimPipelineMatch:
        measurement_hits = [m for m in measurements if self._measurement_matches(claim, m)]
        finding_hits = [f for f in findings if self._finding_matches(claim, f)]
        evidence_hits = [e for e in evidence_items if self._evidence_matches(claim, e, measurement_hits, finding_hits)]
        evidence_ids = {e.evidence_id for e in evidence_hits}
        measurement_ids = {m.measurement_id for m in measurement_hits}
        finding_ids = {f.finding_id for f in finding_hits}
        claim_hits = [c for c in atomic_claims if self._atomic_claim_matches(claim, c, evidence_ids, measurement_ids, finding_ids)]
        surfaced_sentence = self._surface_sentence(claim, final_report)

        if surfaced_sentence:
            match_stage = "surfaced"
            suppression_stage = "none"
            category = "gt_required_surfaced"
        elif claim_hits:
            match_stage = "atomic_claim"
            suppression_stage = self._suppression_from_claims(claim_hits, evidence_hits, claim)
            category = self._category_from_stage(suppression_stage, claim, evidence_hits)
        elif evidence_hits:
            match_stage = "evidence_item"
            suppression_stage = "atomic_claim_not_generated"
            category = "gt_required_but_no_atomic_claim_generated"
        elif measurement_hits or finding_hits:
            match_stage = "measurement_only"
            suppression_stage = "not_converted_to_evidence_item"
            category = "gt_required_but_not_converted_to_evidence_item"
        else:
            match_stage = "no_measurement"
            suppression_stage = "none"
            category = "gt_required_but_missing_from_evidence_extraction"

        salvageability = self._salvageability(claim, evidence_hits, claim_hits, match_stage)
        return GTClaimPipelineMatch(
            gt_claim_id=claim.gt_claim_id,
            case_id=claim.case_id,
            matched_measurement_ids=[m.measurement_id for m in measurement_hits],
            matched_finding_ids=[f.finding_id for f in finding_hits],
            matched_evidence_ids=[e.evidence_id for e in evidence_hits],
            matched_atomic_claim_ids=[c.plan_id for c in claim_hits],
            surfaced_sentence=surfaced_sentence,
            match_stage=match_stage,
            suppression_stage=suppression_stage,
            suppression_reason=self._suppression_reason(claim, evidence_hits, claim_hits, match_stage, surfaced_sentence),
            category=category,
            salvageability=salvageability,
            rationale=self._rationale(claim, evidence_hits, claim_hits, match_stage, surfaced_sentence, salvageability),
        )

    def aggregate(self, audits: Sequence[GTSuppressionAuditResult], variant: str) -> GTSuppressionAggregate:
        matches = [match for audit in audits for match in audit.gt_claim_matches]
        total = len(matches) or 1
        measurement = sum(1 for m in matches if m.matched_measurement_ids or m.matched_finding_ids)
        evidence = sum(1 for m in matches if m.matched_evidence_ids)
        atomic = sum(1 for m in matches if m.matched_atomic_claim_ids)
        surfaced = sum(1 for m in matches if m.match_stage == "surfaced")
        suppressed = sum(1 for m in matches if m.match_stage in {"measurement_only", "evidence_item", "atomic_claim"})
        salvageable = sum(
            1
            for m in matches
            if m.match_stage in {"measurement_only", "evidence_item", "atomic_claim"}
            and m.salvageability in {"allow_candidate", "caveat_candidate"}
        )
        detector_gap = sum(1 for m in matches if m.match_stage == "no_measurement")
        adapter_gap = sum(1 for m in matches if m.suppression_stage == "not_converted_to_evidence_item")
        planner_gap = sum(1 for m in matches if m.suppression_stage == "atomic_claim_not_generated")
        policy_gap = sum(1 for m in matches if m.suppression_stage in {"atomic_claim_blocked", "surface_policy_rejected", "reportability_blocked"})
        metrics = {
            "GTClaimMeasurementAvailability": measurement / total,
            "GTClaimEvidenceItemAvailability": evidence / total,
            "GTClaimAtomicClaimAvailability": atomic / total,
            "GTClaimSurfaceRate": surfaced / total,
            "GTRequiredSuppressionRate": suppressed / total,
            "SalvageableGTClaimRate": salvageable / max(suppressed, 1),
            "DetectorGapRate": detector_gap / total,
            "AdapterGapRate": adapter_gap / total,
            "ClaimPlannerGapRate": planner_gap / total,
            "SurfacePolicyGapRate": policy_gap / total,
        }
        recommendation = _stage3_recommendation(metrics)
        return GTSuppressionAggregate(
            variant=variant,
            num_cases=len(audits),
            num_gt_claims=len(matches),
            metrics=metrics,
            category_counts=dict(Counter(m.category for m in matches)),
            suppression_stage_counts=dict(Counter(m.suppression_stage for m in matches)),
            salvageability_counts=dict(Counter(m.salvageability for m in matches)),
            stage3_recommendation=recommendation,
        )

    def _measurement_matches(self, claim: GTAtomicClaim, measurement: MeasurementValue) -> bool:
        text = _norm(" ".join([measurement.measurement_id, measurement.measurement_name, str(measurement.categorical_value or ""), str(measurement.metadata)]))
        identity = _norm(" ".join([measurement.measurement_id, measurement.measurement_name, str(measurement.categorical_value or "")]))
        if claim.claim_type == "event_amplitude":
            return "event_amplitude" in identity
        if _claim_is_seizure_absent(claim):
            return "seizure" in text and any(key in text for key in ("label", "specific", "absence", "absent", "none"))
        if claim.claim_type.startswith("epileptiform") and any(bad in text for bad in ("candidate_burden", "burden", "support_score", "likelihood")):
            return False
        if claim.claim_type.startswith("localization") or claim.claim_type in {"electrode_maxima", "field"}:
            if not any(key in identity for key in ("localization", "laterality", "field", "electrode", "maxima", "peak")):
                return False
            if any(bad in text for bad in ("ratio", "index", "concentration")) and not _has_space_like_value(measurement):
                return False
        if _target_keywords(claim.claim_type, text):
            if claim.unit and measurement.quantitation and measurement.quantitation.unit:
                return _unit_norm(measurement.quantitation.unit) == _unit_norm(claim.unit)
            return True
        return False

    def _finding_matches(self, claim: GTAtomicClaim, finding: FindingObject) -> bool:
        text = _norm(" ".join([finding.finding_id, finding.finding_type, finding.summary_label or "", " ".join(finding.tags)]))
        identity = _norm(" ".join([finding.finding_id, finding.finding_type, finding.summary_label or ""]))
        if claim.claim_type == "event_amplitude":
            return "event_amplitude" in text
        if _claim_is_seizure_absent(claim):
            return "seizure" in text and any(key in text for key in ("label", "specific", "absence", "absent", "none"))
        if claim.claim_type.startswith("epileptiform") and any(bad in text for bad in ("candidate_burden", "burden", "support_score", "likelihood")):
            return False
        if claim.claim_type.startswith("localization") or claim.claim_type in {"electrode_maxima", "field"}:
            if not any(key in identity for key in ("localization", "laterality", "field", "electrode", "maxima", "peak")):
                return False
            if any(bad in text for bad in ("ratio", "index", "concentration")) and not finding.provenance:
                return False
        return _target_keywords(claim.claim_type, text)

    def _evidence_matches(
        self,
        claim: GTAtomicClaim,
        evidence: EvidenceItem,
        measurement_hits: Sequence[MeasurementValue],
        finding_hits: Sequence[FindingObject],
    ) -> bool:
        if any(mid in evidence.measurement_ids for mid in [m.measurement_id for m in measurement_hits]):
            return True
        if any(fid in evidence.finding_ids for fid in [f.finding_id for f in finding_hits]):
            return True
        target = str(getattr(evidence.clinical_target, "value", evidence.clinical_target))
        text = _norm(" ".join([evidence.evidence_id, evidence.source_module, target, str(evidence.value), str(evidence.normalized_value), str(evidence.rationale), str(evidence.space_provenance)]))
        if claim.claim_type == "event_amplitude":
            return target in {ClinicalTarget.EVENT_CANDIDATE.value, ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value} and "amplitude" in text and _unit_norm(evidence.unit or "") == "uV"
        if _claim_is_pdr(claim):
            if target != ClinicalTarget.PDR.value:
                return False
            value = _evidence_numeric_value(evidence)
            if _unit_norm(evidence.unit or "") == "Hz" and value is not None and value < 6.0:
                return False
            return True
        if claim.claim_type == "background_amplitude":
            return target == ClinicalTarget.BACKGROUND_AMPLITUDE.value or ("amplitude" in text and _unit_norm(evidence.unit or "") == "uV")
        if _claim_is_seizure_absent(claim):
            return target == ClinicalTarget.SEIZURE_EVIDENCE.value and evidence.evidence_type in {EvidenceType.DIRECT, EvidenceType.METADATA, EvidenceType.DERIVED}
        if claim.claim_type in {"photic_status", "photic_response", "hyperventilation_status"}:
            return target == ClinicalTarget.PROTOCOL.value or _target_keywords(claim.claim_type, text)
        if claim.claim_type == "epileptiform_absent":
            return target == ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value and "absent" in text
        if claim.claim_type.startswith("epileptiform") or claim.claim_type in {"generalized_discharge", "focal_discharge"}:
            if target != ClinicalTarget.EPILEPTIFORM_MORPHOLOGY.value:
                return False
            return not _is_internal_score_or_burden(evidence)
        if claim.claim_type.startswith("localization") or claim.claim_type in {"electrode_maxima", "field"}:
            if target != ClinicalTarget.LOCALIZATION.value:
                return False
            return _space_supports_localization(evidence) or _target_keywords(claim.claim_type, text)
        if claim.claim_type in {"awake", "drowsy", "sleep", "stage_ii_sleep", "sleep_architecture", "event_state_sleep"}:
            return target == ClinicalTarget.STATE.value or "sleep" in text or "awake" in text or "drows" in text
        if claim.claim_type in {"push_button_absent"}:
            return target in {ClinicalTarget.PROTOCOL.value, ClinicalTarget.CONTEXT.value, ClinicalTarget.SEIZURE_EVIDENCE.value} and "push" in text
        return _target_keywords(claim.claim_type, text)

    def _atomic_claim_matches(
        self,
        claim: GTAtomicClaim,
        atomic_claim: AtomicClaimPlan,
        evidence_ids: set[str],
        measurement_ids: set[str],
        finding_ids: set[str],
    ) -> bool:
        if evidence_ids and any(eid in atomic_claim.evidence_ids for eid in evidence_ids):
            return True
        if measurement_ids and any(mid in atomic_claim.linked_measurement_ids for mid in measurement_ids):
            return True
        if finding_ids and any(fid in atomic_claim.linked_finding_ids for fid in finding_ids):
            return True
        text = _norm(" ".join([atomic_claim.plan_id, atomic_claim.claim_type, atomic_claim.proposed_text, atomic_claim.rationale or ""]))
        if _claim_is_pdr(claim):
            return False
        if claim.claim_type == "event_amplitude":
            return "event_amplitude" in text
        if _claim_is_seizure_absent(claim):
            return False
        return _target_keywords(claim.claim_type, text)

    def _surface_sentence(self, claim: GTAtomicClaim, final_report: Mapping[str, str]) -> str | None:
        for section_name, section_text in final_report.items():
            if not _section_compatible(claim, section_name):
                continue
            for sentence in _sentences(section_text):
                s = _norm(sentence)
                if _claim_is_seizure_absent(claim):
                    if "seizures" in s and "none" in s:
                        return sentence
                    continue
                if claim.claim_type == "push_button_absent":
                    if "push button" in s and "none" in s:
                        return sentence
                    continue
                if not _target_keywords(claim.claim_type, s):
                    continue
                if claim.unit:
                    if _numeric_overlap_claim_text(claim, sentence):
                        return sentence
                    continue
                return sentence
        return None

    def _suppression_from_claims(self, claims: Sequence[AtomicClaimPlan], evidence_items: Sequence[EvidenceItem], gt_claim: GTAtomicClaim) -> str:
        actions = {claim.surface_action for claim in claims}
        if any(_appropriately_blocked(gt_claim, item) for item in evidence_items):
            if not all(_appropriately_blocked(gt_claim, item) for item in evidence_items):
                return "reportability_blocked"
            return "appropriately_blocked"
        if ClaimSurfaceAction.BLOCK in actions:
            return "atomic_claim_blocked"
        if ClaimSurfaceAction.DEBUG_ONLY in actions:
            return "surface_policy_rejected"
        return "final_prose_missing"

    def _category_from_stage(self, suppression_stage: str, gt_claim: GTAtomicClaim, evidence_items: Sequence[EvidenceItem]) -> str:
        if suppression_stage == "appropriately_blocked" or any(_appropriately_blocked(gt_claim, item) for item in evidence_items):
            if not all(_appropriately_blocked(gt_claim, item) for item in evidence_items):
                return "gt_required_but_surfacepolicy_blocked"
            return "gt_required_but_appropriately_blocked_due_to_unsafe_support"
        if suppression_stage in {"atomic_claim_blocked", "surface_policy_rejected", "reportability_blocked"}:
            return "gt_required_but_surfacepolicy_blocked"
        return "gt_required_but_suppressed"

    def _salvageability(
        self,
        claim: GTAtomicClaim,
        evidence_items: Sequence[EvidenceItem],
        atomic_claims: Sequence[AtomicClaimPlan],
        match_stage: str,
    ) -> str:
        del atomic_claims
        if match_stage == "no_measurement":
            return "detector_gap"
        if not evidence_items and match_stage == "measurement_only":
            return "adapter_gap"
        if _claim_is_seizure_absent(claim):
            return "allow_candidate" if any(str(getattr(e.clinical_target, "value", e.clinical_target)) == ClinicalTarget.SEIZURE_EVIDENCE.value for e in evidence_items) else "detector_gap"
        if _claim_is_pdr(claim):
            for item in evidence_items:
                value = _evidence_numeric_value(item)
                if value is not None and 6.0 <= value <= 13.0:
                    if _space_supports_posterior(item):
                        return "caveat_candidate"
                    return "keep_blocked"
            return "detector_gap"
        if claim.claim_type in {"background_amplitude", "awake", "drowsy", "sleep", "stage_ii_sleep", "sleep_architecture", "push_button_absent"}:
            return "allow_candidate" if evidence_items else "detector_gap"
        if claim.claim_type.startswith("localization") or claim.claim_type in {"electrode_maxima", "field"}:
            return "caveat_candidate" if any(_space_supports_localization(e) and not _is_ratio_only(e) for e in evidence_items) else "keep_blocked"
        if claim.claim_type.startswith("epileptiform") or claim.claim_type in {"generalized_discharge", "focal_discharge"}:
            return "caveat_candidate" if any(not _is_internal_score_or_burden(e) for e in evidence_items) else "keep_blocked"
        if evidence_items:
            return "caveat_candidate"
        return "claim_planner_gap"

    def _suppression_reason(
        self,
        claim: GTAtomicClaim,
        evidence_items: Sequence[EvidenceItem],
        atomic_claims: Sequence[AtomicClaimPlan],
        match_stage: str,
        surfaced_sentence: str | None,
    ) -> str:
        if surfaced_sentence:
            return "surfaced in final prose"
        if match_stage == "no_measurement":
            return "no matching safe Measurement/Finding was found for this GT claim"
        if match_stage == "measurement_only":
            return "matching Measurement/Finding did not become an EvidenceItem"
        if match_stage == "evidence_item":
            return "matching EvidenceItem did not produce an AtomicClaimPlan"
        actions = Counter(str(getattr(c.surface_action, "value", c.surface_action)) for c in atomic_claims)
        if any(_appropriately_blocked(claim, e) for e in evidence_items):
            if not all(_appropriately_blocked(claim, e) for e in evidence_items):
                return f"AtomicClaimPlan exists but did not surface; actions={dict(actions)}, evidence_types={dict(Counter(str(getattr(e.evidence_type, 'value', e.evidence_type)) for e in evidence_items))}"
            return "upstream evidence exists but is unsafe or insufficient for this GT claim"
        return f"AtomicClaimPlan exists but did not surface; actions={dict(actions)}"

    def _rationale(
        self,
        claim: GTAtomicClaim,
        evidence_items: Sequence[EvidenceItem],
        atomic_claims: Sequence[AtomicClaimPlan],
        match_stage: str,
        surfaced_sentence: str | None,
        salvageability: str,
    ) -> str:
        if surfaced_sentence:
            return "GT claim appears in final prose."
        pieces = [f"match_stage={match_stage}", f"salvageability={salvageability}"]
        if evidence_items:
            pieces.append("evidence=" + ",".join(f"{e.evidence_id}:{getattr(e.evidence_type, 'value', e.evidence_type)}" for e in evidence_items[:5]))
        if atomic_claims:
            pieces.append("plans=" + ",".join(f"{c.plan_id}:{getattr(c.surface_action, 'value', c.surface_action)}" for c in atomic_claims[:5]))
        if _claim_is_seizure_absent(claim):
            pieces.append("seizure absence requires seizure-specific evidence or validated metadata; event candidates do not count")
        return "; ".join(pieces)

    def _case_recommendations(self, matches: Sequence[GTClaimPipelineMatch]) -> list[str]:
        counts = Counter(match.suppression_stage for match in matches)
        recs: list[str] = []
        if counts.get("none", 0) and any(match.match_stage == "no_measurement" for match in matches):
            recs.append("Stage 3A: improve extraction for GT claims with no safe Measurement/Finding.")
        if counts.get("not_converted_to_evidence_item", 0):
            recs.append("Stage 3B: repair Measurement/Finding to EvidenceItem conversion for GT claims.")
        if counts.get("atomic_claim_not_generated", 0):
            recs.append("Stage 3D: refine claim planner for GT-matched EvidenceItems.")
        if counts.get("atomic_claim_blocked", 0) or counts.get("surface_policy_rejected", 0) or counts.get("reportability_blocked", 0):
            recs.append("Stage 3C/3E: calibrate reportability/SurfacePolicy only for GT-matched, safe evidence.")
        return recs or ["No dominant GT-required suppression bottleneck detected."]


def _numeric_mentions(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in _NUMERIC_RE.finditer(text):
        unit = _unit_norm(m.group("unit"))
        a = float(m.group("a"))
        b = float(m.group("b")) if m.group("b") else None
        value: Any = {"lower": min(a, b), "upper": max(a, b)} if b is not None else a
        out.append({"value": value, "unit": unit, "raw_text": m.group(0)})
    return out


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(str(text)) if part.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("µ", "u").lower()).strip()


def _unit_norm(unit: str) -> str:
    u = _norm(unit)
    if u in {"hz"}:
        return "Hz"
    if u in {"uv", "microvolt", "microvolts"}:
        return "uV"
    if u in {"sec", "second", "seconds", "s"}:
        return "sec"
    if u in {"%", "percent"}:
        return "percent"
    return unit


def _electrodes(text: str) -> list[str]:
    found: list[str] = []
    for match in _ELECTRODE_RE.finditer(text):
        for token in re.split(r"\s*/\s*", match.group(0)):
            token = token.upper().replace("FP", "Fp")
            if token and token not in found:
                found.append(token)
    return found


def _state_from_sentence(s: str) -> str | None:
    if "stage ii" in s or "sleep" in s or "spindle" in s or "k-complex" in s:
        return "sleep"
    if "drows" in s:
        return "drowsy"
    if "awake" in s or "wakefulness" in s:
        return "awake"
    return None


def _certainty(s: str) -> str:
    if any(key in s for key in ("uncertain", "not definitively", "could", "may", "possible")):
        return "uncertain"
    return "asserted"


def _claim_is_pdr(claim: GTAtomicClaim) -> bool:
    return claim.claim_type == "pdr_frequency"


def _claim_is_seizure_absent(claim: GTAtomicClaim) -> bool:
    return claim.claim_type == "seizure_absent"


def _target_keywords(claim_type: str, text: str) -> bool:
    groups = {
        "pdr_frequency": ("pdr", "posterior dominant", "posterior alpha"),
        "background_amplitude": ("background_amplitude", "amplitude"),
        "pdr_symmetry": ("symmetry", "symmetric"),
        "pdr_reactivity": ("reactivity", "reactive"),
        "background_organization": ("organization", "anterior"),
        "background_slowing": ("slowing", "delta", "theta"),
        "excess_beta": ("excess_beta", "beta"),
        "awake": ("awake", "wakefulness"),
        "drowsy": ("drows",),
        "sleep": ("sleep",),
        "stage_ii_sleep": ("stage", "sleep", "spindle", "k_complex", "k-complex", "vertex"),
        "sleep_architecture": ("spindle", "k_complex", "k-complex", "vertex", "sleep_architecture"),
        "event_state_sleep": ("sleep",),
        "epileptiform_morphology_spike_wave": ("spike_wave", "spike/wave", "spike-wave", "spike wave"),
        "epileptiform_morphology_sharp": ("sharp", "morphology"),
        "generalized_discharge": ("generalized", "discharge"),
        "focal_discharge": ("focal", "discharge"),
        "uncertain_epileptiformity": ("uncertain", "not definitively", "uncertainty"),
        "alternative_sleep_architecture": ("sleep architecture", "irregular", "alternative"),
        "event_amplitude": ("event_amplitude", "amplitude"),
        "event_duration": ("duration", "train"),
        "event_frequency": ("frequency", "hz"),
        "localization_laterality": ("laterality", "left", "right"),
        "localization_region": ("localization", "region", "frontal", "temporal", "posterior"),
        "electrode_maxima": ("electrode", "maxima", "channel", "f3", "f7", "t3", "t5"),
        "field": ("field",),
        "push_button_absent": ("push", "button"),
        "photic_status": ("photic",),
        "photic_response": ("photic", "driving"),
        "hyperventilation_status": ("hyperventilation", "hv"),
        "epileptiform_absent": ("epileptiform", "absent", "no epileptiform"),
    }
    keys = groups.get(claim_type, (claim_type,))
    return any(key in text for key in keys)


def _section_compatible(claim: GTAtomicClaim, section_name: str) -> bool:
    section = _norm(section_name)
    if claim.claim_type.startswith("pdr") or claim.claim_type.startswith("background") or claim.claim_type in {"awake", "drowsy", "stage_ii_sleep", "sleep_architecture", "excess_beta"}:
        return "background" in section or "detail" in section or "eeg description" in section
    if claim.claim_type.startswith("epileptiform") or claim.claim_type in {"generalized_discharge", "focal_discharge", "event_amplitude", "event_duration", "event_frequency", "localization_laterality", "localization_region", "electrode_maxima", "field", "event_state_sleep", "uncertain_epileptiformity", "alternative_sleep_architecture"}:
        return "epileptiform" in section or "interictal" in section or "detail" in section
    if claim.claim_type in {"seizure_absent", "push_button_absent"}:
        return "seizure" in section or "event" in section
    if claim.claim_type in {"photic_status", "photic_response", "hyperventilation_status"}:
        return "background" in section or "detail" in section or "description" in section
    return True


def _numeric_overlap_claim_text(claim: GTAtomicClaim, text: str) -> bool:
    for num in _numeric_mentions(text):
        if claim.unit and _unit_norm(claim.unit) != num["unit"]:
            continue
        if _range_overlap(claim.normalized_value, num["value"]):
            return True
    return False


def _range_overlap(a: Any, b: Any) -> bool:
    lo_a, hi_a = _to_range(a)
    lo_b, hi_b = _to_range(b)
    if lo_a is None or lo_b is None:
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _to_range(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, Mapping):
        lo = value.get("lower", value.get("min"))
        hi = value.get("upper", value.get("max"))
        if lo is not None and hi is not None:
            return float(lo), float(hi)
        exact = value.get("exact")
        if exact is not None:
            return float(exact), float(exact)
    if isinstance(value, (int, float)):
        return float(value), float(value)
    return None, None


def _evidence_numeric_value(evidence: EvidenceItem) -> float | None:
    for value in (evidence.normalized_value, evidence.value):
        lo, hi = _to_range(value)
        if lo is not None and hi is not None:
            return (lo + hi) / 2.0
    return None


def _has_space_like_value(measurement: MeasurementValue) -> bool:
    text = _norm(" ".join([str(measurement.categorical_value or ""), str(measurement.metadata)]))
    return any(key in text for key in ("left", "right", "frontal", "temporal", "f3", "f7", "t3", "t5"))


def _space_supports_posterior(evidence: EvidenceItem) -> bool:
    space = evidence.space_provenance or {}
    text = _norm(str(space))
    return any(key in text for key in ("posterior", "occipital", "o1", "o2"))


def _space_supports_localization(evidence: EvidenceItem) -> bool:
    space = evidence.space_provenance or {}
    text = _norm(" ".join([str(space), str(evidence.value), str(evidence.normalized_value), evidence.rationale or ""]))
    has_region = any(key in text for key in ("left", "right", "frontal", "temporal", "posterior", "generalized", "bifrontal"))
    has_channel = bool(space.get("channels") or space.get("electrode_maxima") or _electrodes(text))
    return has_region and has_channel


def _is_ratio_only(evidence: EvidenceItem) -> bool:
    text = _norm(" ".join([evidence.evidence_id, str(evidence.value), str(evidence.normalized_value), evidence.rationale or ""]))
    return any(key in text for key in ("ratio", "index", "concentration")) and not _space_supports_localization(evidence)


def _is_internal_score_or_burden(evidence: EvidenceItem) -> bool:
    text = _norm(" ".join([evidence.evidence_id, str(evidence.value), str(evidence.normalized_value), evidence.rationale or "", str(evidence.debug_payload)]))
    return any(key in text for key in ("candidate burden", "candidate_burden", "support score", "support_score", "likelihood", "ratio", "index", "field concentration")) or evidence.evidence_type == EvidenceType.DEBUG


def _appropriately_blocked(claim: GTAtomicClaim, evidence: EvidenceItem) -> bool:
    if _claim_is_pdr(claim):
        value = _evidence_numeric_value(evidence)
        text = _norm(" ".join([evidence.evidence_id, evidence.rationale or ""]))
        return (_unit_norm(evidence.unit or "") == "Hz" and value is not None and value < 6.0) or "boundary" in text
    if _claim_is_seizure_absent(claim):
        return str(getattr(evidence.clinical_target, "value", evidence.clinical_target)) != ClinicalTarget.SEIZURE_EVIDENCE.value
    if claim.claim_type.startswith("epileptiform") or claim.claim_type in {"generalized_discharge", "focal_discharge"}:
        return _is_internal_score_or_burden(evidence)
    if claim.claim_type.startswith("localization") or claim.claim_type in {"electrode_maxima", "field"}:
        return _is_ratio_only(evidence)
    return False


def _stage3_recommendation(metrics: Mapping[str, float]) -> str:
    if metrics.get("DetectorGapRate", 0.0) >= 0.45:
        return "Stage 3A: evidence extraction improvement is needed before broad reportability calibration."
    if metrics.get("AdapterGapRate", 0.0) >= 0.20:
        return "Stage 3B: repair Measurement/Finding to EvidenceItem conversion."
    if metrics.get("ClaimPlannerGapRate", 0.0) >= 0.20:
        return "Stage 3D: refine AtomicClaimPlan generation for GT-matched EvidenceItems."
    if metrics.get("SurfacePolicyGapRate", 0.0) >= 0.20:
        return "Stage 3C/3E: reportability and SurfacePolicy calibration is justified by GT-required suppression."
    return "Proceed cautiously: GT-aligned audit does not show one dominant downstream bottleneck."
