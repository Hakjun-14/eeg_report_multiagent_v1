from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List

from eeg_report_multiagent.llm import OpenAIReportSynthesisAdapter
from eeg_report_multiagent.modules.clinical_reference_registry import clinical_reference_payloads
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.modules.surface_policy import SurfacePolicy
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction


@dataclass(frozen=True)
class LLMReportSynthesisResult:
    section_texts: Dict[str, str]
    trace: Dict[str, Any]


class EvidenceBoardLLMReportSynthesizer:
    """Method D: surface-policy-gated LLM report synthesis.

    The LLM receives only AtomicClaimPlan entries that were already allowed or
    caveated by SurfacePolicy. It never receives raw EEG arrays, pkl paths,
    GT/reference report text, full measurement payloads, or debug/proxy scores.
    """

    synthesizer_name = "evidence_board_llm_report_synthesizer"
    synthesis_version = "D_v2_surface_policy_gated"

    def __init__(
        self,
        adapter: OpenAIReportSynthesisAdapter | None = None,
        report_synthesizer: ReportSynthesizer | None = None,
        surface_policy: SurfacePolicy | None = None,
    ) -> None:
        self.adapter = adapter or OpenAIReportSynthesisAdapter()
        self.surface_policy = surface_policy or SurfacePolicy()
        self.report_synthesizer = report_synthesizer or ReportSynthesizer(surface_policy=self.surface_policy)
        self.section_router = SectionRouter()

    def synthesize_celm_sections(
        self,
        board: EvidenceBoard,
        target_section_names: List[str],
        clinical_context: Dict[str, Any] | None = None,
        claim_plan_override: List[AtomicClaimPlan] | None = None,
    ) -> LLMReportSynthesisResult:
        payload = self._build_payload(board, target_section_names, clinical_context or {}, claim_plan_override)
        model_output = self.adapter.synthesize(payload)
        render_coverage = self._render_coverage_trace(
            model_output,
            [item["claim_id"] for item in payload.get("claim_render_checklist", [])],
        )
        section_texts = self._validate_and_align_sections(
            model_output,
            target_section_names,
            payload["surface_payload_summary"].get("surface_allowed_or_caveated_claim_plans_by_section", {}),
        )
        trace = {
            "synthesizer_name": self.synthesizer_name,
            "synthesis_version": self.synthesis_version,
            "model_name": self.adapter.model,
            "raw_eeg_used": bool(model_output.get("raw_eeg_used")),
            "gt_report_used": bool(model_output.get("gt_report_used")),
            "target_section_names": target_section_names,
            "global_limitations": model_output.get("global_limitations", []),
            "rendered_claim_ids": model_output.get("rendered_claim_ids", []),
            "omitted_claims": model_output.get("omitted_claims", []),
            "render_coverage": render_coverage,
            "claim_render_checklist": payload.get("claim_render_checklist", []),
            "model_response_id": model_output.get("_response_id"),
            "model_report_sections": model_output.get("report_sections", []),
            "privacy_contract": payload["privacy_contract"],
            "surface_payload_summary": payload["surface_payload_summary"],
            "claim_plan_source": payload["claim_plan_source"],
        }
        if trace["raw_eeg_used"] or trace["gt_report_used"]:
            raise ValueError("D synthesis violated input contract")
        return LLMReportSynthesisResult(section_texts=section_texts, trace=trace)

    def _build_payload(
        self,
        board: EvidenceBoard,
        target_section_names: List[str],
        clinical_context: Dict[str, Any],
        claim_plan_override: List[AtomicClaimPlan] | None = None,
    ) -> Dict[str, Any]:
        claim_plans = claim_plan_override if claim_plan_override is not None else self.report_synthesizer.build_atomic_claim_plan(board)
        claim_plan_source = "artifact_atomic_claim_plan" if claim_plan_override is not None else "rule_based_fallback"
        shared_board = board.ensure_shared_evidence_board()
        surface_decisions = self.report_synthesizer.build_surface_decisions(
            claim_plans,
            shared_board,
        )
        decision_by_claim = {decision.claim_id: decision for decision in surface_decisions}
        surface_plans = [
            plan
            for plan in claim_plans
            if (
                decision_by_claim.get(plan.plan_id, self.surface_policy.decide(plan)).surface_action
                in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}
            )
            and not self.surface_policy.contains_forbidden_surface_text(plan.proposed_text)
        ]
        surface_claim_ids = {plan.plan_id for plan in surface_plans}
        surface_decision_payloads = [
            self._surface_decision_payload(decision)
            for decision in surface_decisions
            if decision.claim_id in surface_claim_ids
        ]
        surface_decision_by_claim = {item.get("claim_id"): item for item in surface_decision_payloads}
        atomic_claim_payloads = [
            self._claim_plan_payload(plan, shared_board, surface_decision_by_claim.get(plan.plan_id))
            for plan in surface_plans
        ]
        return {
            "session_id": board.session_id,
            "claim_plan_source": claim_plan_source,
            "target_section_names": target_section_names,
            "section_descriptions": self._section_descriptions(target_section_names),
            "clinical_context": self._compact_clinical_context(clinical_context),
            "atomic_claim_plans": atomic_claim_payloads,
            "claim_render_checklist": self._claim_render_checklist(
                atomic_claim_payloads,
                surface_decision_payloads,
                target_section_names,
            ),
            "surface_decisions": surface_decision_payloads,
            "surface_payload_summary": {
                "total_atomic_claim_plans": len(claim_plans),
                "surface_allowed_or_caveated_claim_plans": len(surface_plans),
                "surface_allowed_or_caveated_claim_plans_by_section": self._surface_plan_counts_by_section(
                    surface_plans,
                    surface_decisions,
                    target_section_names,
                ),
                "blocked_or_debug_only_claim_plans": len(claim_plans) - len(surface_plans),
                "audit_only_review_record_count": self._audit_record_count(board),
            },
            "style_policy": {
                "tone": "formal_clinical_eeg_report",
                "allowed_claims": "verbalize only atomic_claim_plans supplied in this payload",
                "linked_evidence_use": (
                    "Use linked_reportable_evidence only to preserve clinically reportable values, units, "
                    "and provenance for the supplied atomic claim. Do not create new claims from evidence alone."
                ),
                "uncertainty_handling": "preserve caveated wording from the atomic claim plan and do not upgrade certainty",
                "section_behavior": "generate each requested section once and do not add extra sections",
                "debug_surface_separation": (
                    "Do not mention internal detector scores, proxy labels, reviewer raw text, or measurements "
                    "that are not already translated into an atomic claim plan."
                ),
            },
            "forbidden_inputs": [
                "raw_eeg_arrays",
                "processed_pkl_payloads",
                "reference_gt_report_text",
                "full_measurement_payloads",
                "full_evidence_item_payloads",
                "unsafe_evidence_item_fields",
                "raw_evidence_reviewer_text",
                "debug_or_proxy_score_payloads",
                "unbounded_external_tools",
            ],
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
                "contains_source_pkl_paths": False,
                "contains_full_measurements": False,
                "contains_full_evidence_items": False,
                "contains_debug_scores": False,
            },
        }

    def _compact_clinical_context(self, clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Keep report-relevant context while preventing report LLM context overflow."""
        banned_keys = {
            "report_json_path_eval_only",
            "gt_report_json_path_eval_only",
            "reference_report",
            "gt_report",
            "report_text",
            "detail_text",
            "impression_text",
        }

        def compact(value: Any, depth: int = 0) -> Any:
            if depth > 3:
                return None
            if isinstance(value, str):
                return value[:2500]
            if isinstance(value, (int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                return [compact(item, depth + 1) for item in value[:20]]
            if isinstance(value, dict):
                out: Dict[str, Any] = {}
                for key, raw_value in value.items():
                    key_str = str(key)
                    if key_str in banned_keys or key_str.endswith("_eval_only"):
                        continue
                    out[key_str] = compact(raw_value, depth + 1)
                return out
            return str(value)[:500]

        compacted = compact(clinical_context)
        return compacted if isinstance(compacted, dict) else {}

    def _claim_render_checklist(
        self,
        atomic_claim_payloads: List[Dict[str, Any]],
        surface_decision_payloads: List[Dict[str, Any]],
        target_section_names: List[str],
    ) -> List[Dict[str, Any]]:
        decision_by_claim = {item.get("claim_id"): item for item in surface_decision_payloads}
        checklist: List[Dict[str, Any]] = []
        for payload in atomic_claim_payloads:
            claim_id = str(payload.get("plan_id") or "")
            decision = decision_by_claim.get(claim_id, {})
            checklist.append(
                {
                    "claim_id": claim_id,
                    "claim_type": payload.get("claim_type"),
                    "surface_action": decision.get("surface_action") or payload.get("surface_action"),
                    "target_section_names": self._candidate_section_names(
                        payload.get("allowed_sections") or decision.get("allowed_sections") or [],
                        target_section_names,
                    ),
                    "required_text_basis": payload.get("proposed_text"),
                    "must_render_values": payload.get("must_render_values") or [],
                    "numeric_claims": payload.get("numeric_claims") or [],
                    "evidence_ids": payload.get("evidence_ids") or [],
                    "clinical_reference_ids": payload.get("clinical_reference_ids") or decision.get("clinical_reference_ids") or [],
                    "clinical_reference_rules": [
                        item.get("short_rule")
                        for item in payload.get("clinical_references", [])
                        if item.get("short_rule")
                    ],
                    "rationale": payload.get("rationale"),
                }
            )
        return checklist

    def _candidate_section_names(self, allowed_sections: List[str], target_section_names: List[str]) -> List[str]:
        if not allowed_sections:
            return list(target_section_names)
        allowed = {str(item) for item in allowed_sections}
        out = [
            section_name
            for section_name in target_section_names
            if self.section_router.role_for_section(section_name).value in allowed
        ]
        return out or list(target_section_names)

    def _render_coverage_trace(self, model_output: Dict[str, Any], expected_claim_ids: List[str]) -> Dict[str, Any]:
        expected = [claim_id for claim_id in expected_claim_ids if claim_id]
        rendered = [
            str(claim_id)
            for claim_id in model_output.get("rendered_claim_ids", [])
            if str(claim_id) in set(expected)
        ]
        omitted_raw = model_output.get("omitted_claims", []) or []
        omitted_claim_ids = [
            str(item.get("claim_id"))
            for item in omitted_raw
            if isinstance(item, dict) and str(item.get("claim_id")) in set(expected)
        ]
        accounted = set(rendered) | set(omitted_claim_ids)
        unaccounted = [claim_id for claim_id in expected if claim_id not in accounted]
        return {
            "expected_claim_ids": expected,
            "rendered_claim_ids": rendered,
            "omitted_claim_ids": omitted_claim_ids,
            "unaccounted_claim_ids": unaccounted,
            "rendered_claim_count": len(set(rendered)),
            "expected_claim_count": len(expected),
            "rendered_claim_rate": (len(set(rendered)) / len(expected)) if expected else 1.0,
        }

    def synthesize_evidence_direct_sections(
        self,
        board: EvidenceBoard,
        target_section_names: List[str],
        clinical_context: Dict[str, Any] | None = None,
        evidence_ids: List[str] | None = None,
        evidence_selection_mode: str = "all_safe",
        payload_mode: str = "evidence_view",
    ) -> LLMReportSynthesisResult:
        """Diagnostic ablation: synthesize report sections directly from EvidenceItems."""
        base_payload = self._build_evidence_direct_payload(
            board=board,
            target_section_names=target_section_names,
            clinical_context=clinical_context or {},
            evidence_ids=evidence_ids,
            evidence_selection_mode=evidence_selection_mode,
        )
        payload = (
            self._build_slot_checklist_payload(base_payload)
            if payload_mode == "slot_checklist"
            else base_payload
        )
        model_output = self.adapter.synthesize_from_evidence_view(payload)
        section_texts = self._validate_and_align_sections(model_output, target_section_names)
        section_texts = {
            section_name: self._strip_evidence_id_mentions(section_text)
            for section_name, section_text in section_texts.items()
        }
        trace = {
            "synthesizer_name": self.synthesizer_name,
            "synthesis_version": (
                "slot_checklist_diagnostic_v1"
                if payload_mode == "slot_checklist"
                else "evidence_direct_diagnostic_v1"
            ),
            "model_name": self.adapter.model,
            "raw_eeg_used": bool(model_output.get("raw_eeg_used")),
            "gt_report_used": bool(model_output.get("gt_report_used")),
            "target_section_names": target_section_names,
            "global_limitations": model_output.get("global_limitations", []),
            "model_response_id": model_output.get("_response_id"),
            "model_report_sections": model_output.get("report_sections", []),
            "privacy_contract": payload["privacy_contract"],
            "evidence_view_summary": payload["evidence_view_summary"],
            "evidence_selection_mode": evidence_selection_mode,
            "payload_mode": payload_mode,
        }
        if trace["raw_eeg_used"] or trace["gt_report_used"]:
            raise ValueError("Evidence-direct synthesis violated input contract")
        return LLMReportSynthesisResult(section_texts=section_texts, trace=trace)

    def _strip_evidence_id_mentions(self, text: str) -> str:
        """Keep evidence IDs in trace JSON only, never in clinical prose."""
        cleaned = re.sub(
            r"\s*\((?:supported|caveated)?\s*evidence IDs?:\s*[^)]*\)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*evidence IDs?:\s*ev[-_a-zA-Z0-9., ]+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _build_evidence_direct_payload(
        self,
        board: EvidenceBoard,
        target_section_names: List[str],
        clinical_context: Dict[str, Any],
        evidence_ids: List[str] | None,
        evidence_selection_mode: str,
    ) -> Dict[str, Any]:
        shared_board = board.ensure_shared_evidence_board()
        selected_ids = set(evidence_ids or [])
        evidence_for_report: List[Dict[str, Any]] = []
        skipped_evidence_ids: List[str] = []
        for item in shared_board.list_evidence():
            if selected_ids and item.evidence_id not in selected_ids:
                continue
            payload = self._surface_safe_evidence_payload(item)
            if payload and self._has_surface_safe_content(payload):
                evidence_for_report.append(payload)
            else:
                skipped_evidence_ids.append(item.evidence_id)
        evidence_for_report.sort(key=lambda item: (str(item.get("clinical_target") or ""), str(item.get("evidence_id") or "")))
        return {
            "session_id": board.session_id,
            "diagnostic_mode": "evidence_direct_report_synthesis",
            "evidence_selection_mode": evidence_selection_mode,
            "target_section_names": target_section_names,
            "section_descriptions": self._section_descriptions(target_section_names),
            "clinical_context": clinical_context,
            "evidence_for_report": evidence_for_report,
            "evidence_view_summary": {
                "total_shared_evidence_items": len(shared_board.list_evidence()),
                "requested_evidence_ids": len(selected_ids),
                "surface_safe_evidence_count": len(evidence_for_report),
                "skipped_evidence_count": len(skipped_evidence_ids),
                "skipped_evidence_ids": skipped_evidence_ids[:200],
            },
            "style_policy": {
                "tone": "formal_clinical_eeg_report",
                "allowed_claims": "verbalize only evidence_for_report entries supplied in this diagnostic payload",
                "numeric_policy": "copy numeric values only from evidence_for_report values with evidence IDs",
                "uncertainty_handling": "preserve uncertainty; do not upgrade caveated evidence to definite abnormality",
                "section_behavior": "generate each requested section once and do not add extra sections",
            },
            "forbidden_surface_terms": [
                "candidate burden",
                "support score",
                "likelihood score",
                "field concentration ratio",
                "laterality index",
                "longest candidate train",
                "missing_slots",
                "values_preview",
            ],
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
                "contains_source_pkl_paths": False,
                "contains_full_measurements": False,
                "contains_full_evidence_items": False,
                "contains_debug_scores": False,
            },
        }

    def _build_slot_checklist_payload(self, evidence_payload: Dict[str, Any]) -> Dict[str, Any]:
        checklist = self._section_slot_checklist(evidence_payload.get("evidence_for_report", []))
        out = dict(evidence_payload)
        out["diagnostic_mode"] = "slot_checklist_report_synthesis"
        out["section_slot_checklist"] = checklist
        out["composition_policy"] = {
            "primary_instruction": "Use section_slot_checklist as the writing plan; evidence_for_report is supporting trace context.",
            "background_sentence": (
                "When available, compose PDR frequency, amplitude, symmetry, reactivity, "
                "and organization into one formal EEG background sentence."
            ),
            "protocol_sentence": "Render photic and hyperventilation status explicitly when known, including not_performed.",
            "state_sentence": "Render awake/drowsy/sleep/stage II architecture explicitly when known.",
            "transient_context_sentence": (
                "Render transient morphology/localization/electrode maxima only as caveated transient context; "
                "do not call it epileptiform unless the slot value explicitly says epileptiform."
            ),
            "unknown_slot_policy": (
                "Unknown, unavailable, or not_available slots are omitted from the checklist and must not be "
                "summarized as absence, normality, or 'no definitive abnormality'."
            ),
        }
        out["evidence_view_summary"] = dict(out.get("evidence_view_summary", {}))
        out["evidence_view_summary"]["slot_group_count"] = len(checklist)
        out["evidence_view_summary"]["slot_count"] = sum(len(group.get("slots", {})) for group in checklist.values())
        return out

    def _section_slot_checklist(self, evidence_for_report: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups: Dict[str, Dict[str, Any]] = {
            "background": {"slots": {}, "section_hint": "EEG DESCRIPTION/DETAILS"},
            "activation_protocol": {"slots": {}, "section_hint": "EEG DESCRIPTION/DETAILS"},
            "state": {"slots": {}, "section_hint": "EEG DESCRIPTION/DETAILS"},
            "transient_event_context": {"slots": {}, "section_hint": "EEG DESCRIPTION/DETAILS"},
            "seizure": {"slots": {}, "section_hint": "EEG DESCRIPTION/DETAILS"},
        }
        for evidence in evidence_for_report:
            target = str(evidence.get("clinical_target") or "")
            value = evidence.get("normalized_value")
            if value in (None, "", {}, []):
                value = evidence.get("value")
            if value in (None, "", {}, []):
                continue
            evidence_id = str(evidence.get("evidence_id") or "")
            unit = evidence.get("unit")
            if target == "pdr" and isinstance(value, dict):
                if self._is_truthy_status(value.get("pdr_supported", True)):
                    self._put_slot(groups, "background", "pdr_frequency", value.get("frequency_hz"), unit or "Hz", evidence_id, status="supported")
                self._put_slot(groups, "background", "pdr_reactivity", value.get("reactivity"), None, evidence_id)
                self._put_slot(groups, "background", "pdr_symmetry", value.get("symmetry"), None, evidence_id)
                self._put_slot(groups, "background", "pdr_posterior_support", value.get("posterior_alpha_ratio"), None, evidence_id, status="supporting_context")
            elif target == "background_amplitude":
                if isinstance(value, dict):
                    if isinstance(value.get("background_amplitude_range_uv"), dict):
                        amp_range = value["background_amplitude_range_uv"]
                        self._put_slot(
                            groups,
                            "background",
                            "background_amplitude_range",
                            {"lower": amp_range.get("lower"), "upper": amp_range.get("upper")},
                            unit or "µV",
                            evidence_id,
                        )
                    elif value.get("background_amplitude_range_uv") is not None:
                        self._put_slot(groups, "background", "background_amplitude_typical", value.get("background_amplitude_range_uv"), unit or "µV", evidence_id)
                    self._put_slot(
                        groups,
                        "background",
                        "background_amplitude_best_supported",
                        value.get("background_amplitude_best_supported_uv"),
                        unit or "µV",
                        evidence_id,
                    )
                    self._put_slot(groups, "background", "background_amplitude_typical", value.get("background_amplitude_typical_uv"), unit or "µV", evidence_id)
                else:
                    self._put_slot(groups, "background", "background_amplitude_typical", value, unit or "µV", evidence_id)
            elif target == "background_slowing":
                self._put_slot(groups, "background", "background_slowing", value, unit, evidence_id, status="caveated")
            elif target == "excess_beta":
                self._put_slot(groups, "background", "excess_beta", value, unit, evidence_id, status="caveated")
            elif target == "protocol" and isinstance(value, dict):
                for key, raw_value in value.items():
                    self._put_slot(groups, "activation_protocol", str(key), raw_value, None, evidence_id)
            elif target == "state" and isinstance(value, dict):
                for key, raw_value in value.items():
                    self._put_slot(groups, "state", str(key), raw_value, None, evidence_id)
            elif target == "localization" and isinstance(value, dict):
                for key in ("spatial_pattern", "field_descriptor", "electrode_maxima", "region", "laterality"):
                    self._put_slot(groups, "transient_event_context", f"transient_{key}", value.get(key), None, evidence_id, status="caveated")
            elif target == "epileptiform_morphology" and isinstance(value, dict):
                self._put_slot(groups, "transient_event_context", "transient_morphology", value.get("morphology_descriptor"), None, evidence_id, status="caveated")
                waveform = value.get("event_waveform_numeric")
                if isinstance(waveform, dict):
                    self._put_slot(groups, "transient_event_context", "transient_waveform_amplitude_uv", waveform.get("amplitude_peak_to_peak_typical_uv"), "uV", evidence_id, status="caveated")
                    self._put_slot(groups, "transient_event_context", "transient_waveform_frequency_hz", waveform.get("dominant_frequency_hz"), "Hz", evidence_id, status="caveated")
            elif target == "seizure_evidence":
                self._put_slot(groups, "seizure", "seizure_evidence", value, unit, evidence_id, status="requires_seizure_specific_support")
            elif target == "context":
                self._put_slot(groups, "state", "context", value, unit, evidence_id, status="context_only")
        return {
            group_name: group
            for group_name, group in groups.items()
            if group.get("slots")
        }

    def _put_slot(
        self,
        groups: Dict[str, Dict[str, Any]],
        group_name: str,
        slot_name: str,
        value: Any,
        unit: Any,
        evidence_id: str,
        status: Any | None = None,
    ) -> None:
        if value in (None, "", {}, []):
            return
        if isinstance(value, str) and value.strip().lower() in {"unknown", "not_available", "unavailable"}:
            return
        slot_status = str(status or "present")
        slots = groups.setdefault(group_name, {"slots": {}, "section_hint": "EEG DESCRIPTION/DETAILS"})["slots"]
        existing = slots.get(slot_name)
        if existing is None:
            slots[slot_name] = {
                "status": slot_status,
                "value": value,
                "unit": unit,
                "evidence_ids": [evidence_id] if evidence_id else [],
            }
            return
        if evidence_id and evidence_id not in existing["evidence_ids"]:
            existing["evidence_ids"].append(evidence_id)

    def _is_truthy_status(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "present", "supported", "available"}
        return bool(value)

    def _has_surface_safe_content(self, payload: Dict[str, Any]) -> bool:
        if payload.get("value") not in (None, "", {}, []):
            return True
        if payload.get("normalized_value") not in (None, "", {}, []):
            return True
        if payload.get("time_provenance"):
            return True
        if payload.get("space_provenance"):
            return True
        return False


    def _section_descriptions(self, target_section_names: List[str]) -> Dict[str, str]:
        descriptions = {
            "EEG DESCRIPTION/DETAILS": (
                "Detailed narrative of EEG findings including background activity, sleep stages, "
                "physiologic variants, activation procedures, and abnormalities observed during the recording period."
            ),
            "IMPRESSION/INTERPRETATION": (
                "Concise clinical interpretation summarizing the most important supported EEG findings and limitations."
            ),
            "BACKGROUND ACTIVITY": (
                "Background rhythm description including PDR, organization, symmetry, amplitude, reactivity, slowing, and beta activity when supported."
            ),
            "SLEEP": (
                "Sleep and drowsiness findings including vertex waves, spindles, K-complexes, and state-dependent abnormalities when supported."
            ),
            "EPILEPTIFORM ABNORMALITIES": (
                "Interictal epileptiform findings including morphology, frequency, field, laterality, localization, state dependence, and uncertainty when supported."
            ),
            "INTERICTAL EPILEPTIFORM ABNORMALITIES": (
                "Interictal epileptiform findings including morphology, frequency, field, laterality, localization, state dependence, and uncertainty when supported."
            ),
            "EVENTS/SEIZURES": (
                "Clinical or electrographic events, push-button events, and seizure-related findings only when event or seizure-specific evidence supports them."
            ),
            "SEIZURES": (
                "Seizure presence or absence only when seizure-specific evidence or validated metadata supports the statement."
            ),
        }
        return {section: descriptions.get(section.upper(), "Requested EEG report section; use only supplied allowed/caveated claims.") for section in target_section_names}

    def _claim_plan_payload(
        self,
        plan: AtomicClaimPlan,
        shared_board: Any,
        surface_decision_payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        linked_reportable_evidence = self._linked_reportable_evidence_payload(plan, shared_board)
        reportable_evidence_values = self._reportable_evidence_values(linked_reportable_evidence)
        surface_safe_values = list(plan.surface_safe_values or reportable_evidence_values)
        must_render_values = list(plan.must_render_values or self._surface_value_requirements(reportable_evidence_values))
        numeric_claims = list(plan.numeric_claims or self._numeric_claims_from_surface_values(surface_safe_values))
        clinical_reference_ids = list((surface_decision_payload or {}).get("clinical_reference_ids") or [])
        return {
            "plan_id": plan.plan_id,
            "claim_type": plan.claim_type,
            "proposed_text": plan.proposed_text,
            "evidence_ids": plan.evidence_ids,
            "linked_reportable_evidence": linked_reportable_evidence,
            "reportable_evidence_values": reportable_evidence_values,
            "surface_safe_values": surface_safe_values,
            "must_render_values": must_render_values,
            "surface_value_requirements": must_render_values,
            "numeric_claims": numeric_claims,
            "clinical_reference_ids": clinical_reference_ids,
            "clinical_references": clinical_reference_payloads(clinical_reference_ids),
            "surface_action": plan.surface_action.value,
            "allowed_sections": plan.allowed_sections,
            "clinical_phrase_template_id": plan.clinical_phrase_template_id,
            "linked_measurement_ids": plan.linked_measurement_ids,
            "required_evidence": plan.required_evidence,
            "missing_evidence": plan.missing_evidence,
            "confidence": plan.confidence,
            "rationale": plan.rationale,
        }

    def _reportable_evidence_values(self, evidence_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compact value view for report wording; contains no debug/proxy values."""
        out: List[Dict[str, Any]] = []
        for payload in evidence_payloads:
            value = payload.get("normalized_value")
            if value in (None, "", {}, []):
                value = payload.get("value")
            if value in (None, "", {}, []):
                continue
            out.append(
                {
                    "evidence_id": payload.get("evidence_id"),
                    "clinical_target": payload.get("clinical_target"),
                    "value": value,
                    "unit": payload.get("unit"),
                }
            )
        return out

    def _surface_value_requirements(self, evidence_values: List[Dict[str, Any]]) -> List[str]:
        """Human-readable safe numeric/status values the report LLM should preserve."""
        requirements: list[str] = []
        for item in evidence_values:
            target = str(item.get("clinical_target") or "")
            unit = item.get("unit")
            value = item.get("value")
            if target == "pdr" and isinstance(value, dict):
                freq = value.get("frequency_hz")
                if isinstance(freq, (int, float)):
                    requirements.append(f"preserve PDR candidate frequency: {float(freq):.1f} {unit or 'Hz'}")
            elif target == "background_amplitude" and isinstance(value, dict):
                best_supported = value.get("background_amplitude_best_supported_uv")
                if isinstance(best_supported, (int, float)) and float(best_supported) > 0.0:
                    requirements.append(f"preserve best-supported background amplitude candidate: {float(best_supported):.1f} {unit or 'uV'}")
                    continue
                typical = value.get("background_amplitude_typical_uv")
                if isinstance(typical, (int, float)) and float(typical) > 0.0:
                    requirements.append(f"preserve typical background amplitude: {float(typical):.1f} {unit or 'uV'}")
                    continue
                amp_range = value.get("background_amplitude_range_uv")
                if isinstance(amp_range, dict) and amp_range.get("upper") is not None:
                    lo = amp_range.get("lower", 0.0)
                    hi = amp_range["upper"]
                    requirements.append(f"preserve background amplitude range: {float(lo):.1f}-{float(hi):.1f} {unit or 'uV'}")
            elif target in {"state", "protocol"} and isinstance(value, dict):
                for key, raw_value in value.items():
                    if self._is_known_status(raw_value):
                        requirements.append(f"preserve {key}: {raw_value}")
            elif target == "epileptiform_morphology" and isinstance(value, dict):
                waveform = value.get("event_waveform_numeric")
                if isinstance(waveform, dict):
                    amp = waveform.get("amplitude_peak_to_peak_typical_uv")
                    freq = waveform.get("dominant_frequency_hz")
                    if isinstance(amp, (int, float)):
                        requirements.append(f"preserve caveated event waveform amplitude: {float(amp):.1f} uV")
                    if isinstance(freq, (int, float)):
                        requirements.append(f"preserve caveated event waveform frequency: {float(freq):.1f} Hz")
            elif isinstance(value, (int, float)) and unit:
                requirements.append(f"preserve {target} value: {float(value):.1f} {unit}")
        return requirements

    def _numeric_claims_from_surface_values(self, surface_values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        for item in surface_values:
            target = str(item.get("clinical_target") or "")
            unit = str(item.get("unit") or "")
            evidence_id = str(item.get("evidence_id") or "")
            value = item.get("value")
            if target == "pdr" and isinstance(value, dict):
                freq = value.get("frequency_hz")
                if isinstance(freq, (int, float)) and 8.0 <= float(freq) <= 13.0:
                    claims.append(self._numeric_claim("pdr_frequency", float(freq), unit or "Hz", evidence_id))
            elif target == "background_amplitude" and isinstance(value, dict):
                best_supported = value.get("background_amplitude_best_supported_uv")
                if isinstance(best_supported, (int, float)) and float(best_supported) > 0.0:
                    claims.append(
                        self._numeric_claim(
                            "background_amplitude_best_supported",
                            float(best_supported),
                            unit or "uV",
                            evidence_id,
                        )
                    )
                    continue
                typical = value.get("background_amplitude_typical_uv")
                if isinstance(typical, (int, float)) and float(typical) > 0.0:
                    claims.append(
                        self._numeric_claim(
                            "background_amplitude_typical",
                            float(typical),
                            unit or "uV",
                            evidence_id,
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
                                "background_amplitude_range",
                                {"lower": lo, "upper": hi},
                                unit or "uV",
                                evidence_id,
                            )
                        )
            elif target == "epileptiform_morphology" and isinstance(value, dict):
                waveform = value.get("event_waveform_numeric")
                if not isinstance(waveform, dict):
                    continue
                amp = waveform.get("amplitude_peak_to_peak_typical_uv")
                if isinstance(amp, (int, float)) and float(amp) > 0.0:
                    claims.append(self._numeric_claim("event_waveform_amplitude", float(amp), "uV", evidence_id))
                freq = waveform.get("dominant_frequency_hz")
                if isinstance(freq, (int, float)) and float(freq) > 0.0:
                    claims.append(self._numeric_claim("event_waveform_frequency", float(freq), "Hz", evidence_id))
        return claims

    def _numeric_claim(self, slot: str, value: Any, unit: str, evidence_id: str) -> Dict[str, Any]:
        if isinstance(value, dict):
            render_text = f"{float(value['lower']):.1f}-{float(value['upper']):.1f} {unit}"
        else:
            render_text = f"{float(value):.1f} {unit}"
        return {
            "slot": slot,
            "value": value,
            "unit": unit,
            "evidence_id": evidence_id,
            "render_required": True,
            "render_text": render_text,
            "source": "surface_safe_values",
        }

    def _is_known_status(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in {"", "unknown", "not_available", "unavailable"}
        return True

    def _linked_reportable_evidence_payload(self, plan: AtomicClaimPlan, shared_board: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for evidence_id in plan.evidence_ids:
            try:
                item = shared_board.get_evidence(evidence_id)
            except KeyError:
                continue
            payload = self._surface_safe_evidence_payload(item)
            if payload:
                out.append(payload)
        return out

    def _surface_safe_evidence_payload(self, item: Any) -> Dict[str, Any] | None:
        evidence_type = str(getattr(item.evidence_type, "value", item.evidence_type))
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        if evidence_type in {"debug", "llm_assisted"}:
            return None
        if self._is_internal_or_proxy_evidence_view(item):
            return None
        safe_value_targets = {
            "pdr",
            "background_slowing",
            "background_amplitude",
            "excess_beta",
            "state",
            "protocol",
            "seizure_evidence",
            "localization",
            "epileptiform_morphology",
            "context",
        }
        include_value = target in safe_value_targets and evidence_type in {"direct", "derived", "metadata"}
        return {
            "evidence_id": item.evidence_id,
            "source_module": item.source_module,
            "evidence_type": evidence_type,
            "clinical_target": target,
            "value": item.value if include_value else None,
            "unit": item.unit,
            "normalized_value": item.normalized_value if include_value else None,
            "time_provenance": item.time_provenance,
            "space_provenance": item.space_provenance,
            "measurement_ids": list(item.measurement_ids),
            "report_surface_note": (
                "Use only for the linked atomic claim; do not surface proxy/debug/internal values."
            ),
        }

    def _is_internal_or_proxy_evidence_view(self, item: Any) -> bool:
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
        if target in {"event_candidate", "epileptiform_morphology", "localization"}:
            if evidence_type in {"direct", "derived"} and any(
                safe_key in text
                for safe_key in (
                    "spatial_pattern",
                    "field_descriptor",
                    "electrode_maxima",
                    "morphology_descriptor",
                    "event_waveform_numeric",
                    "sharp_transient_like",
                    "sharp_wave_like",
                    "spike_wave_like",
                    "generalized_spike_wave_like",
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

    def _surface_decision_payload(self, decision: Any) -> Dict[str, Any]:
        return {
            "decision_id": decision.decision_id,
            "claim_id": decision.claim_id,
            "surface_action": decision.surface_action.value,
            "allowed_sections": decision.allowed_sections,
            "forbidden_sections": decision.forbidden_sections,
            "clinical_phrase_template_id": decision.clinical_phrase_template_id,
            "rationale": decision.rationale,
            "hard_deny_reasons": decision.hard_deny_reasons,
            "evidence_ids": decision.evidence_ids,
            "clinical_reference_ids": list(getattr(decision, "clinical_reference_ids", []) or []),
            "clinical_references": clinical_reference_payloads(getattr(decision, "clinical_reference_ids", []) or []),
            "caveat": decision.caveat,
            "decided_by": decision.decided_by,
        }

    def _audit_record_count(self, board: EvidenceBoard) -> int:
        return sum(
            len(d.weak_evidence) + len(d.missing_slots) + len(d.do_not_claim) + len(d.claim_constraints)
            for d in board.deliberations
        )

    def _surface_plan_counts_by_section(
        self,
        surface_plans: List[AtomicClaimPlan],
        surface_decisions: List[Any],
        target_section_names: List[str],
    ) -> Dict[str, int]:
        decision_by_claim = {decision.claim_id: decision for decision in surface_decisions}
        counts = {section_name: 0 for section_name in target_section_names}
        for section_name in target_section_names:
            role = self.section_router.role_for_section(section_name)
            for plan in surface_plans:
                decision = decision_by_claim.get(plan.plan_id, self.surface_policy.decide(plan))
                if decision.surface_action not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                    continue
                if role.value in decision.forbidden_sections:
                    continue
                if decision.allowed_sections and role.value not in decision.allowed_sections:
                    continue
                counts[section_name] += 1
        return counts

    def _validate_and_align_sections(
        self,
        model_output: Dict[str, Any],
        target_section_names: List[str],
        surface_plan_counts_by_section: Dict[str, int] | None = None,
    ) -> Dict[str, str]:
        raw_sections = model_output.get("report_sections") or []
        by_name = {
            str(item.get("section_name", "")).strip().lower(): str(item.get("section_text", "")).strip()
            for item in raw_sections
            if isinstance(item, dict)
        }
        section_texts: Dict[str, str] = {}
        for idx, target in enumerate(target_section_names):
            if surface_plan_counts_by_section is not None and surface_plan_counts_by_section.get(target, 0) <= 0:
                section_texts[target] = self.surface_policy.safe_fallback_for_role(self.section_router.role_for_section(target))
                continue
            key = target.strip().lower()
            text = by_name.get(key)
            if text is None and idx < len(raw_sections) and isinstance(raw_sections[idx], dict):
                text = str(raw_sections[idx].get("section_text", "")).strip()
            text = text or self.surface_policy.safe_fallback_for_role(self.section_router.role_for_section(target))
            section_texts[target] = self._sanitize_section_text(target, text)
        return section_texts

    def _sanitize_section_text(self, section_name: str, text: str) -> str:
        text = text.replace("μV", "µV").replace("μv", "µV")
        if self.surface_policy.contains_forbidden_surface_text(text):
            return self.surface_policy.safe_fallback_for_role(self.section_router.role_for_section(section_name))
        return text
