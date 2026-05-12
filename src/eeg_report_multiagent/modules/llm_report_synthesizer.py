from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from eeg_report_multiagent.llm import OpenAIReportSynthesisAdapter
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.finding import FindingObject
from eeg_report_multiagent.schemas.measurement import MeasurementValue


@dataclass(frozen=True)
class LLMReportSynthesisResult:
    section_texts: Dict[str, str]
    trace: Dict[str, Any]


class EvidenceBoardLLMReportSynthesizer:
    """Method D: EvidenceBoard-only LLM report synthesis.

    The LLM receives only typed evidence summaries and target section names. It
    never receives raw EEG arrays, pkl paths, or GT/reference report text.
    """

    synthesizer_name = "evidence_board_llm_report_synthesizer"
    synthesis_version = "D_v1"

    def __init__(self, adapter: OpenAIReportSynthesisAdapter | None = None) -> None:
        self.adapter = adapter or OpenAIReportSynthesisAdapter()

    def synthesize_celm_sections(
        self,
        board: EvidenceBoard,
        target_section_names: List[str],
    ) -> LLMReportSynthesisResult:
        payload = self._build_payload(board, target_section_names)
        model_output = self.adapter.synthesize(payload)
        section_texts = self._validate_and_align_sections(model_output, target_section_names)
        trace = {
            "synthesizer_name": self.synthesizer_name,
            "synthesis_version": self.synthesis_version,
            "model_name": self.adapter.model,
            "raw_eeg_used": bool(model_output.get("raw_eeg_used")),
            "gt_report_used": bool(model_output.get("gt_report_used")),
            "target_section_names": target_section_names,
            "global_limitations": model_output.get("global_limitations", []),
            "model_response_id": model_output.get("_response_id"),
            "model_report_sections": model_output.get("report_sections", []),
            "privacy_contract": payload["privacy_contract"],
        }
        if trace["raw_eeg_used"] or trace["gt_report_used"]:
            raise ValueError("D synthesis violated input contract")
        return LLMReportSynthesisResult(section_texts=section_texts, trace=trace)

    def _build_payload(self, board: EvidenceBoard, target_section_names: List[str]) -> Dict[str, Any]:
        measurement_index = {m.measurement_id: m for m in board.measurements}
        return {
            "session_id": board.session_id,
            "target_section_names": target_section_names,
            "findings": [self._finding_payload(f, measurement_index) for f in board.findings],
            "measurements": [self._measurement_payload(m) for m in board.measurements],
            "deliberation_constraints": self._deliberation_payload(board),
            "style_policy": {
                "tone": "formal_clinical_eeg_report",
                "allowed_claims": "claims directly supported by typed evidence only",
                "uncertainty_handling": "explicitly state candidate/automated/limited evidence when appropriate",
                "section_behavior": "generate each requested section once and do not add extra sections",
            },
            "forbidden_inputs": [
                "raw_eeg_arrays",
                "processed_pkl_payloads",
                "reference_gt_report_text",
                "unbounded_external_tools",
            ],
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
                "contains_source_pkl_paths": False,
            },
        }

    def _finding_payload(
        self,
        finding: FindingObject,
        measurement_index: Dict[str, MeasurementValue],
    ) -> Dict[str, Any]:
        q = finding.quantitation
        measurements = [measurement_index[mid] for mid in finding.measurement_ids if mid in measurement_index]
        return {
            "finding_id": finding.finding_id,
            "finding_type": finding.finding_type,
            "assertion": finding.assertion.value,
            "confidence": finding.confidence,
            "source_module": finding.source_module,
            "summary_label": finding.summary_label,
            "quantitation": self._quantitation_payload(q),
            "measurement_ids": finding.measurement_ids,
            "measurement_names": [m.measurement_name for m in measurements],
            "provenance_summary": self._provenance_summary(finding.provenance),
            "tags": finding.tags,
        }

    def _measurement_payload(self, measurement: MeasurementValue) -> Dict[str, Any]:
        return {
            "measurement_id": measurement.measurement_id,
            "measurement_name": measurement.measurement_name,
            "quantitation": self._quantitation_payload(measurement.quantitation),
            "status": measurement.status_value.status.value if measurement.status_value else None,
            "status_reason": measurement.status_value.reason if measurement.status_value else None,
            "categorical_value": measurement.categorical_value,
            "boolean_value": measurement.boolean_value,
            "confidence": measurement.confidence,
            "provenance_summary": self._provenance_summary([measurement.provenance]),
        }

    def _quantitation_payload(self, q: Any) -> Dict[str, Any] | None:
        if q is None:
            return None
        return {
            "kind": q.kind.value,
            "unit": q.unit,
            "exact": q.exact,
            "lower": q.lower,
            "upper": q.upper,
            "values_count": len(q.values),
            "values_preview": q.values[:8],
        }

    def _provenance_summary(self, provenance: List[Any]) -> Dict[str, Any]:
        windows: list[int] = []
        channels: list[str] = []
        regions: list[str] = []
        lateralities: list[str] = []
        tools: list[str] = []
        source_types: list[str] = []
        has_time = False
        has_space = False
        for p in provenance:
            source_types.append(p.source_type.value)
            if p.time.window_indices:
                has_time = True
                windows.extend(int(x) for x in p.time.window_indices[:12])
            if p.time.start_sec is not None or p.time.end_sec is not None:
                has_time = True
            if p.space.channels:
                has_space = True
                channels.extend(str(x) for x in p.space.channels[:12])
            if p.space.region:
                has_space = True
                regions.append(str(p.space.region))
            if p.space.laterality:
                has_space = True
                lateralities.append(str(p.space.laterality))
            if p.measurement:
                tools.append(str(p.measurement.tool_name))
        return {
            "source_types": sorted(set(source_types)),
            "has_time": has_time,
            "window_indices_preview": sorted(set(windows))[:12],
            "has_space": has_space,
            "channels_preview": sorted(set(channels))[:12],
            "regions": sorted(set(regions)),
            "lateralities": sorted(set(lateralities)),
            "tool_names": sorted(set(tools)),
            "provenance_count": len(provenance),
        }

    def _deliberation_payload(self, board: EvidenceBoard) -> Dict[str, Any]:
        return {
            "weak_evidence": [
                {
                    "severity": item.severity.value,
                    "target_type": item.target_type,
                    "target_id": item.target_id,
                    "reason": item.reason,
                    "recommendation": item.recommendation,
                    "linked_finding_ids": item.linked_finding_ids,
                }
                for d in board.deliberations
                for item in d.weak_evidence
            ],
            "missing_slots": [
                {
                    "slot_name": item.slot_name,
                    "target_module": item.target_module,
                    "severity": item.severity.value,
                    "reason": item.reason,
                    "expected_evidence": item.expected_evidence,
                    "linked_finding_ids": item.linked_finding_ids,
                }
                for d in board.deliberations
                for item in d.missing_slots
            ],
            "do_not_claim": [
                {
                    "text": item.text,
                    "rationale": item.rationale,
                    "linked_finding_ids": item.linked_finding_ids,
                }
                for d in board.deliberations
                for item in d.do_not_claim
            ],
            "claim_constraints": [
                {
                    "target": item.target,
                    "constraint": item.constraint,
                    "rationale": item.rationale,
                    "linked_finding_ids": item.linked_finding_ids,
                }
                for d in board.deliberations
                for item in d.claim_constraints
            ],
        }

    def _validate_and_align_sections(self, model_output: Dict[str, Any], target_section_names: List[str]) -> Dict[str, str]:
        raw_sections = model_output.get("report_sections") or []
        by_name = {
            str(item.get("section_name", "")).strip().lower(): str(item.get("section_text", "")).strip()
            for item in raw_sections
            if isinstance(item, dict)
        }
        section_texts: Dict[str, str] = {}
        for idx, target in enumerate(target_section_names):
            key = target.strip().lower()
            text = by_name.get(key)
            if text is None and idx < len(raw_sections) and isinstance(raw_sections[idx], dict):
                text = str(raw_sections[idx].get("section_text", "")).strip()
            section_texts[target] = text or "No supported structured evidence was available for this section."
        return section_texts
