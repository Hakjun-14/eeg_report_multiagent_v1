from __future__ import annotations

from typing import Dict, List

import numpy as np

from eeg_report_multiagent.agents.event_agent import EventAgent
from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue, StatusSemantic
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord
from eeg_report_multiagent.tools.registry import ToolRegistry


def _extract_distribution_values(measurements: List[MeasurementValue], measurement_name: str) -> np.ndarray:
    for m in measurements:
        if m.measurement_name == measurement_name and m.quantitation is not None:
            return np.asarray(m.quantitation.values, dtype=float)
    return np.zeros((0,), dtype=float)


def _finding_from_measurement(m: MeasurementValue) -> Finding:
    finding_type = "event_measurement"
    assertion = StatusSemantic.UNKNOWN

    if m.measurement_name == "event_candidate_burden_ratio":
        finding_type = "epileptiform_event_candidate_burden"
        v = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (v is not None and v > 0.05) else StatusSemantic.ABSENT
    elif m.measurement_name == "event_train_duration_upper_sec":
        finding_type = "event_train_duration"
        assertion = StatusSemantic.PRESENT if (m.quantitation and (m.quantitation.upper or 0.0) > 0.0) else StatusSemantic.ABSENT
    elif m.measurement_name == "event_laterality_index":
        finding_type = "event_laterality"
        assertion = StatusSemantic.PRESENT
    elif m.measurement_name == "event_clinical_localization_label":
        finding_type = "event_clinical_localization"
        assertion = StatusSemantic.PRESENT if m.categorical_value != "unknown" else StatusSemantic.UNKNOWN
    elif m.measurement_name == "event_localization_concentration_ratio":
        finding_type = "event_localization_support"
        assertion = StatusSemantic.PRESENT if (m.quantitation and (m.quantitation.exact or 0.0) > 1.5) else StatusSemantic.UNKNOWN
    elif m.measurement_name == "event_peak_localization_label":
        finding_type = "event_peak_localization"
        assertion = StatusSemantic.PRESENT if m.categorical_value != "unknown" else StatusSemantic.UNKNOWN
    elif m.measurement_name == "event_peak_field_concentration_ratio":
        finding_type = "event_peak_field_support"
        assertion = StatusSemantic.PRESENT if (m.quantitation and (m.quantitation.exact or 0.0) > 1.5) else StatusSemantic.UNKNOWN
    elif m.measurement_name == "event_peak_laterality_index":
        finding_type = "event_peak_laterality"
        assertion = StatusSemantic.PRESENT
    elif m.measurement_name == "event_bifrontal_ratio":
        finding_type = "event_focality_bifrontal_spread"
        assertion = StatusSemantic.PRESENT if (m.quantitation and (m.quantitation.exact or 0.0) > 1.1) else StatusSemantic.ABSENT
    elif m.measurement_name == "event_morphology_proxy_class":
        finding_type = "event_morphology_class"
        assertion = StatusSemantic.PRESENT if m.categorical_value != "insufficient_morphology_evidence" else StatusSemantic.UNKNOWN
    elif m.measurement_name == "event_morphology_support_score":
        finding_type = "event_morphology_support"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 1.0) else StatusSemantic.ABSENT
    elif m.measurement_name == "event_field_concentration_ratio":
        finding_type = "event_field_concentration"
        assertion = StatusSemantic.PRESENT if (m.quantitation and (m.quantitation.exact or 0.0) > 1.5) else StatusSemantic.ABSENT
    elif m.measurement_name == "epileptiform_candidate_likelihood_score":
        finding_type = "epileptiform_candidate_likelihood"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 0.50) else StatusSemantic.ABSENT
    elif m.measurement_name == "electrographic_seizure_likelihood_score":
        finding_type = "electrographic_seizure_likelihood"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 0.70) else StatusSemantic.ABSENT

    return Finding(
        finding_id=f"f_{m.measurement_id}",
        finding_type=finding_type,
        assertion=assertion,
        quantitation=m.quantitation,
        measurement_ids=[m.measurement_id],
    )


class EventModule:
    def __init__(self, registry: ToolRegistry, agent: EventAgent) -> None:
        self.registry = registry
        self.agent = agent

    def run(
        self,
        signal_nct,
        channels: List[str],
        source_ref: str,
        window_seconds: int,
        scout_summary: Dict[str, float],
    ) -> Dict[str, object]:
        tools = self.agent.select_tools(scout_summary)

        measurements: List[MeasurementValue] = []
        findings: List[Finding] = []
        invocations: List[ToolInvocationRecord] = []

        # Always run transient candidate first
        output, rec = self.registry.dispatch(
            "transient_candidate_score",
            signal_nct=signal_nct,
            source_ref=source_ref,
        )
        invocations.append(rec)
        if isinstance(output, list):
            measurements.extend([m for m in output if isinstance(m, MeasurementValue)])

        score_dist = _extract_distribution_values(measurements, "event_candidate_score_distribution")
        suspicious = np.where(score_dist > np.percentile(score_dist, 90))[0].tolist() if score_dist.size else []

        for tool_name in tools:
            if tool_name == "transient_candidate_score":
                continue
            if tool_name == "burst_train_duration_estimate":
                output, rec = self.registry.dispatch(
                    tool_name,
                    score_distribution=score_dist,
                    window_seconds=window_seconds,
                    source_ref=source_ref,
                )
            elif tool_name in {
                "channel_spread_laterality_summary",
                "event_localization_normalizer",
                "event_peak_topography_localizer",
                "focality_bifrontal_summary",
            }:
                output, rec = self.registry.dispatch(
                    tool_name,
                    signal_nct=signal_nct,
                    channels=channels,
                    suspicious_windows=suspicious,
                    source_ref=source_ref,
                )
            elif tool_name == "morphology_feature_encoder":
                output, rec = self.registry.dispatch(
                    tool_name,
                    signal_nct=signal_nct,
                    channels=channels,
                    suspicious_windows=suspicious,
                    source_ref=source_ref,
                )
            elif tool_name == "event_type_separation_classifier":
                output, rec = self.registry.dispatch(
                    tool_name,
                    signal_nct=signal_nct,
                    channels=channels,
                    suspicious_windows=suspicious,
                    score_distribution=score_dist,
                    window_seconds=window_seconds,
                    source_ref=source_ref,
                )
            else:
                continue
            invocations.append(rec)
            if isinstance(output, list):
                measurements.extend([m for m in output if isinstance(m, MeasurementValue)])

        for m in measurements:
            findings.append(_finding_from_measurement(m))

        return {
            "measurements": measurements,
            "findings": findings,
            "tool_invocations": invocations,
            "focused_windows": suspicious,
        }
