from __future__ import annotations

from typing import Dict, List

import numpy as np

from eeg_report_multiagent.agents.event_agent import EventAgent
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord
from eeg_report_multiagent.tools.registry import ToolRegistry


def _extract_distribution_values(measurements: List[MeasurementValue], measurement_name: str) -> np.ndarray:
    for m in measurements:
        if m.measurement_name == measurement_name and m.quantitation is not None:
            return np.asarray(m.quantitation.values, dtype=float)
    return np.zeros((0,), dtype=float)


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

        return {
            "measurements": measurements,
            "tool_invocations": invocations,
            "focused_windows": suspicious,
        }
