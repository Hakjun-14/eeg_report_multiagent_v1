from __future__ import annotations

from typing import Dict, List

from eeg_report_multiagent.agents.background_agent import BackgroundAgent
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord
from eeg_report_multiagent.tools.registry import ToolRegistry


class BackgroundModule:
    def __init__(self, registry: ToolRegistry, agent: BackgroundAgent) -> None:
        self.registry = registry
        self.agent = agent

    def run(
        self,
        signal_nct,
        fs: int,
        source_ref: str,
        scout_summary: Dict[str, float],
        channels: List[str] | None = None,
    ) -> Dict[str, object]:
        tools = self.agent.select_tools(scout_summary)

        measurements: List[MeasurementValue] = []
        invocations: List[ToolInvocationRecord] = []

        for tool_name in tools:
            output, rec = self.registry.dispatch(
                tool_name,
                signal_nct=signal_nct,
                fs=fs,
                source_ref=source_ref,
                channels=channels,
            ) if tool_name in {
                "posterior_dominant_rhythm_candidate",
                "posterior_dominant_rhythm_spectral_v2",
                "background_organization_proxy",
                "state_signal_summary",
                "bandpower_summary",
                "slowing_score",
                "beta_excess_score",
            } else self.registry.dispatch(
                tool_name,
                source_ref=source_ref,
            ) if tool_name in {"background_unavailable_slot_status"} else self.registry.dispatch(
                tool_name,
                signal_nct=signal_nct,
                source_ref=source_ref,
                channels=channels,
            ) if tool_name in {"amplitude_summary"} else self.registry.dispatch(
                tool_name,
                signal_nct=signal_nct,
                fs=fs,
                source_ref=source_ref,
            )
            invocations.append(rec)
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, MeasurementValue):
                        measurements.append(item)

        return {
            "measurements": measurements,
            "tool_invocations": invocations,
        }
