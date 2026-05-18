from __future__ import annotations

from typing import Dict, List

from eeg_report_multiagent.agents.background_agent import BackgroundAgent
from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue, StatusSemantic
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord
from eeg_report_multiagent.tools.registry import ToolRegistry


def _finding_from_measurement(m: MeasurementValue) -> Finding:
    finding_type = "background_measurement"
    assertion = StatusSemantic.UNKNOWN

    if m.measurement_name == "background_dominant_frequency_hz":
        finding_type = "background_frequency"
        hz = m.quantitation.exact if m.quantitation else None
        if hz is not None and (hz <= 0.51 or hz >= 29.99):
            assertion = StatusSemantic.UNKNOWN
        elif hz is not None and hz < 8.0:
            assertion = StatusSemantic.PRESENT
        else:
            assertion = StatusSemantic.ABSENT
    elif m.measurement_name == "pdr_candidate_frequency_hz":
        finding_type = "background_pdr_frequency"
        assertion = StatusSemantic.PRESENT if m.metadata.get("pdr_supported") == "true" else StatusSemantic.UNKNOWN
    elif m.measurement_name == "pdr_candidate_confidence_score":
        finding_type = "background_pdr_support"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 0.35) else StatusSemantic.UNKNOWN
    elif m.measurement_name == "pdr_posterior_anterior_alpha_ratio":
        finding_type = "background_pdr_topography"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 1.2) else StatusSemantic.UNKNOWN
    elif m.measurement_name == "pdr_symmetry_score":
        finding_type = "background_pdr_symmetry"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 0.65) else StatusSemantic.UNKNOWN
    elif m.measurement_name == "background_ap_organization_score":
        finding_type = "background_ap_organization"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 0.35) else StatusSemantic.UNKNOWN
    elif m.measurement_name == "background_reactivity_status":
        finding_type = "background_reactivity"
        assertion = m.status_value.status if m.status_value else StatusSemantic.UNKNOWN
    elif m.measurement_name == "sleep_architecture_status":
        finding_type = "sleep_architecture"
        assertion = m.status_value.status if m.status_value else StatusSemantic.UNKNOWN
    elif m.measurement_name == "background_amplitude_range_uv":
        finding_type = "background_amplitude_range"
        assertion = StatusSemantic.PRESENT
    elif m.measurement_name == "slowing_score":
        finding_type = "background_slowing"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 1.0) else StatusSemantic.ABSENT
    elif m.measurement_name == "beta_excess_score":
        finding_type = "excess_beta"
        score = m.quantitation.exact if m.quantitation else None
        assertion = StatusSemantic.PRESENT if (score is not None and score >= 0.35) else StatusSemantic.ABSENT

    return Finding(
        finding_id=f"f_{m.measurement_id}",
        finding_type=finding_type,
        assertion=assertion,
        quantitation=m.quantitation,
        measurement_ids=[m.measurement_id],
    )


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
        findings: List[Finding] = []
        invocations: List[ToolInvocationRecord] = []

        for tool_name in tools:
            output, rec = self.registry.dispatch(
                tool_name,
                signal_nct=signal_nct,
                fs=fs,
                source_ref=source_ref,
                channels=channels,
            ) if tool_name in {"posterior_dominant_rhythm_candidate", "background_organization_proxy"} else self.registry.dispatch(
                tool_name,
                source_ref=source_ref,
            ) if tool_name in {"background_unavailable_slot_status"} else self.registry.dispatch(
                tool_name,
                signal_nct=signal_nct,
                fs=fs,
                source_ref=source_ref,
            ) if tool_name not in {"amplitude_summary"} else self.registry.dispatch(
                tool_name,
                signal_nct=signal_nct,
                source_ref=source_ref,
            )
            invocations.append(rec)
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, MeasurementValue):
                        measurements.append(item)
                        findings.append(_finding_from_measurement(item))

        return {
            "measurements": measurements,
            "findings": findings,
            "tool_invocations": invocations,
        }
