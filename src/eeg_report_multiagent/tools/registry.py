from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord, ToolSchemaRef, ToolSpec
from eeg_report_multiagent.tools.background import signal_tools as bg_tools
from eeg_report_multiagent.tools.event import signal_tools as ev_tools
from eeg_report_multiagent.tools.parser import text_tools as p_tools


ToolCallable = Callable[..., Any]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    fn: ToolCallable


class ToolRegistry:
    def __init__(self, registry_name: str) -> None:
        self.registry_name = registry_name
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, fn: ToolCallable) -> None:
        self._tools[spec.tool_name] = RegisteredTool(spec=spec, fn=fn)

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def dispatch(self, tool_name: str, **kwargs: Any) -> tuple[Any, ToolInvocationRecord]:
        if tool_name not in self._tools:
            raise KeyError(f"tool not found in registry '{self.registry_name}': {tool_name}")

        tool = self._tools[tool_name]
        invocation_id = str(uuid.uuid4())
        try:
            output = tool.fn(**kwargs)
            measurement_ids: List[str] = []
            if isinstance(output, list) and output and isinstance(output[0], MeasurementValue):
                measurement_ids = [m.measurement_id for m in output]
            rec = ToolInvocationRecord(
                invocation_id=invocation_id,
                tool_name=tool_name,
                module_name=tool.spec.module_name,
                input_digest={k: type(v).__name__ for k, v in kwargs.items()},
                output_measurement_ids=measurement_ids,
                status="ok",
            )
            return output, rec
        except Exception as exc:  # pragma: no cover - defensive logging path
            rec = ToolInvocationRecord(
                invocation_id=invocation_id,
                tool_name=tool_name,
                module_name=tool.spec.module_name,
                input_digest={k: type(v).__name__ for k, v in kwargs.items()},
                status="error",
                error_message=str(exc),
            )
            raise RuntimeError(f"tool dispatch failed: {tool_name}: {exc}") from exc


def _spec(tool_name: str, description: str, module_name: str) -> ToolSpec:
    return ToolSpec(
        tool_name=tool_name,
        description=description,
        module_name=module_name,
        input_schema=ToolSchemaRef(schema_name=f"{tool_name}_input"),
        output_schema=ToolSchemaRef(schema_name=f"{tool_name}_output"),
    )


def build_background_registry() -> ToolRegistry:
    reg = ToolRegistry("background")
    reg.register(_spec("psd_power_spectrum_summary", "Dominant frequency summary", "background"), bg_tools.psd_power_spectrum_summary)
    reg.register(_spec("posterior_dominant_rhythm_candidate", "Posterior alpha/PDR candidate summary", "background"), bg_tools.posterior_dominant_rhythm_candidate)
    reg.register(_spec("background_organization_proxy", "Anterior-posterior organization proxy", "background"), bg_tools.background_organization_proxy)
    reg.register(_spec("background_unavailable_slot_status", "Nullable background slot status declarations", "background"), bg_tools.background_unavailable_slot_status)
    reg.register(_spec("bandpower_summary", "Relative bandpower summary", "background"), bg_tools.bandpower_summary)
    reg.register(_spec("amplitude_summary", "Amplitude range summary", "background"), bg_tools.amplitude_summary)
    reg.register(_spec("slowing_score", "Background slowing score", "background"), bg_tools.slowing_score)
    reg.register(_spec("beta_excess_score", "Beta excess score", "background"), bg_tools.beta_excess_score)
    return reg


def build_event_registry() -> ToolRegistry:
    reg = ToolRegistry("event")
    reg.register(_spec("transient_candidate_score", "Per-window candidate score", "event"), ev_tools.transient_candidate_score)
    reg.register(_spec("burst_train_duration_estimate", "Burst/train duration estimator", "event"), ev_tools.burst_train_duration_estimate)
    reg.register(_spec("channel_spread_laterality_summary", "Laterality spread summary", "event"), ev_tools.channel_spread_laterality_summary)
    reg.register(_spec("event_localization_normalizer", "Coarse channel-to-clinical-region normalizer", "event"), ev_tools.event_localization_normalizer)
    reg.register(_spec("event_peak_topography_localizer", "Peak-centered event topography localizer", "event"), ev_tools.event_peak_topography_localizer)
    reg.register(_spec("focality_bifrontal_summary", "Focality/bifrontal summary", "event"), ev_tools.focality_bifrontal_summary)
    reg.register(_spec("morphology_feature_encoder", "Local morphology feature encoder proxy", "event"), ev_tools.morphology_feature_encoder)
    reg.register(_spec("event_type_separation_classifier", "Separate event-candidate, epileptiform-candidate, and seizure likelihood", "event"), ev_tools.event_type_separation_classifier)
    return reg


def build_parser_registry() -> ToolRegistry:
    reg = ToolRegistry("parser")
    reg.register(_spec("report_section_splitter", "Split report into sections", "parser"), p_tools.report_section_splitter)
    reg.register(_spec("metadata_normalizer", "Normalize metadata availability", "parser"), p_tools.metadata_normalizer)
    reg.register(_spec("status_semantics_extractor", "Extract status semantics", "parser"), p_tools.status_semantics_extractor)
    reg.register(_spec("comparison_history_parser", "Parse comparison/history presence", "parser"), p_tools.comparison_history_parser)
    return reg
