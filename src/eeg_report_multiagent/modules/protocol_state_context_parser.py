from __future__ import annotations

from typing import Dict, List

from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord
from eeg_report_multiagent.tools.registry import ToolRegistry


class ProtocolStateContextParser:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def run(self, note_text: str, metadata: Dict[str, str], source_ref: str) -> Dict[str, object]:
        invocations: List[ToolInvocationRecord] = []
        measurements: List[MeasurementValue] = []

        _sections, rec_sections = self.registry.dispatch(
            "report_section_splitter",
            note_text=note_text,
            source_ref=source_ref,
        )
        invocations.append(rec_sections)

        status_meas, rec_status = self.registry.dispatch(
            "status_semantics_extractor",
            note_text=note_text,
            source_ref=source_ref,
        )
        invocations.append(rec_status)
        if isinstance(status_meas, list):
            measurements.extend([m for m in status_meas if isinstance(m, MeasurementValue)])

        metadata_meas, rec_meta = self.registry.dispatch(
            "metadata_normalizer",
            metadata=metadata,
            source_ref=source_ref,
        )
        invocations.append(rec_meta)
        if isinstance(metadata_meas, list):
            measurements.extend([m for m in metadata_meas if isinstance(m, MeasurementValue)])

        history_meas, rec_hist = self.registry.dispatch(
            "comparison_history_parser",
            note_text=note_text,
            source_ref=source_ref,
        )
        invocations.append(rec_hist)
        if isinstance(history_meas, list):
            measurements.extend([m for m in history_meas if isinstance(m, MeasurementValue)])

        return {
            "measurements": measurements,
            "findings": [],
            "tool_invocations": invocations,
        }
