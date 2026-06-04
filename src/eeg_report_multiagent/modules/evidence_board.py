from __future__ import annotations

from typing import Iterable, List

from eeg_report_multiagent.schemas.evidence import RuntimeEvidenceBundle
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.report import ClaimRecord
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord
from eeg_report_multiagent.modules.evidence_item_adapter import build_shared_evidence_board


class EvidenceBoardAssembler:
    def merge(
        self,
        session_id: str,
        measurement_groups: Iterable[List[MeasurementValue]],
        tool_invocation_groups: Iterable[List[ToolInvocationRecord]],
        claims: List[ClaimRecord] | None = None,
    ) -> RuntimeEvidenceBundle:
        measurements: List[MeasurementValue] = []
        invocations: List[ToolInvocationRecord] = []

        for group in measurement_groups:
            measurements.extend(group)
        for group in tool_invocation_groups:
            invocations.extend(group)

        board = RuntimeEvidenceBundle(
            session_id=session_id,
            measurements=measurements,
            claims=claims or [],
            tool_invocations=invocations,
            shared_evidence_board=build_shared_evidence_board(
                recording_id=session_id,
                measurements=measurements,
            ),
        )
        return board
