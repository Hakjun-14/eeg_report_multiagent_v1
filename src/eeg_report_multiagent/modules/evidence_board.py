from __future__ import annotations

from typing import Iterable, List

from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.finding import FindingObject
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.report import ClaimRecord
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord


class EvidenceBoardAssembler:
    def merge(
        self,
        session_id: str,
        measurement_groups: Iterable[List[MeasurementValue]],
        finding_groups: Iterable[List[FindingObject]],
        tool_invocation_groups: Iterable[List[ToolInvocationRecord]],
        claims: List[ClaimRecord] | None = None,
    ) -> EvidenceBoard:
        measurements: List[MeasurementValue] = []
        findings: List[FindingObject] = []
        invocations: List[ToolInvocationRecord] = []

        for group in measurement_groups:
            measurements.extend(group)
        for group in finding_groups:
            findings.extend(group)
        for group in tool_invocation_groups:
            invocations.extend(group)

        board = EvidenceBoard(
            session_id=session_id,
            measurements=measurements,
            findings=findings,
            claims=claims or [],
            tool_invocations=invocations,
        )
        board.rebuild_index()
        return board
