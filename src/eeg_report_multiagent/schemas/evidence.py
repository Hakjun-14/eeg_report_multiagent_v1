from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .agent import AgentDeliberationRecord
from .measurement import MeasurementValue
from .report import ClaimRecord
from .shared_evidence import SharedEvidenceBoard
from .tooling import ToolInvocationRecord


class RuntimeEvidenceBundle(BaseModel):
    """Runtime container around canonical SharedEvidenceBoard.

    This is not the canonical evidence store. It carries measurements, tool
    traces, generated claims, review records, and the SharedEvidenceBoard for
    one pipeline run.
    """

    session_id: str
    measurements: List[MeasurementValue] = Field(default_factory=list)
    claims: List[ClaimRecord] = Field(default_factory=list)
    tool_invocations: List[ToolInvocationRecord] = Field(default_factory=list)
    deliberations: List[AgentDeliberationRecord] = Field(default_factory=list)
    shared_evidence_board: SharedEvidenceBoard | None = None

    def ensure_shared_evidence_board(self) -> SharedEvidenceBoard:
        if self.shared_evidence_board is None:
            from eeg_report_multiagent.modules.evidence_item_adapter import build_shared_evidence_board

            self.shared_evidence_board = build_shared_evidence_board(
                recording_id=self.session_id,
                measurements=self.measurements,
            )
        return self.shared_evidence_board


EvidenceBoard = RuntimeEvidenceBundle
