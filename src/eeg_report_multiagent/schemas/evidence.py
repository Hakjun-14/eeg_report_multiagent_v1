from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from .agent import AgentDeliberationRecord
from .finding import Finding
from .measurement import MeasurementValue
from .report import ClaimRecord
from .shared_evidence import SharedEvidenceBoard
from .tooling import ToolInvocationRecord


class EvidenceBoard(BaseModel):
    session_id: str
    measurements: List[MeasurementValue] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    claims: List[ClaimRecord] = Field(default_factory=list)
    tool_invocations: List[ToolInvocationRecord] = Field(default_factory=list)
    deliberations: List[AgentDeliberationRecord] = Field(default_factory=list)
    shared_evidence_board: SharedEvidenceBoard | None = None
    index: Dict[str, List[str]] = Field(default_factory=dict)

    def rebuild_index(self) -> None:
        self.index = {}
        for finding in self.findings:
            self.index.setdefault(finding.finding_type, []).append(finding.finding_id)

    def ensure_shared_evidence_board(self) -> SharedEvidenceBoard:
        if self.shared_evidence_board is None:
            from eeg_report_multiagent.modules.evidence_item_adapter import build_shared_evidence_board

            self.shared_evidence_board = build_shared_evidence_board(
                recording_id=self.session_id,
                measurements=self.measurements,
                findings=self.findings,
            )
        return self.shared_evidence_board
