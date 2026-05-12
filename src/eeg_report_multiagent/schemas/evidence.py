from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from .agent import AgentDeliberationRecord
from .finding import FindingObject
from .measurement import MeasurementValue
from .report import ClaimRecord
from .tooling import ToolInvocationRecord


class EvidenceBoard(BaseModel):
    session_id: str
    measurements: List[MeasurementValue] = Field(default_factory=list)
    findings: List[FindingObject] = Field(default_factory=list)
    claims: List[ClaimRecord] = Field(default_factory=list)
    tool_invocations: List[ToolInvocationRecord] = Field(default_factory=list)
    deliberations: List[AgentDeliberationRecord] = Field(default_factory=list)
    index: Dict[str, List[str]] = Field(default_factory=dict)

    def rebuild_index(self) -> None:
        self.index = {}
        for finding in self.findings:
            self.index.setdefault(finding.finding_type, []).append(finding.finding_id)
