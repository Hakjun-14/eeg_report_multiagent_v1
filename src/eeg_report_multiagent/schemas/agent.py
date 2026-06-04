from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceGapSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceGap(BaseModel):
    gap_id: str
    evidence_target: str
    severity: EvidenceGapSeverity
    reason: str
    linked_measurement_ids: List[str] = Field(default_factory=list)


class ToolRequestProposal(BaseModel):
    proposal_id: str
    target_module: str
    tool_name: str
    rationale: str
    expected_measurement: str
    linked_gap_ids: List[str] = Field(default_factory=list)
    linked_measurement_ids: List[str] = Field(default_factory=list)


class RejectedToolRequestProposal(ToolRequestProposal):
    rejection_reason: str


class WeakEvidenceRecord(BaseModel):
    weakness_id: str
    severity: EvidenceGapSeverity
    target_type: str
    target_id: str
    reason: str
    linked_measurement_ids: List[str] = Field(default_factory=list)
    recommendation: str


class MissingSlotRecord(BaseModel):
    slot_id: str
    slot_name: str
    target_module: str
    severity: EvidenceGapSeverity
    reason: str
    expected_evidence: str
    linked_measurement_ids: List[str] = Field(default_factory=list)


class DoNotClaimRecord(BaseModel):
    item_id: str
    text: str
    rationale: str
    linked_measurement_ids: List[str] = Field(default_factory=list)
    linked_evidence_ids: List[str] = Field(default_factory=list)


class ClaimConstraintRecord(BaseModel):
    constraint_id: str
    target: str
    constraint: str
    rationale: str
    linked_measurement_ids: List[str] = Field(default_factory=list)
    linked_evidence_ids: List[str] = Field(default_factory=list)


class AgentDeliberationRecord(BaseModel):
    review_id: str
    reviewer_name: str
    status: str
    review_version: str = "v1"
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list)
    weak_evidence: List[WeakEvidenceRecord] = Field(default_factory=list)
    missing_slots: List[MissingSlotRecord] = Field(default_factory=list)
    do_not_claim: List[DoNotClaimRecord] = Field(default_factory=list)
    claim_constraints: List[ClaimConstraintRecord] = Field(default_factory=list)
    tool_request_proposals: List[ToolRequestProposal] = Field(default_factory=list)
    rejected_tool_request_proposals: List[RejectedToolRequestProposal] = Field(default_factory=list)
    summary: str
    raw_eeg_used: bool = False
    gt_report_used: bool = False
    model_name: Optional[str] = None
    error_message: Optional[str] = None
