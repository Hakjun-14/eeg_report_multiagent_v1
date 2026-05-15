from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from eeg_report_multiagent.schemas.gt_suppression import GTAtomicClaim


class GeneratedClaimMatch(BaseModel):
    case_id: str
    variant: str
    generated_claim_id: str
    generated_claim: GTAtomicClaim
    matched_gt_claim_ids: List[str] = Field(default_factory=list)
    is_extra_claim: bool = True


class GTClaimRecallMatch(BaseModel):
    case_id: str
    variant: str
    gt_claim_id: str
    gt_claim: GTAtomicClaim
    matched_generated_claim_ids: List[str] = Field(default_factory=list)
    is_missing: bool = True


class GeneratedClaimAuditResult(BaseModel):
    case_id: str
    variant: str
    gt_claims: List[GTAtomicClaim] = Field(default_factory=list)
    generated_claims: List[GTAtomicClaim] = Field(default_factory=list)
    generated_claim_matches: List[GeneratedClaimMatch] = Field(default_factory=list)
    gt_claim_recall_matches: List[GTClaimRecallMatch] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
