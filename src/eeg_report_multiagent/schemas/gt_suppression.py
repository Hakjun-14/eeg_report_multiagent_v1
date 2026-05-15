from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GTAtomicClaim(BaseModel):
    gt_claim_id: str
    case_id: str
    section: str
    claim_type: str
    normalized_value: Optional[Any] = None
    unit: Optional[str] = None
    state: Optional[str] = None
    topography: Optional[Dict[str, Any]] = None
    certainty: str = "asserted"
    source_text: str


class GTClaimPipelineMatch(BaseModel):
    gt_claim_id: str
    case_id: str
    matched_measurement_ids: List[str] = Field(default_factory=list)
    matched_finding_ids: List[str] = Field(default_factory=list)
    matched_evidence_ids: List[str] = Field(default_factory=list)
    matched_atomic_claim_ids: List[str] = Field(default_factory=list)
    surfaced_sentence: Optional[str] = None
    match_stage: str
    suppression_stage: str = "none"
    suppression_reason: str = ""
    category: str = "not_gt_required"
    salvageability: str = "keep_blocked"
    rationale: str = ""


class GTSuppressionAuditResult(BaseModel):
    case_id: str
    variant: str
    gt_claims: List[GTAtomicClaim] = Field(default_factory=list)
    gt_claim_matches: List[GTClaimPipelineMatch] = Field(default_factory=list)
    gt_required_suppressed_claims: List[str] = Field(default_factory=list)
    gt_required_missing_evidence: List[str] = Field(default_factory=list)
    gt_required_surface_rate: float = 0.0
    gt_required_claim_recovery_rate: float = 0.0
    recommendations: List[str] = Field(default_factory=list)


class GTSuppressionAggregate(BaseModel):
    variant: str
    num_cases: int
    num_gt_claims: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    category_counts: Dict[str, int] = Field(default_factory=dict)
    suppression_stage_counts: Dict[str, int] = Field(default_factory=dict)
    salvageability_counts: Dict[str, int] = Field(default_factory=dict)
    stage3_recommendation: str
