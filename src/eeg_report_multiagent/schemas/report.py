from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ReportSectionType(str, Enum):
    DETAIL = "detail"
    IMPRESSION = "impression"


class ReportSection(BaseModel):
    section_type: ReportSectionType
    text: str
    claim_ids: List[str] = Field(default_factory=list)


class ClaimSurfaceAction(str, Enum):
    ALLOW = "allow"
    CAVEAT = "caveat"
    BLOCK = "block"
    DEBUG_ONLY = "debug_only"


class AtomicClaimPlan(BaseModel):
    """Planned clinical claim before it is allowed onto the report surface."""

    plan_id: str
    section_type: ReportSectionType
    claim_type: str
    proposed_text: str
    linked_finding_ids: List[str] = Field(default_factory=list)
    linked_measurement_ids: List[str] = Field(default_factory=list)
    required_evidence: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    surface_action: ClaimSurfaceAction
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    allowed_sections: List[str] = Field(default_factory=list)
    forbidden_sections: List[str] = Field(default_factory=list)
    clinical_phrase_template_id: Optional[str] = None
    debug_payload: dict[str, Any] = Field(default_factory=dict)


class SurfaceDecision(BaseModel):
    """Central decision for whether evidence may reach clinical report prose."""

    surface_action: ClaimSurfaceAction
    allowed_sections: List[str] = Field(default_factory=list)
    forbidden_sections: List[str] = Field(default_factory=list)
    clinical_phrase_template_id: Optional[str] = None
    rationale: str
    evidence_ids: List[str] = Field(default_factory=list)
    debug_payload: dict[str, Any] = Field(default_factory=dict)


class ClaimSupportLabel(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    MISSING = "missing"


class ClaimRecord(BaseModel):
    claim_id: str
    section_type: ReportSectionType
    text: str
    linked_finding_ids: List[str] = Field(default_factory=list)


class VerificationRecord(BaseModel):
    claim_id: str
    support_label: ClaimSupportLabel
    evidence_finding_ids: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
