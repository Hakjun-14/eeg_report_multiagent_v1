from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DebugLeakType(str, Enum):
    DEBUG_SCORE = "debug_score"
    PROXY_CONCEPT = "proxy_concept"
    INTERNAL_REVIEWER_TEXT = "internal_reviewer_text"
    MEASUREMENT_ARTIFACT = "measurement_artifact"
    UNSUPPORTED_TOOL_OUTPUT = "unsupported_tool_output"


class NumericMatchStatus(str, Enum):
    EXACT = "exact"
    RANGE_CONTAINED = "range_contained"
    UNIT_MISMATCH = "unit_mismatch"
    NO_MATCH = "no_match"
    MATCHED_BUT_NOT_REPORTABLE = "matched_but_not_reportable"
    MATCHED_BUT_WRONG_SECTION = "matched_but_wrong_section"


class NumericMention(BaseModel):
    raw_text: str
    value: Any
    unit: Optional[str] = None
    normalized_value: Any = None
    section_name: str
    sentence: str
    char_start: int
    char_end: int


class DebugLeak(BaseModel):
    term: str
    section_name: str
    sentence: str
    leak_type: DebugLeakType


class SectionLeakage(BaseModel):
    section_name: str
    sentence: str
    leakage_type: str
    rationale: str


class NumericProvenanceMatch(BaseModel):
    numeric_mention: NumericMention
    matched_evidence_id: Optional[str] = None
    match_status: NumericMatchStatus
    rationale: str


class ClaimSurfaceMatch(BaseModel):
    section_name: str
    sentence: str
    matched_plan_id: Optional[str] = None
    matched_evidence_ids: List[str] = Field(default_factory=list)
    match_status: str
    rationale: str


class FinalProseAuditResult(BaseModel):
    unsupported_numeric_mentions: List[NumericProvenanceMatch] = Field(default_factory=list)
    supported_numeric_mentions: List[NumericProvenanceMatch] = Field(default_factory=list)
    debug_leaks: List[DebugLeak] = Field(default_factory=list)
    section_leakages: List[SectionLeakage] = Field(default_factory=list)
    seizure_gate_violations: List[SectionLeakage] = Field(default_factory=list)
    unmatched_surface_claims: List[ClaimSurfaceMatch] = Field(default_factory=list)
    missing_required_evidence_links: List[ClaimSurfaceMatch] = Field(default_factory=list)
    pass_fail: str
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
