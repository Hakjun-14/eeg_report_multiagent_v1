from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SlotFlowRecord(BaseModel):
    case_id: str
    section_name: str
    clinical_slot: str
    measurement_exists: bool = False
    evidence_item_exists: bool = False
    measurement_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_type_counts: Dict[str, int] = Field(default_factory=dict)
    reportability_counts: Dict[str, int] = Field(default_factory=dict)
    atomic_claim_exists: bool = False
    atomic_claim_ids: List[str] = Field(default_factory=list)
    surface_action_counts: Dict[str, int] = Field(default_factory=dict)
    surfaced_in_final_prose: bool = False
    final_sentence: Optional[str] = None
    suppression_reasons: List[str] = Field(default_factory=list)
    useful_but_suppressed: bool = False
    notes: Optional[str] = None


class EvidenceFlowAuditResult(BaseModel):
    case_id: str
    variant: str
    slot_records: List[SlotFlowRecord] = Field(default_factory=list)
    suppression_reason_counts: Dict[str, int] = Field(default_factory=dict)
    useful_suppressed_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    surfaced_slots: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    case_diagnosis: str


class EvidenceFlowAggregate(BaseModel):
    variant: str
    num_cases: int
    per_slot_availability: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    per_slot_surface_rate: Dict[str, float] = Field(default_factory=dict)
    per_slot_suppression_reason_counts: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    useful_suppressed_top_examples: List[Dict[str, Any]] = Field(default_factory=list)
    aggregate_recommendation_for_stage3: str
