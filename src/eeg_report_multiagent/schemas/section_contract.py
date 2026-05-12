from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SectionRole(str, Enum):
    DETAIL = "detail"
    BACKGROUND = "background"
    EPILEPTIFORM = "epileptiform"
    EVENTS_SEIZURES = "events_seizures"
    SEIZURES = "seizures"
    SLEEP = "sleep"
    IMPRESSION = "impression"
    OTHER = "other"


class SectionSlotRequirement(BaseModel):
    slot_name: str
    finding_types: List[str] = Field(default_factory=list)
    required: bool = True
    nullable: bool = False
    reason: str


class TargetReportSection(BaseModel):
    raw_name: str
    standardized_name: str
    role: SectionRole
    required_slots: List[SectionSlotRequirement] = Field(default_factory=list)
    optional_slots: List[SectionSlotRequirement] = Field(default_factory=list)
    generation_policy: str


class TargetSectionContract(BaseModel):
    """Benchmark section contract without GT section text leakage."""

    contract_id: str
    report_id: str
    source: str = "celm_split_target_section_names"
    target_sections: List[TargetReportSection]
    reference_text_available: bool = True
    reference_text_allowed_as_inference_input: bool = False
    eval_only_reference_json_path: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
