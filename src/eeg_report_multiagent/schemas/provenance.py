from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    SIGNAL = "signal"
    REPORT_TEXT = "report_text"
    METADATA = "metadata"
    SYNTHESIZER = "synthesizer"


class TimeProvenance(BaseModel):
    window_indices: List[int] = Field(default_factory=list)
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


class SpaceProvenance(BaseModel):
    channels: List[str] = Field(default_factory=list)
    region: Optional[str] = None
    laterality: Optional[str] = None


class MeasurementProvenance(BaseModel):
    tool_name: str
    function_name: str
    measurement_ids: List[str] = Field(default_factory=list)


class ClaimProvenance(BaseModel):
    claim_id: Optional[str] = None
    supports_finding_id: Optional[str] = None


class ProvenanceRecord(BaseModel):
    source_type: SourceType
    source_ref: Optional[str] = None
    time: TimeProvenance = Field(default_factory=TimeProvenance)
    space: SpaceProvenance = Field(default_factory=SpaceProvenance)
    measurement: Optional[MeasurementProvenance] = None
    claim: Optional[ClaimProvenance] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    value_span: Optional[Tuple[float, float]] = None
