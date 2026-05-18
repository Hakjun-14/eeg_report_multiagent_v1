from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .measurement import QuantitationValue, StatusSemantic
from .provenance import ProvenanceRecord


class Finding(BaseModel):
    finding_id: str
    finding_type: str
    assertion: StatusSemantic
    quantitation: Optional[QuantitationValue] = None
    summary_label: Optional[str] = None
    measurement_ids: List[str] = Field(default_factory=list)
    provenance: List[ProvenanceRecord] = Field(default_factory=list)
    confidence: Optional[float] = None
    source_module: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
