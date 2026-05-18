from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .measurement import QuantitationValue, StatusSemantic
from .provenance import ProvenanceRecord


class Finding(BaseModel):
    """Clinical grouping over measurements.

    `Finding` is intentionally thin: provenance and tool confidence should be
    read from linked `MeasurementValue` records through `measurement_ids`.
    The legacy fields below remain only so older artifacts/tests can load.
    """

    finding_id: str
    finding_type: str
    assertion: StatusSemantic
    quantitation: Optional[QuantitationValue] = None
    summary_label: Optional[str] = None
    measurement_ids: List[str] = Field(default_factory=list)
    provenance: List[ProvenanceRecord] = Field(default_factory=list, description="Deprecated compatibility; use linked MeasurementValue.provenance.")
    confidence: Optional[float] = Field(default=None, description="Deprecated compatibility; use linked MeasurementValue.confidence or SurfaceDecision confidence.")
    source_module: Optional[str] = Field(default=None, description="Deprecated compatibility; infer source from linked MeasurementValue provenance/tool.")
    tags: List[str] = Field(default_factory=list)
