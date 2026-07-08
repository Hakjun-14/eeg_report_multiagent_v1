from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ClinicalReferenceItem(BaseModel):
    """Clinical interpretation reference used to contextualize a claim.

    This is not patient evidence. Patient-specific support remains in
    EvidenceItem IDs; this object records which guideline/knowledge rule was
    used to interpret that evidence safely.
    """

    reference_id: str
    concept: str
    short_rule: str
    source_name: str
    source_path: Optional[str] = None
    source_kind: str = "clinical_reference"
    applicable_targets: List[str] = Field(default_factory=list)
    applicable_claim_types: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
