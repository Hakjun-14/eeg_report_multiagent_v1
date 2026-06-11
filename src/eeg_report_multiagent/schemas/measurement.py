from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .provenance import ProvenanceRecord


class StatusSemantic(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    NOT_OBSERVED = "not_observed"
    NOT_PERFORMED = "not_performed"
    NO_RESPONSE = "no_response"
    UNKNOWN = "unknown"


class QuantitationKind(str, Enum):
    EXACT = "exact"
    RANGE = "range"
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    DISTRIBUTION = "distribution"


class MeasurementContextDependency(str, Enum):
    SIGNAL_ONLY = "signal_only"
    CONTEXT_SELECTED = "context_selected"
    CONTEXT_STATUS = "context_status"
    CROSS_MODAL = "cross_modal"
    UNKNOWN = "unknown"


class MeasurementRole(str, Enum):
    CLINICAL_MEASUREMENT = "clinical_measurement"
    SUPPORT_FEATURE = "support_feature"
    PROXY_SCORE = "proxy_score"
    STATUS_OBSERVATION = "status_observation"
    DEBUG_DIAGNOSTIC = "debug_diagnostic"
    UNKNOWN = "unknown"


class QuantitationValue(BaseModel):
    kind: QuantitationKind
    unit: Optional[str] = None
    exact: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    values: List[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_kind(self) -> "QuantitationValue":
        if self.kind == QuantitationKind.EXACT and self.exact is None:
            raise ValueError("exact quantitation requires exact")
        if self.kind == QuantitationKind.RANGE and (self.lower is None or self.upper is None):
            raise ValueError("range quantitation requires lower and upper")
        if self.kind == QuantitationKind.UPPER_BOUND and self.upper is None:
            raise ValueError("upper_bound quantitation requires upper")
        if self.kind == QuantitationKind.LOWER_BOUND and self.lower is None:
            raise ValueError("lower_bound quantitation requires lower")
        if self.kind == QuantitationKind.DISTRIBUTION and not self.values:
            raise ValueError("distribution quantitation requires values")
        return self


class StatusValue(BaseModel):
    status: StatusSemantic
    reason: Optional[str] = None


class MeasurementValue(BaseModel):
    """Deterministic tool output.

    Measurements are produced by bounded functions, not by LLM judgment, so
    they intentionally do not carry a standalone confidence score. If a tool
    needs to expose quality metadata, store it in typed payload/provenance or
    debug-only evidence, not as report-surface confidence.
    """

    measurement_id: str
    measurement_name: str
    quantitation: Optional[QuantitationValue] = None
    status_value: Optional[StatusValue] = None
    categorical_value: Optional[str] = None
    boolean_value: Optional[bool] = None
    provenance: ProvenanceRecord
    metadata: Dict[str, str] = Field(default_factory=dict)
    context_dependency: MeasurementContextDependency = MeasurementContextDependency.UNKNOWN
    measurement_role: MeasurementRole = MeasurementRole.UNKNOWN

    @model_validator(mode="after")
    def validate_payload(self) -> "MeasurementValue":
        payload_count = sum(
            [
                self.quantitation is not None,
                self.status_value is not None,
                self.categorical_value is not None,
                self.boolean_value is not None,
            ]
        )
        if payload_count == 0:
            raise ValueError("measurement must include at least one typed payload")
        return self
