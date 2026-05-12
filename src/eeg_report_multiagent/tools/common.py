from __future__ import annotations

from typing import Iterable, List, Optional

from eeg_report_multiagent.schemas.measurement import (
    MeasurementValue,
    QuantitationKind,
    QuantitationValue,
    StatusSemantic,
    StatusValue,
)
from eeg_report_multiagent.schemas.provenance import (
    MeasurementProvenance,
    ProvenanceRecord,
    SourceType,
    SpaceProvenance,
    TimeProvenance,
)


def make_provenance(
    tool_name: str,
    function_name: str,
    source_ref: str,
    source_type: SourceType = SourceType.SIGNAL,
    window_indices: Optional[Iterable[int]] = None,
    channels: Optional[Iterable[str]] = None,
    region: Optional[str] = None,
    laterality: Optional[str] = None,
    reason: Optional[str] = None,
    confidence: Optional[float] = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=source_type,
        source_ref=source_ref,
        time=TimeProvenance(window_indices=list(window_indices or [])),
        space=SpaceProvenance(channels=list(channels or []), region=region, laterality=laterality),
        measurement=MeasurementProvenance(tool_name=tool_name, function_name=function_name),
        reason=reason,
        confidence=confidence,
    )


def make_exact_measurement(
    measurement_id: str,
    measurement_name: str,
    value: float,
    unit: Optional[str],
    provenance: ProvenanceRecord,
    confidence: Optional[float] = None,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=float(value), unit=unit),
        provenance=provenance,
        confidence=confidence,
    )


def make_range_measurement(
    measurement_id: str,
    measurement_name: str,
    lower: float,
    upper: float,
    unit: Optional[str],
    provenance: ProvenanceRecord,
    confidence: Optional[float] = None,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.RANGE, lower=float(lower), upper=float(upper), unit=unit),
        provenance=provenance,
        confidence=confidence,
    )


def make_upper_bound_measurement(
    measurement_id: str,
    measurement_name: str,
    upper: float,
    unit: Optional[str],
    provenance: ProvenanceRecord,
    confidence: Optional[float] = None,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.UPPER_BOUND, upper=float(upper), unit=unit),
        provenance=provenance,
        confidence=confidence,
    )


def make_distribution_measurement(
    measurement_id: str,
    measurement_name: str,
    values: List[float],
    unit: Optional[str],
    provenance: ProvenanceRecord,
    confidence: Optional[float] = None,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.DISTRIBUTION, values=[float(x) for x in values], unit=unit),
        provenance=provenance,
        confidence=confidence,
    )


def make_status_measurement(
    measurement_id: str,
    measurement_name: str,
    status: StatusSemantic,
    provenance: ProvenanceRecord,
    reason: Optional[str] = None,
    confidence: Optional[float] = None,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        status_value=StatusValue(status=status, reason=reason),
        provenance=provenance,
        confidence=confidence,
    )


def make_categorical_measurement(
    measurement_id: str,
    measurement_name: str,
    value: str,
    provenance: ProvenanceRecord,
    confidence: Optional[float] = None,
) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        categorical_value=value,
        provenance=provenance,
        confidence=confidence,
    )
