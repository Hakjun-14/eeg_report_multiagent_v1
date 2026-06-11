from __future__ import annotations

from typing import Iterable, List, Optional

from eeg_report_multiagent.schemas.measurement import (
    MeasurementContextDependency,
    MeasurementRole,
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


def infer_measurement_tags(
    measurement_name: str,
    *,
    is_status: bool = False,
    is_categorical: bool = False,
) -> tuple[MeasurementContextDependency, MeasurementRole]:
    """Classify tool output without making a report-surface decision.

    These tags describe how a MeasurementValue was obtained and how it should
    be treated downstream. They are not reportability or surface-action labels.
    """
    name = measurement_name.lower()
    if is_status:
        return MeasurementContextDependency.CONTEXT_STATUS, MeasurementRole.STATUS_OBSERVATION

    if any(term in name for term in ("candidate_burden", "duration_distribution", "score_distribution", "train_duration")):
        role = MeasurementRole.DEBUG_DIAGNOSTIC
    elif any(term in name for term in ("likelihood", "score", "ratio", "bandpower", "morphology_proxy", "laterality_index")):
        role = MeasurementRole.PROXY_SCORE
    elif any(term in name for term in ("localization", "laterality", "field_concentration", "bifrontal")):
        role = MeasurementRole.SUPPORT_FEATURE
    elif name in {
        "pdr_candidate_frequency_hz",
        "pdr_v2_frequency_hz",
        "background_amplitude_range_uv",
        "background_amplitude_typical_uv",
    }:
        role = MeasurementRole.CLINICAL_MEASUREMENT
    elif is_categorical:
        role = MeasurementRole.SUPPORT_FEATURE
    else:
        role = MeasurementRole.SUPPORT_FEATURE

    # Current signal tools do not yet select windows using state/protocol
    # metadata. Future awake/PDR or HV/photic pre-post tools can override this
    # by constructing MeasurementValue directly or extending helper arguments.
    return MeasurementContextDependency.SIGNAL_ONLY, role


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
) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=source_type,
        source_ref=source_ref,
        time=TimeProvenance(window_indices=list(window_indices or [])),
        space=SpaceProvenance(channels=list(channels or []), region=region, laterality=laterality),
        measurement=MeasurementProvenance(tool_name=tool_name, function_name=function_name),
        reason=reason,
    )


def make_exact_measurement(
    measurement_id: str,
    measurement_name: str,
    value: float,
    unit: Optional[str],
    provenance: ProvenanceRecord,
) -> MeasurementValue:
    context_dependency, measurement_role = infer_measurement_tags(measurement_name)
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=float(value), unit=unit),
        provenance=provenance,
        context_dependency=context_dependency,
        measurement_role=measurement_role,
    )


def make_range_measurement(
    measurement_id: str,
    measurement_name: str,
    lower: float,
    upper: float,
    unit: Optional[str],
    provenance: ProvenanceRecord,
) -> MeasurementValue:
    context_dependency, measurement_role = infer_measurement_tags(measurement_name)
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.RANGE, lower=float(lower), upper=float(upper), unit=unit),
        provenance=provenance,
        context_dependency=context_dependency,
        measurement_role=measurement_role,
    )


def make_upper_bound_measurement(
    measurement_id: str,
    measurement_name: str,
    upper: float,
    unit: Optional[str],
    provenance: ProvenanceRecord,
) -> MeasurementValue:
    context_dependency, measurement_role = infer_measurement_tags(measurement_name)
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.UPPER_BOUND, upper=float(upper), unit=unit),
        provenance=provenance,
        context_dependency=context_dependency,
        measurement_role=measurement_role,
    )


def make_distribution_measurement(
    measurement_id: str,
    measurement_name: str,
    values: List[float],
    unit: Optional[str],
    provenance: ProvenanceRecord,
) -> MeasurementValue:
    context_dependency, measurement_role = infer_measurement_tags(measurement_name)
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        quantitation=QuantitationValue(kind=QuantitationKind.DISTRIBUTION, values=[float(x) for x in values], unit=unit),
        provenance=provenance,
        context_dependency=context_dependency,
        measurement_role=measurement_role,
    )


def make_status_measurement(
    measurement_id: str,
    measurement_name: str,
    status: StatusSemantic,
    provenance: ProvenanceRecord,
    reason: Optional[str] = None,
) -> MeasurementValue:
    context_dependency, measurement_role = infer_measurement_tags(measurement_name, is_status=True)
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        status_value=StatusValue(status=status, reason=reason),
        provenance=provenance,
        context_dependency=context_dependency,
        measurement_role=measurement_role,
    )


def make_categorical_measurement(
    measurement_id: str,
    measurement_name: str,
    value: str,
    provenance: ProvenanceRecord,
) -> MeasurementValue:
    context_dependency, measurement_role = infer_measurement_tags(measurement_name, is_categorical=True)
    return MeasurementValue(
        measurement_id=measurement_id,
        measurement_name=measurement_name,
        categorical_value=value,
        provenance=provenance,
        context_dependency=context_dependency,
        measurement_role=measurement_role,
    )
