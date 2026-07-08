from __future__ import annotations

from typing import Any, Dict, Iterable, List

from eeg_report_multiagent.llm import OpenAIEvidenceGroupingAdapter
from eeg_report_multiagent.schemas.measurement import MeasurementRole, MeasurementValue
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


class LLMEvidenceGrouper:
    """Group deterministic measurements into EvidenceItems with an LLM."""

    def __init__(self, adapter: OpenAIEvidenceGroupingAdapter | None = None) -> None:
        self.adapter = adapter or OpenAIEvidenceGroupingAdapter()

    def run(
        self,
        *,
        recording_id: str,
        measurements: Iterable[MeasurementValue],
        clinical_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        measurement_list = list(measurements)
        payload = self._payload(recording_id, measurement_list, clinical_context or {})
        result = self.adapter.group(payload)
        board = self._board_from_result(recording_id, measurement_list, result)
        return {
            "status": "ok",
            "model_name": self.adapter.model,
            "summary": str(result.get("summary", "")),
            "raw_eeg_used": bool(result.get("raw_eeg_used", False)),
            "gt_report_used": bool(result.get("gt_report_used", False)),
            "raw_result": result,
            "shared_evidence_board": board,
        }

    def _payload(
        self,
        recording_id: str,
        measurements: List[MeasurementValue],
        clinical_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "recording_id": recording_id,
            "clinical_context": clinical_context,
            "allowed_clinical_targets": [target.value for target in ClinicalTarget],
            "allowed_evidence_types": [evidence_type.value for evidence_type in EvidenceType],
            "allowed_sections": [role.value for role in SectionRole],
            "measurements": [self._measurement_payload(measurement) for measurement in measurements],
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
            },
        }

    def _measurement_payload(self, measurement: MeasurementValue) -> Dict[str, Any]:
        q = measurement.quantitation
        return {
            "measurement_id": measurement.measurement_id,
            "measurement_name": measurement.measurement_name,
            "quantitation": {
                "kind": q.kind.value,
                "unit": q.unit,
                "exact": q.exact,
                "lower": q.lower,
                "upper": q.upper,
                "values_count": len(q.values),
            }
            if q is not None
            else None,
            "status": measurement.status_value.status.value if measurement.status_value else None,
            "categorical_value": measurement.categorical_value,
            "boolean_value": measurement.boolean_value,
            "metadata": measurement.metadata,
            "provenance": {
                "source_type": measurement.provenance.source_type.value,
                "tool_name": measurement.provenance.measurement.tool_name if measurement.provenance.measurement else None,
                "function_name": measurement.provenance.measurement.function_name if measurement.provenance.measurement else None,
                "has_time": bool(
                    measurement.provenance.time.window_indices
                    or measurement.provenance.time.start_sec is not None
                    or measurement.provenance.time.end_sec is not None
                ),
                "channels": measurement.provenance.space.channels,
                "region": measurement.provenance.space.region,
                "laterality": measurement.provenance.space.laterality,
            },
        }

    def _board_from_result(
        self,
        recording_id: str,
        measurements: List[MeasurementValue],
        result: Dict[str, Any],
    ) -> SharedEvidenceBoard:
        measurement_index = {measurement.measurement_id: measurement for measurement in measurements}
        board = SharedEvidenceBoard(board_id=f"seb_llm_{recording_id}", recording_id=recording_id)
        for idx, group in enumerate(result.get("evidence_groups", [])):
            linked_ids = [mid for mid in group.get("linked_measurement_ids", []) if mid in measurement_index]
            linked_measurements = [measurement_index[mid] for mid in linked_ids]
            if not linked_measurements:
                continue
            evidence_type = EvidenceType(group.get("evidence_type", EvidenceType.LLM_ASSISTED.value))
            target = ClinicalTarget(group.get("clinical_target", ClinicalTarget.UNKNOWN.value))
            target = self._correct_target_from_measurements(target, linked_measurements)
            action = self._compat_reportability(evidence_type, target)
            report_linked_measurements = self._report_linked_measurements(target, linked_measurements)
            report_linked_ids = [measurement.measurement_id for measurement in report_linked_measurements]
            value = self._group_value(target, report_linked_measurements)
            item = EvidenceItem(
                evidence_id=self._safe_evidence_id(str(group.get("evidence_id") or f"llm_group_{idx}")),
                source_module="llm_evidence_grouper",
                evidence_type=evidence_type,
                clinical_target=target,
                value=value,
                unit=self._group_unit(report_linked_measurements),
                normalized_value=value,
                confidence=None,
                reliability=None,
                time_provenance=self._time_provenance(report_linked_measurements),
                space_provenance=self._space_provenance(report_linked_measurements),
                measurement_ids=report_linked_ids,
                reportability=action,
                allowed_sections=[section for section in group.get("allowed_sections", []) if section in {role.value for role in SectionRole}],
                rationale=None,
                caveat=None,
                debug_payload={
                    "llm_evidence_grouping": True,
                    "rationale": str(group.get("rationale", "")),
                    "value_summary": str(group.get("value_summary", "")),
                    "clinical_knowledge_reference": group.get("clinical_knowledge_reference", {}),
                    "all_linked_measurement_ids": linked_ids,
                    "measurement_names": [measurement.measurement_name for measurement in linked_measurements],
                },
                created_by="llm_evidence_grouper",
                created_at=EvidenceItem.now_iso(),
            )
            board.add_evidence(item)
        return board

    def _compat_reportability(self, evidence_type: EvidenceType, target: ClinicalTarget) -> ClaimSurfaceAction:
        if evidence_type in {EvidenceType.DEBUG, EvidenceType.LLM_ASSISTED}:
            return ClaimSurfaceAction.DEBUG_ONLY
        if evidence_type == EvidenceType.PROXY:
            return ClaimSurfaceAction.BLOCK
        if target in {ClinicalTarget.STATE, ClinicalTarget.PROTOCOL}:
            return ClaimSurfaceAction.ALLOW
        return ClaimSurfaceAction.CAVEAT

    def _correct_target_from_measurements(
        self,
        target: ClinicalTarget,
        measurements: List[MeasurementValue],
    ) -> ClinicalTarget:
        names = {measurement.measurement_name for measurement in measurements}
        if names.intersection(
            {
                "event_waveform_amplitude_peak_to_peak_typical_uv",
                "event_waveform_amplitude_peak_to_peak_range_uv",
                "event_waveform_dominant_frequency_hz",
                "event_waveform_duration_typical_sec",
                "event_waveform_duration_upper_sec",
            }
        ):
            return ClinicalTarget.EPILEPTIFORM_MORPHOLOGY
        if (
            target == ClinicalTarget.BACKGROUND_SLOWING
            and "background_amplitude_range_uv" in names
            and not names.intersection({"slowing_score", "background_dominant_frequency_hz"})
        ):
            return ClinicalTarget.BACKGROUND_AMPLITUDE
        return target

    def _report_linked_measurements(
        self,
        target: ClinicalTarget,
        measurements: List[MeasurementValue],
    ) -> List[MeasurementValue]:
        if target == ClinicalTarget.PDR:
            reportable = [
                measurement
                for measurement in measurements
                if measurement.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
                or "reactivity" in measurement.measurement_name.lower()
                or measurement.measurement_name in {"pdr_symmetry_status", "background_organization_status"}
            ]
            return reportable or measurements
        return measurements

    def _group_value(self, target: ClinicalTarget, measurements: List[MeasurementValue]) -> Dict[str, Any]:
        if target == ClinicalTarget.PDR:
            return self._pdr_group_value(measurements)
        if target == ClinicalTarget.BACKGROUND_AMPLITUDE:
            return {
                "background_amplitude_range_uv": self._measurement_value(
                    self._first_exact_name(measurements, "background_amplitude_range_uv")
                ),
                "background_amplitude_typical_uv": self._measurement_value(
                    self._first_exact_name(measurements, "background_amplitude_typical_uv")
                ),
                "background_amplitude_peak_to_peak_typical_uv": self._measurement_value(
                    self._first_exact_name(measurements, "background_amplitude_peak_to_peak_typical_uv")
                ),
                "background_amplitude_best_supported_uv": self._measurement_value(
                    self._first_exact_name(measurements, "background_amplitude_best_supported_uv")
                ),
            }

        values: Dict[str, Any] = {}
        for measurement in measurements:
            value = self._measurement_value(measurement)
            if value is None:
                continue
            values[measurement.measurement_name] = value
        return values

    def _first_exact_name(self, measurements: List[MeasurementValue], measurement_name: str) -> MeasurementValue | None:
        for measurement in measurements:
            if measurement.measurement_name == measurement_name:
                return measurement
        return None

    def _pdr_group_value(self, measurements: List[MeasurementValue]) -> Dict[str, Any]:
        """Keep reportable PDR values separate from support/debug measurements."""
        values: Dict[str, Any] = {}
        freq = self._first_exact_measurement_value(
            measurements,
            preferred_names=("pdr_v2_frequency_hz", "pdr_candidate_frequency_hz"),
        )
        if isinstance(freq, (int, float)):
            values["frequency_hz"] = float(freq)
            if 8.0 <= float(freq) <= 13.0 and self._has_posterior_provenance(measurements):
                values["pdr_supported"] = "true"
        symmetry = self._first_named_value(measurements, "pdr_symmetry_status")
        if isinstance(symmetry, str):
            values["symmetry"] = symmetry
        organization = self._first_named_value(measurements, "background_organization_status")
        if isinstance(organization, str):
            values["organization"] = organization
        values["reactivity"] = self._first_named_value(measurements, "reactivity") or "unknown"
        return values

    def _measurement_value(self, measurement: MeasurementValue | None) -> Any:
        if measurement is None:
            return None
        if measurement.quantitation is not None:
            q = measurement.quantitation
            if q.exact is not None:
                return q.exact
            if q.lower is not None or q.upper is not None:
                return {"lower": q.lower, "upper": q.upper}
            if q.values:
                return q.values[:20]
        if measurement.status_value is not None:
            return measurement.status_value.status.value
        if measurement.categorical_value is not None:
            return measurement.categorical_value
        if measurement.boolean_value is not None:
            return measurement.boolean_value
        return None

    def _group_unit(self, measurements: List[MeasurementValue]) -> str | None:
        clinical_units = {
            measurement.quantitation.unit
            for measurement in measurements
            if measurement.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
            and measurement.quantitation is not None
            and measurement.quantitation.unit
        }
        if len(clinical_units) == 1:
            return sorted(clinical_units)[0]
        units = {
            measurement.quantitation.unit
            for measurement in measurements
            if measurement.quantitation is not None and measurement.quantitation.unit
        }
        return sorted(units)[0] if len(units) == 1 else None

    def _first_exact_measurement_value(
        self,
        measurements: List[MeasurementValue],
        *,
        preferred_names: tuple[str, ...],
    ) -> Any:
        by_name = {measurement.measurement_name.lower(): measurement for measurement in measurements}
        for name in preferred_names:
            measurement = by_name.get(name.lower())
            if measurement is not None:
                return self._measurement_value(measurement)
        for measurement in measurements:
            if measurement.measurement_role != MeasurementRole.CLINICAL_MEASUREMENT:
                continue
            value = self._measurement_value(measurement)
            if value is not None:
                return value
        return None

    def _first_named_measurement(self, measurements: List[MeasurementValue], *needles: str) -> MeasurementValue | None:
        for measurement in measurements:
            name = measurement.measurement_name.lower()
            if all(needle in name for needle in needles):
                return measurement
        return None

    def _first_named_value(self, measurements: List[MeasurementValue], *needles: str) -> Any:
        for measurement in measurements:
            name = measurement.measurement_name.lower()
            if all(needle in name for needle in needles):
                value = self._measurement_value(measurement)
                if value is not None:
                    return value
        return None

    def _has_posterior_provenance(self, measurements: List[MeasurementValue]) -> bool:
        posterior_channels = {"o1", "o2", "oz", "p3", "p4", "pz"}
        for measurement in measurements:
            region = (measurement.provenance.space.region or "").lower()
            if region in {"posterior", "occipital", "parietal"}:
                return True
            if any(channel.lower() in posterior_channels for channel in measurement.provenance.space.channels):
                return True
            if "posterior" in measurement.measurement_name.lower():
                return True
        return False

    def _safe_evidence_id(self, evidence_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in evidence_id.strip())
        if not safe.startswith("ev"):
            safe = f"ev_llm_{safe}"
        return safe

    def _time_provenance(self, measurements: List[MeasurementValue]) -> Dict[str, Any] | None:
        windows: list[int] = []
        for measurement in measurements:
            windows.extend(measurement.provenance.time.window_indices)
        if not windows:
            return None
        return {"window_indices": sorted(set(windows)), "window_id": sorted(set(windows))[0]}

    def _space_provenance(self, measurements: List[MeasurementValue]) -> Dict[str, Any] | None:
        channels: list[str] = []
        regions: list[str] = []
        sides: list[str] = []
        for measurement in measurements:
            channels.extend(measurement.provenance.space.channels)
            if measurement.provenance.space.region:
                regions.append(measurement.provenance.space.region)
            if measurement.provenance.space.laterality:
                sides.append(measurement.provenance.space.laterality)
        if not channels and not regions and not sides:
            return None
        return {
            "channels": sorted(set(channels)),
            "region": sorted(set(regions))[0] if regions else None,
            "side": sorted(set(sides))[0] if sides else None,
            "electrode_maxima": sorted(set(channels))[:4],
        }
