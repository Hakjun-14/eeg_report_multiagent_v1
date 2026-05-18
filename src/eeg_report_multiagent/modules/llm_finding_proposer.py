from __future__ import annotations

from typing import Any, Dict, List

from eeg_report_multiagent.llm import OpenAIFindingProposalAdapter
from eeg_report_multiagent.schemas.agent import FindingProposalRecord
from eeg_report_multiagent.schemas.evidence import EvidenceBoard


ALLOWED_FINDING_TYPES = {
    "background_pdr_frequency",
    "background_ap_organization",
    "background_reactivity",
    "sleep_architecture",
    "background_slowing",
    "excess_beta",
    "epileptiform_candidate_likelihood",
    "electrographic_seizure_likelihood",
    "event_morphology_support",
    "event_focality_bifrontal_spread",
}


class LLMFindingProposalModule:
    """Optional ablation: LLM proposes finding labels from typed measurements only."""

    def __init__(self, adapter: OpenAIFindingProposalAdapter | None = None) -> None:
        self.adapter = adapter or OpenAIFindingProposalAdapter()

    def run(self, board: EvidenceBoard) -> Dict[str, Any]:
        payload = self._build_payload(board)
        try:
            result = self.adapter.propose(payload)
            proposals = self._validate(result.get("finding_proposals", []), board)
            return {
                "status": "ok",
                "model_name": self.adapter.model,
                "summary": str(result.get("summary", "")),
                "raw_eeg_used": bool(result.get("raw_eeg_used", False)),
                "gt_report_used": bool(result.get("gt_report_used", False)),
                "finding_proposals": proposals,
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "model_name": self.adapter.model,
                "summary": "LLM finding proposal was unavailable; no proposal findings were applied.",
                "raw_eeg_used": False,
                "gt_report_used": False,
                "finding_proposals": [],
                "error_message": str(exc),
            }

    def _build_payload(self, board: EvidenceBoard) -> Dict[str, Any]:
        existing_types = sorted({f.finding_type for f in board.findings})
        measurements: List[Dict[str, Any]] = []
        for measurement in board.measurements:
            q = measurement.quantitation
            measurements.append(
                {
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
                    "metadata": measurement.metadata,
                    "provenance": {
                        "source_type": measurement.provenance.source_type.value,
                        "tool_name": measurement.provenance.measurement.tool_name if measurement.provenance.measurement else None,
                        "has_time": bool(measurement.provenance.time.window_indices),
                        "has_space": bool(
                            measurement.provenance.space.channels
                            or measurement.provenance.space.region
                            or measurement.provenance.space.laterality
                        ),
                    },
                }
            )
        return {
            "session_id": board.session_id,
            "allowed_finding_types": sorted(ALLOWED_FINDING_TYPES),
            "existing_finding_types": existing_types,
            "measurements": measurements,
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
            },
        }

    def _validate(self, proposals: List[Dict[str, Any]], board: EvidenceBoard) -> List[FindingProposalRecord]:
        measurement_ids = {m.measurement_id for m in board.measurements}
        out: List[FindingProposalRecord] = []
        for raw in proposals:
            if not isinstance(raw, dict):
                continue
            linked = [str(x) for x in raw.get("linked_measurement_ids", []) if str(x) in measurement_ids]
            finding_type = str(raw.get("finding_type", ""))
            accepted = finding_type in ALLOWED_FINDING_TYPES and bool(linked) and not bool(raw.get("raw_eeg_used"))
            rejection = None
            if finding_type not in ALLOWED_FINDING_TYPES:
                rejection = f"finding_type_not_allowed:{finding_type}"
            elif not linked:
                rejection = "no_valid_linked_measurement_ids"
            out.append(
                FindingProposalRecord(
                    proposal_id=str(raw.get("proposal_id", f"proposal_{len(out)}")),
                    finding_type=finding_type,
                    assertion=str(raw.get("assertion", "unknown")),
                    confidence=float(raw.get("confidence", 0.0)),
                    rationale=str(raw.get("rationale", "")),
                    linked_measurement_ids=linked,
                    provenance_policy=str(raw.get("provenance_policy", "measurement_linked_proposal_only")),
                    accepted=accepted,
                    rejection_reason=rejection,
                )
            )
        return out
