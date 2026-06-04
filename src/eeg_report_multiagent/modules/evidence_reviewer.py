from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List

from eeg_report_multiagent.llm import OpenAIEvidenceReviewAdapter
from eeg_report_multiagent.schemas.agent import (
    AgentDeliberationRecord,
    ClaimConstraintRecord,
    DoNotClaimRecord,
    EvidenceGap,
    MissingSlotRecord,
    RejectedToolRequestProposal,
    ToolRequestProposal,
    WeakEvidenceRecord,
)
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.tools import build_background_registry, build_event_registry, build_parser_registry


class EvidenceReviewModule:
    """Rule+LLM review over structured evidence only.

    This module is intentionally placed after EvidenceBoard assembly. It never
    receives raw EEG arrays or GT report text; it can only critique typed
    measurements and suggest bounded local tools.
    """

    reviewer_name = "rule_plus_llm_evidence_reviewer"
    review_version = "v2"

    def __init__(self, adapter: OpenAIEvidenceReviewAdapter | None = None) -> None:
        self.adapter = adapter or OpenAIEvidenceReviewAdapter()

    def run(self, board: EvidenceBoard) -> AgentDeliberationRecord:
        payload = self._build_payload(board)
        available_tools = payload["available_tools"]
        local_review = self._local_structured_review(board, available_tools)

        try:
            llm_result = self.adapter.review(payload)
            valid_proposals, rejected_proposals = self._validate_tool_proposals(
                llm_result.get("tool_request_proposals", []),
                available_tools,
            )
            rejected_proposals.extend(local_review["rejected_tool_request_proposals"])
            summary_parts = [
                "Local evidence safety review completed.",
                str(llm_result.get("summary", "LLM evidence review completed.")),
            ]
            if rejected_proposals:
                summary_parts.append(
                    f"Rejected {len(rejected_proposals)} proposal(s) outside the bounded tool registry."
                )
            return AgentDeliberationRecord(
                review_id=f"review_{uuid.uuid4().hex}",
                reviewer_name=self.reviewer_name,
                review_version=self.review_version,
                status="ok",
                summary=" ".join(x.strip() for x in summary_parts if x.strip()),
                evidence_gaps=self._merge_by_id(
                    EvidenceGap,
                    "gap_id",
                    local_review["evidence_gaps"],
                    llm_result.get("evidence_gaps", []),
                ),
                weak_evidence=self._merge_by_id(
                    WeakEvidenceRecord,
                    "weakness_id",
                    local_review["weak_evidence"],
                    llm_result.get("weak_evidence", []),
                ),
                missing_slots=self._merge_by_id(
                    MissingSlotRecord,
                    "slot_id",
                    local_review["missing_slots"],
                    llm_result.get("missing_slots", []),
                ),
                do_not_claim=self._merge_by_id(
                    DoNotClaimRecord,
                    "item_id",
                    local_review["do_not_claim"],
                    llm_result.get("do_not_claim", []),
                ),
                claim_constraints=self._merge_by_id(
                    ClaimConstraintRecord,
                    "constraint_id",
                    local_review["claim_constraints"],
                    llm_result.get("claim_constraints", []),
                ),
                tool_request_proposals=local_review["tool_request_proposals"] + valid_proposals,
                rejected_tool_request_proposals=rejected_proposals,
                raw_eeg_used=False,
                gt_report_used=False,
                model_name=self.adapter.model,
            )
        except Exception as exc:
            return AgentDeliberationRecord(
                review_id=f"review_{uuid.uuid4().hex}",
                reviewer_name=self.reviewer_name,
                review_version=self.review_version,
                status="local_only",
                summary=(
                    "LLM evidence review was unavailable; deterministic local evidence safety review was retained."
                ),
                evidence_gaps=[EvidenceGap(**x) for x in local_review["evidence_gaps"]],
                weak_evidence=[WeakEvidenceRecord(**x) for x in local_review["weak_evidence"]],
                missing_slots=[MissingSlotRecord(**x) for x in local_review["missing_slots"]],
                do_not_claim=[DoNotClaimRecord(**x) for x in local_review["do_not_claim"]],
                claim_constraints=[ClaimConstraintRecord(**x) for x in local_review["claim_constraints"]],
                tool_request_proposals=local_review["tool_request_proposals"],
                rejected_tool_request_proposals=local_review["rejected_tool_request_proposals"],
                raw_eeg_used=False,
                gt_report_used=False,
                model_name=self.adapter.model,
                error_message=str(exc),
            )

    def _build_payload(self, board: EvidenceBoard) -> Dict[str, Any]:
        measurements_by_id = {m.measurement_id: m for m in board.measurements}

        measurements = []
        for m in board.measurements:
            q = m.quantitation
            provenance = m.provenance
            measurements.append(
                {
                    "measurement_id": m.measurement_id,
                    "measurement_name": m.measurement_name,
                    "payload_type": self._measurement_payload_type(m),
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
                    "status": m.status_value.status.value if m.status_value is not None else None,
                    "provenance_summary": {
                        "source_type": provenance.source_type.value,
                        "has_time": bool(
                            provenance.time.window_indices
                            or provenance.time.start_sec is not None
                            or provenance.time.end_sec is not None
                        ),
                        "has_space": bool(
                            provenance.space.channels
                            or provenance.space.region
                            or provenance.space.laterality
                        ),
                        "tool_name": provenance.measurement.tool_name if provenance.measurement else None,
                    },
                                    }
            )

        return {
            "session_id": board.session_id,
            "measurements": measurements,
            "measurement_count": len(measurements_by_id),
            "tool_invocations": [
                {
                    "tool_name": t.tool_name,
                    "module_name": t.module_name,
                    "status": t.status,
                    "output_measurement_ids": t.output_measurement_ids,
                }
                for t in board.tool_invocations
            ],
            "available_tools": self._available_tools(),
            "review_policy": {
                "allowed_role": "evidence_review_and_bounded_tool_policy_only",
                "forbidden": [
                    "raw_eeg_interpretation",
                    "gt_report_use",
                    "unsupported_new_evidence_creation",
                    "unregistered_tool_request",
                ],
                "preferred_outputs": [
                    "weak_evidence",
                    "missing_slots",
                    "do_not_claim",
                    "claim_constraints",
                    "tool_request_proposals",
                ],
            },
            "privacy_contract": {
                "contains_raw_eeg": False,
                "contains_gt_report_text": False,
                "contains_source_pkl_paths": False,
            },
        }

    def _available_tools(self) -> Dict[str, List[str]]:
        return {
            "background": build_background_registry().list_tools(),
            "event": build_event_registry().list_tools(),
            "parser": build_parser_registry().list_tools(),
        }

    def _local_structured_review(
        self,
        board: EvidenceBoard,
        available_tools: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        evidence_gaps: List[Dict[str, Any]] = []
        weak_evidence: List[Dict[str, Any]] = []
        missing_slots: List[Dict[str, Any]] = []
        do_not_claim: List[Dict[str, Any]] = []
        claim_constraints: List[Dict[str, Any]] = []
        proposals: List[ToolRequestProposal] = []
        rejected: List[RejectedToolRequestProposal] = []

        signal_measurements = [m for m in board.measurements if m.provenance.source_type.value == "signal"]
        missing_space = [m for m in signal_measurements if not (m.provenance.space.channels or m.provenance.space.region or m.provenance.space.laterality)]
        if missing_space:
            linked_measurements = sorted({m.measurement_id for m in missing_space})
            evidence_gaps.append(
                {
                    "gap_id": "gap_signal_spatial_provenance",
                    "evidence_target": "signal_spatial_provenance",
                    "severity": "high",
                    "reason": "Several signal-derived measurements lack channel, region, or laterality provenance.",
                    "linked_measurement_ids": linked_measurements,
                                    }
            )
            weak_evidence.append(
                {
                    "weakness_id": "weak_signal_spatial_provenance",
                    "severity": "high",
                    "target_type": "provenance",
                    "target_id": "signal_measurements",
                    "reason": "Time/tool provenance is present, but spatial provenance is incomplete for multiple signal measurements.",
                    "linked_measurement_ids": linked_measurements,
                                        "recommendation": "Avoid focal or lateralized claims unless a region/channel-aware tool supports them.",
                }
            )
            missing_slots.append(
                {
                    "slot_id": "slot_regional_signal_summary",
                    "slot_name": "regional_signal_summary",
                    "target_module": "background",
                    "severity": "high",
                    "reason": "Current background measurements are mostly global and do not localize slowing or amplitude changes.",
                    "expected_evidence": "region-wise bandpower/amplitude/slowing measurements with channel or region provenance",
                    "linked_measurement_ids": linked_measurements,
                                    }
            )
            claim_constraints.append(
                {
                    "constraint_id": "constraint_no_focal_without_space",
                    "target": "focal_or_lateralized_signal_claims",
                    "constraint": "Do not make focal, lateralized, or regional claims without spatial provenance.",
                    "rationale": "The evidence board cannot yet identify reliable channel/region/laterality for these measurements.",
                    "linked_measurement_ids": linked_measurements,
                                    }
            )

        dominant = self._measurement_by_name(board, "background_dominant_frequency_hz")
        if dominant and dominant.quantitation and dominant.quantitation.exact in {0.5, 30.0}:
            weak_evidence.append(
                {
                    "weakness_id": "weak_background_frequency_boundary",
                    "severity": "medium",
                    "target_type": "measurement",
                    "target_id": dominant.measurement_id,
                    "reason": "Dominant frequency lies exactly on the spectral search boundary.",
                    "linked_measurement_ids": [dominant.measurement_id],
                                        "recommendation": "Treat this as a weak global spectral hint, not a clinical posterior dominant rhythm estimate.",
                }
            )
            claim_constraints.append(
                {
                    "constraint_id": "constraint_pdr_boundary_language",
                    "target": "background_frequency",
                    "constraint": "Use cautious language for background frequency when the estimate is at a search boundary.",
                    "rationale": "Boundary estimates can reflect drift/artifact or poor frequency localization.",
                                    }
            )

        if "focality_bifrontal_summary" in available_tools.get("event", []):
            proposals.append(
                ToolRequestProposal(
                    proposal_id="proposal_event_bifrontal_recheck",
                    target_module="event",
                    tool_name="focality_bifrontal_summary",
                    rationale="Bifrontal spread should be kept bounded to the registered local event tool.",
                    expected_measurement="event_bifrontal_ratio with channel provenance",
                    linked_gap_ids=["gap_signal_spatial_provenance"] if missing_space else [],
                )
            )

        return {
            "evidence_gaps": evidence_gaps,
            "weak_evidence": weak_evidence,
            "missing_slots": missing_slots,
            "do_not_claim": do_not_claim,
            "claim_constraints": claim_constraints,
            "tool_request_proposals": proposals,
            "rejected_tool_request_proposals": rejected,
        }

    def _validate_tool_proposals(
        self,
        proposals: List[Dict[str, Any]],
        available_tools: Dict[str, List[str]],
    ) -> tuple[List[ToolRequestProposal], List[RejectedToolRequestProposal]]:
        valid: List[ToolRequestProposal] = []
        rejected: List[RejectedToolRequestProposal] = []
        for proposal in proposals:
            target_module = str(proposal.get("target_module", ""))
            tool_name = str(proposal.get("tool_name", ""))
            if tool_name in available_tools.get(target_module, []):
                valid.append(ToolRequestProposal(**proposal))
                continue
            rejected.append(
                RejectedToolRequestProposal(
                    **proposal,
                    rejection_reason=(
                        f"tool '{tool_name}' is not registered under module '{target_module}'"
                    ),
                )
            )
        return valid, rejected

    def _merge_by_id(self, model_cls, id_field: str, *groups: Iterable[Dict[str, Any]]) -> list[Any]:
        out: dict[str, Any] = {}
        for group in groups:
            for item in group or []:
                if not isinstance(item, dict) or id_field not in item:
                    continue
                key = str(item[id_field])
                if key not in out:
                    out[key] = model_cls(**item)
        return list(out.values())

    def _measurement_payload_type(self, measurement: MeasurementValue) -> str:
        if measurement.quantitation is not None:
            return "quantitation"
        if measurement.status_value is not None:
            return "status"
        if measurement.categorical_value is not None:
            return "categorical"
        if measurement.boolean_value is not None:
            return "boolean"
        return "unknown"

    def _measurement_by_name(self, board: EvidenceBoard, name: str) -> MeasurementValue | None:
        for measurement in board.measurements:
            if measurement.measurement_name == name:
                return measurement
        return None
