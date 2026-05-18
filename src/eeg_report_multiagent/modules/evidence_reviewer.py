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
from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.tools import build_background_registry, build_event_registry, build_parser_registry


class EvidenceReviewModule:
    """Rule+LLM review over structured evidence only.

    This module is intentionally placed after EvidenceBoard assembly. It never
    receives raw EEG arrays or GT report text; it can only critique typed
    measurements/findings and suggest bounded local tools.
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
        findings: List[Dict[str, Any]] = []
        for finding in board.findings:
            q = finding.quantitation
            linked_measurements = [measurements_by_id[mid] for mid in finding.measurement_ids if mid in measurements_by_id]
            provenance = [m.provenance for m in linked_measurements] or list(finding.provenance)
            source_types = sorted({str(p.source_type.value) for p in provenance})
            has_time = any(p.time.window_indices or p.time.start_sec is not None or p.time.end_sec is not None for p in provenance)
            has_space = any(p.space.channels or p.space.region or p.space.laterality for p in provenance)
            has_measurement_provenance = any(p.measurement is not None for p in provenance)
            findings.append(
                {
                    "finding_id": finding.finding_id,
                    "finding_type": finding.finding_type,
                    "assertion": finding.assertion.value,
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
                    "measurement_ids": finding.measurement_ids,
                    "provenance_summary": {
                        "source_types": source_types,
                        "provenance_count": len(provenance),
                        "has_time": has_time,
                        "has_space": has_space,
                        "has_measurement_provenance": has_measurement_provenance,
                    },
                }
            )

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
                    "linked_finding_ids": [
                        f.finding_id for f in board.findings if m.measurement_id in f.measurement_ids
                    ],
                }
            )

        return {
            "session_id": board.session_id,
            "findings": findings,
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
                    "unsupported_new_finding_creation",
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

        findings_by_type = {f.finding_type: f for f in board.findings}
        measurements_by_id = {m.measurement_id: m for m in board.measurements}
        signal_findings = [f for f in board.findings if self._is_signal_finding(f, measurements_by_id)]
        missing_space = [f for f in signal_findings if not self._has_space_provenance(f, measurements_by_id)]
        if missing_space:
            linked = [f.finding_id for f in missing_space]
            evidence_gaps.append(
                {
                    "gap_id": "gap_signal_spatial_provenance",
                    "finding_type": "signal_spatial_provenance",
                    "severity": "high",
                    "reason": "Several signal-derived findings lack channel, region, or laterality provenance.",
                    "linked_finding_ids": linked,
                }
            )
            weak_evidence.append(
                {
                    "weakness_id": "weak_signal_spatial_provenance",
                    "severity": "high",
                    "target_type": "provenance",
                    "target_id": "signal_findings",
                    "reason": "Time/tool provenance is present, but spatial provenance is incomplete for multiple signal findings.",
                    "linked_measurement_ids": self._measurement_ids_for_findings(missing_space),
                    "linked_finding_ids": linked,
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
                    "linked_finding_ids": linked,
                }
            )
            claim_constraints.append(
                {
                    "constraint_id": "constraint_no_focal_without_space",
                    "target": "focal_or_lateralized_signal_claims",
                    "constraint": "Do not make focal, lateralized, or regional claims without spatial provenance.",
                    "rationale": "The evidence board cannot yet identify reliable channel/region/laterality for these findings.",
                    "linked_finding_ids": linked,
                }
            )

        dominant = self._measurement_by_name(board, "background_dominant_frequency_hz")
        if dominant and dominant.quantitation and dominant.quantitation.exact in {0.5, 30.0}:
            linked_findings = self._linked_finding_ids(board, dominant.measurement_id)
            weak_evidence.append(
                {
                    "weakness_id": "weak_background_frequency_boundary",
                    "severity": "medium",
                    "target_type": "measurement",
                    "target_id": dominant.measurement_id,
                    "reason": "Dominant frequency lies exactly on the spectral search boundary.",
                    "linked_measurement_ids": [dominant.measurement_id],
                    "linked_finding_ids": linked_findings,
                    "recommendation": "Treat this as a weak global spectral hint, not a clinical posterior dominant rhythm estimate.",
                }
            )
            claim_constraints.append(
                {
                    "constraint_id": "constraint_pdr_boundary_language",
                    "target": "background_frequency",
                    "constraint": "Use cautious language for background frequency when the estimate is at a search boundary.",
                    "rationale": "Boundary estimates can reflect drift/artifact or poor frequency localization.",
                    "linked_finding_ids": linked_findings,
                }
            )

        event_burden = findings_by_type.get("epileptiform_event_candidate_burden")
        morphology_support = findings_by_type.get("event_morphology_support")
        has_morphology_proxy = morphology_support is not None and morphology_support.assertion.value == "present"
        if event_burden is not None and not has_morphology_proxy:
            missing_slots.append(
                {
                    "slot_id": "slot_event_morphology",
                    "slot_name": "event_morphology",
                    "target_module": "event",
                    "severity": "high",
                    "reason": "Event burden is estimated, but spike/sharp morphology evidence is not present.",
                    "expected_evidence": "focused morphology descriptors such as sharpness, duration, field, and after-going slow component",
                    "linked_finding_ids": [event_burden.finding_id],
                }
            )
            do_not_claim.append(
                {
                    "item_id": "do_not_claim_definite_epileptiform",
                    "text": "Do not claim definite epileptiform discharges from candidate burden alone.",
                    "rationale": "The current evidence board has event-candidate screening evidence but no morphology confirmation.",
                    "linked_finding_ids": [event_burden.finding_id],
                }
            )
            claim_constraints.append(
                {
                    "constraint_id": "constraint_event_candidate_language",
                    "target": "event_findings",
                    "constraint": "Use event-like or candidate language unless morphology-specific evidence is available.",
                    "rationale": "Burden/duration/laterality screening scores do not establish epileptiform morphology.",
                    "linked_finding_ids": [event_burden.finding_id],
                }
            )
        elif event_burden is not None and has_morphology_proxy:
            linked = [event_burden.finding_id, morphology_support.finding_id]
            weak_evidence.append(
                {
                    "weakness_id": "weak_event_morphology_proxy_not_classifier",
                    "severity": "medium",
                    "target_type": "finding",
                    "target_id": morphology_support.finding_id,
                    "reason": "Local morphology encoder evidence is a proxy feature summary, not a validated epileptiform morphology classifier.",
                    "linked_measurement_ids": self._measurement_ids_for_findings([morphology_support]),
                    "linked_finding_ids": linked,
                    "recommendation": "Use morphology-support wording and keep definite epileptiform claims provisional.",
                }
            )
            claim_constraints.append(
                {
                    "constraint_id": "constraint_morphology_proxy_language",
                    "target": "event_findings",
                    "constraint": "Local morphology-feature encoder support may strengthen event-candidate language but does not prove definite epileptiform discharges or seizures.",
                    "rationale": "The encoder tool is local and bounded, but not yet calibrated as a clinical epileptiform classifier.",
                    "linked_finding_ids": linked,
                }
            )
            do_not_claim.append(
                {
                    "item_id": "do_not_claim_definite_epileptiform",
                    "text": "Do not claim definite epileptiform discharges from morphology proxy evidence alone.",
                    "rationale": "The current evidence board has local encoder morphology support, but not a validated clinical epileptiform classifier.",
                    "linked_finding_ids": linked,
                }
            )
            claim_constraints.append(
                {
                    "constraint_id": "constraint_event_candidate_language",
                    "target": "event_findings",
                    "constraint": "Use event-like or morphology-supported candidate language unless a validated morphology classifier or human review confirms epileptiform morphology.",
                    "rationale": "The local encoder strengthens candidate characterization but does not establish clinical epileptiform morphology by itself.",
                    "linked_finding_ids": linked,
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
                    linked_finding_ids=[f.finding_id for f in signal_findings if f.finding_type.startswith("event") or f.finding_type.startswith("epileptiform")],
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

    def _is_signal_finding(self, finding: Finding, measurements_by_id: Dict[str, MeasurementValue]) -> bool:
        provenance = self._linked_provenance(finding, measurements_by_id)
        return any(p.source_type.value == "signal" for p in provenance)

    def _has_space_provenance(self, finding: Finding, measurements_by_id: Dict[str, MeasurementValue]) -> bool:
        provenance = self._linked_provenance(finding, measurements_by_id)
        return any(p.space.channels or p.space.region or p.space.laterality for p in provenance)

    def _linked_provenance(self, finding: Finding, measurements_by_id: Dict[str, MeasurementValue]):
        records = [measurements_by_id[mid].provenance for mid in finding.measurement_ids if mid in measurements_by_id]
        return records or list(finding.provenance)

    def _measurement_ids_for_findings(self, findings: Iterable[Finding]) -> List[str]:
        ids: List[str] = []
        for finding in findings:
            ids.extend(finding.measurement_ids)
        return sorted(set(ids))

    def _measurement_by_name(self, board: EvidenceBoard, name: str) -> MeasurementValue | None:
        for measurement in board.measurements:
            if measurement.measurement_name == name:
                return measurement
        return None

    def _linked_finding_ids(self, board: EvidenceBoard, measurement_id: str) -> List[str]:
        return [f.finding_id for f in board.findings if measurement_id in f.measurement_ids]
