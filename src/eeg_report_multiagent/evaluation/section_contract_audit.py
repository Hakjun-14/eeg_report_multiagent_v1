from __future__ import annotations

from typing import Any, Dict, List

from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.section_contract import TargetSectionContract


def audit_section_contract(
    contract: TargetSectionContract,
    board: EvidenceBoard,
    generated_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit section coverage without using GT section text."""

    router = SectionRouter()
    shared_board = board.ensure_shared_evidence_board()
    evidence_targets = {
        str(getattr(item.clinical_target, "value", item.clinical_target))
        for item in shared_board.evidence_items
    }
    generated_sections = {}
    for section in generated_report.get("report_sections") or []:
        name = str(section.get("section_name") or "").strip()
        generated_sections[name.upper()] = str(section.get("section_text") or "")

    section_rows: List[Dict[str, Any]] = []
    for section in contract.target_sections:
        generated_text = generated_sections.get(section.standardized_name.upper(), "")
        required_coverage = router.covered_slots(evidence_targets, section.required_slots)
        missing_required = [
            req.slot_name
            for req in section.required_slots
            if req.required and not req.nullable and not required_coverage.get(req.slot_name, False)
        ]
        missing_nullable = [
            req.slot_name
            for req in section.required_slots
            if req.nullable and not required_coverage.get(req.slot_name, False)
        ]
        text_lower = generated_text.lower()
        unsafe_candidate_route = (
            section.role.value == "seizures"
            and "candidate transient burden" in text_lower
        )
        section_rows.append(
            {
                "section_name": section.standardized_name,
                "role": section.role.value,
                "generated_present": bool(generated_text.strip()),
                "generated_chars": len(generated_text),
                "required_slot_coverage": required_coverage,
                "missing_required_slots": missing_required,
                "missing_nullable_slots": missing_nullable,
                "unsafe_candidate_route_to_seizures": unsafe_candidate_route,
                "generation_policy": section.generation_policy,
            }
        )

    return {
        "contract_id": contract.contract_id,
        "report_id": contract.report_id,
        "reference_text_allowed_as_inference_input": contract.reference_text_allowed_as_inference_input,
        "target_section_count": len(contract.target_sections),
        "generated_section_count": len(generated_report.get("report_sections") or []),
        "sections": section_rows,
        "missing_required_slot_count": sum(len(row["missing_required_slots"]) for row in section_rows),
        "unsafe_candidate_route_count": sum(1 for row in section_rows if row["unsafe_candidate_route_to_seizures"]),
        "all_target_sections_generated": all(row["generated_present"] for row in section_rows),
    }
