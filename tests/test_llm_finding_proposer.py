from eeg_report_multiagent.modules.llm_finding_proposer import LLMFindingProposalModule
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.measurement import QuantitationKind, QuantitationValue, MeasurementValue
from eeg_report_multiagent.schemas.provenance import MeasurementProvenance, ProvenanceRecord, SourceType


class FakeProposalAdapter:
    model = "fake"

    def propose(self, payload):
        return {
            "summary": "fake proposal",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "finding_proposals": [
                {
                    "proposal_id": "p1",
                    "finding_type": "background_pdr_frequency",
                    "assertion": "present",
                    "confidence": 0.8,
                    "rationale": "linked to PDR measurement",
                    "linked_measurement_ids": ["m1"],
                    "provenance_policy": "measurement_linked_proposal_only",
                },
                {
                    "proposal_id": "p2",
                    "finding_type": "not_allowed",
                    "assertion": "present",
                    "confidence": 0.8,
                    "rationale": "bad",
                    "linked_measurement_ids": ["m1"],
                    "provenance_policy": "measurement_linked_proposal_only",
                },
            ],
        }


def test_llm_finding_proposer_validates_allowed_types_and_measurement_links():
    board = EvidenceBoard(
        session_id="s",
        measurements=[
            MeasurementValue(
                measurement_id="m1",
                measurement_name="pdr_candidate_frequency_hz",
                quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=10.0, unit="Hz"),
                provenance=ProvenanceRecord(
                    source_type=SourceType.SIGNAL,
                    measurement=MeasurementProvenance(tool_name="t", function_name="f"),
                ),
            )
        ],
    )

    result = LLMFindingProposalModule(adapter=FakeProposalAdapter()).run(board)

    assert result["status"] == "ok"
    assert result["finding_proposals"][0].accepted is True
    assert result["finding_proposals"][1].accepted is False
    assert result["finding_proposals"][1].rejection_reason.startswith("finding_type_not_allowed")
