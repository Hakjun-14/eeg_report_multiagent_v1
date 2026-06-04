from eeg_report_multiagent.modules.evidence_board import EvidenceBoardAssembler
from eeg_report_multiagent.schemas.measurement import MeasurementValue, QuantitationKind, QuantitationValue
from eeg_report_multiagent.schemas.provenance import ProvenanceRecord, SourceType


def test_evidence_board_merge() -> None:
    prov = ProvenanceRecord(source_type=SourceType.SIGNAL, source_ref="s")
    meas = MeasurementValue(
        measurement_id="m1",
        measurement_name="slowing_score",
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=1.2, unit="ratio"),
        provenance=prov,
    )
    board = EvidenceBoardAssembler().merge(
        session_id="s",
        measurement_groups=[[meas]],
        tool_invocation_groups=[[]],
    )
    assert board.session_id == "s"
    assert len(board.measurements) == 1
    shared = board.ensure_shared_evidence_board()
    assert len(shared.evidence_items) == 1
    assert shared.evidence_items[0].measurement_ids == ["m1"]
