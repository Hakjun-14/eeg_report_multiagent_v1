from eeg_report_multiagent.modules.evidence_board import EvidenceBoardAssembler
from eeg_report_multiagent.schemas.finding import Finding
from eeg_report_multiagent.schemas.measurement import MeasurementValue, QuantitationKind, QuantitationValue, StatusSemantic
from eeg_report_multiagent.schemas.provenance import ProvenanceRecord, SourceType


def test_evidence_board_merge() -> None:
    prov = ProvenanceRecord(source_type=SourceType.SIGNAL, source_ref="s")
    meas = MeasurementValue(
        measurement_id="m1",
        measurement_name="slowing_score",
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=1.2, unit="ratio"),
        provenance=prov,
    )
    finding = Finding(
        finding_id="f1",
        finding_type="background_slowing",
        assertion=StatusSemantic.PRESENT,
        measurement_ids=["m1"],
        provenance=[prov],
    )

    board = EvidenceBoardAssembler().merge(
        session_id="s",
        measurement_groups=[[meas]],
        finding_groups=[[finding]],
        tool_invocation_groups=[[]],
    )
    assert board.session_id == "s"
    assert len(board.measurements) == 1
    assert board.index["background_slowing"] == ["f1"]
