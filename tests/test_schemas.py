from eeg_report_multiagent.schemas import (
    AtomicClaimPlan,
    ClaimRecord,
    ClaimSurfaceAction,
    EvidenceBoard,
    MeasurementValue,
    ProvenanceRecord,
    QuantitationValue,
    ReportSection,
    ToolInvocationRecord,
    VerificationRecord,
)
from eeg_report_multiagent.schemas.measurement import QuantitationKind, StatusSemantic, StatusValue
from eeg_report_multiagent.schemas.provenance import SourceType
from eeg_report_multiagent.schemas.report import ClaimSupportLabel, ReportSectionType


def test_schema_instantiation_smoke() -> None:
    prov = ProvenanceRecord(source_type=SourceType.SIGNAL, source_ref="session_x")
    q = QuantitationValue(kind=QuantitationKind.EXACT, exact=10.0, unit="Hz")
    meas = MeasurementValue(
        measurement_id="m1",
        measurement_name="background_dominant_frequency_hz",
        quantitation=q,
        provenance=prov,
    )
    board = EvidenceBoard(session_id="s1", measurements=[meas])

    detail = ReportSection(section_type=ReportSectionType.DETAIL, text="x", claim_ids=["c1"])
    claim = ClaimRecord(claim_id="c1", section_type=ReportSectionType.DETAIL, text="x", linked_evidence_ids=["ev1"])
    verify = VerificationRecord(
        claim_id="c1",
        support_label=ClaimSupportLabel.SUPPORTED,
        evidence_ids=["ev1"],
    )
    inv = ToolInvocationRecord(invocation_id="i1", tool_name="t", module_name="m")
    plan = AtomicClaimPlan(
        plan_id="p1",
        section_type=ReportSectionType.DETAIL,
        claim_type="background_frequency",
        proposed_text="Background frequency is present.",
        linked_measurement_ids=["m1"],
        evidence_ids=["ev1"],
        required_evidence=["signal_measurement_provenance"],
        surface_action=ClaimSurfaceAction.ALLOW,
    )

    assert detail.claim_ids == ["c1"]
    assert plan.surface_action == ClaimSurfaceAction.ALLOW
    assert claim.linked_evidence_ids == ["ev1"]
    assert verify.support_label == ClaimSupportLabel.SUPPORTED
    assert inv.tool_name == "t"
    assert board.measurements[0].measurement_id == "m1"


def test_status_measurement_schema() -> None:
    prov = ProvenanceRecord(source_type=SourceType.REPORT_TEXT, source_ref="report")
    meas = MeasurementValue(
        measurement_id="m_status",
        measurement_name="photic_stimulation_status",
        status_value=StatusValue(status=StatusSemantic.NOT_PERFORMED, reason="na"),
        provenance=prov,
    )
    assert meas.status_value.status == StatusSemantic.NOT_PERFORMED
