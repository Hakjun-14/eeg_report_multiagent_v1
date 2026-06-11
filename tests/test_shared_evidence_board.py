from __future__ import annotations

import pytest
from pydantic import ValidationError

from eeg_report_multiagent.modules.evidence_item_adapter import build_shared_evidence_board
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.modules.surface_policy import SurfacePolicy
from eeg_report_multiagent.schemas import EvidenceBoard, MeasurementValue, QuantitationValue
from eeg_report_multiagent.schemas.measurement import MeasurementRole
from eeg_report_multiagent.schemas.measurement import QuantitationKind
from eeg_report_multiagent.schemas.provenance import MeasurementProvenance, ProvenanceRecord, SourceType, SpaceProvenance, TimeProvenance
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction, ReportSectionType, SurfaceDecision
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


FORBIDDEN = [
    "candidate burden",
    "longest candidate train",
    "laterality index",
    "bifrontal spread tendency",
    "morphology screen",
    "support score",
    "likelihood score",
    "field concentration ratio",
    "missing_slots",
]


def _prov(*, channels=None, region=None, side=None, windows=None, source_type=SourceType.SIGNAL) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type=source_type,
        source_ref="s",
        time=TimeProvenance(window_indices=windows or [0], start_sec=0.0, end_sec=10.0),
        space=SpaceProvenance(channels=channels or [], region=region, laterality=side),
        measurement=MeasurementProvenance(tool_name="tool", function_name="fn"),
    )


def _exact(mid: str, name: str, value: float, unit: str = "score", prov: ProvenanceRecord | None = None) -> MeasurementValue:
    return MeasurementValue(
        measurement_id=mid,
        measurement_name=name,
        quantitation=QuantitationValue(kind=QuantitationKind.EXACT, exact=value, unit=unit),
        provenance=prov or _prov(),
    )


def test_evidence_item_validation_rejects_missing_and_invalid_enums() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            source_module="event",
            evidence_type=EvidenceType.PROXY,
            clinical_target=ClinicalTarget.EVENT_CANDIDATE,
            reportability=ClaimSurfaceAction.DEBUG_ONLY,
            created_by="test",
        )
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="ev_bad_type",
            source_module="event",
            evidence_type="not_a_type",
            clinical_target=ClinicalTarget.EVENT_CANDIDATE,
            reportability=ClaimSurfaceAction.DEBUG_ONLY,
            created_by="test",
        )
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="ev_bad_reportability",
            source_module="event",
            evidence_type=EvidenceType.PROXY,
            clinical_target=ClinicalTarget.EVENT_CANDIDATE,
            reportability="maybe",
            created_by="test",
        )


def test_measurement_grouped_adapter_conservative_mappings() -> None:
    burden = _exact("m_burden", "event_candidate_burden_ratio", 0.2, "ratio")
    support = _exact("m_support", "event_morphology_support_score", 1.4, "score")
    ratio = _exact("m_field", "event_peak_field_concentration_ratio", 2.1, "ratio")
    pdr_prov = _prov(channels=["O1", "O2"], region="occipital", side="bilateral")
    pdr = _exact("m_pdr", "pdr_frequency_hz", 9.5, "Hz", prov=pdr_prov)
    pdr.metadata["pdr_supported"] = "true"

    board = build_shared_evidence_board(recording_id="s", measurements=[burden, support, ratio, pdr])

    by_id = {item.evidence_id: item for item in board.evidence_items}
    assert by_id["evgrp_event_candidate"].evidence_type == EvidenceType.PROXY
    assert by_id["evgrp_event_candidate"].clinical_target == ClinicalTarget.EVENT_CANDIDATE
    assert by_id["evgrp_event_candidate"].reportability == ClaimSurfaceAction.DEBUG_ONLY
    assert by_id["evgrp_epileptiform_morphology"].evidence_type == EvidenceType.DEBUG
    assert by_id["evgrp_epileptiform_morphology"].clinical_target == ClinicalTarget.EPILEPTIFORM_MORPHOLOGY
    assert by_id["evgrp_epileptiform_morphology"].reportability == ClaimSurfaceAction.DEBUG_ONLY
    assert by_id["evgrp_localization"].evidence_type == EvidenceType.PROXY
    assert by_id["evgrp_localization"].clinical_target == ClinicalTarget.LOCALIZATION
    assert by_id["evgrp_localization"].reportability == ClaimSurfaceAction.DEBUG_ONLY
    assert by_id["evgrp_pdr"].clinical_target == ClinicalTarget.PDR
    assert by_id["evgrp_pdr"].reportability == ClaimSurfaceAction.CAVEAT
    assert by_id["evgrp_pdr"].space_provenance["region"] == "occipital"


def test_pdr_group_splits_primary_frequency_from_proxy_support() -> None:
    pdr_prov = _prov(channels=["O1", "O2", "P3", "P4"], region="posterior", side="bilateral")
    freq = _exact("m_pdr_v2_frequency", "pdr_v2_frequency_hz", 9.5, "Hz", prov=pdr_prov)
    freq.measurement_role = MeasurementRole.CLINICAL_MEASUREMENT
    freq.metadata["pdr_supported"] = "true"
    freq.metadata["posterior_alpha_ratio"] = "0.07"
    freq.metadata["posterior_anterior_alpha_ratio"] = "1.4"
    support = _exact("m_pdr_v2_support_score", "pdr_v2_support_score", 0.64, "score", prov=pdr_prov)
    support.measurement_role = MeasurementRole.PROXY_SCORE
    ratio = _exact("m_pdr_v2_posterior_alpha_ratio", "pdr_v2_posterior_alpha_ratio", 0.07, "ratio", prov=pdr_prov)
    ratio.measurement_role = MeasurementRole.PROXY_SCORE

    board = build_shared_evidence_board(recording_id="s", measurements=[freq, support, ratio])
    by_id = {item.evidence_id: item for item in board.evidence_items}

    assert by_id["evgrp_pdr"].clinical_target == ClinicalTarget.PDR
    assert by_id["evgrp_pdr"].evidence_type == EvidenceType.DERIVED
    assert by_id["evgrp_pdr"].reportability == ClaimSurfaceAction.CAVEAT
    assert by_id["evgrp_pdr"].unit == "Hz"
    assert by_id["evgrp_pdr"].measurement_ids == ["m_pdr_v2_frequency"]
    assert by_id["evgrp_pdr"].value["frequency_hz"] == 9.5
    assert by_id["evgrp_pdr_support_debug"].evidence_type == EvidenceType.DEBUG
    assert by_id["evgrp_pdr_support_debug"].reportability == ClaimSurfaceAction.DEBUG_ONLY
    assert sorted(by_id["evgrp_pdr_support_debug"].measurement_ids) == [
        "m_pdr_v2_posterior_alpha_ratio",
        "m_pdr_v2_support_score",
    ]


def test_shared_evidence_board_queries_and_snapshot() -> None:
    item_debug = EvidenceItem(
        evidence_id="ev_debug",
        source_module="event",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.EVENT_CANDIDATE,
        value=0.2,
        unit="ratio",
        reportability=ClaimSurfaceAction.DEBUG_ONLY,
        created_by="test",
    )
    item_reportable = EvidenceItem(
        evidence_id="ev_reportable",
        source_module="metadata",
        evidence_type=EvidenceType.METADATA,
        clinical_target=ClinicalTarget.PROTOCOL,
        value="not_performed",
        reportability=ClaimSurfaceAction.ALLOW,
        allowed_sections=[SectionRole.DETAIL.value],
        created_by="test",
    )
    board = SharedEvidenceBoard(board_id="seb", recording_id="s")
    board.add_evidence(item_debug)
    board.add_evidence(item_reportable)

    assert board.get_evidence("ev_debug") == item_debug
    assert board.query_by_target("event_candidate") == [item_debug]
    assert board.query_by_section("detail") == [item_reportable]
    assert board.query_reportable("detail") == [item_reportable]
    assert board.query_debug_only() == [item_debug]
    assert board.query_for_surface_decisions(
        [
            SurfaceDecision(
                claim_id="p_status",
                surface_action=ClaimSurfaceAction.CAVEAT,
                allowed_sections=[SectionRole.DETAIL.value],
                rationale="SurfaceDecision is authoritative for report synthesis.",
                evidence_ids=["ev_debug"],
            )
        ],
        "detail",
    ) == [item_debug]
    snap = board.snapshot()
    assert snap.summary_by_type["proxy"] == 1
    assert snap.reportable_items == ["ev_reportable"]
    assert snap.debug_only_items == ["ev_debug"]


def test_atomic_claim_plan_links_to_shared_evidence_ids_and_validates_missing_links() -> None:
    amp = MeasurementValue(
        measurement_id="m_amp",
        measurement_name="background_amplitude_range_uv",
        quantitation=QuantitationValue(kind=QuantitationKind.RANGE, lower=70.0, upper=90.0, unit="uV"),
        provenance=_prov(),
    )
    board = EvidenceBoard(session_id="s", measurements=[amp])

    detail, _impression, claims = ReportSynthesizer().synthesize(board)
    plans = ReportSynthesizer().build_atomic_claim_plan(board)

    assert plans[0].evidence_ids == ["evgrp_background_amplitude"]
    assert board.ensure_shared_evidence_board().claim_evidence_links["c_p_evgrp_background_amplitude"] == ["evgrp_background_amplitude"]
    assert claims
    assert "background amplitude" in detail.text.lower()
    with pytest.raises(ValueError):
        board.ensure_shared_evidence_board().link_to_claim("bad_claim", ["ev_missing"])


def test_surface_policy_blocks_debug_proxy_evidence_items_and_seizure_event_candidate_route() -> None:
    item = EvidenceItem(
        evidence_id="ev_candidate",
        source_module="event",
        evidence_type=EvidenceType.PROXY,
        clinical_target=ClinicalTarget.EVENT_CANDIDATE,
        reportability=ClaimSurfaceAction.DEBUG_ONLY,
        created_by="test",
    )
    burden = _exact("m_burden", "event_candidate_burden_ratio", 0.2, "ratio")
    plan = AtomicClaimPlan(
        plan_id="p_ev_candidate",
        section_type=ReportSectionType.DETAIL,
        claim_type="event_candidate",
        proposed_text="No surface-allowed structured evidence was available for this section.",
        evidence_ids=[item.evidence_id],
        linked_measurement_ids=[burden.measurement_id],
        surface_action=ClaimSurfaceAction.BLOCK,
        rationale="Proxy event evidence must be claim-planned before surface use.",
    )
    decision = SurfacePolicy().decide(plan)

    assert decision.surface_action == ClaimSurfaceAction.BLOCK
    assert "ev_candidate" in decision.evidence_ids

    sections = ReportSynthesizer().synthesize_celm_sections(
        EvidenceBoard(session_id="s", measurements=[burden]),
        ["SEIZURES"],
    )
    assert sections["SEIZURES"] == "Seizures: no seizure-specific evidence was produced by the current structured tools."


def test_row189_style_shared_evidence_canary_remains_safe() -> None:
    measurements = [
        _exact("m_freq", "background_dominant_frequency_hz", 0.5, "Hz"),
        _exact("m_burden", "event_candidate_burden_ratio", 0.22, "ratio"),
        _exact("m_field", "event_peak_field_concentration_ratio", 2.4, "ratio"),
        _exact("m_like", "epileptiform_candidate_likelihood_score", 0.8, "score"),
    ]
    board = EvidenceBoard(session_id="row189_like", measurements=measurements)

    sections = ReportSynthesizer().synthesize_celm_sections(
        board,
        ["EEG DESCRIPTION/DETAILS", "EPLEPTIFORM ABNORMALITIES", "EVENTS/SEIZURES", "SEIZURES", "IMPRESSION/INTERPRETATION"],
    )

    rendered = "\n".join(sections.values()).lower()
    for snippet in FORBIDDEN:
        assert snippet not in rendered
    assert sections["SEIZURES"] == "Seizures: no seizure-specific evidence was produced by the current structured tools."
    assert board.ensure_shared_evidence_board().query_debug_only()
    assert not board.ensure_shared_evidence_board().query_reportable("seizures")
