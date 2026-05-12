import json

from eeg_report_multiagent.evaluation.section_contract_audit import audit_section_contract
from eeg_report_multiagent.io.celm_dataset import load_celm_split_sample
from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.section_contract import SectionRole

from test_celm_dataset_loader import _write_fake_celm_row


def test_target_section_contract_does_not_include_gt_text(tmp_path):
    _write_fake_celm_row(tmp_path)

    sample = load_celm_split_sample(data_root=tmp_path, site="S0001", split="test", row_index=0)
    encoded = sample.target_section_contract.model_dump_json()

    assert sample.target_section_contract.reference_text_allowed_as_inference_input is False
    assert "THIS TARGET TEXT MUST NOT BE USED AS INPUT" not in encoded
    assert sample.target_section_contract.target_sections[0].standardized_name == "EEG DESCRIPTION/DETAILS"


def test_section_router_treats_seizures_separately():
    router = SectionRouter()

    assert router.role_for_section("SEIZURES").value == "seizures"
    assert router.role_for_section("EVENTS/SEIZURES").value == "events_seizures"
    assert router.generation_policy(router.role_for_section("SEIZURES")).startswith("Do not claim")


def test_section_contract_audit_flags_candidate_route_to_seizures():
    router = SectionRouter()
    contract = router.build_contract(
        report_id="NeuroReport_fake",
        target_section_names_raw=["seizures:"],
        target_section_names_standardized=["SEIZURES"],
    )
    board = EvidenceBoard(session_id="s1")
    generated = {
        "report_sections": [
            {
                "section_name": "SEIZURES",
                "section_text": "candidate transient burden was routed into seizures",
            }
        ]
    }

    audit = audit_section_contract(contract, board, generated)

    assert audit["target_section_count"] == 1
    assert audit["unsafe_candidate_route_count"] == 1
    assert json.dumps(audit)


def test_report_synthesizer_preserves_events_seizures_role_value():
    text = ReportSynthesizer()._event_section_text(EvidenceBoard(session_id="s1"), {}, {}, SectionRole.EVENTS_SEIZURES)

    assert text.startswith("Events/seizures:")
    assert "Push-button event status" in text
