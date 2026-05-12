from __future__ import annotations

from pathlib import Path

from eeg_report_multiagent.evaluation.clinical_provenance_audit import audit_case


def test_clinical_provenance_audit_flags_near_exact_reference_match() -> None:
    case_payload = {
        "row_index": 783,
        "report_id": "NeuroReport_test",
        "patient_id": "patient-1",
        "target_sections": ["EEG DESCRIPTION/DETAILS"],
        "gt_sections": {
            "EEG DESCRIPTION/DETAILS": "Awake EEG shows a posterior dominant rhythm. No seizures are recorded.",
        },
        "variants": {
            "CELM": {
                "generated_sections": {
                    "EEG DESCRIPTION/DETAILS": "Awake EEG shows a posterior dominant rhythm. No seizures are recorded.",
                },
                "section_comparisons": [
                    {
                        "section_name": "EEG DESCRIPTION/DETAILS",
                        "gt_text": "Awake EEG shows a posterior dominant rhythm. No seizures are recorded.",
                        "generated_text": "Awake EEG shows a posterior dominant rhythm. No seizures are recorded.",
                        "missing_concepts": [],
                        "extra_concepts": [],
                        "numeric_missing": {},
                        "numeric_extra": {},
                    }
                ],
                "aggregate": {"rougeL": 1.0, "meteor": 1.0, "concept_f1_mean": 1.0},
            }
        },
    }

    audit = audit_case(
        case_payload,
        slot_schema={},
        failure_taxonomy={"version": "test"},
        claim_gate_policy={"version": "test"},
        artifact_roots={},
        variants=["CELM"],
    )

    decisions = [card["decision"] for card in audit["claim_cards"]]
    assert "possible_leakage_or_memorization" in decisions
    leakage_rows = [row for row in audit["critical_slot_table"] if row["slot"] == "leakage_audit"]
    assert leakage_rows
    assert leakage_rows[0]["CELM"] == "possible_leakage_or_memorization"


def test_clinical_provenance_audit_marks_reference_miss_as_over_cautious() -> None:
    case_payload = {
        "row_index": 548,
        "report_id": "NeuroReport_test2",
        "patient_id": "patient-2",
        "target_sections": ["EEG DESCRIPTION/DETAILS"],
        "gt_sections": {"EEG DESCRIPTION/DETAILS": "The recording contains epileptiform spike-wave runs."},
        "variants": {
            "Our_B": {
                "generated_sections": {"EEG DESCRIPTION/DETAILS": "Transient candidates were detected."},
                "section_comparisons": [
                    {
                        "section_name": "EEG DESCRIPTION/DETAILS",
                        "gt_text": "The recording contains epileptiform spike-wave runs.",
                        "generated_text": "Transient candidates were detected.",
                        "missing_concepts": ["event:epileptiform", "event:runs"],
                        "extra_concepts": [],
                        "numeric_missing": {},
                        "numeric_extra": {},
                    }
                ],
                "aggregate": {"rougeL": 0.1, "meteor": 0.1, "concept_f1_mean": 0.0},
            }
        },
    }

    audit = audit_case(
        case_payload,
        slot_schema={},
        failure_taxonomy={"version": "test"},
        claim_gate_policy={"version": "test"},
        artifact_roots={},
        variants=["Our_B"],
    )

    cards = audit["claim_cards"]
    assert {card["decision"] for card in cards} == {"over_cautious_false_negative"}
    assert {card["slot"] for card in cards} == {"epileptiform_morphology", "burst_duration"}
