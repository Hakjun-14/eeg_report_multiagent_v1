import csv
import json
import pickle
from pathlib import Path

import numpy as np

from eeg_report_multiagent.io.celm_dataset import (
    load_celm_split_sample,
    make_celm_generated_report,
    standardize_section_name,
)


def _write_fake_celm_row(root: Path) -> None:
    split_dir = root / "random_split_data_by_patient"
    report_id = "NeuroReport_fake_20260504"
    report_dir = root / "matched_eeg_recordings_report" / "S0001" / report_id
    session_dir = report_dir / "processed_eeg" / "sub-S0001fake_ses-1"
    split_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)

    report_payload = {
        "patient_history_section_llm_extractions": {
            "CLINICAL_sections": [
                {"section_name": "indication:", "section_text": "spells concerning for seizure"},
            ]
        },
        "EEG_section_llm_extractions": {
            "EEG_sections": [
                {"section_name": "detail:", "section_text": "THIS TARGET TEXT MUST NOT BE USED AS INPUT"},
            ]
        },
    }
    (report_dir / f"{report_id}.json").write_text(json.dumps(report_payload), encoding="utf-8")
    with (session_dir / "seg_0_sub-S0001fake_ses-1.pkl").open("wb") as f:
        pickle.dump({"signal": np.zeros((22, 2000), dtype="float32")}, f)

    row = {
        "DeidentifiedName(Reports)": f"{report_id}.txt",
        "BDSPPatientID": "p1",
        "VisitTypeDSC": "ROUTINE EEG",
        "SessionIDs": "sub-S0001fake_ses-1",
        "NumberOfSessions": "1",
        "Processed_EEG_Paths": "processed_eeg/sub-S0001fake_ses-1",
        "Extracted_EEG_sections": "detail:",
        "Extracted_Clinical_sections": "indication:",
        "Avg_Age": "21.0",
        "Gender": "Female",
    }
    with (split_dir / "S0001_test_split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def test_load_celm_split_sample_uses_celm_path_contract(tmp_path: Path) -> None:
    _write_fake_celm_row(tmp_path)

    sample = load_celm_split_sample(data_root=tmp_path, site="S0001", split="test", row_index=0)

    assert sample.report_id == "NeuroReport_fake_20260504"
    assert sample.report_json_path == tmp_path / "matched_eeg_recordings_report" / "S0001" / sample.report_id / f"{sample.report_id}.json"
    assert sample.session_dirs == [tmp_path / "matched_eeg_recordings_report" / "S0001" / sample.report_id / "processed_eeg" / "sub-S0001fake_ses-1"]
    assert sample.target_section_names_standardized == ["EEG DESCRIPTION/DETAILS"]
    assert "age: 21.0" in sample.clinical_history
    assert "gender: Female" in sample.clinical_history
    assert "spells concerning for seizure" in sample.clinical_history


def test_celm_study_context_does_not_include_target_eeg_text(tmp_path: Path) -> None:
    _write_fake_celm_row(tmp_path)

    sample = load_celm_split_sample(data_root=tmp_path, site="S0001", split="test", row_index=0)
    encoded = json.dumps(sample.study_context)

    assert "THIS TARGET TEXT MUST NOT BE USED AS INPUT" not in encoded
    assert "note_text" not in sample.study_context
    assert "report_json_path_eval_only" in sample.study_context["metadata"]
    assert sample.study_context["target_section_names"] == ["EEG DESCRIPTION/DETAILS"]


def test_celm_generated_report_uses_evaluator_shape() -> None:
    payload = make_celm_generated_report(
        target_section_names=["EEG DESCRIPTION/DETAILS", "IMPRESSION/INTERPRETATION"],
        detail_text="detail body",
        impression_text="impression body",
    )

    assert list(payload.keys()) == ["report_sections"]
    assert payload["report_sections"][0] == {"section_name": "EEG DESCRIPTION/DETAILS", "section_text": "detail body"}
    assert payload["report_sections"][1] == {"section_name": "IMPRESSION/INTERPRETATION", "section_text": "impression body"}


def test_celm_generated_report_prefers_section_specific_texts() -> None:
    payload = make_celm_generated_report(
        target_section_names=["BACKGROUND ACTIVITY", "EPLEPTIFORM ABNORMALITIES"],
        detail_text="generic detail",
        impression_text="generic impression",
        section_texts={
            "BACKGROUND ACTIVITY": "background-only",
            "EPLEPTIFORM ABNORMALITIES": "event-only",
        },
    )

    assert payload["report_sections"][0] == {"section_name": "BACKGROUND ACTIVITY", "section_text": "background-only"}
    assert payload["report_sections"][1] == {"section_name": "EPLEPTIFORM ABNORMALITIES", "section_text": "event-only"}


def test_section_standardization_matches_celm_detail_mapping() -> None:
    assert standardize_section_name("detail:") == "EEG DESCRIPTION/DETAILS"
    assert standardize_section_name("impression:") == "IMPRESSION/INTERPRETATION"
