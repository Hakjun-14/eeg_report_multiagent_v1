from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.schemas.section_contract import TargetSectionContract


SECTION_STANDARDIZATION_MAPPING = {
    "details:": "EEG DESCRIPTION/DETAILS",
    "detail:": "EEG DESCRIPTION/DETAILS",
    "description:": "EEG DESCRIPTION/DETAILS",
    "impression:": "IMPRESSION/INTERPRETATION",
    "interpretation:": "IMPRESSION/INTERPRETATION",
    "background:": "BACKGROUND ACTIVITY",
    "background activity:": "BACKGROUND ACTIVITY",
    "seizures:": "SEIZURES",
    "events/seizures:": "EVENTS/SEIZURES",
    "epileptiform abnormalities:": "EPLEPTIFORM ABNORMALITIES",
    "interictal epileptiform abnormalities:": "INTERICTAL EPLEPTIFORM ABNORMALITIES",
    "sleep:": "SLEEP",
}


@dataclass(frozen=True)
class CELMSplitSample:
    data_root: Path
    site: str
    split_type: str
    split: str
    row_index: int
    row: Dict[str, str]
    report_id: str
    report_dir: Path
    report_json_path: Path
    processed_eeg_paths: List[str]
    session_dirs: List[Path]
    target_section_names_raw: List[str]
    target_section_names_standardized: List[str]
    target_section_contract: TargetSectionContract
    clinical_history: str
    study_context: Dict[str, Any]


def split_csv_path(data_root: Path, site: str, split: str, split_type: str = "random_split_data_by_patient") -> Path:
    return data_root / split_type / f"{site}_{split}_split.csv"


def read_split_rows(data_root: Path, site: str, split: str, split_type: str = "random_split_data_by_patient") -> List[Dict[str, str]]:
    path = split_csv_path(data_root, site=site, split=split, split_type=split_type)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report_id_from_row(row: Dict[str, str]) -> str:
    return row["DeidentifiedName(Reports)"].replace(".txt", "")


def processed_paths_from_row(row: Dict[str, str]) -> List[str]:
    return [x.strip() for x in str(row.get("Processed_EEG_Paths") or "").split(",") if x.strip()]


def target_sections_from_row(row: Dict[str, str]) -> List[str]:
    sections = [x.strip() for x in str(row.get("Extracted_EEG_sections") or "").split(",") if x.strip()]
    return sections or ["detail:"]


def standardize_section_name(section_name: str) -> str:
    key = section_name.strip().lower()
    return SECTION_STANDARDIZATION_MAPPING.get(key, section_name.strip())


def build_clinical_history(row: Dict[str, str], report_json: Dict[str, Any]) -> str:
    age = row.get("Avg_Age", row.get("AgeAtVisit", ""))
    gender = row.get("Gender", row.get("SexDSC", ""))
    clinical_history = f"age: {age}\ngender: {gender}\n"
    history_payload = report_json.get("patient_history_section_llm_extractions") or {}
    for section in history_payload.get("CLINICAL_sections") or []:
        section_name = str(section.get("section_name") or "").strip()
        section_text = str(section.get("section_text") or "").strip()
        if section_name or section_text:
            clinical_history += f"{section_name}\n{section_text}\n\n"
    return clinical_history


def load_celm_split_sample(
    data_root: Path,
    site: str,
    split: str,
    row_index: int,
    split_type: str = "random_split_data_by_patient",
) -> CELMSplitSample:
    rows = read_split_rows(data_root=data_root, site=site, split=split, split_type=split_type)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index out of range: {row_index} for {len(rows)} rows")

    row = rows[row_index]
    report_id = report_id_from_row(row)
    report_dir = data_root / "matched_eeg_recordings_report" / site / report_id
    report_json_path = report_dir / f"{report_id}.json"
    processed_paths = processed_paths_from_row(row)
    session_dirs = [report_dir / p for p in processed_paths]

    if not report_json_path.exists():
        raise FileNotFoundError(f"CELM report JSON not found: {report_json_path}")
    missing_sessions = [str(p) for p in session_dirs if not p.exists()]
    if missing_sessions:
        raise FileNotFoundError(f"CELM processed EEG session path(s) not found: {missing_sessions}")

    report_json = json.loads(report_json_path.read_text(encoding="utf-8"))
    target_raw = target_sections_from_row(row)
    target_std = [standardize_section_name(name) for name in target_raw]
    target_contract = SectionRouter().build_contract(
        report_id=report_id,
        target_section_names_raw=target_raw,
        target_section_names_standardized=target_std,
        eval_only_reference_json_path=str(report_json_path),
    )
    clinical_history = build_clinical_history(row, report_json)

    metadata = dict(row)
    metadata.update(
        {
            "site": site,
            "split": split,
            "split_type": split_type,
            "row_index": str(row_index),
            "report_id": report_id,
            "report_dir": str(report_dir),
            "report_json_path_eval_only": str(report_json_path),
            "clinical_history": clinical_history,
            "target_section_names": json.dumps(target_std, ensure_ascii=False),
            "target_section_contract_id": target_contract.contract_id,
            "input_contract": (
                "CELM-compatible clinical context only. EEG target section text is not included "
                "as inference input."
            ),
        }
    )
    study_context = {
        "context_type": "celm_harvard_compatible_study_context",
        "clinical_history": clinical_history,
        "target_section_names": target_std,
        "target_section_contract_id": target_contract.contract_id,
        "metadata": metadata,
    }

    return CELMSplitSample(
        data_root=data_root,
        site=site,
        split_type=split_type,
        split=split,
        row_index=row_index,
        row=row,
        report_id=report_id,
        report_dir=report_dir,
        report_json_path=report_json_path,
        processed_eeg_paths=processed_paths,
        session_dirs=session_dirs,
        target_section_names_raw=target_raw,
        target_section_names_standardized=target_std,
        target_section_contract=target_contract,
        clinical_history=clinical_history,
        study_context=study_context,
    )


def make_celm_generated_report(
    target_section_names: List[str],
    detail_text: str,
    impression_text: str,
    section_texts: Dict[str, str] | None = None,
) -> Dict[str, List[Dict[str, str]]]:
    sections = []
    for section_name in target_section_names:
        normalized = section_name.strip().upper()
        if section_texts and section_name in section_texts:
            section_text = section_texts[section_name]
        elif normalized == "IMPRESSION/INTERPRETATION":
            section_text = impression_text
        elif normalized == "EEG DESCRIPTION/DETAILS":
            section_text = detail_text
        elif normalized in {
            "BACKGROUND ACTIVITY",
            "EPLEPTIFORM ABNORMALITIES",
            "INTERICTAL EPLEPTIFORM ABNORMALITIES",
            "EVENTS/SEIZURES",
            "SEIZURES",
            "SLEEP",
        }:
            section_text = detail_text
        else:
            section_text = detail_text
        sections.append({"section_name": section_name, "section_text": section_text})
    return {"report_sections": sections}
