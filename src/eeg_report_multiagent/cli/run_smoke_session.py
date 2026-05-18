from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


METADATA_COLUMNS = [
    "DeidentifiedName(Reports)",
    "BDSPPatientID",
    "SessionIDs",
    "NumberOfSessions",
    "MatchedSavePath",
    "VisitTypeDSC",
    "ProcedureDSC",
    "RecordType",
    "AgeAtVisit",
    "SexDSC",
    "ProcedureDSC(Reports)",
    "Extracted_EEG_sections",
    "Empty_EEG_sections",
    "Number_of_Extracted_EEG_sections",
    "Number_of_Empty_EEG_sections",
    "Extracted_Clinical_sections",
    "Empty_Clinical_sections",
    "Number_of_Extracted_Clinical_sections",
    "Number_of_Empty_Clinical_sections",
    "Processed_EEG_Paths",
]


def _read_rows(split_csv: Path) -> List[Dict[str, str]]:
    with split_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _report_id(row: Dict[str, str]) -> str:
    return row["DeidentifiedName(Reports)"].replace(".txt", "")


def _processed_paths(row: Dict[str, str]) -> List[str]:
    return [x.strip() for x in (row.get("Processed_EEG_Paths") or "").split(",") if x.strip()]


def _write_study_context(row: Dict[str, str], session_index: int, session_dir: Path, out_dir: Path) -> Path:
    rid = _report_id(row)
    metadata = {k: row.get(k, "") for k in METADATA_COLUMNS if k in row}
    metadata.update(
        {
            "site": "S0001",
            "report_id": rid,
            "selected_session_index": str(session_index),
            "selected_session_dir": str(session_dir),
            "input_contract": "Study context only. GT report text is not included as inference input.",
        }
    )
    payload = {
        "context_type": "celm_smoke_study_context",
        "metadata": metadata,
    }
    path = out_dir / "study_context.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CELM S0001 smoke split row/session without GT report as inference input")
    parser.add_argument("--split-csv", required=True, help="Path to S0001 split CSV")
    parser.add_argument("--row-index", type=int, default=0, help="Zero-based row index in split CSV")
    parser.add_argument("--session-index", type=int, default=0, help="Zero-based session index after Processed_EEG_Paths explode")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument("--monitor", action="store_true", help="Show terminal monitor while evidence board is built")
    parser.add_argument("--no-langgraph", action="store_true", help="Force sequential fallback runner")
    parser.add_argument("--no-verify", action="store_true", help="Disable optional claim verification")
    parser.add_argument("--enable-llm-evidence-grouping", action="store_true", help="Use LLM to group typed measurements into EvidenceItems")
    parser.add_argument("--enable-llm-review", action="store_true", help="Run optional evidence-board-only LLM review")
    args = parser.parse_args()

    split_csv = Path(args.split_csv)
    rows = _read_rows(split_csv)
    if args.row_index < 0 or args.row_index >= len(rows):
        raise IndexError(f"row-index out of range: {args.row_index} for {len(rows)} rows")

    row = rows[args.row_index]
    rid = _report_id(row)
    processed_paths = _processed_paths(row)
    if args.session_index < 0 or args.session_index >= len(processed_paths):
        raise IndexError(f"session-index out of range: {args.session_index} for {len(processed_paths)} sessions")

    base = Path(row["MatchedSavePath"])
    session_dir = base / processed_paths[args.session_index]
    gt_report_json = base / f"{rid}.json"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study_context_json = _write_study_context(row, args.session_index, session_dir, out_dir)

    if not session_dir.exists():
        raise FileNotFoundError(f"session_dir not found: {session_dir}")

    cmd = [
        sys.executable,
        "-m",
        "eeg_report_multiagent.cli.run_session",
        "--session-dir",
        str(session_dir),
        "--study-context-json",
        str(study_context_json),
        "--gt-report-json",
        str(gt_report_json),
        "--output-dir",
        str(out_dir),
    ]
    if args.monitor:
        cmd.append("--monitor")
    if args.no_langgraph:
        cmd.append("--no-langgraph")
    if args.no_verify:
        cmd.append("--no-verify")
    if args.enable_llm_evidence_grouping:
        cmd.append("--enable-llm-evidence-grouping")
    if args.enable_llm_review:
        cmd.append("--enable-llm-review")

    print("Running smoke session")
    print(f"  split_csv: {split_csv}")
    print(f"  row_index: {args.row_index}")
    print(f"  report_id: {rid}")
    print(f"  session_index: {args.session_index}")
    print(f"  session_dir: {session_dir}")
    print(f"  study_context_json: {study_context_json}")
    print(f"  gt_report_json_eval_only: {gt_report_json}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
