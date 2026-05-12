from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_report_multiagent.io import load_celm_split_sample, read_split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CELM-compatible split/path contract")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--row-index", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = read_split_rows(data_root=data_root, site=args.site, split=args.split, split_type=args.split_type)
    sample = load_celm_split_sample(
        data_root=data_root,
        site=args.site,
        split=args.split,
        row_index=args.row_index,
        split_type=args.split_type,
    )
    payload = {
        "data_root": str(data_root),
        "site": args.site,
        "split_type": args.split_type,
        "split": args.split,
        "row_count": len(rows),
        "row_index": args.row_index,
        "report_id": sample.report_id,
        "patient_id": sample.row.get("BDSPPatientID", ""),
        "visit_type": sample.row.get("VisitTypeDSC", ""),
        "number_of_sessions": sample.row.get("NumberOfSessions", ""),
        "report_dir": str(sample.report_dir),
        "report_json_path": str(sample.report_json_path),
        "report_json_exists": sample.report_json_path.exists(),
        "processed_eeg_paths": sample.processed_eeg_paths,
        "session_dirs": [str(p) for p in sample.session_dirs],
        "session_dirs_exist": [p.exists() for p in sample.session_dirs],
        "pkl_counts": [len(list(p.glob("seg_*.pkl"))) for p in sample.session_dirs],
        "target_sections_raw": sample.target_section_names_raw,
        "target_sections_standardized": sample.target_section_names_standardized,
        "clinical_history_preview": sample.clinical_history[:500],
        "required_columns_present": {
            name: name in sample.row
            for name in [
                "DeidentifiedName(Reports)",
                "BDSPPatientID",
                "VisitTypeDSC",
                "SessionIDs",
                "NumberOfSessions",
                "Processed_EEG_Paths",
                "Extracted_EEG_sections",
                "Extracted_Clinical_sections",
                "Avg_Age",
                "Gender",
            ]
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
