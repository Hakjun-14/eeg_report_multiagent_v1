from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_artifact_dir(text: str) -> Path:
    path = Path(text)
    if path.exists():
        return path
    if text.startswith("/workspace/"):
        candidate = Path(text.replace("/workspace", "/home/hjlee/Desktop", 1))
        if candidate.exists():
            return candidate
    return path


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "row_index",
        "report_id",
        "section_name",
        "role",
        "generated_present",
        "generated_chars",
        "missing_required_slots",
        "missing_nullable_slots",
        "unsafe_candidate_route_to_seizures",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def audit_batch_sections(batch_root: Path, output_csv: Path) -> Dict[str, Any]:
    summary_rows = _read_csv_dicts(batch_root / "batch_summary.csv")
    out_rows: List[Dict[str, Any]] = []
    missing_audits = 0
    for row in summary_rows:
        artifact_dir = _resolve_artifact_dir(row.get("artifact_dir", ""))
        audit = _read_json(artifact_dir / "section_contract_audit.json")
        if not audit:
            missing_audits += 1
            continue
        for section in audit.get("sections") or []:
            out_rows.append(
                {
                    "row_index": row.get("row_index", ""),
                    "report_id": row.get("report_id", ""),
                    "section_name": section.get("section_name", ""),
                    "role": section.get("role", ""),
                    "generated_present": section.get("generated_present", ""),
                    "generated_chars": section.get("generated_chars", ""),
                    "missing_required_slots": "|".join(section.get("missing_required_slots") or []),
                    "missing_nullable_slots": "|".join(section.get("missing_nullable_slots") or []),
                    "unsafe_candidate_route_to_seizures": section.get("unsafe_candidate_route_to_seizures", ""),
                }
            )
    _write_csv(output_csv, out_rows)
    summary = {
        "batch_root": str(batch_root),
        "rows_in_batch_summary": len(summary_rows),
        "section_rows": len(out_rows),
        "missing_section_audit_files": missing_audits,
        "output_csv": str(output_csv),
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-section contract audits from a batch artifact root.")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    print(json.dumps(audit_batch_sections(Path(args.batch_root), Path(args.output_csv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
