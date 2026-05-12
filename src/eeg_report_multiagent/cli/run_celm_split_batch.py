from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from eeg_report_multiagent.io import read_split_rows, report_id_from_row


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in text)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "report_id",
        "patient_id",
        "status",
        "elapsed_sec",
        "artifact_dir",
        "generated_json_path",
        "error_message",
        "llm_review_status",
        "llm_model_name",
        "measurements",
        "findings",
        "claims",
        "weak_evidence_records",
        "missing_slot_records",
        "do_not_claim_records",
        "claim_constraint_records",
        "overall_pass",
        "input_contract_pass",
        "target_section_count",
        "section_missing_required_slots",
        "unsafe_candidate_routes",
        "all_target_sections_generated",
        "llm_finding_proposal_status",
        "llm_finding_proposals",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_row_indices_file(path: Path) -> List[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [int(row["row_index"]) for row in reader if row.get("row_index")]
    out: List[int] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(int(line))
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _summarize_success(row_index: int, row: Dict[str, str], artifact_dir: Path, celm_results_dir: Path | None) -> Dict[str, Any]:
    report_id = report_id_from_row(row)
    audit = _load_json(artifact_dir / "method_audit.json")
    deliberations = []
    delib_payload = json.loads((artifact_dir / "agent_deliberations.json").read_text(encoding="utf-8")) if (artifact_dir / "agent_deliberations.json").exists() else []
    if isinstance(delib_payload, list):
        deliberations = [x for x in delib_payload if isinstance(x, dict)]
    first_delib = deliberations[0] if deliberations else {}
    generated_json_path = ""
    if celm_results_dir is not None:
        generated_json_path = str(celm_results_dir / "generated_reports_json" / f"GENERATED_REPORT_{report_id}.json")

    counts = audit.get("counts", {}) if isinstance(audit.get("counts"), dict) else {}
    input_contract = audit.get("input_contract", {}) if isinstance(audit.get("input_contract"), dict) else {}
    section_audit = _load_json(artifact_dir / "section_contract_audit.json")
    proposal_payload = _load_json(artifact_dir / "llm_finding_proposals.json")
    return {
        "row_index": row_index,
        "report_id": report_id,
        "patient_id": row.get("BDSPPatientID", ""),
        "status": "ok",
        "artifact_dir": str(artifact_dir),
        "generated_json_path": generated_json_path,
        "error_message": "",
        "llm_review_status": first_delib.get("status", ""),
        "llm_model_name": first_delib.get("model_name", ""),
        "measurements": counts.get("measurements", ""),
        "findings": counts.get("findings", ""),
        "claims": counts.get("claims", ""),
        "weak_evidence_records": counts.get("weak_evidence_records", ""),
        "missing_slot_records": counts.get("missing_slot_records", ""),
        "do_not_claim_records": counts.get("do_not_claim_records", ""),
        "claim_constraint_records": counts.get("claim_constraint_records", ""),
        "overall_pass": audit.get("overall_pass", ""),
        "input_contract_pass": input_contract.get("pass", ""),
        "target_section_count": section_audit.get("target_section_count", ""),
        "section_missing_required_slots": section_audit.get("missing_required_slot_count", ""),
        "unsafe_candidate_routes": section_audit.get("unsafe_candidate_route_count", ""),
        "all_target_sections_generated": section_audit.get("all_target_sections_generated", ""),
        "llm_finding_proposal_status": proposal_payload.get("status", ""),
        "llm_finding_proposals": len(proposal_payload.get("finding_proposals", []) or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CELM-compatible split batch through eeg_report_multiagent_v1")
    parser.add_argument("--data-root", required=True, help="CELM data root, e.g. /workspace/eeg_data/celm_s_sites_pipeline")
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows. Omit for all rows from start-row.")
    parser.add_argument("--row-indices-file", default=None, help="Optional txt/csv file containing explicit row_index values to run")
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--output-root", required=True, help="Batch artifact root. Per-row artifacts are written under rows/.")
    parser.add_argument("--celm-results-dir", default=None, help="CELM-compatible generated report root.")
    parser.add_argument("--no-langgraph", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--enable-llm-review", action="store_true")
    parser.add_argument("--enable-llm-finding-proposals", action="store_true")
    parser.add_argument("--enable-local-encoder", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip rows that already have method_audit.json and celm_generated_report.json")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Optional sleep between rows, useful for API rate limiting")
    parser.add_argument("--row-timeout-sec", type=int, default=0, help="Optional per-row subprocess timeout. 0 disables timeout.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    rows_root = output_root / "rows"
    celm_results_dir = Path(args.celm_results_dir) if args.celm_results_dir else output_root / "celm_results"
    output_root.mkdir(parents=True, exist_ok=True)
    rows_root.mkdir(parents=True, exist_ok=True)

    split_rows = read_split_rows(data_root=data_root, site=args.site, split=args.split, split_type=args.split_type)
    if args.row_indices_file:
        indices = _read_row_indices_file(Path(args.row_indices_file))
        if args.max_rows is not None:
            indices = indices[: args.max_rows]
        stop = None
    else:
        stop = len(split_rows) if args.max_rows is None else min(len(split_rows), args.start_row + args.max_rows)
        indices = list(range(args.start_row, stop))
    invalid = [i for i in indices if i < 0 or i >= len(split_rows)]
    if invalid:
        raise IndexError(f"row index out of range for split with {len(split_rows)} rows: {invalid[:10]}")

    batch_config = {
        "data_root": str(data_root),
        "site": args.site,
        "split_type": args.split_type,
        "split": args.split,
        "split_row_count": len(split_rows),
        "start_row": args.start_row,
        "stop_row_exclusive": stop,
        "row_indices_file": args.row_indices_file,
        "requested_rows": len(indices),
        "variant": (
            "local_encoder_plus_rule_llm_review"
            if args.enable_local_encoder and args.enable_llm_review
            else "local_encoder_plus_rule"
            if args.enable_local_encoder
            else "rule_plus_llm_review"
            if args.enable_llm_review
            else "rule_only"
        ),
        "use_langgraph": not args.no_langgraph,
        "verify_claims": not args.no_verify,
        "enable_local_encoder": args.enable_local_encoder,
        "enable_llm_finding_proposals": args.enable_llm_finding_proposals,
        "celm_results_dir": str(celm_results_dir),
        "resume": args.resume,
    }
    _write_json(output_root / "batch_config.json", batch_config)

    summary_rows: List[Dict[str, Any]] = []
    print("Running CELM-compatible batch")
    print(json.dumps(batch_config, ensure_ascii=False, indent=2))

    for n, row_index in enumerate(indices, start=1):
        row = split_rows[row_index]
        report_id = report_id_from_row(row)
        row_dir = rows_root / f"row_{row_index:06d}_{_safe_name(report_id)}"
        audit_path = row_dir / "method_audit.json"
        generated_path = row_dir / "celm_generated_report.json"
        print(f"[{n}/{len(indices)}] row={row_index} report_id={report_id}", flush=True)

        start_ts = time.time()
        if args.resume and audit_path.exists() and generated_path.exists():
            item = _summarize_success(row_index, row, row_dir, celm_results_dir)
            item["status"] = "skipped_existing"
            item["elapsed_sec"] = 0.0
            summary_rows.append(item)
            _write_csv(output_root / "batch_summary.csv", summary_rows)
            _write_json(output_root / "batch_summary.json", {"config": batch_config, "rows": summary_rows})
            continue

        cmd = [
            sys.executable,
            "-m",
            "eeg_report_multiagent.cli.run_celm_split_session",
            "--data-root",
            str(data_root),
            "--site",
            args.site,
            "--split-type",
            args.split_type,
            "--split",
            args.split,
            "--row-index",
            str(row_index),
            "--session-index",
            str(args.session_index),
            "--output-dir",
            str(row_dir),
            "--celm-results-dir",
            str(celm_results_dir),
        ]
        if args.no_langgraph:
            cmd.append("--no-langgraph")
        if args.no_verify:
            cmd.append("--no-verify")
        if args.enable_llm_review:
            cmd.append("--enable-llm-review")
        if args.enable_llm_finding_proposals:
            cmd.append("--enable-llm-finding-proposals")
        if args.enable_local_encoder:
            cmd.append("--enable-local-encoder")

        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=args.row_timeout_sec if args.row_timeout_sec > 0 else None,
            )
            item = _summarize_success(row_index, row, row_dir, celm_results_dir)
            item["elapsed_sec"] = round(time.time() - start_ts, 3)
        except Exception as exc:
            item = {
                "row_index": row_index,
                "report_id": report_id,
                "patient_id": row.get("BDSPPatientID", ""),
                "status": "error",
                "elapsed_sec": round(time.time() - start_ts, 3),
                "artifact_dir": str(row_dir),
                "generated_json_path": "",
                "error_message": str(exc),
            }
            print(f"ERROR row={row_index}: {exc}", file=sys.stderr, flush=True)
            if args.stop_on_error:
                summary_rows.append(item)
                _write_csv(output_root / "batch_summary.csv", summary_rows)
                _write_json(output_root / "batch_summary.json", {"config": batch_config, "rows": summary_rows})
                raise

        summary_rows.append(item)
        _write_csv(output_root / "batch_summary.csv", summary_rows)
        _write_json(output_root / "batch_summary.json", {"config": batch_config, "rows": summary_rows})
        if args.sleep_sec > 0 and n < len(indices):
            time.sleep(args.sleep_sec)

    ok = sum(1 for r in summary_rows if r.get("status") in {"ok", "skipped_existing"})
    errors = sum(1 for r in summary_rows if r.get("status") == "error")
    final_summary = {
        "config": batch_config,
        "completed_rows": len(summary_rows),
        "ok_or_skipped": ok,
        "errors": errors,
        "summary_csv": str(output_root / "batch_summary.csv"),
        "summary_json": str(output_root / "batch_summary.json"),
        "celm_results_dir": str(celm_results_dir),
    }
    _write_json(output_root / "batch_final_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
