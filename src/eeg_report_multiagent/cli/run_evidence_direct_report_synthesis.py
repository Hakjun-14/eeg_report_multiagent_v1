from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from eeg_report_multiagent.cli.run_d_synthesis_batch import (
    _clinical_context,
    _load_env_file,
    _read_csv_dicts,
    _read_json,
    _resolve_workspace_path,
    _target_sections,
    _write_csv,
    _write_json,
)
from eeg_report_multiagent.io import make_celm_generated_report
from eeg_report_multiagent.llm import OpenAIReportSynthesisAdapter
from eeg_report_multiagent.modules import EvidenceBoardLLMReportSynthesizer
from eeg_report_multiagent.schemas.evidence import EvidenceBoard


SUMMARY_COLUMNS = [
    "row_index",
    "report_id",
    "source_artifact_dir",
    "status",
    "elapsed_sec",
    "output_artifact_dir",
    "generated_json_path",
    "model_name",
    "evidence_selection_mode",
    "payload_mode",
    "surface_safe_evidence_count",
    "slot_count",
    "requested_evidence_ids",
    "raw_eeg_used",
    "gt_report_used",
    "gt_matched_selection_eval_only",
    "error_message",
]


def _gt_matched_evidence_ids(gt_audit_root: Path, row_index: str, report_id: str) -> List[str]:
    if not gt_audit_root.exists():
        raise FileNotFoundError(f"GT audit root does not exist: {gt_audit_root}")
    per_case = gt_audit_root / "per_case"
    if not per_case.exists():
        per_case = gt_audit_root
    if str(row_index).isdigit():
        exact = per_case / f"row_{int(row_index):06d}_{report_id}_gt_claim_audit.json"
        candidates = [exact] if exact.exists() else []
        if not candidates:
            candidates = sorted(per_case.glob(f"row_{int(row_index):06d}_{report_id}*_gt_claim_audit.json"))
    else:
        candidates = sorted(per_case.glob(f"*{report_id}*_gt_claim_audit.json"))
    if not candidates:
        raise FileNotFoundError(f"No GT claim audit JSON found for row={row_index} report_id={report_id} in {per_case}")
    payload = _read_json(candidates[0])
    out: List[str] = []
    seen: set[str] = set()
    for match in payload.get("gt_claim_matches") or []:
        for evidence_id in match.get("matched_evidence_ids") or []:
            evidence_id = str(evidence_id)
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                out.append(evidence_id)
    return out


def run_evidence_direct_report_synthesis_batch(
    source_batch_root: Path,
    output_root: Path,
    model: str | None,
    evidence_selection: str,
    payload_mode: str,
    gt_audit_root: Path | None,
    limit: int | None,
    force: bool,
) -> Dict[str, Any]:
    rows = _read_csv_dicts(source_batch_root / "batch_summary.csv")
    if limit is not None:
        rows = rows[:limit]

    results_dir = output_root / "celm_results"
    json_dir = results_dir / "generated_reports_json"
    txt_dir = results_dir / "generated_reports_txt"
    out_rows_dir = output_root / "rows"
    adapter = OpenAIReportSynthesisAdapter(model=model)
    synth = EvidenceBoardLLMReportSynthesizer(adapter=adapter)
    summary_rows: List[Dict[str, Any]] = []

    if evidence_selection == "gt_matched" and gt_audit_root is None:
        raise ValueError("--gt-audit-root is required when --evidence-selection gt_matched")

    for source_row in rows:
        start = time.time()
        report_id = source_row.get("report_id", "")
        row_index = source_row.get("row_index", "")
        source_artifact_dir = _resolve_workspace_path(source_row.get("artifact_dir") or source_row.get("output_artifact_dir", ""))
        output_artifact_dir = out_rows_dir / f"row_{int(row_index):06d}_{report_id}" if str(row_index).isdigit() else out_rows_dir / report_id
        generated_json_path = json_dir / f"GENERATED_REPORT_{report_id}.json"

        if generated_json_path.exists() and not force:
            summary_rows.append(
                {
                    "row_index": row_index,
                    "report_id": report_id,
                    "source_artifact_dir": str(source_artifact_dir),
                    "status": "skipped_existing",
                    "elapsed_sec": round(time.time() - start, 3),
                    "output_artifact_dir": str(output_artifact_dir),
                    "generated_json_path": str(generated_json_path),
                    "model_name": adapter.model,
                    "evidence_selection_mode": evidence_selection,
                    "payload_mode": payload_mode,
                    "surface_safe_evidence_count": "",
                    "slot_count": "",
                    "requested_evidence_ids": "",
                    "raw_eeg_used": "false",
                    "gt_report_used": "false",
                    "gt_matched_selection_eval_only": str(evidence_selection == "gt_matched").lower(),
                    "error_message": "",
                }
            )
            continue

        try:
            if source_row.get("status") not in {"ok", "skipped_existing"}:
                raise RuntimeError(f"Source row is not successful: {source_row.get('status')}")
            board = EvidenceBoard.model_validate_json((source_artifact_dir / "evidence_board.json").read_text(encoding="utf-8"))
            target_sections = _target_sections(source_artifact_dir)
            evidence_ids = None
            if evidence_selection == "gt_matched":
                evidence_ids = _gt_matched_evidence_ids(gt_audit_root or Path(""), row_index, report_id)

            result = synth.synthesize_evidence_direct_sections(
                board,
                target_sections,
                clinical_context=_clinical_context(source_artifact_dir),
                evidence_ids=evidence_ids,
                evidence_selection_mode=evidence_selection,
                payload_mode=payload_mode,
            )
            generated_report = make_celm_generated_report(
                target_section_names=target_sections,
                detail_text="",
                impression_text="",
                section_texts=result.section_texts,
            )

            output_artifact_dir.mkdir(parents=True, exist_ok=True)
            for filename in [
                "study_context.json",
                "evidence_board.json",
                "agent_deliberations.json",
                "atomic_claim_plan.json",
                "surface_decisions.json",
            ]:
                src = source_artifact_dir / filename
                if src.exists():
                    shutil.copy2(src, output_artifact_dir / filename)
            _write_json(output_artifact_dir / "evidence_direct_section_texts.json", result.section_texts)
            _write_json(output_artifact_dir / "evidence_direct_synthesis_trace.json", result.trace)
            _write_json(output_artifact_dir / "celm_generated_report.json", generated_report)
            _write_json(generated_json_path, generated_report)
            txt_dir.mkdir(parents=True, exist_ok=True)
            (txt_dir / f"GENERATED_REPORT_{report_id}.txt").write_text(
                json.dumps(generated_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            status = "ok"
            error_message = ""
            raw_eeg_used = str(result.trace["raw_eeg_used"]).lower()
            gt_report_used = str(result.trace["gt_report_used"]).lower()
            evidence_summary = result.trace.get("evidence_view_summary", {})
            surface_safe_count = evidence_summary.get("surface_safe_evidence_count", "")
            slot_count = evidence_summary.get("slot_count", "")
            requested_evidence_ids = evidence_summary.get("requested_evidence_ids", "")
        except Exception as exc:  # pragma: no cover - batch guard
            output_artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output_artifact_dir / "evidence_direct_error.json", {"error": repr(exc), "report_id": report_id})
            status = "error"
            error_message = repr(exc)
            raw_eeg_used = "false"
            gt_report_used = "false"
            surface_safe_count = ""
            slot_count = ""
            requested_evidence_ids = ""

        summary_rows.append(
            {
                "row_index": row_index,
                "report_id": report_id,
                "source_artifact_dir": str(source_artifact_dir),
                "status": status,
                "elapsed_sec": round(time.time() - start, 3),
                "output_artifact_dir": str(output_artifact_dir),
                "generated_json_path": str(generated_json_path),
                "model_name": adapter.model,
                "evidence_selection_mode": evidence_selection,
                "payload_mode": payload_mode,
                "surface_safe_evidence_count": surface_safe_count,
                "slot_count": slot_count,
                "requested_evidence_ids": requested_evidence_ids,
                "raw_eeg_used": raw_eeg_used,
                "gt_report_used": gt_report_used,
                "gt_matched_selection_eval_only": str(evidence_selection == "gt_matched").lower(),
                "error_message": error_message,
            }
        )
        _write_csv(output_root / "batch_summary.csv", summary_rows, SUMMARY_COLUMNS)
        print(
            f"[EvidenceDirect] {len(summary_rows)}/{len(rows)} row_index={row_index} report_id={report_id} status={status}",
            flush=True,
        )

    summary = {
        "source_batch_root": str(source_batch_root),
        "output_root": str(output_root),
        "model_name": adapter.model,
        "rows": len(summary_rows),
        "ok": sum(1 for row in summary_rows if row["status"] in {"ok", "skipped_existing"}),
        "errors": sum(1 for row in summary_rows if row["status"] == "error"),
        "generated_reports_json": str(json_dir),
        "method_variant": f"evidence_direct_{evidence_selection}",
        "payload_mode": payload_mode,
        "raw_eeg_used": False,
        "gt_report_used": False,
        "gt_matched_selection_eval_only": evidence_selection == "gt_matched",
    }
    _write_csv(output_root / "batch_summary.csv", summary_rows, SUMMARY_COLUMNS)
    _write_json(output_root / "batch_final_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diagnostic evidence-direct LLM report synthesis over an existing batch artifact root.")
    parser.add_argument("--source-batch-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--evidence-selection", choices=["all_safe", "gt_matched"], default="all_safe")
    parser.add_argument("--payload-mode", choices=["evidence_view", "slot_checklist"], default="evidence_view")
    parser.add_argument("--gt-audit-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _load_env_file(Path(args.env_file) if args.env_file else None)
    summary = run_evidence_direct_report_synthesis_batch(
        source_batch_root=Path(args.source_batch_root),
        output_root=Path(args.output_root),
        model=args.model,
        evidence_selection=args.evidence_selection,
        payload_mode=args.payload_mode,
        gt_audit_root=Path(args.gt_audit_root) if args.gt_audit_root else None,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
