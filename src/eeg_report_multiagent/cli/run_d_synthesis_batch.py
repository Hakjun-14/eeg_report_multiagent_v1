from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from eeg_report_multiagent.io import make_celm_generated_report
from eeg_report_multiagent.llm import OpenAIReportSynthesisAdapter
from eeg_report_multiagent.modules import EvidenceBoardLLMReportSynthesizer
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.report import AtomicClaimPlan


SUMMARY_COLUMNS = [
    "row_index",
    "report_id",
    "source_artifact_dir",
    "status",
    "elapsed_sec",
    "output_artifact_dir",
    "generated_json_path",
    "model_name",
    "raw_eeg_used",
    "gt_report_used",
    "error_message",
]


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_workspace_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    project_root = _project_root()
    text = str(path)
    if text.startswith("/workspace/eeg_report_multiagent_v1"):
        candidate = Path(text.replace("/workspace/eeg_report_multiagent_v1", str(project_root), 1))
        if candidate.exists():
            return candidate
    if text.startswith("/workspace/"):
        candidate = Path(text.replace("/workspace", str(project_root.parent), 1))
        if candidate.exists():
            return candidate
    return path


def _target_sections(artifact_dir: Path) -> List[str]:
    context = _read_json(artifact_dir / "study_context.json")
    sections = context.get("target_section_names") or []
    if not isinstance(sections, list):
        raise ValueError(f"target_section_names is not a list in {artifact_dir / 'study_context.json'}")
    return [str(section) for section in sections]


_EVAL_ONLY_METADATA_KEYS = {
    "report_json_path_eval_only",
    "gt_report_json_path",
    "gt_report_text_path",
    "reference_report_path",
    "reference_gt_report_text",
}

_SAFE_CLINICAL_METADATA_KEYS = {
    "AgeAtVisit",
    "Avg_Age",
    "Gender",
    "SexDSC",
    "ProcedureDSC",
    "ProcedureDSC(Reports)",
    "VisitTypeDSC",
    "RecordType",
    "NumberOfSessions",
    "site",
    "split",
    "split_type",
}


def _clinical_context(artifact_dir: Path) -> Dict[str, Any]:
    context = _read_json(artifact_dir / "study_context.json")
    metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
    safe_metadata = {
        str(key): "" if value is None else str(value)
        for key, value in metadata.items()
        if str(key) in _SAFE_CLINICAL_METADATA_KEYS and str(key) not in _EVAL_ONLY_METADATA_KEYS
    }
    clinical_text = _sanitize_patient_history_text(str(
        context.get("clinical_history")
        or context.get("patient_history")
        or metadata.get("clinical_history")
        or ""
    ))
    target_sections = context.get("target_section_names") or metadata.get("target_section_names") or []
    return {
        "patient_history_and_eeg_description": str(clinical_text),
        "target_section_names": target_sections if isinstance(target_sections, list) else str(target_sections),
        "context_type": str(context.get("context_type", "")),
        "metadata": safe_metadata,
        "gt_report_text_included": False,
    }


def _sanitize_patient_history_text(text: str) -> str:
    """Keep patient/history context; strip GT EEG detail/impression text."""
    lowered = text.lower()
    cut = len(text)
    for marker in (
        "\nmethod:",
        " method:",
        "\ndetail:",
        " detail:",
        "\neeg description",
        " eeg description",
        "\nimpression",
        " impression:",
        "\ncomparison:",
        " comparison:",
        "\nstart time:",
        " start time:",
    ):
        idx = lowered.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def _atomic_claim_plan(artifact_dir: Path) -> List[AtomicClaimPlan] | None:
    path = artifact_dir / "atomic_claim_plan.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"atomic_claim_plan.json is not a list in {artifact_dir}")
    return [AtomicClaimPlan.model_validate(item) for item in payload]


def run_d_synthesis_batch(
    source_batch_root: Path,
    output_root: Path,
    model: str | None,
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

    for source_row in rows:
        start = time.time()
        report_id = source_row.get("report_id", "")
        row_index = source_row.get("row_index", "")
        source_artifact_dir = _resolve_workspace_path(source_row.get("artifact_dir", ""))
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
                    "raw_eeg_used": "false",
                    "gt_report_used": "false",
                    "error_message": "",
                }
            )
            continue

        try:
            if source_row.get("status") not in {"ok", "skipped_existing"}:
                raise RuntimeError(f"Source row is not successful: {source_row.get('status')}")
            board = EvidenceBoard.model_validate_json((source_artifact_dir / "evidence_board.json").read_text(encoding="utf-8"))
            target_sections = _target_sections(source_artifact_dir)
            claim_plan = _atomic_claim_plan(source_artifact_dir)
            result = synth.synthesize_celm_sections(
                board,
                target_sections,
                clinical_context=_clinical_context(source_artifact_dir),
                claim_plan_override=claim_plan,
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
            _write_json(output_artifact_dir / "d_section_texts.json", result.section_texts)
            _write_json(output_artifact_dir / "d_synthesis_trace.json", result.trace)
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
        except Exception as exc:  # pragma: no cover - batch guard
            output_artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output_artifact_dir / "d_error.json", {"error": repr(exc), "report_id": report_id})
            status = "error"
            error_message = repr(exc)
            raw_eeg_used = "false"
            gt_report_used = "false"

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
                "raw_eeg_used": raw_eeg_used,
                "gt_report_used": gt_report_used,
                "error_message": error_message,
            }
        )
        _write_csv(output_root / "batch_summary.csv", summary_rows, SUMMARY_COLUMNS)
        print(
            f"[D] {len(summary_rows)}/{len(rows)} row_index={row_index} report_id={report_id} status={status}",
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
        "method_variant": "D_evidence_board_llm_synthesis",
        "raw_eeg_used": False,
        "gt_report_used": False,
    }
    _write_csv(output_root / "batch_summary.csv", summary_rows, SUMMARY_COLUMNS)
    _write_json(output_root / "batch_final_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run method D: EvidenceBoard-only LLM synthesis over an existing batch.")
    parser.add_argument("--source-batch-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _load_env_file(Path(args.env_file) if args.env_file else None)
    summary = run_d_synthesis_batch(
        source_batch_root=Path(args.source_batch_root),
        output_root=Path(args.output_root),
        model=args.model,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
