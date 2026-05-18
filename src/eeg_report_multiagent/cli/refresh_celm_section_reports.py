from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from eeg_report_multiagent.evaluation.section_contract_audit import audit_section_contract
from eeg_report_multiagent.io import make_celm_generated_report
from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.section_contract import TargetSectionContract


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_jsonable(payload):
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, list):
        return [_to_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _to_jsonable(value) for key, value in payload.items()}
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _target_sections(artifact_dir: Path) -> List[str]:
    context_path = artifact_dir / "study_context.json"
    context = _read_json(context_path)
    sections = context.get("target_section_names") or []
    if not isinstance(sections, list):
        raise ValueError(f"target_section_names is not a list in {context_path}")
    return [str(section) for section in sections]


def refresh_batch(batch_root: Path, celm_results_dir: Path | None = None) -> Dict[str, Any]:
    summary_path = batch_root / "batch_summary.csv"
    rows = _read_csv_dicts(summary_path)
    results_dir = celm_results_dir or batch_root / "celm_results"
    json_dir = results_dir / "generated_reports_json"
    txt_dir = results_dir / "generated_reports_txt"
    synth = ReportSynthesizer()
    router = SectionRouter()

    refreshed = 0
    skipped = 0
    errors: List[Dict[str, str]] = []
    for row in rows:
        if row.get("status") not in {"ok", "skipped_existing"}:
            skipped += 1
            continue
        report_id = row.get("report_id", "")
        artifact_dir = Path(row.get("artifact_dir", ""))
        if not artifact_dir.exists() and str(artifact_dir).startswith("/workspace/"):
            local_workspace = Path.cwd().parent if Path.cwd().name == "eeg_report_multiagent_v1" else Path.cwd()
            artifact_dir = Path(str(artifact_dir).replace("/workspace", str(local_workspace), 1))
        try:
            evidence_board_path = artifact_dir / "evidence_board.json"
            board = EvidenceBoard.model_validate_json(evidence_board_path.read_text(encoding="utf-8"))
            target_sections = _target_sections(artifact_dir)
            contract_path = artifact_dir / "target_section_contract.json"
            if contract_path.exists():
                contract = TargetSectionContract.model_validate_json(contract_path.read_text(encoding="utf-8"))
            else:
                contract = router.build_contract(
                    report_id=report_id,
                    target_section_names_raw=target_sections,
                    target_section_names_standardized=target_sections,
                    eval_only_reference_json_path=None,
                )
                _write_json(contract_path, contract)
            detail_text = (artifact_dir / "detail.txt").read_text(encoding="utf-8")
            impression_text = (artifact_dir / "impression.txt").read_text(encoding="utf-8")
            claim_plan = synth.build_atomic_claim_plan(board)
            surface_decisions = synth.build_surface_decisions(claim_plan, board.ensure_shared_evidence_board())
            section_texts = synth.synthesize_celm_sections(board, target_sections)
            generated_report = make_celm_generated_report(
                target_section_names=target_sections,
                detail_text=detail_text,
                impression_text=impression_text,
                section_texts=section_texts,
            )
            _write_json(artifact_dir / "atomic_claim_plan.json", claim_plan)
            _write_json(artifact_dir / "surface_decisions.json", surface_decisions)
            _write_json(artifact_dir / "celm_section_texts.json", section_texts)
            _write_json(artifact_dir / "celm_generated_report.json", generated_report)
            _write_json(artifact_dir / "section_contract_audit.json", audit_section_contract(contract, board, generated_report))
            _write_json(json_dir / f"GENERATED_REPORT_{report_id}.json", generated_report)
            txt_dir.mkdir(parents=True, exist_ok=True)
            (txt_dir / f"GENERATED_REPORT_{report_id}.txt").write_text(
                json.dumps(generated_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            refreshed += 1
        except Exception as exc:  # pragma: no cover - defensive batch guard
            errors.append({"report_id": report_id, "error": repr(exc)})

    summary = {
        "batch_root": str(batch_root),
        "celm_results_dir": str(results_dir),
        "rows": len(rows),
        "refreshed": refreshed,
        "skipped": skipped,
        "errors": errors,
    }
    _write_json(batch_root / "refresh_celm_section_reports_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh CELM-compatible generated reports from saved EvidenceBoard artifacts.")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--celm-results-dir", default=None)
    args = parser.parse_args()

    summary = refresh_batch(
        batch_root=Path(args.batch_root),
        celm_results_dir=Path(args.celm_results_dir) if args.celm_results_dir else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
