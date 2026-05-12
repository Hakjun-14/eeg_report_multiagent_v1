from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from eeg_report_multiagent.evaluation.method_audit import audit_artifact_dir, render_audit_markdown
from eeg_report_multiagent.evaluation.section_contract_audit import audit_section_contract
from eeg_report_multiagent.io import load_celm_split_sample, make_celm_generated_report
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.schemas.evidence import EvidenceBoard


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one CELM-compatible split row through eeg_report_multiagent_v1"
    )
    parser.add_argument("--data-root", required=True, help="CELM data root, e.g. /workspace/eeg_data/celm_s_sites_pipeline")
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True, help="Per-run artifact directory")
    parser.add_argument(
        "--celm-results-dir",
        default=None,
        help="Optional CELM-compatible results directory. Writes generated_reports_json/txt under this path.",
    )
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--no-langgraph", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--enable-llm-review", action="store_true")
    parser.add_argument("--enable-llm-finding-proposals", action="store_true")
    parser.add_argument("--enable-local-encoder", action="store_true")
    args = parser.parse_args()

    sample = load_celm_split_sample(
        data_root=Path(args.data_root),
        site=args.site,
        split=args.split,
        row_index=args.row_index,
        split_type=args.split_type,
    )
    if args.session_index < 0 or args.session_index >= len(sample.session_dirs):
        raise IndexError(f"session-index out of range: {args.session_index} for {len(sample.session_dirs)} sessions")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study_context_json = out_dir / "study_context.json"
    _write_json(study_context_json, sample.study_context)
    _write_json(out_dir / "target_section_contract.json", sample.target_section_contract)

    session_dir = sample.session_dirs[args.session_index]
    cmd = [
        sys.executable,
        "-m",
        "eeg_report_multiagent.cli.run_session",
        "--session-dir",
        str(session_dir),
        "--study-context-json",
        str(study_context_json),
        "--gt-report-json",
        str(sample.report_json_path),
        "--output-dir",
        str(out_dir),
    ]
    if args.monitor:
        cmd.append("--monitor")
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

    print("Running CELM-compatible split session")
    print(f"  data_root: {sample.data_root}")
    print(f"  site/split: {sample.site}/{sample.split_type}/{sample.split}")
    print(f"  row_index: {sample.row_index}")
    print(f"  report_id: {sample.report_id}")
    print(f"  patient_id: {sample.row.get('BDSPPatientID', '')}")
    print(f"  session_index: {args.session_index}")
    print(f"  session_dir: {session_dir}")
    print(f"  gt_report_json_eval_only: {sample.report_json_path}")
    print(f"  target_sections: {sample.target_section_names_standardized}")
    subprocess.run(cmd, check=True)

    detail_text = (out_dir / "detail.txt").read_text(encoding="utf-8") if (out_dir / "detail.txt").exists() else ""
    impression_text = (out_dir / "impression.txt").read_text(encoding="utf-8") if (out_dir / "impression.txt").exists() else ""
    section_texts = None
    evidence_board_path = out_dir / "evidence_board.json"
    if evidence_board_path.exists():
        board = EvidenceBoard.model_validate_json(evidence_board_path.read_text(encoding="utf-8"))
        section_texts = ReportSynthesizer().synthesize_celm_sections(
            board=board,
            target_section_names=sample.target_section_names_standardized,
        )
        _write_json(out_dir / "celm_section_texts.json", section_texts)
    generated_report = make_celm_generated_report(
        target_section_names=sample.target_section_names_standardized,
        detail_text=detail_text,
        impression_text=impression_text,
        section_texts=section_texts,
    )
    _write_json(out_dir / "celm_generated_report.json", generated_report)
    if evidence_board_path.exists():
        board = EvidenceBoard.model_validate_json(evidence_board_path.read_text(encoding="utf-8"))
        section_audit = audit_section_contract(sample.target_section_contract, board, generated_report)
        _write_json(out_dir / "section_contract_audit.json", section_audit)
    audit = audit_artifact_dir(out_dir)
    _write_json(out_dir / "method_audit.json", audit)
    (out_dir / "method_audit.md").write_text(render_audit_markdown(audit), encoding="utf-8")

    if args.celm_results_dir:
        results_dir = Path(args.celm_results_dir)
        json_dir = results_dir / "generated_reports_json"
        txt_dir = results_dir / "generated_reports_txt"
        json_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"GENERATED_REPORT_{sample.report_id}.json"
        txt_path = txt_dir / f"GENERATED_REPORT_{sample.report_id}.txt"
        _write_json(json_path, generated_report)
        txt_path.write_text(json.dumps(generated_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved CELM-compatible JSON: {json_path}")
        print(f"Saved CELM-compatible TXT: {txt_path}")

    print(f"Saved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
