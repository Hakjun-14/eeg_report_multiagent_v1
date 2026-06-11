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
from eeg_report_multiagent.modules.final_prose_auditor import FinalProseAuditor
from eeg_report_multiagent.modules.llm_report_synthesizer import EvidenceBoardLLMReportSynthesizer
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.report import AtomicClaimPlan


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


def _sanitize_patient_history_text(text: str) -> str:
    """Keep CELM-style clinical context while stripping target report text."""
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


def _clinical_context_from_study_context(study_context: dict) -> dict:
    metadata = study_context.get("metadata") if isinstance(study_context.get("metadata"), dict) else {}
    safe_metadata = {
        str(key): "" if value is None else str(value)
        for key, value in metadata.items()
        if str(key) in _SAFE_CLINICAL_METADATA_KEYS and str(key) not in _EVAL_ONLY_METADATA_KEYS
    }
    clinical_text = _sanitize_patient_history_text(
        str(
            study_context.get("clinical_history")
            or study_context.get("patient_history")
            or metadata.get("clinical_history")
            or ""
        )
    )
    target_sections = study_context.get("target_section_names") or metadata.get("target_section_names") or []
    return {
        "patient_history_and_eeg_description": str(clinical_text),
        "target_section_names": target_sections if isinstance(target_sections, list) else str(target_sections),
        "context_type": str(study_context.get("context_type", "")),
        "metadata": safe_metadata,
        "gt_report_text_included": False,
    }


def _read_atomic_claim_plan(path: Path) -> list[AtomicClaimPlan]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [AtomicClaimPlan.model_validate(item) for item in payload]


def _fallback_section_texts_from_claim_plan(
    synth: ReportSynthesizer,
    board: EvidenceBoard,
    claim_plan: list[AtomicClaimPlan],
    target_section_names: list[str],
) -> dict[str, str]:
    """Compatibility fallback that still preserves the existing LLM claim plan."""
    router = SectionRouter()
    surface_decisions = synth.build_surface_decisions(claim_plan, board.ensure_shared_evidence_board())
    return {
        section_name: synth._section_text_from_plans(  # noqa: SLF001
            claim_plan,
            router.role_for_section(section_name),
            surface_decisions,
        )
        for section_name in target_section_names
    }


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
    parser.add_argument("--enable-llm-evidence-grouping", action="store_true")
    parser.add_argument("--enable-llm-claim-planning", action="store_true")
    parser.add_argument("--enable-llm-review", action="store_true")
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
    if args.enable_llm_evidence_grouping:
        cmd.append("--enable-llm-evidence-grouping")
    if args.enable_llm_claim_planning:
        cmd.append("--enable-llm-claim-planning")
    if args.enable_llm_review:
        cmd.append("--enable-llm-review")
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
    claim_plan: list[AtomicClaimPlan] = []
    if evidence_board_path.exists():
        board = EvidenceBoard.model_validate_json(evidence_board_path.read_text(encoding="utf-8"))
        synth = ReportSynthesizer()
        claim_plan = _read_atomic_claim_plan(out_dir / "atomic_claim_plan.json") or synth.build_atomic_claim_plan(board)
        try:
            result = EvidenceBoardLLMReportSynthesizer(report_synthesizer=synth).synthesize_celm_sections(
                board=board,
                target_section_names=sample.target_section_names_standardized,
                clinical_context=_clinical_context_from_study_context(sample.study_context),
                claim_plan_override=claim_plan,
            )
            section_texts = result.section_texts
            _write_json(out_dir / "llm_report_synthesis.json", result.trace)
        except Exception as exc:
            # Keep the batch runnable if the report LLM/API is unavailable, but
            # do not fall back to rebuilding grouped rule claims.
            _write_json(
                out_dir / "llm_report_synthesis_error.json",
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "fallback": "existing_atomic_claim_plan_template_rendering",
                },
            )
            section_texts = _fallback_section_texts_from_claim_plan(
                synth,
                board,
                claim_plan,
                sample.target_section_names_standardized,
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
        synth = ReportSynthesizer()
        claim_plan = claim_plan or _read_atomic_claim_plan(out_dir / "atomic_claim_plan.json") or synth.build_atomic_claim_plan(board)
        surface_decisions = synth.build_surface_decisions(claim_plan, board.ensure_shared_evidence_board())
        _write_json(out_dir / "surface_decisions.json", surface_decisions)
        final_prose_audit = FinalProseAuditor().audit_report(
            generated_report,
            board.ensure_shared_evidence_board(),
            claim_plan,
        )
        _write_json(out_dir / "final_prose_audit.json", final_prose_audit)
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
