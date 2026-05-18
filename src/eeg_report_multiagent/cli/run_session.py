from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from eeg_report_multiagent.graph import run_pipeline
from eeg_report_multiagent.modules.report_synthesizer import ReportSynthesizer
from eeg_report_multiagent.modules.final_prose_auditor import FinalProseAuditor


CODE_PATH_TRACE = [
    {
        "block": "orchestration",
        "file": "src/eeg_report_multiagent/graph/builder.py",
        "functions": ["run_pipeline", "build_graph"],
    },
    {
        "block": "graph_nodes",
        "file": "src/eeg_report_multiagent/graph/nodes.py",
        "functions": [
            "load_inputs_node",
            "scout_pass_node",
            "background_module_node",
            "event_module_node",
            "protocol_parser_node",
            "evidence_merge_node",
            "evidence_review_node",
            "finding_proposal_node",
            "report_synthesize_node",
            "optional_verify_node",
            "finalize_node",
        ],
    },
    {
        "block": "background",
        "file": "src/eeg_report_multiagent/modules/background_module.py",
        "functions": ["BackgroundModule.run"],
    },
    {
        "block": "event",
        "file": "src/eeg_report_multiagent/modules/event_module.py",
        "functions": ["EventModule.run"],
    },
    {
        "block": "event_tools",
        "file": "src/eeg_report_multiagent/tools/event/signal_tools.py",
        "functions": [
            "transient_candidate_score",
            "burst_train_duration_estimate",
            "channel_spread_laterality_summary",
            "event_peak_topography_localizer",
            "focality_bifrontal_summary",
            "morphology_feature_encoder",
            "event_type_separation_classifier",
        ],
    },
    {
        "block": "parser",
        "file": "src/eeg_report_multiagent/modules/protocol_state_context_parser.py",
        "functions": ["ProtocolStateContextParser.run"],
    },
    {
        "block": "evidence_board",
        "file": "src/eeg_report_multiagent/modules/evidence_board.py",
        "functions": ["EvidenceBoardAssembler.merge"],
    },
    {
        "block": "llm_evidence_review",
        "file": "src/eeg_report_multiagent/modules/evidence_reviewer.py",
        "functions": ["EvidenceReviewModule.run"],
    },
    {
        "block": "llm_finding_proposal",
        "file": "src/eeg_report_multiagent/modules/llm_finding_proposer.py",
        "functions": ["LLMFindingProposalModule.run"],
    },
    {
        "block": "llm_adapter",
        "file": "src/eeg_report_multiagent/llm/openai_adapter.py",
        "functions": ["OpenAIEvidenceReviewAdapter.review"],
    },
    {
        "block": "synthesis",
        "file": "src/eeg_report_multiagent/modules/report_synthesizer.py",
        "functions": [
            "ReportSynthesizer.build_atomic_claim_plan",
            "ReportSynthesizer.synthesize",
            "ReportSynthesizer.synthesize_celm_sections",
        ],
    },
    {
        "block": "shared_evidence",
        "file": "src/eeg_report_multiagent/schemas/shared_evidence.py",
        "functions": ["SharedEvidenceBoard.add_evidence", "SharedEvidenceBoard.snapshot"],
    },
    {
        "block": "final_prose_audit",
        "file": "src/eeg_report_multiagent/modules/final_prose_auditor.py",
        "functions": [
            "FinalProseAuditor.audit_report",
            "FinalProseAuditor.extract_numeric_mentions",
            "FinalProseAuditor.detect_banned_debug_terms",
            "FinalProseAuditor.detect_section_leakage",
            "FinalProseAuditor.match_numeric_to_evidence",
            "FinalProseAuditor.match_text_claims_to_atomic_plans",
        ],
    },
    {
        "block": "section_contract",
        "file": "src/eeg_report_multiagent/modules/section_router.py",
        "functions": ["SectionRouter.build_contract", "SectionRouter.role_for_section"],
    },
]


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, BaseModel):
        return x.model_dump(mode="json")
    if isinstance(x, list):
        return [_to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_dict(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _context_metadata(payload: Dict[str, Any]) -> Dict[str, str]:
    metadata = payload.get("metadata", payload)
    if not isinstance(metadata, dict):
        return {}
    return {str(k): "" if v is None else str(v) for k, v in metadata.items()}


class TerminalMonitor:
    STAGES = [
        "load_inputs",
        "scout_pass",
        "background_module",
        "event_module",
        "protocol_parser",
        "evidence_merge",
        "evidence_review",
        "finding_proposal",
        "report_synthesize",
        "optional_verify",
        "finalize",
    ]

    STAGE_LABELS = {
        "load_inputs": "Load Inputs",
        "scout_pass": "Scout Pass",
        "background_module": "Background Module",
        "event_module": "Event Module",
        "protocol_parser": "Protocol Parser",
        "evidence_merge": "Evidence Merge",
        "evidence_review": "LLM Evidence Review",
        "finding_proposal": "LLM Finding Proposal",
        "report_synthesize": "Report Synthesize",
        "optional_verify": "Claim Verify",
        "finalize": "Finalize",
        "graph_invoke": "LangGraph Invoke",
    }

    ICONS = {
        "pending": "[ ]",
        "running": "[~]",
        "done": "[x]",
        "error": "[!]",
    }

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.stage_status = {s: "pending" for s in self.STAGES}
        self.current_stage = ""
        self.current_phase = ""
        self.spinner = "|"
        self.elapsed_sec = 0.0

    def update(self, node_name: str, phase: str, state: Dict[str, Any], meta: Dict[str, Any] | None) -> None:
        if not self.enabled:
            return
        if meta is None:
            meta = {}

        self.current_stage = node_name
        self.current_phase = phase
        self.spinner = str(meta.get("spinner", self.spinner))
        self.elapsed_sec = float(meta.get("elapsed_sec", self.elapsed_sec))

        if node_name in self.stage_status:
            if phase in {"start", "running"}:
                self.stage_status[node_name] = "running"
            elif phase == "end":
                self.stage_status[node_name] = "done"
            elif phase == "error":
                self.stage_status[node_name] = "error"

        self._render(state, error_text=str(meta.get("error", "")) if phase == "error" else "")

    def _render(self, state: Dict[str, Any], error_text: str = "") -> None:
        # Clear screen and redraw monitor panel in-place.
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("EEG Multi-Agent Monitor (v1)\n")

        stage_label = self.STAGE_LABELS.get(self.current_stage, self.current_stage or "-")
        if self.current_phase == "running":
            current_line = f"Current: {stage_label} {self.spinner} ({self.elapsed_sec:.1f}s)"
        else:
            current_line = f"Current: {stage_label} [{self.current_phase or '-'}] ({self.elapsed_sec:.1f}s)"
        sys.stdout.write(current_line + "\n")
        if error_text:
            sys.stdout.write(f"Error: {error_text}\n")
        sys.stdout.write("\n")

        sys.stdout.write("Stages\n")
        for s in self.STAGES:
            icon = self.ICONS[self.stage_status[s]]
            sys.stdout.write(f"  {icon} {self.STAGE_LABELS[s]}\n")
        sys.stdout.write("\n")

        manifest = state.get("manifest")
        if manifest is not None:
            try:
                session_id = getattr(manifest, "session_id", "-")
                shape = getattr(manifest, "shape_nct", None)
                sys.stdout.write(f"Session: {session_id}  shape={shape}\n")
            except Exception:
                pass

        scout = state.get("scout_summary", {})
        if scout:
            sys.stdout.write(
                "Scout: "
                f"amp={scout.get('global_amp_hint', 0.0):.3f}, "
                f"slow={scout.get('global_slowing_hint', 0.0):.3f}, "
                f"event={scout.get('event_density_hint', 0.0):.3f}\n"
            )

        bg_f = state.get("background_findings", [])
        ev_f = state.get("event_findings", [])
        ps_f = state.get("parser_findings", [])
        claims = []
        board = state.get("evidence_board")
        if board is not None and hasattr(board, "claims"):
            claims = list(board.claims)
        ver = state.get("verification", [])
        deliberations = state.get("agent_deliberations", [])
        focused = state.get("focused_windows", [])

        sys.stdout.write(
            "Counts: "
            f"bg_m={len(state.get('background_measurements', []))}, "
            f"bg_f={len(bg_f)}, "
            f"ev_m={len(state.get('event_measurements', []))}, "
            f"ev_f={len(ev_f)}, "
            f"parser_m={len(state.get('parser_measurements', []))}, "
            f"parser_f={len(ps_f)}, "
            f"review={len(deliberations)}, "
            f"claims={len(claims)}, "
            f"verify={len(ver)}\n"
        )
        sys.stdout.write(f"Focused windows: {len(focused)}\n")

        def _preview(findings: list[Any], title: str) -> None:
            if not findings:
                return
            preview = []
            for f in findings[:3]:
                try:
                    preview.append(f"{f.finding_type}:{f.assertion.value}")
                except Exception:
                    try:
                        preview.append(f"{f['finding_type']}:{f['assertion']}")
                    except Exception:
                        continue
            if preview:
                sys.stdout.write(f"{title}: " + ", ".join(preview) + "\n")

        _preview(bg_f, "BG preview")
        _preview(ev_f, "Event preview")
        _preview(ps_f, "Parser preview")

        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one session end-to-end and save artifacts")
    parser.add_argument("--session-dir", required=True, help="Path to processed_eeg/<session_folder>")
    parser.add_argument("--study-context-json", default=None, help="Optional inference context json path (not GT report)")
    parser.add_argument("--study-context-text", default=None, help="Optional inference context text path (not GT report)")
    parser.add_argument("--gt-report-json", default=None, help="Optional GT report json path for evaluation bookkeeping only")
    parser.add_argument("--report-json", default=None, help="Deprecated alias for --study-context-json")
    parser.add_argument("--report-text", default=None, help="Deprecated alias for --study-context-text")
    parser.add_argument("--metadata-json", default=None, help="Optional metadata json path (dict)")
    parser.add_argument("--output-dir", default=None, help="Artifact output directory")
    parser.add_argument("--no-langgraph", action="store_true", help="Force sequential fallback runner")
    parser.add_argument("--no-verify", action="store_true", help="Disable optional claim verification")
    parser.add_argument("--monitor", action="store_true", help="Show live stage monitor in terminal")
    parser.add_argument("--enable-llm-review", action="store_true", help="Run optional evidence-board-only LLM review")
    parser.add_argument("--enable-llm-finding-proposals", action="store_true", help="Run optional measurement-to-finding LLM proposal ablation")
    parser.add_argument("--enable-local-encoder", action="store_true", help="Run bounded local EEG encoder proxy tool inside event module")
    args = parser.parse_args()

    study_context_json_path = args.study_context_json or args.report_json
    study_context_text_path = args.study_context_text or args.report_text

    metadata: Dict[str, str] = {}
    study_context = _load_json_dict(study_context_json_path)
    metadata.update(_context_metadata(study_context))
    if args.metadata_json:
        metadata.update(_load_json_dict(args.metadata_json))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("artifacts") / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "session_dir": args.session_dir,
        "study_context_json_path": study_context_json_path,
        "study_context_text_path": study_context_text_path,
        "gt_report_json_path": args.gt_report_json,
        "report_json_path": study_context_json_path,
        "report_text_path": study_context_text_path,
        "metadata": metadata,
        "verify_claims": not args.no_verify,
        "enable_llm_review": args.enable_llm_review,
        "enable_llm_finding_proposals": args.enable_llm_finding_proposals,
        "enable_local_encoder": args.enable_local_encoder,
        "run_log": [],
    }

    monitor = TerminalMonitor(enabled=args.monitor)

    use_langgraph = not args.no_langgraph
    final_state = run_pipeline(state, use_langgraph=use_langgraph, node_callback=monitor.update if args.monitor else None)

    _write_json(out_dir / "manifest.json", final_state["manifest"])
    _write_json(out_dir / "scout_summary.json", final_state.get("scout_summary", {}))
    _write_json(out_dir / "background_findings.json", final_state.get("background_findings", []))
    _write_json(out_dir / "event_findings.json", final_state.get("event_findings", []))
    _write_json(out_dir / "parsed_context.json", final_state.get("parser_findings", []))
    _write_json(out_dir / "evidence_board.json", final_state.get("evidence_board"))
    board = final_state.get("evidence_board")
    shared_evidence_snapshot = board.ensure_shared_evidence_board().snapshot() if board is not None else None
    _write_json(out_dir / "shared_evidence_board.json", shared_evidence_snapshot)

    detail_text = final_state.get("detail_section").text if final_state.get("detail_section") else ""
    impression_text = final_state.get("impression_section").text if final_state.get("impression_section") else ""
    (out_dir / "detail.txt").write_text(detail_text, encoding="utf-8")
    (out_dir / "impression.txt").write_text(impression_text, encoding="utf-8")

    _write_json(out_dir / "verification.json", final_state.get("verification", []))

    claims = board.claims if board is not None and hasattr(board, "claims") else []
    report_synthesizer = ReportSynthesizer()
    atomic_claim_plan = report_synthesizer.build_atomic_claim_plan(board) if board is not None else []
    surface_decisions = (
        report_synthesizer.build_surface_decisions(atomic_claim_plan, board.ensure_shared_evidence_board())
        if board is not None
        else []
    )
    _write_json(out_dir / "atomic_claim_plan.json", atomic_claim_plan)
    _write_json(out_dir / "surface_decisions.json", surface_decisions)
    final_prose_audit = None
    if board is not None:
        final_prose_audit = FinalProseAuditor().audit_report(
            {
                "EEG DESCRIPTION/DETAILS": detail_text,
                "IMPRESSION/INTERPRETATION": impression_text,
            },
            board.ensure_shared_evidence_board(),
            atomic_claim_plan,
        )
    _write_json(out_dir / "final_prose_audit.json", final_prose_audit)
    inference_trace = {
        "inputs": {
            "session_dir": args.session_dir,
            "study_context_json_path": study_context_json_path,
            "study_context_text_path": study_context_text_path,
            "gt_report_json_path_eval_only": args.gt_report_json,
            "legacy_report_json_alias_used": bool(args.report_json and not args.study_context_json),
            "legacy_report_text_alias_used": bool(args.report_text and not args.study_context_text),
            "metadata_json_path": args.metadata_json,
            "use_langgraph": use_langgraph,
            "verify_claims": not args.no_verify,
            "monitor_enabled": args.monitor,
            "llm_review_enabled": args.enable_llm_review,
            "llm_finding_proposals_enabled": args.enable_llm_finding_proposals,
            "local_encoder_enabled": args.enable_local_encoder,
            "input_contract": "GT report is evaluation/supervision only and is not passed to parser or signal modules.",
        },
        "scout_summary": final_state.get("scout_summary", {}),
        "background_module": {
            "measurements": final_state.get("background_measurements", []),
            "findings": final_state.get("background_findings", []),
            "tool_invocations": final_state.get("background_tool_invocations", []),
        },
        "event_module": {
            "measurements": final_state.get("event_measurements", []),
            "findings": final_state.get("event_findings", []),
            "tool_invocations": final_state.get("event_tool_invocations", []),
            "focused_windows": final_state.get("focused_windows", []),
        },
        "protocol_parser": {
            "measurements": final_state.get("parser_measurements", []),
            "findings": final_state.get("parser_findings", []),
            "tool_invocations": final_state.get("parser_tool_invocations", []),
        },
        "evidence_board": board,
        "shared_evidence_board": shared_evidence_snapshot,
        "agent_deliberations": final_state.get("agent_deliberations", []),
        "llm_finding_proposals": final_state.get("llm_finding_proposals", {}),
        "report_synthesis": {
            "detail_section": final_state.get("detail_section"),
            "impression_section": final_state.get("impression_section"),
            "atomic_claim_plan": atomic_claim_plan,
            "surface_decisions": surface_decisions,
            "claims": claims,
        },
        "final_prose_audit": final_prose_audit,
        "verification": final_state.get("verification", []),
        "run_log": final_state.get("run_log", []),
        "stats": {
            "background_measurements": len(final_state.get("background_measurements", [])),
            "background_findings": len(final_state.get("background_findings", [])),
            "event_measurements": len(final_state.get("event_measurements", [])),
            "event_findings": len(final_state.get("event_findings", [])),
            "parser_measurements": len(final_state.get("parser_measurements", [])),
            "parser_findings": len(final_state.get("parser_findings", [])),
            "agent_deliberations": len(final_state.get("agent_deliberations", [])),
            "llm_finding_proposals": len((final_state.get("llm_finding_proposals", {}) or {}).get("finding_proposals", [])),
            "verification_records": len(final_state.get("verification", [])),
            "atomic_claim_plans": len(atomic_claim_plan),
            "surface_decisions": len(surface_decisions),
            "audit_pass": final_prose_audit.pass_fail == "pass" if final_prose_audit else None,
            "unsupported_numeric_count": len(final_prose_audit.unsupported_numeric_mentions) if final_prose_audit else 0,
            "debug_leak_count": len(final_prose_audit.debug_leaks) if final_prose_audit else 0,
            "section_leakage_count": len(final_prose_audit.section_leakages) if final_prose_audit else 0,
            "seizure_gate_violation_count": len(final_prose_audit.seizure_gate_violations) if final_prose_audit else 0,
            "claim_trace_coverage": final_prose_audit.metrics.get("ClaimTraceCoverage") if final_prose_audit else None,
        },
    }
    _write_json(out_dir / "inference_trace.json", inference_trace)
    _write_json(out_dir / "agent_deliberations.json", final_state.get("agent_deliberations", []))
    _write_json(out_dir / "llm_finding_proposals.json", final_state.get("llm_finding_proposals", {}))
    (out_dir / "run.log").write_text("\n".join(final_state.get("run_log", [])), encoding="utf-8")

    generated_files = [
        "manifest.json",
        "scout_summary.json",
        "background_findings.json",
        "event_findings.json",
        "parsed_context.json",
        "evidence_board.json",
        "shared_evidence_board.json",
        "detail.txt",
        "impression.txt",
        "verification.json",
        "atomic_claim_plan.json",
        "surface_decisions.json",
        "final_prose_audit.json",
        "inference_trace.json",
        "agent_deliberations.json",
        "llm_finding_proposals.json",
        "run.log",
        "run_artifact_manifest.json",
    ]
    artifact_manifest = {
        "generated_files": [str(out_dir / name) for name in generated_files],
        "code_path_trace": CODE_PATH_TRACE,
        "inference_inputs": inference_trace["inputs"],
        "research_contract": {
            "raw_eeg_external_api": "forbidden",
            "gt_report_as_inference_input": "forbidden",
            "intermediate_representation": "measurement -> finding -> evidence_board -> report_text",
            "primary_artifacts": ["manifest.json", "evidence_board.json", "surface_decisions.json", "inference_trace.json"],
            "human_readable_artifacts": ["detail.txt", "impression.txt"],
        },
    }
    _write_json(out_dir / "run_artifact_manifest.json", artifact_manifest)

    print(f"Saved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
