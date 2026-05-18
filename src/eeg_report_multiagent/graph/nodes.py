from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from eeg_report_multiagent.agents import BackgroundAgent, EventAgent
from eeg_report_multiagent.io import (
    build_session_manifest,
    get_note_text,
    load_report_json,
    load_report_text,
    load_session_from_processed_dir,
)
from eeg_report_multiagent.modules import (
    BackgroundModule,
    ClaimVerifier,
    EvidenceBoardAssembler,
    EvidenceReviewModule,
    EventModule,
    LLMClaimPlanner,
    LLMFindingProposalModule,
    LLMEvidenceGrouper,
    ProtocolStateContextParser,
    ReportSynthesizer,
)
from eeg_report_multiagent.modules.evidence_item_adapter import append_deliberation_evidence
from eeg_report_multiagent.tools import build_background_registry, build_event_registry, build_parser_registry


def _infer_uv_scale(signal_nct: np.ndarray) -> float:
    return 1_000_000.0 if float(np.percentile(np.abs(signal_nct), 95.0)) < 1.0 else 1.0


def _scout_band_ratio(signal_nct: np.ndarray, fs: int) -> float:
    centered = signal_nct - np.mean(signal_nct, axis=-1, keepdims=True)
    freqs = np.fft.rfftfreq(centered.shape[-1], d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(centered, axis=-1)) ** 2) / centered.shape[-1]

    def band_sum(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return float(psd[..., mask].sum(axis=-1).mean())

    slow = band_sum(0.5, 8.0)
    faster = band_sum(8.0, 30.0)
    return slow / (faster + 1e-12)


def _append_log(state: Dict, message: str) -> None:
    state.setdefault("run_log", []).append(message)


def load_inputs_node(state: Dict) -> Dict:
    session_dir = Path(state["session_dir"])
    session = load_session_from_processed_dir(session_dir)
    context_json_path = state.get("study_context_json_path") or state.get("report_json_path")
    context_text_path = state.get("study_context_text_path") or state.get("report_text_path")
    context_json = load_report_json(Path(context_json_path) if context_json_path else None)
    context_text = load_report_text(Path(context_text_path) if context_text_path else None)
    note_text = get_note_text(context_json, fallback_text=context_text)

    manifest = build_session_manifest(
        session=session,
        study_context_json_path=context_json_path,
        study_context_text_path=context_text_path,
        gt_report_json_path=state.get("gt_report_json_path"),
        metadata_row_available=bool(state.get("metadata")),
    )

    state["session"] = session
    state["manifest"] = manifest
    state["note_text"] = note_text
    _append_log(state, f"Loaded session: {session.session_id} with shape {session.signals.shape}")
    return state


def scout_pass_node(state: Dict) -> Dict:
    signal_nct = state["session"].signals
    fs = state["manifest"].sample_rate_hz

    # coarse scout metrics
    abs_sig = np.abs(signal_nct * _infer_uv_scale(signal_nct))
    global_amp = float(np.percentile(abs_sig, 95))

    # Scout slowing is a coarse spectral ratio, not a final clinical claim.
    global_slowing_hint = _scout_band_ratio(signal_nct, fs=fs)

    # transient hint
    deriv = np.diff(signal_nct, axis=-1)
    per_window = np.percentile(np.abs(deriv), 99, axis=(1, 2))
    event_density_hint = float(np.mean(per_window > np.percentile(per_window, 90)))

    state["scout_summary"] = {
        "global_amp_hint": global_amp,
        "global_slowing_hint": global_slowing_hint,
        "event_density_hint": event_density_hint,
        "enable_local_encoder": bool(state.get("enable_local_encoder", False)),
        "amp_unit": "uV",
        "slowing_hint_method": "detrended_spectral_ratio_0p5_8_over_8_30",
    }
    _append_log(state, "Completed scout pass")
    return state


def background_module_node(state: Dict) -> Dict:
    session = state["session"]
    module = BackgroundModule(registry=build_background_registry(), agent=BackgroundAgent())
    result = module.run(
        signal_nct=session.signals,
        fs=state["manifest"].sample_rate_hz,
        source_ref=state["manifest"].session_id,
        scout_summary=state["scout_summary"],
        channels=session.channels,
    )
    state["background_measurements"] = result["measurements"]
    state["background_findings"] = result["findings"]
    state["background_tool_invocations"] = result["tool_invocations"]
    _append_log(state, f"Background module done with {len(result['findings'])} findings")
    return state


def event_module_node(state: Dict) -> Dict:
    session = state["session"]
    module = EventModule(registry=build_event_registry(), agent=EventAgent())
    result = module.run(
        signal_nct=session.signals,
        channels=session.channels,
        source_ref=state["manifest"].session_id,
        window_seconds=state["manifest"].window_seconds,
        scout_summary=state["scout_summary"],
    )
    state["event_measurements"] = result["measurements"]
    state["event_findings"] = result["findings"]
    state["event_tool_invocations"] = result["tool_invocations"]
    state["focused_windows"] = result["focused_windows"]
    _append_log(state, f"Event module done with {len(result['findings'])} findings")
    return state


def protocol_parser_node(state: Dict) -> Dict:
    parser = ProtocolStateContextParser(registry=build_parser_registry())
    result = parser.run(
        note_text=state.get("note_text", ""),
        metadata=state.get("metadata", {}),
        source_ref=state["manifest"].session_id,
    )
    state["parser_measurements"] = result["measurements"]
    state["parser_findings"] = result["findings"]
    state["parser_tool_invocations"] = result["tool_invocations"]
    _append_log(state, f"Protocol parser done with {len(result['findings'])} findings")
    return state


def evidence_merge_node(state: Dict) -> Dict:
    assembler = EvidenceBoardAssembler()
    board = assembler.merge(
        session_id=state["manifest"].session_id,
        measurement_groups=[
            state.get("background_measurements", []),
            state.get("event_measurements", []),
            state.get("parser_measurements", []),
        ],
        finding_groups=[
            state.get("background_findings", []),
            state.get("event_findings", []),
            state.get("parser_findings", []),
        ],
        tool_invocation_groups=[
            state.get("background_tool_invocations", []),
            state.get("event_tool_invocations", []),
            state.get("parser_tool_invocations", []),
        ],
    )
    if state.get("enable_llm_evidence_grouping", False):
        grouper = LLMEvidenceGrouper()
        grouping = grouper.run(recording_id=state["manifest"].session_id, measurements=board.measurements)
        board.shared_evidence_board = grouping["shared_evidence_board"]
        state["llm_evidence_grouping"] = {
            key: value for key, value in grouping.items() if key != "shared_evidence_board"
        }
        _append_log(
            state,
            "LLM evidence grouping completed "
            f"groups={len(board.shared_evidence_board.evidence_items)} raw_eeg_used={grouping.get('raw_eeg_used')} "
            f"gt_report_used={grouping.get('gt_report_used')}",
        )
    else:
        state["llm_evidence_grouping"] = {
            "status": "skipped",
            "summary": "",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "raw_result": {"evidence_groups": []},
        }
    state["evidence_board"] = board
    shared_count = len(board.ensure_shared_evidence_board().evidence_items)
    _append_log(state, f"Evidence board assembled with {len(board.findings)} findings and {shared_count} evidence items")
    return state


def evidence_review_node(state: Dict) -> Dict:
    if not state.get("enable_llm_review", False):
        state["agent_deliberations"] = []
        _append_log(state, "Skipped LLM evidence review")
        return state

    reviewer = EvidenceReviewModule()
    deliberation = reviewer.run(state["evidence_board"])
    state["agent_deliberations"] = [deliberation]
    state["evidence_board"].deliberations = [deliberation]
    append_deliberation_evidence(state["evidence_board"].ensure_shared_evidence_board(), deliberation)
    _append_log(state, f"LLM evidence review completed with status={deliberation.status}")
    return state


def finding_proposal_node(state: Dict) -> Dict:
    if not state.get("enable_llm_finding_proposals", False):
        state["llm_finding_proposals"] = {
            "status": "skipped",
            "finding_proposals": [],
            "raw_eeg_used": False,
            "gt_report_used": False,
        }
        _append_log(state, "Skipped LLM measurement-to-finding proposal ablation")
        return state

    proposer = LLMFindingProposalModule()
    result = proposer.run(state["evidence_board"])
    state["llm_finding_proposals"] = result
    _append_log(
        state,
        f"LLM finding proposal completed with status={result.get('status')} proposals={len(result.get('finding_proposals', []))}",
    )
    return state


def report_synthesize_node(state: Dict) -> Dict:
    synth = ReportSynthesizer()
    claim_plan_override = None
    if state.get("enable_llm_claim_planning", False):
        planner = LLMClaimPlanner()
        planning = planner.run(state["evidence_board"].ensure_shared_evidence_board())
        claim_plan_override = planning["atomic_claim_plan"]
        state["llm_claim_planning"] = {
            key: value for key, value in planning.items() if key != "atomic_claim_plan"
        }
        _append_log(
            state,
            "LLM claim planning completed "
            f"claims={len(claim_plan_override)} raw_eeg_used={planning.get('raw_eeg_used')} "
            f"gt_report_used={planning.get('gt_report_used')}",
        )
    else:
        state["llm_claim_planning"] = {
            "status": "skipped",
            "summary": "",
            "raw_eeg_used": False,
            "gt_report_used": False,
            "raw_result": {"atomic_claims": []},
        }

    detail, impression, claims = synth.synthesize(state["evidence_board"], claim_plan_override=claim_plan_override)
    atomic_claim_plan = claim_plan_override if claim_plan_override is not None else synth.build_atomic_claim_plan(state["evidence_board"])
    surface_decisions = synth.build_surface_decisions(
        atomic_claim_plan,
        state["evidence_board"].ensure_shared_evidence_board(),
    )
    state["detail_section"] = detail
    state["impression_section"] = impression
    state["evidence_board"].claims = claims
    state["atomic_claim_plan"] = atomic_claim_plan
    state["surface_decisions"] = surface_decisions
    _append_log(state, f"Synthesized report with {len(claims)} claims")
    return state


def optional_verify_node(state: Dict) -> Dict:
    if not state.get("verify_claims", False):
        state["verification"] = []
        _append_log(state, "Skipped claim verification")
        return state

    verifier = ClaimVerifier()
    verification = verifier.verify(state["evidence_board"])
    state["verification"] = verification
    _append_log(state, f"Claim verification completed ({len(verification)} records)")
    return state


def finalize_node(state: Dict) -> Dict:
    state["run_artifacts"] = {
        "manifest": state["manifest"],
        "scout_summary": state.get("scout_summary", {}),
        "background_findings": state.get("background_findings", []),
        "event_findings": state.get("event_findings", []),
        "parsed_context": state.get("parser_findings", []),
        "evidence_board": state.get("evidence_board"),
        "agent_deliberations": state.get("agent_deliberations", []),
        "llm_finding_proposals": state.get("llm_finding_proposals", {}),
        "llm_claim_planning": state.get("llm_claim_planning", {}),
        "atomic_claim_plan": state.get("atomic_claim_plan", []),
        "surface_decisions": state.get("surface_decisions", []),
        "detail": state.get("detail_section"),
        "impression": state.get("impression_section"),
        "verification": state.get("verification", []),
        "run_log": state.get("run_log", []),
    }
    _append_log(state, "Finalize done")
    return state
