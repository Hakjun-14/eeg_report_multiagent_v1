from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict

from eeg_report_multiagent.graph import nodes

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = "END"
    StateGraph = None


def _run_node_with_monitor(
    node_name: str,
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    state: Dict[str, Any],
    node_callback: NodeCallback | None,
) -> Dict[str, Any]:
    start_ts = time.time()
    spinner_frames = "|/-\\"
    monitor_interval_sec = 0.75
    _emit_node_event(node_callback, node_name, "start", state, {"elapsed_sec": 0.0, "spinner": spinner_frames[0]})

    if node_callback is None:
        out = fn(state)
        _emit_node_event(node_callback, node_name, "end", out, {"elapsed_sec": time.time() - start_ts})
        return out

    holder: Dict[str, Any] = {}
    error: Dict[str, Exception] = {}

    def _run_step() -> None:
        try:
            holder["state"] = fn(state)
        except Exception as exc:  # pragma: no cover - defensive path
            error["exc"] = exc

    t = threading.Thread(target=_run_step, daemon=True)
    t.start()
    spin_idx = 0
    while t.is_alive():
        _emit_node_event(
            node_callback,
            node_name,
            "running",
            state,
            {
                "elapsed_sec": time.time() - start_ts,
                "spinner": spinner_frames[spin_idx % len(spinner_frames)],
            },
        )
        spin_idx += 1
        time.sleep(monitor_interval_sec)

    t.join()
    if "exc" in error:
        _emit_node_event(
            node_callback,
            node_name,
            "error",
            state,
            {"elapsed_sec": time.time() - start_ts, "error": str(error["exc"])},
        )
        raise error["exc"]

    out = holder["state"]
    _emit_node_event(node_callback, node_name, "end", out, {"elapsed_sec": time.time() - start_ts})
    return out


def _maybe_monitored(
    node_name: str,
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    node_callback: NodeCallback | None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    if node_callback is None:
        return fn

    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        return _run_node_with_monitor(node_name, fn, state, node_callback)

    return _wrapped


def build_graph(include_optional_verifier: bool = True, node_callback: NodeCallback | None = None):
    if StateGraph is None:
        raise ImportError("langgraph is not available")

    graph = StateGraph(dict)
    graph.add_node("load_inputs", _maybe_monitored("load_inputs", nodes.load_inputs_node, node_callback))
    graph.add_node("scout_pass", _maybe_monitored("scout_pass", nodes.scout_pass_node, node_callback))
    graph.add_node("background_module", _maybe_monitored("background_module", nodes.background_module_node, node_callback))
    graph.add_node("event_module", _maybe_monitored("event_module", nodes.event_module_node, node_callback))
    graph.add_node("protocol_parser", _maybe_monitored("protocol_parser", nodes.protocol_parser_node, node_callback))
    graph.add_node("evidence_merge", _maybe_monitored("evidence_merge", nodes.evidence_merge_node, node_callback))
    graph.add_node("evidence_review", _maybe_monitored("evidence_review", nodes.evidence_review_node, node_callback))
    graph.add_node("finding_proposal", _maybe_monitored("finding_proposal", nodes.finding_proposal_node, node_callback))
    graph.add_node("report_synthesize", _maybe_monitored("report_synthesize", nodes.report_synthesize_node, node_callback))
    graph.add_node("optional_verify", _maybe_monitored("optional_verify", nodes.optional_verify_node, node_callback))
    graph.add_node("finalize", _maybe_monitored("finalize", nodes.finalize_node, node_callback))

    graph.set_entry_point("load_inputs")
    graph.add_edge("load_inputs", "scout_pass")
    graph.add_edge("scout_pass", "background_module")
    graph.add_edge("background_module", "event_module")
    graph.add_edge("event_module", "protocol_parser")
    graph.add_edge("protocol_parser", "evidence_merge")
    graph.add_edge("evidence_merge", "evidence_review")
    graph.add_edge("evidence_review", "finding_proposal")
    graph.add_edge("finding_proposal", "report_synthesize")

    if include_optional_verifier:
        graph.add_edge("report_synthesize", "optional_verify")
        graph.add_edge("optional_verify", "finalize")
    else:
        graph.add_edge("report_synthesize", "finalize")

    graph.add_edge("finalize", END)
    return graph.compile()


NodeCallback = Callable[[str, str, Dict[str, Any], Dict[str, Any] | None], None]


def _emit_node_event(
    node_callback: NodeCallback | None,
    node_name: str,
    phase: str,
    state: Dict[str, Any],
    meta: Dict[str, Any] | None = None,
) -> None:
    if node_callback is None:
        return
    try:
        node_callback(node_name, phase, state, meta)
    except Exception:
        # Monitoring callback should never break inference execution.
        pass


def run_pipeline(
    initial_state: Dict,
    use_langgraph: bool = True,
    node_callback: NodeCallback | None = None,
) -> Dict:
    if use_langgraph and StateGraph is not None:
        app = build_graph(include_optional_verifier=True, node_callback=node_callback)
        _emit_node_event(node_callback, "graph_invoke", "start", dict(initial_state), {})
        out = app.invoke(initial_state)
        _emit_node_event(node_callback, "graph_invoke", "end", out, {})
        return out

    # Sequential fallback preserving the same node order.
    state = dict(initial_state)
    steps = (
        ("load_inputs", nodes.load_inputs_node),
        ("scout_pass", nodes.scout_pass_node),
        ("background_module", nodes.background_module_node),
        ("event_module", nodes.event_module_node),
        ("protocol_parser", nodes.protocol_parser_node),
        ("evidence_merge", nodes.evidence_merge_node),
        ("evidence_review", nodes.evidence_review_node),
        ("finding_proposal", nodes.finding_proposal_node),
        ("report_synthesize", nodes.report_synthesize_node),
        ("optional_verify", nodes.optional_verify_node),
        ("finalize", nodes.finalize_node),
    )
    for node_name, fn in steps:
        state = _run_node_with_monitor(node_name, fn, state, node_callback)
    return state
