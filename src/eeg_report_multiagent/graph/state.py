from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from eeg_report_multiagent.io.manifest_builder import SessionManifest
from eeg_report_multiagent.io.session_loader import EEGSessionData
from eeg_report_multiagent.schemas.agent import AgentDeliberationRecord
from eeg_report_multiagent.schemas.evidence import EvidenceBoard
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.report import ReportSection, VerificationRecord
from eeg_report_multiagent.schemas.tooling import ToolInvocationRecord


class PipelineState(TypedDict, total=False):
    session_dir: str
    report_json_path: Optional[str]
    report_text_path: Optional[str]
    metadata: Dict[str, str]
    verify_claims: bool
    enable_llm_review: bool

    session: EEGSessionData
    manifest: SessionManifest
    note_text: str
    clinical_context: Dict[str, Any]

    scout_summary: Dict[str, float]

    background_measurements: List[MeasurementValue]
    background_tool_invocations: List[ToolInvocationRecord]

    event_measurements: List[MeasurementValue]
    event_tool_invocations: List[ToolInvocationRecord]
    focused_windows: List[int]

    parser_measurements: List[MeasurementValue]
    parser_tool_invocations: List[ToolInvocationRecord]

    evidence_board: EvidenceBoard
    agent_deliberations: List[AgentDeliberationRecord]
    detail_section: ReportSection
    impression_section: ReportSection
    verification: List[VerificationRecord]

    run_log: List[str]
    run_artifacts: Dict[str, Any]
