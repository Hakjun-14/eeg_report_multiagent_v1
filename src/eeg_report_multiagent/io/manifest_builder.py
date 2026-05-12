from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .session_loader import EEGSessionData


class WindowManifestItem(BaseModel):
    window_index: int
    start_sec: float
    end_sec: float
    source_pkl_path: str


class SessionManifest(BaseModel):
    session_id: str
    sample_rate_hz: int
    window_seconds: int
    shape_nct: List[int]
    channels: List[str]
    source_pkl_paths: List[str]
    windows: List[WindowManifestItem]
    metadata_availability: dict = Field(default_factory=dict)


def build_session_manifest(
    session: EEGSessionData,
    sample_rate_hz: int = 200,
    window_seconds: int = 10,
    report_json_path: Optional[str] = None,
    report_text_path: Optional[str] = None,
    study_context_json_path: Optional[str] = None,
    study_context_text_path: Optional[str] = None,
    gt_report_json_path: Optional[str] = None,
    metadata_row_available: bool = False,
) -> SessionManifest:
    windows = []
    for w in session.windows:
        start_sec = float(w.window_index * window_seconds)
        end_sec = float(start_sec + window_seconds)
        windows.append(
            WindowManifestItem(
                window_index=w.window_index,
                start_sec=start_sec,
                end_sec=end_sec,
                source_pkl_path=str(w.file_path),
            )
        )

    return SessionManifest(
        session_id=session.session_id,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        shape_nct=[int(x) for x in session.signals.shape],
        channels=session.channels,
        source_pkl_paths=[str(w.file_path) for w in session.windows],
        windows=windows,
        metadata_availability={
            "study_context_json": bool(study_context_json_path or report_json_path),
            "study_context_text": bool(study_context_text_path or report_text_path),
            "gt_report_json_eval_only": bool(gt_report_json_path),
            "metadata_row": bool(metadata_row_available),
        },
    )
