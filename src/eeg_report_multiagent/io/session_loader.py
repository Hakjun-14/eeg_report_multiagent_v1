from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .pkl_reader import EEGWindowData, load_window_pkl


@dataclass
class EEGSessionData:
    session_id: str
    session_dir: Path
    windows: List[EEGWindowData]
    signals: np.ndarray  # (N, C, T)
    channels: List[str]


def load_session_from_processed_dir(session_dir: Path) -> EEGSessionData:
    if not session_dir.exists() or not session_dir.is_dir():
        raise FileNotFoundError(f"session_dir does not exist: {session_dir}")

    files = sorted(session_dir.glob("seg_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"no seg_*.pkl files under: {session_dir}")

    windows = [load_window_pkl(fp) for fp in files]
    windows = sorted(windows, key=lambda w: w.window_index)

    signals = np.stack([w.signal for w in windows], axis=0)
    channels = windows[0].available_channels

    return EEGSessionData(
        session_id=session_dir.name,
        session_dir=session_dir,
        windows=windows,
        signals=signals,
        channels=channels,
    )
