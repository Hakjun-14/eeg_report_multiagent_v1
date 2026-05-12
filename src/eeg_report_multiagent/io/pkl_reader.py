from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class EEGWindowData:
    file_path: Path
    window_index: int
    signal: np.ndarray
    available_channels: List[str]
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]


def parse_window_index(file_path: Path) -> int:
    stem = file_path.stem
    # expected: seg_<idx>_...
    parts = stem.split("_")
    if len(parts) < 2 or parts[0] != "seg":
        raise ValueError(f"unexpected segment filename format: {file_path.name}")
    return int(parts[1])


def load_window_pkl(file_path: Path) -> EEGWindowData:
    with file_path.open("rb") as f:
        payload = pickle.load(f)

    signal = np.asarray(payload["signal"], dtype=np.float32)
    available_channels = list(payload.get("available_channels", []))
    mean = payload.get("mean_eeg_data", payload.get("mean"))
    std = payload.get("std_eeg_data", payload.get("std"))

    return EEGWindowData(
        file_path=file_path,
        window_index=parse_window_index(file_path),
        signal=signal,
        available_channels=available_channels,
        mean=None if mean is None else np.asarray(mean, dtype=np.float32),
        std=None if std is None else np.asarray(std, dtype=np.float32),
    )
