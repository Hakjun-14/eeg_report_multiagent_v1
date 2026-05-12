import json
import pickle
from pathlib import Path

import numpy as np

from eeg_report_multiagent.graph.builder import run_pipeline


def _write_seg(path: Path, idx: int) -> None:
    signal = np.random.randn(22, 2000).astype("float32")
    payload = {
        "available_channels": [
            "C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"
        ],
        "mean_eeg_data": signal.mean(axis=1),
        "std_eeg_data": signal.std(axis=1),
        "signal": signal,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_end_to_end_smoke(tmp_path: Path) -> None:
    session_dir = tmp_path / "sub-test_ses-1"
    session_dir.mkdir(parents=True)
    _write_seg(session_dir / "seg_0_sub-test_ses-1.pkl", 0)
    _write_seg(session_dir / "seg_1_sub-test_ses-1.pkl", 1)

    report_json = tmp_path / "report.json"
    report_json.write_text(
        json.dumps(
            {
                "note_text": "awake state. hyperventilation: na. photic stimulation: no response.",
            }
        ),
        encoding="utf-8",
    )

    state = {
        "session_dir": str(session_dir),
        "report_json_path": str(report_json),
        "report_text_path": None,
        "metadata": {"ekg_available": "true", "video_available": "true"},
        "verify_claims": True,
        "run_log": [],
    }

    out = run_pipeline(state, use_langgraph=False)

    assert out["manifest"].shape_nct == [2, 22, 2000]
    assert "global_slowing_hint" in out["scout_summary"]
    assert len(out["background_findings"]) > 0
    assert len(out["event_findings"]) > 0
    assert len(out["parser_findings"]) > 0
    assert out["evidence_board"].session_id == "sub-test_ses-1"
    assert out["detail_section"].text
    assert out["impression_section"].text
