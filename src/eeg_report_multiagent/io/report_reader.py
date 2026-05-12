from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_report_json(report_json_path: Optional[Path]) -> Dict[str, Any]:
    if report_json_path is None:
        return {}
    with report_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_report_text(report_text_path: Optional[Path]) -> str:
    if report_text_path is None:
        return ""
    return report_text_path.read_text(encoding="utf-8", errors="ignore")


def get_note_text(report_json: Dict[str, Any], fallback_text: str = "") -> str:
    note = report_json.get("note_text")
    if isinstance(note, str) and note.strip():
        return note
    return fallback_text
