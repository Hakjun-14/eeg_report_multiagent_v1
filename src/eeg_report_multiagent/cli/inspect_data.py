from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_report_multiagent.io import build_session_manifest, load_session_from_processed_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one processed EEG session directory")
    parser.add_argument("--session-dir", required=True, help="Path to processed_eeg/<session_folder>")
    args = parser.parse_args()

    session = load_session_from_processed_dir(Path(args.session_dir))
    manifest = build_session_manifest(session)

    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
