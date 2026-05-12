from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_report_multiagent.evaluation.method_audit import audit_artifact_dir, render_audit_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one eeg_report_multiagent_v1 artifact directory")
    parser.add_argument("--artifact-dir", required=True, help="Run artifact directory containing evidence_board.json, manifest.json, etc.")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path. Defaults to artifact-dir/method_audit.json")
    parser.add_argument("--output-md", default=None, help="Optional output Markdown path. Defaults to artifact-dir/method_audit.md")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    audit = audit_artifact_dir(artifact_dir)
    output_json = Path(args.output_json) if args.output_json else artifact_dir / "method_audit.json"
    output_md = Path(args.output_md) if args.output_md else artifact_dir / "method_audit.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_audit_markdown(audit), encoding="utf-8")

    print(json.dumps({
        "overall_pass": audit.get("overall_pass"),
        "input_contract_pass": audit.get("input_contract", {}).get("pass"),
        "artifact_dir": str(artifact_dir),
        "output_json": str(output_json),
        "output_md": str(output_md),
        "counts": audit.get("counts"),
        "weak_measurement_flag_count": len(audit.get("weak_measurement_flags", [])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
