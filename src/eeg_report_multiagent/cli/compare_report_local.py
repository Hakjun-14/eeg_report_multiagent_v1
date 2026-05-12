from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_report_multiagent.evaluation.report_text_comparison import compare_generated_report_to_gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Locally compare GT report text with generated detail/impression")
    parser.add_argument("--gt-report-json", required=True, help="GT report JSON path; evaluation only")
    parser.add_argument("--artifact-dir", required=True, help="Run artifact directory containing detail.txt and impression.txt")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    result = compare_generated_report_to_gt(
        gt_report_json_path=Path(args.gt_report_json),
        generated_detail_path=artifact_dir / "detail.txt",
        generated_impression_path=artifact_dir / "impression.txt",
    )

    output_path = Path(args.output_json) if args.output_json else artifact_dir / "local_report_comparison.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved local comparison to: {output_path}")
    print(
        "detail "
        f"p={result['detail']['metrics']['precision']:.3f} "
        f"r={result['detail']['metrics']['recall']:.3f} "
        f"f1={result['detail']['metrics']['f1']:.3f}"
    )
    print(
        "impression "
        f"p={result['impression']['metrics']['precision']:.3f} "
        f"r={result['impression']['metrics']['recall']:.3f} "
        f"f1={result['impression']['metrics']['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
