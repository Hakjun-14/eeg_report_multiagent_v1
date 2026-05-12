from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from eeg_report_multiagent.evaluation.clinical_provenance_audit import DEFAULT_VARIANTS, audit_cases


def _parse_row_indices(text: str | None) -> List[int] | None:
    if not text:
        return None
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_artifact_root(items: list[str] | None) -> Dict[str, Path]:
    roots: Dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--artifact-root must use VARIANT=/path syntax")
        name, path = item.split("=", 1)
        roots[name.strip()] = Path(path)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-pass clinical provenance audit over GT/generated report comparisons.")
    parser.add_argument(
        "--comparison-root",
        default="artifacts/gt_generated_comparison_selected50_UpgradeLLMProp",
        help="Root containing per_case_json from GT/generated comparison.",
    )
    parser.add_argument("--output-dir", default="artifacts/provenance_case_audits")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--row-indices", default=None, help="Comma-separated row indices, e.g. 189,548,783. Omit for all cases.")
    parser.add_argument("--variant", action="append", default=None, help="Variant name to include. Repeatable.")
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=None,
        help="Optional VARIANT=/path/to/rows mapping for EvidenceBoard lookup. Repeatable.",
    )
    args = parser.parse_args()

    variants = args.variant or DEFAULT_VARIANTS
    artifact_roots = {
        "Our_B": Path("artifacts/batch_s0001_test_B_selected50/rows"),
        "Our_D": Path("artifacts/batch_s0001_test_D_selected50/rows"),
        "Our_B_QFv2": Path("artifacts/batch_s0001_test_B_quality_floor_v2_selected50/rows"),
        "Our_Upgrade_LLMProp": Path("artifacts/batch_s0001_test_onepass_upgrade_llmprop_selected50/rows"),
    }
    artifact_roots.update(_parse_artifact_root(args.artifact_root))

    summary = audit_cases(
        comparison_root=Path(args.comparison_root),
        output_dir=Path(args.output_dir),
        row_indices=_parse_row_indices(args.row_indices),
        variants=variants,
        artifact_roots=artifact_roots,
        config_dir=Path(args.config_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
