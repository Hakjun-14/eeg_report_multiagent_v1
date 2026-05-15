from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from eeg_report_multiagent.io.celm_dataset import read_split_rows, report_id_from_row
from eeg_report_multiagent.modules.generated_claim_auditor import GeneratedClaimAuditor
from eeg_report_multiagent.schemas.generated_claim_audit import GeneratedClaimAuditResult

DEFAULT_DATA_ROOT = Path("/exHDD_8T/hjlee_data/eeg_data/celm_s_sites_pipeline")
DEFAULT_OUTPUT_DIR = Path("artifacts/stage2_95_atomic_claim_variant_audit")
DEFAULT_SELECTED_ROWS = Path("artifacts/experiment_ledgers/S0001_test_B_selected50_row_indices.txt")
DEFAULT_CELM_GENERATED_DIR = Path(
    "/home/hjlee/Desktop/eegagent/CELM_upstream/results/eeg_llm_projection_only/"
    "CELM_SCA_projector_inference_S0001_strict/cbramod_Qwen3-4B-Instruct-2507/"
    "inference_results_S0001/checkpoint_epoch_9/generated_reports_json"
)
DEFAULT_VARIANTS = {
    "CELM": DEFAULT_CELM_GENERATED_DIR,
    "Our_EvidenceGated_v1": Path("artifacts/batch_s0001_test_Our_EvidenceGated_v1_selected50/celm_results/generated_reports_json"),
    "FormatFitAggressive_v0": Path("artifacts/batch_s0001_test_FormatFitAggressive_v0/celm_results/generated_reports_json"),
}
ANCHOR_CASES = {
    "row_000189_NeuroReport_13450252291_593274052_20180914": "row189_claim_comparison.md",
    "row_000548_NeuroReport_13341375710_527779290_20190820": "row548_claim_comparison.md",
    "row_000783_NeuroReport_13324097354_322634096_20180129": "row783_claim_comparison.md",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated reports against GT using symmetric atomic claim extraction.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--row-indices-file", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant mapping NAME=/path/to/generated_reports_json. If omitted, standard selected50 variants are used.",
    )
    parser.add_argument(
        "--require-all-variants",
        action="store_true",
        help="Evaluate only rows where every requested variant has a generated report. Use for fair paired comparison.",
    )
    args = parser.parse_args()

    variants = _parse_variants(args.variant) if args.variant else DEFAULT_VARIANTS
    rows = read_split_rows(args.data_root, site=args.site, split=args.split, split_type=args.split_type)
    selected_indices = _read_indices(args.row_indices_file)
    if args.require_all_variants:
        selected_indices = _filter_indices_with_all_variants(selected_indices, rows, variants)
    auditor = GeneratedClaimAuditor()

    out = args.output_dir
    per_case_dir = out / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[GeneratedClaimAuditResult] = []
    generated_rows: list[dict[str, Any]] = []
    gt_recall_rows: list[dict[str, Any]] = []
    missing_examples: list[dict[str, Any]] = []
    extra_examples: list[dict[str, Any]] = []

    for row_index in selected_indices:
        row = rows[row_index]
        report_id = report_id_from_row(row)
        case_id = f"row_{row_index:06d}_{report_id}"
        gt_report_json = args.data_root / "matched_eeg_recordings_report" / args.site / report_id / f"{report_id}.json"
        for variant, generated_dir in variants.items():
            generated_report = generated_dir / f"GENERATED_REPORT_{report_id}.json"
            if not generated_report.exists():
                print(f"[warn] missing generated report for {variant} {case_id}: {generated_report}")
                continue
            result = auditor.audit_case(
                case_id=case_id,
                variant=variant,
                gt_report_json=gt_report_json,
                generated_report_json=generated_report,
            )
            all_results.append(result)
            _write_json(per_case_dir / f"{variant}_{case_id}_claim_audit.json", result.model_dump(mode="json"))
            for match in result.generated_claim_matches:
                row_payload = _generated_match_row(result, match)
                generated_rows.append(row_payload)
                if match.is_extra_claim:
                    extra_examples.append(row_payload)
            for match in result.gt_claim_recall_matches:
                row_payload = _gt_recall_row(result, match)
                gt_recall_rows.append(row_payload)
                if match.is_missing:
                    missing_examples.append(row_payload)

    aggregate_rows = _aggregate(all_results)
    _write_csv(out / "variant_claim_metrics.csv", aggregate_rows)
    _write_json(out / "variant_claim_metrics.json", aggregate_rows)
    _write_csv(out / "generated_claim_matches.csv", generated_rows)
    _write_csv(out / "gt_claim_recall_matches.csv", gt_recall_rows)
    _write_text(out / "variant_claim_comparison.md", _variant_markdown(aggregate_rows, all_results))
    _write_text(out / "extra_claim_examples.md", _examples_md("Extra Generated Atomic Claims", extra_examples, limit=40))
    _write_text(out / "missing_gt_claim_examples.md", _examples_md("Missing GT Atomic Claims", missing_examples, limit=40))

    for case_id, file_name in ANCHOR_CASES.items():
        case_results = [result for result in all_results if result.case_id == case_id]
        if case_results:
            _write_text(out / file_name, _case_markdown(case_id, case_results))

    print(f"Wrote generated-claim audit artifacts to: {out}")


def _filter_indices_with_all_variants(indices: list[int], rows: list[dict[str, str]], variants: dict[str, Path]) -> list[int]:
    kept: list[int] = []
    for row_index in indices:
        report_id = report_id_from_row(rows[row_index])
        if all((generated_dir / f"GENERATED_REPORT_{report_id}.json").exists() for generated_dir in variants.values()):
            kept.append(row_index)
    return kept


def _parse_variants(values: Iterable[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Variant must be NAME=/path, got: {value}")
        name, raw_path = value.split("=", 1)
        variants[name] = Path(raw_path)
    return variants


def _read_indices(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _generated_match_row(result: GeneratedClaimAuditResult, match: Any) -> dict[str, Any]:
    claim = match.generated_claim
    return {
        "variant": result.variant,
        "case_id": result.case_id,
        "generated_claim_id": match.generated_claim_id,
        "claim_type": claim.claim_type,
        "section": claim.section,
        "normalized_value": _jsonish(claim.normalized_value),
        "unit": claim.unit or "",
        "certainty": claim.certainty,
        "source_text": claim.source_text,
        "matched_gt_claim_ids": "|".join(match.matched_gt_claim_ids),
        "is_extra_claim": int(match.is_extra_claim),
    }


def _gt_recall_row(result: GeneratedClaimAuditResult, match: Any) -> dict[str, Any]:
    claim = match.gt_claim
    return {
        "variant": result.variant,
        "case_id": result.case_id,
        "gt_claim_id": match.gt_claim_id,
        "claim_type": claim.claim_type,
        "section": claim.section,
        "normalized_value": _jsonish(claim.normalized_value),
        "unit": claim.unit or "",
        "certainty": claim.certainty,
        "source_text": claim.source_text,
        "matched_generated_claim_ids": "|".join(match.matched_generated_claim_ids),
        "is_missing": int(match.is_missing),
    }


def _aggregate(results: list[GeneratedClaimAuditResult]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[GeneratedClaimAuditResult]] = defaultdict(list)
    for result in results:
        by_variant[result.variant].append(result)
    rows: list[dict[str, Any]] = []
    for variant, variant_results in sorted(by_variant.items()):
        sums = Counter()
        for result in variant_results:
            sums.update({key: value for key, value in result.metrics.items() if key.endswith("Count")})
        total_gt = int(sum(result.metrics.get("GTClaimCount", 0.0) for result in variant_results))
        total_gen = int(sum(result.metrics.get("GeneratedClaimCount", 0.0) for result in variant_results))
        matched_gt = int(sum(result.metrics.get("MatchedGTClaimCount", 0.0) for result in variant_results))
        matched_gen = int(sum(result.metrics.get("MatchedGeneratedClaimCount", 0.0) for result in variant_results))
        numeric_gt = int(sum(result.metrics.get("NumericGTClaimCount", 0.0) for result in variant_results))
        numeric_gen = int(sum(result.metrics.get("NumericGeneratedClaimCount", 0.0) for result in variant_results))
        numeric_gt_matched = int(sum(1 for result in variant_results for match in result.gt_claim_recall_matches if match.gt_claim.claim_type in {"pdr_frequency", "background_amplitude", "event_amplitude", "event_duration", "event_frequency"} and not match.is_missing))
        numeric_gen_matched = int(sum(1 for result in variant_results for match in result.generated_claim_matches if match.generated_claim.claim_type in {"pdr_frequency", "background_amplitude", "event_amplitude", "event_duration", "event_frequency"} and not match.is_extra_claim))
        rows.append(
            {
                "variant": variant,
                "num_cases": len(variant_results),
                "GTClaimCount": total_gt,
                "GeneratedClaimCount": total_gen,
                "MatchedGTClaimCount": matched_gt,
                "MatchedGeneratedClaimCount": matched_gen,
                "GTClaimRecall": _safe_div(matched_gt, total_gt),
                "GeneratedClaimPrecision": _safe_div(matched_gen, total_gen),
                "MissingGTClaimRate": _safe_div(total_gt - matched_gt, total_gt),
                "ExtraClaimRate": _safe_div(total_gen - matched_gen, total_gen),
                "AvgGTClaimsPerReport": mean([result.metrics.get("GTClaimCount", 0.0) for result in variant_results]),
                "AvgGeneratedClaimsPerReport": mean([result.metrics.get("GeneratedClaimCount", 0.0) for result in variant_results]),
                "NumericGTClaimRecall": _safe_div(numeric_gt_matched, numeric_gt),
                "NumericGeneratedClaimPrecision": _safe_div(numeric_gen_matched, numeric_gen),
                "TraceabilityMode": "text_only_for_CELM; evidence_trace_available_only_for_OURS_artifacts",
            }
        )
    return rows


def _variant_markdown(rows: list[dict[str, Any]], results: list[GeneratedClaimAuditResult]) -> str:
    lines = ["# Stage 2.95 Atomic Claim Variant Audit", ""]
    lines.append("This audit uses the same atomic claim extractor for GT and generated reports. It is a text-level clinical-slot comparison, not signal provenance validation.")
    lines.append("")
    headers = ["variant", "cases", "GT claims", "generated claims", "GT recall", "generated precision", "extra rate", "missing rate", "numeric GT recall"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join([
            str(row["variant"]),
            str(row["num_cases"]),
            str(row["GTClaimCount"]),
            str(row["GeneratedClaimCount"]),
            f"{row['GTClaimRecall']:.3f}",
            f"{row['GeneratedClaimPrecision']:.3f}",
            f"{row['ExtraClaimRate']:.3f}",
            f"{row['MissingGTClaimRate']:.3f}",
            f"{row['NumericGTClaimRecall']:.3f}",
        ]) + " |")
    lines.extend([
        "",
        "## Interpretation Guardrails",
        "",
        "- CELM is evaluated with the same generated-text atomic claim extractor as OURS variants.",
        "- These numbers should be reported alongside BLEU/ROUGE/METEOR, not instead of them.",
        "- For CELM, this audit cannot prove patient-specific signal grounding because CELM outputs do not include EvidenceItem or AtomicClaimPlan provenance.",
        "- For OURS, low recall with high safety means the gate is still under-informative; high precision without traceability would not be enough for the paper claim.",
    ])
    claim_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        for match in result.gt_claim_recall_matches:
            if match.is_missing:
                claim_type_counts[result.variant][match.gt_claim.claim_type] += 1
    lines.extend(["", "## Top Missing GT Claim Types", ""])
    for variant, counts in sorted(claim_type_counts.items()):
        lines.append(f"### {variant}")
        for claim_type, count in counts.most_common(10):
            lines.append(f"- `{claim_type}`: {count}")
        lines.append("")
    return "\n".join(lines)


def _case_markdown(case_id: str, results: list[GeneratedClaimAuditResult]) -> str:
    lines = [f"# {case_id} Atomic Claim Comparison", ""]
    for result in sorted(results, key=lambda r: r.variant):
        metrics = result.metrics
        lines.append(f"## {result.variant}")
        lines.append(f"- GTClaimRecall: `{metrics['GTClaimRecall']:.3f}`")
        lines.append(f"- GeneratedClaimPrecision: `{metrics['GeneratedClaimPrecision']:.3f}`")
        lines.append(f"- ExtraClaimRate: `{metrics['ExtraClaimRate']:.3f}`")
        lines.append(f"- GeneratedClaimCount: `{int(metrics['GeneratedClaimCount'])}`")
        lines.append("")
        lines.append("### Extra Generated Claims")
        extras = [m for m in result.generated_claim_matches if m.is_extra_claim]
        if extras:
            for match in extras[:12]:
                lines.append(f"- `{match.generated_claim.claim_type}`: {match.generated_claim.source_text[:180]}")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("### Missing GT Claims")
        missing = [m for m in result.gt_claim_recall_matches if m.is_missing]
        if missing:
            for match in missing[:12]:
                lines.append(f"- `{match.gt_claim.claim_type}`: {match.gt_claim.source_text[:180]}")
        else:
            lines.append("- None")
        lines.append("")
    return "\n".join(lines)


def _examples_md(title: str, rows: list[dict[str, Any]], limit: int) -> str:
    lines = [f"# {title}", ""]
    for idx, row in enumerate(rows[:limit], start=1):
        lines.append(f"## {idx}. {row['variant']} / {row['case_id']} / {row['claim_type']}")
        lines.append(f"- text: {row['source_text']}")
        if "matched_gt_claim_ids" in row:
            lines.append(f"- matched_gt_claim_ids: `{row['matched_gt_claim_ids']}`")
        if "matched_generated_claim_ids" in row:
            lines.append(f"- matched_generated_claim_ids: `{row['matched_generated_claim_ids']}`")
        lines.append("")
    if not rows:
        lines.append("No examples found.")
    return "\n".join(lines)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonish(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
