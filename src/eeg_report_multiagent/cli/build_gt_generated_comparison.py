from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from eeg_report_multiagent.evaluation.report_text_comparison import compare_text_concepts, normalize_text
from eeg_report_multiagent.io.celm_dataset import (
    read_split_rows,
    report_id_from_row,
    standardize_section_name,
    target_sections_from_row,
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_row_indices(path: Path | None, total_rows: int) -> List[int]:
    if path is None:
        return list(range(total_rows))
    out: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(int(line))
    return out


def _parse_variant(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--variant must use NAME=/path syntax")
    name, path = text.split("=", 1)
    root = Path(path)
    if (root / "generated_reports_json").exists():
        root = root / "generated_reports_json"
    return name.strip(), root


def _section_map_from_gt(report_json: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, List[str]] = {}
    eeg_payload = report_json.get("EEG_section_llm_extractions") or {}
    for section in eeg_payload.get("EEG_sections") or []:
        name = standardize_section_name(str(section.get("section_name") or ""))
        text = str(section.get("section_text") or "").strip()
        if name or text:
            out.setdefault(name, []).append(text)
    return {name: "\n".join(parts).strip() for name, parts in out.items()}


def _section_map_from_generated(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    out: Dict[str, List[str]] = {}
    for section in payload.get("report_sections") or []:
        name = standardize_section_name(str(section.get("section_name") or ""))
        text = str(section.get("section_text") or "").strip()
        if name or text:
            out.setdefault(name, []).append(text)
    return {name: "\n".join(parts).strip() for name, parts in out.items()}


def _tokens(text: str) -> List[str]:
    return normalize_text(text).split()


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _lexical_metrics(reference: str, generated: str) -> Dict[str, float]:
    ref = _tokens(reference)
    pred = _tokens(generated)
    if not ref or not pred:
        return {"bleu1": 0.0, "rouge1": 0.0, "rougeL": 0.0, "meteor": 0.0}
    ref_counts: Dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = 0
    for tok in pred:
        if ref_counts.get(tok, 0) > 0:
            overlap += 1
            ref_counts[tok] -= 1
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lcs = _lcs_len(ref, pred)
    rouge_l = lcs / len(ref)
    return {"bleu1": precision, "rouge1": f1, "rougeL": rouge_l, "meteor": f1}


def _section_comparison(section_name: str, gt_text: str, generated_text: str) -> Dict[str, Any]:
    concept = compare_text_concepts(gt_text, generated_text)
    lexical = _lexical_metrics(gt_text, generated_text)
    metrics = concept["metrics"]
    return {
        "section_name": section_name,
        "gt_text": gt_text,
        "generated_text": generated_text,
        "concept_precision": metrics["precision"],
        "concept_recall": metrics["recall"],
        "concept_f1": metrics["f1"],
        "missing_concepts": concept["missing_concepts"],
        "extra_concepts": concept["extra_concepts"],
        "numeric_missing": concept["numeric_missing"],
        "numeric_extra": concept["numeric_extra"],
        **lexical,
    }


def _aggregate(row_index: int, report_id: str, patient_id: str, variant: str, comps: List[Dict[str, Any]], exists: bool) -> Dict[str, Any]:
    def avg(key: str) -> float:
        vals = [float(c.get(key) or 0.0) for c in comps]
        return mean(vals) if vals else 0.0

    return {
        "row_index": row_index,
        "report_id": report_id,
        "patient_id": patient_id,
        "variant": variant,
        "generated_exists": "true" if exists else "false",
        "target_section_count": len(comps),
        "generated_char_len": sum(len(str(c.get("generated_text") or "")) for c in comps),
        "gt_char_len": sum(len(str(c.get("gt_text") or "")) for c in comps),
        "concept_precision_mean": avg("concept_precision"),
        "concept_recall_mean": avg("concept_recall"),
        "concept_f1_mean": avg("concept_f1"),
        "numeric_missing_count": sum(
            len(values)
            for c in comps
            for values in (c.get("numeric_missing") or {}).values()
        ),
        "numeric_extra_count": sum(
            len(values)
            for c in comps
            for values in (c.get("numeric_extra") or {}).values()
        ),
        "missing_concepts_top": "|".join(sorted({x for c in comps for x in c.get("missing_concepts", [])})[:24]),
        "extra_concepts_top": "|".join(sorted({x for c in comps for x in c.get("extra_concepts", [])})[:24]),
        "bleu1": avg("bleu1"),
        "rouge1": avg("rouge1"),
        "rougeL": avg("rougeL"),
        "meteor": avg("meteor"),
        "bertscore_f1": 0.0,
    }


def build_comparison(
    data_root: Path,
    site: str,
    split_type: str,
    split: str,
    row_indices_file: Path | None,
    variants: List[Tuple[str, Path]],
    output_dir: Path,
) -> Dict[str, Any]:
    split_rows = read_split_rows(data_root=data_root, site=site, split=split, split_type=split_type)
    row_indices = _read_row_indices(row_indices_file, len(split_rows))
    per_case_json = output_dir / "per_case_json"
    per_case_markdown = output_dir / "per_case_markdown"
    long_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []

    for row_index in row_indices:
        row = split_rows[row_index]
        report_id = report_id_from_row(row)
        report_json_path = data_root / "matched_eeg_recordings_report" / site / report_id / f"{report_id}.json"
        gt_map_all = _section_map_from_gt(_read_json(report_json_path))
        target_sections = [standardize_section_name(x) for x in target_sections_from_row(row)]
        gt_sections = {section: gt_map_all.get(section, "") for section in target_sections}
        case_payload: Dict[str, Any] = {
            "row_index": row_index,
            "report_id": report_id,
            "patient_id": row.get("BDSPPatientID", ""),
            "target_sections": target_sections,
            "gt_sections": gt_sections,
            "variants": {},
        }
        wide_row: Dict[str, Any] = {"row_index": row_index, "report_id": report_id, "patient_id": row.get("BDSPPatientID", "")}
        md_lines = [f"# row_{row_index:06d} {report_id}", ""]
        for variant, root in variants:
            generated_path = root / f"GENERATED_REPORT_{report_id}.json"
            generated_map = _section_map_from_generated(generated_path)
            comps = [
                _section_comparison(section, gt_sections.get(section, ""), generated_map.get(section, ""))
                for section in target_sections
            ]
            aggregate = _aggregate(row_index, report_id, row.get("BDSPPatientID", ""), variant, comps, generated_path.exists())
            case_payload["variants"][variant] = {
                "generated_sections": {section: generated_map.get(section, "") for section in target_sections},
                "section_comparisons": comps,
                "aggregate": aggregate,
            }
            long_rows.append(aggregate)
            for key, value in aggregate.items():
                if key not in {"row_index", "report_id", "patient_id", "variant"}:
                    wide_row[f"{variant}_{key}"] = value
            md_lines.extend([f"## {variant}", "", f"- generated_exists: {aggregate['generated_exists']}"])
            md_lines.append(f"- concept_f1_mean: {aggregate['concept_f1_mean']:.3f}")
            md_lines.append(f"- rougeL: {aggregate['rougeL']:.3f}")
            md_lines.append("")
        wide_rows.append(wide_row)
        _write_json(per_case_json / f"row_{row_index:06d}_{report_id}.json", case_payload)
        (per_case_markdown / f"row_{row_index:06d}_{report_id}.md").parent.mkdir(parents=True, exist_ok=True)
        (per_case_markdown / f"row_{row_index:06d}_{report_id}.md").write_text("\n".join(md_lines), encoding="utf-8")

    _write_csv(output_dir / "comparison_long_by_variant.csv", long_rows)
    _write_csv(output_dir / "comparison_wide_by_report.csv", wide_rows)
    summary_rows: List[Dict[str, Any]] = []
    for variant, _ in variants:
        subset = [r for r in long_rows if r["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "cases": len(subset),
                "generated_exists": sum(1 for r in subset if r["generated_exists"] == "true"),
                "concept_f1_mean": mean(float(r["concept_f1_mean"]) for r in subset) if subset else 0.0,
                "rougeL_mean": mean(float(r["rougeL"]) for r in subset) if subset else 0.0,
                "meteor_mean": mean(float(r["meteor"]) for r in subset) if subset else 0.0,
                "numeric_missing_mean": mean(float(r["numeric_missing_count"]) for r in subset) if subset else 0.0,
                "numeric_extra_mean": mean(float(r["numeric_extra_count"]) for r in subset) if subset else 0.0,
            }
        )
    _write_csv(output_dir / "comparison_summary_by_variant.csv", summary_rows)
    _write_json(output_dir / "comparison_summary_by_variant.json", summary_rows)
    (output_dir / "README.md").write_text(
        "# GT vs Generated Comparison\n\n"
        "Local lexical/concept comparison only. GT report text is used for evaluation, not inference.\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), "cases": len(row_indices), "variants": [name for name, _ in variants]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local GT/generated comparison artifacts for clinical provenance audit.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test")
    parser.add_argument("--row-indices-file", default=None)
    parser.add_argument("--variant", action="append", required=True, help="NAME=/path/to/generated_reports_json_or_results_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = build_comparison(
        data_root=Path(args.data_root),
        site=args.site,
        split_type=args.split_type,
        split=args.split,
        row_indices_file=Path(args.row_indices_file) if args.row_indices_file else None,
        variants=[_parse_variant(x) for x in args.variant],
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
