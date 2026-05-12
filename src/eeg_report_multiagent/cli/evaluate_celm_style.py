from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from eeg_report_multiagent.io.celm_dataset import read_split_rows, report_id_from_row, standardize_section_name


SCORE_COLUMNS = [
    "bleu-1",
    "bleu-4",
    "bleu-1-smooth",
    "bleu-4-smooth",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "perplexity",
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "meteor",
]


ZERO_ROW = {
    "bleu-1": 0.0,
    "bleu-4": 0.0,
    "bleu-1-smooth": 0.0,
    "bleu-4-smooth": 0.0,
    "bertscore_precision": 0.0,
    "bertscore_recall": 0.0,
    "bertscore_f1": 0.0,
    "perplexity": 0.0,
    "rouge1": 0.0,
    "rouge2": 0.0,
    "rougeL": 0.0,
    "rougeLsum": 0.0,
    "meteor": 0.0,
}


def _load_evaluate():
    try:
        import evaluate  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing evaluation dependency 'evaluate'. Install with: pip install -e '.[eval]'"
        ) from exc
    return evaluate


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_scores_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["deidentified_name"] + SCORE_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


class CELMStyleEvaluator:
    """Local reproduction of CELM section-wise text metrics without touching raw EEG."""

    def __init__(self, model_name: str, ignore_bertscore: bool = False, include_perplexity: bool = False) -> None:
        self.model_name = model_name
        self.ignore_bertscore = ignore_bertscore
        self.include_perplexity = include_perplexity
        evaluate = _load_evaluate()
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.bertscore = None if ignore_bertscore else evaluate.load("bertscore")
        self.perplexity = evaluate.load("perplexity") if include_perplexity else None

    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, list):
            return float(mean(float(x) for x in value)) if value else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _calculate_metrics(self, references: List[str], predictions: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            out["bleu_score_results"] = {
                "bleu-1": self.bleu.compute(references=references, predictions=predictions, max_order=1, smooth=False),
                "bleu-4": self.bleu.compute(references=references, predictions=predictions, max_order=4, smooth=False),
                "bleu-1-smooth": self.bleu.compute(
                    references=references, predictions=predictions, max_order=1, smooth=True
                ),
                "bleu-4-smooth": self.bleu.compute(
                    references=references, predictions=predictions, max_order=4, smooth=True
                ),
            }
        except Exception as exc:
            out["bleu_error"] = repr(exc)
            out["bleu_score_results"] = {
                "bleu-1": {"bleu": 0.0},
                "bleu-4": {"bleu": 0.0},
                "bleu-1-smooth": {"bleu": 0.0},
                "bleu-4-smooth": {"bleu": 0.0},
            }

        if self.bertscore is None:
            out["bertscore_results"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            try:
                out["bertscore_results"] = self.bertscore.compute(
                    references=references,
                    predictions=predictions,
                    lang="en",
                    model_type="distilbert-base-uncased",
                    device="cpu",
                )
            except Exception as exc:
                out["bertscore_error"] = repr(exc)
                out["bertscore_results"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if self.perplexity is None:
            out["perplexity_results"] = {"mean_perplexity": 0.0, "perplexities": 0.0}
        else:
            try:
                out["perplexity_results"] = self.perplexity.compute(predictions=predictions, model_id=self.model_name)
            except Exception as exc:
                out["perplexity_error"] = repr(exc)
                out["perplexity_results"] = {"mean_perplexity": 0.0, "perplexities": 0.0}

        try:
            out["rouge_score_results"] = self.rouge.compute(references=references, predictions=predictions)
        except Exception as exc:
            out["rouge_error"] = repr(exc)
            out["rouge_score_results"] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        try:
            out["meteor_score_results"] = self.meteor.compute(references=references, predictions=predictions)
        except Exception as exc:
            out["meteor_error"] = repr(exc)
            out["meteor_score_results"] = {"meteor": 0.0}

        return out

    def section_wise_metrics(
        self,
        reference_report: Dict[str, Any],
        generated_report: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        raw_section_names = reference_report.get("extracted_eeg_section_names") or []
        section_names = [standardize_section_name(str(name)) for name in raw_section_names]
        reference_sections = {}
        eeg_payload = reference_report.get("EEG_section_llm_extractions") or {}
        for section in eeg_payload.get("EEG_sections") or []:
            section_name = standardize_section_name(str(section.get("section_name") or ""))
            reference_sections.setdefault(section_name, []).append(str(section.get("section_text") or ""))

        generated_sections = {}
        for section in generated_report.get("report_sections") or []:
            section_name = str(section.get("section_name") or "").strip()
            generated_sections.setdefault(section_name.lower(), []).append(str(section.get("section_text") or ""))

        section_scores: Dict[str, Optional[Dict[str, Any]]] = {}
        for section_name in section_names:
            generated_texts = generated_sections.get(section_name.lower().strip())
            reference_texts = reference_sections.get(section_name)
            if generated_texts and reference_texts:
                section_scores[section_name] = self._calculate_metrics(reference_texts, generated_texts)
            else:
                section_scores[section_name] = None
        return section_scores

    def overall_metrics(self, section_scores: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, float]:
        if not section_scores:
            return dict(ZERO_ROW)

        rows = []
        for score in section_scores.values():
            if score is None:
                rows.append(dict(ZERO_ROW))
                continue
            bleu = score["bleu_score_results"]
            bert = score["bertscore_results"]
            ppl = score["perplexity_results"]
            rouge = score["rouge_score_results"]
            meteor = score["meteor_score_results"]
            rows.append(
                {
                    "bleu-1": self._scalar(bleu["bleu-1"].get("bleu")),
                    "bleu-4": self._scalar(bleu["bleu-4"].get("bleu")),
                    "bleu-1-smooth": self._scalar(bleu["bleu-1-smooth"].get("bleu")),
                    "bleu-4-smooth": self._scalar(bleu["bleu-4-smooth"].get("bleu")),
                    "bertscore_precision": self._scalar(bert.get("precision")),
                    "bertscore_recall": self._scalar(bert.get("recall")),
                    "bertscore_f1": self._scalar(bert.get("f1")),
                    "perplexity": self._scalar(ppl.get("mean_perplexity")),
                    "rouge1": self._scalar(rouge.get("rouge1")),
                    "rouge2": self._scalar(rouge.get("rouge2")),
                    "rougeL": self._scalar(rouge.get("rougeL")),
                    "rougeLsum": self._scalar(rouge.get("rougeLsum")),
                    "meteor": self._scalar(meteor.get("meteor")),
                }
            )
        return {col: mean(row[col] for row in rows) for col in SCORE_COLUMNS}


def evaluate_results(
    data_root: Path,
    results_saved_path: Path,
    site: str,
    split_type: str,
    split: str,
    model_name: str,
    ignore_bertscore: bool,
    include_perplexity: bool,
) -> Dict[str, Any]:
    split_rows = read_split_rows(data_root=data_root, site=site, split=split, split_type=split_type)
    generated_dir = results_saved_path / "generated_reports_json"
    score_dir = results_saved_path / "scores"
    evaluator = CELMStyleEvaluator(
        model_name=model_name,
        ignore_bertscore=ignore_bertscore,
        include_perplexity=include_perplexity,
    )

    output_rows: List[Dict[str, Any]] = []
    generated_count = 0
    for row in split_rows:
        report_id = report_id_from_row(row)
        reference_path = data_root / "matched_eeg_recordings_report" / site / report_id / f"{report_id}.json"
        generated_path = generated_dir / f"GENERATED_REPORT_{report_id}.json"
        if generated_path.exists():
            generated_count += 1
            reference_report = _read_json(reference_path)
            generated_report = _read_json(generated_path)
            section_scores = evaluator.section_wise_metrics(reference_report, generated_report)
            overall_scores = evaluator.overall_metrics(section_scores)
            final_scores = {"section_wise_scores": section_scores, "overall_scores": overall_scores}
        else:
            overall_scores = dict(ZERO_ROW)
            final_scores = {"section_wise_scores": None, "overall_scores": None}

        _write_json(score_dir / f"{report_id}.json", final_scores)
        output_rows.append({"deidentified_name": report_id, **overall_scores})

    _write_scores_csv(results_saved_path / "overall_scores.csv", output_rows)
    summary = {
        "split_rows": len(split_rows),
        "generated_reports": generated_count,
        "overall_scores_csv": str(results_saved_path / "overall_scores.csv"),
        "scores_dir": str(score_dir),
        "ignore_bertscore": ignore_bertscore,
        "include_perplexity": include_perplexity,
    }
    _write_json(results_saved_path / "evaluation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated report JSONs with CELM-style section-wise metrics.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-saved-path", required=True)
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--ignore-bertscore", action="store_true")
    parser.add_argument("--include-perplexity", action="store_true")
    args = parser.parse_args()

    summary = evaluate_results(
        data_root=Path(args.data_root),
        results_saved_path=Path(args.results_saved_path),
        site=args.site,
        split_type=args.split_type,
        split=args.split,
        model_name=args.model_name,
        ignore_bertscore=args.ignore_bertscore,
        include_perplexity=args.include_perplexity,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
