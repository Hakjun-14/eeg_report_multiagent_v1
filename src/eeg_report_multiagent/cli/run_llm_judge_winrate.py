from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from eeg_report_multiagent.io.celm_dataset import (
    read_split_rows,
    report_id_from_row,
    standardize_section_name,
    target_sections_from_row,
)

JUDGE_PROMPT = """You are an expert clinical EEG report evaluator. Your task is to evaluate generated EEG clinical reports by comparing them against a ground-truth EEG report.

You must act as a strict, evidence-grounded judge. Do not reward medically plausible statements unless they are supported by the ground-truth report. Do not infer additional clinical findings that are not present in the ground truth. Penalize hallucinated abnormalities, missing clinically important findings, incorrect normal/abnormal impressions, wrong lateralization, wrong frequency bands, wrong morphology, and unsupported diagnostic conclusions.

You will be given:

1. The ground-truth EEG report.
2. Multiple anonymized generated EEG reports from different systems.

Evaluate each generated report independently against the ground truth. Do not assume any report is produced by a particular model. Do not compare reports based on writing style alone. Focus on clinical fidelity, findings coverage, and clinical usefulness.

Use the following scoring scale for each metric:
1 = Very poor: clinically misleading, mostly incorrect, or missing major findings.
2 = Poor: contains some relevant information but has major omissions or clinically important errors.
3 = Fair: partially correct, captures some key findings, but misses or distorts important details.
4 = Good: mostly correct, covers most key findings, with only minor omissions or minor wording issues.
5 = Very good: clinically faithful to the ground truth, complete, and suitable as a draft clinical report.

Evaluate each report on the following three primary metrics:

Metric 1: Clinical Correctness
Assess whether the generated report is clinically accurate compared with the ground truth. Consider whether it correctly identifies normal vs abnormal EEG, presence or absence of seizures, epileptiform abnormalities, slowing, background abnormalities, sleep/drowsiness, and the final impression. Penalize hallucinated findings and incorrect clinical conclusions.

Metric 2: Findings Coverage and Attribute Accuracy
Assess whether the generated report covers the key findings in the ground truth and accurately describes their attributes. Consider lateralization, localization, frequency, morphology, amplitude if present, temporal pattern, state dependence, and whether important negative findings are preserved. Penalize omissions of clinically important findings and incorrect attributes.

Metric 3: Report Completeness and Real-World Readiness
Assess whether the report is complete, coherent, and usable as a clinical EEG report draft. Consider whether all relevant report sections are addressed, whether the impression follows from the described findings, whether the report avoids irrelevant or unsupported content, and whether it would be appropriate for clinician review.

For each generated report, provide:

* clinical_correctness_score: integer from 1 to 5
* findings_coverage_attribute_accuracy_score: integer from 1 to 5
* completeness_real_world_readiness_score: integer from 1 to 5
* overall_score: the average of the three scores, rounded to two decimal places
* major_omissions: a short list of important findings from the ground truth that are missing
* major_errors_or_hallucinations: a short list of clinically incorrect or unsupported statements
* brief_rationale: 2-4 sentences explaining the scores

After evaluating all reports, provide a preference ranking from best to worst. Ties are allowed only if the reports are clinically indistinguishable in quality.

Return only valid JSON. Do not include markdown, commentary, or any text outside the JSON object.
"""

OUTPUT_SCHEMA_HINT = """Required output format:
{
  "case_id": "<CASE_ID>",
  "evaluations": [
    {
      "report_id": "A",
      "clinical_correctness_score": 0,
      "findings_coverage_attribute_accuracy_score": 0,
      "completeness_real_world_readiness_score": 0,
      "overall_score": 0.0,
      "major_omissions": [],
      "major_errors_or_hallucinations": [],
      "brief_rationale": ""
    }
  ],
  "preference_ranking": [
    {"rank": 1, "report_id": "", "reason": ""}
  ]
}
"""

SECTION_KEYS = [
    "EEG_DESCRIPTION_DETAILS",
    "BACKGROUND_ACTIVITY",
    "EPILEPTIFORM_ABNORMALITIES",
    "INTERICTAL_EPILEPTIFORM_ABNORMALITIES",
    "EVENTS_SEIZURES",
    "IMPRESSION_INTERPRETATION",
]

SECTION_ALIASES = {
    "EEG DESCRIPTION/DETAILS": "EEG_DESCRIPTION_DETAILS",
    "EEG DESCRIPTION": "EEG_DESCRIPTION_DETAILS",
    "DETAILS": "EEG_DESCRIPTION_DETAILS",
    "BACKGROUND ACTIVITY": "BACKGROUND_ACTIVITY",
    "BACKGROUND": "BACKGROUND_ACTIVITY",
    "EPILEPTIFORM ABNORMALITIES": "EPILEPTIFORM_ABNORMALITIES",
    "INTERICTAL EPILEPTIFORM ABNORMALITIES": "INTERICTAL_EPILEPTIFORM_ABNORMALITIES",
    "EVENTS/SEIZURES": "EVENTS_SEIZURES",
    "EVENTS SEIZURES": "EVENTS_SEIZURES",
    "SEIZURES": "EVENTS_SEIZURES",
    "IMPRESSION/INTERPRETATION": "IMPRESSION_INTERPRETATION",
    "IMPRESSION": "IMPRESSION_INTERPRETATION",
    "INTERPRETATION": "IMPRESSION_INTERPRETATION",
}


@dataclass(frozen=True)
class CaseSpec:
    row_index: int
    report_id: str
    patient_id: str
    gt_sections: dict[str, str]
    target_sections: list[str]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _canonical_section_key(name: str) -> str:
    std = standardize_section_name(str(name or ""))
    return SECTION_ALIASES.get(std, std.replace("/", "_").replace(" ", "_"))


def _empty_section_dict() -> dict[str, str]:
    return {key: "" for key in SECTION_KEYS}


def _section_map_from_gt(report_json: dict[str, Any]) -> dict[str, str]:
    out = _empty_section_dict()
    eeg_payload = report_json.get("EEG_section_llm_extractions") or {}
    for section in eeg_payload.get("EEG_sections") or []:
        key = _canonical_section_key(str(section.get("section_name") or ""))
        text = str(section.get("section_text") or "").strip()
        if text:
            out[key] = "\n".join([x for x in [out.get(key, ""), text] if x]).strip()
    return out


def _section_map_from_generated(path: Path) -> dict[str, str]:
    out = _empty_section_dict()
    if not path.exists():
        return out
    payload = _read_json(path)
    if isinstance(payload, dict):
        sections = payload.get("report_sections") or []
    elif isinstance(payload, list):
        sections = payload
    else:
        sections = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        key = _canonical_section_key(str(section.get("section_name") or ""))
        text = str(section.get("section_text") or "").strip()
        if text:
            out[key] = "\n".join([x for x in [out.get(key, ""), text] if x]).strip()
    return out


def _report_text_len(section_map: dict[str, str]) -> int:
    return sum(len(v) for v in section_map.values())


def _make_report_payload(section_map: dict[str, str]) -> dict[str, str]:
    out = _empty_section_dict()
    for key in SECTION_KEYS:
        out[key] = str(section_map.get(key) or "")
    # Preserve extra normalized section keys if present, but keep primary keys first.
    for key, value in section_map.items():
        if key not in out:
            out[key] = str(value or "")
    return out


def _candidate_generated_paths(root: Path, report_id: str, row_index: int | None = None) -> list[Path]:
    candidates: list[Path] = []
    roots = [root]
    if (root / "generated_reports_json").is_dir():
        roots.insert(0, root / "generated_reports_json")
    for base in roots:
        candidates.extend([
            base / f"GENERATED_REPORT_{report_id}.json",
            base / f"{report_id}.json",
        ])
    if row_index is not None:
        row_glob = f"row_{row_index:06d}_{report_id}"
        candidates.extend([
            root / "rows" / row_glob / "celm_generated_report.json",
            root / "rows" / row_glob / "d_section_texts.json",
            root / row_glob / "celm_generated_report.json",
        ])
    return candidates


def _find_generated_path(root: Path, report_id: str, row_index: int | None = None) -> Path | None:
    for path in _candidate_generated_paths(root, report_id, row_index):
        if path.exists():
            return path
    if row_index is not None and (root / "rows").is_dir():
        pattern = f"row_{row_index:06d}_*/celm_generated_report.json"
        matches = sorted((root / "rows").glob(pattern))
        if matches:
            return matches[0]
    return None


def _infer_row_indices_from_ours(ours_root: Path) -> list[int]:
    rows_dir = ours_root / "rows"
    if not rows_dir.is_dir():
        return []
    indices: list[int] = []
    for path in rows_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("row_"):
            continue
        try:
            indices.append(int(path.name.split("_", 2)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(indices))


def _load_cases(
    data_root: Path,
    site: str,
    split_type: str,
    split: str,
    row_indices: list[int],
) -> list[CaseSpec]:
    split_rows = read_split_rows(data_root=data_root, site=site, split=split, split_type=split_type)
    cases: list[CaseSpec] = []
    for row_index in row_indices:
        if row_index < 0 or row_index >= len(split_rows):
            continue
        row = split_rows[row_index]
        report_id = report_id_from_row(row)
        report_json_path = data_root / "matched_eeg_recordings_report" / site / report_id / f"{report_id}.json"
        if not report_json_path.exists():
            continue
        target_sections = [_canonical_section_key(x) for x in target_sections_from_row(row)]
        gt_sections = _section_map_from_gt(_read_json(report_json_path))
        cases.append(
            CaseSpec(
                row_index=row_index,
                report_id=report_id,
                patient_id=str(row.get("BDSPPatientID", "")),
                gt_sections=gt_sections,
                target_sections=target_sections,
            )
        )
    return cases


def _load_row_indices(path: Path | None) -> list[int]:
    if path is None:
        return []
    indices: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            indices.append(int(line))
    return indices


def _build_judge_input(case: CaseSpec, reports: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "case_id": f"row_{case.row_index:06d}_{case.report_id}",
        "ground_truth_report": _make_report_payload(case.gt_sections),
        "generated_reports": [
            {"report_id": blinded_id, "report_sections": _make_report_payload(section_map)}
            for blinded_id, section_map in reports.items()
        ],
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _call_openai_compatible_chat(
    *,
    api_base_url: str,
    api_key: str,
    model: str,
    payload: dict[str, Any],
    timeout_sec: int,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT + "\n" + OUTPUT_SCHEMA_HINT + "\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict clinical EEG report evaluator. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    # Local Qwen/vLLM usually ignores response_format if unsupported, but OpenAI-compatible servers commonly accept it.
    body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(api_base_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            content = response_payload["choices"][0]["message"]["content"]
            result = _extract_json_object(content)
            result["_raw_response_id"] = response_payload.get("id", "")
            return result
        except Exception as exc:  # noqa: BLE001 - keep CLI robust across local servers.
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"LLM judge request failed after {max_retries} attempts: {last_error}")


class LocalTransformersJudge:
    """Small local HF Transformers backend for offline/OpenAI-server-free judging."""

    def __init__(
        self,
        *,
        model: str,
        model_path: str,
        device: str,
        temperature: float,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_ref = model_path or model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = JUDGE_PROMPT + "\n" + OUTPUT_SCHEMA_HINT + "\nInput JSON:\n" + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        messages = [
            {"role": "system", "content": "You are a strict clinical EEG report evaluator. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            text_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
        else:
            generate_kwargs["do_sample"] = False
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        result = _extract_json_object(decoded)
        result["_raw_response_id"] = "local_transformers"
        return result


def _score_map(judge_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in judge_result.get("evaluations") or []:
        if isinstance(item, dict) and item.get("report_id"):
            out[str(item["report_id"])] = item
    return out


def _ranking_winner(judge_result: dict[str, Any]) -> tuple[str | None, bool]:
    ranking = judge_result.get("preference_ranking") or []
    if not ranking:
        return None, False
    first_rank = ranking[0].get("rank", 1) if isinstance(ranking[0], dict) else 1
    top = [r for r in ranking if isinstance(r, dict) and r.get("rank", first_rank) == first_rank]
    ids = [str(r.get("report_id")) for r in top if r.get("report_id")]
    if len(ids) == 1:
        return ids[0], False
    return None, True


def _decide_winner(judge_result: dict[str, Any], label_to_variant: dict[str, str]) -> tuple[str, str]:
    top_label, ranking_tie = _ranking_winner(judge_result)
    if top_label and top_label in label_to_variant:
        return label_to_variant[top_label], "ranking"
    scores = _score_map(judge_result)
    if len(scores) >= 2:
        best_label = None
        best_score = -1.0
        tie = False
        for label in label_to_variant:
            score = float(scores.get(label, {}).get("overall_score") or 0.0)
            if score > best_score:
                best_label = label
                best_score = score
                tie = False
            elif score == best_score:
                tie = True
        if best_label and not tie:
            return label_to_variant[best_label], "overall_score"
    return "tie", "tie" if ranking_tie else "undetermined"


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    ours_root = Path(args.ours_dir)
    celm_root = Path(args.celm_dir)
    output_dir = Path(args.output_dir)
    row_indices = _load_row_indices(Path(args.row_indices_file) if args.row_indices_file else None)
    if not row_indices:
        row_indices = _infer_row_indices_from_ours(ours_root)
    if args.max_cases is not None:
        row_indices = row_indices[: args.max_cases]
    if not row_indices:
        raise RuntimeError("No row indices found. Provide --row-indices-file or an --ours-dir with rows/row_* directories.")

    cases = _load_cases(
        data_root=Path(args.data_root),
        site=args.site,
        split_type=args.split_type,
        split=args.split,
        row_indices=row_indices,
    )
    if not cases:
        raise RuntimeError("No evaluable cases found from selected rows.")

    rng = random.Random(args.seed)
    per_case_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    payload_dir = output_dir / "judge_payloads"
    result_dir = output_dir / "judge_results"

    api_base_url = args.api_base_url.rstrip("/")
    if api_base_url.endswith("/v1"):
        api_base_url = api_base_url + "/chat/completions"
    api_key = os.getenv(args.api_key_env, "") or ("EMPTY" if "localhost" in api_base_url or "127.0.0.1" in api_base_url else "")
    transformers_judge = None
    if args.backend == "transformers" and not args.dry_run:
        transformers_judge = LocalTransformersJudge(
            model=args.model,
            model_path=args.model_path,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    for case in cases:
        ours_path = _find_generated_path(ours_root, case.report_id, case.row_index)
        celm_path = _find_generated_path(celm_root, case.report_id, case.row_index)
        if ours_path is None or celm_path is None:
            failures.append(
                {
                    "row_index": case.row_index,
                    "report_id": case.report_id,
                    "reason": "missing_generated_report",
                    "ours_path": str(ours_path or ""),
                    "celm_path": str(celm_path or ""),
                }
            )
            continue
        ours_sections = _section_map_from_generated(ours_path)
        celm_sections = _section_map_from_generated(celm_path)
        if _report_text_len(ours_sections) == 0 or _report_text_len(celm_sections) == 0:
            failures.append(
                {
                    "row_index": case.row_index,
                    "report_id": case.report_id,
                    "reason": "empty_generated_report",
                    "ours_chars": _report_text_len(ours_sections),
                    "celm_chars": _report_text_len(celm_sections),
                }
            )
            continue

        labels = ["A", "B"]
        rng.shuffle(labels)
        label_to_variant = {labels[0]: "OURS", labels[1]: "CELM"}
        variant_to_label = {variant: label for label, variant in label_to_variant.items()}
        blinded_reports = {
            variant_to_label["OURS"]: ours_sections,
            variant_to_label["CELM"]: celm_sections,
        }
        payload = _build_judge_input(case, blinded_reports)
        payload["_blind_mapping_hidden_from_judge"] = label_to_variant
        payload_path = payload_dir / f"row_{case.row_index:06d}_{case.report_id}.json"
        _write_json(payload_path, payload)
        judge_payload = {k: v for k, v in payload.items() if not k.startswith("_")}
        if args.dry_run:
            judge_result = {
                "case_id": payload["case_id"],
                "evaluations": [],
                "preference_ranking": [],
                "_dry_run": True,
            }
        elif args.backend == "transformers":
            if transformers_judge is None:
                raise RuntimeError("Transformers judge backend was not initialized")
            judge_result = transformers_judge.judge(judge_payload)
        else:
            judge_result = _call_openai_compatible_chat(
                api_base_url=api_base_url,
                api_key=api_key,
                model=args.model,
                payload=judge_payload,
                timeout_sec=args.timeout_sec,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
        judge_result["_blind_mapping"] = label_to_variant
        judge_result["_source_paths"] = {"OURS": str(ours_path), "CELM": str(celm_path)}
        result_path = result_dir / f"row_{case.row_index:06d}_{case.report_id}.json"
        _write_json(result_path, judge_result)
        winner, winner_source = _decide_winner(judge_result, label_to_variant)
        scores = _score_map(judge_result)

        def metric(variant: str, key: str) -> Any:
            label = variant_to_label[variant]
            return scores.get(label, {}).get(key, "")

        row = {
            "row_index": case.row_index,
            "report_id": case.report_id,
            "patient_id": case.patient_id,
            "winner": winner,
            "winner_source": winner_source,
            "ours_label": variant_to_label["OURS"],
            "celm_label": variant_to_label["CELM"],
            "ours_overall": metric("OURS", "overall_score"),
            "celm_overall": metric("CELM", "overall_score"),
            "ours_clinical_correctness": metric("OURS", "clinical_correctness_score"),
            "celm_clinical_correctness": metric("CELM", "clinical_correctness_score"),
            "ours_findings_coverage_attribute_accuracy": metric("OURS", "findings_coverage_attribute_accuracy_score"),
            "celm_findings_coverage_attribute_accuracy": metric("CELM", "findings_coverage_attribute_accuracy_score"),
            "ours_completeness_real_world_readiness": metric("OURS", "completeness_real_world_readiness_score"),
            "celm_completeness_real_world_readiness": metric("CELM", "completeness_real_world_readiness_score"),
            "ours_chars": _report_text_len(ours_sections),
            "celm_chars": _report_text_len(celm_sections),
            "ours_path": str(ours_path),
            "celm_path": str(celm_path),
            "judge_result_path": str(result_path),
            "judge_payload_path": str(payload_path),
        }
        per_case_rows.append(row)
        case_results.append({"case": row, "judge_result": judge_result})

    ours_wins = sum(1 for r in per_case_rows if r.get("winner") == "OURS")
    celm_wins = sum(1 for r in per_case_rows if r.get("winner") == "CELM")
    ties = sum(1 for r in per_case_rows if r.get("winner") == "tie")
    decided = ours_wins + celm_wins + ties
    ours_scores = [float(r["ours_overall"]) for r in per_case_rows if str(r.get("ours_overall", "")) not in {"", "None"}]
    celm_scores = [float(r["celm_overall"]) for r in per_case_rows if str(r.get("celm_overall", "")) not in {"", "None"}]
    summary = {
        "backend": args.backend,
        "model": args.model,
        "model_path": args.model_path,
        "api_base_url": api_base_url,
        "cases_requested": len(row_indices),
        "cases_loaded": len(cases),
        "cases_judged": len(per_case_rows),
        "failures": len(failures),
        "ours_wins": ours_wins,
        "celm_wins": celm_wins,
        "ties": ties,
        "ours_winrate_including_ties_half": (ours_wins + 0.5 * ties) / decided if decided else 0.0,
        "celm_winrate_including_ties_half": (celm_wins + 0.5 * ties) / decided if decided else 0.0,
        "ours_overall_mean": _mean(ours_scores),
        "celm_overall_mean": _mean(celm_scores),
        "ours_dir": str(ours_root),
        "celm_dir": str(celm_root),
        "data_root": str(Path(args.data_root)),
        "site": args.site,
        "split_type": args.split_type,
        "split": args.split,
        "dry_run": bool(args.dry_run),
    }
    _write_csv(output_dir / "per_case_judge_results.csv", per_case_rows)
    _write_csv(output_dir / "failures.csv", failures)
    _write_json(output_dir / "case_results.json", case_results)
    _write_json(output_dir / "summary.json", summary)
    md = [
        "# LLM-as-Judge Winrate: OURS vs CELM",
        "",
        f"- model: `{args.model}`",
        f"- judged cases: {summary['cases_judged']}",
        f"- failures: {summary['failures']}",
        f"- OURS wins: {ours_wins}",
        f"- CELM wins: {celm_wins}",
        f"- ties: {ties}",
        f"- OURS winrate, tie=0.5: {summary['ours_winrate_including_ties_half']:.3f}",
        f"- CELM winrate, tie=0.5: {summary['celm_winrate_including_ties_half']:.3f}",
        f"- OURS mean overall score: {summary['ours_overall_mean']:.3f}",
        f"- CELM mean overall score: {summary['celm_overall_mean']:.3f}",
        "",
        "## Notes",
        "",
        "- GT report text is used only for evaluation.",
        "- Report IDs are blinded as A/B before judging; mapping is saved only in output artifacts.",
        "- This is an LLM-as-a-judge evaluation, not a replacement for claim-level clinical audit.",
    ]
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blinded LLM-as-judge winrate between latest OURS and CELM EEG reports.")
    parser.add_argument("--data-root", default="/exHDD_8T/hjlee_data/eeg_data/celm_s_sites_pipeline")
    parser.add_argument("--site", default="S0001")
    parser.add_argument("--split-type", default="random_split_data_by_patient")
    parser.add_argument("--split", default="test")
    parser.add_argument("--row-indices-file", default="")
    parser.add_argument(
        "--ours-dir",
        default="artifacts/stage3f_spatiomorph_v2_2_selected50_20260626",
        help="OUR latest run root with rows/row_* dirs or generated_reports_json.",
    )
    parser.add_argument(
        "--celm-dir",
        default="artifacts/celm_style_eval_selected50_ToolV2LLM_20260611/CELM_Upstream/generated_reports_json",
        help="CELM generated_reports_json dir or run root.",
    )
    parser.add_argument("--output-dir", default="artifacts/llm_judge_winrate_latest_vs_celm_20260626")
    parser.add_argument("--backend", choices=["openai_chat", "transformers"], default=os.getenv("LLM_JUDGE_BACKEND", "openai_chat"))
    parser.add_argument("--model", default=os.getenv("LLM_JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--model-path", default=os.getenv("LLM_JUDGE_MODEL_PATH", ""))
    parser.add_argument("--device", default=os.getenv("LLM_JUDGE_DEVICE", "auto"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LLM_JUDGE_MAX_NEW_TOKENS", "2048")))
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("LLM_JUDGE_BASE_URL", "http://localhost:8000/v1/chat/completions"),
        help="OpenAI-compatible chat completions URL. If ending in /v1, /chat/completions is appended.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only build judge payloads and manifest; do not call LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
