from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from eeg_report_multiagent.evaluation.metrics import precision_recall_f1


TOKEN_RE = re.compile(r"[a-z0-9]+")

CONCEPT_PATTERNS: Dict[str, Dict[str, Sequence[str]]] = {
    "state": {
        "awake": (r"\bawake\b",),
        "drowsy": (r"\bdrows",),
        "sleep": (r"\bsleep|asleep|spindle|k-complex|vertex",),
    },
    "protocol": {
        "photic": (r"\bphotic\b",),
        "photic_no_response": (r"photic[^.]{0,80}(no|not|did not|without)[^.]{0,80}(response|driving)",),
        "hyperventilation": (r"\bhyperventilation\b",),
        "hyperventilation_not_performed": (r"hyperventilation[^.]{0,50}not performed",),
        "ekg": (r"\bekg\b|ecg",),
        "video": (r"\bvideo\b",),
        "comparison": (r"\bcomparison\b|compared|prior|previous",),
    },
    "background": {
        "posterior_dominant_rhythm": (r"posterior dominant rhythm|\bpdr\b",),
        "slowing": (r"\bslowing\b|theta/delta|delta/theta",),
        "reactivity": (r"\breactiv",),
        "amplitude_asymmetry": (r"amplitude[^.]{0,80}(higher|asym|left|right)",),
        "excess_beta": (r"excess beta|beta excess",),
    },
    "event": {
        "epileptiform": (r"epileptiform|spike|sharp",),
        "spike_wave": (r"spike[- ]?wave|spike and wave",),
        "multifocal": (r"multifocal",),
        "seizure_absent": (r"no[^.]{0,80}(seizure|organized|evolving)",),
        "seizure": (r"\bseizure",),
        "runs": (r"\bruns?\b|continuous",),
    },
    "laterality_location": {
        "left": (r"\bleft\b",),
        "right": (r"\bright\b",),
        "bifrontal": (r"\bbifrontal\b",),
        "frontal": (r"\bfrontal\b",),
        "temporal": (r"\btemporal\b",),
        "posterior": (r"\bposterior\b",),
        "occipital": (r"\boccipital\b",),
        "hemisphere": (r"\bhemisphere\b",),
    },
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_gt_sections(gt_json: Dict[str, Any]) -> Dict[str, str]:
    note_text = str(gt_json.get("note_text") or "")
    detail_parts: List[str] = []
    eeg_extractions = gt_json.get("EEG_section_llm_extractions") or {}
    for section in eeg_extractions.get("EEG_sections") or []:
        if str(section.get("section_name", "")).strip().lower().startswith("detail"):
            detail_parts.append(str(section.get("section_text") or ""))

    lowered = note_text.lower()
    cut_points = [
        idx
        for marker in (" comparison:", " indication:", " method:", " detail:")
        if (idx := lowered.find(marker)) >= 0
    ]
    impression = note_text[: min(cut_points)] if cut_points else note_text

    return {
        "note_text": note_text,
        "detail": "\n".join(detail_parts).strip(),
        "impression": impression.strip(),
    }


def extract_numbers(text: str) -> Dict[str, List[str]]:
    norm = normalize_text(text)
    return {
        "frequency_hz": sorted(set(re.findall(r"\b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*hz\b", norm))),
        "amplitude_uv": sorted(set(re.findall(r"\b\d+(?:\.\d+)?\s*u?v\b", norm))),
        "duration_sec": sorted(set(re.findall(r"\b\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b", norm))),
    }


def extract_concepts(text: str) -> Dict[str, List[str]]:
    norm = normalize_text(text)
    out: Dict[str, List[str]] = {}
    for category, patterns in CONCEPT_PATTERNS.items():
        hits = []
        for concept, regexes in patterns.items():
            if any(re.search(regex, norm) for regex in regexes):
                hits.append(concept)
        out[category] = sorted(hits)
    return out


def _flatten(items: Dict[str, Iterable[str]]) -> set[str]:
    return {f"{category}:{value}" for category, values in items.items() for value in values}


def compare_text_concepts(gt_text: str, generated_text: str) -> Dict[str, Any]:
    gt_concepts = extract_concepts(gt_text)
    pred_concepts = extract_concepts(generated_text)
    gt_numbers = extract_numbers(gt_text)
    pred_numbers = extract_numbers(generated_text)

    gt_flat = _flatten(gt_concepts)
    pred_flat = _flatten(pred_concepts)
    matched = sorted(gt_flat & pred_flat)
    missing = sorted(gt_flat - pred_flat)
    extra = sorted(pred_flat - gt_flat)

    metrics = precision_recall_f1(tp=len(matched), fp=len(extra), fn=len(missing))
    return {
        "metrics": metrics,
        "matched_concepts": matched,
        "missing_concepts": missing,
        "extra_concepts": extra,
        "gt_concepts": gt_concepts,
        "generated_concepts": pred_concepts,
        "gt_numbers": gt_numbers,
        "generated_numbers": pred_numbers,
        "numeric_overlap": {
            key: sorted(set(gt_numbers[key]) & set(pred_numbers[key]))
            for key in gt_numbers
        },
        "numeric_missing": {
            key: sorted(set(gt_numbers[key]) - set(pred_numbers[key]))
            for key in gt_numbers
        },
        "numeric_extra": {
            key: sorted(set(pred_numbers[key]) - set(gt_numbers[key]))
            for key in gt_numbers
        },
    }


def compare_generated_report_to_gt(
    gt_report_json_path: Path,
    generated_detail_path: Path,
    generated_impression_path: Path,
) -> Dict[str, Any]:
    gt = json.loads(gt_report_json_path.read_text(encoding="utf-8"))
    gt_sections = extract_gt_sections(gt)
    generated_detail = generated_detail_path.read_text(encoding="utf-8") if generated_detail_path.exists() else ""
    generated_impression = generated_impression_path.read_text(encoding="utf-8") if generated_impression_path.exists() else ""

    return {
        "input_contract": "local lexical/concept comparison only; no external API judge is used",
        "gt_report_json_path": str(gt_report_json_path),
        "generated_detail_path": str(generated_detail_path),
        "generated_impression_path": str(generated_impression_path),
        "detail": compare_text_concepts(gt_sections["detail"], generated_detail),
        "impression": compare_text_concepts(gt_sections["impression"], generated_impression),
        "gt_section_lengths": {k: len(v) for k, v in gt_sections.items()},
        "generated_lengths": {
            "detail": len(generated_detail),
            "impression": len(generated_impression),
        },
    }
