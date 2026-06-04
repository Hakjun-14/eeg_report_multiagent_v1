from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from eeg_report_multiagent.evaluation.report_text_comparison import (
    compare_text_concepts,
    extract_concepts,
    extract_numbers,
    normalize_text,
)

DEFAULT_VARIANTS = ["CELM", "Our_B", "Our_D", "Our_B_QFv2", "Our_Upgrade_LLMProp"]

CONCEPT_TO_SLOT = {
    "state:awake": ("awake_state", "state_sleep"),
    "state:drowsy": ("drowsiness", "state_sleep"),
    "state:sleep": ("sleep_architecture", "state_sleep"),
    "protocol:photic": ("photic_status", "activation_protocols"),
    "protocol:photic_no_response": ("photic_driving_response", "activation_protocols"),
    "protocol:hyperventilation": ("hyperventilation_status", "activation_protocols"),
    "protocol:hyperventilation_not_performed": ("hyperventilation_status", "activation_protocols"),
    "protocol:ekg": ("ekg_availability", "recording_context"),
    "protocol:video": ("video_availability", "recording_context"),
    "protocol:comparison": ("comparison_history", "recording_context"),
    "background:posterior_dominant_rhythm": ("pdr_frequency", "background_activity"),
    "background:slowing": ("background_slowing", "background_activity"),
    "background:reactivity": ("pdr_reactivity", "background_activity"),
    "background:amplitude_asymmetry": ("background_amplitude_asymmetry", "background_activity"),
    "background:excess_beta": ("excess_beta", "background_activity"),
    "event:epileptiform": ("epileptiform_morphology", "epileptiform_interictal"),
    "event:spike_wave": ("epileptiform_morphology", "epileptiform_interictal"),
    "event:multifocal": ("localization_laterality", "epileptiform_interictal"),
    "event:runs": ("burst_duration", "epileptiform_interictal"),
    "event:seizure_absent": ("seizure_absence", "events_seizures"),
    "event:seizure": ("electrographic_seizure_present", "events_seizures"),
    "laterality_location:left": ("localization_laterality", "epileptiform_interictal"),
    "laterality_location:right": ("localization_laterality", "epileptiform_interictal"),
    "laterality_location:bifrontal": ("localization_laterality", "epileptiform_interictal"),
    "laterality_location:frontal": ("localization_laterality", "epileptiform_interictal"),
    "laterality_location:temporal": ("localization_laterality", "epileptiform_interictal"),
    "laterality_location:posterior": ("localization_laterality", "epileptiform_interictal"),
    "laterality_location:occipital": ("localization_laterality", "epileptiform_interictal"),
    "laterality_location:hemisphere": ("localization_laterality", "epileptiform_interictal"),
}

SLOT_SEVERITY_DEFAULTS = {
    "pdr_frequency": "major",
    "pdr_reactivity": "moderate",
    "background_slowing": "major",
    "excess_beta": "moderate",
    "awake_state": "moderate",
    "drowsiness": "moderate",
    "sleep_architecture": "major",
    "epileptiform_morphology": "critical",
    "burst_duration": "major",
    "localization_laterality": "major",
    "electrographic_seizure_present": "critical",
    "seizure_absence": "critical",
    "photic_status": "moderate",
    "photic_driving_response": "moderate",
    "hyperventilation_status": "moderate",
    "hyperventilation_effect": "major",
    "ekg_availability": "minor",
    "video_availability": "minor",
    "comparison_history": "minor",
    "numeric_quantitation": "major",
    "debug_surface_separation": "moderate",
    "leakage_audit": "critical",
}

CLINICAL_RULES = {
    "pdr_frequency": "A PDR claim requires posterior/occipital alpha-range activity and state/reactivity context when available; global dominant frequency alone is insufficient.",
    "epileptiform_morphology": "A definite epileptiform claim requires morphology-specific evidence and spatial field/localization support, not candidate burden alone.",
    "electrographic_seizure_present": "A seizure claim requires seizure-specific evolution/duration and/or event-level correlate; transient candidates alone are insufficient.",
    "seizure_absence": "A no-seizure claim should be scoped to available reviewed evidence and should not include candidate transient burden as seizure evidence.",
    "photic_status": "Photic claims require protocol provenance and response status when surfaced.",
    "photic_driving_response": "Photic driving/no-response claims require photic stimulation provenance and response evidence.",
    "hyperventilation_status": "Hyperventilation claims require protocol provenance.",
    "hyperventilation_effect": "Hyperventilation effect claims require hyperventilation performance plus effect evidence.",
    "numeric_quantitation": "Numeric values should be tied to supported clinical claims and patient-specific provenance; internal proxy values should remain debug-only.",
    "debug_surface_separation": "Internal detector scores, likelihoods, ratios, and candidate burden values should not appear as clinical prose unless translated into supported clinical claims.",
}

DEBUG_LEAKAGE_PATTERNS = [
    r"\bcandidate transient burden\b",
    r"\blikelihood score\b",
    r"\bsupport score\b",
    r"\bfield concentration\b",
    r"\bslowing index\b",
    r"\bbeta ratio\b",
    r"\bposterior organization proxy\b",
    r"\bproxy score\b",
    r"\binternal\b[^.]{0,60}\bscore\b",
]

EVIDENCE_KEYWORDS_BY_SLOT = {
    "pdr_frequency": ["background_pdr_frequency", "pdr_candidate_frequency", "background_pdr_support"],
    "pdr_reactivity": ["background_reactivity"],
    "background_slowing": ["background_slowing", "slowing_score", "background_frequency"],
    "excess_beta": ["excess_beta", "beta_excess"],
    "sleep_architecture": ["sleep_architecture"],
    "epileptiform_morphology": ["epileptiform", "morphology", "spike", "sharp"],
    "burst_duration": ["train_duration", "burst", "duration"],
    "localization_laterality": ["laterality", "localization", "focality", "bifrontal", "spread"],
    "electrographic_seizure_present": ["seizure_likelihood", "electrographic_seizure"],
    "seizure_absence": ["seizure_likelihood", "electrographic_seizure"],
    "photic_status": ["photic"],
    "photic_driving_response": ["photic"],
    "hyperventilation_status": ["hyperventilation"],
    "hyperventilation_effect": ["hyperventilation"],
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def flatten_concepts(text: str) -> List[str]:
    concepts = extract_concepts(text)
    return sorted(f"{category}:{value}" for category, values in concepts.items() for value in values)


def flatten_numeric(numbers: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    return sorted((kind, value) for kind, values in numbers.items() for value in values)


def slot_for_concept(concept: str) -> Tuple[str, str]:
    return CONCEPT_TO_SLOT.get(concept, (concept.replace(":", "_"), "unknown"))


def severity_for_slot(slot: str, slot_schema: Dict[str, Any]) -> str:
    for group in (slot_schema.get("slot_groups") or {}).values():
        slots = group.get("slots") if isinstance(group, dict) else None
        if isinstance(slots, dict) and slot in slots:
            return str(slots[slot].get("severity_if_wrong") or SLOT_SEVERITY_DEFAULTS.get(slot, "moderate"))
    return SLOT_SEVERITY_DEFAULTS.get(slot, "moderate")


def rule_for_slot(slot: str) -> str:
    return CLINICAL_RULES.get(slot, f"Clinical criteria are required to support the `{slot}` claim.")


def reference_status_for(decision: str) -> str:
    if decision in {"supported_present", "supported_absent", "over_cautious_false_negative", "under_specified"}:
        return "reference_consistent"
    if decision == "contradicted":
        return "reference_contradicted"
    if decision == "unsupported":
        return "reference_missing"
    return "ambiguous"


def _compact_time(time_obj: Dict[str, Any]) -> str:
    if not isinstance(time_obj, dict):
        return "unknown"
    windows = time_obj.get("window_indices") or []
    if isinstance(windows, list) and windows:
        if len(windows) <= 6:
            return f"windows={windows}"
        return f"windows={windows[0]}..{windows[-1]} (n={len(windows)})"
    if time_obj.get("start_sec") is not None or time_obj.get("end_sec") is not None:
        return f"{time_obj.get('start_sec')}..{time_obj.get('end_sec')} sec"
    return "unknown"


def _compact_measurement(quant: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(quant, dict):
        return {}
    unit = str(quant.get("unit") or "")
    out: Dict[str, str] = {}
    exact = quant.get("exact")
    lower = quant.get("lower")
    upper = quant.get("upper")
    if exact is not None:
        key = "frequency_hz" if unit.lower() == "hz" else "confidence" if unit.lower() == "score" else "value"
        out[key] = f"{exact} {unit}".strip()
    elif lower is not None or upper is not None:
        out["value"] = f"{lower}..{upper} {unit}".strip()
    return out


def extract_signal_provenance(evidence_board: Dict[str, Any], slot: str, limit: int = 3) -> List[Dict[str, Any]]:
    if not evidence_board:
        return []
    keywords = EVIDENCE_KEYWORDS_BY_SLOT.get(slot, [slot])
    rows: List[Dict[str, Any]] = []
    for item_type, items in (("measurement", evidence_board.get("measurements") or []),):
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            haystack = " ".join(
                str(item.get(key) or "")
                for key in ("measurement_id", "measurement_name", "assertion")
            ).lower()
            if not any(keyword.lower() in haystack for keyword in keywords):
                continue
            prov = item.get("provenance")
            if isinstance(prov, dict):
                provenance_rows = [prov]
            elif isinstance(prov, list):
                provenance_rows = [p for p in prov if isinstance(p, dict)]
            else:
                provenance_rows = [{}]
            quant = item.get("quantitation") if isinstance(item.get("quantitation"), dict) else {}
            for p in provenance_rows[:1]:
                space = p.get("space") if isinstance(p.get("space"), dict) else {}
                measurement = _compact_measurement(quant)
                rows.append(
                    {
                        "recording_id": p.get("source_ref") or evidence_board.get("session_id") or "unknown",
                        "time_window": _compact_time(p.get("time") if isinstance(p.get("time"), dict) else {}),
                        "channels_or_regions": space.get("channels") or [space.get("region") or space.get("laterality") or "unknown"],
                        "state": "unknown",
                        "protocol": "unknown",
                        "measurement": measurement,
                        "evidence_type": "direct" if item_type == "measurement" else "derived",
                        "source_object": item.get("measurement_id") or item.get("evidence_id") or "unknown",
                    }
                )
            if len(rows) >= limit:
                return rows[:limit]
    return rows[:limit]


def source_level_for_variant(variant: str, evidence_board: Dict[str, Any], slot: str) -> str:
    if variant == "CELM":
        return "generated_text_only"
    signal = extract_signal_provenance(evidence_board, slot, limit=1)
    if signal:
        return signal[0].get("evidence_type", "signal_derived")
    return "generated_text_only"


def make_claim_card(
    *,
    case_id: str,
    model: str,
    section: str,
    claim: str,
    claim_type: str,
    slot: str,
    severity: str,
    decision: str,
    evidence_board: Dict[str, Any],
    recommended_action: str,
    recommended_revision: str,
    negative_reason: str = "",
    debug_feature: str = "",
    debug_value: str = "",
) -> Dict[str, Any]:
    signal_prov = extract_signal_provenance(evidence_board, slot)
    negative = []
    if negative_reason:
        negative.append({"blocked_claim": claim, "reason": negative_reason})
    debug = []
    if debug_feature:
        debug.append(
            {
                "feature": debug_feature,
                "value": debug_value,
                "reason_not_surface_text": "Internal detector/proxy/debug values are not clinical report prose without claim-level support.",
            }
        )
    return {
        "case_id": case_id,
        "model": model,
        "section": section,
        "claim": claim,
        "claim_type": claim_type,
        "slot": slot,
        "severity": severity,
        "decision": decision,
        "reference_status": reference_status_for(decision),
        "required_clinical_knowledge": [
            {
                "rule": rule_for_slot(slot),
                "source_status": "assumed_clinical_knowledge",
            }
        ],
        "patient_signal_provenance": signal_prov,
        "provenance_level": "signal_derived" if signal_prov else ("generated_text_only" if model == "CELM" else "absent"),
        "negative_provenance": negative,
        "debug_only_evidence": debug,
        "recommended_action": recommended_action,
        "recommended_revision": recommended_revision,
    }


def card_for_missing_concept(case_id: str, variant: str, section: str, concept: str, evidence_board: Dict[str, Any], slot_schema: Dict[str, Any]) -> Dict[str, Any]:
    slot, _ = slot_for_concept(concept)
    return make_claim_card(
        case_id=case_id,
        model=variant,
        section=section,
        claim=f"Reference concept not generated: {concept}",
        claim_type="missed_reference_concept",
        slot=slot,
        severity=severity_for_slot(slot, slot_schema),
        decision="over_cautious_false_negative",
        evidence_board=evidence_board,
        recommended_action="human_adjudication" if not evidence_board else "revise",
        recommended_revision=f"Add `{concept}` only if patient-specific evidence supports the `{slot}` slot; otherwise keep as evaluation-only reference miss.",
        negative_reason="The generated report omitted a reference concept; this is a recall failure, not a generation-time license to copy GT wording.",
    )


def card_for_extra_concept(case_id: str, variant: str, section: str, concept: str, evidence_board: Dict[str, Any], slot_schema: Dict[str, Any]) -> Dict[str, Any]:
    slot, _ = slot_for_concept(concept)
    signal = extract_signal_provenance(evidence_board, slot, limit=1)
    decision = "under_specified" if signal else "unsupported"
    action = "caveat" if signal else "block"
    if concept == "event:seizure":
        decision = "unsupported" if not signal else "under_specified"
        action = "block" if not signal else "caveat"
    return make_claim_card(
        case_id=case_id,
        model=variant,
        section=section,
        claim=f"Generated extra concept not found in reference: {concept}",
        claim_type="extra_generated_concept",
        slot=slot,
        severity=severity_for_slot(slot, slot_schema),
        decision=decision,
        evidence_board=evidence_board,
        recommended_action=action,
        recommended_revision=f"Surface `{concept}` only with explicit claim-level provenance for `{slot}`; otherwise remove or move to audit output.",
        negative_reason="The concept is present in generated text but absent from the reference comparison for the target section.",
    )


def card_for_numeric_missing(case_id: str, variant: str, section: str, kind: str, value: str, evidence_board: Dict[str, Any], slot_schema: Dict[str, Any]) -> Dict[str, Any]:
    slot = "numeric_quantitation"
    return make_claim_card(
        case_id=case_id,
        model=variant,
        section=section,
        claim=f"Reference numeric value not generated: {kind}={value}",
        claim_type="numeric_missing",
        slot=slot,
        severity=severity_for_slot(slot, slot_schema),
        decision="under_specified",
        evidence_board=evidence_board,
        recommended_action="revise" if evidence_board else "human_adjudication",
        recommended_revision="Only add numeric quantitation if linked to a supported clinical claim and patient-specific measurement provenance.",
        negative_reason="Numeric values should not be copied from GT; they require signal/metadata provenance for generation.",
    )


def card_for_numeric_extra(case_id: str, variant: str, section: str, kind: str, value: str, evidence_board: Dict[str, Any], slot_schema: Dict[str, Any]) -> Dict[str, Any]:
    slot = "numeric_quantitation"
    suspicious = bool(re.search(r"\b(?:0\.5\s*hz|\d{2,}\s*(?:s|sec|seconds))\b", value.lower()))
    return make_claim_card(
        case_id=case_id,
        model=variant,
        section=section,
        claim=f"Generated numeric value absent from reference: {kind}={value}",
        claim_type="numeric_extra",
        slot=slot,
        severity=severity_for_slot(slot, slot_schema),
        decision="debug_leakage" if suspicious else "unsupported",
        evidence_board=evidence_board,
        recommended_action="move_to_debug_only" if suspicious else "block",
        recommended_revision="Remove unsupported numeric values from clinical prose unless linked to a supported clinical claim with provenance.",
        negative_reason="Generated numeric value is absent from the reference comparison and may reflect proxy/debug evidence rather than clinical quantitation.",
        debug_feature="numeric_proxy_or_internal_value" if suspicious else "",
        debug_value=value if suspicious else "",
    )




def card_for_possible_leakage(case_id: str, variant: str, evidence_board: Dict[str, Any], rouge_l: float, meteor: float) -> Dict[str, Any]:
    return make_claim_card(
        case_id=case_id,
        model=variant,
        section="evaluation_integrity",
        claim=f"Generated report nearly exactly matches the reference text (ROUGE-L={rouge_l:.3f}, METEOR={meteor:.3f}).",
        claim_type="possible_leakage_or_memorization",
        slot="leakage_audit",
        severity="critical",
        decision="possible_leakage_or_memorization",
        evidence_board=evidence_board,
        recommended_action="human_adjudication",
        recommended_revision="Run leakage audit before interpreting this as model quality: check same-patient overlap, duplicate report family, masked token preservation, train/test contamination, and input-side target leakage.",
        negative_reason="Near-exact reference reproduction can reflect legitimate success, template duplication, or leakage; it cannot be accepted without an explicit audit.",
    )

def detect_debug_leakage_cards(case_id: str, variant: str, section: str, text: str, evidence_board: Dict[str, Any], slot_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for pattern in DEBUG_LEAKAGE_PATTERNS:
        for match in re.finditer(pattern, normalize_text(text)):
            feature = match.group(0)
            cards.append(
                make_claim_card(
                    case_id=case_id,
                    model=variant,
                    section=section,
                    claim=f"Debug/proxy phrase exposed in clinical prose: {feature}",
                    claim_type="debug_surface_leakage",
                    slot="debug_surface_separation",
                    severity=severity_for_slot("debug_surface_separation", slot_schema),
                    decision="debug_leakage",
                    evidence_board=evidence_board,
                    recommended_action="move_to_debug_only",
                    recommended_revision="Move internal scores/proxy phrases to provenance JSON; clinical prose should use supported clinical terms only.",
                    negative_reason="Internal detector/proxy values should not surface as final clinical report text.",
                    debug_feature=feature,
                    debug_value=feature,
                )
            )
    return cards


def load_evidence_board_for_variant(
    variant: str,
    row_index: int,
    report_id: str,
    artifact_roots: Mapping[str, Path],
) -> Dict[str, Any]:
    root = artifact_roots.get(variant)
    if not root:
        return {}
    row_dir = root / f"row_{row_index:06d}_{report_id}"
    path = row_dir / "evidence_board.json"
    if path.exists():
        return read_json(path)
    return {}


def audit_case(
    case_payload: Dict[str, Any],
    *,
    slot_schema: Dict[str, Any],
    failure_taxonomy: Dict[str, Any],
    claim_gate_policy: Dict[str, Any],
    artifact_roots: Mapping[str, Path],
    variants: Sequence[str] = DEFAULT_VARIANTS,
) -> Dict[str, Any]:
    row_index = int(case_payload["row_index"])
    report_id = str(case_payload["report_id"])
    case_id = f"row_{row_index:06d}_{report_id}"
    target_sections = [str(x) for x in case_payload.get("target_sections") or []]
    gt_sections = case_payload.get("gt_sections") if isinstance(case_payload.get("gt_sections"), dict) else {}
    variant_payloads = case_payload.get("variants") if isinstance(case_payload.get("variants"), dict) else {}

    reference_concepts_by_section = {sec: flatten_concepts(str(gt_sections.get(sec, ""))) for sec in target_sections}
    reference_numbers_by_section = {sec: extract_numbers(str(gt_sections.get(sec, ""))) for sec in target_sections}

    claim_cards: List[Dict[str, Any]] = []
    critical_slot_table: Dict[str, Dict[str, Any]] = {}

    for section in target_sections:
        for concept in reference_concepts_by_section.get(section, []):
            slot, group = slot_for_concept(concept)
            row = critical_slot_table.setdefault(
                f"{section}:{slot}",
                {
                    "section": section,
                    "slot": slot,
                    "slot_group": group,
                    "ground_truth_reference": concept,
                    "severity": severity_for_slot(slot, slot_schema),
                    "clinical_impact": f"{severity_for_slot(slot, slot_schema)} reference slot; requires symmetric model assessment.",
                    "provenance_issue": "Reference text only until checked against EvidenceBoard or human adjudication.",
                },
            )
            row.setdefault("reference_concepts", []).append(concept)
        for kind, value in flatten_numeric(reference_numbers_by_section.get(section, {})):
            row = critical_slot_table.setdefault(
                f"{section}:numeric_quantitation",
                {
                    "section": section,
                    "slot": "numeric_quantitation",
                    "slot_group": "numeric_quantitation",
                    "ground_truth_reference": "",
                    "severity": severity_for_slot("numeric_quantitation", slot_schema),
                    "clinical_impact": "Numeric quantitation can alter clinical specificity and report fidelity.",
                    "provenance_issue": "Reference numeric values require patient-specific measurement provenance before generation.",
                },
            )
            row.setdefault("reference_numbers", []).append(f"{kind}={value}")

    variant_summaries: Dict[str, Any] = {}
    for variant in variants:
        vp = variant_payloads.get(variant)
        if not isinstance(vp, dict):
            continue
        evidence_board = load_evidence_board_for_variant(variant, row_index, report_id, artifact_roots)
        section_comparisons = vp.get("section_comparisons") or []
        aggregate = vp.get("aggregate") if isinstance(vp.get("aggregate"), dict) else {}
        variant_cards: List[Dict[str, Any]] = []
        for comp in section_comparisons:
            if not isinstance(comp, dict):
                continue
            section = str(comp.get("section_name") or "")
            generated_text = str(comp.get("generated_text") or "")
            for concept in comp.get("missing_concepts") or []:
                card = card_for_missing_concept(case_id, variant, section, str(concept), evidence_board, slot_schema)
                claim_cards.append(card)
                variant_cards.append(card)
                slot, _ = slot_for_concept(str(concept))
                key = f"{section}:{slot}"
                if key in critical_slot_table:
                    critical_slot_table[key][variant] = "missing_reference_concept"
            for concept in comp.get("extra_concepts") or []:
                card = card_for_extra_concept(case_id, variant, section, str(concept), evidence_board, slot_schema)
                claim_cards.append(card)
                variant_cards.append(card)
                slot, _ = slot_for_concept(str(concept))
                key = f"{section}:{slot}"
                row = critical_slot_table.setdefault(
                    key,
                    {
                        "section": section,
                        "slot": slot,
                        "slot_group": slot_for_concept(str(concept))[1],
                        "ground_truth_reference": "not_detected_in_reference_text",
                        "severity": severity_for_slot(slot, slot_schema),
                        "clinical_impact": f"Generated extra `{concept}` may be unsupported or require adjudication.",
                        "provenance_issue": "Generated text requires claim-level evidence support.",
                    },
                )
                row[variant] = "extra_generated_concept"
            for kind, values in (comp.get("numeric_missing") or {}).items():
                for value in values or []:
                    card = card_for_numeric_missing(case_id, variant, section, str(kind), str(value), evidence_board, slot_schema)
                    claim_cards.append(card)
                    variant_cards.append(card)
                    key = f"{section}:numeric_quantitation"
                    if key in critical_slot_table:
                        critical_slot_table[key][variant] = "missing_numeric"
            for kind, values in (comp.get("numeric_extra") or {}).items():
                for value in values or []:
                    card = card_for_numeric_extra(case_id, variant, section, str(kind), str(value), evidence_board, slot_schema)
                    claim_cards.append(card)
                    variant_cards.append(card)
                    key = f"{section}:numeric_quantitation"
                    critical_slot_table.setdefault(
                        key,
                        {
                            "section": section,
                            "slot": "numeric_quantitation",
                            "slot_group": "numeric_quantitation",
                            "ground_truth_reference": "not_detected_in_reference_text",
                            "severity": severity_for_slot("numeric_quantitation", slot_schema),
                            "clinical_impact": "Unsupported numeric values can imply false quantitation.",
                            "provenance_issue": "Generated numeric value needs measurement provenance.",
                        },
                    )[variant] = "extra_numeric"
            for card in detect_debug_leakage_cards(case_id, variant, section, generated_text, evidence_board, slot_schema):
                claim_cards.append(card)
                variant_cards.append(card)
                key = f"{section}:debug_surface_separation"
                critical_slot_table.setdefault(
                    key,
                    {
                        "section": section,
                        "slot": "debug_surface_separation",
                        "slot_group": "report_integrity",
                        "ground_truth_reference": "not_applicable",
                        "severity": severity_for_slot("debug_surface_separation", slot_schema),
                        "clinical_impact": "Debug values in report text reduce clinical usability and may mislead readers.",
                        "provenance_issue": "Debug evidence must stay in provenance structures.",
                    },
                )[variant] = "debug_leakage"

        for row in critical_slot_table.values():
            if variant not in row:
                section = str(row.get("section") or "")
                slot = str(row.get("slot") or "")
                ref_concepts = set(row.get("reference_concepts") or [])
                if ref_concepts:
                    generated_concepts = set()
                    for comp in section_comparisons:
                        if str(comp.get("section_name") or "") != section:
                            continue
                        generated_text = str(comp.get("generated_text") or "")
                        generated_concepts.update(flatten_concepts(generated_text))
                    row[variant] = "matched_or_not_applicable" if ref_concepts & generated_concepts else row.get(variant, "not_assessed")
        rouge_l = float(aggregate.get("rougeL") or 0.0)
        meteor = float(aggregate.get("meteor") or 0.0)
        if rouge_l >= 0.95 and meteor >= 0.95:
            leakage_card = card_for_possible_leakage(case_id, variant, evidence_board, rouge_l, meteor)
            claim_cards.append(leakage_card)
            variant_cards.append(leakage_card)
            key = "evaluation_integrity:leakage_audit"
            critical_slot_table.setdefault(
                key,
                {
                    "section": "evaluation_integrity",
                    "slot": "leakage_audit",
                    "slot_group": "evaluation_integrity",
                    "ground_truth_reference": "near_exact_generated_reference_match",
                    "severity": "critical",
                    "clinical_impact": "A near-exact match may be correct, duplicated, or leaked; metric interpretation requires audit.",
                    "provenance_issue": "Evaluate split integrity, duplicate report family, and input-side target leakage before counting as clinical success.",
                },
            )[variant] = "possible_leakage_or_memorization"

        decision_counts = Counter(card["decision"] for card in variant_cards)
        severity_counts = Counter(card["severity"] for card in variant_cards)
        variant_summaries[variant] = {
            "card_count": len(variant_cards),
            "decision_counts": dict(decision_counts),
            "severity_counts": dict(severity_counts),
            "metric_context": {
                "concept_f1_mean": aggregate.get("concept_f1_mean"),
                "rougeL": aggregate.get("rougeL"),
                "meteor": aggregate.get("meteor"),
                "bertscore_f1": aggregate.get("bertscore_f1"),
            },
            "evidence_board_available": bool(evidence_board),
        }

    slot_rows = []
    for row in critical_slot_table.values():
        if isinstance(row.get("reference_concepts"), list):
            row["reference_concepts"] = "|".join(sorted(set(row["reference_concepts"])))
        if isinstance(row.get("reference_numbers"), list):
            row["reference_numbers"] = "|".join(sorted(set(row["reference_numbers"])))
        slot_rows.append(row)

    return {
        "case_id": case_id,
        "row_index": row_index,
        "report_id": report_id,
        "patient_id": case_payload.get("patient_id"),
        "input_contract": "GT/reference text is evaluation-only. Audit labels are first-pass annotations and require human adjudication for clinical claims.",
        "target_sections": target_sections,
        "critical_slot_table": sorted(slot_rows, key=lambda r: (str(r.get("section")), str(r.get("slot")))),
        "claim_cards": claim_cards,
        "variant_summaries": variant_summaries,
        "taxonomy_version": failure_taxonomy.get("version"),
        "claim_gate_policy_version": claim_gate_policy.get("version"),
    }


def render_case_audit_markdown(audit: Dict[str, Any], variants: Sequence[str] = DEFAULT_VARIANTS, max_cards_per_variant: int = 12) -> str:
    lines: List[str] = []
    lines.append(f"# Clinical Provenance Audit: {audit['case_id']}")
    lines.append("")
    lines.append(audit["input_contract"])
    lines.append("")
    lines.append(f"Patient: `{audit.get('patient_id')}`")
    lines.append(f"Target sections: {', '.join(audit.get('target_sections') or [])}")
    lines.append("")
    lines.append("## Variant Summary")
    lines.append("| Variant | evidence board | cards | top decisions | critical/major cards | ROUGE-L | METEOR |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for variant in variants:
        summary = (audit.get("variant_summaries") or {}).get(variant)
        if not summary:
            continue
        decisions = summary.get("decision_counts") or {}
        severities = summary.get("severity_counts") or {}
        top_decisions = ", ".join(f"{k}:{v}" for k, v in sorted(decisions.items(), key=lambda x: (-x[1], x[0]))[:4])
        crit_major = int(severities.get("critical", 0)) + int(severities.get("major", 0))
        metrics = summary.get("metric_context") or {}
        lines.append(
            f"| {variant} | {summary.get('evidence_board_available')} | {summary.get('card_count')} | {top_decisions} | {crit_major} | {float(metrics.get('rougeL') or 0):.3f} | {float(metrics.get('meteor') or 0):.3f} |"
        )
    lines.append("")
    lines.append("## Critical Slot Table")
    header = ["Slot", "GT/reference", *variants, "Severity", "Provenance issue"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in audit.get("critical_slot_table") or []:
        gt = row.get("reference_concepts") or row.get("reference_numbers") or row.get("ground_truth_reference") or ""
        vals = [str(row.get("slot", "")), str(gt)]
        vals.extend(str(row.get(v, "")) for v in variants)
        vals.extend([str(row.get("severity", "")), str(row.get("provenance_issue", ""))])
        vals = [v.replace("|", ";") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Representative Claim Cards")
    cards_by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for card in audit.get("claim_cards") or []:
        cards_by_variant[str(card.get("model"))].append(card)
    decision_priority = {
        "contradicted": 0,
        "debug_leakage": 1,
        "unsupported": 2,
        "over_cautious_false_negative": 3,
        "under_specified": 4,
        "needs_human_adjudication": 5,
        "supported_present": 6,
        "supported_absent": 7,
    }
    severity_priority = {"critical": 0, "major": 1, "moderate": 2, "minor": 3, "debug_only": 4}
    for variant in variants:
        cards = cards_by_variant.get(variant, [])
        if not cards:
            continue
        lines.append(f"### {variant}")
        cards = sorted(cards, key=lambda c: (severity_priority.get(str(c.get("severity")), 9), decision_priority.get(str(c.get("decision")), 9)))
        for card in cards[:max_cards_per_variant]:
            lines.append(
                f"- `{card.get('severity')}` `{card.get('decision')}` [{card.get('section')}] {card.get('claim')}"
            )
            lines.append(f"  - action: `{card.get('recommended_action')}`")
            lines.append(f"  - revision: {card.get('recommended_revision')}")
            prov = card.get("patient_signal_provenance") or []
            if prov:
                lines.append(f"  - provenance: {prov[0].get('source_object')} / {prov[0].get('time_window')} / {prov[0].get('channels_or_regions')}")
            else:
                lines.append("  - provenance: no patient-specific signal provenance linked in this first-pass audit")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def audit_cases(
    *,
    comparison_root: Path,
    output_dir: Path,
    row_indices: Optional[Sequence[int]],
    variants: Sequence[str],
    artifact_roots: Mapping[str, Path],
    config_dir: Path,
) -> Dict[str, Any]:
    slot_schema = load_yaml(config_dir / "clinical_slot_schema.yaml")
    failure_taxonomy = load_yaml(config_dir / "evaluation_failure_taxonomy.yaml")
    claim_gate_policy = load_yaml(config_dir / "claim_gate_policy.yaml")
    case_json_dir = comparison_root / "per_case_json"
    if not case_json_dir.exists():
        raise FileNotFoundError(f"per_case_json not found: {case_json_dir}")

    selected_files = sorted(case_json_dir.glob("row_*.json"))
    if row_indices is not None:
        wanted = {int(x) for x in row_indices}
        selected_files = [p for p in selected_files if int(p.name.split("_")[1]) in wanted]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_cards: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for path in selected_files:
        payload = read_json(path)
        audit = audit_case(
            payload,
            slot_schema=slot_schema,
            failure_taxonomy=failure_taxonomy,
            claim_gate_policy=claim_gate_policy,
            artifact_roots=artifact_roots,
            variants=variants,
        )
        case_id = audit["case_id"]
        write_json(output_dir / "per_case_json" / f"{case_id}.json", audit)
        (output_dir / "per_case_markdown" / f"{case_id}.md").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "per_case_markdown" / f"{case_id}.md").write_text(
            render_case_audit_markdown(audit, variants=variants), encoding="utf-8"
        )
        all_cards.extend(audit.get("claim_cards") or [])
        for variant, summary in (audit.get("variant_summaries") or {}).items():
            decisions = summary.get("decision_counts") or {}
            severities = summary.get("severity_counts") or {}
            metrics = summary.get("metric_context") or {}
            case_rows.append(
                {
                    "case_id": case_id,
                    "row_index": audit.get("row_index"),
                    "report_id": audit.get("report_id"),
                    "variant": variant,
                    "card_count": summary.get("card_count"),
                    "critical_cards": severities.get("critical", 0),
                    "major_cards": severities.get("major", 0),
                    "unsupported": decisions.get("unsupported", 0),
                    "over_cautious_false_negative": decisions.get("over_cautious_false_negative", 0),
                    "debug_leakage": decisions.get("debug_leakage", 0),
                    "under_specified": decisions.get("under_specified", 0),
                    "evidence_board_available": summary.get("evidence_board_available"),
                    "rougeL": metrics.get("rougeL"),
                    "meteor": metrics.get("meteor"),
                    "concept_f1_mean": metrics.get("concept_f1_mean"),
                }
            )

    card_rows = []
    for card in all_cards:
        card_rows.append(
            {
                "case_id": card.get("case_id"),
                "model": card.get("model"),
                "section": card.get("section"),
                "slot": card.get("slot"),
                "claim_type": card.get("claim_type"),
                "severity": card.get("severity"),
                "decision": card.get("decision"),
                "reference_status": card.get("reference_status"),
                "recommended_action": card.get("recommended_action"),
                "provenance_level": card.get("provenance_level"),
                "claim": card.get("claim"),
            }
        )
    write_csv(output_dir / "clinical_audit_case_summary.csv", case_rows)
    write_csv(output_dir / "clinical_audit_claim_cards.csv", card_rows)

    by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        rows = [row for row in case_rows if row.get("variant") == variant]
        cards = [card for card in all_cards if card.get("model") == variant]
        if not rows:
            continue
        by_variant[variant] = {
            "cases": len(rows),
            "cards": len(cards),
            "critical_cards": sum(int(row.get("critical_cards") or 0) for row in rows),
            "major_cards": sum(int(row.get("major_cards") or 0) for row in rows),
            "decision_counts": dict(Counter(str(card.get("decision")) for card in cards)),
            "severity_counts": dict(Counter(str(card.get("severity")) for card in cards)),
            "mean_rougeL": mean(float(row.get("rougeL") or 0.0) for row in rows),
            "mean_meteor": mean(float(row.get("meteor") or 0.0) for row in rows),
            "mean_concept_f1": mean(float(row.get("concept_f1_mean") or 0.0) for row in rows),
        }
    summary = {
        "input_contract": "GT/reference text is evaluation-only; audit is first-pass annotation for human review.",
        "comparison_root": str(comparison_root),
        "output_dir": str(output_dir),
        "case_count": len(selected_files),
        "variants": list(variants),
        "by_variant": by_variant,
    }
    write_json(output_dir / "clinical_audit_summary.json", summary)
    (output_dir / "README.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    return summary


def render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Clinical Provenance Audit Summary",
        "",
        summary["input_contract"],
        "",
        f"Cases: {summary['case_count']}",
        "",
        "| Variant | cards | critical | major | mean concept F1 | mean ROUGE-L | mean METEOR | top decisions |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, row in (summary.get("by_variant") or {}).items():
        decisions = row.get("decision_counts") or {}
        top = ", ".join(f"{k}:{v}" for k, v in sorted(decisions.items(), key=lambda x: (-x[1], x[0]))[:4])
        lines.append(
            f"| {variant} | {row['cards']} | {row['critical_cards']} | {row['major_cards']} | {row['mean_concept_f1']:.3f} | {row['mean_rougeL']:.3f} | {row['mean_meteor']:.3f} | {top} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "- `per_case_json/`: machine-readable per-case audits",
            "- `per_case_markdown/`: human-readable per-case audits",
            "- `clinical_audit_case_summary.csv`: one row per case/model",
            "- `clinical_audit_claim_cards.csv`: one row per claim card",
            "- `clinical_audit_summary.json`: aggregate summary",
        ]
    )
    return "\n".join(lines) + "\n"
