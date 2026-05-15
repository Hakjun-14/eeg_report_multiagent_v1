from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from eeg_report_multiagent.modules.gt_required_suppression_auditor import GTClaimExtractor, _norm, _range_overlap
from eeg_report_multiagent.schemas.generated_claim_audit import (
    GeneratedClaimAuditResult,
    GeneratedClaimMatch,
    GTClaimRecallMatch,
)
from eeg_report_multiagent.schemas.gt_suppression import GTAtomicClaim


_NUMERIC_CLAIM_TYPES = {"pdr_frequency", "background_amplitude", "event_amplitude", "event_duration", "event_frequency"}
_STRICT_VALUE_CLAIM_TYPES = {
    "seizure_absent",
    "push_button_absent",
    "photic_status",
    "photic_response",
    "hyperventilation_status",
    "localization_laterality",
    "localization_region",
}


class GeneratedClaimAuditor:
    """Symmetric text-level atomic-claim audit for generated reports.

    This intentionally does not require a SharedEvidenceBoard. It answers a different
    question from provenance audits: how many GT report claims are recovered by the
    generated text, and how many generated claims are extra relative to the GT text?
    """

    def __init__(self) -> None:
        self.extractor = GTClaimExtractor()

    def audit_case(
        self,
        *,
        case_id: str,
        variant: str,
        gt_report_json: Path,
        generated_report_json: Path,
    ) -> GeneratedClaimAuditResult:
        gt_claims = self.extractor.extract_from_report_json(gt_report_json, case_id=case_id)
        generated_sections = self.extract_sections_from_generated_report(generated_report_json)
        generated_claims = self.extractor.extract_from_sections(generated_sections, case_id=f"{case_id}__{variant}")
        generated_claims = [claim.model_copy(update={"gt_claim_id": f"gen_{idx+1:04d}_{claim.claim_type}"}) for idx, claim in enumerate(generated_claims)]

        generated_matches: list[GeneratedClaimMatch] = []
        for generated in generated_claims:
            matched_ids = [gt.gt_claim_id for gt in gt_claims if claims_match(gt, generated)]
            generated_matches.append(
                GeneratedClaimMatch(
                    case_id=case_id,
                    variant=variant,
                    generated_claim_id=generated.gt_claim_id,
                    generated_claim=generated,
                    matched_gt_claim_ids=matched_ids,
                    is_extra_claim=not bool(matched_ids),
                )
            )

        recall_matches: list[GTClaimRecallMatch] = []
        for gt in gt_claims:
            matched_ids = [generated.gt_claim_id for generated in generated_claims if claims_match(gt, generated)]
            recall_matches.append(
                GTClaimRecallMatch(
                    case_id=case_id,
                    variant=variant,
                    gt_claim_id=gt.gt_claim_id,
                    gt_claim=gt,
                    matched_generated_claim_ids=matched_ids,
                    is_missing=not bool(matched_ids),
                )
            )

        metrics = _metrics(gt_claims, generated_claims, generated_matches, recall_matches)
        return GeneratedClaimAuditResult(
            case_id=case_id,
            variant=variant,
            gt_claims=gt_claims,
            generated_claims=generated_claims,
            generated_claim_matches=generated_matches,
            gt_claim_recall_matches=recall_matches,
            metrics=metrics,
            notes=[
                "Text-only atomic claim comparison. GT reports are used only for evaluation-time matching.",
                "No patient-specific signal provenance is inferred for CELM or any text-only variant in this audit.",
            ],
        )

    def extract_sections_from_generated_report(self, path: Path) -> dict[str, str]:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            sections = self.extractor._extract_sections(payload)  # noqa: SLF001 - reuse one parser symmetrically.
            if sections:
                return sections
            if isinstance(payload.get("report_sections"), list):
                return self.extractor._extract_sections({"EEG_sections": payload["report_sections"]})  # noqa: SLF001
        raise ValueError(f"Could not extract report sections from generated report: {path}")


def claims_match(gt_claim: GTAtomicClaim, generated_claim: GTAtomicClaim) -> bool:
    if gt_claim.claim_type != generated_claim.claim_type:
        return False
    if gt_claim.claim_type in _NUMERIC_CLAIM_TYPES:
        if _unit_norm(gt_claim.unit) != _unit_norm(generated_claim.unit):
            return False
        return _range_overlap(gt_claim.normalized_value, generated_claim.normalized_value)
    if gt_claim.claim_type == "electrode_maxima":
        return bool(set(_as_str_list(gt_claim.normalized_value)) & set(_as_str_list(generated_claim.normalized_value)))
    if gt_claim.claim_type in _STRICT_VALUE_CLAIM_TYPES:
        return _norm(str(gt_claim.normalized_value)) == _norm(str(generated_claim.normalized_value))
    if gt_claim.normalized_value is not None and generated_claim.normalized_value is not None:
        return _norm(str(gt_claim.normalized_value)) == _norm(str(generated_claim.normalized_value))
    return True


def _metrics(
    gt_claims: Sequence[GTAtomicClaim],
    generated_claims: Sequence[GTAtomicClaim],
    generated_matches: Sequence[GeneratedClaimMatch],
    recall_matches: Sequence[GTClaimRecallMatch],
) -> dict[str, float]:
    gt_total = len(gt_claims)
    gen_total = len(generated_claims)
    matched_gen = sum(1 for match in generated_matches if not match.is_extra_claim)
    matched_gt = sum(1 for match in recall_matches if not match.is_missing)
    numeric_gt = [claim for claim in gt_claims if claim.claim_type in _NUMERIC_CLAIM_TYPES]
    numeric_gen = [claim for claim in generated_claims if claim.claim_type in _NUMERIC_CLAIM_TYPES]
    numeric_gt_ids = {claim.gt_claim_id for claim in numeric_gt}
    numeric_gen_ids = {claim.gt_claim_id for claim in numeric_gen}
    matched_numeric_gt = sum(1 for match in recall_matches if match.gt_claim_id in numeric_gt_ids and not match.is_missing)
    matched_numeric_gen = sum(1 for match in generated_matches if match.generated_claim_id in numeric_gen_ids and not match.is_extra_claim)
    return {
        "GTClaimCount": float(gt_total),
        "GeneratedClaimCount": float(gen_total),
        "MatchedGeneratedClaimCount": float(matched_gen),
        "MatchedGTClaimCount": float(matched_gt),
        "GeneratedClaimPrecision": _safe_div(matched_gen, gen_total),
        "GTClaimRecall": _safe_div(matched_gt, gt_total),
        "ExtraClaimRate": _safe_div(gen_total - matched_gen, gen_total),
        "MissingGTClaimRate": _safe_div(gt_total - matched_gt, gt_total),
        "NumericGeneratedClaimPrecision": _safe_div(matched_numeric_gen, len(numeric_gen)),
        "NumericGTClaimRecall": _safe_div(matched_numeric_gt, len(numeric_gt)),
        "NumericGTClaimCount": float(len(numeric_gt)),
        "NumericGeneratedClaimCount": float(len(numeric_gen)),
    }


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _unit_norm(unit: str | None) -> str:
    if not unit:
        return ""
    u = str(unit).lower().replace("µ", "u")
    if u in {"hz"}:
        return "Hz"
    if u in {"uv", "microvolt", "microvolts"}:
        return "uV"
    if u in {"s", "sec", "second", "seconds"}:
        return "sec"
    if u in {"%", "percent"}:
        return "percent"
    return str(unit)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_norm(str(item)) for item in value]
    return [_norm(str(value))] if value is not None else []
