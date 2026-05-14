from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping

from eeg_report_multiagent.modules.section_router import SectionRouter
from eeg_report_multiagent.modules.surface_policy import SurfacePolicy
from eeg_report_multiagent.schemas.final_prose_audit import (
    ClaimSurfaceMatch,
    DebugLeak,
    DebugLeakType,
    FinalProseAuditResult,
    NumericMatchStatus,
    NumericMention,
    NumericProvenanceMatch,
    SectionLeakage,
)
from eeg_report_multiagent.schemas.report import AtomicClaimPlan, ClaimSurfaceAction
from eeg_report_multiagent.schemas.section_contract import SectionRole
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceItem, EvidenceType, SharedEvidenceBoard


@dataclass(frozen=True)
class _SectionText:
    name: str
    text: str


class FinalProseAuditor:
    """Post-generation audit for numeric provenance and debug surface leakage."""

    BANNED_SURFACE_TERMS: Mapping[str, DebugLeakType] = {
        "candidate burden": DebugLeakType.PROXY_CONCEPT,
        "burden ratio": DebugLeakType.PROXY_CONCEPT,
        "longest candidate train": DebugLeakType.PROXY_CONCEPT,
        "train duration": DebugLeakType.PROXY_CONCEPT,
        "laterality index": DebugLeakType.PROXY_CONCEPT,
        "peak laterality index": DebugLeakType.PROXY_CONCEPT,
        "bifrontal spread tendency": DebugLeakType.PROXY_CONCEPT,
        "morphology screen": DebugLeakType.PROXY_CONCEPT,
        "morphology proxy": DebugLeakType.PROXY_CONCEPT,
        "support score": DebugLeakType.DEBUG_SCORE,
        "likelihood score": DebugLeakType.DEBUG_SCORE,
        "field concentration ratio": DebugLeakType.PROXY_CONCEPT,
        "beta ratio": DebugLeakType.DEBUG_SCORE,
        "slowing score": DebugLeakType.DEBUG_SCORE,
        "weak evidence": DebugLeakType.INTERNAL_REVIEWER_TEXT,
        "do_not_claim": DebugLeakType.INTERNAL_REVIEWER_TEXT,
        "claim_constraints": DebugLeakType.INTERNAL_REVIEWER_TEXT,
        "missing_slots": DebugLeakType.INTERNAL_REVIEWER_TEXT,
        "values_preview": DebugLeakType.MEASUREMENT_ARTIFACT,
        "screen classified": DebugLeakType.PROXY_CONCEPT,
        "local morphology-feature support": DebugLeakType.PROXY_CONCEPT,
        "boundary peak": DebugLeakType.MEASUREMENT_ARTIFACT,
        "search boundary": DebugLeakType.MEASUREMENT_ARTIFACT,
    }

    _RANGE_NUMERIC_RE = re.compile(
        r"(?P<raw>(?:approximately\s+|about\s+|around\s+)?(?P<lo>\d+(?:\.\d+)?)\s*[-–]\s*(?P<hi>\d+(?:\.\d+)?)\s*(?P<unit>Hz|hz|uV|µV|microvolts?|sec|seconds?|s|percent|%)(?=\W|$))",
        re.IGNORECASE,
    )
    _UNIT_NUMERIC_RE = re.compile(
        r"(?P<raw>(?:approximately\s+|about\s+|around\s+)?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>Hz|hz|uV|µV|microvolts?|sec|seconds?|s|percent|%)(?=\W|$))",
        re.IGNORECASE,
    )
    _PREFIX_NUMERIC_RE = re.compile(
        r"(?P<raw>\b(?P<unit>score|ratio|likelihood)\s*(?:of\s*)?(?P<value>\d+(?:\.\d+)?))\b",
        re.IGNORECASE,
    )

    def __init__(self, surface_policy: SurfacePolicy | None = None) -> None:
        self.surface_policy = surface_policy or SurfacePolicy()
        self.router = SectionRouter()

    def audit_report(
        self,
        report_sections: Any,
        shared_evidence_board: SharedEvidenceBoard,
        atomic_claim_plans: Iterable[AtomicClaimPlan],
    ) -> FinalProseAuditResult:
        sections = self._normalize_report_sections(report_sections)
        plans = list(atomic_claim_plans)
        all_numeric: List[NumericMention] = []
        supported: List[NumericProvenanceMatch] = []
        unsupported: List[NumericProvenanceMatch] = []
        debug_leaks: List[DebugLeak] = []
        section_leaks: List[SectionLeakage] = []
        seizure_gate_violations: List[SectionLeakage] = []
        unmatched_claims: List[ClaimSurfaceMatch] = []
        missing_evidence_links: List[ClaimSurfaceMatch] = []
        all_claim_matches: List[ClaimSurfaceMatch] = []

        evidence_items = shared_evidence_board.list_evidence()
        for section in sections:
            numeric_mentions = self.extract_numeric_mentions(section.text, section.name)
            all_numeric.extend(numeric_mentions)
            for mention in numeric_mentions:
                match = self.match_numeric_to_evidence(mention, evidence_items)
                if match.match_status in {NumericMatchStatus.EXACT, NumericMatchStatus.RANGE_CONTAINED}:
                    supported.append(match)
                else:
                    unsupported.append(match)
            debug_leaks.extend(self.detect_banned_debug_terms(section.text, section.name))
            leaks = self.detect_section_leakage(section.name, section.text)
            section_leaks.extend(leaks)
            seizure_gate_violations.extend(self._detect_seizure_gate_violations(section.name, section.text, evidence_items))
            matches = self.match_text_claims_to_atomic_plans(section.name, section.text, plans)
            all_claim_matches.extend(matches)
            unmatched_claims.extend([m for m in matches if m.match_status in {"unmatched_surface_claim", "surface_policy_violation"}])
            missing_evidence_links.extend([m for m in matches if m.match_status == "missing_required_evidence_links"])

        high_risk_count = (
            len(unsupported)
            + len(debug_leaks)
            + len(seizure_gate_violations)
            + len(section_leaks)
            + len([m for m in unmatched_claims if m.match_status == "surface_policy_violation"])
        )
        metrics = self._metrics(
            all_numeric=all_numeric,
            supported=supported,
            unsupported=unsupported,
            debug_leaks=debug_leaks,
            section_leaks=section_leaks,
            seizure_gate_violations=seizure_gate_violations,
            claim_matches=all_claim_matches,
            sections=sections,
        )
        warnings: List[str] = []
        if unsupported:
            warnings.append(f"unsupported_numeric_count={len(unsupported)}")
        if debug_leaks:
            warnings.append(f"debug_leak_count={len(debug_leaks)}")
        if section_leaks:
            warnings.append(f"section_leakage_count={len(section_leaks)}")
        if seizure_gate_violations:
            warnings.append(f"seizure_gate_violation_count={len(seizure_gate_violations)}")
        if missing_evidence_links:
            warnings.append(f"missing_required_evidence_links={len(missing_evidence_links)}")

        return FinalProseAuditResult(
            unsupported_numeric_mentions=unsupported,
            supported_numeric_mentions=supported,
            debug_leaks=debug_leaks,
            section_leakages=section_leaks,
            seizure_gate_violations=seizure_gate_violations,
            unmatched_surface_claims=unmatched_claims,
            missing_required_evidence_links=missing_evidence_links,
            pass_fail="pass" if high_risk_count == 0 and not missing_evidence_links else "fail",
            warnings=warnings,
            metrics=metrics,
        )

    def extract_numeric_mentions(self, text: str, section_name: str = "") -> List[NumericMention]:
        mentions: List[NumericMention] = []
        occupied: list[range] = []
        for match in self._RANGE_NUMERIC_RE.finditer(text):
            mention_range = range(match.start(), match.end())
            occupied.append(mention_range)
            lo = float(match.group("lo"))
            hi = float(match.group("hi"))
            unit = self._normalize_unit(match.group("unit"))
            mentions.append(
                NumericMention(
                    raw_text=match.group("raw"),
                    value={"lower": lo, "upper": hi},
                    unit=unit,
                    normalized_value={"lower": lo, "upper": hi},
                    section_name=section_name,
                    sentence=self._sentence_for_span(text, match.start(), match.end()),
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
        for regex in (self._UNIT_NUMERIC_RE, self._PREFIX_NUMERIC_RE):
            for match in regex.finditer(text):
                if any(match.start() >= r.start and match.end() <= r.stop for r in occupied):
                    continue
                value = float(match.group("value"))
                unit = self._normalize_unit(match.group("unit"))
                mentions.append(
                    NumericMention(
                        raw_text=match.group("raw"),
                        value=value,
                        unit=unit,
                        normalized_value=value,
                        section_name=section_name,
                        sentence=self._sentence_for_span(text, match.start(), match.end()),
                        char_start=match.start(),
                        char_end=match.end(),
                    )
                )
        return sorted(mentions, key=lambda m: m.char_start)

    def detect_banned_debug_terms(self, text: str, section_name: str = "") -> List[DebugLeak]:
        leaks: List[DebugLeak] = []
        lowered = text.lower()
        for term, leak_type in self.BANNED_SURFACE_TERMS.items():
            start = 0
            while True:
                idx = lowered.find(term, start)
                if idx < 0:
                    break
                leaks.append(
                    DebugLeak(
                        term=term,
                        section_name=section_name,
                        sentence=self._sentence_for_span(text, idx, idx + len(term)),
                        leak_type=leak_type,
                    )
                )
                start = idx + len(term)
        return leaks

    def detect_section_leakage(self, section_name: str, section_text: str) -> List[SectionLeakage]:
        role = self.router.role_for_section(section_name)
        leaks: List[SectionLeakage] = []
        for sentence in self._sentences(section_text):
            low = sentence.lower()
            if role == SectionRole.BACKGROUND:
                if self._has_any(low, ["seizure", "epileptiform", "spike-wave", "spike wave", "transient candidate", "candidate burden", "train duration", "support score", "likelihood score"]):
                    leaks.append(self._section_leak(section_name, sentence, "background_section_contamination", "Background section contains seizure/event/debug language."))
                if re.search(r"\b0\.5\s*hz\b", low) and self._has_any(low, ["dominant rhythm", "pdr", "posterior dominant"]):
                    leaks.append(self._section_leak(section_name, sentence, "background_pdr_boundary_misuse", "Boundary/global frequency is being used as clinical rhythm/PDR."))
            elif role == SectionRole.EPILEPTIFORM:
                if "seizure" in low and self._has_any(low, ["confirmed", "consist", "recorded", "electrographic"]):
                    leaks.append(self._section_leak(section_name, sentence, "seizure_claim_in_epileptiform_section", "Epileptiform section contains seizure confirmation language."))
                if self._has_any(low, ["support score", "likelihood score", "field concentration ratio"]):
                    leaks.append(self._section_leak(section_name, sentence, "internal_score_in_epileptiform_section", "Internal score language appears in epileptiform section."))
            elif role == SectionRole.EVENTS_SEIZURES:
                if self._has_any(low, ["candidate burden", "support score", "likelihood score", "field concentration ratio"]):
                    leaks.append(self._section_leak(section_name, sentence, "events_section_proxy_leakage", "Events/seizures section contains proxy/debug language."))
            elif role == SectionRole.SEIZURES:
                if self._has_any(low, ["epileptiform", "interictal", "transient candidate", "candidate burden", "spike-wave", "spike wave"]):
                    leaks.append(self._section_leak(section_name, sentence, "seizure_section_interictal_contamination", "Seizure section contains interictal/transient-candidate language."))
            elif role == SectionRole.IMPRESSION:
                if self._has_any(low, ["candidate burden", "field concentration ratio", "support score", "likelihood score"]):
                    leaks.append(self._section_leak(section_name, sentence, "impression_from_proxy_debug_evidence", "Impression contains proxy/debug evidence language."))
                if "seizure" in low and not self._is_safe_no_seizure_fallback(low):
                    leaks.append(self._section_leak(section_name, sentence, "impression_seizure_claim", "Impression contains seizure language requiring seizure-specific evidence."))
        return leaks

    def match_numeric_to_evidence(self, numeric_mention: NumericMention, evidence_items: Iterable[EvidenceItem]) -> NumericProvenanceMatch:
        section_role = self.router.role_for_section(numeric_mention.section_name).value if numeric_mention.section_name else ""
        unit_matches_wrong_status: list[tuple[EvidenceItem, NumericMatchStatus, str]] = []
        for item in evidence_items:
            value_match = self._numeric_value_match(numeric_mention, item)
            if value_match is None:
                continue
            if numeric_mention.unit and item.unit and self._normalize_unit(item.unit) != numeric_mention.unit:
                unit_matches_wrong_status.append((item, NumericMatchStatus.UNIT_MISMATCH, f"Evidence {item.evidence_id} matched value but unit differs."))
                continue
            if item.reportability not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT} or item.evidence_type == EvidenceType.DEBUG:
                return NumericProvenanceMatch(
                    numeric_mention=numeric_mention,
                    matched_evidence_id=item.evidence_id,
                    match_status=NumericMatchStatus.MATCHED_BUT_NOT_REPORTABLE,
                    rationale=f"Numeric mention matches evidence {item.evidence_id}, but evidence is {item.reportability.value}/{item.evidence_type.value}.",
                )
            if item.allowed_sections and section_role and section_role not in item.allowed_sections:
                return NumericProvenanceMatch(
                    numeric_mention=numeric_mention,
                    matched_evidence_id=item.evidence_id,
                    match_status=NumericMatchStatus.MATCHED_BUT_WRONG_SECTION,
                    rationale=f"Numeric mention matches evidence {item.evidence_id}, but section {section_role} is not allowed.",
                )
            if not self._unit_clinically_meaningful(numeric_mention.unit, item):
                return NumericProvenanceMatch(
                    numeric_mention=numeric_mention,
                    matched_evidence_id=item.evidence_id,
                    match_status=NumericMatchStatus.UNIT_MISMATCH,
                    rationale=f"Unit {numeric_mention.unit} is not clinically meaningful for target {item.clinical_target}.",
                )
            return NumericProvenanceMatch(
                numeric_mention=numeric_mention,
                matched_evidence_id=item.evidence_id,
                match_status=value_match,
                rationale=f"Numeric mention is supported by reportable evidence {item.evidence_id}.",
            )
        if unit_matches_wrong_status:
            item, status, rationale = unit_matches_wrong_status[0]
            return NumericProvenanceMatch(numeric_mention=numeric_mention, matched_evidence_id=item.evidence_id, match_status=status, rationale=rationale)
        return NumericProvenanceMatch(
            numeric_mention=numeric_mention,
            matched_evidence_id=None,
            match_status=NumericMatchStatus.NO_MATCH,
            rationale="No EvidenceItem value/normalized_value matched this numeric mention.",
        )

    def match_text_claims_to_atomic_plans(
        self,
        section_name: str,
        section_text: str,
        atomic_claim_plans: Iterable[AtomicClaimPlan],
    ) -> List[ClaimSurfaceMatch]:
        plans = list(atomic_claim_plans)
        matches: List[ClaimSurfaceMatch] = []
        for sentence in self._sentences(section_text):
            if not self._is_surface_claim_sentence(sentence):
                continue
            plan = self._best_plan_match(sentence, plans)
            if plan is None:
                matches.append(
                    ClaimSurfaceMatch(
                        section_name=section_name,
                        sentence=sentence,
                        match_status="unmatched_surface_claim",
                        rationale="Clinical-looking sentence did not match any AtomicClaimPlan.",
                    )
                )
                continue
            if plan.surface_action not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                matches.append(
                    ClaimSurfaceMatch(
                        section_name=section_name,
                        sentence=sentence,
                        matched_plan_id=plan.plan_id,
                        matched_evidence_ids=plan.evidence_ids,
                        match_status="surface_policy_violation",
                        rationale=f"Sentence matched plan {plan.plan_id}, but plan action is {plan.surface_action.value}.",
                    )
                )
                continue
            if not plan.evidence_ids:
                matches.append(
                    ClaimSurfaceMatch(
                        section_name=section_name,
                        sentence=sentence,
                        matched_plan_id=plan.plan_id,
                        matched_evidence_ids=[],
                        match_status="missing_required_evidence_links",
                        rationale=f"Sentence matched plan {plan.plan_id}, but plan has no evidence_ids.",
                    )
                )
                continue
            matches.append(
                ClaimSurfaceMatch(
                    section_name=section_name,
                    sentence=sentence,
                    matched_plan_id=plan.plan_id,
                    matched_evidence_ids=plan.evidence_ids,
                    match_status="matched_allowed_claim",
                    rationale=f"Sentence matched allowed/caveated plan {plan.plan_id}.",
                )
            )
        return matches

    def _detect_seizure_gate_violations(self, section_name: str, section_text: str, evidence_items: Iterable[EvidenceItem]) -> List[SectionLeakage]:
        has_seizure_evidence = any(
            item.clinical_target == ClinicalTarget.SEIZURE_EVIDENCE
            and item.reportability in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}
            and item.evidence_type in {EvidenceType.DIRECT, EvidenceType.METADATA, EvidenceType.DERIVED}
            for item in evidence_items
        )
        violations: List[SectionLeakage] = []
        for sentence in self._sentences(section_text):
            low = sentence.lower()
            if self._is_safe_no_seizure_fallback(low):
                continue
            if "seizure" in low and not has_seizure_evidence:
                violations.append(self._section_leak(section_name, sentence, "seizure_claim_without_seizure_evidence", "Seizure language appears without reportable seizure_evidence EvidenceItem."))
        return violations

    def _normalize_report_sections(self, report_sections: Any) -> List[_SectionText]:
        if isinstance(report_sections, dict):
            if "report_sections" in report_sections and isinstance(report_sections["report_sections"], list):
                return self._normalize_report_sections(report_sections["report_sections"])
            return [_SectionText(str(k), str(v)) for k, v in report_sections.items()]
        if isinstance(report_sections, list):
            out: List[_SectionText] = []
            for item in report_sections:
                if isinstance(item, dict):
                    out.append(_SectionText(str(item.get("section_name", "")), str(item.get("section_text", ""))))
                else:
                    name = str(getattr(item, "section_name", getattr(item, "section_type", "")))
                    text = str(getattr(item, "section_text", getattr(item, "text", "")))
                    out.append(_SectionText(name, text))
            return out
        name = str(getattr(report_sections, "section_name", getattr(report_sections, "section_type", "report")))
        text = str(getattr(report_sections, "section_text", getattr(report_sections, "text", report_sections)))
        return [_SectionText(name, text)]

    def _numeric_value_match(self, mention: NumericMention, item: EvidenceItem) -> NumericMatchStatus | None:
        candidates = [item.normalized_value, item.value]
        for value in candidates:
            if value is None:
                continue
            if isinstance(mention.normalized_value, dict):
                mlo = float(mention.normalized_value["lower"])
                mhi = float(mention.normalized_value["upper"])
                if isinstance(value, dict) and value.get("lower") is not None and value.get("upper") is not None:
                    elo = float(value["lower"])
                    ehi = float(value["upper"])
                    if abs(mlo - elo) <= 0.05 and abs(mhi - ehi) <= 0.05:
                        return NumericMatchStatus.EXACT
                    if elo <= mlo <= mhi <= ehi:
                        return NumericMatchStatus.RANGE_CONTAINED
                if isinstance(value, (int, float)) and mlo <= float(value) <= mhi:
                    return NumericMatchStatus.RANGE_CONTAINED
            else:
                mv = float(mention.normalized_value)
                if isinstance(value, (int, float)) and abs(mv - float(value)) <= max(0.05, abs(float(value)) * 0.01):
                    return NumericMatchStatus.EXACT
                if isinstance(value, dict) and value.get("lower") is not None and value.get("upper") is not None:
                    if float(value["lower"]) <= mv <= float(value["upper"]):
                        return NumericMatchStatus.RANGE_CONTAINED
                if isinstance(value, list) and any(isinstance(x, (int, float)) and abs(mv - float(x)) <= max(0.05, abs(float(x)) * 0.01) for x in value):
                    return NumericMatchStatus.EXACT
        return None

    def _unit_clinically_meaningful(self, unit: str | None, item: EvidenceItem) -> bool:
        if not unit:
            return False
        target = str(getattr(item.clinical_target, "value", item.clinical_target))
        if unit == "score":
            return False
        if unit == "ratio":
            return False
        if unit == "Hz":
            return target in {"pdr", "background_slowing", "epileptiform_morphology"}
        if unit == "uV":
            return target in {"background_amplitude", "epileptiform_morphology"}
        if unit == "sec":
            return target in {"seizure_evidence"}
        if unit == "percent":
            return target not in {"event_candidate", "localization", "uncertainty"}
        return True

    def _best_plan_match(self, sentence: str, plans: List[AtomicClaimPlan]) -> AtomicClaimPlan | None:
        sentence_norm = self._norm(sentence)
        best: tuple[float, AtomicClaimPlan] | None = None
        for plan in plans:
            plan_norm = self._norm(plan.proposed_text)
            if not plan_norm:
                continue
            if plan_norm in sentence_norm or sentence_norm in plan_norm:
                score = 1.0
            else:
                s_tokens = set(sentence_norm.split())
                p_tokens = set(plan_norm.split())
                if not s_tokens or not p_tokens:
                    score = 0.0
                else:
                    score = len(s_tokens & p_tokens) / max(len(s_tokens), 1)
            if score >= 0.55 and (best is None or score > best[0]):
                best = (score, plan)
        return best[1] if best else None

    def _metrics(
        self,
        *,
        all_numeric: List[NumericMention],
        supported: List[NumericProvenanceMatch],
        unsupported: List[NumericProvenanceMatch],
        debug_leaks: List[DebugLeak],
        section_leaks: List[SectionLeakage],
        seizure_gate_violations: List[SectionLeakage],
        claim_matches: List[ClaimSurfaceMatch],
        sections: List[_SectionText],
    ) -> dict[str, float]:
        numeric_total = len(all_numeric)
        claim_total = len(claim_matches)
        matched_claims = [m for m in claim_matches if m.match_status == "matched_allowed_claim"]
        evidence_linked = [m for m in matched_claims if m.matched_evidence_ids]
        high_risk = len(unsupported) + len(debug_leaks) + len(section_leaks) + len(seizure_gate_violations)
        return {
            "UnsupportedNumericRate": len(unsupported) / numeric_total if numeric_total else 0.0,
            "InternalArtifactExposureRate": 1.0 if debug_leaks else 0.0,
            "DebugLeakCount": float(len(debug_leaks)),
            "SectionLeakageRate": len(section_leaks) / max(len(sections), 1),
            "SeizureGateViolationRate": 1.0 if seizure_gate_violations else 0.0,
            "NumericProvenanceAccuracy": len(supported) / numeric_total if numeric_total else 1.0,
            "ClaimTraceCoverage": len(matched_claims) / claim_total if claim_total else 1.0,
            "EvidenceLinkedClaimRate": len(evidence_linked) / len(matched_claims) if matched_claims else 1.0,
            "AuditPassRate": 1.0 if high_risk == 0 else 0.0,
        }

    def _sentence_for_span(self, text: str, start: int, end: int) -> str:
        left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
        right_dot = text.find(".", end)
        right_new = text.find("\n", end)
        candidates = [x for x in [right_dot, right_new] if x >= 0]
        right = min(candidates) if candidates else len(text)
        return text[left + 1 : right + (1 if right == right_dot else 0)].strip()

    def _sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _normalize_unit(self, unit: str | None) -> str | None:
        if unit is None:
            return None
        low = unit.lower().strip()
        if low == "hz":
            return "Hz"
        if low in {"uv", "µv", "microvolt", "microvolts"}:
            return "uV"
        if low in {"sec", "second", "seconds", "s"}:
            return "sec"
        if low in {"%", "percent"}:
            return "percent"
        if low in {"score", "ratio", "likelihood"}:
            return low
        return unit

    def _section_leak(self, section_name: str, sentence: str, leakage_type: str, rationale: str) -> SectionLeakage:
        return SectionLeakage(section_name=section_name, sentence=sentence, leakage_type=leakage_type, rationale=rationale)

    def _has_any(self, text: str, needles: list[str]) -> bool:
        return any(needle in text for needle in needles)

    def _is_safe_no_seizure_fallback(self, low_sentence: str) -> bool:
        return "no seizure-specific evidence was produced by the current structured tools" in low_sentence

    def _is_surface_claim_sentence(self, sentence: str) -> bool:
        low = sentence.lower()
        if not low or low.startswith("no surface-allowed") or self._is_safe_no_seizure_fallback(low):
            return False
        claim_keywords = [
            "posterior", "pdr", "rhythm", "slowing", "beta", "amplitude", "seizure", "epileptiform",
            "spike", "sharp", "photic", "hyperventilation", "sleep", "awake", "localization", "lateral",
            "reactivity", "frequency", "background",
            "candidate", "burden", "score", "ratio",
        ]
        return self._has_any(low, claim_keywords)

    def _norm(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. -]+", " ", text.lower())).strip()
