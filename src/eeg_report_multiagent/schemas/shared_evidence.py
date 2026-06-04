from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .report import ClaimSurfaceAction, SurfaceDecision


class EvidenceType(str, Enum):
    DIRECT = "direct"
    PROXY = "proxy"
    METADATA = "metadata"
    DEBUG = "debug"
    DERIVED = "derived"
    LLM_ASSISTED = "llm_assisted"


class ClinicalTarget(str, Enum):
    PDR = "pdr"
    BACKGROUND_SLOWING = "background_slowing"
    BACKGROUND_AMPLITUDE = "background_amplitude"
    EXCESS_BETA = "excess_beta"
    EPILEPTIFORM_MORPHOLOGY = "epileptiform_morphology"
    EVENT_CANDIDATE = "event_candidate"
    SEIZURE_EVIDENCE = "seizure_evidence"
    LOCALIZATION = "localization"
    STATE = "state"
    PROTOCOL = "protocol"
    ARTIFACT = "artifact"
    UNCERTAINTY = "uncertainty"
    CONTEXT = "context"
    UNKNOWN = "unknown"


class EvidenceItem(BaseModel):
    """Typed patient-specific fact/provenance unit.

    The policy-like fields (`reportability`, section lists, `rationale`,
    `caveat`) are retained for artifact compatibility only. New surface
    decisions should be made by `SurfaceDecision`, not by this schema.
    """

    evidence_id: str
    source_module: str
    evidence_type: EvidenceType
    clinical_target: ClinicalTarget | str
    value: Any = None
    unit: Optional[str] = None
    normalized_value: Optional[Any] = None
    confidence: Optional[float] = Field(default=None, description="Deprecated compatibility; prefer measurement confidence or SurfaceDecision metadata.")
    reliability: Optional[float] = Field(default=None, description="Deprecated compatibility; evidence weighting should live outside EvidenceItem.")
    time_provenance: Optional[Dict[str, Any]] = None
    space_provenance: Optional[Dict[str, Any]] = None
    measurement_ids: List[str] = Field(default_factory=list)
    reportability: ClaimSurfaceAction = Field(description="Deprecated compatibility; authoritative surface action is SurfaceDecision.surface_action.")
    allowed_sections: List[str] = Field(default_factory=list, description="Deprecated compatibility; authoritative section gating is SurfaceDecision.")
    forbidden_sections: List[str] = Field(default_factory=list, description="Deprecated compatibility; authoritative section gating is SurfaceDecision.")
    rationale: Optional[str] = Field(default=None, description="Deprecated compatibility; use SurfaceDecision.rationale for report-surface judgment.")
    caveat: Optional[str] = Field(default=None, description="Deprecated compatibility; use SurfaceDecision.caveat for surface wording.")
    debug_payload: Dict[str, Any] = Field(default_factory=dict)
    created_by: str
    created_at: Optional[str] = None

    @classmethod
    def now_iso(cls) -> str:
        return datetime.now(timezone.utc).isoformat()


class EvidenceBoardSnapshot(BaseModel):
    board_id: str
    recording_id: str
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    summary_by_type: Dict[str, int] = Field(default_factory=dict)
    summary_by_target: Dict[str, int] = Field(default_factory=dict)
    blocked_items: List[str] = Field(default_factory=list)
    debug_only_items: List[str] = Field(default_factory=list)
    reportable_items: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)


class SharedEvidenceBoard(BaseModel):
    """Central, queryable store for structured evidence before claim planning."""

    board_id: str
    recording_id: str
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    claim_evidence_links: Dict[str, List[str]] = Field(default_factory=dict)

    def add_evidence(self, item: EvidenceItem) -> None:
        if any(existing.evidence_id == item.evidence_id for existing in self.evidence_items):
            raise ValueError(f"duplicate evidence_id: {item.evidence_id}")
        self.evidence_items.append(item)

    def get_evidence(self, evidence_id: str) -> EvidenceItem:
        for item in self.evidence_items:
            if item.evidence_id == evidence_id:
                return item
        raise KeyError(evidence_id)

    def list_evidence(self) -> List[EvidenceItem]:
        return list(self.evidence_items)

    def query_by_target(self, clinical_target: str) -> List[EvidenceItem]:
        return [item for item in self.evidence_items if str(item.clinical_target) == clinical_target or getattr(item.clinical_target, "value", None) == clinical_target]

    def query_by_section(self, section_name: str) -> List[EvidenceItem]:
        key = section_name.strip().lower()
        return [
            item
            for item in self.evidence_items
            if any(section.strip().lower() == key for section in item.allowed_sections)
        ]

    def query_reportable(self, section_name: str) -> List[EvidenceItem]:
        """Deprecated legacy helper.

        Prefer `query_for_surface_decisions()`, which uses authoritative
        SurfaceDecision objects rather than EvidenceItem compatibility policy
        fields.
        """
        key = section_name.strip().lower()
        out: List[EvidenceItem] = []
        for item in self.evidence_items:
            if item.reportability not in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                continue
            if item.evidence_type == EvidenceType.DEBUG:
                continue
            if any(section.strip().lower() == key for section in item.forbidden_sections):
                continue
            if not item.allowed_sections or any(section.strip().lower() == key for section in item.allowed_sections):
                out.append(item)
        return out

    def query_debug_only(self) -> List[EvidenceItem]:
        """Legacy helper for evidence-board audits, not report synthesis."""
        return [
            item
            for item in self.evidence_items
            if item.reportability == ClaimSurfaceAction.DEBUG_ONLY or item.evidence_type == EvidenceType.DEBUG
        ]

    def query_for_surface_decisions(
        self,
        decisions: List[SurfaceDecision],
        section_name: str | None = None,
        actions: set[ClaimSurfaceAction] | None = None,
    ) -> List[EvidenceItem]:
        """Return evidence linked to authoritative surface decisions."""

        allowed_actions = actions or {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}
        section_key = section_name.strip().lower() if section_name else None
        evidence_ids: set[str] = set()
        for decision in decisions:
            if decision.surface_action not in allowed_actions:
                continue
            if section_key:
                forbidden = {section.strip().lower() for section in decision.forbidden_sections}
                allowed = {section.strip().lower() for section in decision.allowed_sections}
                if section_key in forbidden:
                    continue
                if allowed and section_key not in allowed:
                    continue
            evidence_ids.update(decision.evidence_ids)
        return [item for item in self.evidence_items if item.evidence_id in evidence_ids]

    def link_to_claim(self, claim_id: str, evidence_ids: List[str]) -> None:
        missing = [evidence_id for evidence_id in evidence_ids if not any(item.evidence_id == evidence_id for item in self.evidence_items)]
        if missing:
            raise ValueError(f"cannot link missing evidence ids to claim {claim_id}: {missing}")
        self.claim_evidence_links[claim_id] = list(evidence_ids)

    def validate(self) -> List[str]:  # type: ignore[override]
        warnings: List[str] = []
        seen: set[str] = set()
        for item in self.evidence_items:
            if item.evidence_id in seen:
                warnings.append(f"duplicate evidence_id: {item.evidence_id}")
            seen.add(item.evidence_id)
            if item.reportability in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT} and item.evidence_type == EvidenceType.DEBUG:
                warnings.append(f"debug evidence cannot be reportable: {item.evidence_id}")
            if item.clinical_target == ClinicalTarget.SEIZURE_EVIDENCE and item.reportability in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                if item.evidence_type not in {EvidenceType.DIRECT, EvidenceType.METADATA, EvidenceType.DERIVED}:
                    warnings.append(f"seizure evidence must be direct/metadata/derived: {item.evidence_id}")
            if item.clinical_target == ClinicalTarget.LOCALIZATION and item.reportability in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT}:
                space = item.space_provenance or {}
                has_location = bool(space.get("side") or space.get("region"))
                has_channels = bool(space.get("channels") or space.get("electrode_maxima"))
                if not (has_location and has_channels and item.source_module):
                    warnings.append(f"surface localization lacks required space provenance: {item.evidence_id}")
            if item.reportability in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT} and item.value is not None:
                if isinstance(item.value, (int, float)) and not item.unit:
                    warnings.append(f"numeric reportable evidence lacks unit: {item.evidence_id}")
        for claim_id, evidence_ids in self.claim_evidence_links.items():
            for evidence_id in evidence_ids:
                if evidence_id not in seen:
                    warnings.append(f"claim {claim_id} links missing evidence_id: {evidence_id}")
        return warnings

    def snapshot(self) -> EvidenceBoardSnapshot:
        by_type = Counter(item.evidence_type.value for item in self.evidence_items)
        by_target = Counter(str(getattr(item.clinical_target, "value", item.clinical_target)) for item in self.evidence_items)
        blocked = [item.evidence_id for item in self.evidence_items if item.reportability == ClaimSurfaceAction.BLOCK]
        debug_only = [item.evidence_id for item in self.query_debug_only()]
        reportable = [
            item.evidence_id
            for item in self.evidence_items
            if item.reportability in {ClaimSurfaceAction.ALLOW, ClaimSurfaceAction.CAVEAT} and item.evidence_type != EvidenceType.DEBUG
        ]
        return EvidenceBoardSnapshot(
            board_id=self.board_id,
            recording_id=self.recording_id,
            evidence_items=self.list_evidence(),
            summary_by_type=dict(by_type),
            summary_by_target=dict(by_target),
            blocked_items=blocked,
            debug_only_items=debug_only,
            reportable_items=reportable,
            validation_warnings=self.validate(),
        )
