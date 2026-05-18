# Role Separation Compatibility Refactor

## Code Flow Reconnaissance
1. `MeasurementValue` objects are produced by bounded signal/status tools in `tools/common.py`, background/event/parser modules, and test fixtures.
2. `Finding` objects are produced in `background_module.py`, `event_module.py`, and `protocol_state_context_parser.py` from those measurements.
3. `EvidenceItem` objects are created by `evidence_item_adapter.py` when `EvidenceBoard.ensure_shared_evidence_board()` materializes the `SharedEvidenceBoard`.
4. `EvidenceItem.reportability`, `allowed_sections`, and `forbidden_sections` remain compatibility fields but should no longer be treated as the final surface judgment.
5. `AtomicClaimPlan` objects are produced in `ReportSynthesizer.build_atomic_claim_plan()` from findings plus linked EvidenceItems.
6. `SurfacePolicy` and reportability calibration currently decide `allow/caveat/block/debug_only`; this patch emits those judgments as explicit `SurfaceDecision` objects.
7. `surface_decisions.json` is the new first-class artifact for report-surface decisions.
8. Final deterministic report text is produced in `ReportSynthesizer.synthesize()` and `synthesize_celm_sections()` from allow/caveat SurfaceDecision-linked claims only.
9. LLM report synthesis receives only allow/caveat claim plans plus sanitized SurfaceDecision payloads; raw evidence/debug payloads remain excluded.
10. `FinalProseAuditor` runs after synthesis in session/selected50 paths to detect unsupported numerics, debug leakage, section leakage, and seizure-gate violations.

## Minimal Migration Design
- `EvidenceItem` is treated conceptually as fact/provenance only; legacy policy fields remain for backward compatibility until a later schema migration.
- `SurfaceDecision` is the authoritative surface judgment object and is persisted separately as `surface_decisions.json`.
- `AtomicClaimPlan.surface_action` remains as a legacy mirror so older audits and tests continue to work.
- Report synthesis now routes section text through linked SurfaceDecision actions in `{allow, caveat}` plus deterministic safe fallbacks.
- Hard deny checks remain deterministic and cannot be bypassed by future LLM surface review.

## Stage 2.98 Dependency Removal Notes
- Report synthesis no longer creates `cal_*` reportable EvidenceItem copies for calibrated claims.
- Calibrated surface permission is represented by `AtomicClaimPlan.evidence_ids` plus `SurfaceDecision.surface_action`, not by mutating or cloning EvidenceItems into reportable objects.
- `SurfacePolicy` no longer uses `EvidenceItem.reportability` as an evidence-gate input; it uses evidence type, clinical target, and deterministic hard-deny rules.
- `FinalProseAuditor` treats numeric support as valid when the numeric EvidenceItem is linked to an allow/caveat AtomicClaimPlan; legacy EvidenceItem reportability is only a fallback when no claim-plan trace is supplied.
- Remaining `EvidenceItem.reportability` reads are compatibility diagnostics in `SharedEvidenceBoard` and evidence-flow audit outputs, not authoritative report-surface routing.

## Stage 2.99 Schema Slimming
- `Finding` keeps only clinical grouping semantics in new runtime outputs: `finding_id`, `finding_type`, `assertion`, quantitation, and linked `measurement_ids`.
- Legacy `Finding.provenance`, `Finding.confidence`, and `Finding.source_module` remain loadable but are no longer populated by background/event/protocol modules.
- `EvidenceItem` remains the patient-specific evidence pool; policy fields are compatibility metadata, not authoritative report-surface decisions.
- Report-surface judgment remains `AtomicClaimPlan -> SurfaceDecision -> SurfacePolicy -> Report`.

## Stage 3.0 Confidence and Surface Query Cleanup
- `MeasurementValue.confidence` was removed from the runtime schema.
- Tool helper functions may still accept a legacy `confidence` argument for call-site compatibility, but it is not stored on `MeasurementValue`.
- Newly generated `EvidenceItem` records no longer populate compatibility `confidence` or `reliability` values.
- `SharedEvidenceBoard.query_for_surface_decisions()` now provides the preferred SurfaceDecision-based evidence subset lookup.
