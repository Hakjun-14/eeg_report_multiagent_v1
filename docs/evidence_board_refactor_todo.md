# Evidence Board Refactor TODO

## Current Direction

Canonical path:

`MeasurementValue -> EvidenceItem -> SharedEvidenceBoard -> AtomicClaimPlan -> SurfaceDecision -> report_text`

`EvidenceItem` should not be a runtime bridge between tool measurements and evidence.

## TODO

1. Stabilize `MeasurementValue -> EvidenceItem` conversion.
   - Keep `evidence_item_adapter.py` as the deterministic baseline.
   - Make measurement-name to clinical-target mappings explicit and auditable.
   - Add tests for grouped evidence IDs, clinical targets, evidence types, and provenance.

2. Add LLM/API prompt generation for EvidenceItem creation.
   - Add a prompt builder that receives only typed `MeasurementValue` summaries, not raw EEG or GT report.
   - Require JSON output with fixed fields: `clinical_target`, `evidence_type`, `value`, `measurement_ids`, `time_provenance`, `space_provenance`, `rationale`.
   - Reject unknown clinical targets unless they map to an existing enum or registry entry.
   - Save raw prompt/response into artifacts for audit.
   - Keep deterministic adapter as fallback and comparison baseline.

3. Rebuild `evidence_reviewer.py` around `EvidenceItem`.
   - Review `SharedEvidenceBoard.evidence_items`, not `EvidenceItem`.
   - Emit audit-only `EvidenceItem` records for weak evidence, missing support, and do-not-claim constraints.
   - Avoid creating clinical claims in reviewer output.

4. Rename or demote legacy `EvidenceBoard`.
   - Treat it as a runtime container for measurements, tools, claims, deliberations, and `SharedEvidenceBoard`.
   - Avoid calling it the canonical evidence board in docs/artifacts.
   - Consider a later rename such as `RuntimeEvidenceBundle`.

5. Update downstream audits.
   - Replace evidence item-centric metrics with evidence-centric metrics.
   - Track measurement coverage, evidence-item coverage, claim-plan coverage, surface-decision coverage, and final-prose coverage.

6. Run validation.
   - Restore a project environment with `pydantic` and `pytest`.
   - Run focused tests first, then one smoke `run_session`.
   - Compare generated artifacts before/after the refactor.

7. Discuss iterative tool-parameter calibration for claim confidence.
   - Current tools often emit broad ranges or conservative proxy scores in a single pass.
   - Add a design discussion for iterative parameter selection, where bounded tools can rerun with alternative safe parameters when the first measurement is too broad for a clinically useful claim.
   - Examples: amplitude estimator region/percentile selection, PDR posterior-channel selection, artifact trimming threshold, event localization peak window, morphology window selection.
   - Constraint: tuning may improve MeasurementValue precision, but must not use GT report text at inference time and must not let debug/proxy scores directly surface.
   - Target output: narrower, better-provenanced EvidenceItems that increase AtomicClaimPlan and SurfaceDecision confidence without weakening hard deny rules.

8. Tool v2 measurement-quality track.
   - Stage 3C reportability weighting is deferred until the measurement layer is stronger.
   - PDR v2.1 now uses stable posterior window/channel alpha peaks instead of one averaged posterior PSD peak.
   - Selected50 PDR check improved from 22/47 to 33/47 within 1 Hz of GT PDR.
   - Artifact: `artifacts/tool_v2_1_pdr_selected50_20260612/pdr_v2_1_summary.md`.
   - Amplitude v2.1 preserves the existing traceable range and adds `background_amplitude_typical_uv` for report-suitable exact amplitude.
   - Selected50 amplitude typical check: 30/39 within 20 uV of GT midpoint, while range overlap remains 28/39.
   - Artifact: `artifacts/tool_v2_1_amplitude_selected50_20260612/amplitude_v2_1_summary.md`.
   - Next targets: morphology/localization v2, then report synthesis rendering.
