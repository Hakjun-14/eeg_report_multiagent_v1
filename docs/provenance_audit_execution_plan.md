# Provenance Audit Execution Plan

## Goal
Turn the current GT-vs-generated selected50 artifacts into a defensible clinical error analysis without converting the benchmark reference text into a generation-time crutch.

## Current Inputs
- Comparison root: `artifacts/gt_generated_comparison_selected50_UpgradeLLMProp/`
- Filtered comparison root: `artifacts/gt_generated_comparison_selected50_UpgradeFiltered/`
- Diagnostic markdown cases:
  - row 189: low CELM / difficult case
  - row 548: mid CELM / PDR and event-gating diagnostic case
  - row 783: high CELM / possible metric-clinical mismatch or leakage diagnostic case
- Variant score ledger: `artifacts/experiment_ledgers/S0001_test_CELM_vs_B_vs_D_vs_BQFv2_vs_UpgradeLLMProp_selected50_scores.csv`
- Structured evidence, when available: per-row `evidence_board.json`, `section_contract_audit.json`, and `method_audit.json`

## 2026-05-08 Execution Status

Completed through human-review subset preparation:
- Debug/surface separation and numeric provenance filter were applied in `ReportSynthesizer`.
- PDR candidate evidence was extended with posterior/anterior support and posterior alpha symmetry evidence.
- Event evidence was extended with morphology screen class and coarse localization screen outputs.
- State/protocol parser now extracts drowsy/sleep status and section-style photic/hyperventilation status patterns.
- Existing selected50 `Our_Upgrade_LLMProp` EvidenceBoards were refreshed into `Our_Upgrade_Filtered` CELM-compatible generated reports.
- Local GT/generated comparison artifacts were rebuilt for CELM, `Our_Upgrade_LLMProp`, and `Our_Upgrade_Filtered`.
- Clinical provenance audit was rerun over selected50 for the three variants.
- A 12-case human-review subset was selected using pre-specified strata.

Key outputs:
- Filtered generated reports: `artifacts/batch_s0001_test_onepass_filtered_selected50/celm_results/generated_reports_json/`
- Filtered comparison root: `artifacts/gt_generated_comparison_selected50_UpgradeFiltered/`
- Filtered clinical audit: `artifacts/clinical_provenance_audit_selected50_debug_numeric_filtered/`
- Human-review subset: `artifacts/human_review_subset_selected50_debug_numeric_filtered/`

Selected50 audit deltas:
- `Our_Upgrade_LLMProp`: 1373 claim cards, 547 debug-leakage cards, mean concept F1 0.535.
- `Our_Upgrade_Filtered`: 823 claim cards, 0 debug-leakage cards, mean concept F1 0.491.
- Regex surface audit: old generated JSONs had 438 debug/proxy phrase hits across 50/50 files; filtered JSONs had 0 hits across 48 refreshed files.

Interpretation:
- The filter improves clinical surface safety and provenance discipline.
- It does not solve missing PDR, morphology, localization, or state/protocol evidence; those remain the next tool-level upgrades.
- Lower lexical/concept score after filtering should be interpreted as removal of unsupported internal/debug text, not necessarily worse clinical safety.

## Execution Phases

### Phase 1: Case-Level Audit Pilot
Run claim-level clinical/provenance audit for rows 189, 548, and 783.

Output per case:
- critical slot table
- generated claim cards
- blocked / caveated / revised claims
- metric-vs-clinical-correctness notes
- severity labels

Decision rule:
- If the audit cannot verify signal truth from EvidenceBoard, label the claim as `reference_text_only` or `needs_human_adjudication` rather than treating GT as absolute truth.

### Phase 2: Rubric Calibration
Review whether the slot schema and failure labels are too favorable to OURS.

Checks:
- Does the rubric penalize OURS over-cautious false negatives?
- Does the rubric penalize CELM unsupported or contradicted claims?
- Are severity labels symmetric across models?
- Are style-only errors separated from clinically critical errors?

### Phase 3: selected50 Structured Error Table
Apply the calibrated rubric to all selected50 cases with local first-pass automation.

Output:
- `clinical_audit_long.csv`: one claim/slot/model row
- `clinical_audit_summary_by_variant.csv`
- `failure_taxonomy_counts.json`

### Phase 4: Human Review Subset
Select a blinded subset for neurologist or trained clinical reviewer adjudication.

Suggested strata:
- high metric / clinically suspicious
- low metric / clinically safer
- seizure or epileptiform cases
- activation protocol cases
- possible leakage or memorization cases

Current subset:
- `artifacts/human_review_subset_selected50_debug_numeric_filtered/human_review_subset.csv`
- `artifacts/human_review_subset_selected50_debug_numeric_filtered/review_cases/`
- Forced anchor rows: 189, 548, 783.
- Additional strata: high CELM metric with errors, highest filtered critical burden, old debug leakage, numeric provenance issue, morphology failure, localization failure, low filtered concept F1.

### Phase 5: Method Redesign Translation
Use failure counts, not raw GT text, to update the method.

Priority upgrades:
1. PDR detector
2. Epileptiform morphology detector
3. Localization normalizer
4. State/protocol parser
5. Numeric provenance filter
6. Section-specific claim gate
7. Debug/surface separation
8. Leakage audit

## Non-Goals
- Do not use GT wording as synthesis templates.
- Do not make LLM judge the final clinical evaluator.
- Do not claim signal truth when only report text was compared.
- Do not tune solely to ROUGE/METEOR.

## Immediate Next Command-Level Target
Next command-level targets:
1. Run human review on the 12-case subset using the same claim-card rubric for CELM and OURS variants.
2. Convert adjudicated errors into tool requirements for PDR, morphology, localization, and state/protocol upgrades.
3. Add critical-slot metrics that are separate from ROUGE/METEOR.
4. Re-run selected50 after each tool upgrade and keep `Our_Upgrade_Filtered` as the safety baseline.

## 2026-05-11 Implementation Update

CELM contract check:
- Public generated reports should preserve the CELM section contract: generate only requested section names, in strict JSON, with section-wise evaluation.
- Internal ontology, measurements, evidence items, and atomic claims are allowed as audit artifacts, not as extra public report sections.

Localization v2:
- Added `event_peak_topography_localizer`.
- It replaces the event agent's default localization choice while leaving the legacy mean-energy normalizer available in the registry for ablation.
- The v2 tool uses focused suspicious windows, identifies peak samples, summarizes the peak-centered topographic field, and stores top channels, active channels, peak windows, and peak samples in measurement metadata.

Claim planning:
- Added `AtomicClaimPlan` with `allow`, `caveat`, `block`, and `debug_only` surface actions.
- Added `atomic_claim_plan.json` to single-run artifacts.
- Current design keeps clinical report sections CELM-compatible while allowing claim-level provenance audit downstream.

Validation:
- Unit tests passed in the eval Docker image: `32 passed, 1 warning`.
- One smoke run confirmed `event_peak_topography_localizer` was invoked and `atomic_claim_plan.json` was written:
  `artifacts/localization_v2_atomic_claim_smoke/`.
