# Decision Log

## 2026-04-27
- Keep 10-second window as storage unit.
Reason: preserves preprocessing compatibility with CELM/HEEDB and allows deterministic time provenance.

- Use 2-pass processing (scout -> focused).
Reason: supports long recordings with sparse events while retaining recording-level background coverage.

- Separate module vs agent.
Reason: modules compute deterministic signal/parser outputs; agents only control bounded tool selection.

- No external API for raw EEG interpretation or GT comparison.
Reason: privacy/safety and reproducible local evidence traceability.

- Ban free-form strings in intermediate signal layers.
Reason: enforce typed measurement/finding structure and prevent hidden hallucinated semantics.

- Core vs optional.
Core: background, event, parser, evidence board, synthesizer.
Optional: claim verifier/evaluation hooks.

- Separate inference context from GT reports.
Reason: GT report text is supervision/evaluation material, not an inference input. Smoke runners pass GT report paths only as evaluation bookkeeping.

- Treat report text artifacts as renderings.
Reason: the scientific object is structured evidence with provenance; `detail.txt` and `impression.txt` are neurologist-facing views over that evidence.

- Allow optional LLM evidence review only after evidence board assembly.
Reason: API use is limited to structured evidence gap review and local tool proposal generation; raw EEG and GT report text remain outside the external API boundary.

## 2026-05-04
- Use Rule+LLM Evidence Review as the first LLM-enhanced method path.
Reason: this avoids expanding a brittle handcrafted rule forest while preserving the core constraint that raw EEG interpretation stays in local signal tools. The LLM receives only structured measurements/findings/tool registry summaries and returns typed weak-evidence, missing-slot, do-not-claim, claim-constraint, and bounded tool-proposal records.

- Treat LLM finding proposals and local EEG encoders as later variants/ablations.
Reason: LLM structured finding proposal requires stricter schema/provenance validators, and a single local EEG encoder should be evaluated as an assistive signal tool rather than assumed to solve all clinical slots.

## 2026-05-05
- Define method D as EvidenceBoard-only LLM report synthesis.
Reason: D tests whether low lexical metrics are caused by report wording/organization rather than signal evidence quality. The LLM receives typed EvidenceBoard summaries and target section names only; it does not receive raw EEG, source pkl payloads, or GT report text.

- Keep method E as a bounded local signal-tool variant, not an end-to-end encoder-to-report model.
Reason: E should strengthen signal-side evidence while preserving the measurement -> finding -> evidence board -> report contract. The initial E implementation adds a focused-pass `morphology_feature_encoder` proxy tool that emits typed morphology-support and field-concentration measurements with time/channel provenance.

- Do not run full E over selected50 without addressing pkl I/O cost.
Reason: D can reuse existing EvidenceBoard artifacts, but E requires reading processed EEG pkl windows. Row-level monitoring showed the main E runtime bottleneck is `Load Inputs` from many `seg_*.pkl` files on external storage, not the focused encoder computation itself.

- Treat CELM target section names as an inference contract, but never GT section text.
Reason: CELM generation/evaluation is section-wise over `extracted_eeg_section_names`; generated outputs must include the requested section names. The section names are benchmark metadata, while reference section text remains evaluation-only.

- Add one-pass section-aware quality floor before iterative/memory variants.
Reason: B failed partly because event-candidate evidence was routed into seizure-oriented sections and because normal-background slots such as PDR/reactivity/organization were not explicitly represented as unavailable. The quality floor routes evidence by target section role and separates transient candidates from seizure confirmation.

- Split global dominant frequency from posterior dominant rhythm candidate.
Reason: a global PSD argmax at 0.5 Hz should not be verbalized as PDR or definitive background rhythm. PDR is now a separate posterior alpha candidate measurement with its own confidence and support flag.

- Add conservative local event-type separation before seizure language.
Reason: event-candidate burden, epileptiform-candidate likelihood, and electrographic seizure likelihood are distinct clinical claims. The one-pass local classifier keeps seizure likelihood low unless sustained candidate runs, burden, rhythmicity, and score prominence jointly support it.

- Treat LLM measurement-to-finding mapping as an optional ablation, not core inference.
Reason: the LLM may propose structured finding labels from typed measurements, but proposals must link to existing measurement IDs, use an allowlist, and remain separate from raw EEG interpretation or GT report comparison.

## 2026-05-07
- Treat the long clinical audit prompt as a master evaluation specification, not as a generation prompt.
Reason: using GT-derived critique directly for report synthesis risks benchmark overfitting. The protocol is now split into case-level audit, cross-case taxonomy, method redesign, and manuscript summary modes.

- Add blinded/severity-aware clinical audit requirements.
Reason: a provenance-aware rubric can look favorable to OURS unless the same slots, severity labels, and adjudication process are applied to CELM and OURS. Human-audited subsets and inter-rater/adjudication reporting are needed for manuscript-level claims.

- Separate general clinical slot schema from selected50 observed failure examples.
Reason: S0001 examples such as 0.5 Hz boundary peaks, F3/F7 localization, and specific amplitude/duration values are useful diagnostics but should not become dataset-specific hard-coded method logic.

## 2026-05-08
- Apply debug/surface separation before further report-style optimization.
Reason: selected50 clinical provenance audit showed that `Our_Upgrade_LLMProp` often exposed internal detector/proxy language as clinical prose. The updated synthesizer keeps candidate burden, likelihood/support scores, field concentration, slowing index, beta ratio, and organization proxy terms in provenance/debug artifacts rather than final section text.

- Add a numeric provenance filter to report synthesis.
Reason: numeric values are clinically meaningful only when tied to supported claims. Debug-like values such as candidate train duration, internal ratios, and unsupported boundary-frequency evidence should not be surfaced as formal EEG report quantitation.

- Extend local evidence tools without changing the external-API boundary.
Reason: PDR topography/symmetry, morphology proxy class, coarse localization normalization, and drowsy/sleep/protocol parsing improve structured evidence coverage while preserving the rule that raw EEG and GT reports are not sent to external APIs for interpretation.

- Treat filtered selected50 outputs as a safety-oriented method variant, not a final performance claim.
Reason: `Our_Upgrade_Filtered` reduced debug leakage to zero by regex audit on refreshed selected50 report JSONs, but lexical/concept metrics decreased relative to the leakage-prone version. This is expected because unsafe internal details were removed; clinical adequacy still requires morphology/PDR/localization tool upgrades and human adjudication.

- Create a human review subset from audit strata.
Reason: reviewer-facing clinical claims require human adjudication. The subset is selected from metric-clinical mismatch, high critical burden, pre-filter debug leakage, numeric provenance issues, morphology/localization failures, and forced anchor cases rather than cherry-picked wins.

- Keep CELM-style metric evaluation in a separate eval-enabled Docker image.
Reason: `evaluate`, ROUGE, METEOR, and related metric dependencies are not needed for core inference and make the runtime image heavier. The project now supports `INSTALL_EVAL_DEPS=1` to build `eeg-report-multiagent-v1:eval` for exact CELM-style metric recomputation while preserving a lean default image.

- Keep BERTScore dependencies optional even inside the eval image.
Reason: full BERTScore recomputation pulls heavier transformer/torch dependencies and model downloads. The project now supports `INSTALL_EVAL_DEPS=1 INSTALL_BERTSCORE_DEPS=1` when full CELM-style metric parity is required.

## 2026-05-11
- Treat CELM target sections as the public report contract.
Reason: CELM reads `extracted_eeg_section_names`, generates only those sections in strict JSON, and evaluates section-wise against the matching reference sections. OURS should keep richer ontology/claim structures internal and render only CELM-compatible target sections for fair comparison.

- Add event-peak-centered localization v2.
Reason: 10-second mean channel energy is too coarse for clinical topography. The new `event_peak_topography_localizer` selects event-like peaks inside focused windows, summarizes the peak-centered channel field, and emits typed localization, laterality, and field-concentration measurements with window/channel provenance. It remains a proxy, not a definitive clinical localization claim.

- Add `AtomicClaimPlan` before report-surface claims.
Reason: keyword rules, LLM review, and local encoders should create typed evidence only. Clinical claims are now planned in `ReportSynthesizer.build_atomic_claim_plan`, where each proposed sentence is marked `allow`, `caveat`, `block`, or `debug_only` before becoming a `ClaimRecord` or report text.

- Save claim plans as first-class run artifacts.
Reason: `atomic_claim_plan.json` and `inference_trace.report_synthesis.atomic_claim_plan` make it possible to audit which findings were surfaced, caveated, blocked, or retained as debug-only evidence.

- Run localization v2 + atomic claim planning over selected50 as a local-tool validation batch.
Reason: selected50 is a fixed case subset, while `32 passed` refers only to unit tests. The selected50 batch `batch_s0001_test_localization_v2_atomic_claim_selected50` completed 50/50 rows with no errors, invoked `event_peak_topography_localizer` in all rows, and wrote `atomic_claim_plan.json` in all row artifacts.

- Treat localization v2 as provenance/debug evidence until stricter surface gating is added.
Reason: selected50 GT-reference audit showed v2 produces a localization proxy for all rows and current report synthesis surfaces localization language in all 50 rows. Against refined GT abnormal-localization slots, 24/50 rows had a GT localization mention, but 16 rows had GT absence and 10 rows had no abnormal-localization mention. This means v2 is useful for topographic provenance, but report-surface localization should require event/morphology support or remain debug-only.

- Gate localization v2 report-surface text by section role and multi-proxy event support.
Reason: localization should not be surfaced from generic detail sections or from peak topography alone. The updated claim gate keeps `event_peak_localization` debug-only by default and allows surface text only in epileptiform/events sections when candidate burden, peak-field concentration, epileptiform likelihood, and morphology support are jointly present. Re-rendering selected50 reduced localization surface mentions from 50/50 to 5/50, reduced unsupported GT-not-mentioned surface cases from 10 to 0, and reduced GT-absent false-positive surface cases from 16 to 2.

## 2026-05-13
- Unify report-surface policy behind `SurfacePolicy` and `AtomicClaimPlan`.
Reason: Stage 0 leakage audit found that CELM-compatible synthesis and LLM synthesis could still verbalize Measurement/Finding objects or raw reviewer constraints directly. Final clinical prose now must come from `allow`/`caveat` atomic claim plans or deterministic safe fallbacks; `block` and `debug_only` entries remain available for audit/provenance only.

- Treat event localization, candidate burden, train duration, laterality ratios, morphology screens, likelihood/support scores, and field-concentration ratios as debug/proxy evidence by default.
Reason: these values may guide future claim gating, but they are not clinical report language by themselves. This prevents proxy features from becoming unsupported epileptiform, seizure, localization, or impression-level abnormalities.

- Restrict LLM report synthesis to surface-approved atomic claim plans.
Reason: method D can still use an LLM for wording/organization, but the LLM should not receive full measurements, findings, values previews, debug scores, or raw evidence-review text. This keeps external API usage downstream of the structured evidence gate and prevents raw/proxy evidence leakage.

## 2026-05-14
- Add `EvidenceItem` and `SharedEvidenceBoard` as the typed evidence layer before `AtomicClaimPlan`.
Reason: measurements and findings are necessary but too close to raw tool outputs for report-surface governance. Stage 1 introduces a queryable evidence layer with evidence type, clinical target, reportability, time/space provenance, and measurement/finding links before any claim planning occurs.

- Convert Measurement/Finding objects into conservative EvidenceItems before report synthesis.
Reason: proxy values such as candidate burden, localization ratios, field concentration, morphology support, and likelihood scores should be preserved for provenance and future gating while remaining `debug_only` by default. Reportable numeric values now need an EvidenceItem with unit, clinical target, section policy, and allow/caveat reportability.

- Link `AtomicClaimPlan` entries to SharedEvidenceBoard evidence IDs.
Reason: report-surface claims should be traceable to evidence items, not only to raw measurement/finding IDs. This supports future claim verification, clinical audit, and human-review tooling without weakening the Stage 0.5 surface policy.

- Add final-prose audit after report synthesis.
Reason: Stage 1 makes evidence traceable before claim planning, but final output still needs an independent check that numeric values, section placement, seizure language, and debug/proxy phrases actually obey the EvidenceItem and AtomicClaimPlan links. Stage 2 therefore writes `final_prose_audit.json` and reports UnsupportedNumericRate, NumericProvenanceAccuracy, ClaimTraceCoverage, debug leakage, section leakage, and seizure-gate violations.

- Treat final-prose audit as a warning/evaluation layer, not a detector or wording upgrade.
Reason: this stage should not add EEG detectors, train models, or improve report richness. It only verifies whether the already generated clinical prose is safe, traceable, and free of internal artifact leakage. Controlled tests fail on high-risk violations, while normal CLI runs emit audit artifacts and warnings for later hard-fail configuration.

- Add Stage 2.5 batch final-prose audit with separate text-only and full-trace modes.
Reason: CELM and older OURS outputs do not carry SharedEvidenceBoard/AtomicClaimPlan traces, so they should be evaluated for surface safety without being penalized for missing OURS-specific trace objects. Full traceability metrics are reported only when local evidence and claim-plan artifacts exist.

- Use batch audit as failure-pattern discovery, not method optimization.
Reason: selected50 final-prose audit aggregates debug leakage, unsupported numeric heuristics, section leakage, seizure-gate pressure, and trace coverage. These metrics identify which failure buckets should drive Stage 3, but they do not change report wording, SurfacePolicy, detectors, or evidence weighting.

- Regenerate selected50 as `Our_EvidenceGated_v1` using the latest Stage 0.5/1/2 workflow.
Reason: `Our_Upgrade_LLMProp` was produced before unified SurfacePolicy, SharedEvidenceBoard, and FinalProseAuditor were fully integrated. A fresh variant is required to distinguish current evidence-gated behavior from older leakage-prone outputs.

- Fix final-prose numeric audit rounding for reportable range evidence.
Reason: regenerated selected50 exposed a false-positive audit path where rounded amplitude text such as `0.0-80 uV` failed to match `0.0-79.8 uV` EvidenceItems and instead matched unrelated low-frequency values by value overlap. The auditor now allows clinically harmless display rounding while preserving unit/reportability checks.

- Add Stage 2.75 evidence flow and gate-loss audit before changing detectors or SurfacePolicy.
Reason: `Our_EvidenceGated_v1` is safe but clinically under-informative. The new audit traces each clinical slot from Measurement/Finding through EvidenceItem, AtomicClaimPlan, and final prose so we can distinguish evidence absence, conservative reportability, claim planning failure, and true SurfacePolicy over-suppression.

- Treat Stage 3C as the current data-supported next focus, with targeted Stage 3A repairs for missing event amplitude/frequency, electrode maxima, and push-button metadata.
Reason: selected50 evidence-flow audit shows most slots already have measurements, EvidenceItems, and AtomicClaimPlans, but many are blocked/debug-only due to reportability, morphology/state/protocol support, or internal-score suppression. Some slots remain absent and need extraction work, but the dominant bottleneck is evidence classification/weighting rather than raw loader failure.

- Add minimal Stage 3C reportability calibration before claim planning.
Reason: Stage 2.75 showed that useful evidence often reached EvidenceItem and AtomicClaimPlan but was blocked as `numeric_not_reportable`, `proxy_or_debug_only`, or `surface_policy_rejected`. The new calibrator is deliberately narrow: it can only produce allow/caveat decisions for clinically bounded slots such as metadata/status, posterior-alpha PDR candidates with posterior provenance, and background amplitude, while keeping event burden, duration, scores, ratios, seizure candidates, and debug evidence out of final prose.

- Preserve original EvidenceItems and add explicit calibrated evidence copies for surfaced Stage 3C claims.
Reason: silently mutating EvidenceItem reportability would obscure provenance. Stage 3C therefore creates `cal_*` EvidenceItems only when a safe calibration override is used, with the original evidence ID recorded in debug metadata. LLM-assisted reviewer records remain audit-only and are not converted into calibrated clinical provenance.

- Keep FinalProseAuditor as the post-synthesis safety arbiter after calibration.
Reason: calibration is not a bypass around SurfacePolicy. Selected50 rerendering after Stage 3C preserved zero debug leaks, zero unsupported numerics, zero section leakage, zero seizure-gate violations, and full trace coverage while modestly increasing slot surface rate and reducing useful-suppressed rate.
