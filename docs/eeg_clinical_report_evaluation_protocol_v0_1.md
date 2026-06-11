# EEG Clinical Report Evaluation Protocol v0.1

## Purpose
Evaluate EEG report generation systems beyond lexical similarity. This protocol separates text overlap, GT claim recovery, safety, traceability, and EEG-specific clinical fidelity.

## Evaluation Inputs
- Ground-truth EEG report sections are used only for evaluation.
- Generated report sections from CELM-style and EvidenceGated variants.
- Optional structured evidence artifacts: measurements, evidence board, atomic claims, surface decisions, and final-prose audit.
- Optional perturbation condition metadata for signal reliance experiments: real, zero, white noise, time shuffle, channel shuffle, and cross-patient swap.

## 1. Text Metrics
Use text metrics as reporting-quality and reference-overlap indicators, not as clinical correctness by themselves.

- ROUGE-L: long common subsequence overlap with GT report text.
- METEOR: lexical overlap with some stemming/synonym tolerance.
- BERTScore: semantic similarity proxy.
- BLEU: optional, mainly for n-gram precision.

Interpretation rule: high text score does not prove signal use, provenance, or clinical correctness.

## 2. GT Atomic Claim Metrics
GT reports are decomposed into atomic claims, then generated claims are compared against them.

- GT atomic claim recall: fraction of GT clinical claims recovered in generated report.
- Generated claim precision: fraction of generated clinical claims supported by GT and/or admissible evidence.
- Missing GT claim rate: GT claims absent from generated report.
- Extra claim rate: generated claims not present in GT or not supported by available evidence.
- Numeric claim precision: generated numeric values that match clinically reportable GT/evidence values.

Important: GT absence is not proof that a claim is clinically false; it is only a reference-report mismatch.

## 3. Safety Metrics
These detect clinically risky output behavior.

- Unsupported numeric rate: numeric values in final prose without reportable provenance.
- Section leakage rate: claims placed in clinically wrong sections.
- Seizure gate violation rate: seizure claim without seizure-specific evidence or validated seizure metadata.
- Debug/proxy leakage rate: exposure of candidate burden, support score, likelihood score, field concentration ratio, laterality index, train duration, or raw reviewer/debug text.
- PDR validity violation rate: global/boundary/low-frequency values, such as 0.5 Hz, surfaced as PDR.

## 4. Traceability Metrics
These apply to systems that expose evidence and claim artifacts.

- ClaimTraceCoverage: final clinical sentences matched to allowed/caveated atomic claims.
- EvidenceLinkedClaimRate: matched clinical sentences with at least one linked evidence item.
- NumericProvenanceAccuracy: numeric mentions matched to reportable evidence with valid units and section compatibility.
- SurfaceDecisionPassRate: final prose generated only from allow/caveat decisions or deterministic safe fallbacks.

CELM-style baselines without evidence boards should not be penalized on traceability metrics directly; they should be reported as traceability-unavailable.

## 5. EEG-Specific Clinical Metrics
These are the core clinical dimensions to report by slot.

- State fidelity: awake, drowsy, sleep, and stage II architecture preserved correctly.
- Topographic fidelity: generalized, focal, frontal, temporal, posterior, and regional fields preserved.
- Laterality preservation: left/right/asymmetric observations preserved without collapsing subtle asymmetry.
- Electrode maxima accuracy: maxima such as F3/F7 or F7/T3/T5 preserved when present.
- Uncertainty preservation: caveats such as not definitively epileptiform or possible sleep architecture mimic retained.
- Seizure/interictal consistency: interictal transients not converted into seizures; no-seizure statements preserved when supported.
- PDR validity: PDR requires posterior/occipital alpha-range rhythm and relevant state/topography support; global/boundary frequency is not PDR.
- Activation protocol fidelity: photic and hyperventilation status/response not invented or inverted.

## 6. Signal Reliance Sanity Check
Run each baseline on identical report rows under EEG perturbations.

Conditions:
- Real EEG
- Zero EEG
- White noise EEG
- Time-shuffled EEG
- Channel-shuffled EEG
- Cross-patient swapped EEG

Metrics:
- Text metrics versus GT for each condition.
- Generated report hash and section text comparison across conditions.
- GT atomic claim recall and generated claim precision.
- Numeric claim precision.

Interpretation:
- If random/zero/shuffled EEG preserves nearly identical reports and metrics, the model likely relies heavily on report/history/language prior or has low signal sensitivity.
- If performance materially degrades under random/zero/shuffled EEG, the model is more likely using EEG signal information.
- Hash identity across perturbations is strong evidence of signal insensitivity or a pipeline bug; confirm with PKL hashes, dataloader tensors, EEG token embeddings, and logits.

## 7. LLM-as-Structured-Auditor Policy
LLM-as-a-judge is secondary. It should not replace clinical expert evaluation.

Allowed use:
- Extract candidate claims.
- Label missing/extra/unsupported/contradictory claims using a fixed rubric.
- Generate audit tables for human review.

Disallowed use:
- Claim that the LLM is a clinical ground-truth judge.
- Use GT report as inference input.
- Let LLM override hard safety gates.

## 8. Reporting Format
For each variant, report:
- Text metric table.
- GT claim metric table.
- Safety metric table.
- Traceability metric table if available.
- Three case-level clinical comparisons: row 189, row 548, row 783.
- Signal reliance sanity check table for real/noise/shuffle/zero conditions.

## 9. Decision Rules
- If CELM degrades strongly under zero/noise/shuffle: treat it as a strong signal-using baseline and differentiate EvidenceGated by safety/traceability.
- If CELM remains similar under zero/noise/shuffle: discuss language/report-prior dependence and strengthen signal-reliance evaluation.
- If GT-required claims exist upstream but are suppressed downstream: Stage 3C/3E calibration is justified.
- If GT-required claims are absent at measurement stage: detector/evidence extraction should be prioritized before reportability tuning.
