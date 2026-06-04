# Provenance-Aware EEG Report Error Analysis Protocol

## Purpose
This document defines the evaluation protocol for comparing generated EEG report sections against reference EEG report sections and available structured evidence. It is an evaluation protocol, not an inference prompt and not a clinical replacement workflow.

The protocol is designed to support the project framing:

Long-duration EEG -> structured evidence -> neurologist-facing guidance.

It evaluates whether generated report claims are clinically faithful, section-appropriate, and provenance-supported.

## Scope Boundary
- Reference / GT report text is used only for evaluation.
- GT report text must not be passed to signal modules, parser modules, evidence-board construction, or report generation.
- Raw EEG must not be sent to external APIs.
- LLM-based audit, if used, is a first-pass annotation / error triage tool only.
- Human clinical review or adjudication is required for paper-level claims about clinical correctness.

## Reviewer-Attack Mitigations

### Concern: The rubric favors OURS.
Mitigation:
- Apply the same clinical slots, severity labels, and decision labels to CELM and OURS variants.
- Pre-register the slot schema before reviewing selected qualitative examples.
- Separate general clinical slots from method-specific observed failure examples.
- Report cases where OURS is safer but underpowered, including over-cautious false negatives.

### Concern: GT reports are not absolute truth.
Mitigation:
- Treat GT report as reference interpretation, not oracle truth.
- Use labels such as `reference_consistent`, `reference_contradicted`, and `needs_human_adjudication` when signal provenance is unavailable.
- For ambiguous slots, preserve uncertainty and mark adjudication status.

### Concern: LLM judge cannot replace clinical evaluation.
Mitigation:
- Use any LLM audit only for structured first-pass annotation.
- Require human-audited subset for final manuscript claims.
- Report inter-rater agreement when multiple reviewers are available.

### Concern: Provenance cannot be verified from text alone.
Mitigation:
- Distinguish `reported-provenance consistency` from `patient-specific signal provenance`.
- If only GT and generated text are available, do not claim signal verification.
- If EvidenceBoard artifacts are available, evaluate claim support against measurement/evidence provenance.

## Evaluation Modes

### Mode A: Case-Level Clinical Audit
Input:
- GT/reference EEG section text
- generated report section text
- optional EvidenceBoard
- optional section contract
- text metrics and concept/numeric comparison outputs

Output:
- critical slot table
- claim-level provenance cards
- blocked/revised claims
- severity labels

### Mode B: Cross-Case Failure Taxonomy
Input:
- case audit JSON files
- comparison_long_by_variant.csv
- metric summaries

Output:
- repeated failure patterns by model
- model-specific strengths and failure modes
- metric-clinical mismatch patterns

### Mode C: Method Redesign Translation
Input:
- failure taxonomy only, not GT section text
- current schema/tool registry/module boundaries

Output:
- required tool upgrades
- schema extensions
- claim gates
- ablation plan

### Mode D: Manuscript Summary
Input:
- architecture description
- dataset/split information
- quantitative metrics
- human-audited failure taxonomy summary

Output:
- limitations
- contribution framing
- reviewer-facing defense

## Decision Labels
- `supported_present`: claim matches reference and has adequate evidence.
- `supported_absent`: absence claim is correct and absence evidence is available.
- `unsupported`: claim lacks support from reference or available evidence.
- `contradicted`: claim conflicts with reference or available evidence.
- `under_specified`: directionally plausible but lacks required frequency, amplitude, state, localization, morphology, or protocol details.
- `over_cautious_false_negative`: model avoids a clinically important reference observation by downgrading to vague candidate language.
- `section_contaminated`: claim appears in the wrong section or mixes section semantics.
- `debug_leakage`: internal scores/proxy values appear in clinical prose.
- `possible_leakage_or_memorization`: generated output nearly copies reference text or preserves unusual masked tokens/formatting.
- `needs_human_adjudication`: reference/text/evidence disagreement requires expert review.

## Severity Grades
- `critical`: could materially alter seizure diagnosis, treatment urgency, or major abnormal/normal interpretation.
- `major`: clinically important but less immediately safety-critical; e.g. localization/laterality/morphology errors.
- `moderate`: incomplete quantitation, missing state/protocol details, or section contamination with limited safety impact.
- `minor`: style, concision, or wording issue without material clinical meaning change.
- `debug_only`: affects audit/provenance usability, not clinical report surface text.

## Provenance Levels
- `signal_direct`: direct measurement from patient EEG with time/channel/state/protocol provenance.
- `signal_derived`: derived/proxy measurement from patient EEG; must be caveated if surfaced.
- `metadata_direct`: study metadata or protocol field directly supports claim.
- `reference_text_only`: observed in GT/reference report only; evaluation reference, not signal proof.
- `generated_text_only`: generated claim without support in reference or EvidenceBoard.
- `absent`: no support available.

## Claim Card Schema
```json
{
  "case_id": "...",
  "model": "...",
  "section": "...",
  "claim": "...",
  "claim_type": "...",
  "severity": "critical | major | moderate | minor | debug_only",
  "decision": "supported_present | supported_absent | unsupported | contradicted | under_specified | over_cautious_false_negative | section_contaminated | debug_leakage | possible_leakage_or_memorization | needs_human_adjudication",
  "reference_status": "reference_consistent | reference_contradicted | reference_missing | ambiguous",
  "required_clinical_knowledge": [
    {
      "rule": "...",
      "source_status": "provided | external_needed | assumed_clinical_knowledge"
    }
  ],
  "patient_signal_provenance": [
    {
      "recording_id": "...",
      "time_window": "...",
      "channels_or_regions": ["..."],
      "state": "...",
      "protocol": "...",
      "measurement": {
        "frequency_hz": "...",
        "amplitude_uv": "...",
        "duration_sec": "...",
        "morphology": "...",
        "laterality": "...",
        "localization": "...",
        "reactivity": "...",
        "confidence": "..."
      },
      "evidence_type": "direct | derived | proxy | weak | absent"
    }
  ],
  "negative_provenance": [
    {
      "blocked_claim": "...",
      "reason": "..."
    }
  ],
  "debug_only_evidence": [
    {
      "feature": "...",
      "value": "...",
      "reason_not_surface_text": "..."
    }
  ],
  "recommended_action": "allow | block | caveat | revise | move_to_debug_only | human_adjudication",
  "recommended_revision": "..."
}
```

## Clinical Knowledge Anchors
These are reference anchors for defining clinical concepts. They do not replace site-specific clinical review.

- ACNS Standardized Critical Care EEG Terminology 2021 defines electrographic seizure concepts using frequency/evolution/duration criteria for critical care EEG terminology.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC8135051/
- IFCN/ILAE routine and sleep EEG minimum recording standards discuss routine/sleep EEG recording practice and activation procedures.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10006292/

When a rule is not covered by provided sources, mark `source_status=external_needed` or `assumed_clinical_knowledge`; do not fabricate citations.

## Dataset-Specific vs General Rules
General rules define claim requirements, e.g. PDR requires posterior/occipital alpha-range evidence and state/reactivity context when available.

Dataset-specific observations are diagnostic examples from the S0001 selected50 run, e.g. repeated 0.5 Hz boundary-frequency artifacts or F3/F7 localization errors. These must not be hard-coded as universal facts.

## Minimal Human Review Plan
For a defensible manuscript subset:
- Select cases stratified by metric performance and clinical slot type.
- Blind reviewers to model identity when feasible.
- Use the same slot/rubric for all models.
- Grade severity and decision labels per claim.
- Report agreement or adjudication counts if multiple reviewers are available.
- Use LLM audits only as pre-annotation, not final adjudication.
