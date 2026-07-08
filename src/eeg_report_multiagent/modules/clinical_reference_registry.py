from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from eeg_report_multiagent.schemas.clinical_reference import ClinicalReferenceItem


EEGAGENT_RAG_DOC_ROOT = "EEGAgent/RAG/docs"


CLINICAL_REFERENCE_ITEMS: dict[str, ClinicalReferenceItem] = {
    "acns_posterior_alpha_pdr": ClinicalReferenceItem(
        reference_id="acns_posterior_alpha_pdr",
        concept="posterior alpha / posterior dominant rhythm",
        short_rule=(
            "A PDR/alpha rhythm should be an 8-13 Hz posterior wakefulness rhythm, "
            "best interpreted with eye-closure/opening reactivity context."
        ),
        source_name="EEGAgent RAG ACNS/glossary corpus",
        source_path=f"{EEGAGENT_RAG_DOC_ROOT}/glossary_clean.txt",
        applicable_targets=["pdr"],
        applicable_claim_types=["pdr", "posterior_alpha", "posterior_dominant_rhythm"],
        notes="Derived from EEGAgent RAG chunks on posterior dominant rhythm and alpha rhythm.",
    ),
    "ifcn_background_report_structure": ClinicalReferenceItem(
        reference_id="ifcn_background_report_structure",
        concept="background activity reporting",
        short_rule=(
            "Background reporting should preserve supported rhythm, frequency, amplitude, "
            "symmetry, reactivity, organization, and slowing as distinct report elements."
        ),
        source_name="EEGAgent RAG IFCN EEG reporting guideline corpus",
        source_path=f"{EEGAGENT_RAG_DOC_ROOT}/Guideline7-GuidelinesforEEGReporting_v1.pdf",
        applicable_targets=[
            "background_amplitude",
            "background_slowing",
            "excess_beta",
            "pdr",
        ],
        applicable_claim_types=[
            "background",
            "background_amplitude",
            "background_amplitude_range",
            "background_slowing",
            "excess_beta",
            "pdr",
        ],
        notes="Used as report-structure provenance, not as patient evidence.",
    ),
    "acns_activation_protocol": ClinicalReferenceItem(
        reference_id="acns_activation_protocol",
        concept="activation procedures",
        short_rule=(
            "Activation procedures such as hyperventilation and intermittent photic stimulation "
            "are protocol/status information and should be preserved when known."
        ),
        source_name="EEGAgent RAG ACNS/glossary corpus",
        source_path=f"{EEGAGENT_RAG_DOC_ROOT}/glossary_clean.txt",
        applicable_targets=["protocol"],
        applicable_claim_types=["protocol", "activation", "photic", "hyperventilation"],
        notes="Supports rendering performed/not-performed protocol status without using GT text.",
    ),
    "acns_state_sleep_context": ClinicalReferenceItem(
        reference_id="acns_state_sleep_context",
        concept="state and sleep context",
        short_rule=(
            "Wakefulness, drowsiness, sleep, and sleep architecture are state-context findings; "
            "absence or presence should not be inferred without structured support."
        ),
        source_name="EEGAgent RAG ACNS/glossary/reporting corpus",
        source_path=f"{EEGAGENT_RAG_DOC_ROOT}/glossary_clean.txt",
        applicable_targets=["state", "context"],
        applicable_claim_types=["state", "sleep", "drowsiness", "stage_ii_sleep", "context"],
        notes="Keeps state interpretation separate from background and event morphology.",
    ),
    "acns_spike_sharp_wave_morphology": ClinicalReferenceItem(
        reference_id="acns_spike_sharp_wave_morphology",
        concept="epileptiform morphology",
        short_rule=(
            "Spike/sharp/spike-wave wording requires morphology and context support; "
            "event burden or score alone should not become a definitive epileptiform claim."
        ),
        source_name="EEGAgent RAG ACNS terminology corpus",
        source_path=f"{EEGAGENT_RAG_DOC_ROOT}/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf",
        applicable_targets=["epileptiform_morphology", "event_candidate"],
        applicable_claim_types=[
            "epileptiform",
            "epileptiform_morphology",
            "sharp",
            "spike",
            "spike_wave",
            "event_candidate",
        ],
        notes="Supports caveated morphology interpretation, not seizure diagnosis.",
    ),
    "acns_spatial_field_localization": ClinicalReferenceItem(
        reference_id="acns_spatial_field_localization",
        concept="spatial field and localization",
        short_rule=(
            "Localization/laterality should be supported by spatial channel, region, field, "
            "or electrode-maxima evidence; ratio-only localization is insufficient."
        ),
        source_name="EEGAgent RAG ACNS terminology corpus",
        source_path=f"{EEGAGENT_RAG_DOC_ROOT}/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf",
        applicable_targets=["localization"],
        applicable_claim_types=[
            "localization",
            "laterality",
            "field",
            "region",
            "electrode_maxima",
        ],
        notes="Keeps spatial provenance explicit for traceable report wording.",
    ),
    "multiti_seizure_hard_gate": ClinicalReferenceItem(
        reference_id="multiti_seizure_hard_gate",
        concept="seizure claim safety gate",
        short_rule=(
            "Seizure presence or absence requires seizure-specific evidence or validated seizure "
            "metadata; generic event candidates are not sufficient."
        ),
        source_name="MultiTI hard-deny safety contract with EEGAgent ACNS terminology context",
        source_path=f"{EEGAGENT_RAG_DOC_ROOT}/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf",
        applicable_targets=["seizure_evidence"],
        applicable_claim_types=["seizure", "seizure_absent", "seizure_present", "electrographic_seizure"],
        notes="Project safety rule; the reference only documents terminology context.",
    ),
}


def clinical_reference_ids_for_claim(
    claim_type: str | None,
    clinical_targets: Iterable[str] | None = None,
) -> List[str]:
    """Return deterministic reference IDs relevant to a claim and targets."""

    claim_key = str(claim_type or "").lower()
    targets = {str(target).lower() for target in clinical_targets or [] if target}
    out: list[str] = []
    for reference_id, item in CLINICAL_REFERENCE_ITEMS.items():
        claim_match = any(key and key in claim_key for key in item.applicable_claim_types)
        target_match = bool(targets & {target.lower() for target in item.applicable_targets})
        if claim_match or target_match:
            out.append(reference_id)
    return sorted(dict.fromkeys(out))


def clinical_reference_payloads(reference_ids: Sequence[str]) -> List[dict[str, Any]]:
    """Compact prompt-safe reference view.

    The payload intentionally includes short rules and source paths only; it
    does not include raw guideline chunks or long paper text.
    """

    payloads: list[dict[str, Any]] = []
    for reference_id in reference_ids:
        item = CLINICAL_REFERENCE_ITEMS.get(reference_id)
        if item is None:
            continue
        payloads.append(
            {
                "reference_id": item.reference_id,
                "concept": item.concept,
                "short_rule": item.short_rule,
                "source_name": item.source_name,
                "source_path": item.source_path,
                "source_kind": item.source_kind,
            }
        )
    return payloads
