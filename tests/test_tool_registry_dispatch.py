import numpy as np

from eeg_report_multiagent.agents.background_agent import BackgroundAgent
from eeg_report_multiagent.agents.event_agent import EventAgent
from eeg_report_multiagent.modules.event_module import EventModule
from eeg_report_multiagent.schemas.measurement import MeasurementRole
from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.report import ClaimSurfaceAction
from eeg_report_multiagent.schemas.shared_evidence import ClinicalTarget, EvidenceType
from eeg_report_multiagent.modules.evidence_item_adapter import build_shared_evidence_board
from eeg_report_multiagent.tools.registry import build_background_registry, build_event_registry


def test_background_registry_dispatch() -> None:
    reg = build_background_registry()
    assert "amplitude_summary" in reg.list_tools()
    assert "posterior_dominant_rhythm_candidate" in reg.list_tools()
    assert "posterior_dominant_rhythm_spectral_v2" in reg.list_tools()

    signal = np.random.randn(4, 22, 2000).astype("float32")
    output, rec = reg.dispatch("amplitude_summary", signal_nct=signal, source_ref="s")

    assert rec.status == "ok"
    assert isinstance(output, list)
    assert any(isinstance(x, MeasurementValue) for x in output)


def test_background_agent_uses_pdr_v2_not_legacy_v1() -> None:
    tools = BackgroundAgent().select_tools({})

    assert "posterior_dominant_rhythm_spectral_v2" in tools
    assert "posterior_dominant_rhythm_candidate" not in tools


def test_background_amplitude_summary_is_clinical_measurement_with_channel_provenance() -> None:
    reg = build_background_registry()
    channels = ["C3", "C4", "O1", "O2"]
    signal = np.random.randn(4, 4, 2000).astype("float32") * 20.0

    output, rec = reg.dispatch("amplitude_summary", signal_nct=signal, source_ref="s", channels=channels)

    assert rec.status == "ok"
    amp = next(x for x in output if x.measurement_name == "background_amplitude_range_uv")
    typical = next(x for x in output if x.measurement_name == "background_amplitude_typical_uv")
    peak_to_peak = next(x for x in output if x.measurement_name == "background_amplitude_peak_to_peak_typical_uv")
    best_supported = next(x for x in output if x.measurement_name == "background_amplitude_best_supported_uv")
    assert amp.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
    assert typical.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
    assert peak_to_peak.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
    assert best_supported.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
    assert amp.quantitation.lower >= 0.0
    assert amp.quantitation.upper >= amp.quantitation.lower
    assert amp.quantitation.lower <= typical.quantitation.exact <= amp.quantitation.upper
    assert amp.provenance.space.channels == ["O1", "O2"]
    assert typical.provenance.space.channels == ["O1", "O2"]
    assert peak_to_peak.provenance.space.channels == ["O1", "O2"]
    assert best_supported.provenance.space.channels == ["O1", "O2"]
    assert amp.metadata["amplitude_estimator"] == "per_window_channel_half_of_p95_minus_p5"
    assert typical.metadata["reported_value"] == "median_across_selected_window_channel_amplitudes"
    assert peak_to_peak.metadata["amplitude_estimator"] == "per_window_channel_p95_minus_p5"
    assert best_supported.metadata["amplitude_estimator"] == "best_supported_candidate_from_half_envelope_full_envelope_and_rms"


def test_background_slowing_and_beta_scores_are_bounded_fractions() -> None:
    reg = build_background_registry()
    channels = ["C3", "C4", "O1", "O2"]
    signal = np.random.randn(4, 4, 2000).astype("float32")

    slowing, slow_rec = reg.dispatch("slowing_score", signal_nct=signal, fs=200, source_ref="s", channels=channels)
    beta, beta_rec = reg.dispatch("beta_excess_score", signal_nct=signal, fs=200, source_ref="s", channels=channels)
    organization, org_rec = reg.dispatch("background_organization_proxy", signal_nct=signal, fs=200, source_ref="s", channels=channels)

    assert slow_rec.status == "ok"
    assert beta_rec.status == "ok"
    assert org_rec.status == "ok"
    slow_score = next(x for x in slowing if x.measurement_name == "slowing_score")
    slow_status = next(x for x in slowing if x.measurement_name == "background_slowing_status")
    beta_score = next(x for x in beta if x.measurement_name == "beta_excess_score")
    beta_status = next(x for x in beta if x.measurement_name == "excess_beta_status")
    organization_status = next(x for x in organization if x.measurement_name == "background_organization_status")
    assert 0.0 <= slow_score.quantitation.exact <= 1.0
    assert 0.0 <= beta_score.quantitation.exact <= 1.0
    assert slow_status.categorical_value in {"present", "absent", "uncertain"}
    assert beta_status.categorical_value in {"present", "absent", "uncertain"}
    assert organization_status.categorical_value in {"organized", "poorly_organized", "uncertain"}
    assert slow_score.provenance.space.channels == channels
    assert beta_score.provenance.space.channels == channels


def test_pdr_candidate_uses_posterior_alpha_not_global_boundary() -> None:
    reg = build_background_registry()
    fs = 200
    t = np.arange(2000) / fs
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(3, 22, 2000).astype("float32") * 0.05
    for ch in ["O1", "O2", "P3", "P4", "Pz"]:
        signal[:, channels.index(ch), :] += np.sin(2 * np.pi * 10 * t).astype("float32")

    output, rec = reg.dispatch(
        "posterior_dominant_rhythm_candidate",
        signal_nct=signal,
        fs=fs,
        source_ref="s",
        channels=channels,
    )

    assert rec.status == "ok"
    freq = next(x for x in output if x.measurement_name == "pdr_candidate_frequency_hz")
    assert 8.0 <= freq.quantitation.exact <= 13.0
    assert freq.metadata["pdr_supported"] == "true"


def test_pdr_spectral_v2_uses_posterior_alpha_and_ignores_slow_boundary() -> None:
    reg = build_background_registry()
    fs = 200
    t = np.arange(4000) / fs
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(3, 22, 4000).astype("float32") * 0.02
    slow = (3.0 * np.sin(2 * np.pi * 0.5 * t)).astype("float32")
    alpha = np.sin(2 * np.pi * 10 * t).astype("float32")
    signal += slow[None, None, :]
    for ch in ["O1", "O2", "P3", "P4", "Pz"]:
        signal[:, channels.index(ch), :] += alpha

    output, rec = reg.dispatch(
        "posterior_dominant_rhythm_spectral_v2",
        signal_nct=signal,
        fs=fs,
        source_ref="s",
        channels=channels,
    )

    assert rec.status == "ok"
    names = {x.measurement_name for x in output}
    assert "pdr_v2_frequency_hz" in names
    assert "pdr_v2_support_score" in names
    assert "pdr_symmetry_status" in names
    freq = next(x for x in output if x.measurement_name == "pdr_v2_frequency_hz")
    support = next(x for x in output if x.measurement_name == "pdr_v2_support_score")
    symmetry = next(x for x in output if x.measurement_name == "pdr_symmetry_status")
    assert 9.0 <= freq.quantitation.exact <= 11.0
    assert freq.metadata["pdr_supported"] == "true"
    assert symmetry.categorical_value == "symmetric"
    assert "stable_window_channel_alpha_peak" in freq.metadata["method"]
    assert int(freq.metadata["stable_candidate_count"]) > 0
    assert float(support.quantitation.exact) > 0.35
    assert "specparam" in freq.metadata["package_versions"]


def test_event_registry_has_bounded_local_encoder_tool() -> None:
    reg = build_event_registry()
    assert "morphology_feature_encoder" in reg.list_tools()

    signal = np.random.randn(5, 22, 2000).astype("float32")
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    output, rec = reg.dispatch(
        "morphology_feature_encoder",
        signal_nct=signal,
        channels=channels,
        suspicious_windows=[1, 2],
        source_ref="s",
    )

    assert rec.status == "ok"
    assert isinstance(output, list)
    assert any(x.measurement_name == "event_morphology_support_score" for x in output)


def test_spike_wave_candidate_score_prefers_rhythmic_event_windows() -> None:
    reg = build_event_registry()
    assert "spike_wave_candidate_score" in reg.list_tools()

    fs = 200
    t = np.arange(2000) / fs
    signal = np.random.randn(6, 22, 2000).astype("float32") * 0.2
    signal[1, :, :] += (20.0 * np.sin(2 * np.pi * 0.8 * t)).astype("float32")
    for ch_idx in [5, 7]:
        signal[3, ch_idx, :] += (35.0 * np.sin(2 * np.pi * 4.0 * t)).astype("float32")

    output, rec = reg.dispatch(
        "spike_wave_candidate_score",
        signal_nct=signal,
        window_seconds=10,
        source_ref="s",
    )

    assert rec.status == "ok"
    score = next(x for x in output if x.measurement_name == "spike_wave_candidate_score_distribution")
    values = np.asarray(score.quantitation.values, dtype=float)
    assert int(np.argmax(values)) == 3
    assert values[3] > values[1]


def test_event_module_uses_spike_wave_score_for_focused_windows() -> None:
    reg = build_event_registry()
    fs = 200
    t = np.arange(2000) / fs
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(6, 22, 2000).astype("float32") * 0.2
    signal[1, :, :] += (50.0 * np.sin(2 * np.pi * 0.8 * t)).astype("float32")
    for ch in ["F3", "F7"]:
        signal[3, channels.index(ch), :] += (35.0 * np.sin(2 * np.pi * 4.0 * t)).astype("float32")

    result = EventModule(registry=reg, agent=EventAgent()).run(
        signal_nct=signal,
        channels=channels,
        source_ref="s",
        window_seconds=10,
        scout_summary={"event_density_hint": 0.2, "enable_local_encoder": False},
    )

    assert 3 in result["focused_windows"]
    names = {m.measurement_name for m in result["measurements"]}
    assert "spike_wave_candidate_score_distribution" in names


def test_event_peak_topography_localizer_uses_peak_centered_field() -> None:
    reg = build_event_registry()
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(3, 22, 2000).astype("float32") * 0.01
    for ch, amp in {"F7": 6.0, "T3": 5.0, "T5": 4.5}.items():
        signal[1, channels.index(ch), 1000] += amp
        signal[1, channels.index(ch), 1001] -= amp * 0.5

    output, rec = reg.dispatch(
        "event_peak_topography_localizer",
        signal_nct=signal,
        channels=channels,
        suspicious_windows=[1],
        source_ref="s",
    )

    assert rec.status == "ok"
    label = next(x for x in output if x.measurement_name == "event_peak_localization_label")
    assert "left" in label.categorical_value
    assert "temporal" in label.categorical_value or "frontotemporal" in label.categorical_value
    assert label.metadata["localization_is_peak_centered"] == "true"
    assert "F7" in label.metadata["top_channels"]


def test_event_spatiomorphology_v2_outputs_safe_descriptors_without_scores() -> None:
    reg = build_event_registry()
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(3, 22, 2000).astype("float32") * 0.01
    for ch, amp in {"F3": 5.5, "F7": 6.0, "T3": 4.0}.items():
        signal[1, channels.index(ch), 1000] += amp
        signal[1, channels.index(ch), 1001] -= amp * 0.6

    output, rec = reg.dispatch(
        "event_spatiomorphology_v2",
        signal_nct=signal,
        channels=channels,
        suspicious_windows=[1],
        source_ref="s",
    )

    assert rec.status == "ok"
    names = {x.measurement_name for x in output}
    assert {
        "event_electrode_maxima_v2",
        "event_region_v2",
        "event_laterality_v2",
        "event_spatial_pattern_v2",
        "event_field_descriptor_v2",
        "event_morphology_descriptor_v2",
    }.issubset(names)
    pattern = next(x for x in output if x.measurement_name == "event_spatial_pattern_v2")
    field = next(x for x in output if x.measurement_name == "event_field_descriptor_v2")
    assert "maximal at" in pattern.categorical_value
    assert "event field" in field.categorical_value
    assert pattern.metadata["internal_scores_suppressed"] == "true"
    assert all((m.quantitation is None or m.quantitation.unit != "ratio") for m in output)


def test_event_spatiomorphology_v2_detects_spike_wave_like_pattern() -> None:
    reg = build_event_registry()
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(3, 22, 2000).astype("float32") * 0.02
    fs = 200
    for center in [300, 350, 400, 450, 500, 550, 600]:
        spike = np.exp(-0.5 * ((np.arange(2000) - center) / 3.0) ** 2) * 80.0
        slow = -np.exp(-0.5 * ((np.arange(2000) - (center + 45)) / 25.0) ** 2) * 35.0
        waveform = (spike + slow).astype("float32")
        for ch in ["F3", "F7"]:
            signal[1, channels.index(ch), :] += waveform

    output, rec = reg.dispatch(
        "event_spatiomorphology_v2",
        signal_nct=signal,
        channels=channels,
        suspicious_windows=[1],
        source_ref="s",
        window_seconds=10,
    )

    assert rec.status == "ok"
    descriptor = next(x for x in output if x.measurement_name == "event_morphology_descriptor_v2")
    assert descriptor.categorical_value in {"spike_wave_like", "generalized_spike_wave_like"}
    assert descriptor.metadata["sharp_component"] == "present"
    assert descriptor.metadata["slow_wave_follow"] == "present"
    assert descriptor.metadata["rhythmicity"] == "present"


def test_event_registry_separates_seizure_likelihood_from_event_candidates() -> None:
    reg = build_event_registry()
    signal = np.random.randn(5, 22, 2000).astype("float32")
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    output, rec = reg.dispatch(
        "event_type_separation_classifier",
        signal_nct=signal,
        channels=channels,
        suspicious_windows=[1, 2],
        score_distribution=np.array([0.1, 2.0, 2.2, 0.1, 0.2]),
        window_seconds=10,
        source_ref="s",
    )

    assert rec.status == "ok"
    names = {x.measurement_name for x in output}
    assert "epileptiform_candidate_likelihood_score" in names
    assert "electrographic_seizure_likelihood_score" in names


def test_event_waveform_numeric_v2_outputs_traceable_clinical_values() -> None:
    reg = build_event_registry()
    assert "event_waveform_numeric_v2" in reg.list_tools()

    fs = 200
    t = np.arange(2000) / fs
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(5, 22, 2000).astype("float32") * 2.0
    rhythm = (50.0 * np.sin(2 * np.pi * 4.0 * t)).astype("float32")
    for ch in ["F3", "F7"]:
        signal[2, channels.index(ch), :] += rhythm

    output, rec = reg.dispatch(
        "event_waveform_numeric_v2",
        signal_nct=signal,
        channels=channels,
        suspicious_windows=[2],
        window_seconds=10,
        source_ref="s",
    )

    assert rec.status == "ok"
    names = {x.measurement_name for x in output}
    assert {
        "event_waveform_amplitude_peak_to_peak_typical_uv",
        "event_waveform_amplitude_peak_to_peak_range_uv",
        "event_waveform_dominant_frequency_hz",
    }.issubset(names)
    amplitude = next(x for x in output if x.measurement_name == "event_waveform_amplitude_peak_to_peak_typical_uv")
    frequency = next(x for x in output if x.measurement_name == "event_waveform_dominant_frequency_hz")
    assert amplitude.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
    assert frequency.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
    assert amplitude.quantitation.exact > 80.0
    assert 3.5 <= frequency.quantitation.exact <= 4.5
    if "event_waveform_duration_typical_sec" in names:
        duration = next(x for x in output if x.measurement_name == "event_waveform_duration_typical_sec")
        assert duration.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
        assert 0.0 < duration.quantitation.exact <= 10.0
    assert "F3" in amplitude.provenance.space.channels or "F7" in amplitude.provenance.space.channels
    assert amplitude.metadata["not_seizure_evidence"] == "true"


def test_event_waveform_numeric_groups_into_traceable_evidence_item() -> None:
    reg = build_event_registry()
    channels = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "A1", "A2"]
    signal = np.random.randn(4, 22, 2000).astype("float32") * 2.0
    t = np.arange(2000) / 200
    signal[1, channels.index("F7"), :] += (45.0 * np.sin(2 * np.pi * 5.0 * t)).astype("float32")
    output, _rec = reg.dispatch(
        "event_waveform_numeric_v2",
        signal_nct=signal,
        channels=channels,
        suspicious_windows=[1],
        window_seconds=10,
        source_ref="s",
    )

    board = build_shared_evidence_board(recording_id="r", measurements=output)
    item = next(x for x in board.evidence_items if x.evidence_id == "evgrp_event_waveform_numeric")
    assert item.clinical_target == ClinicalTarget.EPILEPTIFORM_MORPHOLOGY
    assert item.evidence_type == EvidenceType.DERIVED
    assert item.reportability == ClaimSurfaceAction.CAVEAT
    assert item.value["event_waveform_numeric"]["amplitude_peak_to_peak_typical_uv"] is not None
    assert item.value["event_waveform_numeric"]["dominant_frequency_hz"] is not None
    assert item.value["event_waveform_numeric"]["not_seizure_evidence"] is True
