import numpy as np

from eeg_report_multiagent.agents.background_agent import BackgroundAgent
from eeg_report_multiagent.schemas.measurement import MeasurementRole
from eeg_report_multiagent.schemas.measurement import MeasurementValue
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
    assert amp.measurement_role == MeasurementRole.CLINICAL_MEASUREMENT
    assert amp.quantitation.lower >= 0.0
    assert amp.quantitation.upper >= amp.quantitation.lower
    assert amp.provenance.space.channels == ["O1", "O2"]
    assert amp.metadata["amplitude_estimator"] == "per_window_channel_half_of_p95_minus_p5"


def test_background_slowing_and_beta_scores_are_bounded_fractions() -> None:
    reg = build_background_registry()
    channels = ["C3", "C4", "O1", "O2"]
    signal = np.random.randn(4, 4, 2000).astype("float32")

    slowing, slow_rec = reg.dispatch("slowing_score", signal_nct=signal, fs=200, source_ref="s", channels=channels)
    beta, beta_rec = reg.dispatch("beta_excess_score", signal_nct=signal, fs=200, source_ref="s", channels=channels)

    assert slow_rec.status == "ok"
    assert beta_rec.status == "ok"
    slow_score = next(x for x in slowing if x.measurement_name == "slowing_score")
    beta_score = next(x for x in beta if x.measurement_name == "beta_excess_score")
    assert 0.0 <= slow_score.quantitation.exact <= 1.0
    assert 0.0 <= beta_score.quantitation.exact <= 1.0
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
    freq = next(x for x in output if x.measurement_name == "pdr_v2_frequency_hz")
    support = next(x for x in output if x.measurement_name == "pdr_v2_support_score")
    assert 9.0 <= freq.quantitation.exact <= 11.0
    assert freq.metadata["pdr_supported"] == "true"
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
