import numpy as np

from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.tools.registry import build_background_registry, build_event_registry


def test_background_registry_dispatch() -> None:
    reg = build_background_registry()
    assert "amplitude_summary" in reg.list_tools()
    assert "posterior_dominant_rhythm_candidate" in reg.list_tools()

    signal = np.random.randn(4, 22, 2000).astype("float32")
    output, rec = reg.dispatch("amplitude_summary", signal_nct=signal, source_ref="s")

    assert rec.status == "ok"
    assert isinstance(output, list)
    assert any(isinstance(x, MeasurementValue) for x in output)


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
