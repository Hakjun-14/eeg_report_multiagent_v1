from __future__ import annotations

from typing import List

import numpy as np

from eeg_report_multiagent.schemas.measurement import MeasurementValue
from eeg_report_multiagent.schemas.measurement import StatusSemantic
from eeg_report_multiagent.tools.common import (
    make_exact_measurement,
    make_provenance,
    make_range_measurement,
    make_status_measurement,
)


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}
PHYSIOLOGIC_BAND = (0.5, 30.0)
POSTERIOR_CHANNELS = {"O1", "O2", "P3", "P4", "Pz"}
ANTERIOR_CHANNELS = {"Fp1", "Fp2", "Fpz", "F3", "F4", "F7", "F8", "Fz"}


def _detrend(signal_nct: np.ndarray) -> np.ndarray:
    return signal_nct - np.mean(signal_nct, axis=-1, keepdims=True)


def _infer_voltage_scale(signal_nct: np.ndarray) -> tuple[float, str, str]:
    """Return multiplier to microvolts plus a traceable scale assumption."""
    p95 = float(np.percentile(np.abs(signal_nct), 95.0))
    if p95 < 1.0:
        return 1_000_000.0, "uV", "input_magnitude_lt_1_assumed_volts"
    return 1.0, "uV", "input_magnitude_ge_1_assumed_microvolts"


def _compute_psd(signal_nct: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    # signal_nct: (N, C, T)
    signal_nct = _detrend(signal_nct)
    n = signal_nct.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft = np.fft.rfft(signal_nct, axis=-1)
    psd = (np.abs(fft) ** 2) / n
    return freqs, psd


def _bandpower(freqs: np.ndarray, psd_ncf: np.ndarray, band: tuple[float, float], reduce: str = "sum") -> np.ndarray:
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(psd_ncf.shape[:-1], dtype=np.float32)
    band_psd = psd_ncf[..., mask]
    if reduce == "mean":
        return band_psd.mean(axis=-1)
    return band_psd.sum(axis=-1)


def psd_power_spectrum_summary(signal_nct: np.ndarray, fs: int, source_ref: str) -> List[MeasurementValue]:
    freqs, psd = _compute_psd(signal_nct, fs=fs)
    mean_psd = psd.mean(axis=(0, 1))
    mask = (freqs >= PHYSIOLOGIC_BAND[0]) & (freqs <= PHYSIOLOGIC_BAND[1])
    dom_freq = float(freqs[mask][int(np.argmax(mean_psd[mask]))]) if np.any(mask) else 0.0
    provenance = make_provenance(
        tool_name="psd_power_spectrum_summary",
        function_name="psd_power_spectrum_summary",
        source_ref=source_ref,
        window_indices=range(signal_nct.shape[0]),
        reason="detrended_psd_argmax_restricted_to_0p5_30_hz",
    )
    m = make_exact_measurement(
        measurement_id="m_background_dominant_frequency",
        measurement_name="background_dominant_frequency_hz",
        value=dom_freq,
        unit="Hz",
        provenance=provenance,
    )
    m.metadata.update({"frequency_search_hz": "0.5-30", "preprocessing": "per_window_channel_mean_removed"})
    return [m]


def posterior_dominant_rhythm_candidate(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    """Estimate a posterior alpha/PDR candidate without using the 0.5 Hz global argmax.

    This tool intentionally searches posterior channels in the 8-13 Hz range
    and reports a confidence score. It is not a full PDR/reactivity detector.
    """
    if channels is None:
        channels = [str(i) for i in range(signal_nct.shape[1])]
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    posterior_idx = [ch_to_idx[c] for c in channels if c in POSTERIOR_CHANNELS]
    anterior_idx = [ch_to_idx[c] for c in channels if c in ANTERIOR_CHANNELS]
    if not posterior_idx:
        posterior_idx = list(range(signal_nct.shape[1]))

    freqs, psd = _compute_psd(signal_nct[:, posterior_idx, :], fs=fs)
    posterior_psd = psd.mean(axis=(0, 1))
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    if not np.any(alpha_mask):
        peak_hz = 0.0
        confidence = 0.0
        posterior_alpha_ratio = 0.0
        ap_ratio = 0.0
        symmetry_score = 0.0
    else:
        alpha_freqs = freqs[alpha_mask]
        alpha_psd = posterior_psd[alpha_mask]
        peak_idx = int(np.argmax(alpha_psd))
        peak_hz = float(alpha_freqs[peak_idx])
        posterior_total = float(_bandpower(freqs, psd, PHYSIOLOGIC_BAND).mean()) + 1e-12
        posterior_alpha = float(_bandpower(freqs, psd, (8.0, 13.0)).mean())
        posterior_alpha_ratio = posterior_alpha / posterior_total

        if anterior_idx:
            _, ant_psd = _compute_psd(signal_nct[:, anterior_idx, :], fs=fs)
            anterior_alpha = float(_bandpower(freqs, ant_psd, (8.0, 13.0)).mean())
            ap_ratio = posterior_alpha / (anterior_alpha + 1e-12)
        else:
            ap_ratio = 1.0

        if len(posterior_idx) >= 2:
            posterior_channel_alpha = _bandpower(freqs, psd, (8.0, 13.0)).mean(axis=0)
            left_vals = [
                float(posterior_channel_alpha[i])
                for i, channel_idx in enumerate(posterior_idx)
                if channels[channel_idx] in {"O1", "P3"}
            ]
            right_vals = [
                float(posterior_channel_alpha[i])
                for i, channel_idx in enumerate(posterior_idx)
                if channels[channel_idx] in {"O2", "P4"}
            ]
            if left_vals and right_vals:
                left_alpha = float(np.mean(left_vals))
                right_alpha = float(np.mean(right_vals))
                symmetry_score = float(1.0 - abs(left_alpha - right_alpha) / (left_alpha + right_alpha + 1e-12))
                symmetry_score = max(0.0, min(1.0, symmetry_score))
            else:
                symmetry_score = 0.0
        else:
            symmetry_score = 0.0

        prominence = float(np.max(alpha_psd) / (np.median(alpha_psd) + 1e-12))
        confidence = float(
            min(
                1.0,
                0.45 * min(posterior_alpha_ratio / 0.25, 1.0)
                + 0.35 * min(prominence / 3.0, 1.0)
                + 0.20 * min(ap_ratio / 1.5, 1.0),
            )
        )

    pdr_supported = 8.0 <= peak_hz <= 13.0 and confidence >= 0.35
    provenance = make_provenance(
        tool_name="posterior_dominant_rhythm_candidate",
        function_name="posterior_dominant_rhythm_candidate",
        source_ref=source_ref,
        window_indices=range(signal_nct.shape[0]),
        channels=[channels[i] for i in posterior_idx if i < len(channels)],
        region="posterior",
        reason="posterior_channel_alpha_peak_search_8_13_hz_not_global_psd_argmax",
    )
    freq = make_exact_measurement(
        measurement_id="m_pdr_candidate_frequency",
        measurement_name="pdr_candidate_frequency_hz",
        value=peak_hz,
        unit="Hz",
        provenance=provenance,
    )
    freq.metadata.update(
        {
            "pdr_supported": str(pdr_supported).lower(),
            "posterior_alpha_ratio": f"{posterior_alpha_ratio:.6f}",
            "posterior_anterior_alpha_ratio": f"{ap_ratio:.6f}",
            "posterior_alpha_symmetry_score": f"{symmetry_score:.6f}",
            "search_hz": "8-13",
        }
    )
    score = make_exact_measurement(
        measurement_id="m_pdr_candidate_confidence",
        measurement_name="pdr_candidate_confidence_score",
        value=confidence,
        unit="score",
        provenance=provenance,
    )
    score.metadata.update(freq.metadata)
    ap = make_exact_measurement(
        measurement_id="m_pdr_posterior_anterior_alpha_ratio",
        measurement_name="pdr_posterior_anterior_alpha_ratio",
        value=ap_ratio,
        unit="ratio",
        provenance=provenance,
    )
    symmetry = make_exact_measurement(
        measurement_id="m_pdr_symmetry_score",
        measurement_name="pdr_symmetry_score",
        value=symmetry_score,
        unit="score",
        provenance=provenance,
    )
    ap.metadata.update(freq.metadata)
    symmetry.metadata.update(freq.metadata)
    return [freq, score, ap, symmetry]


def posterior_dominant_rhythm_spectral_v2(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    """Posterior alpha/PDR candidate using Welch PSD plus spectral parameterization.

    This runs in parallel with the legacy PDR candidate. It still produces
    bounded MeasurementValue records only. The frequency is a clinical
    measurement candidate; support scores and ratios remain support/proxy
    features for downstream evidence grouping and SurfacePolicy.
    """
    if channels is None:
        channels = [str(i) for i in range(signal_nct.shape[1])]
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    posterior_idx = [ch_to_idx[c] for c in channels if c in POSTERIOR_CHANNELS]
    anterior_idx = [ch_to_idx[c] for c in channels if c in ANTERIOR_CHANNELS]
    if not posterior_idx:
        posterior_idx = list(range(signal_nct.shape[1]))

    posterior = _detrend(signal_nct[:, posterior_idx, :])
    freqs, posterior_psd = _welch_mean_psd(posterior, fs=fs)
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    total_mask = (freqs >= 1.0) & (freqs <= 30.0)

    peak_hz = 0.0
    specparam_peak_hz = 0.0
    specparam_peak_power = 0.0
    specparam_peak_width = 0.0
    stable_peak_hz = 0.0
    stable_candidate_count = 0
    stable_candidate_fraction = 0.0
    peak_prominence_ratio = 0.0
    posterior_alpha_ratio = 0.0
    ap_ratio = 0.0
    support_score = 0.0
    method_notes = ["welch_psd"]

    if np.any(alpha_mask):
        alpha_freqs = freqs[alpha_mask]
        alpha_psd = posterior_psd[alpha_mask]
        peak_hz, peak_power, peak_prominence_ratio = _alpha_peak_from_psd(alpha_freqs, alpha_psd)
        posterior_alpha_power = float(np.trapezoid(alpha_psd, alpha_freqs))
        total_power = float(np.trapezoid(posterior_psd[total_mask], freqs[total_mask])) + 1e-12 if np.any(total_mask) else 1e-12
        posterior_alpha_ratio = posterior_alpha_power / total_power

        spec = _specparam_alpha_peak(freqs, posterior_psd)
        if spec is not None:
            specparam_peak_hz, specparam_peak_power, specparam_peak_width = spec
            peak_hz = specparam_peak_hz
            method_notes.append("specparam_alpha_peak")

        stable = _stable_posterior_alpha_peak(posterior, fs=fs)
        if stable is not None:
            stable_peak_hz, stable_candidate_count, stable_candidate_fraction = stable
            peak_hz = stable_peak_hz
            method_notes.append("stable_window_channel_alpha_peak")

        if anterior_idx:
            anterior = _detrend(signal_nct[:, anterior_idx, :])
            ant_freqs, anterior_psd = _welch_mean_psd(anterior, fs=fs)
            ant_alpha_mask = (ant_freqs >= 8.0) & (ant_freqs <= 13.0)
            anterior_alpha = float(np.trapezoid(anterior_psd[ant_alpha_mask], ant_freqs[ant_alpha_mask])) + 1e-12 if np.any(ant_alpha_mask) else 1e-12
            ap_ratio = posterior_alpha_power / anterior_alpha
        else:
            ap_ratio = 1.0

        ap_ratio = float(min(ap_ratio, 10.0))
        support_score = float(
            min(
                1.0,
                0.40 * min(posterior_alpha_ratio / 0.20, 1.0)
                + 0.35 * min(peak_prominence_ratio / 3.0, 1.0)
                + 0.25 * min(ap_ratio / 1.5, 1.0),
            )
        )

    pdr_supported = bool(8.0 <= peak_hz <= 13.0 and support_score >= 0.35 and posterior_alpha_ratio > 0.05)
    posterior_channels = [channels[i] for i in posterior_idx if i < len(channels)]
    provenance = make_provenance(
        tool_name="posterior_dominant_rhythm_spectral_v2",
        function_name="posterior_dominant_rhythm_spectral_v2",
        source_ref=source_ref,
        window_indices=range(signal_nct.shape[0]),
        channels=posterior_channels,
        region="posterior",
        reason="welch_posterior_alpha_peak_with_optional_specparam_aperiodic_separation",
    )
    base_metadata = {
        "pdr_supported": str(pdr_supported).lower(),
        "search_hz": "8-13",
        "posterior_channels": ",".join(posterior_channels),
        "posterior_alpha_ratio": f"{posterior_alpha_ratio:.6f}",
        "posterior_anterior_alpha_ratio": f"{ap_ratio:.6f}",
        "peak_prominence_ratio": f"{peak_prominence_ratio:.6f}",
        "specparam_peak_hz": f"{specparam_peak_hz:.6f}",
        "specparam_peak_power": f"{specparam_peak_power:.6f}",
        "specparam_peak_width": f"{specparam_peak_width:.6f}",
        "stable_peak_hz": f"{stable_peak_hz:.6f}",
        "stable_candidate_count": str(stable_candidate_count),
        "stable_candidate_fraction": f"{stable_candidate_fraction:.6f}",
        "method": "+".join(method_notes),
        "package_versions": _pdr_v2_package_versions(),
    }
    freq = make_exact_measurement(
        measurement_id="m_pdr_v2_frequency",
        measurement_name="pdr_v2_frequency_hz",
        value=peak_hz,
        unit="Hz",
        provenance=provenance,
    )
    support = make_exact_measurement(
        measurement_id="m_pdr_v2_support_score",
        measurement_name="pdr_v2_support_score",
        value=support_score,
        unit="score",
        provenance=provenance,
    )
    posterior_ratio = make_exact_measurement(
        measurement_id="m_pdr_v2_posterior_alpha_ratio",
        measurement_name="pdr_v2_posterior_alpha_ratio",
        value=posterior_alpha_ratio,
        unit="ratio",
        provenance=provenance,
    )
    ap = make_exact_measurement(
        measurement_id="m_pdr_v2_posterior_anterior_alpha_ratio",
        measurement_name="pdr_v2_posterior_anterior_alpha_ratio",
        value=ap_ratio,
        unit="ratio",
        provenance=provenance,
    )
    for measurement in (freq, support, posterior_ratio, ap):
        measurement.metadata.update(base_metadata)
    return [freq, support, posterior_ratio, ap]


def _stable_posterior_alpha_peak(signal_nct: np.ndarray, fs: int) -> tuple[float, int, float] | None:
    """Return a robust PDR candidate from stable posterior alpha windows/channels.

    This is still a bounded signal measurement. It does not decide whether the
    final report may call the value a PDR; it only avoids letting one averaged
    PSD or one aperiodic fit dominate the numeric candidate.
    """
    freqs, psd = _welch_psd_by_window_channel(signal_nct, fs=fs)
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    total_mask = (freqs >= 1.0) & (freqs <= 30.0)
    if not np.any(alpha_mask) or not np.any(total_mask):
        return None

    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[..., alpha_mask]
    total_psd = psd[..., total_mask]
    peak_indices = np.argmax(alpha_psd, axis=-1)
    peak_hz = alpha_freqs[peak_indices]
    peak_power = np.take_along_axis(alpha_psd, peak_indices[..., None], axis=-1)[..., 0]
    alpha_power = np.trapezoid(alpha_psd, alpha_freqs, axis=-1)
    total_power = np.trapezoid(total_psd, freqs[total_mask], axis=-1) + 1e-12
    alpha_ratio = alpha_power / total_power
    prominence = peak_power / (np.median(alpha_psd, axis=-1) + 1e-12)

    finite = np.isfinite(peak_hz) & np.isfinite(alpha_ratio) & np.isfinite(prominence)
    if not np.any(finite):
        return None
    total_power_cap = np.percentile(total_power[finite], 90.0)
    selected = finite & (alpha_ratio >= 0.05) & (prominence >= 1.5) & (total_power <= total_power_cap)
    if int(np.sum(selected)) < 3:
        selected = finite & (alpha_ratio >= np.percentile(alpha_ratio[finite], 60.0))
    if int(np.sum(selected)) == 0:
        return None

    weights = np.maximum(alpha_ratio[selected] * prominence[selected], 1e-12)
    candidate = _weighted_median(peak_hz[selected], weights)
    return float(candidate), int(np.sum(selected)), float(np.mean(selected))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(max(idx, 0), sorted_values.size - 1)
    return float(sorted_values[idx])


def _welch_mean_psd(signal_nct: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
    except Exception:
        return _compute_psd(signal_nct, fs=fs)

    signal_nct = _detrend(signal_nct)
    nperseg = min(signal_nct.shape[-1], max(int(fs * 4), 16))
    freqs, psd = welch(signal_nct.reshape(-1, signal_nct.shape[-1]), fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd.mean(axis=0)


def _alpha_peak_from_psd(alpha_freqs: np.ndarray, alpha_psd: np.ndarray) -> tuple[float, float, float]:
    if alpha_freqs.size == 0:
        return 0.0, 0.0, 0.0
    peak_idx = int(np.argmax(alpha_psd))
    try:
        from scipy.signal import find_peaks, peak_prominences

        peaks, _props = find_peaks(alpha_psd)
        if peaks.size:
            prominences = peak_prominences(alpha_psd, peaks)[0]
            best = int(peaks[int(np.argmax(prominences))])
            peak_idx = best
    except Exception:
        pass
    peak_power = float(alpha_psd[peak_idx])
    baseline = float(np.median(alpha_psd)) + 1e-12
    return float(alpha_freqs[peak_idx]), peak_power, peak_power / baseline


def _specparam_alpha_peak(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float, float] | None:
    mask = (freqs >= 1.0) & (freqs <= 40.0)
    if not np.any(mask):
        return None
    try:
        from specparam import SpectralModel

        model = SpectralModel(verbose=False)
        model.fit(freqs[mask], psd[mask], [1.0, 40.0])
        peaks = np.asarray(model.get_params("periodic"), dtype=float)
        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)
        alpha_peaks = peaks[(peaks[:, 0] >= 8.0) & (peaks[:, 0] <= 13.0)]
        if alpha_peaks.size == 0:
            return None
        best = alpha_peaks[int(np.argmax(alpha_peaks[:, 1]))]
        return float(best[0]), float(best[1]), float(best[2])
    except Exception:
        return None


def _pdr_v2_package_versions() -> str:
    versions: list[str] = []
    try:
        import scipy

        versions.append(f"scipy={scipy.__version__}")
    except Exception:
        versions.append("scipy=unavailable")
    try:
        import specparam

        versions.append(f"specparam={specparam.__version__}")
    except Exception:
        versions.append("specparam=unavailable")
    return ";".join(versions)


def background_organization_proxy(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    """Typed proxy for anterior-posterior organization from alpha topography."""
    if channels is None:
        channels = [str(i) for i in range(signal_nct.shape[1])]
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    posterior_idx = [ch_to_idx[c] for c in channels if c in POSTERIOR_CHANNELS]
    anterior_idx = [ch_to_idx[c] for c in channels if c in ANTERIOR_CHANNELS]
    if not posterior_idx or not anterior_idx:
        score = 0.0
        region_channels = channels
        reason = "missing_standard_anterior_or_posterior_channels"
    else:
        freqs, post_psd = _welch_psd_by_window_channel(signal_nct[:, posterior_idx, :], fs=fs)
        _, ant_psd = _welch_psd_by_window_channel(signal_nct[:, anterior_idx, :], fs=fs)
        posterior_alpha = float(_bandpower(freqs, post_psd, (8.0, 13.0)).mean())
        anterior_alpha = float(_bandpower(freqs, ant_psd, (8.0, 13.0)).mean())
        ratio = posterior_alpha / (anterior_alpha + 1e-12)
        score = float(min(1.0, ratio / 2.0))
        region_channels = [channels[i] for i in posterior_idx + anterior_idx if i < len(channels)]
        reason = "welch_posterior_over_anterior_alpha_ratio_proxy"

    return [
        make_exact_measurement(
            measurement_id="m_background_ap_organization_score",
            measurement_name="background_ap_organization_score",
            value=score,
            unit="score",
            provenance=make_provenance(
                tool_name="background_organization_proxy",
                function_name="background_organization_proxy",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
                channels=region_channels,
                region="anterior-posterior",
                reason=reason,
            ),
        )
    ]


def background_unavailable_slot_status(source_ref: str) -> List[MeasurementValue]:
    """Declare clinically important background slots that v1 cannot measure yet."""
    provenance = make_provenance(
        tool_name="background_unavailable_slot_status",
        function_name="background_unavailable_slot_status",
        source_ref=source_ref,
        reason="no_eye_opening_markers_sleep_stage_classifier_or_protocol_events_in_processed_pkl",
    )
    return [
        make_status_measurement(
            measurement_id="m_background_reactivity_status",
            measurement_name="background_reactivity_status",
            status=StatusSemantic.UNKNOWN,
            provenance=provenance,
            reason="reactivity requires activation/eye-opening markers not available to v1 signal tools",
        ),
        make_status_measurement(
            measurement_id="m_sleep_architecture_status",
            measurement_name="sleep_architecture_status",
            status=StatusSemantic.UNKNOWN,
            provenance=provenance,
            reason="sleep architecture requires a validated sleep/state detector not available in v1",
        ),
    ]


def bandpower_summary(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    freqs, psd = _welch_psd_by_window_channel(signal_nct, fs=fs)
    total_power = _bandpower(freqs, psd, PHYSIOLOGIC_BAND).mean() + 1e-12
    channel_list = channels or [str(i) for i in range(signal_nct.shape[1])]
    out: List[MeasurementValue] = []
    for band_name, band in BANDS.items():
        bp = float(_bandpower(freqs, psd, band).mean())
        rel = bp / total_power
        m = make_exact_measurement(
            measurement_id=f"m_bandpower_{band_name}",
            measurement_name=f"relative_bandpower_{band_name}",
            value=rel,
            unit="ratio",
            provenance=make_provenance(
                tool_name="bandpower_summary",
                function_name="bandpower_summary",
                source_ref=source_ref,
                window_indices=range(signal_nct.shape[0]),
                channels=channel_list,
                reason="welch_band_power_divided_by_total_0p5_30_hz_power_after_detrending",
            ),
        )
        m.metadata.update({"band_hz": f"{band[0]}-{band[1]}", "denominator_hz": "0.5-30", "method": "welch_psd"})
        out.append(m)
    return out


def amplitude_summary(
    signal_nct: np.ndarray,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    # Robust typical amplitude estimate. Clinical report amplitudes are closer to
    # a half peak-to-peak/background envelope than to full global p95-p5 spread.
    scale, unit, scale_assumption = _infer_voltage_scale(signal_nct)
    channel_list = channels or [str(i) for i in range(signal_nct.shape[1])]
    posterior_idx = [i for i, channel in enumerate(channel_list) if channel in POSTERIOR_CHANNELS]
    region = "posterior" if posterior_idx else None
    selected_idx = posterior_idx or list(range(signal_nct.shape[1]))
    selected_channels = [channel_list[i] for i in selected_idx if i < len(channel_list)]
    signal_uv = signal_nct[:, selected_idx, :] * scale
    robust_low = np.percentile(signal_uv, 5.0, axis=-1)
    robust_high = np.percentile(signal_uv, 95.0, axis=-1)
    robust_half_ptp = np.maximum((robust_high - robust_low) / 2.0, 0.0)
    valid = robust_half_ptp[np.isfinite(robust_half_ptp) & (robust_half_ptp >= 1.0)]
    if valid.size == 0:
        valid = robust_half_ptp[np.isfinite(robust_half_ptp) & (robust_half_ptp > 0.0)]
    if valid.size:
        artifact_cap = float(np.percentile(valid, 90.0))
        valid = valid[valid <= artifact_cap]
    if valid.size == 0:
        lo = 0.0
        hi = 0.0
        typical = 0.0
    else:
        lo = float(np.percentile(valid, 25.0))
        hi = float(np.percentile(valid, 75.0))
        typical = float(np.median(valid))
    range_measurement = make_range_measurement(
        measurement_id="m_background_amplitude_range",
        measurement_name="background_amplitude_range_uv",
        lower=lo,
        upper=hi,
        unit=unit,
        provenance=make_provenance(
            tool_name="amplitude_summary",
            function_name="amplitude_summary",
            source_ref=source_ref,
            window_indices=range(signal_nct.shape[0]),
            channels=selected_channels,
            region=region,
            reason=f"{scale_assumption};posterior_preferred_half_peak_to_peak_iqr_artifact_trimmed",
        ),
    )
    range_measurement.metadata.update(
        {
            "scale_assumption": scale_assumption,
            "amplitude_estimator": "per_window_channel_half_of_p95_minus_p5",
            "reported_range": "iqr_across_selected_window_channel_amplitudes",
            "near_zero_amplitude_rejection_uv": "1.0",
            "artifact_cap_percentile": "90",
            "selected_region": region or "all",
        }
    )
    typical_measurement = make_exact_measurement(
        measurement_id="m_background_amplitude_typical",
        measurement_name="background_amplitude_typical_uv",
        value=typical,
        unit=unit,
        provenance=make_provenance(
            tool_name="amplitude_summary",
            function_name="amplitude_summary",
            source_ref=source_ref,
            window_indices=range(signal_nct.shape[0]),
            channels=selected_channels,
            region=region,
            reason=f"{scale_assumption};posterior_preferred_half_peak_to_peak_median_artifact_trimmed",
        ),
    )
    typical_measurement.metadata.update(
        {
            "scale_assumption": scale_assumption,
            "amplitude_estimator": "per_window_channel_half_of_p95_minus_p5",
            "reported_value": "median_across_selected_window_channel_amplitudes",
            "near_zero_amplitude_rejection_uv": "1.0",
            "artifact_cap_percentile": "90",
            "selected_region": region or "all",
            "paired_range_measurement_id": range_measurement.measurement_id,
        }
    )
    return [range_measurement, typical_measurement]


def slowing_score(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    freqs, psd = _welch_psd_by_window_channel(signal_nct, fs=fs)
    delta = float(_bandpower(freqs, psd, BANDS["delta"]).mean())
    theta = float(_bandpower(freqs, psd, BANDS["theta"]).mean())
    alpha = float(_bandpower(freqs, psd, BANDS["alpha"]).mean())
    beta = float(_bandpower(freqs, psd, BANDS["beta"]).mean())
    total = delta + theta + alpha + beta + 1e-12
    score = float((delta + theta) / total)
    m = make_exact_measurement(
        measurement_id="m_slowing_score",
        measurement_name="slowing_score",
        value=score,
        unit="ratio",
        provenance=make_provenance(
            tool_name="slowing_score",
            function_name="slowing_score",
            source_ref=source_ref,
            window_indices=range(signal_nct.shape[0]),
            channels=channels or [str(i) for i in range(signal_nct.shape[1])],
            reason="welch_delta_theta_fraction_of_delta_theta_alpha_beta_power",
        ),
    )
    m.metadata.update(
        {
            "method": "welch_psd",
            "score_definition": "(delta+theta)/(delta+theta+alpha+beta)",
            "delta_theta_over_alpha_beta": f"{(delta + theta) / (alpha + beta + 1e-12):.6f}",
        }
    )
    return [m]


def beta_excess_score(
    signal_nct: np.ndarray,
    fs: int,
    source_ref: str,
    channels: List[str] | None = None,
) -> List[MeasurementValue]:
    freqs, psd = _welch_psd_by_window_channel(signal_nct, fs=fs)
    beta = float(_bandpower(freqs, psd, BANDS["beta"]).mean())
    others = (
        float(_bandpower(freqs, psd, BANDS["delta"]).mean())
        + float(_bandpower(freqs, psd, BANDS["theta"]).mean())
        + float(_bandpower(freqs, psd, BANDS["alpha"]).mean())
    )
    score = float(beta / (beta + others + 1e-12))
    m = make_exact_measurement(
        measurement_id="m_beta_excess_score",
        measurement_name="beta_excess_score",
        value=score,
        unit="ratio",
        provenance=make_provenance(
            tool_name="beta_excess_score",
            function_name="beta_excess_score",
            source_ref=source_ref,
            window_indices=range(signal_nct.shape[0]),
            channels=channels or [str(i) for i in range(signal_nct.shape[1])],
            reason="welch_beta_fraction_of_delta_theta_alpha_beta_power",
        ),
    )
    m.metadata.update(
        {
            "method": "welch_psd",
            "score_definition": "beta/(delta+theta+alpha+beta)",
            "beta_over_non_beta": f"{beta / (others + 1e-12):.6f}",
        }
    )
    return [m]


def _welch_psd_by_window_channel(signal_nct: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
    except Exception:
        return _compute_psd(signal_nct, fs=fs)

    signal_nct = _detrend(signal_nct)
    nperseg = min(signal_nct.shape[-1], max(int(fs * 4), 16))
    freqs, psd = welch(signal_nct, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd
