"""Probe installed EEG-related package capabilities for this project.

This is an exploratory script. It verifies that candidate package functions
exist and can be called on a small synthetic EEG-like signal where practical.
The output is intended to guide future measurement-tool improvements, not to
validate clinical accuracy.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class CapabilityRecord:
    package: str
    capability: str
    object_path: str
    status: str
    task_relevance: str
    recommended_use: str
    output_summary: dict[str, Any]
    error: str | None = None


def _safe_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "None"}
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "finite": bool(np.isfinite(value).all()),
            "mean": float(np.nanmean(value)) if value.size else None,
        }
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    return {"type": type(value).__name__, "repr": repr(value)[:160]}


def _resolve_object(path: str) -> Any:
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _record(
    package: str,
    capability: str,
    object_path: str,
    task_relevance: str,
    recommended_use: str,
    call: Callable[[Any], Any] | None,
) -> CapabilityRecord:
    try:
        obj = _resolve_object(object_path)
        if call is None:
            return CapabilityRecord(
                package=package,
                capability=capability,
                object_path=object_path,
                status="exists",
                task_relevance=task_relevance,
                recommended_use=recommended_use,
                output_summary={"type": type(obj).__name__},
            )
        output = call(obj)
        return CapabilityRecord(
            package=package,
            capability=capability,
            object_path=object_path,
            status="call_ok",
            task_relevance=task_relevance,
            recommended_use=recommended_use,
            output_summary=_safe_summary(output),
        )
    except Exception as exc:  # pragma: no cover - exploratory diagnostics
        return CapabilityRecord(
            package=package,
            capability=capability,
            object_path=object_path,
            status="failed",
            task_relevance=task_relevance,
            recommended_use=recommended_use,
            output_summary={},
            error=f"{type(exc).__name__}: {exc}",
        )


def _synthetic_eeg(sf: float = 200.0, duration_sec: float = 20.0) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(7)
    t = np.arange(int(sf * duration_sec)) / sf
    posterior_alpha = 35.0 * np.sin(2 * math.pi * 10.0 * t)
    theta = 10.0 * np.sin(2 * math.pi * 6.0 * t)
    slow = 25.0 * np.sin(2 * math.pi * 1.2 * t)
    spindle = np.zeros_like(t)
    mask = (t > 6.0) & (t < 7.2)
    spindle[mask] = 20.0 * np.sin(2 * math.pi * 13.0 * t[mask])
    noise = rng.normal(0.0, 4.0, size=t.shape)
    channels = np.vstack(
        [
            posterior_alpha + noise,
            0.8 * posterior_alpha + rng.normal(0.0, 4.0, size=t.shape),
            theta + spindle + rng.normal(0.0, 4.0, size=t.shape),
            slow + rng.normal(0.0, 4.0, size=t.shape),
        ]
    )
    return channels, t, sf


def _write_markdown(records: list[CapabilityRecord], path: Path) -> None:
    rows = [
        "| package | capability | status | relevance | recommended use | error |",
        "|---|---|---:|---|---|---|",
    ]
    for rec in records:
        rows.append(
            "| {package} | {capability} | {status} | {relevance} | {use} | {error} |".format(
                package=rec.package,
                capability=rec.capability,
                status=rec.status,
                relevance=rec.task_relevance,
                use=rec.recommended_use,
                error=(rec.error or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="artifacts/eeg_package_capability_probe",
        help="Directory for JSON and markdown outputs.",
    )
    args = parser.parse_args()

    x, _t, sf = _synthetic_eeg()
    one = x[0]
    two = x[1]

    records: list[CapabilityRecord] = []

    records.extend(
        [
            _record(
                "scipy.signal",
                "Welch PSD",
                "scipy.signal.welch",
                "high: PDR/bandpower/slowing",
                "Replace ad-hoc PSD with standardized spectral estimates.",
                lambda fn: fn(one, fs=sf, nperseg=int(4 * sf)),
            ),
            _record(
                "scipy.signal",
                "Peak detection",
                "scipy.signal.find_peaks",
                "high: alpha/PDR candidate selection",
                "Find posterior alpha peaks after PSD estimation.",
                lambda fn: fn(np.abs(np.fft.rfft(one)), prominence=10),
            ),
            _record(
                "scipy.signal",
                "Spectrogram",
                "scipy.signal.spectrogram",
                "medium: state/rhythmicity windows",
                "Track frequency changes over time before evidence grouping.",
                lambda fn: fn(one, fs=sf, nperseg=int(2 * sf)),
            ),
            _record(
                "scipy.signal",
                "Coherence",
                "scipy.signal.coherence",
                "medium: posterior symmetry/connectivity proxy",
                "Estimate channel-pair synchrony as support, not direct prose.",
                lambda fn: fn(one, two, fs=sf, nperseg=int(4 * sf)),
            ),
            _record(
                "scipy.signal",
                "Hilbert analytic envelope",
                "scipy.signal.hilbert",
                "medium: amplitude envelope/burst support",
                "Compute envelope features after bandpass filtering.",
                lambda fn: np.abs(fn(one)),
            ),
        ]
    )

    records.extend(
        [
            _record(
                "mne",
                "Create RawArray",
                "mne.io.RawArray",
                "high: channel/montage-aware signal container",
                "Use for montage-aware preprocessing and topographic provenance.",
                lambda cls: cls(x, importlib.import_module("mne").create_info(["O1", "O2", "C3", "C4"], sf, "eeg"), verbose=False),
            ),
            _record(
                "mne",
                "Create standard montage",
                "mne.channels.make_standard_montage",
                "high: electrode maxima/topography",
                "Attach 10-20 coordinates for localization provenance.",
                lambda fn: fn("standard_1020"),
            ),
        ]
    )

    records.extend(
        [
            _record(
                "pyedflib",
                "EDF reader class",
                "pyedflib.EdfReader",
                "low now: raw EDF IO",
                "Useful only if we ingest EDF directly instead of PKL folders.",
                None,
            ),
            _record(
                "pyedflib",
                "EDF writer class",
                "pyedflib.EdfWriter",
                "low now: debug/export",
                "Useful for exporting synthetic/debug signals.",
                None,
            ),
        ]
    )

    records.extend(
        [
            _record(
                "neurodsp",
                "Welch spectrum",
                "neurodsp.spectral.compute_spectrum_welch",
                "high: robust oscillation spectrum",
                "Alternative spectral estimator for PDR/bandpower/slowing.",
                lambda fn: fn(one, sf, nperseg=int(4 * sf), f_range=(0.5, 40.0)),
            ),
            _record(
                "neurodsp",
                "Bandpass filter",
                "neurodsp.filt.filter_signal",
                "high: alpha/sigma/delta feature isolation",
                "Standardize band-limited features before amplitude/envelope estimates.",
                lambda fn: fn(one, sf, "bandpass", (8, 12)),
            ),
            _record(
                "neurodsp",
                "Dual-threshold burst detection",
                "neurodsp.burst.detect_bursts_dual_threshold",
                "medium: rhythmic burst support",
                "Use as support evidence only, not direct epileptiform prose.",
                lambda fn: fn(one, sf, dual_thresh=(1, 2), f_range=(8, 12)),
            ),
        ]
    )

    records.extend(
        [
            _record(
                "yasa",
                "Bandpower table",
                "yasa.bandpower",
                "high: background bandpower/state support",
                "Use for standardized relative bandpower summaries.",
                lambda fn: fn(x, sf=sf, ch_names=["O1", "O2", "C3", "C4"]),
            ),
            _record(
                "yasa",
                "Sleep spindle detector",
                "yasa.spindles_detect",
                "high: stage II sleep architecture support",
                "Detect spindles/K-complex-adjacent sleep evidence when state supports it.",
                lambda fn: fn(x, sf=sf, ch_names=["O1", "O2", "C3", "C4"], verbose=False),
            ),
            _record(
                "yasa",
                "Slow-wave detector",
                "yasa.sw_detect",
                "medium: sleep/slowing support",
                "Potential sleep slow-wave support; avoid conflating with pathology.",
                lambda fn: fn(x, sf=sf, ch_names=["O1", "O2", "C3", "C4"], verbose=False),
            ),
            _record(
                "yasa",
                "Artifact detector",
                "yasa.art_detect",
                "medium: artifact/status support",
                "Flag bad windows before PDR/amplitude claims.",
                lambda fn: fn(x, sf=sf, verbose=False),
            ),
        ]
    )

    records.extend(
        [
            _record(
                "specparam",
                "Spectral parameterization model",
                "specparam.SpectralModel",
                "high: separate 1/f from true oscillatory peaks",
                "Prevent global/boundary peak from being mislabeled as PDR.",
                lambda cls: _fit_specparam(cls, one, sf),
            ),
            _record(
                "fooof",
                "Legacy FOOOF model",
                "fooof.FOOOF",
                "medium: spectral parameterization fallback",
                "Legacy fallback only; prefer specparam in new code.",
                None,
            ),
        ]
    )

    records.extend(
        [
            _record(
                "antropy",
                "Spectral entropy",
                "antropy.spectral_entropy",
                "medium: background organization/complexity support",
                "Use as support feature, not direct clinical claim.",
                lambda fn: fn(one, sf=sf, method="welch", normalize=True),
            ),
            _record(
                "antropy",
                "Hjorth parameters",
                "antropy.hjorth_params",
                "medium: background complexity/mobility support",
                "Use as bounded support features for state/artifact screening.",
                lambda fn: fn(one),
            ),
        ]
    )

    records.extend(
        [
            _record(
                "braindecode",
                "EEGNetv4 model class",
                "braindecode.models.EEGNetv4",
                "future: learned detector baseline",
                "Not for current bounded tools unless we train/validate a detector.",
                None,
            ),
            _record(
                "braindecode",
                "SleepStagerChambon2018 model class",
                "braindecode.models.SleepStagerChambon2018",
                "future: learned sleep staging",
                "Potential future model-based state detector; not zero-shot clinical evidence.",
                None,
            ),
        ]
    )

    records.append(
        _record(
            "sklearn",
            "RobustScaler",
            "sklearn.preprocessing.RobustScaler",
            "medium: feature normalization",
            "Normalize bounded features across windows before evidence grouping.",
            lambda cls: cls().fit_transform(x.T),
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "package_capabilities.json"
    md_path = out_dir / "package_capabilities.md"

    json_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(records, md_path)

    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(counts, indent=2))
    return 0


def _fit_specparam(cls: Any, sig: np.ndarray, sf: float) -> dict[str, Any]:
    from scipy.signal import welch

    freqs, psd = welch(sig, fs=sf, nperseg=int(4 * sf))
    mask = (freqs >= 1.0) & (freqs <= 40.0)
    model = cls(verbose=False)
    model.fit(freqs[mask], psd[mask], [1.0, 40.0])
    return {
        "model_class": type(model).__name__,
        "has_results": bool(getattr(model, "has_model", False)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
