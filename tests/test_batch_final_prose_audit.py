from __future__ import annotations

import json
from pathlib import Path

from eeg_report_multiagent.cli.run_batch_final_prose_audit import run_batch_final_prose_audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_batch_final_prose_audit_text_only_outputs(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "per_case_json"
    _write_json(
        comparison_dir / "row_000189_NeuroReport_x.json",
        {
            "row_index": 189,
            "report_id": "NeuroReport_x",
            "variants": {
                "CELM": {
                    "generated_sections": {
                        "BACKGROUND ACTIVITY": "A 0.5 Hz dominant rhythm is present with candidate burden.",
                        "SEIZURES": "Seizures consist of spike-wave complexes lasting 190 sec.",
                    }
                },
                "OURS_SAFE": {
                    "generated_sections": {
                        "SEIZURES": "Seizures: no seizure-specific evidence was produced by the current structured tools."
                    }
                },
            },
        },
    )
    output_dir = tmp_path / "stage2_5"

    summary = run_batch_final_prose_audit(comparison_dir, output_dir, trace_roots={})

    assert summary["case_count"] == 1
    assert (output_dir / "aggregate_metrics.csv").exists()
    assert (output_dir / "failure_bucket_counts.csv").exists()
    assert (output_dir / "variant_comparison.md").exists()
    assert (output_dir / "high_risk_examples.md").exists()
    assert (output_dir / "row189_audit_summary.md").exists()
    assert len(list((output_dir / "per_report_audits").glob("*_final_prose_audit.json"))) == 2

    aggregate = (output_dir / "aggregate_metrics.csv").read_text(encoding="utf-8")
    assert "CELM" in aggregate
    assert "OURS_SAFE" in aggregate
    high_risk = (output_dir / "high_risk_examples.md").read_text(encoding="utf-8")
    assert "candidate burden" in high_risk
    assert "Seizure Gate Violation" in high_risk

