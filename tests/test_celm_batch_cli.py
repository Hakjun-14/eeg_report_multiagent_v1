from eeg_report_multiagent.cli.run_celm_split_batch import _safe_name


def test_safe_name_for_report_ids() -> None:
    assert _safe_name("NeuroReport_x/y z") == "NeuroReport_x_y_z"
