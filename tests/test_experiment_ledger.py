import csv
from pathlib import Path

from eeg_report_multiagent.cli.build_experiment_ledger import build_ledger
from eeg_report_multiagent.cli.run_celm_split_batch import _read_row_indices_file
from eeg_report_multiagent.cli.select_ledger_subset import select_celm_stratified


def test_read_row_indices_file_txt(tmp_path: Path) -> None:
    p = tmp_path / "rows.txt"
    p.write_text("# comment\n3\n5\n", encoding="utf-8")
    assert _read_row_indices_file(p) == [3, 5]


def test_read_row_indices_file_csv(tmp_path: Path) -> None:
    p = tmp_path / "rows.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "report_id"])
        writer.writeheader()
        writer.writerow({"row_index": "2", "report_id": "a"})
        writer.writerow({"row_index": "7", "report_id": "b"})
    assert _read_row_indices_file(p) == [2, 7]


def test_select_celm_stratified_skips_completed_rows() -> None:
    rows = []
    for i in range(20):
        rows.append(
            {
                "row_index": str(i),
                "celm_generated_exists": "true" if i < 15 else "false",
                "celm_nonzero_text_metric": "true" if i < 15 else "false",
                "celm_rougeL": str(i / 20),
                "our_B_status": "ok" if i == 0 else "not_started",
            }
        )
    selected = select_celm_stratified(rows, n=8, score_column="celm_rougeL", seed=1)
    assert len(selected) == 8
    assert "0" not in {row["row_index"] for row in selected}
    assert any(row["celm_generated_exists"] == "false" for row in selected)


def test_build_ledger_adds_our_b_metric_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    split_dir = data_root / "random_split_data_by_patient"
    split_dir.mkdir(parents=True)
    with (split_dir / "S0001_test_split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "DeidentifiedName(Reports)",
                "BDSPPatientID",
                "VisitTypeDSC",
                "NumberOfSessions",
                "Processed_EEG_Paths",
                "Extracted_EEG_sections",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "DeidentifiedName(Reports)": "NeuroReport_demo.txt",
                "BDSPPatientID": "P1",
                "VisitTypeDSC": "ROUTINE EEG",
                "NumberOfSessions": "1",
                "Processed_EEG_Paths": "processed_eeg/sub-demo_ses-1",
                "Extracted_EEG_sections": "detail:",
            }
        )

    our_batch_root = tmp_path / "our_batch"
    our_batch_root.mkdir()
    with (our_batch_root / "batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["report_id", "status"])
        writer.writeheader()
        writer.writerow({"report_id": "NeuroReport_demo", "status": "ok"})
    our_scores_dir = our_batch_root / "celm_results"
    our_scores_dir.mkdir()
    with (our_scores_dir / "overall_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["deidentified_name", "rougeL", "meteor"])
        writer.writeheader()
        writer.writerow({"deidentified_name": "NeuroReport_demo", "rougeL": "0.25", "meteor": "0.5"})

    rows = build_ledger(
        data_root=data_root,
        site="S0001",
        split_type="random_split_data_by_patient",
        split="test",
        celm_results_dir=None,
        our_batch_root=our_batch_root,
    )

    assert rows[0]["our_B_status"] == "ok"
    assert rows[0]["our_B_nonzero_text_metric"] == "true"
    assert rows[0]["our_B_rougeL"] == "0.25"
    assert rows[0]["our_B_meteor"] == "0.5"
