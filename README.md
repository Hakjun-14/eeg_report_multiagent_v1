# eeg_report_multiagent_v1

Minimal task-specific multi-agent EEG report architecture using:
- Local signal tools for raw/processed EEG
- Bounded tool registries (no open-ended tool discovery)
- LangGraph as orchestration layer
- Typed intermediate objects (`Measurement -> Finding -> Report`)

## Project Scope
Core v1 modules:
1. Background / Activity Module
2. Epileptiform / Event Module
3. Protocol / State / Context Parser
4. Shared Evidence Board
5. Report Synthesizer
6. Optional Claim Verifier

## Workspace and Data
This project expects processed EEG windows in CELM-style pickles:
- per window: `(22, 2000)` at 200 Hz for 10 s
- per session: `(N, 22, 2000)` assembled from `seg_*.pkl`

## Quick Start (Docker)
```bash
cd /home/hjlee/Desktop/eeg_report_multiagent_v1
./scripts/check_api_key.sh
export OPENAI_API_KEY='your-key'
./scripts/start_container.sh
```

The script:
- builds image from this folder
- mounts host workspace as `/workspace`
- sets workdir to `/workspace/eeg_report_multiagent_v1`
- passes `OPENAI_API_KEY` into container

## Local Run (without Docker)
```bash
python -m pip install -e .[dev]
eeg-inspect-data --session-dir <processed_eeg/session_dir>
eeg-run-session --session-dir <processed_eeg/session_dir> --study-context-json <study_context.json>
eeg-run-smoke-session --split-csv <S0001_test_split.csv> --row-index 0 --session-index 0 --output-dir artifacts/smoke_row0
```

`--gt-report-json` is reserved for evaluation bookkeeping. GT report text should not be passed as inference context.

Use `--monitor` with `eeg-run-session` or `eeg-run-smoke-session` to watch sequential module execution and evidence board updates in the terminal.

Use `--enable-llm-review` to add optional Rule+LLM evidence review. This sends only structured evidence summaries and bounded tool names to the API.

The review output is stored in `agent_deliberations.json` with typed fields:
- `weak_evidence`
- `missing_slots`
- `do_not_claim`
- `claim_constraints`
- `tool_request_proposals`

The LLM review is evidence-board-only: it does not receive raw EEG arrays, source PKL payloads, or GT report text.

## CELM-Compatible S0001 Interface
Use the same split CSV, report JSON, and processed EEG PKL layout as the CELM baseline:

```bash
docker run --rm \
  -v /home/hjlee/Desktop:/workspace \
  -v /exHDD_8T/hjlee_data/eeg_data:/workspace/eeg_data \
  -w /workspace/eeg_report_multiagent_v1 \
  eeg-report-multiagent-v1:latest \
  eeg-inspect-celm-split \
  --data-root /workspace/eeg_data/celm_s_sites_pipeline \
  --site S0001 \
  --split-type random_split_data_by_patient \
  --split test \
  --row-index 0
```

Run one split row and also write CELM evaluator-compatible files:

```bash
docker run --rm \
  --env-file /home/hjlee/Desktop/eeg_report_multiagent_v1/.env \
  -v /home/hjlee/Desktop:/workspace \
  -v /exHDD_8T/hjlee_data/eeg_data:/workspace/eeg_data \
  -w /workspace/eeg_report_multiagent_v1 \
  eeg-report-multiagent-v1:latest \
  eeg-run-celm-split-session \
  --data-root /workspace/eeg_data/celm_s_sites_pipeline \
  --site S0001 \
  --split-type random_split_data_by_patient \
  --split test \
  --row-index 0 \
  --session-index 0 \
  --output-dir /workspace/eeg_report_multiagent_v1/artifacts/celm_interface_s0001_test_row0 \
  --celm-results-dir /workspace/eeg_report_multiagent_v1/artifacts/celm_eval_compatible_s0001_test_row0
```

Add `--enable-llm-review` to run the Rule+LLM evidence-review variant.

Run a resumable batch over the S0001 CELM-compatible test split:

```bash
docker run --rm \
  --env-file /home/hjlee/Desktop/eeg_report_multiagent_v1/.env \
  -v /home/hjlee/Desktop:/workspace \
  -v /exHDD_8T/hjlee_data/eeg_data:/workspace/eeg_data \
  -w /workspace/eeg_report_multiagent_v1 \
  eeg-report-multiagent-v1:latest \
  eeg-run-celm-split-batch \
  --data-root /workspace/eeg_data/celm_s_sites_pipeline \
  --site S0001 \
  --split-type random_split_data_by_patient \
  --split test \
  --start-row 0 \
  --session-index 0 \
  --output-root /workspace/eeg_report_multiagent_v1/artifacts/batch_s0001_test_B_full \
  --celm-results-dir /workspace/eeg_report_multiagent_v1/artifacts/batch_s0001_test_B_full/celm_results \
  --no-langgraph \
  --enable-llm-review \
  --resume \
  --sleep-sec 0.5 \
  --row-timeout-sec 240
```

For a small smoke run, add `--max-rows 2`. Batch outputs include `batch_summary.csv`, `batch_summary.json`, `batch_final_summary.json`, per-row artifacts under `rows/`, and CELM-compatible generated reports under `celm_results/generated_reports_json/`.

The CELM-compatible output is written as:
- `generated_reports_json/GENERATED_REPORT_<report_id>.json`
- `generated_reports_txt/GENERATED_REPORT_<report_id>.txt`
- `method_audit.json` / `method_audit.md` in the per-run artifact directory

The generated JSON has the CELM evaluator shape:

```json
{"report_sections": [{"section_name": "EEG DESCRIPTION/DETAILS", "section_text": "..."}]}
```

Clinical context follows CELM's age/gender plus `patient_history_section_llm_extractions`, but EEG target section text is not passed as inference input.

Audit an existing run directory without re-running inference:

```bash
docker run --rm \
  -v /home/hjlee/Desktop:/workspace \
  -w /workspace/eeg_report_multiagent_v1 \
  eeg-report-multiagent-v1:latest \
  eeg-audit-method-run \
  --artifact-dir /workspace/eeg_report_multiagent_v1/artifacts/celm_interface_s0001_test_row0
```

## Output Artifacts
Each run writes:
- `manifest.json`
- `scout_summary.json`
- `background_findings.json`
- `event_findings.json`
- `parsed_context.json`
- `evidence_board.json`
- `detail.txt`
- `impression.txt`
- `verification.json` (optional)
- `run.log`
- `inference_trace.json`
- `agent_deliberations.json`
- `run_artifact_manifest.json`
- `method_audit.json`
- `method_audit.md`
