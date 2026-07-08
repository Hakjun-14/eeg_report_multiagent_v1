# Container Execution Standard

Use the existing `eeg-report-audit` container for MultiTI development, tests, audits, and OpenAI-backed CLI runs.

## Why

Local Python may miss dependencies such as `pydantic`, and Docker-generated artifacts can become root-owned. The wrapper below runs commands inside the container as the host uid/gid by default and provides a permission repair command.

## Wrapper

```bash
scripts/mt_container.sh status
scripts/mt_container.sh compile
scripts/mt_container.sh pytest -q
```

Run an arbitrary command inside `/workspace/eeg_report_multiagent_v1`:

```bash
scripts/mt_container.sh run 'python3 -m eeg_report_multiagent.cli.run_batch_final_prose_audit --help'
```

Run commands that need `.env` / `OPENAI_API_KEY`:

```bash
scripts/mt_container.sh run-env 'python3 -m eeg_report_multiagent.cli.run_evidence_direct_report_synthesis --help'
```

Open shell:

```bash
scripts/mt_container.sh shell
```

Fix root-owned artifacts/caches after older root-based runs:

```bash
scripts/mt_container.sh fix-perms
```

## Defaults

- Container: `eeg-report-audit`
- Container workdir: `/workspace/eeg_report_multiagent_v1`
- Host repo mount: `/home/hjlee/Desktop/eeg_report_multiagent_v1`
- Data mount: `/workspace/eeg_data`

Override when needed:

```bash
MULTITI_CONTAINER=other-container scripts/mt_container.sh status
MULTITI_WORKDIR_IN_CONTAINER=/workspace/eeg_report_multiagent_v1 scripts/mt_container.sh pytest -q
```

## Notes

- Prefer `run-env` for OpenAI-backed synthesis/evidence grouping runs.
- Prefer `pytest`, `compile`, and `run` over direct local Python.
- Avoid heredocs through `run`; use `python3 -c`, a temporary script file, or `shell` for multi-line scripts.
