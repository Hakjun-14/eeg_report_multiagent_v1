# eeg_report_multiagent_v1 메인 실행 경로와 최소 읽기 파일

분석 기준일: 2026-06-16

## 결론

`eeg_report_multiagent_v1`의 기본 메인 엔트리포인트는 `pyproject.toml`의 console script인 `eeg-run-session = "eeg_report_multiagent.cli.run_session:main"`이다. 실제 1개 세션 분석은 `src/eeg_report_multiagent/cli/run_session.py`의 `main()`에서 시작하고, 핵심 실행 함수는 `src/eeg_report_multiagent/graph/builder.py`의 `run_pipeline()`이다.

CELM split 기준 실행은 wrapper CLI가 한 단계 더 있다.

- 단일 CELM row/session: `eeg-run-celm-split-session -> run_celm_split_session.main() -> subprocess로 run_session.main()`
- CELM batch: `eeg-run-celm-split-batch -> run_celm_split_batch.main() -> row별 run_celm_split_session.main()`

## 공식 엔트리포인트

`pyproject.toml:38`의 `[project.scripts]`에 등록된 주요 실행 명령은 다음이다.

- 기본 분석: `eeg-run-session`
- 데이터 검사: `eeg-inspect-data`
- smoke 실행: `eeg-run-smoke-session`
- CELM split 검사: `eeg-inspect-celm-split`
- CELM 단일 실행: `eeg-run-celm-split-session`
- CELM batch 실행: `eeg-run-celm-split-batch`
- 산출물 감사: `eeg-audit-method-run`, `eeg-audit-section-contracts`, `eeg-run-batch-final-prose-audit` 등

## 기본 실행 흐름

```text
eeg-run-session
  -> eeg_report_multiagent.cli.run_session:main
  -> argparse로 session/context/output 설정
  -> run_pipeline(state, use_langgraph=...)
  -> build_graph() 또는 sequential fallback
  -> graph nodes 순차 실행
     1. load_inputs_node
     2. scout_pass_node
     3. background_module_node
     4. event_module_node
     5. protocol_parser_node
     6. evidence_merge_node
     7. evidence_review_node
     8. report_synthesize_node
     9. optional_verify_node
    10. finalize_node
  -> artifact JSON/TXT 저장
```

핵심 라인:

- `src/eeg_report_multiagent/cli/run_session.py:303` 기본 CLI `main()`
- `src/eeg_report_multiagent/cli/run_session.py:335` 초기 state 구성
- `src/eeg_report_multiagent/cli/run_session.py:354` `run_pipeline()` 호출
- `src/eeg_report_multiagent/graph/builder.py:88` LangGraph DAG 구성
- `src/eeg_report_multiagent/graph/builder.py:142` `run_pipeline()`
- `src/eeg_report_multiagent/graph/nodes.py:135` `load_inputs_node()`
- `src/eeg_report_multiagent/graph/nodes.py:310` `report_synthesize_node()`

## CELM 실행 흐름

```text
eeg-run-celm-split-batch
  -> run_celm_split_batch.main()
  -> split CSV row 순회
  -> subprocess: python -m eeg_report_multiagent.cli.run_celm_split_session

run_celm_split_session.main()
  -> load_celm_split_sample()
  -> study_context.json 생성
  -> subprocess: python -m eeg_report_multiagent.cli.run_session
  -> CELM-compatible generated_reports_json/txt 생성
  -> method_audit / section_contract_audit 생성
```

핵심 라인:

- `src/eeg_report_multiagent/cli/run_celm_split_batch.py:132` batch CLI `main()`
- `src/eeg_report_multiagent/cli/run_celm_split_batch.py:262` row별 subprocess 실행
- `src/eeg_report_multiagent/cli/run_celm_split_session.py:138` CELM 단일 CLI `main()`
- `src/eeg_report_multiagent/cli/run_celm_split_session.py:218` `run_session` subprocess 실행

## 최소 읽기 파일 1단계: 메인/그래프

먼저 아래만 읽으면 실행의 뼈대가 잡힌다.

1. `pyproject.toml`
2. `src/eeg_report_multiagent/cli/run_session.py`
3. `src/eeg_report_multiagent/graph/__init__.py`
4. `src/eeg_report_multiagent/graph/builder.py`
5. `src/eeg_report_multiagent/graph/nodes.py`
6. `src/eeg_report_multiagent/graph/state.py`

## 최소 읽기 파일 2단계: 입력 IO

세션 폴더와 context/report metadata를 어떻게 읽는지 확인한다.

1. `src/eeg_report_multiagent/io/__init__.py`
2. `src/eeg_report_multiagent/io/session_loader.py`
3. `src/eeg_report_multiagent/io/pkl_reader.py`
4. `src/eeg_report_multiagent/io/manifest_builder.py`
5. `src/eeg_report_multiagent/io/report_reader.py`
6. CELM을 볼 경우 추가: `src/eeg_report_multiagent/io/celm_dataset.py`

## 최소 읽기 파일 3단계: Agent/Tool 실행

실제 signal tool 선택과 실행 경계를 본다.

1. `src/eeg_report_multiagent/agents/__init__.py`
2. `src/eeg_report_multiagent/agents/background_agent.py`
3. `src/eeg_report_multiagent/agents/event_agent.py`
4. `src/eeg_report_multiagent/tools/__init__.py`
5. `src/eeg_report_multiagent/tools/registry.py`
6. `src/eeg_report_multiagent/tools/common.py`
7. `src/eeg_report_multiagent/tools/background/signal_tools.py`
8. `src/eeg_report_multiagent/tools/event/signal_tools.py`
9. `src/eeg_report_multiagent/tools/parser/text_tools.py`

## 최소 읽기 파일 4단계: Module/Report 핵심

측정값이 증거 보드와 report section으로 바뀌는 지점을 본다.

1. `src/eeg_report_multiagent/modules/background_module.py`
2. `src/eeg_report_multiagent/modules/event_module.py`
3. `src/eeg_report_multiagent/modules/protocol_state_context_parser.py`
4. `src/eeg_report_multiagent/modules/evidence_board.py`
5. `src/eeg_report_multiagent/modules/evidence_item_adapter.py`
6. `src/eeg_report_multiagent/modules/evidence_reviewer.py`
7. `src/eeg_report_multiagent/modules/llm_evidence_grouper.py`
8. `src/eeg_report_multiagent/modules/llm_claim_planner.py`
9. `src/eeg_report_multiagent/modules/report_synthesizer.py`
10. `src/eeg_report_multiagent/modules/surface_policy.py`
11. `src/eeg_report_multiagent/modules/section_router.py`
12. `src/eeg_report_multiagent/modules/claim_verifier.py`
13. `src/eeg_report_multiagent/modules/final_prose_auditor.py`

## 최소 읽기 파일 5단계: Schema

v1은 schema 중심 구조이므로, report 품질을 보려면 schema를 읽어야 한다.

1. `src/eeg_report_multiagent/schemas/measurement.py`
2. `src/eeg_report_multiagent/schemas/evidence.py`
3. `src/eeg_report_multiagent/schemas/shared_evidence.py`
4. `src/eeg_report_multiagent/schemas/tooling.py`
5. `src/eeg_report_multiagent/schemas/report.py`
6. `src/eeg_report_multiagent/schemas/agent.py`
7. `src/eeg_report_multiagent/schemas/provenance.py`
8. `src/eeg_report_multiagent/schemas/section_contract.py`

## 최소 실행 패키지 관점

기본 `eeg-run-session`을 실행 가능한 코드 단위로 압축하면 다음 묶음이다.

```text
pyproject.toml
src/eeg_report_multiagent/__init__.py
src/eeg_report_multiagent/cli/__init__.py
src/eeg_report_multiagent/cli/run_session.py
src/eeg_report_multiagent/graph/*.py
src/eeg_report_multiagent/io/{__init__,manifest_builder,pkl_reader,report_reader,session_loader}.py
src/eeg_report_multiagent/agents/*.py
src/eeg_report_multiagent/tools/**/*.py
src/eeg_report_multiagent/modules/*.py
src/eeg_report_multiagent/schemas/*.py
src/eeg_report_multiagent/llm/*.py
```

CELM 실행까지 포함하면 아래가 추가된다.

```text
src/eeg_report_multiagent/cli/run_celm_split_session.py
src/eeg_report_multiagent/cli/run_celm_split_batch.py
src/eeg_report_multiagent/cli/inspect_celm_split.py
src/eeg_report_multiagent/io/celm_dataset.py
src/eeg_report_multiagent/evaluation/method_audit.py
src/eeg_report_multiagent/evaluation/section_contract_audit.py
```

## 읽기 우선순위

1. `src/eeg_report_multiagent/cli/run_session.py:303`부터 읽기
2. `src/eeg_report_multiagent/graph/builder.py:88`부터 graph 구조 읽기
3. `src/eeg_report_multiagent/graph/nodes.py:135`부터 node별 state 변환 읽기
4. `src/eeg_report_multiagent/modules/background_module.py`와 `event_module.py` 읽기
5. `src/eeg_report_multiagent/tools/background/signal_tools.py`와 `tools/event/signal_tools.py` 읽기
6. `src/eeg_report_multiagent/modules/evidence_board.py`와 `evidence_item_adapter.py` 읽기
7. `src/eeg_report_multiagent/modules/report_synthesizer.py`와 `surface_policy.py` 읽기
8. CELM을 돌릴 경우 `run_celm_split_session.py`, `run_celm_split_batch.py`, `io/celm_dataset.py` 읽기

## 현재 확인된 실행상 주의점

1. 기본 입력은 EDF가 아니라 processed EEG session directory다. `README.md` 기준으로 session은 `seg_*.pkl`을 포함하는 CELM-style processed 폴더다.
2. `--gt-report-json`은 evaluation bookkeeping용이며 inference context가 아니다.
3. LLM 관련 옵션은 선택형이다: `--enable-llm-evidence-grouping`, `--enable-llm-claim-planning`, `--enable-llm-review`.
4. `--no-langgraph`를 주면 LangGraph 없이 동일 node 순서의 sequential fallback으로 실행된다.
5. `modules/__init__.py`가 여러 module을 eager import하므로 core 실행 압축본에는 `modules/*.py` 전체를 넣는 편이 안전하다.
