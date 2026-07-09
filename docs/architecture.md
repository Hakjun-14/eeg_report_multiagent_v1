# Architecture contract (v1)

전체 피겨, 폴더 구조, 실행 명령, 산출물 지도는 [README](../README.md)를 기준으로 한다. 이 문서는 구현이 지켜야 하는 핵심 계약만 요약한다.

## Research framing

이 프로젝트는 장시간 임상 EEG 검토를 보조하는 evidence-first 연구 아키텍처다.

```text
processed EEG
→ local typed measurements
→ provenance-linked evidence
→ atomic claim plans
→ authoritative surface decisions
→ neurologist-facing report text
```

raw sequence를 곧바로 report text로 바꾸는 direct sequence-to-report 구조가 아니다.

## Actual runtime graph

기본 진입점은 `eeg-run-session → cli.run_session.main() → graph.builder.run_pipeline()`이다.

```text
load_inputs
→ scout_pass
→ background_module
→ event_module
→ protocol_parser
→ evidence_merge
→ evidence_review
→ report_synthesize
→ optional_verify
→ finalize
```

- 현재 LangGraph는 위 10개 노드를 순차 실행한다.
- `--no-langgraph` 또는 LangGraph import 실패 시에도 같은 순서의 sequential runner를 사용한다.
- `--monitor`는 sequential runner를 강제하지 않지만 callback이 연결된 각 node를 monitor용 daemon worker thread에서 실행한다. 순서와 반환 state는 동일하다.
- `evidence_review`는 기본적으로 skip된다.
- `optional_verify`는 기본 활성화되고 `--no-verify`에서 skip된다.
- file writing과 `FinalProseAuditor`는 graph 반환 뒤 `run_session.py`에서 실행된다.

## Typed layers

1. `EEGSessionData / SessionManifest`: processed PKL session과 time/channel manifest
2. `MeasurementValue / ToolInvocationRecord`: bounded deterministic tool의 typed output과 호출 trace
3. `EvidenceItem / SharedEvidenceBoard`: 임상 target별 patient-specific evidence와 provenance
4. `AtomicClaimPlan`: report surface 전에 계획된 원자 주장
5. `SurfaceDecision`: `allow / caveat / block / debug_only`와 section routing의 authoritative 판단
6. `ReportSection / ClaimRecord`: 최종 Detail, Impression, claim-evidence link

`evidence_board.json`은 호환성 이름을 유지한 `RuntimeEvidenceBundle`이며 full `SharedEvidenceBoard`와 `claim_evidence_links`를 포함한다. `shared_evidence_board.json`은 links를 제외한 `EvidenceBoardSnapshot`이다.

## Bounded agent and tool contract

- `BackgroundAgent`와 `EventAgent`는 자율 LLM agent가 아니라 rule-based tool selector다.
- 각 selector는 자신의 hard-coded `ToolRegistry`에 등록된 함수만 호출할 수 있다.
- open-ended tool discovery, arbitrary code execution, external signal tool call은 허용하지 않는다.
- raw EEG는 `load_inputs`, `scout_pass`, background/event local code 안에서만 접근한다.
- tool invocation artifact에는 성공한 dispatch만 남는다. 실패 dispatch는 record를 구성한 뒤 예외를 raise하므로 module 결과에 append되지 않는다.

## LLM boundary

기본 `eeg-run-session`의 claim planning과 report rendering은 deterministic하다.

| 기능 | 활성화 | 의미 |
|---|---|---|
| Evidence grouping | `--enable-llm-evidence-grouping` | typed measurements를 추가 EvidenceItem으로 그룹화 |
| Evidence review | `--enable-llm-review` | gap/constraint/tool proposal을 audit-only record로 생성 |
| Claim planning | `--enable-llm-claim-planning` | deterministic AtomicClaimPlan을 LLM plan으로 override |
| CELM section rendering | CELM wrapper에서 자동 시도 | surface-approved plan을 문장화하고 실패 시 template fallback |

Grouping/planning adapter 오류는 core run으로 전파될 수 있다. Review는 API 오류를 `local_only` deliberation으로 바꾸고, CELM final renderer는 기존 atomic plan의 template rendering으로 fallback한다.

CELM-managed path 또는 caller가 input contract를 지킨 direct run의 inference API payload에는 다음을 포함하지 않는다.

- raw EEG arrays
- processed PKL payload 또는 source path
- GT/reference EEG target-section text
- unbounded tool access

- evidence grouping/review payload에는 typed proxy/debug-role measurement 값이 포함될 수 있다.
- claim planning은 EvidenceItem과 clinical context를 받고 이후 SurfaceDecision을 통과한다.
- CELM report synthesis는 surface-approved/caveated claim과 reportable evidence만 받는다.
- Evidence review의 tool proposal은 해당 run에서 자동 실행되지 않는다.
- `eeg-run-llm-judge-winrate`는 GT/generated text를 configurable endpoint 또는 local backend에 전달하는 post-hoc evaluation 예외이며 inference privacy contract의 대상이 아니다.
- Direct `eeg-run-session`은 이 계약을 완전히 강제하지 않는다. caller가 study-context 또는 deprecated report alias에 GT target text를 넣으면 parser와 LLM context로 흘러갈 수 있다.

## Input and evaluation separation

- inference 입력은 processed EEG session, study metadata, clinical context다.
- PKL loader는 `pickle.load()`를 사용하므로 신뢰할 수 있는 processed EEG 파일만 입력해야 한다.
- loader는 window 간 shape, sample rate, channel label/order 일관성을 검증하지 않고 첫 window의 channel 목록을 session 전체에 사용한다.
- GT report path는 evaluation bookkeeping으로 전달할 수 있다. manifest에는 availability boolean만 남고, `inference_trace.json`에는 path가 기록된다. GT/reference EEG target-section text는 inference에 사용할 수 없다.
- CELM report JSON의 clinical-history section은 safe context로 추출할 수 있다.
- patient history sanitizer는 LLM용 `clinical_context`에 적용된다.
- parser `note_text`에는 같은 sanitizer가 적용되지 않으므로 caller가 study-context와 deprecated report alias에 GT target text를 넣지 않아야 한다.
- GT/reference EEG target-section text가 필요한 metric, provenance, suppression, judge workflow는 core inference 뒤 별도 evaluation CLI에서 실행한다.

## Post-generation checks

- `ClaimVerifier`는 `ClaimRecord.linked_evidence_ids`가 아니라 SharedEvidenceBoard의 `claim_evidence_links` map과 evidence ID 존재를 검사한다. 임상 재판독기가 아니다.
- `c_impression_summary`는 항상 생성되지만 map에 link되지 않아 현재 기본 verifier에서 `MISSING`으로 기록된다.
- `FinalProseAuditor`는 numeric provenance, debug leakage, section leakage, seizure gate, claim trace를 검사한다.
- Final prose audit 판정은 report를 수정하거나 gate하지 않는다. 다만 auditor 호출 자체는 `try/except` 밖이므로 예상치 못한 예외는 CLI를 중단시킬 수 있다.
- auditor는 최종 `SurfaceDecision`이 아니라 `AtomicClaimPlan.surface_action`을 기준으로 claim trace를 판정한다. synthesis calibration이 plan을 최종 ALLOW/CAVEAT로 승격한 경우 `surface_policy_violation` false positive가 가능하므로 `surface_decisions.json`과 함께 해석한다.
- CELM wrapper는 final section rendering 뒤 `FinalProseAuditor`를 다시 실행해 core `final_prose_audit.json`을 덮어쓰고, section contract audit와 method audit를 생성한다.

## Source of truth

| 관심사 | 기준 코드 |
|---|---|
| graph와 node 순서 | `src/eeg_report_multiagent/graph/builder.py`, `graph/nodes.py` |
| runtime tool allowlist | `src/eeg_report_multiagent/tools/registry.py` |
| evidence 변환 | `modules/evidence_board.py`, `modules/evidence_item_adapter.py` |
| claim/surface/report | `modules/report_synthesizer.py`, `modules/surface_policy.py` |
| OpenAI payload/API | `llm/openai_adapter.py`, `modules/llm_*.py` |
| GT-aware post-hoc LLM judge | `cli/run_llm_judge_winrate.py` |

`configs/base.yaml`, `configs/graph.yaml`, `configs/tool_registry.yaml`은 현재 core runtime에서 로드하지 않는 참고 snapshot이다. `clinical_slot_schema.yaml`, `evaluation_failure_taxonomy.yaml`, `claim_gate_policy.yaml`은 clinical provenance evaluation에서만 읽으며 inference `SurfacePolicy`를 구성하지 않는다.
