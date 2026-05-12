# Architecture (v1)

## Research Framing
This project is an assistive AI architecture for long-duration clinical EEG review:

Long-duration EEG -> structured clinical evidence -> neurologist-facing guidance.

It is not a direct sequence-to-report demo. The main contribution is task-specific
decomposition with typed evidence before any report text is rendered.

## Layering
1. Measurement layer: numeric/status typed values from tools
2. Finding layer: clinically meaningful assertions with provenance
3. Report layer: natural language generated only from evidence board

## Input Contract
- Inference inputs: processed EEG session, study metadata, and clinically available context.
- GT report text is not an inference input.
- GT report paths may be stored for evaluation bookkeeping only.
- Raw EEG and GT comparison must not be sent to external APIs.

## Orchestration
- LangGraph state tracks session refs, scout summaries, module outputs, evidence board, and report sections.
- Fallback sequential runner is provided for environments without LangGraph import.
- `--monitor` forces the sequential runner so node-level evidence board updates can be watched in the terminal.
- `--enable-llm-review` adds an optional evidence-board-only review node. It sends structured finding/tool summaries, not raw EEG or GT report text.

## Provenance Requirements
Each finding carries:
- Time provenance: window indices/time range
- Space provenance: channels/laterality/region
- Measurement provenance: tool/function and values used
- Claim provenance: link from finding to synthesized claim(s)

## Primary Artifacts
- `manifest.json`
- `evidence_board.json`
- `inference_trace.json`
- `run_artifact_manifest.json`

`detail.txt` and `impression.txt` are human-readable renderings, not the primary scientific evidence.

## LLM Boundary
LLM review is an optional policy layer for evidence gaps and local tool request proposals. It cannot create signal-derived findings and cannot call tools outside the bounded registries.
