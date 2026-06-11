from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict


REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "evidence_gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "gap_id": {"type": "string"},
                    "evidence_target": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "reason": {"type": "string"},
                                    },
                "required": ["gap_id", "evidence_target", "severity", "reason"],
            },
        },
        "weak_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "weakness_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "target_type": {"type": "string", "enum": ["measurement", "evidence", "claim", "provenance", "slot"]},
                    "target_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "linked_measurement_ids": {"type": "array", "items": {"type": "string"}},
                                        "recommendation": {"type": "string"},
                },
                "required": [
                    "weakness_id",
                    "severity",
                    "target_type",
                    "target_id",
                    "reason",
                    "linked_measurement_ids",
                                        "recommendation",
                ],
            },
        },
        "missing_slots": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "slot_id": {"type": "string"},
                    "slot_name": {"type": "string"},
                    "target_module": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "reason": {"type": "string"},
                    "expected_evidence": {"type": "string"},
                                    },
                "required": [
                    "slot_id",
                    "slot_name",
                    "target_module",
                    "severity",
                    "reason",
                    "expected_evidence",
                                    ],
            },
        },
        "do_not_claim": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_id": {"type": "string"},
                    "text": {"type": "string"},
                    "rationale": {"type": "string"},
                                    },
                "required": ["item_id", "text", "rationale"],
            },
        },
        "claim_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "constraint_id": {"type": "string"},
                    "target": {"type": "string"},
                    "constraint": {"type": "string"},
                    "rationale": {"type": "string"},
                                    },
                "required": ["constraint_id", "target", "constraint", "rationale"],
            },
        },
        "tool_request_proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "proposal_id": {"type": "string"},
                    "target_module": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "rationale": {"type": "string"},
                    "expected_measurement": {"type": "string"},
                    "linked_gap_ids": {"type": "array", "items": {"type": "string"}},
                                    },
                "required": [
                    "proposal_id",
                    "target_module",
                    "tool_name",
                    "rationale",
                    "expected_measurement",
                    "linked_gap_ids",
                                    ],
            },
        },
    },
    "required": [
        "summary",
        "evidence_gaps",
        "weak_evidence",
        "missing_slots",
        "do_not_claim",
        "claim_constraints",
        "tool_request_proposals",
    ],
}


REPORT_SYNTHESIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "report_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "section_name": {"type": "string"},
                    "section_text": {"type": "string"},
                    "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_limitations": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "section_name",
                    "section_text",
                    "supporting_evidence_ids",
                    "evidence_limitations",
                ],
            },
        },
        "global_limitations": {"type": "array", "items": {"type": "string"}},
        "raw_eeg_used": {"type": "boolean"},
        "gt_report_used": {"type": "boolean"},
    },
    "required": ["report_sections", "global_limitations", "raw_eeg_used", "gt_report_used"],
}


EVIDENCE_GROUPING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "evidence_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "evidence_id": {"type": "string"},
                    "clinical_target": {
                        "type": "string",
                        "enum": [
                            "pdr",
                            "background_slowing",
                            "background_amplitude",
                            "excess_beta",
                            "epileptiform_morphology",
                            "event_candidate",
                            "seizure_evidence",
                            "localization",
                            "state",
                            "protocol",
                            "artifact",
                            "uncertainty",
                            "context",
                            "unknown",
                        ],
                    },
                    "evidence_type": {
                        "type": "string",
                        "enum": ["direct", "proxy", "metadata", "debug", "derived", "llm_assisted"],
                    },
                    "value_summary": {"type": "string"},
                    "linked_measurement_ids": {"type": "array", "items": {"type": "string"}},
                    "allowed_sections": {"type": "array", "items": {"type": "string"}},
                    "clinical_knowledge_reference": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "reference_type": {
                                "type": "string",
                                "enum": ["provided_guideline", "required_but_not_provided", "internal_reporting_standard"],
                            },
                            "statement": {"type": "string"},
                        },
                        "required": ["reference_type", "statement"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "evidence_id",
                    "clinical_target",
                    "evidence_type",
                    "value_summary",
                    "linked_measurement_ids",
                    "allowed_sections",
                    "clinical_knowledge_reference",
                    "rationale",
                ],
            },
        },
        "raw_eeg_used": {"type": "boolean"},
        "gt_report_used": {"type": "boolean"},
    },
    "required": ["summary", "evidence_groups", "raw_eeg_used", "gt_report_used"],
}


CLAIM_PLANNING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "atomic_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "plan_id": {"type": "string"},
                    "claim_type": {"type": "string"},
                    "proposed_text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "surface_action": {
                        "type": "string",
                        "enum": ["allow", "caveat", "block", "debug_only"],
                    },
                    "allowed_sections": {"type": "array", "items": {"type": "string"}},
                    "required_evidence": {"type": "array", "items": {"type": "string"}},
                    "missing_evidence": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "plan_id",
                    "claim_type",
                    "proposed_text",
                    "evidence_ids",
                    "surface_action",
                    "allowed_sections",
                    "required_evidence",
                    "missing_evidence",
                    "rationale",
                ],
            },
        },
        "raw_eeg_used": {"type": "boolean"},
        "gt_report_used": {"type": "boolean"},
    },
    "required": ["summary", "atomic_claims", "raw_eeg_used", "gt_report_used"],
}


CLINICAL_NEUROPHYSIOLOGIST_SYSTEM = (
    "You are an expert clinical neurophysiologist specializing in EEG interpretation "
    "and clinical report generation. You work inside an evidence-grounded assistive "
    "AI pipeline for long-duration clinical EEG review."
)


REPORT_SYNTHESIS_STYLE_EXAMPLE = (
    "CELM-STYLE REPORT GENERATION FORMAT EXAMPLE\n"
    "This example defines report-generation style and input organization only. It is not "
    "patient evidence and must not introduce claims that are absent from the payload.\n\n"
    "EEGCHANNELS\n"
    "['C3', 'C4', 'O1', 'O2', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2', "
    "'Fpz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2']\n\n"
    "TASK\n"
    "Your task is to generate the specified sections (**SECTIONS TO BE GENERATED**) "
    "of a formal clinical EEG report using the above provided data of EEG recording "
    "sessions and the following information:\n"
    "- Patient history\n"
    "- EEG description\n"
    "- EEG channels\n\n"
    "EEGSECTIONDESCRIPTIONS [STANDARDIZED_SECTION_DESCRIPTIONS]\n"
    "e.g. EEG DESCRIPTION/DETAILS: Detailed narrative of EEG findings including "
    "background activity, sleep stages, physiologic variants, and abnormalities observed "
    "during the recording period.\n\n"
    "GUIDELINES\n"
    "- Do NOT generate any additional sections.\n"
    "- Do NOT repeat the same section more than once.\n"
    "- Do NOT include preamble, markdown, explanation, or audit text.\n"
    "- Do NOT invent unsupported findings; use only supplied allowed/caveated claims.\n\n"
    "OUTPUT FORMAT (STRICT)\n"
    "SECTIONS TO BE GENERATED [SECTION_NAMES]\n"
    "- Generate only the sections listed in **SECTIONS TO BE GENERATED**.\n"
    "- Only generate the output in JSON format and do not include any other text.\n"
    "Return ONLY the following JSON structure:\n"
    "{\n"
    '  "report_sections": [\n'
    "    {\n"
    '      "section_name": "Name of the section as given in SECTIONS TO BE GENERATED",\n'
    '      "section_text": "Generated text for the section as a string"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "e.g. ['EEG DESCRIPTION/DETAILS']\n\n"
    "PATIENT HISTORYANDEEGDESCRIPTION [PATIENT_HISTORY_AND_EEG_DESCRIPTION]\n"
    "e.g. age: 77.0, gender: Female, indication: patient evaluated for transient "
    "altered awareness. pertinent medications: provided medication list if available.\n\n"
    "Now generate the EEG report."
)


def _clinical_stage_prompt_text(
    *,
    task: str,
    provided_data: list[str],
    guidelines: list[str],
    output_format: str,
    pipeline_position: str,
    previous_stage: str,
    next_stage: str,
    payload: Dict[str, Any],
    style_example: str | None = None,
) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    payload_json = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        f"{CLINICAL_NEUROPHYSIOLOGIST_SYSTEM}\n\n"
        f"TASK\n{task}\n\n"
        f"PIPELINE POSITION\n{pipeline_position}\n\n"
        f"PREVIOUS STAGE\n{previous_stage}\n\n"
        f"NEXT STAGE\n{next_stage}\n\n"
        f"PROVIDED DATA\n{bullets(provided_data)}\n\n"
        + (f"{style_example}\n\n" if style_example else "")
        + (
        f"GUIDELINES\n{bullets(guidelines)}\n\n"
        f"OUTPUT FORMAT (STRICT)\n{output_format}\n\n"
        f"PAYLOAD\n{payload_json}\n\n"
        "Return only the required JSON object. Do not include preamble, markdown, or explanation."
        )
    )



class OpenAIEvidenceReviewAdapter:
    """Structured OpenAI adapter for evidence-board-only review.

    This adapter must never receive raw EEG arrays, pkl payloads, or GT report text.
    """

    def __init__(self, model: str | None = None, timeout_sec: int = 60) -> None:
        self.model = model or os.getenv("OPENAI_EVIDENCE_REVIEW_MODEL", "gpt-4o-mini")
        self.timeout_sec = timeout_sec

    def review(self, evidence_payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        prompt = _clinical_stage_prompt_text(
            task=(
                "Review the structured EEG evidence as a clinical neurophysiologist before report "
                "claim planning. Identify weak support, missing clinically relevant evidence, "
                "do-not-claim constraints, and bounded local tool suggestions."
            ),
            pipeline_position="Post-evidence-board clinical review before atomic claim planning.",
            previous_stage="Bounded EEG tools generated MeasurementValue records and the evidence board assembled them.",
            next_stage="The claim planner will use this review only as audit and constraint context.",
            provided_data=[
                "Structured measurement summaries",
                "Measurement provenance summaries",
                "Tool invocation summaries",
                "Available bounded local tool registry",
            ],
            guidelines=[
                "Do not infer new EEG evidence or clinical claims.",
                "Do not request tools outside available_tools.",
                "Do not use raw EEG or GT report text; they are not present in this payload.",
                "Do not convert event candidates into definite epileptiform discharges unless morphology evidence is present.",
                "Do not convert global background measurements into focal/lateralized claims unless spatial provenance is present.",
                "Prefer calibrated uncertainty and do_not_claim records over unsupported clinical claims.",
                "Propose additional local evidence collection only when it would strengthen provenance or uncertainty handling.",
            ],
            output_format=(
                "Return only JSON matching eeg_evidence_review: summary, evidence_gaps, weak_evidence, "
                "missing_slots, do_not_claim, claim_constraints, and tool_request_proposals."
            ),
            payload=evidence_payload,
        )
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": CLINICAL_NEUROPHYSIOLOGIST_SYSTEM,
                },
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "eeg_evidence_review",
                    "strict": True,
                    "schema": REVIEW_SCHEMA,
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI evidence review failed: {exc.code}: {detail}") from exc

        text = self._extract_text(data)
        if not text:
            raise RuntimeError("OpenAI evidence review returned no text payload")
        return json.loads(text)

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]

        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        return "".join(chunks)


class OpenAIReportSynthesisAdapter:
    """Structured OpenAI adapter for EvidenceBoard-only report synthesis.

    This adapter is method variant D: the model may organize and verbalize the
    EvidenceBoard, but it must not receive raw EEG arrays or reference report text.
    """

    def __init__(self, model: str | None = None, timeout_sec: int = 90) -> None:
        self.model = model or os.getenv("OPENAI_REPORT_SYNTHESIS_MODEL", "gpt-4o-mini")
        self.timeout_sec = timeout_sec

    def synthesize(self, evidence_payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        prompt = _clinical_stage_prompt_text(
            task=(
                "Generate the specified sections of a formal clinical EEG report using only the "
                "provided allowed or caveated atomic claims."
            ),
            pipeline_position="Final neurologist-facing report wording after evidence and surface gating.",
            previous_stage="Surface gating selected reportable AtomicClaimPlan entries.",
            next_stage="Final prose auditing will check numeric provenance, debug leakage, seizure gating, and section consistency.",
            provided_data=[
                "Patient history and EEG description clinical context",
                "Requested section names",
                "Allowed or caveated AtomicClaimPlan entries",
                "SurfaceDecision summaries",
                "Evidence limitations attached to those claims",
            ],
            guidelines=[
                "Use only the allowed or caveated atomic_claim_plans in the payload.",
                "Follow the CELM-style section-description and patient-history formatting example for report style only.",
                "Use the payload section_descriptions to understand each requested section's expected content.",
                "When an atomic claim has reportable_evidence_values or linked_reportable_evidence values, preserve clinically meaningful values and units in the report sentence unless the value is marked unknown or unsafe.",
                "If proposed_text is generic but linked reportable evidence contains a safe numeric value, rewrite the sentence to include that value with its unit and the same caveat level.",
                "Do not infer new EEG evidence or clinical claims from general medical knowledge.",
                "Do not claim definite epileptiform discharges or seizures when evidence says event candidates only.",
                "Do not claim focality/laterality without spatial provenance.",
                "Do not verbalize internal detector scores, proxy labels, or raw reviewer/audit text.",
                "Do not add normality, posterior predominance, symmetry, organization, reactivity, seizure absence, or localization details unless they are explicitly present in the supplied claim/evidence payload.",
                "Do not fill empty requested sections with plausible clinical statements; use the conservative fallback when no claim is available for that section.",
                "Do not mention raw EEG review, GT/reference report text, or unavailable context.",
                "Return exactly the requested section names where possible.",
                "If no atomic claim plan is available for a section, use a conservative empty-evidence statement.",
            ],
            output_format=(
                "Return only JSON matching eeg_evidence_report_synthesis: report_sections, "
                "global_limitations, raw_eeg_used=false, and gt_report_used=false."
            ),
            payload=evidence_payload,
            style_example=REPORT_SYNTHESIS_STYLE_EXAMPLE,
        )
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": CLINICAL_NEUROPHYSIOLOGIST_SYSTEM,
                },
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "eeg_evidence_report_synthesis",
                    "strict": True,
                    "schema": REPORT_SYNTHESIS_SCHEMA,
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI report synthesis failed: {exc.code}: {detail}") from exc

        text = self._extract_text(data)
        if not text:
            raise RuntimeError("OpenAI report synthesis returned no text payload")
        result = json.loads(text)
        result["_response_id"] = data.get("id")
        return result

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]

        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        return "".join(chunks)


class OpenAIEvidenceGroupingAdapter:
    """LLM adapter for grouping typed measurements into EvidenceItems."""

    def __init__(self, model: str | None = None, timeout_sec: int = 90) -> None:
        self.model = model or os.getenv("OPENAI_EVIDENCE_GROUPING_MODEL", "gpt-4o-mini")
        self.timeout_sec = timeout_sec

    def group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        prompt = _clinical_stage_prompt_text(
            task=(
                "Convert the provided typed EEG measurements into patient-specific clinical EvidenceItems. "
                "This mirrors the pre-report reasoning step where a clinical neurophysiologist separates "
                "measured observations from reportable prose."
            ),
            pipeline_position="Measurement-to-evidence clinical reasoning stage.",
            previous_stage="Bounded signal/status tools generated MeasurementValue records from EEG-derived statistics and context.",
            next_stage="The claim planner will convert EvidenceItems into AtomicClaimPlan entries.",
            provided_data=[
                "Patient history and EEG description clinical context",
                "Typed EEG measurement summaries",
                "Measurement names, values, status fields, and metadata",
                "Time, channel, region, and laterality provenance summaries",
                "Allowed clinical targets, evidence types, and report sections",
            ],
            guidelines=[
                "Use only measurement IDs present in the payload.",
                "Do not inspect, request, or infer from raw EEG; raw EEG is not present.",
                "Do not use GT/reference report text; it is not present.",
                "Do not create seizure_evidence from event_candidate measurements.",
                "Do not call global or boundary 0.5 Hz activity a PDR.",
                "Candidate burden, likelihood, support score, ratios, and train duration are proxy/debug unless combined with appropriate clinical evidence.",
                "Keep evidence groups compact: group related measurements into clinical targets such as pdr, background_slowing, localization, protocol, state.",
                "Attach a clinical_knowledge_reference for each group. If no source is provided, use required_but_not_provided rather than inventing a citation.",
            ],
            output_format=(
                "Return only JSON matching eeg_evidence_grouping: summary, evidence_groups, "
                "raw_eeg_used=false, and gt_report_used=false. Do not generate report prose."
            ),
            payload=payload,
        )
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": CLINICAL_NEUROPHYSIOLOGIST_SYSTEM,
                },
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "eeg_evidence_grouping",
                    "strict": True,
                    "schema": EVIDENCE_GROUPING_SCHEMA,
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI evidence grouping failed: {exc.code}: {detail}") from exc

        text = self._extract_text(data)
        if not text:
            raise RuntimeError("OpenAI evidence grouping returned no text payload")
        result = json.loads(text)
        result["_response_id"] = data.get("id")
        return result

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        return "".join(chunks)


class OpenAIClaimPlanningAdapter:
    """LLM adapter for EvidenceItem-to-AtomicClaimPlan planning."""

    def __init__(self, model: str | None = None, timeout_sec: int = 90, max_retries: int = 3) -> None:
        self.model = model or os.getenv("OPENAI_CLAIM_PLANNING_MODEL", "gpt-4o-mini")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        prompt = _clinical_stage_prompt_text(
            task=(
                "Convert patient-specific EvidenceItems into minimal AtomicClaimPlan candidates. "
                "This mirrors the clinical step of deciding which evidence can support a concise "
                "report claim, which evidence requires caveat, and which evidence must remain non-reportable."
            ),
            pipeline_position="Evidence-to-atomic-claim clinical reasoning stage.",
            previous_stage="The evidence organizer produced typed EvidenceItems with provenance and reportability context.",
            next_stage="The surface gate will make the final allow/caveat/block/debug_only decision before report prose.",
            provided_data=[
                "Patient history and EEG description clinical context",
                "Typed EvidenceItems",
                "Evidence values and normalized values",
                "Time and space provenance",
                "Allowed sections and allowed surface actions",
            ],
            guidelines=[
                "Use only evidence_ids present in the payload.",
                "Do not inspect, request, or infer from raw EEG; raw EEG is not present.",
                "Do not use GT/reference report text; it is not present.",
                "Include clinically meaningful numeric values only when they are present in linked EvidenceItems.",
                "Do not mention candidate burden, burden ratio, support score, likelihood score, field concentration ratio, laterality index, bifrontal ratio, ratio of, train duration, slowing score, score of, alpha ratio, symmetry score, confidence score, confidence assessment, confidence in this assessment, confidence in the determination, support being marked, analyzed scores, concentration ratios, missing_slots, or values_preview.",
                "Do not create seizure claims unless linked evidence has clinical_target=seizure_evidence and evidence_type is direct, derived, or metadata.",
                "Do not call global or boundary 0.5 Hz activity a PDR.",
                "If an EvidenceItem has clinical_target=pdr with frequency_hz in 8-13 Hz and pdr_supported=true, create an allow/caveat PDR claim using the frequency value.",
                "If an EvidenceItem has clinical_target=background_amplitude with a uV range, create an allow/caveat amplitude claim using the range.",
                "Do not turn background_dominant_frequency_hz=0.5 Hz, slowing_score, beta_excess_score, or other internal/proxy measurements into allow claims.",
                "Use caveated wording when state, morphology, localization, or reactivity support is incomplete.",
                "Return block/debug_only for proxy or internal evidence that cannot safely surface.",
            ],
            output_format=(
                "Return only JSON matching eeg_claim_planning: summary, atomic_claims, "
                "raw_eeg_used=false, and gt_report_used=false. Do not generate full report sections."
            ),
            payload=payload,
        )
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": CLINICAL_NEUROPHYSIOLOGIST_SYSTEM,
                },
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "eeg_claim_planning",
                    "strict": True,
                    "schema": CLAIM_PLANNING_SCHEMA,
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"OpenAI claim planning failed: {exc.code}: {detail}") from exc

        text = self._extract_text(data)
        if not text:
            raise RuntimeError("OpenAI claim planning returned no text payload")
        result = json.loads(text)
        result["_response_id"] = data.get("id")
        return result

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        return "".join(chunks)
