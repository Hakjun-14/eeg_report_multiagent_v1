from __future__ import annotations

import json
import os
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
                    "finding_type": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "reason": {"type": "string"},
                    "linked_finding_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["gap_id", "finding_type", "severity", "reason", "linked_finding_ids"],
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
                    "target_type": {"type": "string", "enum": ["measurement", "finding", "claim", "provenance", "slot"]},
                    "target_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "linked_measurement_ids": {"type": "array", "items": {"type": "string"}},
                    "linked_finding_ids": {"type": "array", "items": {"type": "string"}},
                    "recommendation": {"type": "string"},
                },
                "required": [
                    "weakness_id",
                    "severity",
                    "target_type",
                    "target_id",
                    "reason",
                    "linked_measurement_ids",
                    "linked_finding_ids",
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
                    "linked_finding_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "slot_id",
                    "slot_name",
                    "target_module",
                    "severity",
                    "reason",
                    "expected_evidence",
                    "linked_finding_ids",
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
                    "linked_finding_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["item_id", "text", "rationale", "linked_finding_ids"],
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
                    "linked_finding_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["constraint_id", "target", "constraint", "rationale", "linked_finding_ids"],
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
                    "linked_finding_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "proposal_id",
                    "target_module",
                    "tool_name",
                    "rationale",
                    "expected_measurement",
                    "linked_gap_ids",
                    "linked_finding_ids",
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
                    "supporting_finding_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_limitations": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "section_name",
                    "section_text",
                    "supporting_finding_ids",
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


FINDING_PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "finding_proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "proposal_id": {"type": "string"},
                    "finding_type": {"type": "string"},
                    "assertion": {
                        "type": "string",
                        "enum": ["present", "absent", "not_observed", "not_performed", "no_response", "unknown"],
                    },
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                    "linked_measurement_ids": {"type": "array", "items": {"type": "string"}},
                    "provenance_policy": {"type": "string"},
                },
                "required": [
                    "proposal_id",
                    "finding_type",
                    "assertion",
                    "confidence",
                    "rationale",
                    "linked_measurement_ids",
                    "provenance_policy",
                ],
            },
        },
        "raw_eeg_used": {"type": "boolean"},
        "gt_report_used": {"type": "boolean"},
    },
    "required": ["summary", "finding_proposals", "raw_eeg_used", "gt_report_used"],
}



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

        prompt = {
            "task": (
                "Review structured EEG evidence and return typed JSON for weak evidence, missing slots, "
                "claim constraints, and bounded local tool suggestions."
            ),
            "constraints": [
                "Do not infer new EEG findings.",
                "Do not request tools outside available_tools.",
                "Do not use raw EEG or GT report text; they are not present in this payload.",
                "Do not convert event candidates into definite epileptiform discharges unless morphology evidence is present.",
                "Do not convert global background measurements into focal/lateralized claims unless spatial provenance is present.",
                "Prefer calibrated uncertainty and do_not_claim records over unsupported clinical claims.",
                "Propose additional local evidence collection only when it would strengthen provenance or uncertainty handling.",
            ],
            "payload": evidence_payload,
        }
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": "You are an evidence review policy for clinical EEG assistive AI. You only inspect structured evidence.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
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

        prompt = {
            "task": (
                "Generate only the requested clinical EEG report sections from structured evidence. "
                "Use concise formal report language, but preserve uncertainty and evidence limitations."
            ),
            "constraints": [
                "Use only the allowed or caveated atomic_claim_plans in the payload.",
                "Do not infer new EEG findings from general medical knowledge.",
                "Do not claim definite epileptiform discharges or seizures when evidence says event candidates only.",
                "Do not claim focality/laterality without spatial provenance.",
                "Do not verbalize internal detector scores, proxy labels, or raw reviewer/audit text.",
                "Do not mention raw EEG review, GT/reference report text, or unavailable context.",
                "Return exactly the requested section names where possible.",
                "If no atomic claim plan is available for a section, use a conservative empty-evidence statement.",
            ],
            "payload": evidence_payload,
        }
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical EEG report synthesis policy. You only verbalize structured "
                        "evidence for an assistive AI system and never inspect raw EEG or reference reports."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
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


class OpenAIFindingProposalAdapter:
    """LLM ablation adapter for measurement-to-finding proposals only."""

    def __init__(self, model: str | None = None, timeout_sec: int = 60) -> None:
        self.model = model or os.getenv("OPENAI_FINDING_PROPOSAL_MODEL", "gpt-4o-mini")
        self.timeout_sec = timeout_sec

    def propose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        prompt = {
            "task": "Propose structured EEG finding labels from typed local measurements only.",
            "constraints": [
                "Do not inspect or request raw EEG.",
                "Do not use GT/reference report text.",
                "Every proposal must link to existing measurement IDs.",
                "Use only allowed_finding_types.",
                "Prefer unknown/absent over present when evidence is weak.",
                "Do not create seizure or definite epileptiform findings unless the measurement summary explicitly supports them.",
            ],
            "payload": payload,
        }
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": "You map typed EEG measurements to conservative structured finding proposals.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "eeg_finding_proposals",
                    "strict": True,
                    "schema": FINDING_PROPOSAL_SCHEMA,
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
            raise RuntimeError(f"OpenAI finding proposal failed: {exc.code}: {detail}") from exc

        text = self._extract_text(data)
        if not text:
            raise RuntimeError("OpenAI finding proposal returned no text payload")
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
