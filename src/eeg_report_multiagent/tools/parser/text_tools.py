from __future__ import annotations

import re
from typing import Dict, List

from eeg_report_multiagent.schemas.measurement import MeasurementValue, StatusSemantic
from eeg_report_multiagent.schemas.provenance import SourceType
from eeg_report_multiagent.tools.common import make_provenance, make_status_measurement


SECTION_PATTERN = re.compile(r"^\s*([a-zA-Z_ ]+):\s*$", re.MULTILINE)


def report_section_splitter(note_text: str, source_ref: str) -> Dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(note_text))
    if not matches:
        return {"full_text": note_text}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        sec_name = m.group(1).strip().lower().replace(" ", "_")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note_text)
        sections[sec_name] = note_text[start:end].strip()
    return sections


def _status_from_text(text: str, positive_keywords: List[str], negative_keywords: List[str]) -> StatusSemantic:
    t = text.lower()
    if any(k in t for k in negative_keywords):
        return StatusSemantic.ABSENT
    if any(k in t for k in positive_keywords):
        return StatusSemantic.PRESENT
    return StatusSemantic.UNKNOWN


def status_semantics_extractor(note_text: str, source_ref: str) -> List[MeasurementValue]:
    t = note_text.lower()
    out: List[MeasurementValue] = []

    awake_status = StatusSemantic.PRESENT if "wake" in t or "awake" in t else StatusSemantic.UNKNOWN
    out.append(
        make_status_measurement(
            measurement_id="m_state_awake",
            measurement_name="state_awake",
            status=awake_status,
            provenance=make_provenance(
                "status_semantics_extractor",
                "status_semantics_extractor",
                source_ref,
                source_type=SourceType.REPORT_TEXT,
            ),
        )
    )

    drowsy_status = StatusSemantic.PRESENT if re.search(r"\bdrows", t) else StatusSemantic.UNKNOWN
    out.append(
        make_status_measurement(
            measurement_id="m_state_drowsy",
            measurement_name="state_drowsy",
            status=drowsy_status,
            provenance=make_provenance(
                "status_semantics_extractor",
                "status_semantics_extractor",
                source_ref,
                source_type=SourceType.REPORT_TEXT,
            ),
        )
    )

    sleep_status = (
        StatusSemantic.PRESENT
        if re.search(r"\bsleep\b|asleep|spindle|k-complex|vertex", t)
        else StatusSemantic.UNKNOWN
    )
    out.append(
        make_status_measurement(
            measurement_id="m_state_sleep",
            measurement_name="state_sleep",
            status=sleep_status,
            provenance=make_provenance(
                "status_semantics_extractor",
                "status_semantics_extractor",
                source_ref,
                source_type=SourceType.REPORT_TEXT,
            ),
        )
    )

    hv_status = StatusSemantic.UNKNOWN
    if re.search(r"hyperventilation[\s:_-]{0,20}[^.\n]{0,80}(not performed|was not performed|n/a|\bna\b)", t):
        hv_status = StatusSemantic.NOT_PERFORMED
    elif re.search(r"hyperventilation[\s:_-]{0,20}[^.\n]{0,80}(performed|effort|diffuse slowing|activation)", t):
        hv_status = StatusSemantic.PRESENT
    out.append(
        make_status_measurement(
            measurement_id="m_hyperventilation_status",
            measurement_name="hyperventilation_status",
            status=hv_status,
            provenance=make_provenance(
                "status_semantics_extractor",
                "status_semantics_extractor",
                source_ref,
                source_type=SourceType.REPORT_TEXT,
            ),
        )
    )

    photic_status = StatusSemantic.UNKNOWN
    if re.search(r"photic[\s:_-]{0,30}[^.\n]{0,80}(not performed|was not performed|n/a|\bna\b)", t):
        photic_status = StatusSemantic.NOT_PERFORMED
    elif re.search(r"(photic[\s:_-]{0,30}[^.\n]{0,100}(no|not|without)[^.\n]{0,80}(response|driving))|((no|not|without)[^.\n]{0,80}(response|driving)[^.\n]{0,80}photic)", t):
        photic_status = StatusSemantic.NO_RESPONSE
    elif re.search(r"photic[\s:_-]{0,30}[^.\n]{0,100}(performed|response|driving|stimulation)", t):
        photic_status = StatusSemantic.PRESENT
    out.append(
        make_status_measurement(
            measurement_id="m_photic_status",
            measurement_name="photic_stimulation_status",
            status=photic_status,
            provenance=make_provenance(
                "status_semantics_extractor",
                "status_semantics_extractor",
                source_ref,
                source_type=SourceType.REPORT_TEXT,
            ),
        )
    )

    return out


def metadata_normalizer(metadata: Dict[str, str], source_ref: str) -> List[MeasurementValue]:
    out: List[MeasurementValue] = []
    ekg_present = StatusSemantic.PRESENT if metadata.get("ekg_available", "").lower() in {"1", "true", "yes"} else StatusSemantic.UNKNOWN
    video_present = StatusSemantic.PRESENT if metadata.get("video_available", "").lower() in {"1", "true", "yes"} else StatusSemantic.UNKNOWN

    out.append(
        make_status_measurement(
            measurement_id="m_ekg_availability",
            measurement_name="ekg_availability",
            status=ekg_present,
            provenance=make_provenance(
                "metadata_normalizer",
                "metadata_normalizer",
                source_ref,
                source_type=SourceType.METADATA,
            ),
        )
    )
    out.append(
        make_status_measurement(
            measurement_id="m_video_availability",
            measurement_name="video_availability",
            status=video_present,
            provenance=make_provenance(
                "metadata_normalizer",
                "metadata_normalizer",
                source_ref,
                source_type=SourceType.METADATA,
            ),
        )
    )
    return out


def comparison_history_parser(note_text: str, source_ref: str) -> List[MeasurementValue]:
    t = note_text.lower()
    status = StatusSemantic.PRESENT if "comparison" in t or "prior" in t or "history" in t else StatusSemantic.UNKNOWN
    return [
        make_status_measurement(
            measurement_id="m_comparison_history_presence",
            measurement_name="comparison_history_presence",
            status=status,
            provenance=make_provenance(
                "comparison_history_parser",
                "comparison_history_parser",
                source_ref,
                source_type=SourceType.REPORT_TEXT,
            ),
        )
    ]
