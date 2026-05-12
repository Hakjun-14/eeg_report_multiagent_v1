from eeg_report_multiagent.schemas.measurement import StatusSemantic
from eeg_report_multiagent.tools.parser.text_tools import status_semantics_extractor


def test_status_semantics_extractor() -> None:
    note = """
    sleep:
    patient awake then drowsy.
    hyperventilation:
    na
    photic_stimulation:
    no response to photic stimulation
    """
    out = status_semantics_extractor(note, source_ref="s")
    by_name = {m.measurement_name: m for m in out}

    assert by_name["state_awake"].status_value.status in {StatusSemantic.PRESENT, StatusSemantic.UNKNOWN}
    assert by_name["hyperventilation_status"].status_value.status == StatusSemantic.NOT_PERFORMED
    assert by_name["photic_stimulation_status"].status_value.status == StatusSemantic.NO_RESPONSE
