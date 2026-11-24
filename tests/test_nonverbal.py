"""Referring to objects: every scenario's prompt teaches the marker, the client's object list is kept in the context, and a marker becomes a refer behaviour once the output transport passes it on."""

import json

from pipecat.frames.frames import AggregatedTextFrame, LLMMessagesTransformFrame, TTSTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from agent import scenarios
from agent.game import wrap
from agent.nonverbal import INSTRUCTIONS, NONVERBAL, OBJECTS_TAG, REFER, ReferRelay, SceneObjects, prompt
from agent.observability.study_log import StudyLog


class _RTVI:
    """Stand-in for the worker's RTVI processor, recording the server messages sent through it."""

    def __init__(self):
        """Start with nothing sent."""
        self.sent = []

    async def send_server_message(self, data):
        """Record one server message."""
        self.sent.append(data)


class _Worker:
    """Stand-in for the pipeline worker: applies each queued context transform to a message list, as the user aggregator would."""

    def __init__(self, messages: list):
        """Start from the given context messages."""
        self.messages = messages

    async def queue_frames(self, frames):
        """Apply each context transform and refuse to run the LLM."""
        for frame in frames:
            assert isinstance(frame, LLMMessagesTransformFrame)
            assert frame.run_llm is False
            self.messages = frame.transform(self.messages)


async def _push(relay: ReferRelay, source, frame):
    """Show the relay one frame pushed downstream by the given processor."""
    await relay.on_push_frame(FramePushed(source=source, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0))


def _marker(target: str) -> AggregatedTextFrame:
    """A marker as the reply strip passes it on."""
    return AggregatedTextFrame(text=target, text_type=REFER)


def _logged(path) -> list[dict]:
    """The study log's nonverbal lines, without their stamps."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [{key: line[key] for key in ("behaviour", "target")} for line in lines if line["event"] == NONVERBAL]


def test_every_scenario_prompt_gets_the_instructions():
    """The instructions follow each scenario's own prompt unchanged, so no scenario has to repeat them."""
    for scenario in scenarios.SCENARIOS.values():
        own = scenario.system_prompt("en")
        assert prompt(own) == f"{own}\n\n{INSTRUCTIONS}"


async def test_a_marker_from_the_output_transport_is_sent_and_logged(tmp_path):
    """A marker the output transport passes on becomes one refer behaviour for the client and one nonverbal line in the study log."""
    path = tmp_path / "study.jsonl"
    study = StudyLog(path)
    rtvi = _RTVI()
    relay = ReferRelay(rtvi, study)
    await _push(relay, BaseOutputTransport(TransportParams()), _marker("flask-3"))
    study.close()
    assert rtvi.sent == [{"t": NONVERBAL, "d": {"behaviour": REFER, "target": "flask-3"}}]
    assert _logged(path) == [{"behaviour": REFER, "target": "flask-3"}]


async def test_a_marker_anywhere_else_or_without_an_id_is_not_sent(tmp_path):
    """Only the output transport's push counts, so a marker is sent once and not before the words ahead of it are played; an empty marker, the reply's words and spoken text are never sent."""
    path = tmp_path / "study.jsonl"
    study = StudyLog(path)
    rtvi = _RTVI()
    relay = ReferRelay(rtvi, study)
    output = BaseOutputTransport(TransportParams())
    await _push(relay, FrameProcessor(), _marker("flask-3"))
    await _push(relay, output, _marker("  "))
    await _push(relay, output, AggregatedTextFrame(text="That one.", text_type="sentence"))
    await _push(relay, output, TTSTextFrame(text="flask-3", text_type=REFER))
    study.close()
    assert rtvi.sent == []
    assert _logged(path) == []


async def test_a_new_object_list_replaces_the_previous_one_last_in_the_context():
    """Each list the client sends, such as after a reconnect, takes the place of the one before it, appended last as a tagged developer message."""
    first = [{"id": "ice_bath", "kind": "object"}]
    second = [{"id": "ice_bath", "kind": "object"}, {"id": "bunsen_burner", "kind": "object"}]
    worker = _Worker([{"role": "user", "content": "Hi."}])
    objects = SceneObjects(worker.queue_frames)
    await objects.receive(first)
    await objects.receive(second)
    assert worker.messages == [
        {"role": "user", "content": "Hi."},
        {"role": "developer", "content": wrap(OBJECTS_TAG, second)},
    ]
