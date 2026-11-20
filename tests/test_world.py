"""What the agent and the client tell each other outside of audio: each LLM thought, sent whole, and the client's inputs, logged as sent and handed to the inputs."""

import json

import pytest
from pipecat.frames.frames import LLMThoughtEndFrame, LLMThoughtStartFrame, LLMThoughtTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.models import ClientMessage

from agent.observability.study_log import StudyLog
from agent.world import THOUGHT, ThoughtRelay, attach


class _RTVI:
    """Stand-in for the worker's RTVI processor, recording the server messages sent through it."""

    def __init__(self):
        """Start with nothing sent."""
        self.sent = []

    async def send_server_message(self, data):
        """Record one server message."""
        self.sent.append(data)

    def event_handler(self, name):
        """Keep each handler registered for an event, so a test can raise the event."""

        def register(handler):
            self.handlers[name] = handler
            return handler

        return register


class _Worker:
    """Stand-in for the pipeline worker, holding the RTVI stand-in and ignoring observers."""

    def __init__(self):
        """Start with an RTVI stand-in that has no handlers yet."""
        self.rtvi = _RTVI()
        self.rtvi.handlers = {}

    def add_observer(self, observer):
        """Accept an observer without running it."""


class _Inputs:
    """Stand-in for the inputs, recording each client input handed to them."""

    def __init__(self):
        """Start with nothing received."""
        self.received = []

    async def receive(self, kind, data):
        """Record one client input."""
        self.received.append((kind, data))


async def _push(relay: ThoughtRelay, frame):
    """Show the relay one frame moving downstream."""
    await relay.on_push_frame(
        FramePushed(source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0)
    )


async def test_a_thought_reaches_the_client_whole_once_it_ends():
    """The client gets one server message per thought, not one per streamed chunk."""
    rtvi = _RTVI()
    relay = ThoughtRelay(rtvi)
    await _push(relay, LLMThoughtStartFrame())
    await _push(relay, LLMThoughtTextFrame(text="Ready,"))
    assert rtvi.sent == []
    await _push(relay, LLMThoughtTextFrame(text=" so start."))
    await _push(relay, LLMThoughtEndFrame())
    assert rtvi.sent == [{"t": THOUGHT, "d": {"text": "Ready, so start."}}]


@pytest.mark.parametrize("input_type", ["held", "pointing"])
async def test_a_client_input_is_logged_as_sent(tmp_path, input_type):
    """A held or pointing message from the client lands in the study log as a client_input line with its data unchanged."""
    path = tmp_path / "study.jsonl"
    study = StudyLog(path)
    worker = _Worker()
    inputs = _Inputs()
    attach(worker, greeting=None, study=study, settings={}, game=None, inputs=inputs)
    hands = {"left": None, "right": {"id": "flask-3", "kind": "flask", "shape": "beaker", "substance": None}}
    await worker.rtvi.handlers["on_client_message"](worker.rtvi, ClientMessage(msg_id="1", type=input_type, data=hands))
    study.close()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    logged = [line for line in lines if line["event"] == "client_input"]
    assert [(line["input"], line["data"]) for line in logged] == [(input_type, hands)]
    assert inputs.received == [(input_type, hands)]
