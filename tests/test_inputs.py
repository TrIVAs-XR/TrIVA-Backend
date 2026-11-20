"""Client inputs in the LLM context and the per-call input record: the draft before a call, what the call got and missed, and whether its reply was heard."""

import json

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMMessagesTransformFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import run_test

from agent.game import HEADER as GAME_HEADER
from agent.inputs import HEADERS, INPUTS, ContextCapture, Inputs
from agent.observability.study_log import StudyLog

HANDS = {"left": None, "right": {"id": "flask-3", "kind": "flask"}}
OTHER_HANDS = {"left": {"id": "flask-1", "kind": "flask"}, "right": None}


class _Worker:
    """Stand-in for the worker: applies context transforms as the user aggregator would."""

    def __init__(self):
        """Start from a context holding only the system prompt."""
        self.messages = [{"role": "system", "content": "You are a tutor."}]

    async def queue_frames(self, frames):
        """Apply any context transform among the frames."""
        for frame in frames:
            if isinstance(frame, LLMMessagesTransformFrame):
                self.messages = frame.transform(self.messages)


class _RTVI:
    """Stand-in for the worker's RTVI processor, recording the server messages sent through it."""

    def __init__(self):
        """Start with nothing sent."""
        self.sent = []

    async def send_server_message(self, data):
        """Record one server message."""
        self.sent.append(data)

    def records(self, state: str | None = None) -> list[dict]:
        """The input records sent, optionally only those in one state."""
        return [m["d"] for m in self.sent if m["t"] == INPUTS and (state is None or m["d"]["state"] == state)]


class _Processor:
    """Stand-in for a pipeline processor that is not the LLM."""


# Passes as the LLM for the record's isinstance check without building a real service.
_LLM = object.__new__(LLMService)


@pytest.fixture
def worker():
    """The stand-in worker the inputs queue into."""
    return _Worker()


@pytest.fixture
def rtvi():
    """The stand-in RTVI processor the records go through."""
    return _RTVI()


@pytest.fixture
def study(tmp_path):
    """A real study log in a temporary folder, closed and read back by the test that needs it."""
    log = StudyLog(tmp_path / "P01.jsonl")
    yield log
    log.close()


@pytest.fixture
def capture():
    """A context capture whose sides pass frames nowhere, since no pipeline surrounds them here."""
    pair = ContextCapture()

    async def drop(frame, direction):
        """Let the frame go."""

    pair.before_llm().push_frame = drop
    pair.after_llm().push_frame = drop
    return pair


@pytest.fixture
def inputs(worker, rtvi, study, capture):
    """Inputs with a local STT and a cloud LLM."""
    return Inputs(worker.queue_frames, rtvi, study, capture, stt_location="local", llm_location="cloud")


async def _push(inputs: Inputs, frame, destination=None):
    """Show the inputs one frame moving through the pipeline."""
    await inputs.on_push_frame(
        FramePushed(
            source=_Processor(),
            destination=destination or _Processor(),
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )


async def _call(inputs: Inputs, worker: _Worker, capture: ContextCapture, before_observer=None, upstream=False):
    """Run the LLM on the worker's context: the frame passes the capture side it would meet, then, after an optional change to the live context, reaches the observer."""
    context = LLMContext(messages=worker.messages)
    frame = LLMContextFrame(context=context)
    if upstream:
        await capture.after_llm().process_frame(frame, FrameDirection.UPSTREAM)
    else:
        await capture.before_llm().process_frame(frame, FrameDirection.DOWNSTREAM)
    if before_observer is not None:
        before_observer(context)
    await _push(inputs, frame, _LLM)


def _logged(study: StudyLog, path) -> list[dict]:
    """The input records the study log at this path holds, after closing it."""
    study.close()
    with open(path, encoding="utf-8") as file:
        lines = [json.loads(line) for line in file]
    return [{k: v for k, v in line.items() if k not in ("event", "ts", "iso", "turn")} for line in lines if line["event"] == "input_record"]


async def test_a_newer_state_replaces_the_previous_one_in_the_context(inputs, worker):
    """Only the newest held state is in the context, after the other messages."""
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", HANDS)
    held = [m for m in worker.messages if m["content"].startswith(HEADERS["held"])]
    assert len(held) == 1
    assert json.loads(held[0]["content"].removeprefix(HEADERS["held"] + "\n")) == HANDS
    assert worker.messages[-1] is held[0]


async def test_a_draft_is_sent_when_it_changes_and_only_then(inputs, rtvi):
    """Each change to what the next call is meant to get sends the draft once."""
    await inputs.receive("held", HANDS)
    await inputs.receive("held", HANDS)
    await _push(inputs, UserStartedSpeakingFrame())
    drafts = rtvi.records("draft")
    assert [[e["input"] for e in d["inputs"]] for d in drafts] == [["held"], ["transcript", "held"]]
    assert {d["id"] for d in drafts} == {1}


async def test_a_call_is_recorded_from_the_context_it_reads(inputs, worker, rtvi, study, tmp_path, capture):
    """The sent record names the trigger, each input with its content and where it was processed, and the log holds the same record."""
    await inputs.receive("held", HANDS)
    await _push(inputs, UserStartedSpeakingFrame())
    worker.messages.append({"role": "user", "content": "Is this the right one?"})
    await _call(inputs, worker, capture)
    sent = rtvi.records("sent")
    assert sent == [
        {
            "id": 1,
            "state": "sent",
            "trigger": "user_turn",
            "llm": "cloud",
            "inputs": [
                {"input": "transcript", "data": "Is this the right one?", "where": "local"},
                {"input": "held", "data": HANDS, "where": "client"},
            ],
            "dropped": [],
        }
    ]
    assert _logged(study, tmp_path / "P01.jsonl") == sent


async def test_an_input_that_misses_the_call_is_listed_as_dropped(inputs, worker, rtvi, capture):
    """A pointing state the context has not taken yet shows as dropped, beside the older state the call got."""
    await inputs.receive("pointing", OTHER_HANDS)
    worker.messages.append({"role": "user", "content": "This one?"})
    snapshot = list(worker.messages)
    await inputs.receive("pointing", HANDS)
    worker.messages = snapshot
    await _call(inputs, worker, capture)
    (sent,) = rtvi.records("sent")
    assert {"input": "pointing", "data": OTHER_HANDS, "where": "client"} in sent["inputs"]
    assert sent["dropped"] == [{"input": "pointing", "data": HANDS, "where": "client"}]


async def test_a_transcript_counts_once_and_a_game_snapshot_triggers(inputs, worker, rtvi, capture):
    """A later call on the same user message does not count it again, and a snapshot after it makes the game the trigger."""
    worker.messages.append({"role": "user", "content": "Hello."})
    await _call(inputs, worker, capture)
    worker.messages.append({"role": "assistant", "content": "Hi."})
    snapshot = {"event": "poured", "detail": None}
    worker.messages.append({"role": "developer", "content": f"{GAME_HEADER}\n{json.dumps(snapshot)}"})
    await inputs.note("game", snapshot)
    await _call(inputs, worker, capture)
    first, second = rtvi.records("sent")
    assert [e["input"] for e in first["inputs"]] == ["transcript"]
    assert second["trigger"] == "game"
    assert second["inputs"] == [{"input": "game", "data": snapshot, "where": "client"}]
    assert second["id"] == 2


async def test_a_tool_result_triggers_the_call(inputs, worker, rtvi, capture):
    """A call that answers a tool result names it as the trigger."""
    worker.messages.append({"role": "tool", "content": "{\"started\": true}", "tool_call_id": "1"})
    await _call(inputs, worker, capture)
    assert rtvi.records("sent")[0]["trigger"] == "tool_result"


async def test_an_emotion_reading_is_its_own_input(inputs, worker, rtvi, capture):
    """The reading expressive conversation puts at the head of a user message is recorded apart from the words."""
    worker.messages.append({"role": "user", "content": "([emotion angry=0.81, neutral=0.12]) Where is it?"})
    await _call(inputs, worker, capture)
    (sent,) = rtvi.records("sent")
    assert sent["inputs"] == [
        {"input": "transcript", "data": "Where is it?", "where": "local"},
        {"input": "emotion", "data": "([emotion angry=0.81, neutral=0.12])", "where": "local"},
    ]


async def test_a_reply_heard_to_its_end_is_answered(inputs, worker, rtvi, study, tmp_path, capture):
    """The record closes as answered once the reply has ended and the agent has stopped speaking it."""
    await _call(inputs, worker, capture)
    await _push(inputs, LLMTextFrame(text="Yes."))
    await _push(inputs, BotStartedSpeakingFrame())
    await _push(inputs, LLMFullResponseEndFrame())
    assert rtvi.records("answered") == []
    await _push(inputs, BotStoppedSpeakingFrame())
    assert rtvi.records("answered") == [{"id": 1, "state": "answered"}]
    assert _logged(study, tmp_path / "P01.jsonl")[-1] == {"id": 1, "state": "answered"}


async def test_a_reply_without_words_is_answered_when_it_ends(inputs, worker, rtvi, capture):
    """A call that only calls a tool has nothing to hear, so it closes when its reply ends."""
    await _call(inputs, worker, capture)
    await _push(inputs, LLMFullResponseEndFrame())
    assert rtvi.records("answered") == [{"id": 1, "state": "answered"}]


async def test_a_reply_cut_off_is_interrupted(inputs, worker, rtvi, capture):
    """An interruption before the reply is heard closes the record as interrupted, once."""
    await _call(inputs, worker, capture)
    await _push(inputs, LLMTextFrame(text="Yes, that"))
    await _push(inputs, BotStartedSpeakingFrame())
    await _push(inputs, InterruptionFrame())
    await _push(inputs, BotStoppedSpeakingFrame())
    assert [r["state"] for r in rtvi.records() if r["id"] == 1] == ["sent", "interrupted"]


async def test_the_next_draft_is_sent_right_after_a_call(inputs, worker, rtvi, capture):
    """Inputs that stay in the context are shown as the next call's draft without waiting for a change."""
    await inputs.receive("held", HANDS)
    await _call(inputs, worker, capture)
    assert rtvi.records("draft")[-1] == {"id": 2, "state": "draft", "inputs": [{"input": "held", "data": HANDS, "where": "client"}], "dropped": []}


async def test_a_snapshot_swapped_in_after_the_user_message_leaves_the_user_turn_the_trigger(inputs, worker, rtvi, capture):
    """A game snapshot moving to the end of the context does not make a user turn's call a game call."""
    worker.messages.append({"role": "user", "content": "Done."})
    worker.messages.append({"role": "developer", "content": f"{GAME_HEADER}\n{json.dumps({'event': None})}"})
    await _call(inputs, worker, capture)
    assert rtvi.records("sent")[0]["trigger"] == "user_turn"


async def test_a_call_while_the_participant_speaks_keeps_the_transcript_in_the_draft(inputs, worker, rtvi, capture):
    """A game call during the participant's speech does not take the coming transcript out of the next draft."""
    await _push(inputs, UserStartedSpeakingFrame())
    worker.messages.append({"role": "developer", "content": f"{GAME_HEADER}\n{json.dumps({'event': 'poured'})}"})
    await _call(inputs, worker, capture)
    assert [e["input"] for e in rtvi.records("draft")[-1]["inputs"]] == ["transcript"]


async def test_a_spoken_reply_followed_by_a_tool_call_is_answered_not_interrupted(inputs, worker, rtvi, capture):
    """A reply heard to its end stays answered when the call for its tool result starts before the agent stops speaking."""
    await _call(inputs, worker, capture)
    await _push(inputs, LLMTextFrame(text="Let me start."))
    await _push(inputs, BotStartedSpeakingFrame())
    await _push(inputs, LLMFullResponseEndFrame())
    worker.messages.append({"role": "tool", "content": "{}", "tool_call_id": "1"})
    await _call(inputs, worker, capture)
    await _push(inputs, LLMFullResponseEndFrame())
    await _push(inputs, BotStoppedSpeakingFrame())
    outcomes = [(r["id"], r["state"]) for r in rtvi.records() if r["state"] in ("answered", "interrupted")]
    assert sorted(outcomes) == [(1, "answered"), (2, "answered")]


async def test_the_record_holds_the_context_as_the_call_passed_not_as_it_is_later(inputs, worker, rtvi, capture):
    """A held state swapped into the live context after the call passed the capture is not in the call's record."""
    worker.messages.append({"role": "user", "content": "This one?"})

    def late_swap(context):
        """Append a held state as the user aggregator would just after the call."""
        context.get_messages().append({"role": "developer", "content": f"{HEADERS['held']}\n{json.dumps(HANDS)}"})

    await _call(inputs, worker, capture, before_observer=late_swap)
    assert [e["input"] for e in rtvi.records("sent")[0]["inputs"]] == ["transcript"]


async def test_a_call_arriving_upstream_is_captured_after_the_llm(inputs, worker, rtvi, capture):
    """A call on a tool result comes back from the assistant aggregator, and the side after the LLM captures it as it passed."""
    worker.messages.append({"role": "tool", "content": "{}", "tool_call_id": "1"})

    def late_swap(context):
        """Append a held state just after the call."""
        context.get_messages().append({"role": "developer", "content": f"{HEADERS['held']}\n{json.dumps(HANDS)}"})

    await _call(inputs, worker, capture, before_observer=late_swap, upstream=True)
    (sent,) = rtvi.records("sent")
    assert sent["trigger"] == "tool_result"
    assert sent["inputs"] == []


def test_the_capture_keeps_only_the_newest_copies():
    """Copies of calls that never reached the observer do not pile up."""
    pair = ContextCapture()
    for frame_id in range(ContextCapture.KEPT + 3):
        pair.keep(frame_id, [])
    assert pair.take(0) is None
    assert pair.take(ContextCapture.KEPT + 2) == []


class _PipelineLLM(LLMService):
    """Stand-in for the LLM inside a real pipeline: takes context frames and passes every other frame on."""

    async def process_frame(self, frame, direction):
        """Consume a context frame as the LLM would, without running a model."""
        await FrameProcessor.process_frame(self, frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)


async def test_a_call_through_a_real_pipeline_is_recorded_from_the_capture(worker, rtvi, study):
    """In a running pipeline, where observers see a frame first on its way into the capture, the call is still recorded, from the capture's copy."""
    pair = ContextCapture()
    inputs = Inputs(worker.queue_frames, rtvi, study, pair, stt_location="local", llm_location="cloud")
    worker.messages.append({"role": "user", "content": "Is this it?"})
    frame = LLMContextFrame(context=LLMContext(messages=list(worker.messages)))
    await run_test(Pipeline([pair.before_llm(), _PipelineLLM()]), frames_to_send=[frame], observers=[inputs])
    (sent,) = rtvi.records("sent")
    assert sent["inputs"] == [{"input": "transcript", "data": "Is this it?", "where": "local"}]
    assert pair.take(frame.id) is None
