"""Client inputs in the LLM context and the per-call input record: the draft before a call, what the call got and missed, and whether its reply was heard."""

import asyncio
import json

import pytest
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMMessagesTransformFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import run_test

from agent.game import TAG as GAME_TAG
from agent.game import wrap
from agent.inputs import INPUTS, ContextCapture, Inputs
from agent.nonverbal import REFER
from agent.observability.study_log import StudyLog

FLASK_1 = {"id": "flask-1", "kind": "flask"}
FLASK_3 = {"id": "flask-3", "kind": "flask"}
EMPTY = {"left": None, "right": None}
HANDS = {"left": None, "right": FLASK_3}
OTHER_HANDS = {"left": FLASK_1, "right": None}
# What a held or pointing input reads after OTHER_HANDS, then HANDS, since the participant last started speaking.
BOTH_IN_TURN = {"left": [FLASK_1, None], "right": [None, FLASK_3]}


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


async def _push(inputs: Inputs, frame, destination=None, source=None):
    """Show the inputs one frame moving through the pipeline."""
    await inputs.on_push_frame(
        FramePushed(
            source=source or _Processor(),
            destination=destination or _Processor(),
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )


async def _said(inputs: Inputs, text: str, text_type: str = "sentence"):
    """Show the inputs a unit of the reply as the reply strip passes it on: words, or an object marker with text_type refer."""
    await _push(inputs, AggregatedTextFrame(text=text, text_type=text_type), source=LLMTextProcessor())


async def _ended(inputs: Inputs):
    """Show the inputs the reply's end as the reply strip passes it on, after any words it held back."""
    await _push(inputs, LLMFullResponseEndFrame(), source=LLMTextProcessor())


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
    return [{k: v for k, v in line.items() if k not in ("event", "ts", "iso")} for line in lines if line["event"] == "input_record"]


def _held(worker: _Worker) -> list[str]:
    """The held messages in the worker's context."""
    return [m["content"] for m in worker.messages if m["content"].startswith("<held>")]


async def test_a_newer_state_replaces_the_previous_message_in_the_context(inputs, worker):
    """Only one held message is in the context, after the other messages, reading every state each hand had."""
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", HANDS)
    assert _held(worker) == [wrap("held", BOTH_IN_TURN)]
    assert worker.messages[-1]["content"] == _held(worker)[0]


async def test_empty_hands_read_nothing(inputs, worker):
    """A held state with both hands empty, and nothing held before it, reads nothing as plain text."""
    await inputs.receive("held", EMPTY)
    assert _held(worker) == ["<held>\nnothing\n</held>"]


async def test_an_empty_hand_is_left_out(inputs, worker):
    """A hand that has held nothing does not appear, and a hand that held one object reads that object alone."""
    await inputs.receive("held", HANDS)
    assert _held(worker) == [wrap("held", {"right": FLASK_3})]


async def test_an_object_dropped_before_the_call_still_counts(inputs, worker):
    """Letting go of an object after grabbing it lists the object, then null for the empty hand."""
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", EMPTY)
    assert _held(worker) == [wrap("held", {"left": [FLASK_1, None]})]


async def test_every_state_of_a_hand_reaches_the_context(inputs, worker):
    """Grabbing, letting go and grabbing another object lists all three states, each swapped in as it arrives."""
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", EMPTY)
    await inputs.receive("held", {"left": FLASK_3, "right": None})
    assert _held(worker) == [wrap("held", {"left": [FLASK_1, None, FLASK_3]})]


async def test_a_start_over_after_several_states_reaches_the_context(inputs, worker):
    """Speech start after a hand changed swaps in its current state alone."""
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", {"left": FLASK_3, "right": None})
    await _push(inputs, UserStartedSpeakingFrame())
    assert _held(worker) == [wrap("held", {"left": FLASK_3})]


async def test_two_objects_without_an_id_stay_two_entries(inputs, worker):
    """Objects Unity cannot refer to carry no id, so one following the other is listed as a change."""
    cup = {"id": None, "kind": "unknown", "name": "Cup"}
    lid = {"id": None, "kind": "unknown", "name": "Lid"}
    await inputs.receive("held", {"left": cup, "right": None})
    await inputs.receive("held", {"left": lid, "right": None})
    assert _held(worker) == [wrap("held", {"left": [cup, lid]})]


async def test_the_same_object_described_anew_replaces_its_entry(inputs, worker):
    """A held object whose description changes stays one entry, with its newest description."""
    filled = {**FLASK_1, "substance": "Flame"}
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", {"left": filled, "right": None})
    assert _held(worker) == [wrap("held", {"left": filled})]


async def test_the_history_starts_over_when_the_participant_starts_speaking(inputs, worker):
    """Speech start keeps each hand's current state only, so an object dropped before it reads nothing."""
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", EMPTY)
    await _push(inputs, UserStartedSpeakingFrame())
    assert _held(worker) == ["<held>\nnothing\n</held>"]


async def test_speech_resumed_before_the_call_keeps_the_history(inputs, worker, capture):
    """A second speech start before the turn's call is the same turn, so a drop between them still counts; the next turn starts over."""
    await _push(inputs, UserStartedSpeakingFrame())
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", EMPTY)
    await _push(inputs, UserStartedSpeakingFrame())
    assert _held(worker) == [wrap("held", {"left": [FLASK_1, None]})]
    worker.messages.append({"role": "user", "content": "This one?"})
    await _call(inputs, worker, capture)
    await _push(inputs, UserStartedSpeakingFrame())
    assert _held(worker) == ["<held>\nnothing\n</held>"]


async def test_speech_that_never_reaches_the_llm_keeps_the_history_into_the_next_turn(inputs, worker):
    """A false start with no call on a user message leaves the turn open, so the next speech does not start the hands over."""
    await _push(inputs, UserStartedSpeakingFrame())
    await inputs.receive("held", OTHER_HANDS)
    await inputs.receive("held", EMPTY)
    await _push(inputs, UserStartedSpeakingFrame())
    assert _held(worker) == [wrap("held", {"left": [FLASK_1, None]})]


async def test_the_history_does_not_start_over_at_a_call(inputs, worker, rtvi, capture):
    """A call between two states leaves the history running, so the next call reads both."""
    await inputs.receive("held", OTHER_HANDS)
    await _call(inputs, worker, capture)
    await inputs.receive("held", HANDS)
    await _call(inputs, worker, capture)
    assert [e["data"] for e in rtvi.records("sent")[1]["inputs"]] == [BOTH_IN_TURN]


async def test_a_call_records_nothing_as_its_data(inputs, worker, rtvi, capture):
    """An input reading nothing is recorded with nothing as its data."""
    await inputs.receive("pointing", EMPTY)
    await _call(inputs, worker, capture)
    assert rtvi.records("sent")[0]["inputs"] == [{"input": "pointing", "data": "nothing", "where": "client"}]


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
                {"input": "held", "data": {"right": FLASK_3}, "where": "client"},
            ],
            "dropped": [],
            "turn": 0,
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
    assert {"input": "pointing", "data": {"left": FLASK_1}, "where": "client"} in sent["inputs"]
    assert sent["dropped"] == [{"input": "pointing", "data": BOTH_IN_TURN, "where": "client"}]


async def test_a_transcript_counts_once_and_a_game_snapshot_triggers(inputs, worker, rtvi, capture):
    """A later call on the same user message does not count it again, and a snapshot after it makes the game the trigger."""
    worker.messages.append({"role": "user", "content": "Hello."})
    await _call(inputs, worker, capture)
    worker.messages.append({"role": "assistant", "content": "Hi."})
    snapshot = {"event": "poured", "detail": None}
    worker.messages.append({"role": "developer", "content": wrap(GAME_TAG, snapshot)})
    await inputs.note("game", snapshot)
    await _call(inputs, worker, capture)
    first, second = rtvi.records("sent")
    assert [e["input"] for e in first["inputs"]] == ["transcript"]
    assert second["trigger"] == "game"
    assert second["inputs"] == [{"input": "game", "data": snapshot, "where": "client"}]
    assert second["id"] == 2


async def test_a_record_carries_the_study_log_turn_except_a_draft(inputs, worker, rtvi, study, tmp_path, capture):
    """Once a turn has started, the sent record and its outcome carry its number, as the study log stamps them, and a draft carries none."""
    await study.observers[1].on_push_frame(FramePushed(source=_Processor(), destination=_Processor(), frame=UserStartedSpeakingFrame(), direction=FrameDirection.DOWNSTREAM, timestamp=0))
    # The turn tracker runs its turn-started handler as a task; this lets it run.
    await asyncio.sleep(0)
    worker.messages.append({"role": "user", "content": "Hello."})
    await _call(inputs, worker, capture)
    await _ended(inputs)
    assert [r["turn"] for r in rtvi.records() if r["state"] != "draft"] == [1, 1]
    assert all("turn" not in r for r in rtvi.records("draft"))
    assert [r["turn"] for r in _logged(study, tmp_path / "P01.jsonl")] == [1, 1]


async def test_a_tool_result_triggers_the_call(inputs, worker, rtvi, capture):
    """A call that answers a tool result names it as the trigger."""
    worker.messages.append({"role": "tool", "content": "{\"started\": true}", "tool_call_id": "1"})
    await _call(inputs, worker, capture)
    assert rtvi.records("sent")[0]["trigger"] == "tool_result"


async def test_an_emotion_reading_is_its_own_input(inputs, worker, rtvi, capture):
    """The reading the voice reader puts at the head of a user message is recorded apart from the words."""
    reading = "([tone frustrated=0.71, neutral=0.22],[arousal value=0.34],[laughter value=0.02],[speech_rate value=3.1],[sarcasm_voice value=0.63])"
    worker.messages.append({"role": "user", "content": f"{reading} Where is it?"})
    await _call(inputs, worker, capture)
    (sent,) = rtvi.records("sent")
    assert sent["inputs"] == [
        {"input": "transcript", "data": "Where is it?", "where": "local"},
        {"input": "emotion", "data": reading, "where": "local"},
    ]


async def test_a_reply_heard_to_its_end_is_answered(inputs, worker, rtvi, study, tmp_path, capture):
    """The record closes as answered once the reply has ended and the agent has stopped speaking it."""
    await _call(inputs, worker, capture)
    await _said(inputs, "Yes.")
    await _push(inputs, BotStartedSpeakingFrame())
    await _ended(inputs)
    assert rtvi.records("answered") == []
    await _push(inputs, BotStoppedSpeakingFrame())
    assert rtvi.records("answered") == [{"id": 1, "state": "answered", "turn": 0}]
    assert _logged(study, tmp_path / "P01.jsonl")[-1] == {"id": 1, "state": "answered", "turn": 0}


async def test_a_reply_without_words_is_answered_when_it_ends(inputs, worker, rtvi, capture):
    """A call that only calls a tool has nothing to hear, so it closes when its reply ends."""
    await _call(inputs, worker, capture)
    await _ended(inputs)
    assert rtvi.records("answered") == [{"id": 1, "state": "answered", "turn": 0}]


async def test_a_reply_cut_off_is_interrupted(inputs, worker, rtvi, capture):
    """An interruption before the reply is heard closes the record as interrupted, once."""
    await _call(inputs, worker, capture)
    await _said(inputs, "Yes, that")
    await _push(inputs, BotStartedSpeakingFrame())
    await _push(inputs, InterruptionFrame())
    await _push(inputs, BotStoppedSpeakingFrame())
    assert [r["state"] for r in rtvi.records() if r["id"] == 1] == ["sent", "interrupted"]


async def test_a_question_whose_reply_was_cut_off_unheard_counts_again(inputs, worker, rtvi, capture):
    """When an interruption stops a reply before any of it reaches the context, the next call answers both questions and its record lists both."""
    worker.messages.append({"role": "user", "content": "Which flask do you mean?"})
    await _call(inputs, worker, capture)
    await _push(inputs, InterruptionFrame())
    worker.messages.append({"role": "user", "content": "This one on the left."})
    await _call(inputs, worker, capture)
    second = rtvi.records("sent")[1]
    assert second["trigger"] == "user_turn"
    assert [e["data"] for e in second["inputs"] if e["input"] == "transcript"] == ["Which flask do you mean?", "This one on the left."]


async def test_a_question_whose_reply_was_partly_heard_does_not_count_again(inputs, worker, rtvi, capture):
    """A reply cut off after some of it was spoken stays in the context, so the question before it is not listed in the next call."""
    worker.messages.append({"role": "user", "content": "Which flask do you mean?"})
    await _call(inputs, worker, capture)
    worker.messages.append({"role": "assistant", "content": "The one"})
    worker.messages.append({"role": "user", "content": "This one on the left."})
    await _call(inputs, worker, capture)
    second = rtvi.records("sent")[1]
    assert [e["data"] for e in second["inputs"] if e["input"] == "transcript"] == ["This one on the left."]


async def test_the_next_draft_is_sent_right_after_a_call(inputs, worker, rtvi, capture):
    """Inputs that stay in the context are shown as the next call's draft without waiting for a change."""
    await inputs.receive("held", HANDS)
    await _call(inputs, worker, capture)
    assert rtvi.records("draft")[-1] == {"id": 2, "state": "draft", "inputs": [{"input": "held", "data": {"right": FLASK_3}, "where": "client"}], "dropped": []}


async def test_a_snapshot_swapped_in_after_the_user_message_leaves_the_user_turn_the_trigger(inputs, worker, rtvi, capture):
    """A game snapshot moving to the end of the context does not make a user turn's call a game call."""
    worker.messages.append({"role": "user", "content": "Done."})
    worker.messages.append({"role": "developer", "content": wrap(GAME_TAG, {"event": None})})
    await _call(inputs, worker, capture)
    assert rtvi.records("sent")[0]["trigger"] == "user_turn"


async def test_a_call_while_the_participant_speaks_keeps_the_transcript_in_the_draft(inputs, worker, rtvi, capture):
    """A game call during the participant's speech does not take the coming transcript out of the next draft."""
    await _push(inputs, UserStartedSpeakingFrame())
    worker.messages.append({"role": "developer", "content": wrap(GAME_TAG, {"event": "poured"})})
    await _call(inputs, worker, capture)
    assert [e["input"] for e in rtvi.records("draft")[-1]["inputs"]] == ["transcript"]


async def test_a_spoken_reply_followed_by_a_tool_call_is_answered_not_interrupted(inputs, worker, rtvi, capture):
    """A reply heard to its end stays answered when the call for its tool result starts before the agent stops speaking."""
    await _call(inputs, worker, capture)
    await _said(inputs, "Let me start.")
    await _push(inputs, BotStartedSpeakingFrame())
    await _ended(inputs)
    worker.messages.append({"role": "tool", "content": "{}", "tool_call_id": "1"})
    await _call(inputs, worker, capture)
    await _ended(inputs)
    await _push(inputs, BotStoppedSpeakingFrame())
    outcomes = [(r["id"], r["state"]) for r in rtvi.records() if r["state"] in ("answered", "interrupted")]
    assert sorted(outcomes) == [(1, "answered"), (2, "answered")]


async def test_the_record_holds_the_context_as_the_call_passed_not_as_it_is_later(inputs, worker, rtvi, capture):
    """A held state swapped into the live context after the call passed the capture is not in the call's record."""
    worker.messages.append({"role": "user", "content": "This one?"})

    def late_swap(context):
        """Append a held state as the user aggregator would just after the call."""
        context.get_messages().append({"role": "developer", "content": wrap("held", HANDS)})

    await _call(inputs, worker, capture, before_observer=late_swap)
    assert [e["input"] for e in rtvi.records("sent")[0]["inputs"]] == ["transcript"]


async def test_a_call_arriving_upstream_is_captured_after_the_llm(inputs, worker, rtvi, capture):
    """A call on a tool result comes back from the assistant aggregator, and the side after the LLM captures it as it passed."""
    worker.messages.append({"role": "tool", "content": "{}", "tool_call_id": "1"})

    def late_swap(context):
        """Append a held state just after the call."""
        context.get_messages().append({"role": "developer", "content": wrap("held", HANDS)})

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


async def test_a_reply_that_is_only_a_marker_is_answered_when_it_ends(inputs, worker, rtvi, capture):
    """A reply with nothing left to speak but an object marker has nothing to hear, so it closes when it ends."""
    await _call(inputs, worker, capture)
    await _said(inputs, "flask-3", text_type=REFER)
    await _ended(inputs)
    assert rtvi.records("answered") == [{"id": 1, "state": "answered", "turn": 0}]


async def test_the_llm_ending_before_the_held_back_words_does_not_close_the_record(inputs, worker, rtvi, capture):
    """The reply strip holds the last sentence until the reply ends, so the record waits for the strip's end and then for the words to be heard."""
    await _call(inputs, worker, capture)
    await _push(inputs, LLMFullResponseEndFrame())
    await _said(inputs, "Yes.")
    await _ended(inputs)
    assert rtvi.records("answered") == []
    await _push(inputs, BotStartedSpeakingFrame())
    await _push(inputs, BotStoppedSpeakingFrame())
    assert rtvi.records("answered") == [{"id": 1, "state": "answered", "turn": 0}]
