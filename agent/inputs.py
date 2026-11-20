"""Participant inputs in the LLM context, and the per-call record of which inputs reached the LLM, sent to the client and written to the study log."""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMMessagesTransformFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.services.llm_service import LLMService

from agent.game import HEADER as GAME_HEADER
from agent.observability.study_log import StudyLog

# The RTVI server-message type the agent sends each input record under, as {"t": "inputs", "d": {"id": ..., "state": ..., ...}}.
INPUTS = "inputs"

# Says where each client input in the context comes from, since Mistral receives it as a user message.
HEADERS = {
    "held": "What the participant holds in each hand, from the client. The participant did not say this.",
    "pointing": "What the participant is pointing at with each hand, from the client. The participant did not say this.",
}

# Where an input the client works out was processed.
CLIENT = "client"

# Where the emotion reading of expressive conversation is processed: Pipecat Tone runs on this machine's CPU.
EMOTION_LOCATION = "local"

# The head of a user message that carries an expressive-conversation reading, such as "([emotion angry=0.81]) Where is it?".
READING_START = "(["
READING_END = ") "


def selected(kind: str) -> bool:
    """Whether an input of this kind goes into the LLM context; every input does for now."""
    return True


def _text(message: Any) -> str | None:
    """The text of a context message, or None for one that holds no text."""
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return None


def _client_input(text: str) -> tuple[str, Any] | None:
    """The kind and data of a client input or game snapshot the context holds, or None for any other message."""
    for kind, header in [*HEADERS.items(), ("game", GAME_HEADER)]:
        if text.startswith(header + "\n"):
            return kind, json.loads(text.removeprefix(header + "\n"))
    return None


def _split_reading(text: str) -> tuple[str | None, str]:
    """Split a user message into its expressive-conversation reading, if it starts with one, and the words said."""
    if text.startswith(READING_START) and READING_END in text:
        reading, words = text.split(READING_END, 1)
        return reading + ")", words
    return None, text


class _Tap(FrameProcessor):
    """One side of the context capture: copies the message list of each context frame passing it in one direction, then passes every frame on unchanged."""

    def __init__(self, capture: "ContextCapture", direction: FrameDirection, **kwargs):
        """Take the capture the copies go to and the direction this side copies."""
        super().__init__(**kwargs)
        self._capture = capture
        self._direction = direction

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Copy a context frame's messages on its way to the LLM, then pass the frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == self._direction:
            self._capture.keep(frame.id, list(frame.context.get_messages()))
        await self.push_frame(frame, direction)


class ContextCapture:
    """Copies the message list each LLM call reads as the call passes, since observers see the shared context only after it may have changed; one side sits before the LLM, the other after it for calls that arrive upstream, such as one on a tool result."""

    # At most this many copies wait for the observer; older ones belong to calls the LLM dropped.
    KEPT = 8

    def __init__(self):
        """Start with no copies."""
        self._copies: dict[int, list] = {}
        self._before = _Tap(self, FrameDirection.DOWNSTREAM)
        self._after = _Tap(self, FrameDirection.UPSTREAM)

    def before_llm(self) -> FrameProcessor:
        """The side that goes right before the LLM."""
        return self._before

    def after_llm(self) -> FrameProcessor:
        """The side that goes right after the LLM."""
        return self._after

    def keep(self, frame_id: int, messages: list):
        """Keep one call's message list, dropping the oldest once more than KEPT are waiting."""
        self._copies[frame_id] = messages
        while len(self._copies) > self.KEPT:
            del self._copies[next(iter(self._copies))]

    def take(self, frame_id: int) -> list | None:
        """Hand over and forget the message list of the call with this context frame id, or None if no such call passed."""
        return self._copies.pop(frame_id, None)


class Inputs(BaseObserver):
    """Puts the client's inputs into the LLM context, and keeps the record of each LLM call: the inputs it was meant to get, the inputs it got, and whether its reply was heard."""

    def __init__(
        self,
        queue: Callable[[list[Frame]], Awaitable[None]],
        rtvi: RTVIProcessor,
        study: StudyLog,
        capture: ContextCapture,
        stt_location: str,
        llm_location: str,
        **kwargs,
    ):
        """Take the worker's way to queue frames, its RTVI processor to send records through, the study log, the capture before the LLM, and where the STT and LLM run."""
        super().__init__(observe_every_push=True, **kwargs)
        self._queue = queue
        self._rtvi = rtvi
        self._study = study
        self._capture = capture
        self._stt_location = stt_location
        self._llm_location = llm_location
        self._messages: dict[str, dict] = {}
        self._known: dict[str, Any] = {}
        self._speaking = False
        # Every context message an earlier call has read, by id, with the message itself.
        self._seen: dict[int, Any] = {}
        self._id = 1
        # The draft last sent for the current id, so an unchanged draft is not sent again.
        self._draft: list[dict] = []
        self._draft_sent = False
        # The records whose reply is not yet heard or cut off, oldest first: id, whether the reply has text, whether speech began after it did, and whether it has ended.
        self._open: list[dict] = []

    async def receive(self, kind: str, data: Any):
        """Put the client's newest held or pointing state into the context in place of the previous one, and update the draft."""
        if selected(kind):
            await self._queue([LLMMessagesTransformFrame(transform=self._swap_in(kind, data), run_llm=False)])
        await self.note(kind, data)

    async def note(self, kind: str, data: Any):
        """Note an input the next LLM call is meant to get, such as a game snapshot another component puts into the context."""
        self._known[kind] = data
        await self._update_draft()

    def _swap_in(self, kind: str, data: Any) -> Callable[[list], list]:
        """Build the context transform that drops this kind's previous message and appends the new one."""
        previous = self._messages.get(kind)
        message = {"role": "developer", "content": f"{HEADERS[kind]}\n{json.dumps(data, ensure_ascii=False)}"}
        self._messages[kind] = message

        def swap(messages: list) -> list:
            """Runs inside the user aggregator, on the context the next LLM call reads."""
            return [m for m in messages if m is not previous] + [message]

        return swap

    def _where(self, kind: str) -> str:
        """Where an input of this kind was processed."""
        if kind == "transcript":
            return self._stt_location
        if kind == "emotion":
            return EMOTION_LOCATION
        return CLIENT

    def _entry(self, kind: str, data: Any) -> dict:
        """One input of a record: its kind, its content and where it was processed."""
        return {"input": kind, "data": data, "where": self._where(kind)}

    def _intended(self) -> list[dict]:
        """The inputs the next LLM call is meant to get: every selected input known so far, and a transcript while the participant speaks."""
        draft = [self._entry(kind, data) for kind, data in self._known.items() if selected(kind)]
        if self._speaking:
            draft.insert(0, self._entry("transcript", None))
        return draft

    async def _update_draft(self):
        """Send the draft when it has changed since it was last sent."""
        draft = self._intended()
        if draft == self._draft and self._draft_sent:
            return
        self._draft = draft
        self._draft_sent = True
        await self._send({"id": self._id, "state": "draft", "inputs": draft, "dropped": []}, log=False)

    async def _send(self, record: dict, log: bool):
        """Send a record to the client and, when asked, write it to the study log."""
        await self._rtvi.send_server_message({"t": INPUTS, "d": record})
        if log:
            self._study.write("input_record", **record)

    def _read(self, messages: list) -> tuple[str, list[dict]]:
        """Work out from the context an LLM call reads what started the call and which inputs it got: a new user message, else a tool result last, else a new game snapshot."""
        inputs: list[dict] = []
        new_user = new_game = False
        last_role = None
        for message in messages:
            text = _text(message)
            new = id(message) not in self._seen
            if text is not None and (found := _client_input(text)) is not None:
                inputs.append(self._entry(*found))
                new_game = new_game or (found[0] == "game" and new)
                continue
            role = message.get("role") if isinstance(message, dict) else None
            last_role = role
            if role == "user" and text is not None and new:
                new_user = True
                reading, words = _split_reading(text)
                if reading is not None:
                    inputs.append(self._entry("emotion", reading))
                inputs.append(self._entry("transcript", words))
        self._seen.update((id(message), message) for message in messages)
        if new_user:
            trigger = "user_turn"
        elif last_role == "tool":
            trigger = "tool_result"
        elif new_game:
            trigger = "game"
        else:
            trigger = "other"
        order = {"transcript": 0, "emotion": 1}
        inputs.sort(key=lambda entry: order.get(entry["input"], 2))
        return trigger, inputs

    @staticmethod
    def _dropped(entry: dict, inputs: list[dict]) -> bool:
        """Whether an intended input missed the call: its kind is absent, or, for a client input swapped in here, the call got other data than intended."""
        if entry["input"] in HEADERS:
            return entry not in inputs
        return not any(sent["input"] == entry["input"] for sent in inputs)

    async def _sent(self, frame: LLMContextFrame):
        """Turn the draft into the record of this call, built from the context it reads, and send the next draft."""
        messages = self._capture.take(frame.id)
        if messages is None:
            logger.warning("No captured context for an LLM call, so its input record reads the live context")
            messages = frame.context.get_messages()
        trigger, inputs = self._read(messages)
        dropped = [entry for entry in self._intended() if self._dropped(entry, inputs)]
        record = {"id": self._id, "state": "sent", "trigger": trigger, "llm": self._llm_location, "inputs": inputs, "dropped": dropped}
        await self._send(record, log=True)
        self._open.append({"id": self._id, "has_text": False, "heard": False, "ended": False})
        self._id += 1
        self._draft = []
        self._draft_sent = False
        if any(entry["input"] == "transcript" for entry in inputs):
            self._speaking = False
        await self._update_draft()

    async def _close(self, records: list[dict], outcome: str):
        """Close these open records as answered or interrupted."""
        for record in records:
            self._open.remove(record)
            await self._send({"id": record["id"], "state": outcome}, log=True)

    async def on_push_frame(self, data: FramePushed):
        """Follow the participant's speech for the draft, each LLM call for the record, and each reply until it is heard or cut off."""
        frame = data.frame
        # A call counts on its push into the LLM, which follows the capture's copy; every other frame counts on its first push only.
        if isinstance(frame, LLMContextFrame):
            if isinstance(data.destination, LLMService):
                await self._sent(frame)
            return
        if not data.first_push:
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._speaking = True
            await self._update_draft()
        elif not self._open:
            return
        elif isinstance(frame, InterruptionFrame):
            await self._close(list(self._open), "interrupted")
        elif isinstance(frame, LLMTextFrame) and frame.text.strip():
            self._open[-1]["has_text"] = True
        elif isinstance(frame, BotStartedSpeakingFrame):
            for record in self._open:
                record["heard"] = record["has_text"]
        elif isinstance(frame, LLMFullResponseEndFrame):
            current = self._open[-1]
            current["ended"] = True
            if not current["has_text"]:
                await self._close([current], "answered")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._close([r for r in self._open if r["ended"] and r["heard"]], "answered")
