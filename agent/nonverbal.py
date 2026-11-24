"""Referring to objects: the LLM marks the object it means in its reply, and the agent looks and points at it when the words before the marker have been spoken."""

from collections.abc import Awaitable, Callable
from typing import Any

from pipecat.frames.frames import AggregatedTextFrame, Frame, LLMMessagesTransformFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.transports.base_output import BaseOutputTransport

from agent.game import wrap
from agent.observability.study_log import StudyLog

# The tag the LLM marks an object with in its reply, as <refer>flask-3</refer>, and the text type the reply strip gives the marker, which the TTS and the client's bot output leave out.
REFER = "refer"

# The tag the objects the client lists at the start are wrapped in within the context.
OBJECTS_TAG = "objects"

# The RTVI server-message type the agent sends each nonverbal behaviour under, as {"t": "nonverbal", "d": {"behaviour": ..., ...}}.
NONVERBAL = "nonverbal"

# Added to every scenario's system prompt, so any scenario can refer to the objects the client tells it about.
INSTRUCTIONS = """Referring to objects
You stand in the same space as the participant and can look and point at an object while you talk about it. To do so, write <refer>ID</refer> at the start of the sentence about that object, for example "<refer>flask-3</refer>That one is right." The marker is not spoken; you look and point as you start the sentence.
- You know an object's ID from the <objects> message, which lists the objects in the scene at the start, and from the <held> and <pointing> messages. Use an ID exactly as written there. Never make one up, and never refer to an object whose id is null.
- Refer only when it helps the participant see which object you mean, at most once per sentence. Never answer with a marker alone; it always starts a sentence you say.
- Never say an ID aloud."""


def prompt(scenario_prompt: str) -> str:
    """Return a scenario's system prompt with the instructions for referring to objects after it."""
    return f"{scenario_prompt}\n\n{INSTRUCTIONS}"


class SceneObjects:
    """Keeps the objects the client lists at the start in the LLM context, as one message swapped in place of the previous list, so the LLM knows their IDs."""

    def __init__(self, queue: Callable[[list[Frame]], Awaitable[None]]):
        """Take the worker's way to queue frames."""
        self._queue = queue
        self._message: dict | None = None

    async def receive(self, data: Any):
        """Swap the client's list into the context in place of the previous one, without running the LLM."""
        previous = self._message
        message = {"role": "developer", "content": wrap(OBJECTS_TAG, data)}
        self._message = message

        def swap(messages: list) -> list:
            """Runs inside the user aggregator, on the context the next LLM call reads."""
            return [m for m in messages if m is not previous] + [message]

        await self._queue([LLMMessagesTransformFrame(transform=swap, run_llm=False)])


class ReferRelay(BaseObserver):
    """Sends the client a refer behaviour for each marker in the reply as the output transport passes it on, which is once the audio before it has been played out, and writes it to the study log."""

    def __init__(self, rtvi: RTVIProcessor, study: StudyLog, **kwargs):
        """Send through the worker's RTVI processor and log to the study log."""
        super().__init__(observe_every_push=True, **kwargs)
        self._rtvi = rtvi
        self._study = study

    async def on_push_frame(self, data: FramePushed):
        """Send a marker's behaviour when the output transport pushes it on; a marker without an ID is ignored."""
        frame = data.frame
        if not isinstance(data.source, BaseOutputTransport) or not isinstance(frame, AggregatedTextFrame) or isinstance(frame, TTSTextFrame):
            return
        if frame.text_type != REFER or not frame.text.strip():
            return
        behaviour = {"behaviour": REFER, "target": frame.text.strip()}
        await self._rtvi.send_server_message({"t": NONVERBAL, "d": behaviour})
        self._study.write(NONVERBAL, **behaviour)
