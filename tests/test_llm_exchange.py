"""The LLMExchange observer: every context that enters the LLM is logged, whichever processor hands it over, and the LLM's own output once."""

import pytest
from loguru import logger

from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from agent.observability.llm_exchange import LLMExchange

# The observer checks whether a hop's end is an LLM service and reads its system prompt, so a bare one with a prompt stands in for the LLM.
LLM = LLMService(settings=LLMSettings(model="test", system_instruction="You are Iva."))
BEFORE = object()
AFTER = object()


@pytest.fixture
def records() -> list[dict]:
    """Collect the LLM records the observer logs during one test, as their message plus extra fields."""
    collected: list[dict] = []
    sink_id = logger.add(
        lambda message: collected.append({"message": message.record["message"], **message.record["extra"]}),
        level="DEBUG",
        filter=lambda record: "llm" in record["extra"],
    )
    yield collected
    logger.remove(sink_id)


def _push(frame, source, destination, *, first_push: bool, direction=FrameDirection.DOWNSTREAM) -> FramePushed:
    """Build one hop of a frame between two processors."""
    return FramePushed(
        source=source,
        destination=destination,
        frame=frame,
        direction=direction,
        timestamp=0,
        first_push=first_push,
    )


def _context() -> LLMContextFrame:
    """Build a context frame holding one user message."""
    return LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": "hello"}]))


def test_every_hop_of_a_frame_is_observed():
    """The observer asks pipecat for every push, since a context's hop into the LLM is not its first."""
    assert LLMExchange().observe_every_push is True


async def test_a_context_passed_on_by_another_processor_is_logged_as_it_enters_the_llm(records):
    """A context reaches the LLM on its second hop, after the context capture passed it on, and is logged there with the LLM's system prompt."""
    exchange = LLMExchange()
    frame = _context()
    await exchange.on_push_frame(_push(frame, object(), BEFORE, first_push=True))
    await exchange.on_push_frame(_push(frame, BEFORE, LLM, first_push=False))
    assert [record["message"] for record in records] == ["LLM input"]
    assert records[0]["system_instruction"] == "You are Iva."
    assert records[0]["messages"] == [{"role": "user", "content": "hello"}]


async def test_a_context_moving_between_other_processors_is_not_logged(records):
    """Only the hop into the LLM counts as an LLM call; a context moving anywhere else leaves no record."""
    exchange = LLMExchange()
    await exchange.on_push_frame(_push(_context(), BEFORE, AFTER, first_push=True))
    assert records == []


async def test_a_broadcast_tool_call_and_its_result_are_logged_once_each(records):
    """The LLM sends a tool call and its result as one copy downstream and one upstream, and each is logged once."""
    exchange = LLMExchange()
    call = FunctionCallFromLLM(function_name="start_task", tool_call_id="1", arguments={"step": 1}, context=LLMContext())
    for direction in (FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM):
        await exchange.on_push_frame(_push(FunctionCallsStartedFrame(function_calls=[call]), LLM, AFTER, first_push=True, direction=direction))
        result = FunctionCallResultFrame(function_name="start_task", tool_call_id="1", arguments={"step": 1}, result="ok")
        await exchange.on_push_frame(_push(result, LLM, AFTER, first_push=True, direction=direction))
    assert [(record["message"], record["name"]) for record in records] == [("LLM tool call", "start_task"), ("LLM tool result", "start_task")]
    assert records[0]["arguments"] == {"step": 1}
    assert records[1]["result"] == "ok"


async def test_a_context_arriving_upstream_is_logged(records):
    """A call on a tool result reaches the LLM upstream from the processor after it, and is logged too."""
    exchange = LLMExchange()
    await exchange.on_push_frame(_push(_context(), AFTER, LLM, first_push=False, direction=FrameDirection.UPSTREAM))
    assert [record["message"] for record in records] == ["LLM input"]


async def test_the_llms_output_is_logged_once(records):
    """Text the LLM pushes out is logged once as a whole response."""
    exchange = LLMExchange()
    for frame in (LLMFullResponseStartFrame(), LLMTextFrame(text="Hi "), LLMTextFrame(text="there"), LLMFullResponseEndFrame()):
        await exchange.on_push_frame(_push(frame, LLM, AFTER, first_push=True))
    assert [(record["message"], record.get("text")) for record in records] == [("LLM output", "Hi there")]


async def test_frames_the_llm_only_passes_along_are_not_logged(records):
    """A frame that came from elsewhere and only passes through the LLM leaves no record."""
    exchange = LLMExchange()
    result = FunctionCallResultFrame(function_name="start_task", tool_call_id="1", arguments={}, result="ok")
    await exchange.on_push_frame(_push(result, LLM, BEFORE, first_push=False, direction=FrameDirection.UPSTREAM))
    assert records == []
