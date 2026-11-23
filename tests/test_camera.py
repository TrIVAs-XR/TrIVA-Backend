"""The camera: holds the newest camera frame and keeps it out of the pipeline, so a camera frame can no longer split the agent's reply in two."""

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMContextAssistantTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSTextFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.base_text_aggregator import TextType

from agent.camera import Camera


def _image(shade: int, request: UserImageRequestFrame | None = None) -> UserImageRawFrame:
    """Build a 1x1 camera frame of one grey shade, as the transport pushes it."""
    return UserImageRawFrame(user_id="unity", image=bytes([shade] * 3), size=(1, 1), format="RGB", request=request)


def _spoken(text: str) -> TTSTextFrame:
    """Build a sentence of the agent's reply as the TTS pushes it after speaking it."""
    return TTSTextFrame(text=text, text_type=TextType.SENTENCE)


async def test_camera_frames_stop_and_the_newest_is_held():
    """Unrequested camera frames go no further, and the camera keeps only the latest."""
    camera = Camera()
    assert camera.latest is None
    await run_test(camera, frames_to_send=[_image(1), _image(2)], expected_down_frames=[])
    assert camera.latest is not None and camera.latest.image == bytes([2] * 3)


async def test_an_image_that_answers_a_request_passes_through():
    """An image the pipeline asked for reaches whoever asked, and does not replace the held frame."""
    camera = Camera()
    request = UserImageRequestFrame(user_id="unity")
    await run_test(camera, frames_to_send=[_image(1), _image(2, request)], expected_down_frames=[UserImageRawFrame])
    assert camera.latest.image == bytes([1] * 3)


async def test_a_camera_frame_during_a_reply_leaves_the_reply_whole():
    """With the camera before the assistant aggregator, a frame arriving mid-reply no longer commits half the reply, so the context and the turn carry all of it."""
    context = LLMContext()
    down, _ = await run_test(
        Pipeline([Camera(), LLMContextAggregatorPair(context).assistant()]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            BotStartedSpeakingFrame(),
            _spoken("Nice to meet you!"),
            SleepFrame(sleep=0.1),
            _image(1),
            _spoken("How are you doing today?"),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.1),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
        ],
    )
    whole = "Nice to meet you! How are you doing today?"
    assert [m["content"] for m in context.get_messages() if m.get("role") == "assistant"] == [whole]
    assert [f.text for f in down if isinstance(f, LLMContextAssistantTurnFrame)] == [whole]
