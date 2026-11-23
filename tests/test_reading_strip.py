"""The reading strip: a voice reading the LLM copies into its reply is neither spoken nor kept in the context, and other bracketed text is left alone."""

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMContextAssistantTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

from agent.reading_strip import strip_readings

# The reading session 61's LLM put at the head of a reply, which the TTS then spoke.
LEAKED = "([tone neutral=0.99, positive=0.00, frustrated=0.01],[arousal value=0.30],[laughter value=0.01],[speech_rate value=3.92],[sarcasm_voice value=0.01])"


class _SpokenTTS(TTSService):
    """Stands in for the TTS: records each sentence it is asked to speak and makes no audio."""

    def __init__(self, **kwargs):
        """Start with nothing spoken."""
        super().__init__(sample_rate=24000, settings=TTSSettings(model=None, voice=None, language=None), **kwargs)
        self.spoken: list[str] = []

    def can_generate_metrics(self) -> bool:
        """Report no metrics, as there is no model."""
        return False

    async def run_tts(self, text: str, context_id: str):
        """Record the sentence."""
        self.spoken.append(text)
        yield None


@pytest.mark.parametrize(
    "text, expected",
    [
        (f"{LEAKED} Oh, that's great.", "Oh, that's great."),
        ("([tone frustrated=0.71, neutral=0.22]) Where is it?", "Where is it?"),
        (f"Sure. {LEAKED} Let's start.", "Sure. Let's start."),
        ("([Pitch_f0 value=-1e-05]) Fine.", "Fine."),
        ("Take the beaker [the one on the left] first.", "Take the beaker [the one on the left] first."),
        ("Nice to meet you, [name].", "Nice to meet you, [name]."),
        ("Mix it (slowly) now.", "Mix it (slowly) now."),
    ],
)
async def test_only_a_reading_is_stripped(text, expected):
    """A reading goes wherever it stands in the sentence; brackets and parentheses that are not a reading stay."""
    assert await strip_readings(text, "sentence") == expected


async def test_a_streamed_reply_reaches_the_tts_and_the_context_without_its_reading():
    """Behind the LLM text processor as pipeline.py sets it up, a reading streamed in pieces is gone from what is spoken, from the context and from the turn the study log records."""
    reply = f"{LEAKED} Oh, that's great. Let's start."
    tts = _SpokenTTS()
    context = LLMContext()
    down, _ = await run_test(
        Pipeline([LLMTextProcessor(text_transforms=[("*", strip_readings)]), tts, LLMContextAggregatorPair(context).assistant()]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            BotStartedSpeakingFrame(),
            *[LLMTextFrame(reply[i : i + 7]) for i in range(0, len(reply), 7)],
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.2),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
        ],
    )
    whole = "Oh, that's great. Let's start."
    assert tts.spoken == ["Oh, that's great.", "Let's start."]
    assert [m["content"] for m in context.get_messages()] == [whole]
    assert [f.text for f in down if isinstance(f, LLMContextAssistantTurnFrame)] == [whole]
