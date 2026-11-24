"""The reply strip: a voice reading or a tagged game snapshot the LLM copies into its reply is neither spoken nor kept in the context, an object marker is passed on unspoken between the words around it, and other bracketed text is left alone."""

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMContextAssistantTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

from agent.game import TAG, wrap
from agent.nonverbal import OBJECTS_TAG, REFER
from agent.reply_strip import reply_strip, strip_readings

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


# A snapshot as session 64's LLM made one up, in the tagged form, with a description holding a sentence break so the block spans sentences.
SNAPSHOT = {"event": "flask_placed", "detail": "erlenmeyer", "recipe": {"flask": "erlenmeyer", "note": "Placed. Now add reagents."}, "complete": False}


async def _streamed(reply: str) -> tuple[list[str], list[str], list[str]]:
    """Stream a reply in pieces through the reply strip as pipeline.py sets it up, and return what was spoken, what the context holds and the turns the study log records."""
    tts = _SpokenTTS(skip_text_types=[REFER])
    context = LLMContext()
    down, _ = await run_test(
        Pipeline([reply_strip(), tts, LLMContextAggregatorPair(context).assistant()]),
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
    return tts.spoken, [m["content"] for m in context.get_messages()], [f.text for f in down if isinstance(f, LLMContextAssistantTurnFrame)]


async def test_a_streamed_reply_reaches_the_tts_and_the_context_without_its_reading():
    """A reading streamed in pieces is gone from what is spoken, from the context and from the turn the study log records."""
    spoken, context, turns = await _streamed(f"{LEAKED} Oh, that's great. Let's start.")
    whole = "Oh, that's great. Let's start."
    assert spoken == ["Oh, that's great.", "Let's start."]
    assert context == [whole]
    assert turns == [whole]


async def test_a_made_up_snapshot_is_dropped_and_the_rest_of_the_reply_kept():
    """A tagged snapshot the LLM writes at the head of its reply is dropped whole, even across a sentence break inside it, and the words after it are spoken and kept."""
    spoken, context, turns = await _streamed(f"{wrap(TAG, SNAPSHOT)}\n\nNow tell me the first code on the panel.")
    whole = "Now tell me the first code on the panel."
    assert spoken == [whole]
    assert context == [whole]
    assert turns == [whole]


async def test_a_copied_object_list_is_dropped():
    """An object list the LLM copies into its reply is dropped whole, like a snapshot, and the words after it are spoken and kept."""
    spoken, context, _ = await _streamed(f'{wrap(OBJECTS_TAG, [{"id": "ice_bath", "kind": "object"}])} Good.')
    assert spoken == ["Good."]
    assert context == ["Good."]


async def test_an_unclosed_block_is_dropped_at_the_end_of_the_reply():
    """A block the reply ends inside never reaches the speech or the context, while the words before it do."""
    spoken, context, _ = await _streamed('Sure. <held>\n{"left": "beaker"')
    assert spoken == ["Sure."]
    assert context == ["Sure."]


async def test_a_lone_angle_bracket_is_spoken():
    """Text with a bracket that opens no known tag passes unchanged."""
    spoken, _, _ = await _streamed("Keep it < 40 degrees. Good.")
    assert spoken == ["Keep it < 40 degrees.", "Good."]


async def test_a_marker_is_not_spoken_and_stays_in_the_context():
    """An object marker reaches neither the TTS nor the words around it, and the context keeps it where the LLM wrote it."""
    spoken, context, _ = await _streamed("Yes, <refer>flask-3</refer>that one. Pour it in.")
    assert spoken == ["Yes,", "that one.", "Pour it in."]
    assert len(context) == 1 and "<refer>flask-3</refer>" in context[0]


async def test_a_marker_leaves_the_tts_between_the_words_around_it():
    """The TTS passes a marker on after the words before it and ahead of the words after it, so its behaviour can be sent as those words are played."""
    reply = "Yes, <refer>flask-3</refer>that one. Pour it in."
    down, _ = await run_test(
        Pipeline([reply_strip(), _SpokenTTS(skip_text_types=[REFER])]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            *[LLMTextFrame(reply[i : i + 7]) for i in range(0, len(reply), 7)],
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.2),
        ],
    )
    units = [(f.text_type, f.text) for f in down if isinstance(f, TTSTextFrame) or (isinstance(f, AggregatedTextFrame) and f.text_type == REFER)]
    assert units == [("sentence", "Yes,"), (REFER, "flask-3"), ("sentence", "that one."), ("sentence", "Pour it in.")]
