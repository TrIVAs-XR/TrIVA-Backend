"""The VAD fallback: flushes the STT for speech VAD missed, stays out of turns VAD saw, and lets such a turn reach the LLM once behind the voice reader."""

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.classifiers.base_classifier import BaseClassifier, YesNoQuestion, YesNoResult
from pipecat.expressive.processors.perception_reader import PerceptionQuestion, PerceptionReader, PerceptionSource
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from agent.vad_fallback import VADFallback

SILENCE = 0.1


class _LevelClassifier(BaseClassifier):
    """Stands in for the voice reader's model: answers with the level of the audio it heard, so a reading shows which speech was read."""

    @property
    def supports_audio(self) -> bool:
        """Hears audio, as the voice reader's model does."""
        return True

    @property
    def min_audio_secs(self) -> float:
        """Reads clips as short as the test's speech."""
        return 0.1

    @property
    def audio_sample_rate(self) -> int:
        """Takes clips at the pipeline's input rate."""
        return 16000

    async def _ask(self, state, questions, inputs=()):
        """Answer every question with the clip's first sample, divided by 100."""
        level = inputs[0].audio[0] / 100
        return {name: YesNoResult(probability=level) for name in questions}, None


class _LLMCalls(FrameProcessor):
    """Records the user messages each LLM call would read, as the call passes."""

    def __init__(self, **kwargs):
        """Start with no calls."""
        super().__init__(**kwargs)
        self.calls: list[list[str]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Copy the user messages of each context frame, then pass every frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.calls.append([m["content"] for m in frame.context.get_messages() if m.get("role") == "user"])
        await self.push_frame(frame, direction)


def _speech(level: int, secs: float) -> InputAudioRawFrame:
    """Build a stretch of 16 kHz mono audio whose every sample has this level."""
    return InputAudioRawFrame(audio=bytes([level, 0]) * int(16000 * secs), sample_rate=16000, num_channels=1)


def _interim(text: str) -> InterimTranscriptionFrame:
    """Build an interim transcript as the STT would push it."""
    return InterimTranscriptionFrame(text=text, user_id="unity", timestamp="")


def _final(text: str) -> TranscriptionFrame:
    """Build a final transcript as the STT would push it."""
    return TranscriptionFrame(text=text, user_id="unity", timestamp="")


async def test_interims_without_vad_flush_stt():
    """Silence after VAD-less interims flushes the STT, and the final that follows passes through alone."""
    await run_test(
        VADFallback(silence_secs=SILENCE),
        frames_to_send=[_interim("Test"), SleepFrame(sleep=SILENCE * 3), _final("Test")],
        expected_down_frames=[InterimTranscriptionFrame, TranscriptionFrame],
        expected_up_frames=[VADUserStoppedSpeakingFrame],
    )


async def test_each_interim_restarts_the_silence_window():
    """Interims arriving faster than the window keep the fallback from firing."""
    await run_test(
        VADFallback(silence_secs=SILENCE * 3),
        frames_to_send=[
            _interim("Ja"),
            SleepFrame(sleep=SILENCE * 2),
            _interim("Ja, ich"),
            SleepFrame(sleep=SILENCE * 2),
            _final("Ja, ich"),
        ],
        expected_down_frames=[InterimTranscriptionFrame, InterimTranscriptionFrame, TranscriptionFrame],
        expected_up_frames=[],
    )


async def test_interims_after_vad_start_leave_the_turn_to_vad():
    """Once VAD has heard the user, interims and silence do not trigger the fallback."""
    await run_test(
        VADFallback(silence_secs=SILENCE),
        frames_to_send=[VADUserStartedSpeakingFrame(), _interim("Hallo"), SleepFrame(sleep=SILENCE * 3)],
        expected_down_frames=[VADUserStartedSpeakingFrame, InterimTranscriptionFrame],
        expected_up_frames=[],
    )


async def test_final_without_fallback_passes_through():
    """A final transcript on the normal path passes through alone."""
    await run_test(
        VADFallback(silence_secs=SILENCE),
        frames_to_send=[_interim("Hallo"), _final("Hallo")],
        expected_down_frames=[InterimTranscriptionFrame, TranscriptionFrame],
        expected_up_frames=[],
    )


async def test_a_turn_vad_missed_reaches_the_llm_once_and_unread():
    """Behind the fallback, the voice reader and the user aggregator as pipeline.py sets them up, a turn VAD missed reaches the LLM in one call, without the reading of the turn before it."""
    reader = PerceptionReader(
        sources=[
            PerceptionSource(
                classifier=_LevelClassifier(),
                questions={"level": PerceptionQuestion(question=YesNoQuestion(instructions="How loud?"), description="The level.")},
            )
        ]
    )
    user = LLMContextAggregatorPair(
        LLMContext(messages=[{"role": "system", "content": "Test."}]),
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]),
        ),
    ).user()
    calls = _LLMCalls()
    await run_test(
        Pipeline([VADFallback(silence_secs=SILENCE), reader, user, calls]),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _speech(11, 1.0),
            VADUserStoppedSpeakingFrame(stop_secs=0.2),
            _final("First turn."),
            SleepFrame(sleep=3.0),
            _interim("Second"),
            SleepFrame(sleep=SILENCE * 3),
            _final("Second turn."),
            SleepFrame(sleep=3.0),
        ],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )
    assert calls.calls == [
        ["([level value=0.11]) First turn."],
        ["([level value=0.11]) First turn.", "Second turn."],
    ]
