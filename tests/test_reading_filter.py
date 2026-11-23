"""The reading filter: the LLM and the client get only what stands out in a reading, or none, the LLM is not taught the dropped signals, and the study log keeps each turn's full reading once."""

import json

import pytest
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, PerceptionTerm, merge_perception
from pipecat.frames.frames import (
    Frame,
    LLMContextUserTurnFrame,
    PerceptionMetadataFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver
from pipecat.tests.utils import SleepFrame, run_test

from agent.observability.study_log import StudyLog
from agent.reading_filter import ReadingFilter, notable

# How long a test waits for a frame sent back upstream to be handled, and between turns.
PAUSE = 0.2

SIGNALS = ["tone", "arousal", "laughter", "speech_rate", "sarcasm_voice"]


def _reading(tone: list[tuple[str, float]], arousal: float, laughter: float, speech_rate: float, sarcasm: float) -> list[PerceptionSignal]:
    """Build a reading as Pipecat Voice Reader makes one."""
    return [
        PerceptionSignal("tone", labels=tone),
        PerceptionSignal("arousal", value=arousal),
        PerceptionSignal("laughter", value=laughter),
        PerceptionSignal("speech_rate", value=speech_rate),
        PerceptionSignal("sarcasm_voice", value=sarcasm),
    ]


# "What the hell was that?" from a session with a real voice: frustrated, and laughing.
FRUSTRATED = _reading([("frustrated", 0.71), ("positive", 0.16)], 0.6, 0.91, 4.93, 0.04)

# "It shows P2." from the same session: nothing stands out.
CALM = _reading([("neutral", 0.94), ("sad", 0.03)], 0.19, 0.0, 3.72, 0.01)


def _transcription(perception: list[PerceptionSignal] | None) -> TranscriptionFrame:
    """Build a final transcription as the voice reader passes it on."""
    return TranscriptionFrame(text="Words.", user_id="unity", timestamp="", perception=perception)


def _end_of_turn(speculation: bool = False) -> LLMContextUserTurnFrame:
    """Tell the stand-in aggregator to write the turn so far, as the real one does when the turn ends or is speculated."""
    return LLMContextUserTurnFrame(text="", timestamp="", speculation=speculation)


class _Aggregator(FrameProcessor):
    """Stands in for the user aggregator: merges the readings it gets, drops transcriptions while muted, and sends the turn back upstream with them when told the turn has ended."""

    def __init__(self, **kwargs):
        """Start with no turn, unmuted."""
        super().__init__(**kwargs)
        self._perception: list[PerceptionSignal] | None = None
        self._muted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Follow a mute and send it back upstream, drop or merge a transcription's reading, answer an end of turn with the turn upstream, and pass everything else on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, (UserMuteStartedFrame, UserMuteStoppedFrame)):
            self._muted = isinstance(frame, UserMuteStartedFrame)
            await self.push_frame(frame, FrameDirection.UPSTREAM)
            return
        if isinstance(frame, TranscriptionFrame) and self._muted:
            return
        if isinstance(frame, TranscriptionFrame) and frame.perception:
            self._perception = merge_perception(self._perception, frame.perception)
        elif isinstance(frame, LLMContextUserTurnFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(LLMContextUserTurnFrame(text="Words.", timestamp="", perception=self._perception, speculation=frame.speculation), FrameDirection.UPSTREAM)
            if not frame.speculation:
                self._perception = None
            return
        await self.push_frame(frame, direction)


class _STT(FrameProcessor):
    """Stands in for the STT: turns a text frame into a final transcription it pushes itself, as the STT is the first to push one."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Push a transcription in place of a text frame, and pass everything else on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            frame = _transcription(None)
        await self.push_frame(frame, direction)


class _Reader(FrameProcessor):
    """Stands in for the voice reader: declares its signals at start and puts a frustrated reading on each transcription."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Declare the signals after the start frame, read each transcription, and pass every frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            frame.perception = list(FRUSTRATED)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(PerceptionMetadataFrame(service_name="reader", perception=PerceptionCapabilities([PerceptionTerm(name) for name in SIGNALS])))


class _Client(RTVIObserver):
    """The RTVI observer, keeping the user-input messages it would send the client instead of sending them."""

    def __init__(self):
        """Keep no messages yet."""
        super().__init__(None)
        self.inputs: list[dict] = []

    async def send_rtvi_message(self, model, exclude_none: bool = True):
        """Keep a user-input message's data."""
        message = model.model_dump()
        if message["type"] == "user-input":
            self.inputs.append(message["data"])


@pytest.fixture
def path(tmp_path):
    """Where the session under test writes its log."""
    return tmp_path / "study" / "P01.jsonl"


@pytest.fixture
def study(path):
    """A study log for the session under test."""
    log = StudyLog(path)
    yield log
    log.close()


def _readings(path) -> list[dict]:
    """The voice_reading events written so far."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line["event"] == "voice_reading"]


async def _turns(study: StudyLog, frames: list[Frame]) -> list[Frame]:
    """Run frames through the filter and the stand-in aggregator, and return what reached the aggregator's far side."""
    down, _ = await run_test(Pipeline([ReadingFilter(study), _Aggregator()]), frames_to_send=[*frames, SleepFrame(sleep=PAUSE)])
    return down


def test_only_labels_other_than_neutral_and_chances_that_reach_the_threshold_are_kept():
    """Arousal and speech rate never pass, and neither does a weak label or a small chance."""
    assert notable(FRUSTRATED) == [PerceptionSignal("tone", labels=[("frustrated", 0.71)]), PerceptionSignal("laughter", value=0.91)]


def test_a_sure_neutral_tone_is_not_notable():
    """Neutral says nothing stands out, however sure the reading is."""
    assert notable(CALM) == []


async def test_a_transcription_carries_only_the_notable_part_and_the_turn_logs_the_full_reading(study, path):
    """The LLM gets the filtered reading, and the study log both readings once the turn is written."""
    down = await _turns(study, [_transcription(FRUSTRATED), _end_of_turn()])
    [transcription] = [frame for frame in down if isinstance(frame, TranscriptionFrame)]
    assert transcription.perception == notable(FRUSTRATED)
    [line] = _readings(path)
    assert line["reading"].startswith("([tone frustrated=0.71, positive=0.16],[arousal value=0.6]")
    assert line["kept"] == "([tone frustrated=0.71],[laughter value=0.91])"


async def test_a_turn_of_several_transcriptions_logs_one_merged_reading(study, path):
    """Each signal keeps its highest value across the turn, as the LLM's reading does."""
    await _turns(study, [_transcription(CALM), _transcription(FRUSTRATED), _end_of_turn()])
    [line] = _readings(path)
    assert line["reading"].startswith("([tone neutral=0.94, frustrated=0.71, positive=0.16, sad=0.03],[arousal value=0.6]")
    assert line["kept"] == "([tone frustrated=0.71],[laughter value=0.91])"


async def test_each_turn_logs_its_own_reading(study, path):
    """A written turn starts the next one over."""
    await _turns(study, [_transcription(FRUSTRATED), _end_of_turn(), SleepFrame(sleep=PAUSE), _transcription(CALM), _end_of_turn()])
    assert [line["kept"] for line in _readings(path)] == ["([tone frustrated=0.71],[laughter value=0.91])", None]


async def test_a_speculated_turn_logs_nothing(study, path):
    """Only the turn written to the context counts, so a speculation ahead of it neither logs nor starts the turn over."""
    await _turns(study, [_transcription(FRUSTRATED), _end_of_turn(speculation=True), SleepFrame(sleep=PAUSE), _end_of_turn()])
    [line] = _readings(path)
    assert line["kept"] == "([tone frustrated=0.71],[laughter value=0.91])"


async def test_a_reading_while_muted_stays_out_of_the_next_turn(study, path):
    """Speech over the greeting never reaches the context, so its reading is not logged with the first turn."""
    await _turns(study, [UserMuteStartedFrame(), SleepFrame(sleep=PAUSE), _transcription(FRUSTRATED), SleepFrame(sleep=PAUSE), UserMuteStoppedFrame(), SleepFrame(sleep=PAUSE), _transcription(CALM), _end_of_turn()])
    [line] = _readings(path)
    assert line["reading"].startswith("([tone neutral=0.94, sad=0.03]")
    assert line["kept"] is None


async def test_a_reading_with_nothing_notable_is_taken_off_the_transcription(study, path):
    """A calm turn reaches the LLM as words only, and is still logged in full."""
    down = await _turns(study, [_transcription(CALM), _end_of_turn()])
    [transcription] = [frame for frame in down if isinstance(frame, TranscriptionFrame)]
    assert transcription.perception is None
    [line] = _readings(path)
    assert line["kept"] is None
    assert "speech_rate" in line["reading"]


async def test_a_turn_without_a_reading_is_not_logged(study, path):
    """A turn too short for the voice reader has no reading to filter or log."""
    down = await _turns(study, [_transcription(None), _end_of_turn()])
    [transcription] = [frame for frame in down if isinstance(frame, TranscriptionFrame)]
    assert transcription.perception is None
    assert _readings(path) == []


async def test_the_declared_signals_leave_out_the_dropped_ones(study):
    """The LLM is taught every signal it can still get, and not arousal or speech rate."""
    declared = PerceptionMetadataFrame(service_name="reader", perception=PerceptionCapabilities([PerceptionTerm(name) for name in SIGNALS]))
    down, _ = await run_test(ReadingFilter(study), frames_to_send=[declared], expected_down_frames=[PerceptionMetadataFrame])
    assert [term.name for term in down[0].perception.signals] == ["tone", "laughter", "sarcasm_voice"]


async def test_the_client_is_sent_the_filtered_reading_when_the_reader_follows_the_stt(study):
    """The filter declares the signals itself, so the RTVI observer sends a transcription only once it has passed the filter."""
    client = _Client()
    await run_test(Pipeline([_STT(), _Reader(), ReadingFilter(study)]), frames_to_send=[TextFrame(text="go")], observers=[client])
    assert [data["perception"] for data in client.inputs] == [[{"name": "tone", "labels": [{"label": "frustrated", "confidence": 0.71}], "value": None}, {"name": "laughter", "labels": [], "value": 0.91}]]
