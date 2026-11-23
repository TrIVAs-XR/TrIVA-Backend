"""Cuts each voice reading down to what stands out before the LLM gets it, and writes each user turn's full reading to the study log."""

from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, format_perception, merge_perception
from pipecat.frames.frames import (
    Frame,
    LLMContextUserTurnFrame,
    PerceptionMetadataFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.observability.study_log import StudyLog

# How sure a label or how likely a chance has to be for the LLM to get it.
THRESHOLD = 0.7

# Signals the LLM never gets.
DROPPED = ("arousal", "speech_rate")

# The tone label that says nothing stands out.
NEUTRAL = "neutral"


def notable(signals: list[PerceptionSignal]) -> list[PerceptionSignal]:
    """Keep the labels other than neutral and the chances that reach THRESHOLD, from every signal not in DROPPED; a signal with nothing left is gone."""
    kept: list[PerceptionSignal] = []
    for signal in signals:
        if signal.name in DROPPED:
            continue
        if signal.labels:
            labels = [(label, score) for label, score in signal.labels if label != NEUTRAL and score >= THRESHOLD]
            if labels:
                kept.append(PerceptionSignal(signal.name, labels=labels))
        elif signal.value is not None and signal.value >= THRESHOLD:
            kept.append(PerceptionSignal(signal.name, value=signal.value))
    return kept


class ReadingFilter(FrameProcessor):
    """Sits between the voice reader and the user aggregator: gives each transcription only the notable part of its reading, none when nothing stands out, and declares the signals itself without the dropped ones, so the LLM is not taught them."""

    def __init__(self, study: StudyLog, **kwargs):
        """Take the study log each turn's full reading is written to."""
        super().__init__(**kwargs)
        self._study = study
        # The full readings of the turn so far, merged as the user aggregator merges what the LLM gets.
        self._turn: list[PerceptionSignal] | None = None
        # Whether the user aggregator drops transcriptions, as it does until the greeting has finished.
        self._muted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Filter a transcription's reading and replace the declared signals on their way down, follow whether the user is muted, and log the turn's reading when its user turn comes back up."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, PerceptionMetadataFrame):
            # Declared from here, so the RTVI observer counts this filter as a perceiver and holds a transcription pushed to it until it has passed.
            signals = [term for term in frame.perception.signals if term.name not in DROPPED]
            await self.push_frame(PerceptionMetadataFrame(service_name=self.name, perception=PerceptionCapabilities(signals)), direction)
            return
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame) and frame.perception:
            if not self._muted:
                self._turn = merge_perception(self._turn, frame.perception)
            frame.perception = notable(frame.perception) or None
        elif direction == FrameDirection.UPSTREAM and isinstance(frame, (UserMuteStartedFrame, UserMuteStoppedFrame)):
            self._muted = isinstance(frame, UserMuteStartedFrame)
        elif direction == FrameDirection.UPSTREAM and isinstance(frame, LLMContextUserTurnFrame) and not frame.speculation:
            self._log_turn(frame)
        await self.push_frame(frame, direction)

    def _log_turn(self, frame: LLMContextUserTurnFrame):
        """Write the turn's full reading and what the LLM got of it, if the turn had a reading, and start the next turn's over."""
        if self._turn:
            self._study.write("voice_reading", reading=format_perception(self._turn), kept=format_perception(frame.perception) if frame.perception else None)
        self._turn = None
