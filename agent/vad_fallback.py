"""Makes the STT finish a transcript of speech VAD missed, which a streaming STT such as Voxtral only does once VAD reports the end of speech."""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SILENCE_SECS = 1.5


class VADFallback(FrameProcessor):
    """Sits right after the STT and, when interim transcripts arrive without VAD activity and then go quiet, sends the STT the end of speech VAD never reported."""

    def __init__(self, *, silence_secs: float = SILENCE_SECS, **kwargs):
        """Set how long interim transcripts must stay quiet before the fallback acts."""
        super().__init__(**kwargs)
        self._silence_secs = silence_secs
        self._vad_heard = False
        self._timer: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass every frame on, arming the fallback on interims that arrive without any VAD activity."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_heard = True
            await self._disarm()
        elif isinstance(frame, InterimTranscriptionFrame) and not self._vad_heard:
            await self._disarm()
            self._timer = self.create_task(self._wait_for_silence(), f"{self}::silence")
        elif isinstance(frame, TranscriptionFrame):
            await self._disarm()
            self._vad_heard = False

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Stop a pending silence timer."""
        await super().cleanup()
        await self._disarm()

    async def _wait_for_silence(self):
        """After the silence window, make the STT finalize as if VAD had reported the end of speech."""
        await asyncio.sleep(self._silence_secs)
        self._timer = None
        logger.info(f"{self}: VAD missed the user's speech, flushing STT after {self._silence_secs}s of silence")
        await self.push_frame(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _disarm(self):
        """Cancel the silence timer if one is running."""
        if self._timer:
            await self.cancel_task(self._timer)
            self._timer = None
