"""Holds the newest camera frame the client sends, for whatever decides to put it into the LLM context; until then the frame goes no further."""

from pipecat.frames.frames import Frame, UserImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class Camera(FrameProcessor):
    """Sits right after the transport input, after the frame saver, keeps each camera frame in place of the one before and stops it there, and passes every other frame on."""

    def __init__(self, **kwargs):
        """Start with no frame held."""
        super().__init__(**kwargs)
        self._latest: UserImageRawFrame | None = None

    @property
    def latest(self) -> UserImageRawFrame | None:
        """The newest camera frame, or None before the first one arrives."""
        return self._latest

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Hold a camera frame nobody asked for; pass on every other frame, including an image that answers a request."""
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRawFrame) and frame.request is None:
            self._latest = frame
            return
        await self.push_frame(frame, direction)
