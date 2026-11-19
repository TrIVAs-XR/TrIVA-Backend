"""Writes every camera frame the client sends to disk as a PNG, for checking by eye what the agent would see."""

import asyncio
from datetime import datetime
from pathlib import Path

from loguru import logger
from PIL import Image
from pipecat.frames.frames import Frame, UserImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DIRECTORY = Path(__file__).resolve().parent.parent / "logs" / "frames"


class FrameSaver(FrameProcessor):
    """Sits right after the transport input, saves each incoming image to its session folder and passes every frame on unchanged."""

    def __init__(self, directory: Path, **kwargs):
        """Set the folder the session's frames are written to; it is created with the first frame."""
        super().__init__(**kwargs)
        self._directory = directory

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Save user images off the event loop, then pass every frame on."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRawFrame):
            await asyncio.to_thread(self._save, frame, datetime.now())

        await self.push_frame(frame, direction)

    def _save(self, frame: UserImageRawFrame, received: datetime):
        """Encode one frame as a PNG named by its arrival time to the millisecond."""
        stamp = f"{received:%Y%m%d_%H%M%S_%f}"[:-3]
        path = self._directory / f"{stamp}.png"
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            Image.frombytes(frame.format or "RGB", frame.size, frame.image).save(path)
        except (ValueError, OSError) as error:
            logger.warning(f"Frame from {frame.user_id} not saved: {error}")
