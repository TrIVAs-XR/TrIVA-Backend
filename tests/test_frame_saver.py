"""The frame saver: writes each camera frame as a PNG and passes every frame on unchanged."""

from PIL import Image
from pipecat.frames.frames import UserImageRawFrame
from pipecat.tests.utils import run_test

from agent.frame_saver import FrameSaver

SIZE = (4, 2)


def _image(fmt: str = "RGB") -> UserImageRawFrame:
    """Build a camera frame as the LiveKit transport would push it."""
    channels = len(fmt)
    return UserImageRawFrame(user_id="unity", image=bytes(range(SIZE[0] * SIZE[1] * channels)), size=SIZE, format=fmt)


async def test_each_camera_frame_is_saved_as_png_and_passed_on(tmp_path):
    """Every image lands in the session folder as a readable PNG of the same size, and is passed on."""
    directory = tmp_path / "P04"
    await run_test(
        FrameSaver(directory),
        frames_to_send=[_image(), _image("RGBA")],
        expected_down_frames=[UserImageRawFrame, UserImageRawFrame],
    )

    saved = sorted(directory.glob("*.png"))
    assert len(saved) == 2
    for path in saved:
        with Image.open(path) as image:
            assert image.size == SIZE


async def test_a_frame_whose_bytes_do_not_match_its_size_is_skipped(tmp_path):
    """A malformed frame is logged and dropped from disk, but still passed on so the pipeline is unaffected."""
    broken = UserImageRawFrame(user_id="unity", image=b"\x00", size=SIZE, format="RGB")
    await run_test(
        FrameSaver(tmp_path),
        frames_to_send=[broken],
        expected_down_frames=[UserImageRawFrame],
    )

    assert not list(tmp_path.glob("*.png"))
