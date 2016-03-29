import os
import tempfile
import unittest

from daps.utils.video import count_frames, dump_video, get_clip


@unittest.skip("Skipping until correct installation of OpenCV")
def test_dump_video():
    video = 'data/videos/example.mp4'
    with tempfile.NamedTemporaryFile() as f:
        filename = f.name
    clip = get_clip(video, 3, 30)
    dump_video(filename, clip)
    assert os.path.isfile(filename) == True
    assert (count_frames(filename) > 0) == True
    os.remove(filename)
