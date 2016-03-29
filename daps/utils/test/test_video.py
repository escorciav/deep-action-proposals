import unittest

from daps.utils.video import count_frames, duration, frame_rate


class test_video(unittest.TestCase):
    def setUp(self):
        self.video = 'data/videos/example.mp4'
        self.video_dir = 'data/videos/example'

    def test_count_frames(self):
        filename = 'not_existent_video.avi'
        assert count_frames(filename) == 0
        assert count_frames(self.video_dir, method='dir') == 1507
        assert count_frames(self.video) == 1507
        assert count_frames(self.video, 'ffprobe') == 1507

    @unittest.skip("A contribution is required")
    def test_dump_frames(self):
        pass

    def test_duration(self):
        assert isinstance(duration(self.video), float)
        assert duration('nonexistent.video') == 0.0

    @unittest.skip("A contribution is required")
    def get_clip(self):
        pass

    def test_frame_rate(self):
        assert isinstance(frame_rate(self.video), float)
        assert frame_rate('nonexistent.video') == 0.0
