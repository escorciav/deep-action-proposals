import utils


def test_count_frames():
    filename = 'not_existent_video.avi'
    assert utils.count_frames(filename) == 0
    filename = 'data/videos/example.mp4'
    assert utils.count_frames(filename) == 1507
    assert utils.count_frames(filename, 'ffprobe') == 1507
