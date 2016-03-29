import cv2


def dump_video(filename, clip, fourcc_str='X264', fps=30.0):
    """Write video on disk from a stack of images

    Parameters
    ----------
    filename : str
        Fullpath of video-file to generate
    clip : ndarray
        ndarray where first dimension is used to refer to i-th frame
    fourcc_str : str
        str to retrieve fourcc from opencv
    fps : float
        frame rate of create video-stream

    """
    fourcc = cv2.cv.CV_FOURCC(**list(fourcc_str))
    fid = cv2.VideoWriter(filename, fourcc, fps, clip.shape[0:2])
    if fid.isOpened():
        for i in xrange(clip.shape[0]):
                fid.write(clip[i, ...])
        return True
    else:
        return False
