import __future__
import glob
import os
from subprocess import check_output

import cv2
import natsort
import numpy as np


def count_frames(filename, method=None, ext='*.jpg'):
    """Count number of frames of a video

    Parameters
    ----------
    filename : str
        fullpath of video file
    method : str, optional
        algorithm to use (None, 'ffprobe')
    ext : str, optional
        image extension

    Outputs
    -------
    counter : int
        number of frames

    """
    counter, fail_ffprobe = 0, False
    if isinstance(method, str):
        if method == 'ffprobe':
            cmd = ['ffprobe', '-v', 'error', '-count_frames',
                   '-select_streams', 'v:0', '-show_entries',
                   'stream=nb_read_frames', '-of',
                   'default=nokey=1:noprint_wrappers=1', filename]
            try:
                counter = int(check_output(cmd).replace('\n', ''))
            except:
                counter, fail_ffprobe = 0, True
        else:
            if os.path.isdir(filename):
                counter = len(glob.glob(os.path.join(filename, ext)))

    if method is None or fail_ffprobe:
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret = cap.grab()
            if ret:
                counter += 1
            else:
                break
        cap.release()
    return counter


def dump_frames(filename, output_folder, basename_format=None):
    """Dump frames of a video-file into a folder

    Parameters
    ----------
    filename : str
        Fullpath of video-file
    output_folder : str
        Fullpath of folder to place frames
    basename_format: (None, str)
        String format used to save video frames. If None, the
        format is assigned according the length of the video

    Outputs
    -------
    success : bool

    Note: this function makes use of ffmpeg and its results depends on it.

    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    if basename_format:
        fbasename = basename_format + '.jpg'
    else:
        n_frames, n = count_frames(filename, 'ffprobe'), 0
        while n_frames > 0:
            n_frames /= 10
            n += 1
        fbasename = '%0' + str(max(6, n)) + 'd.jpg'

    output_format = os.path.join(output_folder, fbasename)
    cmd = ['ffmpeg', '-v', 'error', '-i', filename, '-qscale:v', '2', '-f',
           'image2', output_format]
    try:
        check_output(cmd)
    except:
        return False
    return True


def duration(filename):
    """Return duration on seconds of a video

    Parameters
    ----------
    filename : str
        Fullpath of video-file

    Outputs
    -------
    frame_rate : float

    Note: this function makes use of ffprobe and its results depends on it.

    """
    if os.path.isfile(filename):
        cmd = ('ffprobe -v 0 -of flat=s=_ -select_streams v:0 -show_entries ' +
               'stream=duration -of default=nokey=1:noprint_wrappers=1 ' +
               filename).split()
        fr_exp = check_output(cmd)
        return eval(compile(fr_exp, '<string>', 'eval',
                            __future__.division.compiler_flag))
    else:
        return 0.0


def frame_rate(filename):
    """Return frame-rate of video

    Parameters
    ----------
    filename : str
        Fullpath of video-file

    Outputs
    -------
    frame_rate : float

    Note: this function makes use of ffprobe and its results depends on it.

    """
    if os.path.isfile(filename):
        cmd = ('ffprobe -v 0 -of flat=s=_ -select_streams v:0 -show_entries ' +
               'stream=avg_frame_rate -of default=nokey=1:noprint_wrappers=1' +
               ' ' + filename).split()
        fr_exp = check_output(cmd)
        return eval(compile(fr_exp, '<string>', 'eval',
                            __future__.division.compiler_flag))
    else:
        return 0.0


def get_clip(filename, i_frame=0, duration=1, ext='.jpg', img_list=None):
    """Return a clip from a video

    Parameters
    ----------
    filename : str
        Fullpath of video-stream or img-dir
    i_frame : int, optional
        Index of initial frame to capture, 0-indexed.
    duration : int, optional
        duration of clip
    ext : str
        Extension of image-files in case filename is dir
    img_list : list, optional
        list, is a set of strings with basename of images to stack.

    Outputs
    -------
    clip : ndarray
        numpy array of stacked frames

    """
    clip = []
    if os.path.isdir(filename):
        if img_list is None:
            img_files = glob.glob(os.path.join(filename, '*' + ext))
            img_files_s = natsort.natsorted(img_files)
            img_list = img_files_s[i_frame:i_frame + duration]

        # Make a clip from a list of images in filename dir
        if isinstance(img_list, list):
            for i in img_list:
                img_name = i
                if filename not in i:
                    img_name = os.path.join(filename, i)

                if os.path.isfile(img_name):
                    img = cv2.imread(img_name)
                    if img is not None:
                        clip.append(img)
                else:
                    raise IOError('unknown file {}'.format(img_name))
    elif os.path.isfile(filename):
        cap = cv2.VideoCapture(filename)
        for i in xrange(0, i_frame):
            success = cap.grab()
        for i in xrange(0, duration):
            success, img = cap.read()
            if success:
                clip.append(img)
            else:
                break
        cap.release()
    else:
        return None
    return np.stack(clip)
