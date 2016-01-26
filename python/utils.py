import glob
import os
import __future__
from subprocess import check_output

import cv2
import numpy as np
import natsort
import pandas as pd


def c3d_input_file_generator(filename, output_file, t_size=16, step_size=8,
                             output_folder=None, mkdir=True):
    """Generate textfile(s) used by C3D code

    Parameters
    ----------
    filename : string
        fullpath of CSV file (space separated of 5 columns without header) with
        data about the videos to process.
    output_file : string or list
        list of output files to generate
    t_size : int, optional
        size of temporal field of your C3D network.
    step_size : int, optional
        control how dense do you plan to extract clips of your long video.
    output_folder : string, optional
        fullpath of folder to allocate outputs. Pass it to generate output-file
        for extract-image-features.
    mkdir : bool, optional
        create folder to allocate outputs for C3D extract-image-features

    Outputs
    -------
    summary : dict
        statistics about the process

    """
    summary = dict(success=False, output=None)
    col_names = ['video-name', 'num-frame', 'i-frame', 'duration', 'label']
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=col_names)
    except:
        return summary
    if isinstance(output_file, str):
        output_file = [output_file]

    n_segments = df.shape[0]
    idx_bool_keep = df['duration'] >= t_size
    # Compute number of clips and from where to extract clips from each
    # activity-segment
    n_clips = ((df['duration'] - t_size) / step_size + 1).astype('int')
    init_end_frame = np.concatenate(
        [np.arange(df.loc[i, 'i-frame'],
                   df.loc[i, 'i-frame'] + df.loc[i, 'duration'],
                   step_size)[:n_clips[i]]
         for i in df.index[idx_bool_keep]])

    # Create new data frame with data required for C3D input layer
    idx_keep_expanded = np.repeat(np.array(df.index[idx_bool_keep]),
                                  n_clips[idx_bool_keep])
    df_c3d = pd.DataFrame({1: df.loc[idx_keep_expanded, 'video-name'],
                           2: init_end_frame,
                           3: df.loc[idx_keep_expanded, 'label']})
    try:
        df_c3d.to_csv(output_file[0], sep=' ', header=False, index=False)
    except:
        return summary

    # Dump outpu file
    if len(output_file) > 1 and isinstance(output_folder, str):
        skip_mkdir = False
        try:
            sr_out = (output_folder + os.path.sep +
                      df.loc[idx_keep_expanded, 'video-name'].astype(str) +
                      os.path.sep + init_end_frame.astype(str))
            # Avoid redundacy such that A//B introduce by previous cmd. A
            # principled solution is welcome.
            sr_out = sr_out.apply(os.path.normpath)
            sr_out.to_csv(output_file[1], header=None, index=False)
        except:
            skip_mkdir = True

        if not skip_mkdir:
            summary['output'] = output_file[1]
            if mkdir:
                for i in sr_out:
                    os.makedirs(i)

    # Update summary
    summary['success'] = True
    summary['pctg-skipped-segments'] = ((n_segments - idx_bool_keep.sum()) *
                                        1.0 / n_segments)
    summary['ratio-clips-segments'] = (idx_keep_expanded.size * 1.0 /
                                       n_segments)
    return summary

# General purpose functions


# Video utilities
def count_frames(filename, method=None, ext='*.jpg'):
    """Count number of frames of a video

    Parameters
    ----------
    filename : string
        fullpath of video file
    method : string, optional
        algorithm to use (None, 'ffprobe')
    ext : string, optional
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


def dump_frames(filename, output_folder):
    """Dump frames of a video-file into a folder

    Parameters
    ----------
    filename : string
        Fullpath of video-file
    output_folder : string
        Fullpath of folder to place frames

    Outputs
    -------
    success : bool

    Note: this function makes use of ffmpeg and its results depends on it.

    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    n_frames, n = count_frames(filename, 'ffprobe'), 0
    while n_frames > 0:
        n_frames /= 10
        n += 1

    output_format = os.path.join(output_folder,
                                 '%0' + str(max(6, n)) + 'd.jpg')
    cmd = ['ffmpeg', '-v', 'error', '-i', filename, '-qscale:v', '2', '-f',
           'image2', output_format]
    try:
        check_output(cmd)
    except:
        return False
    return True


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


def frame_rate(filename):
    """Return frame-rate of video

    Parameters
    ----------
    filename : stri
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


def video_duration(filename):
    """Return frame-rate of video

    Parameters
    ----------
    filename : stri
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


# String utilities
def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance btw two strings

    Note
    ----
    Taken from wikibooks.org-wiki-Algorithm_Implementation

    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one
            # character longer than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# IO utilities
def file_as_folder(filename):
    """Return a filename ending with os-path-separator

    Parameters
    ----------
    filename : str
        Fullpath filename

    Outputs
    -------
    filename : str
        Fullpath filename ending with os-path-separator

    """
    return os.path.splitext(filename)[0] + os.path.sep
