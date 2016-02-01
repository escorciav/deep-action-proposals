import glob
import json
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


# Sampling
def sampling_with_uniform_groups(x, bin_edges, strict=True, rng=None):
    """
    Sample values of x such that the distribution on bin-edges is as uniform as
    possible

    Parameters
    ----------
    x : ndarray
        1-dim array to sample
    bin_edges : ndarray
        1-dim array with intervals. See numpy.digitize for more details.
    strict : bool, optional
        If true, every bucket will have the same number of samples.
    rng : numpy.random.RandomState, optional
        pseudo-rnd number generator instance

    Outputs
    -------
    keep_idx : ndarray
        1-dim array of indexes from the elements of x to keep

    """
    if rng is None:
        rng = np.random.RandomState()

    n_bins = len(bin_edges) - 1
    idx = np.digitize(x, bin_edges) - 1
    counts = np.bincount(idx, minlength=n_bins)

    min_samples = counts.min()
    if strict:
        samples_per_bin = np.repeat(min_samples, n_bins)
        # Sample from the same distrib withot matter strict value
        rng.rand(n_bins)
    else:
        samples_per_bin = np.minimum(min_samples+counts.std()*rng.rand(n_bins),
                                     counts)

    keep_idx_list = []
    for i in xrange(n_bins):
        tmp = (idx == i).nonzero()[0]
        keep_idx_list.append(rng.permutation(tmp)[:samples_per_bin[i]])
    keep_idx = np.hstack(keep_idx_list)
    return keep_idx


# Segment utilities
def segment_intersection(target_segments, test_segments,
                         return_ratio_target=False):
    """Compute intersection btw segments

    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    return_ratio_target : bool, optional
        extra ndarray output with ratio btw size of intersection over size of
        target-segments

    Outputs
    -------
    intersect : ndarray
        3-dim array in format [m, n, 2:=[init, end]]
    ratio_target : ndarray
        2-dim array [m x n] with ratio btw size of intersect over size of
        target segment

    Note: It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')
    m, n = target_segments.shape[0], test_segments.shape[0]
    if return_ratio_target:
        ratio_target = np.zeros((m, n))

    intersect = np.zeros((m, n, 2))
    for i in xrange(m):
        target_size = target_segments[i, 1] - target_segments[i, 0] + 1.0
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        intersect[i, :, 0], intersect[i, :, 1] = tt1, tt2
        if return_ratio_target:
            isegs_size = (tt2 - tt1 + 1.0).clip(0)
            ratio_target[i, :] = isegs_size / target_size

    if return_ratio_target:
        return intersect, ratio_target
    return intersect


def segment_format(X, mthd='c2b'):
    """Transform temporal annotations

    Parameters
    ----------
    X : ndarray
        [n x 2] array with temporal annotations
    mthd : str
        Type of conversion:
        'c2b': transform [central-frame, duration] onto [f-init, f-end]

    Outputs
    -------
    Y : ndarray
        [n x 2] array with transformed temporal annotations

    """
    if mthd == 'c2b':
        Xinit = np.round(X[:, 0] - 0.5*X[:, 1])
        return np.stack([Xinit, Xinit + X[:, 1]], axis=-1)


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
def dump_json(filename, data, **kwargs):
    """Serialize data as JSON-file

    Parameters
    ----------
    filename : str
    data : list, dict, etc.
        Data to save. Chech json.dump for more details and description.

    """
    with open(filename, 'w') as f:
        json.dump(data, f, **kwargs)
    return None


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
