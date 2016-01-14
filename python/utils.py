import os
from subprocess import check_output

import cv2
import numpy as np
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


def count_frames(filename, method=None):
    """Count number of frames of a video

    Parameters
    ----------
    filename : string
        fullpath of video file
    method : string, optional
        algorithm to use (None, 'ffprobe')

    Outputs
    -------
    counter : int
        number of frames

    """
    counter, fail_ffprobe = 0, False
    if method == 'ffprobe':
        cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams',
               'v:0', '-show_entries', 'stream=nb_read_frames', '-of',
               'default=nokey=1:noprint_wrappers=1', filename]
        try:
            counter = int(check_output(cmd).replace('\n', ''))
        except:
            counter, fail_ffprobe = 0, True

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


def get_clip(filename, i_frame, duration):
    """Return a clip from a video
    """
    if not os.path.isfile(filename):
        return None

    cap = cv2.VideoCapture(filename)
    record, counter, length, clip = False, 0, 0, []
    for i in xrange(0, i_frame):
        success = cap.grab()
    for i in xrange(0, duration):
        success, img = cap.read()
        if success:
            clip.append(img)
        else:
            break
    cap.release()
    return np.stack(clip)
