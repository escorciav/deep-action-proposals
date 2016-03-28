import array
import os

import numpy as np
import pandas as pd


def input_file_generator(filename, output_file, t_size=16, step_size=8,
                         output_folder=None, mkdir=True):
    """Generate textfile(s) used by C3D code

    Parameters
    ----------
    filename : string
        fullpath of CSV file (space separated of 4-5 columns with header) with
        data about the videos to process. Expected columns are:
        ['video-name', 'num-frame', 'i-frame', 'duration', 'label']
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
    req_cols = set(['video-name', 'num-frame', 'i-frame', 'duration'])
    try:
        df = pd.read_csv(filename, sep=' ')
    except:
        raise ValueError('Unable to open file {}'.format(filename))

    cols = df.columns.tolist()
    n_segments = df.shape[0]
    if not req_cols.issubset(cols):
        msg = 'Not enough information or incorrect format in {}'
        raise ValueError(msg.format(filename))

    if 'label' not in cols:
        dummy = np.zeros(n_segments, int)
        df = pd.concat([df, pd.DataFrame({'label': dummy})], axis=1)

    if isinstance(output_file, str):
        output_file = [output_file]

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
    df_c3d.drop_duplicates(keep='first', inplace=True)

    # Dump DataFrame for C3D-input
    try:
        df_c3d.to_csv(output_file[0], sep=' ', header=False, index=False)
    except:
        msg = 'Unable to create input list for C3D ({}). Check permissions.'
        raise ValueError(msg.format(output_file[0]))

    # C3D-output
    if len(output_file) > 1 and isinstance(output_folder, str):
        sr_out = (output_folder + os.path.sep +
                  df.loc[idx_keep_expanded, 'video-name'].astype(str) +
                  os.path.sep + init_end_frame.astype(str))
        # Avoid redundacy such that A//B introduce by previous cmd. A
        # principled solution is welcome.
        sr_out = sr_out.apply(os.path.normpath)
        sr_out.drop_duplicates(keep='first', inplace=True)
        skip_mkdir = False

        # Dump DataFrame for C3D-output
        try:
            sr_out.to_csv(output_file[1], header=None, index=False)
        except:
            skip_mkdir = True

        # Create dirs to place C3D-features
        if not skip_mkdir:
            summary['output'] = output_file[1]
            if mkdir:
                for i in df['video-name'].unique():
                    os.makedirs(os.path.join(output_folder, i))

    # Update summary
    summary['success'] = True
    summary['pctg-skipped-segments'] = ((n_segments - idx_bool_keep.sum()) *
                                        1.0 / n_segments)
    summary['ratio-clips-segments'] = (idx_keep_expanded.size * 1.0 /
                                       n_segments)
    return summary


def read_feature(filename):
    """Read feature dump by C3D

    Parameters
    ----------
    filename : str
        Fullpath of file to read

    Outputs
    -------
    x : ndarray
        numpy array of features

    Note: It accomplishes the same purpose of this code:
        C3D/examples/c3d_feature_extraction/script/read_binary_blob.m

    """
    s_parr, d_parr = array.array('i'), array.array('f')
    with open(filename, 'r') as f:
        s_parr.fromfile(f, 5)
        s = np.array(s_parr)
        m = np.cumprod(s)[-1]

        d_parr.fromfile(f, m)
    return s, np.array(d_parr)
