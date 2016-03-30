import os

import numpy as np
import pandas as pd

from daps.utils.extra import file_as_folder
from daps.utils.video import count_frames


def dump_video_list(filename):
    """Read all train-test list of UCF-101 and create csv with all videos

    Parameters
    ----------
    filename : string
        Fullpath name of the CSV-file with all the videos of UCF-101

    """
    arr = []
    prefix = 'data/ucf101/ucfTrainTestlist'
    ucf_lists = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']
    for i in ucf_lists:
        arr.append(np.array(pd.read_csv(os.path.join(prefix, i),
                                        sep=' ', header=None)))
    video_list = np.concatenate(arr, axis=0)
    _, idx = np.unique(video_list[:, 0], return_index=True)
    df = pd.DataFrame(video_list[idx, :])
    df.to_csv(filename, sep=' ', header=None, index=False)


def list_zero_indexded(filename, new_file=None, no_ext=True):
    """Return a DataFrame from UCF train/test list using 0-indexed labels

    Parameters
    ----------
    filename : str
        Fullpath of UCF-list.
    new_file : str, optional
        Fullpath with new list.
    no_ext : bool, optional
        replace extension by path separator on each file.

    Outputs
    -------
    new_df : pandas.DataFrame
        table with video-list and labels.

    """
    df = pd.read_csv(filename, sep=' ', header=None)
    if df.shape[1] > 1:
        df.loc[:, 1] -= 1

    if no_ext:
        df_videos = df.loc[:, 0].apply(file_as_folder)
    else:
        df_videos = df.loc[:, 0]

    if df.shape[1] > 1:
        new_df = pd.concat([df_videos,  df.loc[:, 1]], axis=1)
    else:
        new_df = df_videos

    if new_file is not None:
        new_df.to_csv(new_file, header=None, sep=' ', index=False)
    return new_df


def dump_segment_list(filename, new_file, cf_method='dir', cf_ext='*.jpg'):
    """Dump CSV required for extracting C3D features

    Parameters
    ----------
    filename : str
        Fullpath of csv-file with list of videos
    new_file : str
        Fullpath of new csv-file
    cf_method : str, optional
        method used by count_frame function
    cf_ext : str, optional
        image extension used by count_frame if cf_method == 'dir'

    TODO
    - Read database or JSON instead on counting every-time
    """
    df = list_zero_indexded(filename)
    n_videos, n_cols = df.shape

    if n_cols > 1:
        labels = df.loc[:, 1]
    else:
        labels = np.zeros(n_videos, dtype=int)

    if cf_method != 'dir':
        i_frame = np.zeros(n_videos, dtype=int)
    else:
        i_frame = np.ones(n_videos, dtype=int)

    duration = df.loc[:, 0].apply(count_frames, args=(cf_method, cf_ext))
    new_df = pd.DataFrame({1: df.loc[:, 0], 2: duration, 3: i_frame,
                           4: duration, 5: labels})
    new_df.to_csv(new_file, sep=' ', header=None, index=False)
