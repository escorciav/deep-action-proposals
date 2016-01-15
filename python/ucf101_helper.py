import os

import numpy as np
import pandas as pd

from utils import file_as_folder
# TODO: Make a class packing all this function


def csv_to_dump_frames(filename):
    """Read all train-test list of UCF-101 and create csv with all videos

    Parameters
    ----------
    filename : string
        Fullpath name of the CSV-file with all the videos of UCF-101

    """
    arr = []
    prefix = 'data/ucf101/ucfTrainTestlist'
    ucf_lists = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt',
                 'testlist01.txt', 'testlist02.txt', 'testlist03.txt']
    for i in ucf_lists:
        arr.append(np.array(pd.read_csv(os.path.join(prefix, i),
                                        sep=' ', header=None).loc[:, 0]))
    video_list = np.hstack(arr)
    df = pd.DataFrame(np.unique(video_list))
    df.to_csv(filename, sep=' ', header=None, index=False)


def list_zero_indexded(filename, new_file, no_ext=True):
    """Save UCF train/test list using 0-indexing for labels

    Parameters
    ----------
    filename : str
        Fullpath of UCF-list
    new_file : str
        Fullpath of new list
    no_ext : bool, optional
        replace extension by path separator on each file

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

    new_df.to_csv(new_file, header=None, sep=' ', index=False)
    return None
