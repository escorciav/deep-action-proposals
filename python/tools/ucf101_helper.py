import os

import numpy as np
import pandas as pd


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

